"""Exercise three real EP groups through the opt-in runtime P/D API.

This traffic driver also acts as the test router. It reserves both stages before
prefill, stops assigning a draining group, and preserves already assigned KV
handoffs. It never retries an executed request or hides request failures.
"""

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

import httpx

PROMPTS = [
    "The capital of France is",
    "Calculate 19 plus 23. Explain briefly:",
    "Write a Python function to reverse a list:",
    "Explain why the sky appears blue in two sentences:",
    "Continue the sequence: 2, 4, 8, 16,",
    "Name three planets in the solar system:",
    "Translate 'good morning' into Spanish:",
    "Write a short story about a cat named Luna:",
]
PROMPTS = [
    "Context: Mira keeps a notebook about astronomy, languages, and mathematics. "
    "She uses clear explanations and checks her calculations. "
    * 12
    + "\nAnswer this question: "
    + prompt
    for prompt in PROMPTS
]


def pct(values, q):
    if not values:
        return None
    return sorted(values)[min(len(values) - 1, int(len(values) * q))]


class Fleet:
    def __init__(self, client, urls, model):
        self.client, self.urls, self.model = client, urls, model
        self.states = []
        self.available = [True] * 3
        self.reservations = [0] * 3
        self.rotation = Counter()
        self.records = []
        self.transitions = []
        self.started = time.monotonic()

    async def initialize(self):
        self.states = [
            (await self.client.get(u + "/v1/pd_role")).json() for u in self.urls
        ]
        assert [s["role"] for s in self.states] == ["prefill", "prefill", "decode"], (
            self.states
        )

    def headers(self, index, state=None):
        state = state or self.states[index]
        return {"x-vllm-pd-role": state["role"], "x-vllm-pd-epoch": str(state["epoch"])}

    def choose(self, role):
        candidates = [
            i
            for i, s in enumerate(self.states)
            if self.available[i] and s["role"] == role
        ]
        assert candidates, f"No ready {role} group"
        index = candidates[self.rotation[role] % len(candidates)]
        self.rotation[role] += 1
        return index

    async def request(self, prompt_index, max_tokens=64):
        # Both reservations are acquired before the first await.
        p, d = self.choose("prefill"), self.choose("decode")
        p_headers, d_headers = self.headers(p), self.headers(d)
        self.reservations[p] += 1
        self.reservations[d] += 1
        p_pending, d_pending = True, True
        started = time.monotonic()
        record = {
            "start": started - self.started,
            "p": p,
            "d": d,
            "prompt": prompt_index,
        }
        body = {
            "model": self.model,
            "prompt": PROMPTS[prompt_index],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 42,
            "ignore_eos": True,
            "cache_salt": f"pd-validation-{time.monotonic_ns()}",
        }
        token_times, parts = [], []
        try:
            prefill = await self.client.post(
                self.urls[p] + "/v1/completions",
                headers=p_headers,
                json={
                    **body,
                    "max_tokens": 1,
                    "stream": False,
                    "kv_transfer_params": {
                        "do_remote_decode": True,
                        "do_remote_prefill": False,
                    },
                },
            )
            prefill.raise_for_status()
            kv = prefill.json().get("kv_transfer_params")
            assert kv and kv.get("do_remote_prefill") and kv.get("remote_block_ids"), (
                prefill.text
            )
            self.reservations[p] -= 1
            p_pending = False
            record["kv_remote_engine_id"] = kv["remote_engine_id"]
            record["remote_num_tokens"] = kv.get("remote_num_tokens")
            record["remote_blocks"] = sum(len(g) for g in kv["remote_block_ids"])
            finish_reason, done = None, False
            async with self.client.stream(
                "POST",
                self.urls[d] + "/v1/completions",
                headers=d_headers,
                json={**body, "stream": True, "kv_transfer_params": kv},
            ) as response:
                response.raise_for_status()
                self.reservations[d] -= 1
                d_pending = False
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        done = True
                        continue
                    chunk = json.loads(payload)
                    assert "error" not in chunk, chunk
                    for choice in chunk["choices"]:
                        if choice.get("text"):
                            parts.append(choice["text"])
                            token_times.append(time.monotonic())
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
            assert done and finish_reason == "length", (done, finish_reason)
            record.update(
                text="".join(parts),
                finish_reason=finish_reason,
                ttft=token_times[0] - started,
                intervals=[b - a for a, b in zip(token_times, token_times[1:])],
                chunks=len(token_times),
                token_times=[t - self.started for t in token_times],
            )
        except Exception as exc:
            record["error"] = repr(exc)
        finally:
            if p_pending:
                self.reservations[p] -= 1
            if d_pending:
                self.reservations[d] -= 1
            record["end"] = time.monotonic() - self.started
            self.records.append(record)
        return record

    async def switch(self, index, role, timeout=120):
        start = time.monotonic()
        previous = self.states[index].copy()
        self.available[index] = False
        # A D reservation may still be waiting for its P response. Do not fence
        # the engine until those assigned handoffs have reached HTTP admission.
        while self.reservations[index]:
            assert time.monotonic() - start < timeout, "Reservation drain timeout"
            await asyncio.sleep(0.005)
        response = await self.client.post(
            self.urls[index] + "/v1/pd_role",
            json={
                "role": role,
                "expected_epoch": previous["epoch"],
                "drain_timeout": timeout,
            },
        )
        response.raise_for_status()
        accepted = time.monotonic()
        samples = [response.json()]
        while True:
            state = (await self.client.get(self.urls[index] + "/v1/pd_role")).json()
            samples.append(state)
            assert state["phase"] != "failed", state
            if state["phase"] == "ready":
                assert (
                    state["role"] == role and state["epoch"] == previous["epoch"] + 1
                ), state
                break
            assert time.monotonic() - start < timeout + 5, samples[-1]
            await asyncio.sleep(0.01)
        end = time.monotonic()
        self.states[index] = state
        self.available[index] = True
        event = {
            "from": previous["role"],
            "to": role,
            "start": start - self.started,
            "end": end - self.started,
            "reservation_drain_seconds": accepted - start,
            "engine_transition_seconds": end - accepted,
            "samples": samples,
        }
        self.transitions.append(event)
        return event


async def main(args):
    async with httpx.AsyncClient(
        timeout=180, limits=httpx.Limits(max_connections=128)
    ) as client:
        fleet = Fleet(client, args.urls, args.model)
        await fleet.initialize()
        initial_states = [s.copy() for s in fleet.states]
        metrics_before = [(await client.get(u + "/metrics")).text for u in args.urls]
        startup_probes = [(await client.get(u + "/pd_probe")).json() for u in args.urls]
        # Establish greedy references through real P/D transfer before transitions.
        references = {}
        for i in range(len(PROMPTS)):
            result = await fleet.request(i, args.tokens)
            assert "error" not in result, result
            references[i] = result["text"]
        if args.references:
            dense = json.loads(Path(args.references).read_text())["references"]
            for index, value in references.items():
                assert value == dense[str(index)], {
                    "prompt": index,
                    "pd": value,
                    "dense": dense[str(index)],
                }
        initial_probes = [(await client.get(u + "/pd_probe")).json() for u in args.urls]
        reference_count = len(fleet.records)
        stop = asyncio.Event()

        async def load(worker_id):
            iteration = worker_id
            while not stop.is_set():
                await fleet.request(iteration % len(PROMPTS), args.tokens)
                iteration += args.concurrency

        transition_probes = []
        fatal_error = None
        worker_errors = []
        workers = [asyncio.create_task(load(i)) for i in range(args.concurrency)]
        try:
            await asyncio.sleep(args.settle)
            for _ in range(args.cycles):
                for role in ("decode", "prefill"):
                    event = await fleet.switch(1, role)
                    print(
                        json.dumps({k: v for k, v in event.items() if k != "samples"}),
                        flush=True,
                    )
                    probe = await client.get(args.urls[1] + "/pd_probe")
                    probe.raise_for_status()
                    transition_probes.append(probe.json())
                    await asyncio.sleep(args.settle)
        except Exception as exc:
            fatal_error = repr(exc)
        finally:
            stop.set()
            worker_results = await asyncio.gather(*workers, return_exceptions=True)
            worker_errors = [
                f"worker {i}: {value!r}"
                for i, value in enumerate(worker_results)
                if isinstance(value, BaseException)
            ]
            if worker_errors and fatal_error is None:
                fatal_error = "; ".join(worker_errors)
            Path(args.output + ".partial").write_text(
                json.dumps(
                    {
                        "error": fatal_error,
                        "worker_errors": worker_errors,
                        "records": fleet.records,
                        "transitions": fleet.transitions,
                        "transition_probes": transition_probes,
                    },
                    indent=2,
                )
            )
        final_probes = [(await client.get(u + "/pd_probe")).json() for u in args.urls]
        metrics_after = [(await client.get(u + "/metrics")).text for u in args.urls]
        results = fleet.records[reference_count:]
        failures = [r for r in results if "error" in r]
        mismatches = [
            r
            for r in results
            if "error" not in r and r["text"] != references[r["prompt"]]
        ]
        summary = {
            "fatal_error": fatal_error,
            "worker_errors": worker_errors,
            "initial_epochs": [s["epoch"] for s in initial_states],
            "final_epochs": [s["epoch"] for s in fleet.states],
            "requests": len(results),
            "errors": len(failures),
            "output_mismatches": len(mismatches),
            "transitions": len(fleet.transitions),
            "ttft_p50": pct([r["ttft"] for r in results if "ttft" in r], 0.5),
            "ttft_p99": pct([r["ttft"] for r in results if "ttft" in r], 0.99),
            "inter_chunk_p99": pct(
                [t for r in results for t in r.get("intervals", [])], 0.99
            ),
            "groups_used": {
                str((p, d)): sum(r["p"] == p and r["d"] == d for r in results)
                for p, d in ((0, 2), (1, 2), (0, 1))
            },
        }
        for event in fleet.transitions:
            times = sorted(
                t
                for r in results
                if r["d"] == 2
                for t in r.get("token_times", [])
                if event["start"] <= t <= event["end"]
            )
            boundaries = [event["start"], *times, event["end"]]
            event["unaffected_decode_chunks_during_transition"] = len(times)
            event["unaffected_decode_max_gap_seconds"] = max(
                b - a for a, b in zip(boundaries, boundaries[1:])
            )
        artifact = {
            "summary": summary,
            "args": vars(args),
            "initial_probes": initial_probes,
            "startup_probes": startup_probes,
            "transition_probes": transition_probes,
            "final_probes": final_probes,
            "transitions": fleet.transitions,
            "records": fleet.records,
            "references": references,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
        }
        Path(args.output).write_text(json.dumps(artifact, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        assert not fatal_error and not failures and not mismatches, summary
        if args.cycles:
            assert all(summary["groups_used"].values()), summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urls",
        nargs=3,
        default=[
            "http://127.0.0.1:8100",
            "http://127.0.0.1:8110",
            "http://127.0.0.1:8120",
        ],
    )
    parser.add_argument("--model", default="Qwen3-30B-A3B")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--references")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--settle", type=float, default=3)
    parser.add_argument("--output", default="/workspace/validation/results.json")
    asyncio.run(main(parser.parse_args()))
