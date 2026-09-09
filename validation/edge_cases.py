"""Real KV lease timeout/rollback and stale-admission checks on running groups."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
from exercise import PROMPTS


async def main(args):
    async with httpx.AsyncClient(timeout=120) as client:
        p, d = args.prefill, args.decode
        pstate = (await client.get(p + "/v1/pd_role")).json()
        dstate = (await client.get(d + "/v1/pd_role")).json()
        assert pstate["role"] == "prefill" and pstate["phase"] == "ready"
        assert dstate["role"] == "decode" and dstate["phase"] == "ready"
        pheaders = {
            "x-vllm-pd-role": "prefill",
            "x-vllm-pd-epoch": str(pstate["epoch"]),
        }
        dheaders = {"x-vllm-pd-role": "decode", "x-vllm-pd-epoch": str(dstate["epoch"])}
        probe_before = (await client.get(p + "/pd_probe")).json()
        body = {
            "model": args.model,
            "prompt": PROMPTS[0],
            "max_tokens": 64,
            "temperature": 0,
            "seed": 42,
            "ignore_eos": True,
            "cache_salt": f"pd-held-lease-{time.monotonic_ns()}",
        }
        response = await client.post(
            p + "/v1/completions",
            headers=pheaders,
            json={
                **body,
                "max_tokens": 1,
                "kv_transfer_params": {
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                },
            },
        )
        response.raise_for_status()
        kv = response.json()["kv_transfer_params"]
        assert kv["remote_block_ids"]
        switch = await client.post(
            p + "/v1/pd_role",
            json={
                "role": "decode",
                "expected_epoch": pstate["epoch"],
                "drain_timeout": 0.25,
            },
        )
        switch.raise_for_status()
        states = [switch.json()]
        deadline = time.monotonic() + 5
        while True:
            state = (await client.get(p + "/v1/pd_role")).json()
            states.append(state)
            if state["phase"] in ("ready", "failed"):
                break
            assert time.monotonic() < deadline, state
            await asyncio.sleep(0.01)
        assert state["phase"] == "ready" and state["role"] == "prefill", state
        assert state["epoch"] == pstate["epoch"] and "TimeoutError" in state["error"], (
            state
        )
        # The held KV remains usable after timeout: no abort, reset or expired lease.
        completion = await client.post(
            d + "/v1/completions",
            headers=dheaders,
            json={**body, "kv_transfer_params": kv},
        )
        completion.raise_for_status()
        result = completion.json()
        assert result["choices"][0]["finish_reason"] == "length", result
        references = json.loads(Path(args.references).read_text())["references"]
        assert result["choices"][0]["text"] == references["0"], result
        # Wrong epoch must reject even an invalid body before inference/validation.
        stale = await client.post(
            p + "/v1/completions",
            headers={**pheaders, "x-vllm-pd-epoch": str(pstate["epoch"] + 99)},
            json={},
        )
        assert stale.status_code == 409, stale.text
        probe_after = (await client.get(p + "/pd_probe")).json()
        artifact = {
            "held_lease_rollback_passed": True,
            "stale_epoch_status": stale.status_code,
            "states": states,
            "response": result,
            "probe_before": probe_before,
            "probe_after": probe_after,
        }
        Path(args.output).write_text(json.dumps(artifact, indent=2))
        print(
            json.dumps(
                {
                    "held_lease_rollback_passed": True,
                    "stale_epoch_status": stale.status_code,
                    "epoch_preserved": state["epoch"],
                    "output_matches_reference": True,
                }
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill", default="http://127.0.0.1:8110")
    parser.add_argument("--decode", default="http://127.0.0.1:8120")
    parser.add_argument("--model", default="Qwen3-30B-A3B")
    parser.add_argument("--references", default="/workspace/validation/results.json")
    parser.add_argument("--output", default="/workspace/validation/edge-results.json")
    asyncio.run(main(parser.parse_args()))
