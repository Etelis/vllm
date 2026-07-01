#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive GSM8K + MBPP against a vLLM server and collect per-prompt expert usage.

The server must be launched with ``--enable-return-routed-experts`` (so each
completion carries routing) and ``--enable-prompt-tokens-details`` (so each
completion reports ``cached_tokens`` for the prefix-cache correlation).

For every prompt we record a compact per-layer x expert histogram (not the raw
per-token tensor), the number of prompt tokens served from the prefix cache, and
token counts. Results are written per domain under ``--out-dir`` for offline
analysis by ``analyze.py``.

Example:
    .venv/bin/python -m benchmarks.moe_expert_routing.run_experiment \\
        --host http://127.0.0.1 --port 8000 \\
        --num-questions 500 --num-experts 128 --out-dir ./moe_stats_out
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

from benchmarks.moe_expert_routing.datasets import DOMAIN_BUILDERS, Domain
from benchmarks.moe_expert_routing.expert_stats import (
    decode_routed_experts,
    layer_expert_histogram,
)


async def _fetch_metrics(session: aiohttp.ClientSession, base_url: str) -> str:
    try:
        async with session.get(f"{base_url}/metrics") as resp:
            if resp.status == 200:
                return await resp.text()
    except Exception:
        pass
    return ""


def _extract_prefix_cache_metrics(metrics_text: str) -> dict[str, float]:
    wanted = ("prefix_cache", "gpu_prefix_cache")
    out: dict[str, float] = {}
    for line in metrics_text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if any(w in line for w in wanted):
            name, _, value = line.rpartition(" ")
            try:
                out[name.strip()] = float(value)
            except ValueError:
                continue
    return out


async def _one_request(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt: str,
    domain: Domain,
    seed: int,
    num_experts: int,
) -> dict | None:
    """Send one completion; return a compact per-prompt record (or None)."""
    payload = {
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": domain.max_tokens,
        "stop": domain.stop,
        "seed": seed,
    }
    try:
        async with session.post(f"{base_url}/v1/completions", json=payload) as resp:
            resp.raise_for_status()
            result = await resp.json()
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    choice = result["choices"][0]
    routing = decode_routed_experts(choice.get("routed_experts"))
    usage = result.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}

    record: dict = {
        "ok": routing is not None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "finish_reason": choice.get("finish_reason"),
    }
    if routing is None:
        record["error"] = "no routed_experts in response"
        return record

    record["routing_tokens"] = int(routing.shape[0])
    record["num_layers"] = int(routing.shape[1])
    record["top_k"] = int(routing.shape[2])
    record["max_expert_id"] = int(routing.max()) if routing.size else -1
    record["hist"] = layer_expert_histogram(routing, num_experts).astype(np.int32)
    record["raw"] = routing  # kept only for a few samples by the caller
    return record


async def _run_domain(
    domain: Domain,
    base_url: str,
    seed: int,
    num_experts: int,
    max_concurrency: int,
    out_dir: str,
    num_raw_samples: int,
    request_timeout: float,
) -> dict:
    dom_dir = os.path.join(out_dir, domain.name)
    os.makedirs(os.path.join(dom_dir, "samples"), exist_ok=True)

    timeout = aiohttp.ClientTimeout(total=request_timeout)
    sem = asyncio.Semaphore(max_concurrency)
    tic = time.perf_counter()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        metrics_before = _extract_prefix_cache_metrics(
            await _fetch_metrics(session, base_url)
        )
        # Warm the shared preamble once so the rest of the domain hits the
        # prefix cache (this is the intra-domain locality the study relies on).
        print(f"[{domain.name}] warming shared preamble ({domain.num_shots}-shot)...")
        await _one_request(
            session, base_url, domain.prompts[0], domain, seed, num_experts
        )

        async def _guarded(idx: int) -> tuple[int, dict | None]:
            async with sem:
                rec = await _one_request(
                    session, base_url, domain.prompts[idx], domain, seed, num_experts
                )
                return idx, rec

        tasks = [_guarded(i) for i in range(len(domain.prompts))]
        results: list[tuple[int, dict | None]] = await tqdm.gather(
            *tasks, desc=f"[{domain.name}] collecting"
        )
        metrics_after = _extract_prefix_cache_metrics(
            await _fetch_metrics(session, base_url)
        )

    elapsed = time.perf_counter() - tic

    # Assemble compact outputs.
    hists: list[np.ndarray] = []
    kept_idx: list[int] = []
    table: list[dict] = []
    saved_raw = 0
    for idx, rec in sorted(results, key=lambda t: t[0]):
        if rec is None:
            continue
        row = {
            k: rec.get(k)
            for k in (
                "ok",
                "prompt_tokens",
                "completion_tokens",
                "cached_tokens",
                "routing_tokens",
                "finish_reason",
                "max_expert_id",
                "error",
            )
        }
        row["idx"] = idx
        table.append(row)
        if rec.get("ok"):
            hists.append(rec["hist"])
            kept_idx.append(idx)
            if saved_raw < num_raw_samples:
                np.save(os.path.join(dom_dir, "samples", f"{idx}.npy"), rec["raw"])
                saved_raw += 1

    if hists:
        np.save(os.path.join(dom_dir, "hist.npy"), np.stack(hists))
    np.save(os.path.join(dom_dir, "kept_idx.npy"), np.array(kept_idx, dtype=np.int64))
    with open(os.path.join(dom_dir, "table.jsonl"), "w") as fout:
        for row in table:
            fout.write(json.dumps(row) + "\n")
    with open(os.path.join(dom_dir, "preamble.txt"), "w") as fout:
        fout.write(domain.preamble)

    ok_count = len(hists)
    cached = [r["cached_tokens"] for r in table if r.get("cached_tokens") is not None]
    ptoks = [r["prompt_tokens"] for r in table if r.get("prompt_tokens")]
    max_ids = [r["max_expert_id"] for r in table if r.get("max_expert_id") is not None]
    summary = {
        "domain": domain.name,
        "num_prompts": len(domain.prompts),
        "num_ok": ok_count,
        "num_shots": domain.num_shots,
        "preamble_chars": len(domain.preamble),
        "elapsed_s": round(elapsed, 2),
        "mean_cached_tokens": float(np.mean(cached)) if cached else None,
        "mean_prompt_tokens": float(np.mean(ptoks)) if ptoks else None,
        "observed_max_expert_id": max(max_ids) if max_ids else -1,
        "prefix_cache_metrics_before": metrics_before,
        "prefix_cache_metrics_after": metrics_after,
        "num_experts": num_experts,
        "num_layers": int(hists[0].shape[0]) if hists else None,
    }
    with open(os.path.join(dom_dir, "config.json"), "w") as fout:
        json.dump({**summary, "meta": domain.meta}, fout, indent=2)
    print(
        f"[{domain.name}] {ok_count}/{len(domain.prompts)} ok, "
        f"mean cached_tokens={summary['mean_cached_tokens']}, "
        f"elapsed={summary['elapsed_s']}s"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--domains", default="gsm8k,mbpp")
    parser.add_argument("--num-questions", type=int, default=500)
    parser.add_argument("--gsm8k-shots", type=int, default=8)
    parser.add_argument("--mbpp-shots", type=int, default=3)
    parser.add_argument(
        "--num-experts",
        type=int,
        default=128,
        help="Routed experts E (Qwen3-30B-A3B=128); observed max id is validated.",
    )
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--num-raw-samples", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="./moe_stats_out")
    args = parser.parse_args()

    base_url = f"{args.host}:{args.port}"
    os.makedirs(args.out_dir, exist_ok=True)
    shots = {"gsm8k": args.gsm8k_shots, "mbpp": args.mbpp_shots}

    summaries = []
    for name in [d.strip() for d in args.domains.split(",") if d.strip()]:
        if name not in DOMAIN_BUILDERS:
            raise SystemExit(
                f"unknown domain {name!r}; choices={list(DOMAIN_BUILDERS)}"
            )
        print(f"\n=== building {name} ({args.num_questions} questions) ===")
        domain = DOMAIN_BUILDERS[name](
            num_questions=args.num_questions, num_shots=shots.get(name, 3)
        )
        summary = asyncio.run(
            _run_domain(
                domain=domain,
                base_url=base_url,
                seed=args.seed,
                num_experts=args.num_experts,
                max_concurrency=args.max_concurrency,
                out_dir=args.out_dir,
                num_raw_samples=args.num_raw_samples,
                request_timeout=args.request_timeout,
            )
        )
        summaries.append(summary)
        observed_max = summary.get("observed_max_expert_id", -1)
        if observed_max >= args.num_experts:
            print(
                f"WARNING: observed expert id {observed_max} >= --num-experts "
                f"{args.num_experts}; re-run with a larger --num-experts."
            )

    with open(os.path.join(args.out_dir, "run_summary.json"), "w") as fout:
        json.dump(summaries, fout, indent=2)
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
