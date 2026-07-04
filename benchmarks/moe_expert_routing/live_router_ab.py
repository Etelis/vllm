#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive interleaved two-domain traffic through a live KV-cache-aware router.

Unlike ``run_experiment.py`` (which targets one vLLM server, domain by
domain), this sends GSM8K + MBPP prompts *interleaved* through an llm-d /
Gateway-API-Inference-Extension gateway fronting several vLLM replicas, and
records per request which replica served it. Replica attribution comes from
response headers when the gateway echoes the picked endpoint, plus the
completion ``id`` for an ``oc logs``-based join as fallback.

Each replica must run with ``--enable-return-routed-experts`` and
``--enable-prompt-tokens-details``. Run once per router arm (e.g. EPP with and
without the prefix-cache scorer) with a distinct ``--out-dir``; analyze with
``live_analyze.py``.

Example (in-cluster runner):
    python -m benchmarks.moe_expert_routing.live_router_ab \\
        --base-url http://gw-istio.itay-eplb-affinity.svc.cluster.local \\
        --num-questions 300 --num-experts 128 --arm affinity \\
        --replica-urls r0=http://10.0.0.1:8000,r1=http://10.0.0.2:8000 \\
        --out-dir ./live_out/affinity
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

from benchmarks.moe_expert_routing.datasets import DOMAIN_BUILDERS, Domain
from benchmarks.moe_expert_routing.expert_stats import (
    decode_routed_experts,
    layer_expert_histogram,
)

# Response headers that may carry the picked endpoint, most specific first.
ENDPOINT_HEADERS = (
    "x-gateway-destination-endpoint",
    "x-inference-pod",
    "x-served-by",
    "x-envoy-upstream-host",
)


def _endpoint_from_headers(headers) -> str | None:
    for name in ENDPOINT_HEADERS:
        if name in headers:
            return headers[name]
    return None


async def _fetch_replica_metrics(urls: dict[str, str]) -> dict[str, dict[str, float]]:
    """Scrape prefix-cache counters straight from each replica pod."""
    wanted = ("prefix_cache", "gpu_prefix_cache")
    out: dict[str, dict[str, float]] = {}
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for name, url in urls.items():
            vals: dict[str, float] = {}
            try:
                async with session.get(f"{url}/metrics") as resp:
                    text = await resp.text() if resp.status == 200 else ""
            except Exception:
                text = ""
            for line in text.splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                if any(w in line for w in wanted):
                    metric, _, value = line.rpartition(" ")
                    try:
                        vals[metric.strip()] = float(value)
                    except ValueError:
                        continue
            out[name] = vals
    return out


async def _served_model(session: aiohttp.ClientSession, base_url: str) -> str:
    async with session.get(f"{base_url}/v1/models") as resp:
        resp.raise_for_status()
        return (await resp.json())["data"][0]["id"]


async def _one_request(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt: str,
    domain: Domain,
    seed: int,
    num_experts: int,
    max_tokens_override: int | None = None,
    model: str = "Qwen/Qwen3-30B-A3B",
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": max_tokens_override or domain.max_tokens,
        "stop": domain.stop,
        "seed": seed,
    }
    if max_tokens_override:
        payload.pop("stop", None)  # force full-length generations for load runs
    tic = time.perf_counter()
    try:
        async with session.post(f"{base_url}/v1/completions", json=payload) as resp:
            resp.raise_for_status()
            endpoint = _endpoint_from_headers(resp.headers)
            result = await resp.json()
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    latency = time.perf_counter() - tic

    choice = result["choices"][0]
    routing = decode_routed_experts(choice.get("routed_experts"))
    usage = result.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}

    record: dict = {
        "ok": routing is not None,
        "id": result.get("id"),
        "endpoint": endpoint,
        "latency_s": round(latency, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "finish_reason": choice.get("finish_reason"),
    }
    if routing is None:
        record["error"] = "no routed_experts in response"
        return record
    record["hist"] = layer_expert_histogram(routing, num_experts).astype(np.int32)
    return record


async def _run(args, domains: dict[str, Domain]) -> None:
    replica_urls = {}
    for item in (args.replica_urls or "").split(","):
        if "=" in item:
            name, url = item.split("=", 1)
            replica_urls[name.strip()] = url.strip().rstrip("/")

    # Optional client-side routing: domain=replica pairs; requests for a
    # mapped domain go straight to that replica instead of the gateway.
    route_map: dict[str, str] = {}
    for item in (args.route_map or "").split(","):
        if "=" in item:
            dom_name, replica = item.split("=", 1)
            route_map[dom_name.strip()] = replica_urls[replica.strip()]

    def _target(domain_name: str) -> str:
        return route_map.get(domain_name, args.base_url)

    # Interleave: fixed-seed shuffle of all (domain, question) pairs.
    order = [
        (name, i) for name, dom in domains.items() for i in range(len(dom.prompts))
    ]
    random.Random(args.seed).shuffle(order)

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    sem = asyncio.Semaphore(args.max_concurrency)
    os.makedirs(args.out_dir, exist_ok=True)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        model = args.model or await _served_model(session, args.base_url)
        print(f"[{args.arm}] model: {model}")
        metrics_before = await _fetch_replica_metrics(replica_urls)

        # Seed each shared preamble once (first touch pins the prefix under an
        # affinity router; under a load-only router it lands arbitrarily).
        for name, dom in domains.items():
            print(f"[{args.arm}] warming {name} preamble...")
            await _one_request(
                session,
                _target(name),
                dom.prompts[0],
                dom,
                args.seed,
                args.num_experts,
                model=model,
            )

        async def _guarded(pos: int, name: str, idx: int):
            async with sem:
                rec = await _one_request(
                    session,
                    _target(name),
                    domains[name].prompts[idx],
                    domains[name],
                    args.seed,
                    args.num_experts,
                    args.max_tokens,
                    model=model,
                )
                return pos, name, idx, rec

        bench_tic = time.perf_counter()

        tasks = [_guarded(p, name, idx) for p, (name, idx) in enumerate(order)]
        results = await tqdm.gather(*tasks, desc=f"[{args.arm}] interleaved")
        bench_elapsed = time.perf_counter() - bench_tic
        metrics_after = await _fetch_replica_metrics(replica_urls)

    hists: list[np.ndarray] = []
    table: list[dict] = []
    for pos, name, idx, rec in sorted(results, key=lambda t: t[0]):
        row = {
            k: rec.get(k)
            for k in (
                "ok",
                "id",
                "endpoint",
                "latency_s",
                "prompt_tokens",
                "completion_tokens",
                "cached_tokens",
                "finish_reason",
                "error",
            )
        }
        row.update({"pos": pos, "domain": name, "idx": idx, "hist_row": None})
        if rec.get("ok"):
            row["hist_row"] = len(hists)
            hists.append(rec["hist"])
        table.append(row)

    if hists:
        np.save(os.path.join(args.out_dir, "hist.npy"), np.stack(hists))
    with open(os.path.join(args.out_dir, "table.jsonl"), "w") as fout:
        for row in table:
            fout.write(json.dumps(row) + "\n")

    def _lat_summary(rows: list[dict]) -> dict | None:
        lats = sorted(r["latency_s"] for r in rows if r.get("latency_s") is not None)
        if not lats:
            return None
        pct = lambda p: round(lats[min(len(lats) - 1, int(p * len(lats)))], 3)
        return {
            "n": len(lats),
            "mean": round(float(np.mean(lats)), 3),
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p99": pct(0.99),
        }

    completion_tokens = sum(r.get("completion_tokens") or 0 for r in table)
    config = {
        "arm": args.arm,
        "base_url": args.base_url,
        "num_questions": args.num_questions,
        "num_experts": args.num_experts,
        "seed": args.seed,
        "max_concurrency": args.max_concurrency,
        "max_tokens_override": args.max_tokens,
        "route_map": args.route_map,
        "domains": list(domains),
        "num_ok": len(hists),
        "num_total": len(table),
        "endpoint_header_hits": sum(1 for r in table if r.get("endpoint")),
        "bench_elapsed_s": round(bench_elapsed, 2),
        "requests_per_s": round(len(table) / bench_elapsed, 2),
        "output_tokens_per_s": round(completion_tokens / bench_elapsed, 1),
        "latency": _lat_summary(table),
        "latency_by_domain": {
            d: _lat_summary([r for r in table if r["domain"] == d]) for d in domains
        },
        "replica_metrics_before": metrics_before,
        "replica_metrics_after": metrics_after,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as fout:
        json.dump(config, fout, indent=2)
    print(
        f"[{args.arm}] {len(hists)}/{len(table)} ok, "
        f"endpoint header on {config['endpoint_header_hits']} responses"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="gateway base URL")
    parser.add_argument("--arm", required=True, help="label, e.g. affinity/baseline")
    parser.add_argument("--domains", default="gsm8k,mbpp")
    parser.add_argument("--num-questions", type=int, default=300)
    parser.add_argument("--gsm8k-shots", type=int, default=8)
    parser.add_argument("--mbpp-shots", type=int, default=3)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument(
        "--model", default=None, help="served model id (default: query /v1/models)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="override per-domain max_tokens and drop stop strings (load runs)",
    )
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--route-map",
        default="",
        help="domain=replica pairs for client-side routing (bypasses gateway)",
    )
    parser.add_argument(
        "--replica-urls",
        default="",
        help="name=url pairs for direct per-replica /metrics scrapes",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    shots = {
        "gsm8k": args.gsm8k_shots,
        "mbpp": args.mbpp_shots,
        "tenants": args.gsm8k_shots,
    }
    domains: dict[str, Domain] = {}
    for name in [d.strip() for d in args.domains.split(",") if d.strip()]:
        if name not in DOMAIN_BUILDERS:
            raise SystemExit(
                f"unknown domain {name!r}; choices={list(DOMAIN_BUILDERS)}"
            )
        print(f"=== building {name} ({args.num_questions} questions) ===")
        domains[name] = DOMAIN_BUILDERS[name](
            num_questions=args.num_questions, num_shots=shots.get(name, 3)
        )
    asyncio.run(_run(args, domains))


if __name__ == "__main__":
    main()
