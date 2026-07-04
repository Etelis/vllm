# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Closed-loop EPLB redundancy budget controller.

Consumes the router-layer telemetry that llm-d's endpoint picker scrapes for
its scorers (per-replica KV-pool occupancy and prefix-cache hit counters) and
decides how many redundant expert slots the fleet can afford without pushing
the tenant working set over the KV pool - the measured cliff (see RESULTS.md:
30B 69%->4.5% hits at 0->64 slots; 235B 84%->9.5% at 0->8).

Decision rule per window, per replica:
  occupancy >= hot_occupancy and hit_ratio < thrash_hits -> OVERFLOW
      (working set already exceeds the pool: recommend zero slots)
  occupancy >= tight_occupancy -> TIGHT (no slot headroom)
  else -> ROOM: max_safe = (tight_occupancy - occupancy) * pool / slot_cost

Fleet recommendation = min over replicas. The controller never sees the
workload definition - only the live signals the router itself routes on.

Usage:
    python -m benchmarks.moe_expert_routing.budget_controller \\
        --replica-urls r0=http://IP:8000,r1=http://IP:8000 \\
        --pool-tokens 230224 --slot-cost 9316 --requested-slots 8 \\
        --window 30 --once
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
import urllib.request

METRICS = (
    "vllm:gpu_cache_usage_perc",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
)


def scrape(url: str) -> dict[str, float]:
    out = dict.fromkeys(METRICS, 0.0)
    with urllib.request.urlopen(f"{url}/metrics", timeout=10) as resp:
        for line in resp.read().decode().splitlines():
            if line.startswith("#"):
                continue
            for m in METRICS:
                if line.startswith(m):
                    with contextlib.suppress(ValueError):
                        out[m] += float(line.rsplit(" ", 1)[1])
    return out


def classify(
    occupancy: float,
    hit_ratio: float | None,
    pool: int,
    slot_cost: int,
    hot_occupancy: float = 0.97,
    tight_occupancy: float = 0.90,
    thrash_hits: float = 0.5,
) -> tuple[str, int]:
    if occupancy >= hot_occupancy and (hit_ratio or 0.0) < thrash_hits:
        return "overflow", 0
    if occupancy >= tight_occupancy:
        return "tight", 0
    headroom = (tight_occupancy - occupancy) * pool
    return "room", int(headroom // slot_cost)


def decide(args) -> dict:
    replicas = {}
    for item in args.replica_urls.split(","):
        name, url = item.split("=", 1)
        replicas[name.strip()] = url.strip().rstrip("/")

    before = {n: scrape(u) for n, u in replicas.items()}
    time.sleep(args.window)
    after = {n: scrape(u) for n, u in replicas.items()}

    verdict: dict = {"replicas": {}, "window_s": args.window}
    fleet_safe = None
    for n in replicas:
        occ = after[n]["vllm:gpu_cache_usage_perc"]
        dq = (
            after[n]["vllm:prefix_cache_queries_total"]
            - before[n]["vllm:prefix_cache_queries_total"]
        )
        dh = (
            after[n]["vllm:prefix_cache_hits_total"]
            - before[n]["vllm:prefix_cache_hits_total"]
        )
        hit = dh / dq if dq > 0 else None
        regime, safe = classify(occ, hit, args.pool_tokens, args.slot_cost)
        verdict["replicas"][n] = {
            "occupancy": round(occ, 3),
            "window_hit_ratio": round(hit, 3) if hit is not None else None,
            "regime": regime,
            "max_safe_slots": safe,
        }
        fleet_safe = safe if fleet_safe is None else min(fleet_safe, safe)

    granted = min(args.requested_slots, fleet_safe or 0)
    verdict.update(
        {
            "requested_slots": args.requested_slots,
            "max_safe_slots": fleet_safe or 0,
            "granted_slots": granted,
            "clamped": granted < args.requested_slots,
        }
    )
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica-urls", required=True, help="name=url,...")
    parser.add_argument("--pool-tokens", type=int, required=True)
    parser.add_argument("--slot-cost", type=int, required=True)
    parser.add_argument("--requested-slots", type=int, required=True)
    parser.add_argument("--window", type=float, default=30.0)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        print(json.dumps(decide(args), indent=2), flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
