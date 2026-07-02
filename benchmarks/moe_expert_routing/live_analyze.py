#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze one or two ``live_router_ab.py`` runs: per-replica expert skew.

For each arm: group requests by serving replica, aggregate per-replica
layer x expert histograms, and report the domain mix each replica received,
per-replica imbalance ratio, pairwise cross-replica Jensen-Shannon divergence,
and per-replica prefix-cache hits. With two arms, prints the headline
affinity-vs-baseline comparison (the live analogue of ``worker_ab.py``).

Replica attribution: the ``endpoint`` column when present, else an
``id -> replica`` JSON map produced from pod logs (``--attribution FILE``).

Example:
    python -m benchmarks.moe_expert_routing.live_analyze \\
        --arm affinity=./live_out/affinity --arm baseline=./live_out/baseline \\
        --attribution ./live_out/attribution.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from benchmarks.moe_expert_routing.expert_stats import (
    js_divergence_per_layer,
    per_layer_imbalance_ratio,
    usage_distribution,
)


def _load_arm(out_dir: str, attribution: dict[str, str]) -> dict:
    hists = np.load(os.path.join(out_dir, "hist.npy"))
    rows = []
    with open(os.path.join(out_dir, "table.jsonl")) as fin:
        for line in fin:
            rows.append(json.loads(line))
    with open(os.path.join(out_dir, "config.json")) as fin:
        config = json.load(fin)

    by_replica: dict[str, list[int]] = defaultdict(list)
    domain_mix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    cached: dict[str, list[int]] = defaultdict(list)
    unattributed = 0
    for row in rows:
        if not row.get("ok") or row.get("hist_row") is None:
            continue
        replica = row.get("endpoint") or attribution.get(row.get("id") or "")
        if not replica:
            unattributed += 1
            continue
        by_replica[replica].append(row["hist_row"])
        domain_mix[replica][row["domain"]] += 1
        if row.get("cached_tokens") is not None:
            cached[replica].append(row["cached_tokens"])
    return {
        "config": config,
        "hists": hists,
        "by_replica": dict(by_replica),
        "domain_mix": {r: dict(d) for r, d in domain_mix.items()},
        "cached": dict(cached),
        "unattributed": unattributed,
    }


def _replica_report(arm_name: str, arm: dict) -> dict:
    replicas = sorted(arm["by_replica"])
    agg = {}
    for r in replicas:
        agg[r] = arm["hists"][arm["by_replica"][r]].sum(axis=0).astype(np.float64)

    per_replica = {}
    for r in replicas:
        ir = per_layer_imbalance_ratio(agg[r])
        per_replica[r] = {
            "requests": len(arm["by_replica"][r]),
            "domain_mix": arm["domain_mix"][r],
            "mean_layer_ir": round(float(np.mean(ir)), 3),
            "mean_cached_tokens": (
                round(float(np.mean(arm["cached"][r])), 1)
                if arm["cached"].get(r)
                else None
            ),
        }

    js_pairs = {}
    js_vals = []
    for i, a in enumerate(replicas):
        for b in replicas[i + 1 :]:
            js = float(
                np.mean(
                    js_divergence_per_layer(
                        usage_distribution(agg[a]), usage_distribution(agg[b])
                    )
                )
            )
            js_pairs[f"{a}|{b}"] = round(js, 4)
            js_vals.append(js)

    report = {
        "arm": arm_name,
        "num_replicas": len(replicas),
        "unattributed": arm["unattributed"],
        "per_replica": per_replica,
        "cross_replica_js_mean": round(float(np.mean(js_vals)), 4) if js_vals else None,
        "cross_replica_js_max": round(float(np.max(js_vals)), 4) if js_vals else None,
        "cross_replica_js_pairs": js_pairs,
        "mean_per_replica_ir": round(
            float(np.mean([v["mean_layer_ir"] for v in per_replica.values()])), 3
        )
        if per_replica
        else None,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="name=out_dir; repeat for the second arm",
    )
    parser.add_argument("--attribution", default=None, help="id->replica JSON map")
    parser.add_argument("--out", default=None, help="write full report JSON here")
    args = parser.parse_args()

    attribution: dict[str, str] = {}
    if args.attribution and os.path.exists(args.attribution):
        with open(args.attribution) as fin:
            attribution = json.load(fin)

    reports = []
    for spec in args.arm:
        name, _, out_dir = spec.partition("=")
        arm = _load_arm(out_dir, attribution)
        report = _replica_report(name, arm)
        reports.append(report)
        print(f"\n=== arm: {name} ===")
        print(json.dumps(report, indent=2))

    if len(reports) == 2:
        a, b = reports
        print("\n=== headline (arm[0] vs arm[1]) ===")
        print(
            f"cross-replica JS mean: {a['arm']}={a['cross_replica_js_mean']} vs "
            f"{b['arm']}={b['cross_replica_js_mean']}"
        )
        print(
            f"per-replica IR mean:   {a['arm']}={a['mean_per_replica_ir']} vs "
            f"{b['arm']}={b['mean_per_replica_ir']}"
        )

    if args.out:
        with open(args.out, "w") as fout:
            json.dump(reports, fout, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
