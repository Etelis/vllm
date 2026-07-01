#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache-affinity vs shuffled worker A/B: does KV-routing manufacture imbalance?

Emulates ``num_workers`` independent MoE replicas fed by an external router,
reconstructed from the per-prompt expert-usage histograms collected by
``run_experiment.py``. Because per-prompt expert usage is a property of the
prompt (its hidden states drive the router), summing the per-prompt ``(L, E)``
histograms of the prompts assigned to a replica faithfully reproduces that
replica's expert load.

Two routing policies over the *same* request set, with equal prompts per worker:

  * ``affinity``  -- each worker receives one domain (what a KV-cache-aware
    router produces: same-prefix/same-domain prompts co-located).
  * ``shuffled``  -- prompts assigned round-robin ignoring domain (domain-mixed,
    what random / round-robin routing produces).

If experts specialize by domain, ``affinity`` makes each worker's expert load
concentrated and the workers mutually divergent, while ``shuffled`` gives every
worker the same balanced mix. The gap is the routing-induced ("manufactured")
imbalance.

Example:
    python -m benchmarks.moe_expert_routing.worker_ab \\
        --out-dir ./moe_stats_out --domains gsm8k,mbpp --num-workers 4
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from benchmarks.moe_expert_routing.expert_stats import (
    js_divergence_per_layer,
    per_layer_imbalance_ratio,
)


def _load_all(out_dir: str, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return stacked per-prompt histograms ``(N, L, E)`` and domain labels."""
    hists, labels = [], []
    for domain_id, name in enumerate(names):
        hist_nle = np.load(os.path.join(out_dir, name, "hist.npy"))
        hists.append(hist_nle)
        labels.append(np.full(hist_nle.shape[0], domain_id, dtype=np.int64))
    return np.concatenate(hists, axis=0), np.concatenate(labels)


def _worker_loads(
    hist_nle: np.ndarray, assignment: np.ndarray, num_workers: int
) -> np.ndarray:
    """Sum per-prompt histograms per worker -> ``(W, L, E)`` load tensor."""
    num_layers, num_experts = hist_nle.shape[1:]
    loads = np.zeros((num_workers, num_layers, num_experts), dtype=np.int64)
    for worker in range(num_workers):
        mask = assignment == worker
        if mask.any():
            loads[worker] = hist_nle[mask].sum(axis=0)
    return loads


def _affinity_assignment(
    labels: np.ndarray, num_workers: int, num_domains: int
) -> np.ndarray:
    """Domain-homogeneous: split workers across domains, round-robin within."""
    if num_workers % num_domains != 0:
        raise ValueError(
            f"num_workers ({num_workers}) must be divisible by num_domains "
            f"({num_domains}) for a clean affinity split"
        )
    per_domain = num_workers // num_domains
    assignment = np.empty(len(labels), dtype=np.int64)
    for domain_id in range(num_domains):
        idx = np.flatnonzero(labels == domain_id)
        base = domain_id * per_domain
        assignment[idx] = base + (np.arange(len(idx)) % per_domain)
    return assignment


def _shuffled_assignment(
    num_prompts: int, num_workers: int, rng: np.random.Generator
) -> np.ndarray:
    """Domain-mixed: random permutation, round-robin across all workers."""
    perm = rng.permutation(num_prompts)
    assignment = np.empty(num_prompts, dtype=np.int64)
    assignment[perm] = np.arange(num_prompts) % num_workers
    return assignment


def _metrics(loads: np.ndarray) -> dict:
    """Per-worker peakedness (IR) and cross-worker usage divergence (JS)."""
    num_workers = loads.shape[0]
    worker_ir = np.array(
        [float(per_layer_imbalance_ratio(loads[w]).mean()) for w in range(num_workers)]
    )
    pair_js = []
    for a in range(num_workers):
        for b in range(a + 1, num_workers):
            pair_js.append(float(js_divergence_per_layer(loads[a], loads[b]).mean()))
    return {
        "mean_per_worker_IR": float(worker_ir.mean()),
        "mean_cross_worker_JS_bits": float(np.mean(pair_js)) if pair_js else 0.0,
        "max_cross_worker_JS_bits": float(np.max(pair_js)) if pair_js else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="./moe_stats_out")
    parser.add_argument("--domains", default="gsm8k,mbpp")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    names = [d.strip() for d in args.domains.split(",") if d.strip()]
    hist_nle, labels = _load_all(args.out_dir, names)
    rng = np.random.default_rng(args.seed)

    affinity = _affinity_assignment(labels, args.num_workers, len(names))
    shuffled = _shuffled_assignment(len(labels), args.num_workers, rng)

    aff = _metrics(_worker_loads(hist_nle, affinity, args.num_workers))
    shu = _metrics(_worker_loads(hist_nle, shuffled, args.num_workers))

    result = {
        "num_workers": args.num_workers,
        "num_prompts": int(len(labels)),
        "domains": names,
        "affinity": aff,
        "shuffled": shu,
        "manufactured_IR_gap": aff["mean_per_worker_IR"] - shu["mean_per_worker_IR"],
        "manufactured_JS_gap": (
            aff["mean_cross_worker_JS_bits"] - shu["mean_cross_worker_JS_bits"]
        ),
    }
    with open(os.path.join(args.out_dir, "worker_ab.json"), "w") as fout:
        json.dump(result, fout, indent=2)

    print("\n=========== cache-affinity vs shuffled worker A/B ===========")
    print(f"  {args.num_workers} workers, {len(labels)} prompts, domains={names}")
    print("  policy    per-worker IR   cross-worker JS (bits)")
    print(
        f"  affinity     {aff['mean_per_worker_IR']:7.3f}          "
        f"{aff['mean_cross_worker_JS_bits']:.3f}"
    )
    print(
        f"  shuffled     {shu['mean_per_worker_IR']:7.3f}          "
        f"{shu['mean_cross_worker_JS_bits']:.3f}"
    )
    print(
        f"  --> manufactured imbalance: IR gap "
        f"{result['manufactured_IR_gap']:+.3f}, "
        f"cross-worker JS gap {result['manufactured_JS_gap']:+.3f} bits"
    )
    print(f"\nWrote {os.path.join(args.out_dir, 'worker_ab.json')}")


if __name__ == "__main__":
    main()
