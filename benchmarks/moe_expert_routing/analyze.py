#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze collected routed-experts stats from ``run_experiment.py``.

Produces, per domain and cross-domain:
  * aggregated per-layer hot experts + expert-load imbalance (IR / Gini / entropy),
  * cross-domain Jensen-Shannon divergence and top-expert Jaccard overlap per
    layer (the "experts specialize by domain" signal),
  * per-prompt prefix-cache-hit vs expert-usage-concentration correlation,
  * an illustrative per-worker (block-EP) imbalance ratio.

Writes ``analysis_summary.json`` under ``--out-dir`` and prints a console report.

Example:
    .venv/bin/python -m benchmarks.moe_expert_routing.analyze \\
        --out-dir ./moe_stats_out --domains gsm8k,mbpp --num-workers 8
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from benchmarks.moe_expert_routing.expert_stats import (
    aggregate_histograms,
    experts_to_worker_load,
    js_divergence_per_layer,
    per_layer_entropy_bits,
    per_layer_gini,
    per_layer_imbalance_ratio,
    top_experts_per_layer,
)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def load_domain(out_dir: str, name: str) -> dict:
    dom_dir = os.path.join(out_dir, name)
    hist_nle = np.load(os.path.join(dom_dir, "hist.npy"))  # (N, L, E)
    kept_idx = np.load(os.path.join(dom_dir, "kept_idx.npy"))
    with open(os.path.join(dom_dir, "config.json")) as fin:
        config = json.load(fin)
    table_by_idx: dict[int, dict] = {}
    with open(os.path.join(dom_dir, "table.jsonl")) as fin:
        for line in fin:
            row = json.loads(line)
            table_by_idx[int(row["idx"])] = row
    return {
        "name": name,
        "hist_nle": hist_nle,
        "agg_le": aggregate_histograms(list(hist_nle)),
        "kept_idx": kept_idx,
        "table_by_idx": table_by_idx,
        "config": config,
    }


def domain_report(dom: dict, num_workers: int, top_n: int) -> dict:
    agg = dom["agg_le"]  # (L, E)
    num_layers, num_experts = agg.shape
    ir = per_layer_imbalance_ratio(agg)
    gini = per_layer_gini(agg)
    entropy = per_layer_entropy_bits(agg)

    # Illustrative per-worker imbalance under default block-EP placement.
    total_e = agg.sum(axis=0)  # (E,) load summed over layers
    worker_ir = float("nan")
    if num_experts % num_workers == 0:
        worker_load = experts_to_worker_load(total_e, num_workers)
        worker_ir = float(worker_load.max() / max(worker_load.mean(), 1e-9))

    # Per-prompt: cache-hit fraction vs expert-usage concentration (mean Gini).
    hist_nle = dom["hist_nle"]
    cache_frac: list[float] = []
    concentration: list[float] = []
    for row_i, idx in enumerate(dom["kept_idx"].tolist()):
        row = dom["table_by_idx"].get(int(idx), {})
        ptok, cached = row.get("prompt_tokens"), row.get("cached_tokens")
        if not ptok or cached is None:
            continue
        cache_frac.append(cached / ptok)
        concentration.append(float(per_layer_gini(hist_nle[row_i]).mean()))

    corr = _pearson(np.array(cache_frac), np.array(concentration))
    return {
        "name": dom["name"],
        "num_prompts": int(hist_nle.shape[0]),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "mean_IR": float(ir.mean()),
        "max_IR": float(ir.max()),
        "mean_gini": float(gini.mean()),
        "mean_entropy_bits": float(entropy.mean()),
        "worker_IR_illustrative": worker_ir,
        "global_top_experts": np.argsort(-total_e)[:top_n].tolist(),
        "top_experts_deepest_layer": top_experts_per_layer(
            agg[num_layers - 1 : num_layers], top_n
        )[0].tolist(),
        "cache_vs_concentration_pearson": corr,
        "mean_cache_frac": float(np.mean(cache_frac)) if cache_frac else None,
    }


def cross_domain(dom_a: dict, dom_b: dict, top_n: int) -> dict:
    a, b = dom_a["agg_le"], dom_b["agg_le"]
    if a.shape != b.shape:
        raise ValueError(f"layer/expert mismatch {a.shape} vs {b.shape}")
    js = js_divergence_per_layer(a, b)  # (L,)
    top_a = top_experts_per_layer(a, top_n)
    top_b = top_experts_per_layer(b, top_n)
    jaccard = []
    for layer in range(a.shape[0]):
        sa, sb = set(top_a[layer].tolist()), set(top_b[layer].tolist())
        jaccard.append(len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0)
    jaccard = np.array(jaccard)
    return {
        "pair": f"{dom_a['name']}_vs_{dom_b['name']}",
        "mean_JS_bits": float(js.mean()),
        "max_JS_bits": float(js.max()),
        "most_divergent_layer": int(js.argmax()),
        "mean_top{}_jaccard".format(top_n): float(jaccard.mean()),
        "per_layer_JS_bits": [round(float(v), 4) for v in js],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="./moe_stats_out")
    parser.add_argument("--domains", default="gsm8k,mbpp")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()

    names = [d.strip() for d in args.domains.split(",") if d.strip()]
    domains = [load_domain(args.out_dir, name) for name in names]
    reports = [domain_report(d, args.num_workers, args.top_n) for d in domains]

    print("\n================ per-domain expert usage ================")
    for rep in reports:
        print(
            f"\n[{rep['name']}]  prompts={rep['num_prompts']}  "
            f"layers={rep['num_layers']}  experts={rep['num_experts']}"
        )
        print(
            f"  expert-load  mean IR={rep['mean_IR']:.3f}  max IR={rep['max_IR']:.3f}  "
            f"mean Gini={rep['mean_gini']:.3f}  "
            f"mean entropy={rep['mean_entropy_bits']:.2f} bits"
        )
        print(
            f"  illustrative per-worker IR (block-EP, R={args.num_workers}): "
            f"{rep['worker_IR_illustrative']:.3f}"
        )
        print(f"  global hottest experts: {rep['global_top_experts']}")
        print(
            f"  cache-hit vs concentration Pearson r = "
            f"{rep['cache_vs_concentration_pearson']:.3f}  "
            f"(mean cache frac={rep['mean_cache_frac']})"
        )

    cross = None
    if len(domains) >= 2:
        cross = cross_domain(domains[0], domains[1], args.top_n)
        print("\n================ cross-domain specialization ================")
        print(f"  {cross['pair']}")
        print(
            f"  expert-usage JS: mean={cross['mean_JS_bits']:.3f} bits  "
            f"max={cross['max_JS_bits']:.3f} bits "
            f"@layer {cross['most_divergent_layer']}"
        )
        jac_key = f"mean_top{args.top_n}_jaccard"
        print(
            f"  top-{args.top_n} expert overlap (Jaccard): {cross[jac_key]:.3f}  "
            "(low = domains use different experts)"
        )

    summary = {"per_domain": reports, "cross_domain": cross}
    with open(os.path.join(args.out_dir, "analysis_summary.json"), "w") as fout:
        json.dump(summary, fout, indent=2)
    print(f"\nWrote {os.path.join(args.out_dir, 'analysis_summary.json')}")


if __name__ == "__main__":
    main()
