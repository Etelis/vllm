# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end analysis pipeline: routed-experts capture -> config recommendation.

Consumes a capture produced by ``live_router_ab.py`` / ``run_experiment.py``
(``hist.npy`` + ``table.jsonl`` with per-request domains) and produces, for a
target EP deployment:

  1. per-workload expert profiles and cross-workload divergence,
  2. per-rank balancedness under default / fitted / anti-fitted placements
     across EP sizes (vLLM's real EPLB placement algorithm),
  3. redundancy economics: redundant slots needed for a balance target,
     under domain-pure vs mixed routing (uniform-hash replica dispatch,
     matching the engine),
  4. purity economics: cross-replica divergence as a function of routing
     purity,
  5. a recommendation block: placement/redundancy/router-affinity guidance
     for the requested EP size and fabric class.

Usage:
    python -m benchmarks.moe_expert_routing.pipeline \\
        --capture ./live_out/i2a_off --ep 8 --fabric nvlink \\
        --balance-target 0.95 --out ./pipeline_report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types

import numpy as np


def _load_policy(repo: str):
    """Load vLLM's EPLB placement policy without importing vllm/torch."""
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = np.ndarray
    torch_stub.from_numpy = lambda x: x
    torch_stub.tensor = np.asarray
    sys.modules.setdefault("torch", torch_stub)
    path = os.path.join(repo, "vllm/distributed/eplb/policy/default.py")
    with open(path) as fin:
        src = fin.read()
    src = src.replace(
        "from .abstract import AbstractEplbPolicy", "AbstractEplbPolicy = object"
    )
    mod = types.ModuleType("eplb_policy")
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod.DefaultEplbPolicy


class _T:
    """Minimal tensor shim for the policy's public API."""

    def __init__(self, a):
        self.a = a

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def usage(mat: np.ndarray) -> np.ndarray:
    return mat / np.maximum(mat.sum(-1, keepdims=True), 1e-12)


def js_bits(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)

    def kl(a, b):
        r = np.log2(np.maximum(a, 1e-30) / np.maximum(b, 1e-30))
        return np.where(a > 0, a * r, 0).sum(-1)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


class Pipeline:
    def __init__(self, capture: str, repo: str):
        self.h = np.load(os.path.join(capture, "hist.npy")).astype(np.float64)
        with open(os.path.join(capture, "table.jsonl")) as fin:
            rows = [json.loads(line) for line in fin]
        self.idx: dict[str, list[int]] = {}
        for r in rows:
            if r.get("hist_row") is not None:
                self.idx.setdefault(r["domain"], []).append(r["hist_row"])
        self.agg = {d: self.h[i].sum(axis=0) for d, i in self.idx.items()}
        self.L, self.E = self.h.shape[1], self.h.shape[2]
        self.policy = _load_policy(repo)

    # -- stage 1/2: profiles and divergence -------------------------------
    def profiles(self) -> dict:
        out: dict = {"domains": {}}
        for d, w in self.agg.items():
            u = usage(w)
            ir = w.max(axis=1) / np.maximum(w.mean(axis=1), 1e-12)
            ent = -np.where(u > 0, u * np.log2(np.maximum(u, 1e-30)), 0).sum(-1)
            out["domains"][d] = {
                "prompts": len(self.idx[d]),
                "mean_layer_ir": round(float(ir.mean()), 2),
                "mean_entropy_bits": round(float(ent.mean()), 2),
                "top8": sorted(int(x) for x in np.argsort(-w.sum(0))[:8]),
            }
        doms = sorted(self.agg)
        if len(doms) >= 2:
            pair = {}
            for i, a in enumerate(doms):
                for b in doms[i + 1 :]:
                    pair[f"{a}|{b}"] = round(
                        float(js_bits(usage(self.agg[a]), usage(self.agg[b])).mean()), 3
                    )
            out["cross_domain_js_bits"] = pair
        return out

    # -- stage 3: placement model ------------------------------------------
    def fitted(self, w: np.ndarray, ep: int, redundant: int = 0) -> np.ndarray:
        out = self.policy.rebalance_experts(_T(w), self.E + redundant, 1, 1, ep)
        return np.asarray(out[0] if isinstance(out, tuple) else out)

    def balancedness(self, w: np.ndarray, p2l: np.ndarray, ep: int, phys: int) -> float:
        slots = phys // ep
        vals = []
        for li in range(self.L):
            counts = np.bincount(p2l[li], minlength=self.E).astype(np.float64)
            x = w[li][p2l[li]] / np.maximum(counts[p2l[li]], 1)
            loads = x.reshape(ep, slots).sum(axis=1)
            vals.append(loads.mean() / max(loads.max(), 1e-12))
        return float(np.mean(vals))

    def placement_table(self, eps=(2, 4, 8, 16, 32)) -> dict:
        ident = np.tile(np.arange(self.E), (self.L, 1))
        doms = sorted(self.agg)
        other = {doms[0]: doms[-1], doms[-1]: doms[0]} if len(doms) >= 2 else {}
        table: dict = {}
        for d, w in self.agg.items():
            for ep in eps:
                if self.E % ep:
                    continue
                row = {
                    "default": round(self.balancedness(w, ident, ep, self.E), 3),
                    "fitted": round(
                        self.balancedness(w, self.fitted(w, ep), ep, self.E), 3
                    ),
                }
                if other:
                    row["anti"] = round(
                        self.balancedness(
                            w, self.fitted(self.agg[other[d]], ep), ep, self.E
                        ),
                        3,
                    )
                table[f"{d}|EP{ep}"] = row
        return table

    # -- stage 4: redundancy economics ---------------------------------------
    def redundancy_needed(self, w: np.ndarray, ep: int, target: float) -> int | None:
        for r in range(0, 4 * ep + 1, ep):
            p2l = self.fitted(w, ep, r)
            if self.balancedness(w, p2l, ep, self.E + r) >= target:
                return r
        return None

    def redundancy_table(self, eps=(8, 16, 32), target: float = 0.95) -> dict:
        doms = sorted(self.agg)
        mixed = sum(self.agg.values()) / len(self.agg)
        out: dict = {}
        for ep in eps:
            if self.E % ep:
                continue
            out[f"EP{ep}"] = {
                "pure": self.redundancy_needed(self.agg[doms[0]], ep, target),
                "mixed": self.redundancy_needed(mixed, ep, target),
            }
        return out

    # -- stage 5: purity economics -------------------------------------------
    def purity_curve(self) -> dict:
        doms = sorted(self.agg)
        if len(doms) < 2:
            return {}
        a, b = self.agg[doms[0]], self.agg[doms[-1]]
        out = {}
        for p in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            am = p * a + (1 - p) * b
            bm = (1 - p) * a + p * b
            out[f"{p:.1f}"] = round(float(js_bits(usage(am), usage(bm)).mean()), 3)
        return out

    # Measured live on Qwen3-30B-A3B EP4/H100 (redundant slots 0/16/32:
    # KV pool 514,192 / 495,344 / 476,496 tokens per rank - exactly linear).
    KV_TOKENS_PER_REDUNDANT_SLOT_PER_RANK = 4712

    # -- stage 6: recommendation ----------------------------------------------
    def recommend(self, ep: int, fabric: str, target: float) -> dict:
        table = self.placement_table((ep,))
        red = self.redundancy_table((ep,), target)
        default_bal = float(np.mean([v["default"] for v in table.values()]))
        fitted_bal = float(np.mean([v["fitted"] for v in table.values()]))
        headroom = fitted_bal - default_bal
        red_row = red.get(f"EP{ep}", {})
        conv = {"nvlink": 0.01, "tcp": 0.05, "rdma": None}.get(fabric)
        rec = {
            "ep": ep,
            "fabric": fabric,
            "balancedness_default": round(default_bal, 3),
            "balancedness_fitted": round(fitted_bal, 3),
            "headroom_pp": round(headroom * 100, 1),
            "redundant_slots_for_target": red_row,
            "expected_throughput_gain": (
                f"~{conv * 100:.0f}% (measured regime)"
                if conv
                else "unmeasured (validate on this fabric)"
            ),
            "guidance": [],
        }
        g = rec["guidance"]
        if headroom < 0.05:
            g.append(
                "Placement fit is nearly free at this width: run EPLB with the "
                "balancedness threshold trigger only; skip periodic rearrangement."
            )
        else:
            g.append(
                "Meaningful placement headroom: enable EPLB with the threshold "
                "trigger (rebalance_threshold ~0.7-0.8, cooldown >= 2000)."
            )
        pure_r, mixed_r = red_row.get("pure"), red_row.get("mixed")
        if pure_r is not None and mixed_r is not None and pure_r > mixed_r:
            kv_cost = (
                (pure_r - mixed_r) // ep * self.KV_TOKENS_PER_REDUNDANT_SLOT_PER_RANK
            )
            g.append(
                f"Affinity purity costs memory here: pure traffic needs {pure_r} "
                f"redundant slots vs {mixed_r} mixed for balancedness>={target} "
                f"(~{kv_cost:,} KV tokens/rank at the measured "
                f"{self.KV_TOKENS_PER_REDUNDANT_SLOT_PER_RANK:,}/slot). "
                "Cap prefix-affinity scorer weight or spend the slots knowingly."
            )
        elif pure_r is not None and mixed_r is not None and pure_r < mixed_r:
            g.append(
                f"Affinity purity saves memory here ({pure_r} vs {mixed_r} "
                "redundant slots); prefer strong prefix-affinity routing."
            )
        else:
            g.append(
                "Redundancy needs are purity-insensitive at this width; choose "
                "router weights on cache-hit grounds alone."
            )
        g.append(
            "Cache term (dominant when active): if the tenant prefix working "
            "set exceeds one replica's KV pool but fits the fleet partitioned, "
            "prefix-affinity routing is worth far more than any expert-side "
            "term (measured: +77% goodput / -48% p50 at 3x pool overload vs "
            "27.8%-hit thrashing). Compute: working_set = tenants x prefix "
            "tokens; compare against per-replica pool (and remember redundant "
            "slots shrink that pool by 4,712 tokens each)."
        )
        if fabric == "nvlink":
            g.append(
                "On NVLink-class fabric the placement effect on throughput is "
                "~1%: do not trade anything else for balance."
            )
        return rec

    def run(self, ep: int, fabric: str, target: float, out_dir: str) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        report = {
            "profiles": self.profiles(),
            "placement": self.placement_table(),
            "redundancy": self.redundancy_table(target=target),
            "purity_curve": self.purity_curve(),
            "recommendation": self.recommend(ep, fabric, target),
        }
        with open(os.path.join(out_dir, "report.json"), "w") as fout:
            json.dump(report, fout, indent=2)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture", required=True, help="dir with hist.npy+table.jsonl"
    )
    parser.add_argument("--repo", default=os.environ.get("VLLM_REPO", "."))
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--fabric", choices=("nvlink", "tcp", "rdma"), default="nvlink")
    parser.add_argument("--balance-target", type=float, default=0.95)
    parser.add_argument("--out", default="./pipeline_report")
    args = parser.parse_args()

    pipe = Pipeline(args.capture, args.repo)
    report = pipe.run(args.ep, args.fabric, args.balance_target, args.out)
    print(json.dumps(report["recommendation"], indent=2))
    print(f"\nfull report: {args.out}/report.json")


if __name__ == "__main__":
    main()
