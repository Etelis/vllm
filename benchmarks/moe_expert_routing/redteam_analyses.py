# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Red-team addressable analyses from existing data (no GPU).

A. Mixing curve: cross-replica JS heterogeneity vs routing purity
   (addresses "Figure 1 is a purity limiting case").
B. Redundancy requirement vs traffic purity at EP{8,16,32} using vLLM's
   real placement algorithm with replication (addresses "memory story
   never quantified").
C. Small-batch balancedness: does the aggregate-load model hold at
   decode-batch granularity? (addresses "aggregate != per-step").
"""

import json
import os
import sys
import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = sys.argv[1] if len(sys.argv) > 1 else "."
REPO = os.environ.get("VLLM_REPO", ".")

torch_stub = types.ModuleType("torch")
torch_stub.Tensor = np.ndarray
torch_stub.from_numpy = lambda x: x
torch_stub.tensor = np.asarray
sys.modules.setdefault("torch", torch_stub)
with open(f"{REPO}/vllm/distributed/eplb/policy/default.py") as _f:
    src = _f.read()
src = src.replace(
    "from .abstract import AbstractEplbPolicy", "AbstractEplbPolicy = object"
)
mod = types.ModuleType("eplb_default")
exec(compile(src, "default.py", "exec"), mod.__dict__)
Policy = mod.DefaultEplbPolicy


class _T:
    def __init__(self, a):
        self.a = a

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


h = np.load(f"{BASE}/hist.npy").astype(np.float64)
with open(f"{BASE}/table.jsonl") as _fin:
    rows = [json.loads(line) for line in _fin]
idx = {"gsm8k": [], "mbpp": []}
for r in rows:
    if r.get("hist_row") is not None:
        idx[r["domain"]].append(r["hist_row"])
agg = {d: h[i].sum(axis=0) for d, i in idx.items()}
L, E = 48, 128
BLUE, AQUA, YELLOW, RED = "#2a78d6", "#1baf7a", "#eda100", "#e34948"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"


def usage(mat):
    return mat / np.maximum(mat.sum(-1, keepdims=True), 1e-12)


def js_bits(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        return np.where(
            a > 0, a * np.log2(np.maximum(a, 1e-30) / np.maximum(b, 1e-30)), 0
        ).sum(-1)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def ax_style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


# ---------- A. mixing curve ----------
# Replica A gets fraction p of gsm8k + (1-p) of mbpp; replica B mirrors.
ps = np.linspace(0.5, 1.0, 11)
jvals = []
for p in ps:
    a = p * agg["gsm8k"] + (1 - p) * agg["mbpp"]
    b = (1 - p) * agg["gsm8k"] + p * agg["mbpp"]
    jvals.append(float(js_bits(usage(a), usage(b)).mean()))
print("A. mixing curve (purity p -> cross-replica JS bits):")
for p, j in zip(ps, jvals):
    print(f"   p={p:.2f}  JS={j:.3f}")

fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
ax.plot(ps * 100, jvals, color=BLUE, lw=2, marker="o", ms=5)
ax.axhline(jvals[-1], color=INK2, lw=1, ls=":")
measured = [
    (50, 0.001, "load-only (measured)"),
    (100, 0.319, "affinity (measured)"),
]
for x, y, lab in measured:
    ax.plot([x], [y], "s", ms=9, color=RED)
    ax.annotate(
        lab,
        (x, y),
        xytext=(8, -4 if x == 100 else 8),
        textcoords="offset points",
        fontsize=8,
        color=INK,
    )
ax.set_xlabel(
    "routing purity: % of a replica's traffic from its majority domain", color=INK2
)
ax.set_ylabel("cross-replica expert JS (bits)", color=INK2)
ax.set_title(
    "Heterogeneity needs purity: 80% affinity yields only\n"
    "~30% of the divergence - Figure 1 is a high-purity effect",
    color=INK,
    fontsize=11,
    loc="left",
)
ax_style(ax)
fig.tight_layout()
fig.savefig(f"{BASE}/rt_mixing_curve.png", dpi=150, facecolor=SURFACE)


# ---------- B. redundancy requirement vs purity ----------
def fitted(weight, ep, redundant):
    phys = E + redundant
    out = Policy.rebalance_experts(_T(weight), phys, 1, 1, ep)
    phy2log = np.asarray(out[0] if isinstance(out, tuple) else out)
    return phy2log, phys


def balancedness_with_replication(weight, phy2log, ep, phys):
    # Split each logical expert's load evenly across its replicas (ideal
    # dispatch), then measure per-rank balance.
    slots = phys // ep
    bal = []
    for li in range(L):
        counts = np.bincount(phy2log[li], minlength=E).astype(np.float64)
        w = weight[li][phy2log[li]] / np.maximum(counts[phy2log[li]], 1)
        loads = w.reshape(ep, slots).sum(axis=1)
        bal.append(loads.mean() / max(loads.max(), 1e-12))
    return float(np.mean(bal))


mix50 = 0.5 * agg["gsm8k"] + 0.5 * agg["mbpp"]
print("\nB. redundant slots needed for balancedness >= 0.95 (ideal replica dispatch):")
req = {}
for ep in (8, 16, 32):
    for name, w in (("pure", agg["gsm8k"]), ("mixed", mix50)):
        need = None
        for r in range(0, 129, ep):
            p2l, phys = fitted(w, ep, r)
            b = balancedness_with_replication(w, p2l, ep, phys)
            if b >= 0.95:
                need = (r, b)
                break
        req[(ep, name)] = need
        print(
            f"   EP{ep:<3d} {name:6s}: {need[0]} redundant slots (bal {need[1]:.3f})"
            if need
            else f"   EP{ep:<3d} {name:6s}: >128"
        )

fig, ax = plt.subplots(figsize=(6.8, 4.2), facecolor=SURFACE)
eps = [8, 16, 32]
w_ = 0.32
for k, (name, color, lab) in enumerate(
    (("pure", BLUE, "domain-pure traffic"), ("mixed", YELLOW, "50/50 mixed traffic"))
):
    ys = [req[(ep, name)][0] if req[(ep, name)] else 140 for ep in eps]
    ax.bar(np.arange(3) + (k - 0.5) * w_, ys, w_ * 0.9, color=color, label=lab)
    for x, y in zip(np.arange(3) + (k - 0.5) * w_, ys):
        ax.text(x, y + 2, str(int(y)), ha="center", color=INK, fontsize=9)
ax.set_xticks(range(3))
ax.set_xticklabels([f"EP{e}" for e in eps])
ax.set_ylabel("redundant expert slots for balancedness >= 0.95", color=INK2)
ax.set_ylim(0, 40)
ax.set_title(
    "Reversal: domain-pure traffic needs MORE redundancy at EP16 -\n"
    "mixing flattens hot-expert peaks below rank capacity",
    color=INK,
    fontsize=11,
    loc="left",
)
ax.legend(frameon=False, fontsize=9, labelcolor=INK)
ax_style(ax)
fig.tight_layout()
fig.savefig(f"{BASE}/rt_redundancy_vs_purity.png", dpi=150, facecolor=SURFACE)

# ---------- C. small-batch balancedness ----------
rng = np.random.default_rng(0)
gidx = np.array(idx["gsm8k"])
p2l_fit, _ = fitted(agg["gsm8k"], 8, 0)
identity = np.tile(np.arange(E), (L, 1))
print("\nC. per-batch balancedness at EP8 (gsm8k traffic, batches of B prompts):")
res = {}
for B in (4, 16, 64, 256):
    vals = {"fitted": [], "default": []}
    for _ in range(300):
        batch = h[rng.choice(gidx, size=B, replace=True)].sum(axis=0)
        for scen, p2l in (("fitted", p2l_fit), ("default", identity)):
            w = np.take_along_axis(batch, p2l, axis=1)
            loads = w.reshape(L, 8, E // 8).sum(axis=2)
            bal = loads.mean(1) / np.maximum(loads.max(1), 1e-12)
            vals[scen].append(bal.mean())
    res[B] = {s: (float(np.mean(v)), float(np.std(v))) for s, v in vals.items()}
    print(
        f"   B={B:<4d} fitted {res[B]['fitted'][0]:.3f}+-{res[B]['fitted'][1]:.3f}   "
        f"default {res[B]['default'][0]:.3f}+-{res[B]['default'][1]:.3f}"
    )

fig, ax = plt.subplots(figsize=(6.8, 4.2), facecolor=SURFACE)
Bs = list(res)
for scen, color in (("fitted", BLUE), ("default", YELLOW)):
    m = [res[B][scen][0] for B in Bs]
    s = [res[B][scen][1] for B in Bs]
    ax.errorbar(
        range(len(Bs)),
        m,
        yerr=s,
        color=color,
        lw=2,
        marker="o",
        ms=5,
        capsize=3,
        label=scen,
    )
ax.set_xticks(range(len(Bs)))
ax.set_xticklabels([f"B={b}" for b in Bs])
ax.set_ylim(0, 1.02)
ax.set_xlabel("prompts per batch", color=INK2)
ax.set_ylabel("per-batch balancedness (mean +- sd)", color=INK2)
ax.set_title(
    "The fitted-placement advantage survives at small batches\n"
    "(aggregate model is not a large-batch artifact)",
    color=INK,
    fontsize=11,
    loc="left",
)
ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
ax_style(ax)
fig.tight_layout()
fig.savefig(f"{BASE}/rt_smallbatch_balancedness.png", dpi=150, facecolor=SURFACE)
print("\ncharts written")
