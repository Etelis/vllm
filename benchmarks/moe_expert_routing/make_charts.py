# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step 1: expert-distribution characterization charts (regular EP capture)."""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = sys.argv[1] if len(sys.argv) > 1 else "."
BLUE, AQUA = "#2a78d6", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
DOM_COLOR = {"gsm8k": BLUE, "mbpp": AQUA}
DOM_LABEL = {"gsm8k": "GSM8K (math)", "mbpp": "MBPP (code)"}

h = np.load(f"{BASE}/hist.npy").astype(np.float64)  # (N, 48, 128)
with open(f"{BASE}/table.jsonl") as _fin:
    rows = [json.loads(line) for line in _fin]
idx = {"gsm8k": [], "mbpp": []}
for r in rows:
    if r.get("hist_row") is not None:
        idx[r["domain"]].append(r["hist_row"])

agg = {d: h[i].sum(axis=0) for d, i in idx.items()}  # (48, 128) per domain
L, E = 48, 128


def usage(mat):  # per-layer distribution
    s = mat.sum(axis=-1, keepdims=True)
    return mat / np.maximum(s, 1e-12)


def js_bits(p, q):  # per-layer JS divergence in bits
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return np.where(
            mask, a * np.log2(np.maximum(a, 1e-30) / np.maximum(b, 1e-30)), 0
        ).sum(-1)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


ug, um = usage(agg["gsm8k"]), usage(agg["mbpp"])
js = js_bits(ug, um)  # (48,)
top8 = {d: np.argsort(-agg[d], axis=1)[:, :8] for d in agg}
jac = np.array(
    [
        len(set(top8["gsm8k"][li]) & set(top8["mbpp"][li]))
        / len(set(top8["gsm8k"][li]) | set(top8["mbpp"][li]))
        for li in range(L)
    ]
)
ir = {d: (agg[d].max(axis=1) / agg[d].mean(axis=1)) for d in agg}
ent = {
    d: -np.where(
        usage(agg[d]) > 0, usage(agg[d]) * np.log2(np.maximum(usage(agg[d]), 1e-30)), 0
    ).sum(-1)
    for d in agg
}

print("=== STEP-1 VALIDATION (expect ~ RESULTS.md values) ===")
print(
    f"JS mean {js.mean():.3f} bits (expect ~0.318) | "
    f"max {js.max():.3f} at layer {js.argmax()} (expect ~0.516 @ 6)"
)
g8 = np.argsort(-agg["gsm8k"].sum(0))[:8]
m8 = np.argsort(-agg["mbpp"].sum(0))[:8]
gj = len(set(g8) & set(m8)) / len(set(g8) | set(m8))
print(
    f"global top-8 Jaccard {gj:.3f} (expect ~0.132) | "
    f"gsm8k top8 {sorted(g8)} | mbpp top8 {sorted(m8)}"
)
for d in ("gsm8k", "mbpp"):
    print(
        f"{d}: mean layer IR {ir[d].mean():.2f} | "
        f"mean entropy {ent[d].mean():.2f}/7.0 bits"
    )

style = dict(facecolor=SURFACE)


def ax_style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


# --- Chart 1: expert load profile at the most divergent layer
LSHOW = int(js.argmax())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), **style)
top8share = {}
for d in ("gsm8k", "mbpp"):
    share = np.sort(agg[d][LSHOW] / agg[d][LSHOW].sum())[::-1]
    top8share[d] = np.cumsum(share)[7] * 100
    a1.plot(
        np.arange(1, E + 1), share * 100, color=DOM_COLOR[d], lw=2, label=DOM_LABEL[d]
    )
    a2.plot(
        np.arange(1, E + 1),
        np.cumsum(share) * 100,
        color=DOM_COLOR[d],
        lw=2,
        label=DOM_LABEL[d],
    )
a1.set_yscale("log")
a1.set_xlabel(f"expert rank within layer {LSHOW} (sorted by load)", color=INK2)
a1.set_ylabel("share of layer tokens (%)", color=INK2)
a1.set_title(
    f"Layer {LSHOW}: a few experts dominate each workload",
    color=INK,
    fontsize=11,
    loc="left",
)
uniform = 100 / E
a1.axhline(uniform, color=INK2, lw=1, ls=":")
a1.text(70, uniform * 1.25, "uniform (0.78%)", color=INK2, fontsize=8)
a2.axvline(8, color=INK2, lw=1, ls=":")
offsets = {"gsm8k": 6, "mbpp": -9}
for d in ("gsm8k", "mbpp"):
    a2.annotate(
        f"top-8 = {top8share[d]:.0f}%",
        xy=(8, top8share[d]),
        xytext=(14, top8share[d] + offsets[d]),
        color=DOM_COLOR[d],
        fontsize=9,
    )
a2.set_xlabel("top-N experts", color=INK2)
a2.set_ylabel("cumulative share (%)", color=INK2)
a2.set_title(
    f"Layer {LSHOW}: top-8 of 128 carry "
    f"{top8share['gsm8k']:.0f}% (math) / {top8share['mbpp']:.0f}% (code)",
    color=INK,
    fontsize=11,
    loc="left",
)
for a in (a1, a2):
    ax_style(a)
    a.legend(frameon=False, fontsize=9, labelcolor=INK)
fig.tight_layout()
fig.savefig(f"{BASE}/s1_expert_load_profiles.png", dpi=150, facecolor=SURFACE)

# --- Chart 2: per-prompt distributions
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), **style)
for d in ("gsm8k", "mbpp"):
    per_prompt = h[idx[d]]  # (n,48,128)
    srt = np.sort(per_prompt, axis=-1)[:, :, ::-1]
    top8sh = (srt[:, :, :8].sum(-1) / np.maximum(per_prompt.sum(-1), 1e-12)).mean(
        -1
    ) * 100
    u = per_prompt / np.maximum(per_prompt.sum(-1, keepdims=True), 1e-12)
    pent = (-np.where(u > 0, u * np.log2(np.maximum(u, 1e-30)), 0).sum(-1)).mean(
        -1
    )  # mean layer entropy
    a1.hist(
        top8sh, bins=30, histtype="step", lw=2, color=DOM_COLOR[d], label=DOM_LABEL[d]
    )
    a2.hist(
        pent, bins=30, histtype="step", lw=2, color=DOM_COLOR[d], label=DOM_LABEL[d]
    )
a1.axvline(8 / 128 * 100, color=INK2, lw=1, ls=":")
a1.text(8, 5, "uniform (6.25%)", color=INK2, fontsize=8, rotation=90)
a1.set_xlabel(
    "share of a prompt's routing in its top-8 experts, mean over layers (%)", color=INK2
)
a1.set_ylabel("prompts", color=INK2)
a1.set_title(
    "Per prompt, ~30% of routing lands in 8 of 128 experts (5x uniform)",
    color=INK,
    fontsize=11,
    loc="left",
)
a2.set_xlabel("per-prompt mean layer entropy (bits, max 7)", color=INK2)
a2.set_ylabel("prompts", color=INK2)
a2.set_title(
    "...but usage is concentrated well below uniform",
    color=INK,
    fontsize=11,
    loc="left",
)
for a in (a1, a2):
    ax_style(a)
    a.legend(frameon=False, fontsize=9, labelcolor=INK)
fig.tight_layout()
fig.savefig(f"{BASE}/s1_per_prompt_concentration.png", dpi=150, facecolor=SURFACE)

# --- Chart 3: overlap by layer (JS + Jaccard)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), **style)
a1.plot(range(L), js, color=BLUE, lw=2)
a1.axhline(js.mean(), color=INK2, lw=1, ls=":")
a1.text(
    L - 1, js.mean() + 0.01, f"mean {js.mean():.2f}", color=INK2, fontsize=9, ha="right"
)
a1.set_xlabel("MoE layer", color=INK2)
a1.set_ylabel("Jensen-Shannon divergence (bits)", color=INK2)
a1.set_title(
    "Math vs code expert usage diverges at every layer",
    color=INK,
    fontsize=11,
    loc="left",
)
a2.plot(range(L), jac, color=BLUE, lw=2)
a2.axhline(jac.mean(), color=INK2, lw=1, ls=":")
a2.text(
    L - 1,
    jac.mean() + 0.02,
    f"mean {jac.mean():.2f}",
    color=INK2,
    fontsize=9,
    ha="right",
)
a2.set_ylim(0, 1)
a2.set_xlabel("MoE layer", color=INK2)
a2.set_ylabel("top-8 hot-expert Jaccard overlap", color=INK2)
a2.set_title("Hot-expert sets barely overlap", color=INK, fontsize=11, loc="left")
for a in (a1, a2):
    ax_style(a)
fig.tight_layout()
fig.savefig(f"{BASE}/s1_overlap_by_layer.png", dpi=150, facecolor=SURFACE)

# --- Chart 4: per-expert load scatter (identity of the divergence)
fig, ax = plt.subplots(figsize=(6.4, 6), **style)
gs = agg["gsm8k"][LSHOW] / agg["gsm8k"][LSHOW].sum() * 100
ms = agg["mbpp"][LSHOW] / agg["mbpp"][LSHOW].sum() * 100
ax.scatter(gs, ms, s=26, color=BLUE, alpha=0.65, edgecolors=SURFACE, linewidths=1)
lim = max(gs.max(), ms.max()) * 1.15
ax.plot([0.001, lim], [0.001, lim], color=INK2, lw=1, ls=":")
ax.set_xscale("log")
ax.set_yscale("log")
floor = max(1e-4, min(gs[gs > 0].min(), ms[ms > 0].min()) * 0.5)
ax.set_xlim(floor, lim)
ax.set_ylim(floor, lim)
lay_g8 = list(np.argsort(-agg["gsm8k"][LSHOW])[:3])
lay_m8 = list(np.argsort(-agg["mbpp"][LSHOW])[:3])
for e in sorted(set(int(x) for x in lay_g8 + lay_m8)):
    ax.annotate(
        f"e{e}",
        (gs[e], ms[e]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
        color=INK,
    )
ax.set_xlabel(f"share of GSM8K tokens in layer {LSHOW} (%)", color=INK2)
ax.set_ylabel(f"share of MBPP tokens in layer {LSHOW} (%)", color=INK2)
ax.set_title(
    f"Layer {LSHOW}: each workload has its own hot experts"
    "\n(off-diagonal = workload-exclusive)",
    color=INK,
    fontsize=11,
    loc="left",
)
ax_style(ax)
fig.tight_layout()
fig.savefig(f"{BASE}/s1_expert_scatter.png", dpi=150, facecolor=SURFACE)

json.dump(
    {
        "js_mean_bits": round(float(js.mean()), 4),
        "js_max_bits": round(float(js.max()), 4),
        "js_max_layer": int(js.argmax()),
        "global_top8_jaccard": round(gj, 4),
        "layer_top8_jaccard_mean": round(float(jac.mean()), 4),
        "ir_mean": {d: round(float(ir[d].mean()), 2) for d in ir},
        "entropy_mean_bits": {d: round(float(ent[d].mean()), 2) for d in ent},
        "gsm8k_top8": sorted(int(x) for x in g8),
        "mbpp_top8": sorted(int(x) for x in m8),
        "n_prompts": {d: len(i) for d, i in idx.items()},
    },
    open(f"{BASE}/step1_summary.json", "w"),  # noqa: SIM115
    indent=2,
)
print("charts written")
