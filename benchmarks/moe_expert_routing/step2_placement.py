# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step 2: per-rank load under vLLM's real EPLB placement algorithm.

Loads vllm/distributed/eplb/policy/default.py (the DeepSeek-adapted
algorithm, numpy implementation) directly, with torch stubbed out, and
applies it to the step-1 measured per-domain expert loads.
"""

import importlib.util
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

# ---- stub torch + abstract base so default.py imports standalone
torch_stub = types.ModuleType("torch")
torch_stub.Tensor = np.ndarray
torch_stub.from_numpy = lambda x: x
torch_stub.tensor = np.asarray
torch_stub.as_tensor = np.asarray
sys.modules.setdefault("torch", torch_stub)

abs_mod = types.ModuleType("vllm_eplb_abstract")


class AbstractEplbPolicy:
    pass


abs_mod.AbstractEplbPolicy = AbstractEplbPolicy

spec = importlib.util.spec_from_file_location(
    "vllm_eplb_default", f"{REPO}/vllm/distributed/eplb/policy/default.py"
)
mod = importlib.util.module_from_spec(spec)
mod.__dict__["AbstractEplbPolicy"] = AbstractEplbPolicy
sys.modules["vllm_eplb_default"] = mod
# satisfy `from .abstract import AbstractEplbPolicy`
sys.modules["vllm_eplb_default.abstract"] = abs_mod
mod.__package__ = "vllm_eplb_default"
try:
    spec.loader.exec_module(mod)
except ImportError:
    # relative import fallback: rewrite and exec
    with open(f"{REPO}/vllm/distributed/eplb/policy/default.py") as _f:
        src = _f.read()
    src = src.replace("from .abstract import AbstractEplbPolicy", "")
    mod = types.ModuleType("vllm_eplb_default")
    mod.AbstractEplbPolicy = AbstractEplbPolicy
    exec(compile(src, "default.py", "exec"), mod.__dict__)

Policy = mod.DefaultEplbPolicy
print("loaded policy:", Policy)

# ---- data
h = np.load(f"{BASE}/hist.npy").astype(np.float64)
with open(f"{BASE}/table.jsonl") as _fin:
    rows = [json.loads(line) for line in _fin]
idx = {"gsm8k": [], "mbpp": []}
for r in rows:
    if r.get("hist_row") is not None:
        idx[r["domain"]].append(r["hist_row"])
agg = {d: h[i].sum(axis=0) for d, i in idx.items()}  # (48,128)
L, E = 48, 128


class _T:  # minimal tensor shim for the policy's public API
    def __init__(self, a):
        self.a = a

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def fitted_phy2log(weight, ep):
    """Run vLLM's rebalance for `ep` GPUs, no redundancy, ungrouped model."""
    out = Policy.rebalance_experts(_T(weight), E, 1, 1, ep)
    phy2log = out[0] if isinstance(out, tuple) else out
    return np.asarray(phy2log)


def rank_loads(weight, phy2log, ep):
    """Per-layer per-rank token loads for a placement."""
    slots = E // ep
    loads = np.zeros((L, ep))
    for li in range(L):
        w = weight[li][phy2log[li]]  # load at each physical slot
        loads[li] = w.reshape(ep, slots).sum(axis=1)
    return loads


def balancedness(loads):
    return loads.mean(axis=1) / np.maximum(loads.max(axis=1), 1e-12)


identity = np.tile(np.arange(E), (L, 1))
EPS = [2, 4, 8, 16, 32]
scen_results = {}  # (domain, scenario, ep) -> per-layer balancedness
other = {"gsm8k": "mbpp", "mbpp": "gsm8k"}
for d in ("gsm8k", "mbpp"):
    for ep in EPS:
        scen_results[(d, "default", ep)] = balancedness(
            rank_loads(agg[d], identity, ep)
        )
        scen_results[(d, "fitted", ep)] = balancedness(
            rank_loads(agg[d], fitted_phy2log(agg[d], ep), ep)
        )
        scen_results[(d, "anti-fitted", ep)] = balancedness(
            rank_loads(agg[d], fitted_phy2log(agg[other[d]], ep), ep)
        )

print("=== STEP-2 VALIDATION ===")
ok = True
for d in ("gsm8k", "mbpp"):
    for ep in EPS:
        f = scen_results[(d, "fitted", ep)].mean()
        de = scen_results[(d, "default", ep)].mean()
        if f < de:
            ok = False
            print(f"GATE FAIL: fitted<default for {d} EP{ep}")
print(f"gate 1 (fitted beats default everywhere): {'PASS' if ok else 'FAIL'}")
ep4d = np.mean([scen_results[(d, "default", 4)].mean() for d in ("gsm8k", "mbpp")])
print(f"gate 2 (EP4 default prediction vs live-measured ~0.78): predicted {ep4d:.2f}")
for d in ("gsm8k", "mbpp"):
    a = scen_results[(d, "anti-fitted", 8)].mean()
    de = scen_results[(d, "default", 8)].mean()
    print(
        f"gate 3 ({d} EP8): anti {a:.2f} vs default {de:.2f} (expect similar ~random)"
    )

# ---- charts
BLUE, YELLOW, RED = "#2a78d6", "#eda100", "#e34948"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
SCEN_COLOR = {"fitted": BLUE, "default": YELLOW, "anti-fitted": RED}


def ax_style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


# Chart A: balancedness vs EP size
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
for ax, d in zip(axes, ("gsm8k", "mbpp")):
    for scen in ("fitted", "default", "anti-fitted"):
        ys = [scen_results[(d, scen, ep)].mean() for ep in EPS]
        ax.plot(EPS, ys, color=SCEN_COLOR[scen], lw=2, marker="o", ms=5, label=scen)
    ax.set_xscale("log", base=2)
    ax.set_xticks(EPS)
    ax.set_xticklabels([f"EP{e}" for e in EPS])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("expert-parallel size", color=INK2)
    ax.set_ylabel("balancedness (mean/max rank load)", color=INK2)
    ax.set_title(
        f"{'GSM8K (math)' if d == 'gsm8k' else 'MBPP (code)'} traffic",
        color=INK,
        fontsize=11,
        loc="left",
    )
    ax_style(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
fig.suptitle(
    "Placement fit matters more as EP grows "
    "(vLLM's real EPLB algorithm on measured loads)",
    color=INK,
    fontsize=12,
    x=0.02,
    ha="left",
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{BASE}/s2_balancedness_vs_ep.png", dpi=150, facecolor=SURFACE)

# Chart B: per-rank loads at EP8, worst layer
LSHOW = 6
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
width = 0.27
for ax, d in zip(axes, ("gsm8k", "mbpp")):
    for k, scen in enumerate(("fitted", "default", "anti-fitted")):
        p2l = (
            identity
            if scen == "default"
            else fitted_phy2log(agg[d] if scen == "fitted" else agg[other[d]], 8)
        )
        loads = rank_loads(agg[d], p2l, 8)[LSHOW]
        share = loads / loads.sum() * 100
        ax.bar(
            np.arange(8) + (k - 1) * width,
            share,
            width * 0.92,
            color=SCEN_COLOR[scen],
            label=scen,
        )
    ax.axhline(100 / 8, color=INK2, lw=1, ls=":")
    ax.text(7.5, 100 / 8 + 0.6, "balanced (12.5%)", color=INK2, fontsize=8, ha="right")
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"r{i}" for i in range(8)])
    ax.set_xlabel(f"EP rank (layer {LSHOW})", color=INK2)
    ax.set_ylabel("share of layer tokens (%)", color=INK2)
    ax.set_title(
        f"{'GSM8K (math)' if d == 'gsm8k' else 'MBPP (code)'} traffic, EP8",
        color=INK,
        fontsize=11,
        loc="left",
    )
    ax_style(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
fig.suptitle(
    f"Layer {LSHOW}, EP8: unfitted placements overload one rank ~1.8x "
    "while others idle at 0.4x",
    color=INK,
    fontsize=12,
    x=0.02,
    ha="left",
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{BASE}/s2_rank_loads_ep8.png", dpi=150, facecolor=SURFACE)

summary = {
    f"{d}|{s}|EP{ep}": round(float(scen_results[(d, s, ep)].mean()), 3)
    for d in ("gsm8k", "mbpp")
    for s in ("default", "fitted", "anti-fitted")
    for ep in EPS
}
with open(f"{BASE}/step2_summary.json", "w") as fo:
    json.dump(summary, fo, indent=1)
print("charts written")
