# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate A/B arms into a comparison table."""

import glob
import hashlib
import json
import os
import re
import statistics as st

OUT = "/results/sse/out"
MODE_NAME = {0: "stock (pydantic)", 1: "PR #3 template", 2: "derived template"}

rows = []
for path in sorted(glob.glob(f"{OUT}/*.json")):
    if path.endswith(".bench.json"):
        continue
    with open(path) as f:
        r = json.load(f)
    if r["label"] == "smoke":
        continue
    rows.append(r)


def show(phase, pin):
    sel = [r for r in rows if r["pin"] == pin]
    if not sel:
        return
    print(f"\n=== {phase} (frontend pinned={bool(pin)}) ===")
    print(
        f"{'arm':10s} {'mode':18s} {'out tok/s':>10s} {'FE us/tok':>10s} "
        f"{'FE cores':>9s} {'mean TPOT':>10s} {'p99 TPOT':>9s}"
    )
    for r in sel:
        print(
            f"{r['label']:10s} {MODE_NAME[r['mode']]:18s} "
            f"{r['output_throughput'] or 0:10.0f} "
            f"{r.get('frontend_us_per_token') or 0:10.2f} "
            f"{r.get('frontend_cores') or 0:9.3f} "
            f"{r['mean_tpot_ms'] or 0:9.2f}m {r['p99_tpot_ms'] or 0:8.2f}m"
        )
    print(f"{'-' * 72}")
    base = None
    for mode in (0, 1, 2):
        g = [r for r in sel if r["mode"] == mode]
        if not g:
            continue
        thr = st.median([r["output_throughput"] for r in g if r["output_throughput"]])
        cpu = st.median(
            [r["frontend_us_per_token"] for r in g if r.get("frontend_us_per_token")]
        )
        if mode == 0:
            base = (thr, cpu)
        d_thr = f"{(thr / base[0] - 1) * 100:+.1f}%" if base else ""
        d_cpu = f"{(cpu / base[1] - 1) * 100:+.1f}%" if base else ""
        print(
            f"MEDIAN n={len(g)}  {MODE_NAME[mode]:18s} "
            f"{thr:10.0f} {d_thr:>8s}   FE CPU {cpu:6.2f} us/tok {d_cpu:>8s}"
        )


show("Phase A: throughput headline", 1)
show("Phase B: realistic deployment", 0)

# Wire-byte identity across arms (ids/timestamps normalised away).
print("\n=== wire-byte identity (raw SSE off the socket) ===")
digests = {}
for path in sorted(glob.glob(f"{OUT}/*.sse")):
    raw = open(path, "rb").read()
    norm = re.sub(rb'"id":"cmpl-[^"]*"', b'"id":"X"', raw)
    norm = re.sub(rb'"created":\d+', b'"created":0', norm)
    digests.setdefault(hashlib.sha256(norm).hexdigest()[:16], []).append(
        os.path.basename(path).replace(".sse", "")
    )
for d, labels in digests.items():
    modes = sorted({int(x.split("_m")[1]) for x in labels if "_m" in x})
    print(f"  {d}  modes={modes}  arms={labels}")
print(
    "  -> IDENTICAL across all modes"
    if len(digests) == 1
    else f"  -> {len(digests)} DISTINCT byte streams - investigate"
)
