#!/bin/bash
# A/B sweep for one model. Modes 0 (stock) and 2 (derived template) only --
# the hardcoded variant was shown equivalent to the derived one on run 1.
#
# env: BENCH_MODEL, BENCH_OUT, BENCH_GPU_UTIL, BENCH_SERVE_EXTRA,
#      BENCH_NPROMPTS_A, BENCH_NPROMPTS_B
cd "$(dirname "$0")" || exit 1
mkdir -p "$BENCH_OUT"
pkill -f '[v]llm serve'; sleep 5

NA=${BENCH_NPROMPTS_A:-1500}
NB=${BENCH_NPROMPTS_B:-1500}

# Phase A: frontend pinned to one core -> API server is the bottleneck.
for rep in 1 2; do
  for mode in 0 2; do
    echo "=== phaseA rep$rep mode$mode ($BENCH_MODEL) ==="
    python3 run_arm.py "$mode" "a${rep}_m${mode}" 1 "$NA" 512 2>&1 | tail -30
    sleep 5
  done
done

# Phase B: unpinned, ordinary flags.
n=0
for mode in 0 2 0 2; do
  n=$((n+1))
  echo "=== phaseB run$n mode$mode ($BENCH_MODEL) ==="
  python3 run_arm.py "$mode" "b${n}_m${mode}" 0 "$NB" 256 2>&1 | tail -30
  sleep 5
done
echo "SWEEP DONE"
