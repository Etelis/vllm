#!/bin/bash
# A/B/A/B sweep. Arms alternate so any thermal/drift effect hits all modes.
cd /results/sse || exit 1
pkill -f '[v]llm serve'; sleep 5

# Phase A: frontend pinned to one core -> API server is the bottleneck.
for rep in 1 2; do
  for mode in 0 1 2; do
    echo "=== phaseA rep$rep mode$mode ==="
    python3 run_arm.py "$mode" "a${rep}_m${mode}" 1 4000 512 2>&1 | tail -30
    sleep 5
  done
done

# Phase B: unpinned, realistic config -> CPU-per-token in a normal deployment.
for mode in 0 2 0 2; do
  n=$((n+1))
  echo "=== phaseB run$n mode$mode ==="
  python3 run_arm.py "$mode" "b${n}_m${mode}" 0 2000 256 2>&1 | tail -30
  sleep 5
done
echo "SWEEP DONE"
