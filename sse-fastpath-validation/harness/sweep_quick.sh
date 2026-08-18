#!/bin/bash
# One rep of each phase for one model. Used for the small-model replication
# sweep, where breadth across models matters more than reps within a model.
cd "$(dirname "$0")" || exit 1
mkdir -p "$BENCH_OUT"
pkill -f '[v]llm serve'; sleep 5

NA=${BENCH_NPROMPTS_A:-3000}
NB=${BENCH_NPROMPTS_B:-2000}

for mode in 0 2; do
  echo "=== phaseA mode$mode ($BENCH_MODEL) ==="
  python3 run_arm.py "$mode" "a1_m${mode}" 1 "$NA" 512 2>&1 | tail -28
  sleep 5
done

for mode in 0 2; do
  echo "=== phaseB mode$mode ($BENCH_MODEL) ==="
  python3 run_arm.py "$mode" "b1_m${mode}" 0 "$NB" 256 2>&1 | tail -28
  sleep 5
done
echo "SWEEP DONE"
