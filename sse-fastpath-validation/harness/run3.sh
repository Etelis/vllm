#!/bin/bash
# Run 3, answering the two open objections from the adversarial audit:
#   1. frames != tokens (vLLM coalesces undelivered deltas) -- measure the
#      frames-per-token ratio in each arm so the CPU-per-token comparison can
#      be put on a per-frame basis.
#   2. --stream-interval already attacks this cost with a bigger lever --
#      measure how much of the win survives at stream_interval 2 and 4.
cd "$(dirname "$0")" || exit 1
export BENCH_MODEL=Qwen/Qwen3-0.6B
export BENCH_OUT=/results/sse2/out_run3
mkdir -p "$BENCH_OUT"
pkill -f '[v]llm serve'; sleep 5

for si in 1 2 4; do
  for mode in 0 2; do
    echo "=== stream_interval=$si mode=$mode ==="
    BENCH_SERVE_EXTRA="--stream-interval $si" \
      python3 run_arm.py "$mode" "si${si}_m${mode}" 0 3000 256 2>&1 | tail -30
    sleep 5
  done
done
echo "RUN3 DONE"
