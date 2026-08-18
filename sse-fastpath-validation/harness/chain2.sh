#!/bin/bash
# Fourth model: a small dense model close to the measured ~32k tok/s frontend
# ceiling, to test whether the throughput win survives at a real deployment size.
cd "$(dirname "$0")" || exit 1

echo "########## Qwen3-4B ##########"
BENCH_MODEL=Qwen/Qwen3-4B \
BENCH_OUT=/results/sse2/out_qwen4b \
BENCH_GPU_UTIL=0.85 \
BENCH_NPROMPTS_A=3000 BENCH_NPROMPTS_B=3000 \
  ./sweep_model.sh

echo "CHAIN2 DONE"
