#!/bin/bash
# Run the A/B sweep across the second and third models, back to back.
cd "$(dirname "$0")" || exit 1

echo "########## Qwen3-30B-A3B-FP8 ##########"
BENCH_MODEL=Qwen/Qwen3-30B-A3B-FP8 \
BENCH_OUT=/results/sse2/out_qwen30 \
BENCH_GPU_UTIL=0.85 \
BENCH_NPROMPTS_A=1500 BENCH_NPROMPTS_B=1500 \
  ./sweep_model.sh

echo "########## gpt-oss-120b ##########"
BENCH_MODEL=openai/gpt-oss-120b \
BENCH_OUT=/results/sse2/out_gptoss \
BENCH_GPU_UTIL=0.92 \
BENCH_NPROMPTS_A=800 BENCH_NPROMPTS_B=800 \
  ./sweep_model.sh

echo "CHAIN DONE"
