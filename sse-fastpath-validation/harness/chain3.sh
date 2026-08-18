#!/bin/bash
# Small-model replication sweep. Four models spanning three tokenizer
# families (vocab 32k / 49k / 152k), all small enough that the engine can
# plausibly outrun the ~32k tok/s single-frontend ceiling.
cd "$(dirname "$0")" || exit 1

run() {
  echo "########## $1 ##########"
  BENCH_MODEL="$1" BENCH_OUT="$2" BENCH_GPU_UTIL=0.85 \
  BENCH_NPROMPTS_A=3000 BENCH_NPROMPTS_B=2000 \
    ./sweep_quick.sh
}

run Qwen/Qwen3-1.7B                      /results/sse2/out_qwen17
run TinyLlama/TinyLlama-1.1B-Chat-v1.0   /results/sse2/out_tinyllama
run HuggingFaceTB/SmolLM2-1.7B-Instruct  /results/sse2/out_smol17
run HuggingFaceTB/SmolLM2-360M-Instruct  /results/sse2/out_smol360

echo "CHAIN3 DONE"
