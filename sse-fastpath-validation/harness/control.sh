#!/bin/bash
# Control: is the "stock" arm (patched file, VLLM_SSE_FASTPATH=0) really
# equivalent to the pristine, unpatched file? Runs all three back to back in
# one pod so the comparison is paired.
cd "$(dirname "$0")" || exit 1
F=/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/serving.py
export BENCH_MODEL=Qwen/Qwen3-0.6B
export BENCH_OUT=/results/sse2/out_ctrl
mkdir -p "$BENCH_OUT"
pkill -f '[v]llm serve'; sleep 5

echo "=== ctrl: pristine unpatched file ==="
cp "$F.orig" "$F"
python3 run_arm.py 0 ctrl_pristine 1 4000 512 2>&1 | tail -25
sleep 5

echo "=== ctrl: patched file, fast path disabled (mode 0) ==="
python3 patch_serving.py "$F"
python3 run_arm.py 0 ctrl_mode0 1 4000 512 2>&1 | tail -25
sleep 5

echo "=== ctrl: patched file, derived template (mode 2) ==="
python3 run_arm.py 2 ctrl_mode2 1 4000 512 2>&1 | tail -25
echo "CONTROL DONE"
