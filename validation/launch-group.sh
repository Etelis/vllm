#!/bin/bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 {a|b|c} MODEL_PATH MODEL_ALIAS OUTPUT_DIR" >&2
  exit 2
fi
group=$1
model_path=$2
model_alias=$3
output_dir=$4
case "$group" in
  a) devices=0,1; api_port=8100; nixl_port=8200; rpc_port=8300; master_port=8310; role=prefill ;;
  b) devices=2,3; api_port=8110; nixl_port=8210; rpc_port=8330; master_port=8340; role=prefill ;;
  c) devices=4,5; api_port=8120; nixl_port=8220; rpc_port=8360; master_port=8370; role=decode ;;
  *) echo "Unknown group: $group" >&2; exit 2 ;;
esac
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
cd /workspace/patched
export PATH=/workspace/.venv/bin:$PATH
export PYTHONPATH=/workspace/validation:/workspace/patched
export CUDA_VISIBLE_DEVICES=$devices
export PD_TEST_SOURCE_REVISION=1bd6fe4b1014a571d70159b4df71642f7a8e82a2
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_BATCH_INVARIANT=1
export NCCL_IB_HCA="${NCCL_IB_HCA:-=${PD_HCA:-mlx5_0:1}}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-${PD_HCA:-mlx5_0:1}}"
export VLLM_NIXL_SIDE_CHANNEL_PORT=$nixl_port
export VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1
export VLLM_CACHE_ROOT=/workspace/cache-role
export HF_HOME=/workspace/hf-cache
export TRITON_CACHE_DIR=/workspace/triton-cache-role
export FLASHINFER_WORKSPACE_BASE=/workspace/flashinfer
export XDG_CACHE_HOME=/workspace/cache
export CUDA_CACHE_PATH=/workspace/cuda-cache
export TORCHINDUCTOR_CACHE_DIR=/workspace/torchinductor-cache-role
export EP_JIT_CACHE_DIR=/workspace/deep-ep-cache
export VLLM_NO_USAGE_STATS=1
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

kv_config="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"engine_id\":\"pd-$group\",\"pd_role\":\"$role\"}"
command=(
  /workspace/.venv/bin/python -m vllm.entrypoints.cli.main serve "$model_path"
  --served-model-name "$model_alias"
  --host 0.0.0.0 --port "$api_port"
  --tensor-parallel-size 1 --data-parallel-size 2 --data-parallel-size-local 2
  --data-parallel-address 127.0.0.1 --data-parallel-rpc-port "$rpc_port" --master-port "$master_port"
  --enable-expert-parallel --all2all-backend deepep_v2 --api-server-count 1
  --kv-transfer-config "$kv_config"
  --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 64
  --gpu-memory-utilization 0.75
  --worker-extension-cls worker_probe.PDRoleSwitchProbe
  --middleware worker_probe.ProbeMiddleware
)
/workspace/.venv/bin/python - "$output_dir/launch-$group.json" "$group" "${command[@]}" <<'PY'
import datetime
import json
import os
import pathlib
import sys

keys = [
    "CUDA_VISIBLE_DEVICES", "PD_TEST_SOURCE_REVISION", "VLLM_USE_V2_MODEL_RUNNER",
    "VLLM_BATCH_INVARIANT",
    "NCCL_IB_HCA", "UCX_NET_DEVICES", "VLLM_NIXL_SIDE_CHANNEL_PORT",
    "VLLM_NIXL_SIDE_CHANNEL_HOST", "PYTHONPATH", "VLLM_CACHE_ROOT", "HF_HOME",
    "TRITON_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE", "XDG_CACHE_HOME",
    "CUDA_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR", "EP_JIT_CACHE_DIR",
    "VLLM_NO_USAGE_STATS", "NCCL_DEBUG", "PYTHONUNBUFFERED",
]
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "group": sys.argv[2], "cwd": os.getcwd(), "argv": sys.argv[3:],
    "environment": {key: os.environ[key] for key in keys},
}, indent=2) + "\n")
PY
exec "${command[@]}" > "$output_dir/server-$group.log" 2>&1
