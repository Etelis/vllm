#!/bin/bash
set -euo pipefail
model_path=${1:?model path}
model_alias=${2:?served model name}
run_dir=${3:?run directory}
bash /workspace/validation/start-fleet.sh "$model_path" "$model_alias" "$run_dir"
/workspace/.venv/bin/python /workspace/validation/wait-ready.py "$run_dir" \
  > "$run_dir/readiness.log" 2>&1
bash /workspace/validation/commands.sh "$model_alias" "$run_dir"
