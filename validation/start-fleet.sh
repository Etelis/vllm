#!/bin/bash
set -euo pipefail
model_path=${1:?model path}
model_alias=${2:?served model name}
run_dir=${3:?run directory}
mkdir -p "$run_dir"
for group in a b c; do
  if [[ -e "$run_dir/server-$group.pid" ]]; then
    echo "Refusing to overwrite previous server PID" >&2
    exit 1
  fi
  nohup setsid bash /workspace/validation/launch-group.sh \
    "$group" "$model_path" "$model_alias" "$run_dir" \
    > "$run_dir/server-$group.log" 2>&1 < /dev/null &
  echo "$!" > "$run_dir/server-$group.pid"
done
date -u +%FT%TZ > "$run_dir/launch-time.txt"
