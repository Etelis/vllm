#!/bin/bash
# Run after root has launched all three groups and verified readiness.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 MODEL_ALIAS OUTPUT_DIR" >&2
  exit 2
fi
model_alias=$1
output_dir=$2
mkdir -p "$output_dir"
python_bin=/workspace/.venv/bin/python
"$python_bin" /workspace/validation/exercise.py \
  --model "$model_alias" --cycles 0 --concurrency 8 --tokens 64 --settle 20 \
  --output "$output_dir/final-static-results.json" > "$output_dir/static.log" 2>&1
PYTHONPATH=/workspace/validation "$python_bin" - "$output_dir/final-static-results.json" \
  > "$output_dir/static-validation.log" 2>&1 <<'PY'
import json
import sys
from pathlib import Path
from validate_probe import compare

data = json.loads(Path(sys.argv[1]).read_text())
summary = data["summary"]
assert not summary["fatal_error"] and summary["worker_errors"] == [], summary
assert summary["requests"] > 0 and summary["errors"] == 0, summary
assert summary["output_mismatches"] == 0 and summary["transitions"] == 0, summary
records = data["records"][len(data["references"]):]
assert len(records) == summary["requests"]
assert all("error" not in row for row in records)
assert all(row["text"] == data["references"][str(row["prompt"])] for row in records)
assert len(data["startup_probes"]) == len(data["initial_probes"]) == len(data["final_probes"]) == 3
for group in range(3):
    compare(data["startup_probes"][group], data["initial_probes"][group], 2)
    compare(data["initial_probes"][group], data["final_probes"][group], 2)
print(json.dumps({"static_summary": summary, "startup_to_final_identities_unchanged": True}, indent=2))
PY
"$python_bin" /workspace/validation/exercise.py \
  --model "$model_alias" --cycles 10 --concurrency 8 --tokens 64 --settle 3 \
  --references "$output_dir/final-static-results.json" \
  --output "$output_dir/results.json" > "$output_dir/switching.log" 2>&1
"$python_bin" /workspace/validation/summarize_results.py \
  "$output_dir/results.json" --output "$output_dir/switching-summary.json" \
  > "$output_dir/switching-summary.log" 2>&1
"$python_bin" /workspace/validation/edge_cases.py \
  --model "$model_alias" --references "$output_dir/final-static-results.json" \
  --output "$output_dir/edge-results.json" > "$output_dir/edge.log" 2>&1
"$python_bin" /workspace/validation/final_verify.py "$output_dir" \
  > "$output_dir/final-verification.log" 2>&1
