#!/usr/bin/env bash
set -euo pipefail

# Fresh /workspace in image sha256:93f8302560d0bf0b88642c8c92515a6f8c5dc875b91c5853408c6c3a6d9549d3.
# Supply exact `git archive` outputs and `git ls-tree -r b946... -- vllm`.
source_dir=${1:-/workspace/source}
binary_revision=c7e9816c6ab0731165a134fd0a9defed6ab1d748
source_revision=b946138879d46ae6848e6f001f8e294b6cbd0c8c
wheel_name=vllm-0.28.1rc1.dev588+gc7e9816c6-cp38-abi3-manylinux_2_28_x86_64.whl
wheel_url=https://wheels.vllm.ai/c7e9816c6ab0731165a134fd0a9defed6ab1d748/vllm-0.28.1rc1.dev588%2Bgc7e9816c6-cp38-abi3-manylinux_2_28_x86_64.whl

test -f "$source_dir/baseline-c7.tar.gz"
test -f "$source_dir/combined-b946.tar.gz"
test -f "$source_dir/vllm-b946-tree.txt"
for destination in /workspace/.venv /workspace/baseline /workspace/patched; do
    if test -e "$destination"; then
        printf 'Refusing to overwrite existing runtime: %s\n' "$destination" >&2
        exit 1
    fi
done
mkdir -p /workspace/baseline /workspace/patched /workspace/results /workspace/wheels
mkdir -p /workspace/cache /workspace/cache-role /workspace/hf-cache
mkdir -p /workspace/triton-cache-role /workspace/flashinfer /workspace/cuda-cache
mkdir -p /workspace/torchinductor-cache-role /workspace/deep-ep-cache
export XDG_CACHE_HOME=/workspace/cache
export UV_CACHE_DIR=/workspace/cache/uv
export FLASHINFER_WORKSPACE_BASE=/workspace/flashinfer
export EP_JIT_CACHE_DIR=/workspace/deep-ep-cache

tar -xzf "$source_dir/baseline-c7.tar.gz" -C /workspace/baseline
tar -xzf "$source_dir/combined-b946.tar.gz" -C /workspace/patched
uv venv --python /usr/bin/python3.12 --system-site-packages /workspace/.venv

# Build metadata helpers are absent from the image; native builds remain disabled.
uv pip install --python /workspace/.venv/bin/python --no-deps \
    setuptools-rust==1.11.1 semantic-version==2.10.0 wheel==0.45.1

# Preserve the pinned image dependencies; do not resolve or upgrade runtime packages.
/workspace/.venv/bin/python - <<'PY'
import importlib.metadata as metadata
import json

expected = {
    "torch": "2.13.0+cu130", "triton": "3.7.1", "deep-ep": "2.0.0+local",
    "nixl": "1.3.2", "nvidia-nccl-cu13": "2.30.7",
    "nvidia-nvshmem-cu13": "3.4.5", "flashinfer-python": "0.6.18",
}
actual = {name: metadata.version(name) for name in expected}
assert actual == expected, (expected, actual)
print(json.dumps(actual, indent=2))
PY

curl --fail --location --retry 3 --connect-timeout 20 --max-time 600 \
    "$wheel_url" -o "/workspace/wheels/$wheel_name"
printf '%s  %s\n' bedc5e5b491af548b7eb248c031d0d910eabf2dc93f2158ea8466edf9f21a5cd \
    "/workspace/wheels/$wheel_name" | sha256sum --check
sha256sum "/workspace/wheels/$wheel_name" > /workspace/results/precompiled-wheel.sha256
printf '%s\n' "$wheel_url" > /workspace/results/precompiled-wheel-url.txt

cd /workspace/baseline
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION="/workspace/wheels/$wheel_name" \
VLLM_PRECOMPILED_WHEEL_COMMIT="$binary_revision" \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
VLLM_VERSION_OVERRIDE=0.28.1rc1.dev588+gc7e9816c6.precompiled \
uv pip install --python /workspace/.venv/bin/python \
    --no-deps --no-build-isolation --editable /workspace/baseline \
    2>&1 | tee /workspace/results/baseline-install.log

# The wheel supplies native libraries and generated Python/resources absent from Git.
# Existing b946 tracked files always take precedence over the baseline installation.
/workspace/.venv/bin/python - "$source_dir/vllm-b946-tree.txt" "$source_revision" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys

baseline = Path('/workspace/baseline/vllm')
patched = Path('/workspace/patched/vllm')
copied = []
for source in sorted(baseline.rglob('*')):
    relative = source.relative_to(baseline)
    if '__pycache__' in relative.parts or source.is_dir():
        continue
    target = patched / relative
    if target.exists() or target.is_symlink():
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target, follow_symlinks=False)
    copied.append(str(relative))
Path('/workspace/results/generated-artifacts.json').write_text(json.dumps(copied, indent=2))

records = []
for line in Path(sys.argv[1]).read_text().splitlines():
    meta, relative = line.split('\t', 1)
    mode, kind, expected = meta.split()
    target = Path('/workspace/patched') / relative
    assert kind == 'blob'
    assert target.is_symlink() == (mode == '120000'), relative
    content = os.readlink(target).encode() if target.is_symlink() else target.read_bytes()
    actual = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
    assert expected == actual, relative
    if not target.is_symlink():
        assert bool(target.stat().st_mode & stat.S_IXUSR) == (mode == '100755'), relative
    records.append({'path': relative, 'git_blob_sha1': actual})
assert len(records) == 3002, len(records)
for relative in ['_C_stable_libtorch.abi3.so', '_moe_C_stable_libtorch.abi3.so',
                 'third_party/flashmla/flash_mla_interface.py', 'vllm-rs']:
    assert (patched / relative).is_file(), relative
result = {'source_revision': sys.argv[2], 'tracked_files_verified': len(records),
          'generated_files_copied': len(copied), 'records': records}
Path('/workspace/results/source-b946-verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps({key: value for key, value in result.items() if key != 'records'}, indent=2))
PY

printf 'Runtime ready. Launch with PYTHONPATH=/workspace/validation:/workspace/patched. No servers launched.\n'
