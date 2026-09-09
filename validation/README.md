# Validation harness

These are the original sources used for the three-model comparison on
`1bd6fe4b1014a571d70159b4df71642f7a8e82a2`. Python and shell files are unchanged;
no paths, transport settings, prompts, timing, or checks were edited for
publication. The archive's macOS metadata files were omitted.
[Source hashes](SOURCE_SHA256SUMS.json) identify the published copies.

The harness is specific to one Linux machine with six available H100 80GB GPUs,
three EP2 groups, and the recorded vLLM revision. It runs real P/D handoffs and
checks outputs, resource identities, timeout recovery, and final draining. The
probe is test instrumentation, not a production router or management interface.

## Runtime

Use this immutable image with a CUDA 13 compatible NVIDIA driver and GPU/RDMA
access:

```text
docker.io/vllm/vllm-openai@sha256:93f8302560d0bf0b88642c8c92515a6f8c5dc875b91c5853408c6c3a6d9549d3
```

The measured runtime used Python 3.12.3, Torch 2.13.0+cu130, Triton 3.7.1,
DeepEP 2.0.0+local, NIXL 1.3.2, NCCL 2.30.7, NVSHMEM 3.4.5,
FlashInfer 0.6.18, Transformers 5.16.1, and httpx 0.28.1. The setup creates a
venv with system site packages and installs editable vLLM with `--no-deps
--no-build-isolation`, preserving the image's runtime dependencies. Its pinned
build helpers are setuptools-rust 1.11.1, semantic-version 2.10.0, and wheel 0.45.1.

Native and generated vLLM artifacts come from this base-revision wheel:

[vllm c7 CUDA 13 wheel](https://wheels.vllm.ai/c7e9816c6ab0731165a134fd0a9defed6ab1d748/vllm-0.28.1rc1.dev588%2Bgc7e9816c6-cp38-abi3-manylinux_2_28_x86_64.whl)

Its SHA256 is
`bedc5e5b491af548b7eb248c031d0d910eabf2dc93f2158ea8466edf9f21a5cd`
(309,034,969 bytes). The setup checks this hash. Copying native libraries alone
is insufficient: the installation also supplies generated Python wrappers and
resources absent from Git.

## Filesystem and source setup

Provide a fresh writable `/workspace`. Place this entire `validation/` directory
at `/workspace/validation`. The original scripts expect:

| Path | Purpose |
| --- | --- |
| `/workspace/.venv` | GPU runtime environment |
| `/workspace/baseline` | Base source with wheel artifacts installed |
| `/workspace/patched` | Tested Python source plus generated artifacts |
| `/workspace/source` | Source archives for setup |
| `/workspace/runs` | New output directory for each model run |

The following reconstructs the recorded source layout. `setup-runtime.sh`
first installs the original combined revision `b946138…`; the final archive
then overlays tracked files from `1bd6fe4…`, including the rollback-status fix.
Generated files remain in place. These setup instructions are provided for
reproduction; publication did not rerun the models.

```sh
git clone https://github.com/Etelis/vllm.git /workspace/source-repo
git -C /workspace/source-repo checkout 1bd6fe4b1014a571d70159b4df71642f7a8e82a2
mkdir -p /workspace/source /workspace/runs
git -C /workspace/source-repo archive --format=tar.gz \
  --output=/workspace/source/baseline-c7.tar.gz \
  c7e9816c6ab0731165a134fd0a9defed6ab1d748
git -C /workspace/source-repo archive --format=tar.gz \
  --output=/workspace/source/combined-b946.tar.gz \
  b946138879d46ae6848e6f001f8e294b6cbd0c8c
git -C /workspace/source-repo ls-tree -r \
  b946138879d46ae6848e6f001f8e294b6cbd0c8c -- vllm \
  > /workspace/source/vllm-b946-tree.txt
bash /workspace/validation/setup-runtime.sh /workspace/source
git -C /workspace/source-repo archive \
  1bd6fe4b1014a571d70159b4df71642f7a8e82a2 \
  | tar -x -C /workspace/patched
```

The combined revision includes the separate Triton MoE prerequisite described
in the [validation summary](../VALIDATION.md). Checkpoints must match the model
pins there. Model paths and served aliases are explicit arguments; no model
volume or machine-specific checkpoint location is embedded in the harness.

## Launch requirements

`launch-group.sh` sets `PYTHONPATH=/workspace/validation:/workspace/patched`,
`VLLM_USE_V2_MODEL_RUNNER=1`, and `VLLM_BATCH_INVARIANT=1`. It enables the worker
extension `worker_probe.PDRoleSwitchProbe` and middleware
`worker_probe.ProbeMiddleware`; the client requires `/pd_probe`, `/v1/pd_role`,
`/metrics`, and streaming `/v1/completions` on all three frontends.

The launch uses BF16 model defaults, TP1/DP2 with two local DP workers, expert
parallelism, `deepep_v2`, one Python API process, and NIXL `kv_both`. Context and
batch limits are 4,096, maximum sequences 64, and GPU memory utilization 0.75.
The scripts configure writable cache directories beneath `/workspace`, including
FlashInfer and DeepEP JIT caches. Startup captures the graphs used by the checks.

| Group | GPU indices | Initial role | API / NIXL / DP RPC / master ports |
| --- | --- | --- | --- |
| A | 0, 1 | Prefill | 8100 / 8200 / 8300 / 8310 |
| B | 2, 3 | Prefill | 8110 / 8210 / 8330 / 8340 |
| C | 4, 5 | Decode | 8120 / 8220 / 8360 / 8370 |

Clients use loopback; API listeners bind all interfaces. Use an isolated test
environment for the probe endpoints. Raw outputs include process, host, and
allocation metadata; the public [measurement summary](../measurements.json)
contains only the relevant checks and counts.

The measured transport filters were `NCCL_IB_HCA='=mlx5_0:1'` and
`UCX_NET_DEVICES=mlx5_0:1`. The unchanged launcher defaults to those values.
They must identify a reachable RDMA interface in the reproduction environment;
there is no automatic interface discovery. Set both variables explicitly before
launching if the verified interface differs. NIXL side-channel hosts remain
loopback because this harness uses one machine.

## Run

Set the checkpoint path yourself, use the matching alias, and choose a fresh
output directory. The following example uses Qwen; repeat with the other pinned
models after stopping the preceding fleet.

```sh
model_path=/path/to/pinned/Qwen3-30B-A3B
model_alias=Qwen3-30B-A3B
run_dir=/workspace/runs/qwen3
bash /workspace/validation/run-model.sh "$model_path" "$model_alias" "$run_dir"
/workspace/.venv/bin/python /workspace/validation/stop-fleet.py "$run_dir"
```

`run-model.sh` starts all groups and verifies the initial roles and epochs.
`commands.sh` then creates serial references, runs the 20-second static baseline,
performs ten round trips under eight-client traffic, summarizes results, runs the
held-KV and stale-epoch checks, and verifies final quiescence. Output length is
64 tokens; the edge-case test requires references generated at that length.

The stop helper first checks quiescence, terminates the three recorded server
process groups, and waits for GPU processes to exit. Use a dedicated GPU test
environment; its final GPU check expects no remaining compute processes.

The 66 historical control/CLI tests used an isolated venv with pytest 9.1.1,
pytest-asyncio 1.4.0, and tblib 3.2.2, with CUDA hidden:

```sh
uv venv --python /usr/bin/python3.12 --system-site-packages /workspace/.unit-venv
uv pip install --python /workspace/.unit-venv/bin/python --no-deps \
  pytest==9.1.1 pytest-asyncio==1.4.0 tblib==3.2.2
cd /workspace/patched
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/workspace/patched \
  /workspace/.unit-venv/bin/python -m pytest tests/v1/engine/test_pd_role.py \
  tests/entrypoints/serve/middleware/test_pd_role.py \
  tests/entrypoints/launchers/test_cli_args.py -q --tb=short
```

Identity checks compare host metadata and counters. They do not checksum tensor
contents, trace every graph replay, or establish arbitrary-topology determinism.
Output equality and real KV handoffs are checked separately. Timing excludes
router reservation draining and the POST round trip; the complete definition and
limitations are in the [validation summary](../VALIDATION.md).
