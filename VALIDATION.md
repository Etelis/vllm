# Runtime P/D switching: model comparison

We tested [combined source `1bd6fe4b1014a571d70159b4df71642f7a8e82a2`](https://github.com/Etelis/vllm/tree/1bd6fe4b1014a571d70159b4df71642f7a8e82a2), containing [feature revision `1dc2802e7d84c13ca09977ae6c8b945534bc6f17`](https://github.com/Etelis/vllm/tree/1dc2802e7d84c13ca09977ae6c8b945534bc6f17) and a separate unmerged Triton MoE correctness fix. These historical results describe that build.

Six NVIDIA H100 80GB HBM3 GPUs on one node ran three EP2 groups with TP1/DP2, BF16, DeepEPv2, NIXL pull transfer, CUDA graphs, and `VLLM_BATCH_INVARIANT=1`. A stayed prefill, C stayed decode, and B switched roles. Models ran separately. Limits were 4,096 context tokens, 4,096 batch tokens, 64 sequences, and 75% GPU memory utilization.

Eight concurrent clients used eight fixed prompts (287–310 tokens across tokenizers), greedy sampling, fresh cache salts, and 64 output tokens with EOS ignored. Each model passed a 20-second static baseline before ten switches in each direction.

| Model | Load requests | P→D median (range), ms | D→P median (range), ms |
| --- | ---: | ---: | ---: |
| OLMoE-1B-7B-0924-Instruct | 1,679 | 13.0 (12.2–37.1) | 234.2 (189.0–271.9) |
| DeepSeek-V2-Lite-Chat | 751 | 13.1 (12.1–25.0) | 590.6 (523.7–641.1) |
| Qwen3-30B-A3B | 630 | 35.8 (12.7–48.8) | 757.4 (614.0–793.0) |

Timing runs from receipt of HTTP 202 to observing the requested role as ready. It includes draining and readiness polling with 10 ms sleeps; it excludes preceding router reservation draining and the POST round trip. These workload measurements are not latency guarantees.

All 60 transitions and 3,060 load requests passed without request errors, test-worker errors, or differences from serial P/D references. Real NIXL transfers were verified. All workers retained process, model/KV allocation, communicator, NIXL registration, and CUDA graph identities; capture counts remained unchanged. C emitted output during every measured transition interval.

All models passed a deliberately held-KV test: a 250 ms drain timeout restored the original role and epoch, then decode consumed the same KV successfully. Stale epochs returned HTTP 409. Final checks found no failed transfers, failed notifications, or expired leases; all cores drained. The historical control/CLI suite passed 66 tests.

Model pins:

- [allenai/OLMoE-1B-7B-0924-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct/tree/7f1c97f440f06ce36705e4f2b843edb5925f4498)
- [deepseek-ai/DeepSeek-V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/tree/85864749cd611b4353ce1decdb286193298f64c7)
- [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/tree/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39)

[Measurements](measurements.json) contain exact pins, settings, counts, and timing series; [CSV](figures/switch-duration-observations.csv) contains all chart observations.

Recorded driver commands below use normalized local paths. The [original harness and runtime instructions](validation/README.md) provide the source, launch settings, and exact environment.

```sh
model_alias=Qwen3-30B-A3B
.venv/bin/python validation/exercise.py \
  --model "$model_alias" --cycles 0 --concurrency 8 --tokens 64 --settle 20 \
  --output results/final-static-results.json
.venv/bin/python validation/exercise.py \
  --model "$model_alias" --cycles 10 --concurrency 8 --tokens 64 --settle 3 \
  --references results/final-static-results.json --output results/results.json
.venv/bin/python validation/edge_cases.py \
  --model "$model_alias" --references results/final-static-results.json \
  --output results/edge-results.json
.unit-venv/bin/python -m pytest tests/v1/engine/test_pd_role.py \
  tests/entrypoints/serve/middleware/test_pd_role.py \
  tests/entrypoints/launchers/test_cli_args.py -q --tb=short
```

Earlier diagnostic runs are excluded, including OLMoE's static baseline without batch invariance and Qwen's rollback-status race before correction. This experiment does not establish model quality, production performance, EP16, multi-node operation, hardware-failure recovery, or zero latency impact. Continued availability requires another ready group for each role.


## Final draft checks

[Feature revision `69e044805b323020162c9e538aae904bb733d39d`](https://github.com/Etelis/vllm/tree/69e044805b323020162c9e538aae904bb733d39d) passed all **66 control, frontend, and CLI tests** in 36.08 seconds on Linux/Python 3.12.3, with 14 existing Torch deprecation warnings. All 3,002 tracked vLLM source files and the three test files matched that revision. These were CPU tests with GPUs hidden and `VLLM_TARGET_DEVICE=cpu`.

The review shortened comments and documentation, consolidated repeated test setup, and moved readiness publication into the outer transition coordinator. On Python 3.11.16, the original controller published `ready` before finalizing its duration; the final revision keeps admission closed until finalization. The deterministic regression failed on `1dc2802` and passed on `69e0448` on Linux. It extends the existing HTTP/KV-drain test.

The GPU comparison above was not rerun after this cleanup. It remains pinned to `1bd6fe4`; its Python 3.12 execution did not exhibit the scheduling race. Pre-commit passed, and the final feature branch merged cleanly with upstream `3fb676bfad0f1c7099af6296983e739be3fe29cc` during review.
