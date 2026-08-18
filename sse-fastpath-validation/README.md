# SSE delta-frame template: cluster validation

Hardware-backed validation of rendering plain `/v1/completions` delta chunks
from a per-stream template instead of constructing and dumping a
`CompletionStreamResponse` per token.

Run on **CoreWeave waldorf**, 1×H200 / 32 cores, `vllm/vllm-openai:nightly`
(vLLM 0.27.2rc1.dev18+g3d204dfda, Python 3.12.3, pydantic 2.13.4),
`Qwen/Qwen3-0.6B`, `--api-server-count 1` (the default for a non-DP
deployment), `--max-num-seqs 512`, `stream_interval=1`.

## Results

Per-chunk render cost, measured on the image's own protocol models:

| path | µs/chunk | vs stock |
|---|---|---|
| construct + `model_dump_json(exclude_unset=True)` | 7.50 | — |
| hardcoded template | 0.85 | −89% |
| derived template | 0.82 | −89% |

End-to-end, `vllm bench serve` (random, 32 in / 512 out, `--ignore-eos`),
medians of 2 runs per arm, arms alternated:

| phase | arm | output tok/s | frontend CPU µs/token |
|---|---|---|---|
| A — frontend pinned to 1 core | stock | 33,411 | 30.21 |
| A | hardcoded template | 45,061 (**+34.9%**) | 22.44 (**−25.7%**) |
| A | derived template | 44,517 (+33.2%) | 22.74 (−24.7%) |
| B — unpinned, 256 concurrency | stock | 32,972 | 30.54 |
| B | derived template | 43,279 (**+31.3%**) | 21.54 (−29.5%) |

The API server is a single asyncio loop, so it is confined to one core whether
or not it is pinned: `frontend_cores` is 1.006–1.013 in every stock arm. That
is why phase B — no pinning, ordinary flags — shows the same gain as phase A.

Measured throughput matches the frontend's CPU capacity to within 2%
(1e6/30.21 = 33.1k predicted vs 33.4k measured; 1e6/22.44 = 44.6k vs 45.1k),
confirming the API server, not the client or the engine, is what binds.

Phase B's fast-path arms sit at 0.925–0.938 cores — the frontend is no longer
saturated, so something else caps that arm and **+31.3% is a lower bound**.

## Correctness

Raw SSE bytes were captured off the socket for a fixed greedy request set in
every arm. After normalising the per-request `id` and `created`, all 11 arms
(modes stock / hardcoded / derived) hash to one digest, `1b901c44fdf95379`,
over 150 frames — see `results/sample_stock.sse` vs `results/sample_fastpath.sse`.

## Reproducing

```bash
kubectl apply -f harness/benchpod.yaml
kubectl -n etelis-moe cp harness/ etelis-moe/sse-bench:/results/sse/
kubectl -n etelis-moe exec sse-bench -- bash -lc '
  F=/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/serving.py
  python3 /results/sse/patch_serving.py $F
  cd /results/sse && ./sweep.sh'
kubectl -n etelis-moe exec sse-bench -- python3 /results/sse/aggregate.py
```

`patch_serving.py` installs one file whose behaviour is selected at import time
by `VLLM_SSE_FASTPATH` (0 stock / 1 hardcoded / 2 derived), so every arm runs
byte-identical code on disk and the arms differ only by environment.
