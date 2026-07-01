# Cluster run (Qwen3-30B-A3B, DP8 + EP8)

End-to-end run of the harness against a live vLLM server on 8×H100. Captures
per-prompt / per-layer logical expert usage, aggregates per benchmark, joins
with prefix-cache hits, and runs the cache-affinity-vs-shuffled worker A/B that
demonstrates KV-routing-manufactured expert imbalance. Reproduces an earlier
DP4 run (per-domain numbers within noise).

## Setup

- Model: `Qwen/Qwen3-30B-A3B` (128 routed experts, top-8, 48 MoE layers)
- Serving: `--data-parallel-size 8 --enable-expert-parallel --api-server-count 1
  --enable-return-routed-experts --enable-prompt-tokens-details
  --enable-prefix-caching --max-model-len 8192`
- Image: `vllm/vllm-openai:v0.23.0-cu129-ubuntu2404`
- Hardware: 8× NVIDIA H100-80GB (single node)
- Domains: GSM8K 8-shot (math) and MBPP 3-shot (code), 300 prompts each

Each request returned `routed_experts` decoding to `(num_tokens-1, 48, 8)`
`uint8` logical expert ids in `[0, 127]`. All 600 requests succeeded; every
expert was exercised (max id 127).

(`--api-server-count 1` is required at DP8: the default spawns one API-server
process per DP rank, and 8 of them exhaust shared-memory/semaphore objects and
crash on startup. One front-end + 8 engine ranks is stable.)

## Manufactured imbalance — worker A/B (the headline)

`worker_ab.py` emulates 8 independent MoE replicas fed by an external router,
reconstructed from the per-prompt histograms. The *same* 600 prompts are routed
two ways, 75 prompts/worker either way — only the assignment differs:

- **shuffled** (domain-mixed workers): per-worker IR 6.77, cross-worker
  JS **0.002 bits**.
- **cache-affinity** (each worker gets one domain): per-worker IR 8.90,
  cross-worker JS **0.182 bits**.
- **Manufactured imbalance: per-worker IR +2.13, cross-worker JS +0.18 bits
  (0.002 → 0.182, ~90×).**

Under shuffled routing the 8 workers are statistically interchangeable
(cross-worker JS ≈ 0), so one cluster-wide EPLB placement fits all of them.
Cache-affinity routing makes each worker's batch domain-homogeneous, so workers
specialize onto different expert sets (JS 0.18) — routing-induced heterogeneity
that a single global EPLB placement cannot satisfy. The effect is purely the
router's doing: identical request set, only the assignment changed.

## Per-domain expert usage (aggregated over 300 prompts)

- **gsm8k**: mean layer imbalance ratio (max/mean) 9.40 (max 14.34), mean Gini
  0.630, mean entropy 5.92 / 7.0 bits, mean cached tokens 1264. Global top-8
  experts: 27, 99, 4, 18, 22, 44, 46, 87.
- **mbpp**: mean layer imbalance ratio 8.39 (max 14.14), mean Gini 0.608, mean
  entropy 6.02 / 7.0 bits, mean cached tokens 561. Global top-8 experts: 91, 27,
  122, 71, 59, 64, 0, 28.

Both domains show strong intra-layer expert-load imbalance (a few experts take
~9× the average load) — consistent with expert specialization.

## Cross-domain specialization (the mechanism)

- Expert-usage Jensen-Shannon divergence: mean 0.318 bits, max 0.516 bits at
  layer 6 (all 48 layers exceed 0.09).
- Top-8 hot-expert Jaccard overlap: **0.132** (only expert 27 is shared).
- Most divergent layers: 6, 7, 18, 30, 31.

Math and code share only ~13% of their hottest experts — the precondition that
makes cache-affinity routing produce the per-worker skew above.

## Prefix-cache correlation (exploratory)

Per-prompt Pearson r between cache-hit fraction and expert-usage concentration
(mean per-layer Gini): gsm8k +0.51, mbpp −0.56. Signs differ across domains, so
this per-prompt correlation is not a robust causal signal (confounded by prompt
length — the shared preamble is fixed-size, so shorter questions have both a
higher cache fraction and a noisier, more concentrated histogram). The
load-bearing evidence is the worker A/B and the cross-domain divergence.

## Scope / next step

The A/B is an offline reconstruction from real per-prompt routing, using 8
virtual workers that match the DP8 topology served here. It isolates the routing
effect (workload held constant). The remaining step for a fully live demo is a
real multi-replica llm-d / NVIDIA Dynamo deployment with its KV-aware router,
measuring per-worker expert-load skew directly and confirming EPLB cannot close
the gap the router opens.
