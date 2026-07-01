# Validation run (Qwen3-30B-A3B, DP4 + EP)

End-to-end validation of the harness against a live vLLM server. Confirms that
per-prompt / per-layer logical expert usage is captured, aggregates per
benchmark, and joins with prefix-cache hits — and shows the specialization
signal the KV-routing imbalance thesis depends on.

## Setup

- Model: `Qwen/Qwen3-30B-A3B` (128 routed experts, top-8, 48 MoE layers)
- Serving: `--data-parallel-size 4 --enable-expert-parallel
  --enable-return-routed-experts --enable-prompt-tokens-details
  --enable-prefix-caching`
- Image: `vllm/vllm-openai:v0.23.0-cu129-ubuntu2404`
- Hardware: 4× NVIDIA H100-80GB (single node)
- Domains: GSM8K 8-shot (math) and MBPP 3-shot (code), 300 prompts each

Each request returned `routed_experts` decoding to `(num_tokens-1, 48, 8)`
`uint8` logical expert ids in `[0, 127]`. All 600 requests succeeded; the max
expert id observed was 127, so every expert was exercised.

## Per-domain expert usage (aggregated over 300 prompts)

- **gsm8k**: mean layer imbalance ratio (max/mean) 9.40 (max 14.33), mean Gini
  0.629, mean entropy 5.93 / 7.0 bits, mean cached tokens 1264. Global top-8
  experts: 27, 99, 4, 18, 22, 44, 46, 87.
- **mbpp**: mean layer imbalance ratio 8.40 (max 14.13), mean Gini 0.609, mean
  entropy 6.01 / 7.0 bits, mean cached tokens 561. Global top-8 experts: 91, 27,
  122, 71, 59, 64, 0, 28.

Both domains show strong intra-layer expert-load imbalance (a few experts take
~9× the average load) — consistent with expert specialization.

## Cross-domain specialization (the mechanism)

- Expert-usage Jensen-Shannon divergence: mean 0.319 bits, max 0.516 bits at
  layer 6 (all 48 layers exceed 0.09).
- Top-8 hot-expert Jaccard overlap: **0.134**.
- Most divergent layers: 6, 7, 18, 30, 31.

Math and code share only ~13% of their hottest experts (only expert 27 appears
in both top-8 lists). This is the precondition for KV-cache-aware routing to
manufacture per-worker imbalance: if a router pins same-domain prompts to one
worker, that worker's expert load concentrates on that domain's distinct hot
experts.

## Prefix-cache correlation (exploratory)

Per-prompt Pearson r between cache-hit fraction and expert-usage concentration
(mean per-layer Gini): gsm8k +0.49, mbpp −0.58. Signs differ across domains, so
this per-prompt correlation is not a robust causal signal (it is confounded by
prompt length — the shared preamble is a fixed size, so shorter questions have
both a higher cache fraction and a noisier, more concentrated histogram). The
load-bearing evidence is the cross-domain divergence above.

## Scope / next step

This run uses one DP server, which spreads each domain's requests across all
ranks, so the *aggregate* per-worker imbalance ratio is ~1.0 (balanced) — as
expected. Demonstrating the *manufactured* per-worker skew requires pinning
same-domain prompts to a worker (an emulated KV-router, or a real
llm-d / Dynamo deployment) and comparing cache-affinity grouping vs. shuffled
grouping of the identical request set. The harness already emits everything
needed for that A/B.
