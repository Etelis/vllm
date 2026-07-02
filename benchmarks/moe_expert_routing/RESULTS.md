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

---

## Live llm-d deployment (4x / 2x replicas behind the real inference scheduler)

Setup: llm-d inference scheduler (EPP `llm-d-inference-scheduler:v0.7.0`,
default guide profile `queue:2 + kv-cache-utilization:2 + prefix-cache:3`)
behind an Istio Gateway-API gateway, InferencePool over vLLM replicas on
H100-80GB (OpenShift). Driver sends both domains interleaved (fixed-seed
shuffle) through the gateway; replica attribution via completion-id join
against pod logs (`live_router_ab.py` / `live_analyze.py`).

### Live Figure 1 — the router manufactures the skew (4x single-GPU replicas)

Same 600 requests (300 GSM8K + 300 MBPP), identical replicas; only the EPP
profile changes:

- **load-only** (prefix scorer removed): domains mixed per replica
  (~50/50 each), cross-replica expert JS **0.001 bits**, per-replica IR 6.78.
- **default affinity profile**: all 300 MBPP -> one replica, all 300 GSM8K ->
  another (two replicas idle), cross-replica JS **0.319 bits** = the full
  math<->code divergence, per-replica IR 8.89.

Live numbers reproduce the offline reconstruction almost exactly
(IR 8.89/8.90, JS 0.319 vs cross-domain 0.318, per-domain cached tokens
1264/563 vs 1264/561): the offline harness is a faithful router simulator.

### Iteration 1 — standard-stack baselines under load (2x DP4+EP4 replicas)

1000 requests (500+500 interleaved), max_tokens=384 (stop strings dropped),
concurrency 128, affinity profile, `vllm/vllm-openai:v0.23.0`:

| config | out tok/s | p50 | p90 | p99 |
| --- | --- | --- | --- | --- |
| EPLB off | **7574** | 5.86 | 8.23 | 10.36 |
| stock EPLB (defaults, after a full warmup pass) | 6442 | 7.05 | 9.09 | 11.73 |

**Stock EPLB loses ~15% throughput under affinity-pure traffic**: periodic
rearrangement stalls + stats overhead outweigh placement gains, even with
real imbalance available (balancedness mean 0.78, bursts to 0.36). Best
standard config = EPLB off.

### Iteration 2a — router-triggered one-shot specialization (our vLLM patch)

vLLM branch adds `EplbState.schedule_forced_rearrangement()` (fires at an
aligned step so all EP ranks rearrange the same step), forced-pending stats
recording, and worker RPCs `trigger_eplb_rearrange` / `get_eplb_stats`
(reachable via the `/collective_rpc` dev endpoint). Replicas run EPLB
"trigger-only" (`step_interval=1000000`, no periodic rearrangement).

Protocol: warmup pass (router pins domains) -> trigger both replicas ~15 s in
(emulating the router->EPLB channel) -> measured pass on fitted placements.
Both arms on identical branch pods, same seed; arm `off` never triggers.

| arm | out tok/s | p50 | p90 | p99 |
| --- | --- | --- | --- | --- |
| off (EPLB armed, never triggered ~ EPLB off) | 7490 | 5.84 | 7.52 | 9.89 |
| **ours (one-shot trigger after pin)** | **8720 (+16.4%)** | **4.72 (-19%)** | 7.89 | **9.08 (-8%)** |

vs stock EPLB the gain is +35%. Even the warmup pass that absorbed the
rearrangement stall beat the untriggered arm end-to-end (8313 vs 7490).
Branch-vs-release sanity: arm off 7490 ~ iteration-1 EPLB-off 7574.

Repeated with fresh pod restarts (independent placements, caches, nodes):

| arm | rep1 | rep2 | mean | p50 (reps) | p99 (reps) |
| --- | --- | --- | --- | --- | --- |
| off | 7490 | 7608 | 7549 (+-0.8%) | 5.84 / 5.45 | 9.89 / 10.01 |
| ours | 8720 | 8659 | **8689 (+-0.4%), +15.1%** | 4.72 / 4.73 | 9.08 / 9.06 |

The gain is ~19x the off-arm run-to-run spread. The p90 wobble seen in rep1
(7.89 vs 7.52) is inside the off arm's own rep-to-rep p90 range (7.52-8.86):
tail noise, not a regression. The triggered arm is also more reproducible
than the untriggered one (fitted placement removes a variance source).

### Iteration 3 - balancedness-threshold auto-trigger (engine-only ablation)

`EPLBConfig.rebalance_threshold` monitors windowed balancedness (delta of the
all-reduced per-rank load between samples, so it stays fresh even when the
sliding window is not recording) and schedules the same synchronized forced
rearrangement without any router involvement - implementing the balancedness
trigger left as a to-do in the original vLLM EPLB PR. Ablation vs the
router-trigger arm quantifies how much of the win needs router knowledge at
steady state vs a purely engine-side policy (results pending).
