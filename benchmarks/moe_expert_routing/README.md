# MoE expert-routing stats (KV-routing imbalance study)

Tools to measure, per prompt and per layer, **which experts an MoE model
actually used**, aggregate that across a benchmark, and correlate it with
prefix-cache hits. Built to test the hypothesis that KV-cache-aware routing
(llm-d / Dynamo style) manufactures expert imbalance: prompts sharing a
prefix/domain co-activate the same experts, so routing by cache locality
concentrates hot experts on some workers.

This uses vLLM's upstream **routed-experts capture** (PR #28284): when the
server runs with `--enable-return-routed-experts`, every `/v1/completions`
response carries the *logical* top-k expert id for each token at each MoE layer
(captured before EPLB remap), base64-encoded as a NumPy `.npy` blob.

## 1. Serve the model (8×H100, DP + EP)

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --enable-return-routed-experts \
  --enable-prompt-tokens-details \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --port 8000
```

* `--enable-return-routed-experts` — emit per-token routing (required).
* `--enable-prompt-tokens-details` — report per-request `cached_tokens` for the
  prefix-cache correlation (required).
* Add `--enable-eplb` for the later per-worker imbalance A/B (not needed to
  collect per-prompt stats; logical ids are captured pre-EPLB either way).
* Incompatible with pipeline parallelism, context parallelism, and KV
  connectors (PD-disaggregation / KV-offload). DP+EP is fine.

## 2. Collect per-prompt expert usage over both domains

```bash
python -m benchmarks.moe_expert_routing.run_experiment \
  --host http://127.0.0.1 --port 8000 \
  --domains gsm8k,mbpp \
  --num-questions 500 \
  --num-experts 128 \
  --out-dir ./moe_stats_out
```

Domain A = GSM8K 8-shot (grade-school math); Domain B = MBPP 3-shot (Python
code). Each domain reuses one byte-identical few-shot preamble, so requests hit
the prefix cache within the domain. A single warmup request seeds the shared
preamble before the rest fan out. Per domain we store `hist.npy` (an
`(N, L, E)` stack of per-prompt layer×expert count histograms), a per-prompt
`table.jsonl` (prompt/cached/completion tokens), a few raw `(T, L, K)` sample
arrays, and `config.json`.

## 3. Analyze

```bash
python -m benchmarks.moe_expert_routing.analyze \
  --out-dir ./moe_stats_out --domains gsm8k,mbpp --num-workers 8
```

Reports and writes `analysis_summary.json`:

* **Per-benchmark hot experts** — aggregated per-layer, plus expert-load
  imbalance ratio (`max/mean`), Gini, and entropy.
* **Cross-domain specialization** — per-layer Jensen-Shannon divergence and
  top-expert Jaccard overlap between math and code (the mechanism: do the two
  domains use different experts?).
* **Cache ↔ concentration** — per-prompt Pearson correlation between
  prefix-cache-hit fraction and expert-usage concentration.
* **Illustrative per-worker imbalance** — folds the aggregate expert load onto
  `--num-workers` EP ranks under default block placement.

## Scope note

The per-prompt routing tells us *which experts* each prompt used, but not
*which DP worker* served it. Demonstrating true per-worker imbalance under a
live KV-router needs a multi-worker deployment (or an emulated router that pins
same-domain prompts to a worker); the `illustrative_worker_ir` here assumes
logical block placement and is a precursor signal, not the full claim.

## Validate the decode/metrics locally (no GPU)

```bash
python -m benchmarks.moe_expert_routing.test_expert_stats
```
