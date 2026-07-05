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
  (0.002 -> 0.182; the baseline is the sampling noise floor, so the
  meaningful statistic is the +0.18-bit difference, not the ratio).**

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

**RETRACTED (see audit below): the off/ours gap did not survive replication
and is attributed to pod/node-placement confounding, not the trigger.**

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
steady state vs a purely engine-side policy.

Result (same protocol, threshold 0.7, sample every 32 steps, cooldown 2000;
no manual trigger sent): warmup 8252, **measured 8620 tok/s, p50 4.74** -
within noise of the router-triggered arm (8689 +-0.4%), +14.2% over off.
All 4 EP ranks on both replicas logged the same breach (balancedness 0.6975
< 0.70) and the same aligned fire step (750080): the in-band trigger is
deterministically synchronized. Steady-state conclusion: the engine-only
threshold captures the full win; router involvement is not needed to *fit*
a stable mix. The router channel's remaining value is proactivity under
traffic drift (re-pin/scale events, where reactive detection must first
suffer the imbalance and cooldown gates back-to-back refits) and
cross-replica grouping decisions - measured next.

### Audit - the +15% does not survive falsification (RETRACTION)

Controlled decomposition of the iteration-2a gap:

| tok/s | never rearranged | rearranged |
| --- | --- | --- |
| through gateway | 7549 +-0.8% (original) / **8601 (rep3 rerun)** | 8689 +-0.4% |
| direct to pods | 8761 | 8544-8590 |

- Same-pod fitted-vs-anti-fitted A/B (`crux`): **placement fit is worth ~1%**
  at Qwen3-30B-A3B EP4/NVLink - the benefit available to any placement policy
  at this scale is small.
- Confounds ruled out one by one: completion lengths identical (382.7+-0.2
  mean across all arms), prefix-cache hits identical (933-939 mean, zero cold
  requests), no hidden mid-run refits (trigger counts audited), placements
  verified via get_eplb_stats.
- rep3 rerun of the exact original off protocol is fast (8601): the original
  slow off pair is attributed to node/time confounding (restart-per-arm =
  node lottery on a shared cluster; a partially-busy neighbor GPU slows the
  whole DP wave). The +15.1% causal claim is retracted.

What survives the audit:

1. **Figure 1 (unaffected):** cache-affinity routing manufactures
   cross-replica expert heterogeneity (JS 0.001 -> 0.319 live, matching the
   offline reconstruction). This is a routing-layer fact.
2. **Placement fit ~1% at EP4/NVLink (new, solid):** the only same-pod
   controlled comparison in the series. Implication: EPLB benefit at this
   scale is small; its regime is wide-EP / network-bound dispatch /
   many-expert models.
3. **The machinery (solid):** synchronized forced rearrangement (aligned-step
   design verified: all EP ranks fire the same step), balancedness-threshold
   auto-trigger (detected imbalance 0.6975 < 0.70 and refit autonomously),
   EPLB stats RPC, and the rank-uniform-gating bug found and fixed
   (is_dummy-dependent gate around a collective hangs workers).
4. **Iteration-1's stock-EPLB harm (needs re-validation):** B1/B2 ran on
   different pod generations, so the -15% magnitude is exposed to the same
   confound; the rearrangement-stall cost itself is real and independently
   reported upstream.
5. **Drift finding:** at hard re-pins the dominant cost is prefix-cache loss,
   not placement (refit-at-flip bought nothing); threshold 0.7 also slept
   through the flip (anti-fitted ~ random placement does not necessarily
   breach a threshold tuned for onset).

Methodology rule going forward: **same-pod A/Bs only** (flip route-maps or
config on live pods); restart-per-arm comparisons are not trustworthy on a
shared cluster.

### Same-pod stall-cost test (closes the audit)

Control vs two mid-pass forced rearrangements, same warm pods, back-to-back:
8930 vs 8910 tok/s -> **~0.1% per rearrangement** (p99 +3.9 s from requests
in flight during the weight copy). Async rearrangement is essentially free.
Consequence: iteration-1's "stock EPLB -15%" is also retracted (two
rearrangements cost 0.2%; a periodic cadence cannot cost 15%) - same
node-placement confound, different direction.

### Audited bottom line

At Qwen3-30B-A3B / EP4 / NVLink / ~130 concurrent: expert placement fit is
worth ~1%, async rearrangement costs ~0.1%, and EPLB (any policy) is
net-neutral. Every 10%+ arm-vs-arm effect measured across pod restarts on
this shared cluster was placement/time confounding - it passed n=2
replication with tight error bars in both directions, which is the
methodological headline: **restart-per-arm benchmarks on shared clusters
produce convincing false effects; only same-pod A/Bs are trustworthy.**

Survives: Figure 1 (routing manufactures expert heterogeneity - the llm-d
phenomenon); the vLLM machinery (threshold auto-trigger, synchronized forced
rearrange, EPLB stats RPC) as cheap operational-control/observability
features; two real bug finds (rank-divergent gating around a collective;
cumulative-load staleness in naive threshold monitoring). The EPLB
benefit-regime question moves to wide-EP / network-bound / many-expert
scale, and the routing-layer value (grouping, redundancy<->KV memory) to
future work with same-pod protocols.

## Stepwise validated build (post-audit protocol)

### Step 1 - expert distributions (charts: make_charts.py)

1000 live-captured prompts (500/domain). Gates: GSM8K top-8 hot experts
reproduce the original study exactly (4,18,22,27,44,46,87,99); MBPP 7/8;
per-layer top-8 Jaccard mean 0.134 (study: 0.132); layer 6 most divergent in
both. Layer 6: top-8 of 128 experts carry 43% (math) / 41% (code) of tokens;
per-expert workload exclusivity spans 4 orders of magnitude (e40: 10% of code
tokens, 0.0006% of math).

### Step 2 - rank-load model with vLLM's real placement algorithm (step2_placement.py)

vllm/distributed/eplb/policy/default.py (numpy) applied to step-1 loads.
Gates: fitted beats default at every domain x EP in {2..32}; EP4-default
predicted balancedness 0.82 vs 0.78 measured live; anti-fitted == default
(both ~random for the serving workload). Headroom grows with EP: fitted
holds 1.0 through EP8 while default/anti decay 0.82 (EP4) -> 0.66 (EP8) ->
0.50 (EP16) -> 0.36 (EP32).

### Step 3 - empirical cross-node EP8 (2 nodes x 4 GPU, all-to-all over TCP)

Same-pod crux (wep_crux.sh): serve mbpp on a gsm8k-fitted (anti) placement,
trigger refit mid-traffic, remeasure identical traffic:

| pass | tok/s | p50 | p99 |
| --- | --- | --- | --- |
| anti-fitted | 1072 | 35.8 | 74.3 |
| refitted | **1123 (+4.7%)** | 34.6 | **69.4 (-6.6%)** |

Direction confirmed with the trustworthy protocol; all three metrics move
together. Magnitude is capped by comm-latency dominance: cross-node EP over
TCP runs ~8x slower than single-node EP4/NVLink (1.1K vs 8.6K tok/s), so
compute-balance gains are diluted by the latency floor. The realistic
wide-EP regime (RDMA/RoCE, bandwidth-bound all-to-all) remains untested:
first RoCE attempt failed on NCCL device selection (NCCL_IB_HCA=mlx5 also
matches storage SR-IOV VFs -> "GID table changed" crash); needs explicit
HCA pinning. Ops notes: multi-node vLLM DP behind a headless Service needs
publishNotReadyAddresses=true (readiness/DNS bootstrap deadlock).

### Step 4 - RDMA regime attempt (negative result, ops trail)

Goal: rerun the step-3 crux with the all-to-all on RoCE/GDR instead of TCP.
Five configurations were attempted on the 2-node testbed; cross-node NCCL
formation never stabilized:

1. `NCCL_IB_HCA=mlx5` (wildcard) + GID 3: rings formed, then fatal
   "GID table changed" on a flapping PF (mlx5_4).
2. Exclusion list + GID auto: NCCL selected GID 0 (link-local, not routable
   from the pod netns) -> every QP failed INIT->RTR ("No such device").
3. Exclusion list + GID 3: GID-table churn now visible on many PFs
   (mlx5_5/8/9/16/17) - the Multi-NIC CNI rewrites PF GID tables on every
   pod attach/detach cluster-wide, so churn is continuous on a busy cluster.
4. - fabric attachment (multi-nic-compute NAD from
   openshift-sriov-network-operator, 16 net1-* interfaces with stable pod
   IPs, MTU 9000) + IPC_LOCK/SYS_RAWIO + privileged SCC (the recipe copied
   from a working llm-d GDR deployment in vela-on-prem-operations): no
   crash, but formation hung >40 min amid ongoing GID-change rescans.
5. vela-minimal NCCL env (no GID pin, NCCL_SOCKET_IFNAME=net1-0): did not
   form within a 25-minute decision gate.

Conclusion: cross-node NCCL EP on this multi-tenant Multi-NIC fabric is
blocked by continuous GID-table churn; notably the working GDR consumer on
this cluster (vela llm-d P/D pods) never runs cross-node NCCL - its
cross-pod path is UCX/NIXL. The llm-d wide-EP lane (LWS + DeepEP) is the
supported route and a follow-up should test DeepEP's RDMA path instead of
NCCL. Ops notes worth keeping: publishNotReadyAddresses=true is required
for the DP-coordinator headless Service; the fabric-attach recipe is
NAD multi-nic-compute@openshift-sriov-network-operator + rdma/roce_gdr:1 +
IPC_LOCK/SYS_RAWIO + privileged SCC.

**Step-4 verdict stands on TCP: placement refit converts +4.7% tok/s /
-6.6% p99 at cross-node EP8 (same-pod), with conversion capped by the
comm-latency floor; the balancedness headroom model (step 2) predicts the
gap grows with EP width and fabric bandwidth.**

## Red-team pass (adversarial self-review; analyses in redteam_analyses.py)

Attacking our own findings for sense and scope. Six items, two of which
reverse earlier framings:

1. **Figure 1 is a high-purity phenomenon (scope correction).** Mixing-curve
   analysis on the measured histograms: cross-replica JS vs routing purity is
   superlinear - 80% purity yields only ~30% of the full divergence
   (0.083 of 0.275 bits), 60% purity is negligible (0.009). Production
   deployments where load-spill dilutes affinity will see proportionally
   little expert heterogeneity; the measured full-separation case is the
   pure-affinity limit, real at low load and under strict session pinning.
   Also: quote the JS *difference*, never the ratio to the 0.001 noise floor.

2. **The redundancy flywheel spins backwards (hypothesis refuted).** Using
   the real placement algorithm with replication: at EP16, domain-pure
   traffic needs 16 redundant slots for balancedness >= 0.95 while 50/50
   mixed traffic needs 0; at EP8 neither needs any; at EP32 both need 32.
   Mechanism: mixing disjoint hot-sets flattens peaks below rank fair-share;
   purity sharpens the top expert (~10% of layer tokens) past EP16 rank
   capacity (6.25%), forcing replication. The original "purity -> fewer
   redundant experts -> more KV" hypothesis is wrong for this model's
   concentration profile; if anything, affinity routing *raises* the memory
   cost of balance at the widths where redundancy starts to matter.

3. **Step-3 (+4.7%) needs a drift bound (kept, tightened).** Same-config
   late-pass repeats elsewhere in the session bound non-placement
   pass-order drift to <=~0.5-1%; the anti->refit measurement spans two
   pass positions, so the placement-attributable effect is ~3.7-4.5%,
   with p99 co-movement (-6.6%) as supporting evidence. n=1 (the repeat
   was sacrificed to the RoCE roll); treat as a single validated point.

4. **Result files are mutable state (process defect, caught).** The wep_*
   output dirs were partially overwritten by a later racing launch against
   a crashing cluster (mtimes prove it: m_refit 20:45 original; g_*/m_anti
   20:49-20:58 with 400/500 connection errors). The step-3 numbers stand
   because the completed chain printed them at 20:45 before contamination.
   Rule: archive result JSONs off-pod at run completion; unique out-dirs
   per launch.

5. **RoCE root cause is unproven (wording corrected).** "Blocked by GID
   churn" was a correlation; alternates (MTU/routing across 16 ipvlan
   subnets, NCCL cross-NIC pairing) were not isolated. Honest statement:
   formation never stabilized amid continuous GID-change rescans; root
   cause not established; DeepEP-over-LWS is the supported path to test.

6. **Small-batch check (model survives).** Per-batch balancedness at EP8
   with batches of 4/16/64/256 prompts: fitted 0.97-1.00 vs default ~0.66
   at every batch size - the aggregate-load model is not a large-batch
   artifact.

Net position after red-teaming: what makes sense and survives is (i) the
routing->expert-identity phenomenon *in the high-purity regime*, (ii) the
calibrated placement/headroom model incl. small-batch validity, (iii) a
single same-pod ~4% conversion point at network-bound EP8, and (iv) the
trigger/threshold machinery. What does not survive: the redundancy/memory
dividend of purity (reversed), any router-over-engine advantage at steady
state (iter-3), refit value at cache-destroying flips (iter-4), and all
restart-per-arm throughput comparisons. The strongest remaining research
line is characterizing *when* purity pays (wide-EP bandwidth-bound
all-to-all) versus when it costs (redundancy at EP16+), which is a
placement-economics question the router must be aware of - a more nuanced,
and more defensible, version of the original idea.

### Red-team round 2 (attacking the round-1 corrections; redteam_round2.py)

1. **Redundancy reversal survives its own audit.** Threshold sensitivity:
   the EP16 pure-needs-16 / mixed-needs-0 split holds for balancedness
   targets 0.90 and 0.95 (pure r=0 sits at 0.847, mixed at 0.984) and
   converges to equal-16/16 only at 0.99. Dispatch-model sensitivity:
   vLLM selects among an expert's replicas by a Knuth multiplicative hash
   of the token index mod replica count (fused_moe/router/base_router.py),
   i.e. uniform in expectation - matching the analysis's ideal-split
   assumption; the worst-case bracket (all load on one replica, where
   replication would *hurt*) is not the shipped behavior.

2. **Step-1 statistics are stable.** Bootstrap over prompts (500
   resamples): JS mean 0.275 with 95% CI [0.273, 0.278]; per-layer top-8
   Jaccard 0.133 with CI [0.126, 0.143]. No fragility.

3. **The +4.7% is quantitatively coherent with comm dominance.** With the
   TCP/NVLink step-time ratio implying ~87% added-comm share, a refit that
   removes the 0.66->1.0 max-rank penalty predicts +1.9/3.2/4.4% for MoE =
   30/50/70% of the non-comm budget. Observed 4.7% (drift-adjusted
   ~3.7-4.5%) sits in the coherent band for an A3B MoE where expert FFNs
   dominate compute. No anomaly; also explains why the TCP regime caps
   conversion.

Attacks that failed to land are as informative as the two that landed in
round 1: the phenomenon's statistics, the placement model's engine
assumptions, and the single live conversion point are internally
consistent.

### Red-team round 3 (capture semantics, layer scope, asymmetry)

1. **Preamble-artifact attack fails (Figure 1 strengthened).** Verified in
   source: the routed-experts capturer records router `topk_ids` from
   forward passes only (routed_experts_capturer.py; output_processor
   concatenates per-step chunks) - prefix-cache-HIT tokens never route and
   contribute nothing. With ~60-85% of prompt tokens cache-hit in our runs,
   the measured divergence is generation-content-driven, not shared-preamble
   copying. Corollary scope note: histograms under-represent full-prefill
   routing.

2. **Layer-6 charts are labeled correctly but the story is not layer-6-
   specific.** Per-layer default-placement balancedness at EP8 (gsm8k):
   layer 6 = 0.55, median layer = 0.68, worst = 0.50 (layer 45), best =
   0.82 - every layer is materially imbalanced under default placement,
   and all model/redundancy conclusions already use full 48-layer weights.

3. **Asymmetric splits add nothing new:** the 2-replica JS depends on the
   purity margin exactly as the symmetric mixing curve (70/30 -> 0.036,
   90/10 -> 0.156).

4. **Scope label added:** the "~8x TCP collapse" is a single-point
   observation (Qwen3-30B-A3B, DP8+EP8 over 2 nodes, concurrency 128, this
   cluster's pod network) - a regime marker, not a general constant.

Local/analytical attack surface is exhausted after three rounds. Remaining
attacks require new data: other concentration profiles (256-expert models -
does the redundancy reversal generalize?), >2-domain captures, and the
RDMA-regime conversion point.

## Live redundancy <-> KV economics (EP4/H100, single node, measured 2026-07-03)

`num_redundant_experts` swept 0/16/32 on the same deployment (verified in
pod args after an earlier silent sed failure - patch configs via JSON edit,
never sed on YAML):

| redundant slots (per rank) | KV pool (tokens/rank) | delta | tok/s | mean cached |
| --- | --- | --- | --- | --- |
| 0 (0) | 514,192 | - | 7360-7555 | 937 |
| 16 (4) | 495,344 | -3.7% | 7074 | 940 |
| 32 (8) | 476,496 | -7.3% | 6691 | 939 |

- **Exactly linear: 4,712 KV tokens per redundant slot per rank** (both
  deltas divide evenly; the 32-slot pool was predicted to 4 tokens).
- Hit-rate is stable across configs (the workload does not saturate KV);
  the KV cost converts to hit-rate/goodput loss only under KV pressure -
  the pipeline reports the token cost directly.
- Throughput declined ~5.4% from red16 to red32 on the *same node*
  (consecutive pod IPs), suggesting a real compute cost of replication
  (more, smaller expert groups + hash dispatch) beyond the memory cost;
  red0-vs-red16 crosses nodes (restart variance +-1.3% same-config), so
  treat the throughput column with the usual caution, the KV column is
  deterministic engine accounting.
- Closes the loop on the EP16 purity finding: the 16 extra slots that
  domain-pure traffic requires cost a measured ~4,712 KV tokens/rank
  (~0.9% of pool per slot-per-rank at this config) - the price of purity
  is now in real units, and pipeline.py quotes it in recommendations.

### KV pressure: the pool converts to goodput ~1:1 (same node, deterministic)

Saturating closed-loop workload (gsm8k only, 896 requests, concurrency 448,
3072-token generations - vLLM admission-controls, so pressure appears as
queueing, not preemptions; hit counters identical across configs):

| config | KV pool (tok/rank) | goodput | p50 | p99 |
| --- | --- | --- | --- | --- |
| redundant=32 | 476,496 | 6646 tok/s | 178.7 s | 245.9 s |
| redundant=0 | 514,192 (+7.9%) | **7184 tok/s (+8.1%)** | 161.1 s (-9.9%) | 207.6 s (-15.6%) |

Consecutive pod IPs (same node), deterministic pool sizes, identical
prefix-hit counts (1,128,752/1,192,315 = 94.7% both): under KV-bound load,
**goodput scales with the KV pool almost exactly** (+8.1% vs +7.9%).

This prices the whole trade in one currency: at EP16, domain-pure traffic's
16 required redundant slots cost 3.7% of the KV pool = ~3.7% goodput under
KV pressure, against a placement-fit gain of ~1% (NVLink) to ~4%
(network-bound EP8). **On this model at these widths, the router's affinity
weight is net-negative for KV-bound serving at EP16 and net-neutral-to-
positive only where all-to-all is network-bound and KV is not the binding
constraint** - the regime map the pipeline now encodes.

## The cache term completes the equation (multi-tenant, measured live)

Workload: 96 tenants, each with a distinct ~3.6K-token preamble (20-shot
shuffled GSM8K), 960 requests interleaved, decode 64, concurrency 96, four
1-GPU replicas (118,176-token KV pool each) behind llm-d. Prefix working
set ~350K tokens: does not fit one replica, fits the fleet if partitioned.
Both arms cold-started, warmup pass + measured pass (same protocol).

| routing | hit rate | goodput | p50 | p99 |
| --- | --- | --- | --- | --- |
| load-only (prefix scorer off) | 27.8% | 2367 tok/s | 2.50 s | 3.63 s |
| prefix-affinity (llm-d default) | **98.5%** | **4183 tok/s (+77%)** | **1.31 s (-48%)** | 2.63 s (-28%) |

Mechanism: load-only routing duplicates every tenant's prefix across all
replicas, overflowing each 118K pool (thrash: 28% hits, constant 3.6K-token
re-prefills); affinity partitions 24 tenants/replica (~87K, fits) - 98.5%
hits. A mild-window variant (240 tenants x ~0.5K preambles, working set
~fits one pool - caused by a driver default passing 3-shot preambles, kept
as an honest scope point) still showed 98.7% vs 72.9% hits but only +4%
goodput / -9% p99: **the cache term's size is set by the ratio of prefix
working set to per-replica pool, not by hit-rate deltas alone.**

### The complete, measured trade equation for router affinity

  net(affinity) = cache_term + placement_term - redundancy_tax

- **cache_term**: 0 when the working set fits every replica; **+77%
  goodput / -48% p50** when it exceeds one pool but fits the partitioned
  fleet (this section); degrades again if per-replica shares overflow.
- **placement_term**: +1% (NVLink) to +4% (network-bound EP8) via
  trigger-fitted EPLB placements.
- **redundancy_tax**: 0 slots at EP4/EP8; 16 slots at EP16 for pure traffic
  = 4,712 KV tokens/rank each ~ -3.7% goodput when KV-bound.

The earlier "affinity is net-negative at KV-bound EP16" verdict holds only
when the cache term is inactive. When the multi-tenant window is open, the
cache term dominates everything else by an order of magnitude - and the
KV-pool arithmetic (working set vs pool sizes, both of which EPLB
redundancy shrinks) is what the pipeline computes to decide the router
weight and redundancy jointly.

### Window map: the boundaries are predictable arithmetic

Sweeping the tenant prefix working set against fixed pools (118,176
tokens/replica x 4 replicas), measured both routing arms at each point:

| working set | load-only hits / tok/s | affinity hits / tok/s | affinity advantage |
| --- | --- | --- | --- |
| 116K (32 tenants) | 85.9% / 3501 | 99.8% / 4427 | +26% |
| 350K (96 tenants) | 27.8% / 2367 | 98.5% / 4183 | **+77%** |
| 580K (160 tenants) | 17.3% / 2243 | 64.4% / 3103 | +38% |

The measured curves land on the pool-arithmetic boundaries: affinity holds
~99% hits until its per-replica share (WS/4) exceeds one pool (160-tenant
point: 145K > 118K -> 64%), and load-only decays as soon as duplication
inflates the effective working set past the fleet. One refinement over the
naive prediction: load-only pays a duplication tax even *below* the
single-pool boundary (86% vs 99.8% hits at 116K), so affinity never lost in
the tested range - the "window" has a soft lower edge set by the
duplication factor, not a hard wall. The advantage peaks mid-window (+77%)
exactly where the arithmetic says partitioning matters most. This is the
pipeline's core claim demonstrated end-to-end: the regime is computable
from working-set and pool sizes before you deploy anything.

## Keystone: the EPLB-to-cache coupling, measured live

Everything above establishes the links separately: the router manufactures
expert divergence (Fig-1), EPLB redundancy costs KV (4,712 tokens/slot),
KV pressure costs goodput (~1:1), and the cache term dominates when the
multi-tenant window is open (+77%). Honest gap, raised as a direct
challenge: no single experiment had shown an *EPLB decision* changing a
*router cache outcome*. The +77% result involved no EPLB at all.

This experiment closes the loop with one knob. Fleet: 2x TP4+EP4 replicas
(Qwen3-30B-A3B, stock vLLM v0.23.0, one 2.28M-token KV pool per replica),
llm-d prefix-affinity routing, fixed workload of 1,194 tenants x 3.6K-token
prefixes (per-replica share 2.167M tokens, chosen at the midpoint of the
82K-token band that fits at red=0 and overflows at red=32). The only
variable across arms: `num_redundant_experts` in `--eplb-config`. Each arm
cold-starts (clean pod replace), runs a warm-up pass then a measured pass.

| redundant slots | pool/replica | WS fill | hit rate | goodput | p50 |
| --- | --- | --- | --- | --- | --- |
| 0 | 2,279,808 | 95% | **68.6%** | 4,646 tok/s | 0.96 s |
| 32 | 2,129,024 | 102% | **42.3%** | 3,413 tok/s | 1.89 s |
| 64 | 1,978,240 | 110% | **4.5%** | 2,523 tok/s | 2.39 s |

The EPLB knob alone moves the router's cache outcome by -64pp hits, -46%
goodput, 2.5x p50 - a monotone dose-response tracking the pool arithmetic
exactly. At red=64 the measured pass (4.5%) is statistically the cold pass
(3.4%): the prefix cache retains nothing.

Supporting evidence that makes it airtight:

- **Slot cost is topology-invariant.** Pool shrink = 4,712 KV tokens/slot,
  exact at 0/16/32 slots on DP4+EP4 and 0/32/64 on TP4+EP4; the red=64
  pool was predicted (2,279,808 - 64x4,712 = 1,978,240) and measured at
  precisely that value.
- **Not a routing artifact.** Per-replica metrics deltas show near-50/50
  query splits and lockstep hit rates (66.6%/70.7% at red=0, 42.0%/42.7%
  at red=32): uniform pool pressure, not one replica thrashing.
- **Usable pool < nominal pool.** The red=0 control lands at 68.6% (not
  ~99%) because 95% nominal fill already evicts: block quantization,
  active-decode blocks, and the eviction watermark shave ~5-8% off the
  advertised pool. Capacity planning against nominal pool sizes
  overcommits; the pipeline uses a 0.9 safety factor.

Scope notes: hit-frac is the lead metric (pure cache arithmetic, immune to
node placement); nodes pokprod-b93r38s0/b93r44s3 held fixed across arms;
the stock image lacks `--enable-return-routed-experts`, so the driver's
per-request `ok` flag reads false while token/latency/cache fields remain
valid (metrics computed from response rows).

The chain is now closed end-to-end: the router's affinity creates the
imbalance EPLB reacts to; EPLB's fix (redundant experts) spends the very
KV pool the router's cache win depends on; and this experiment shows the
spend destroying the win, live. Joint control of the two - the pipeline's
recommendation output - is not an optimization nicety; the dose-response
says the interaction term is first-order.

## From findings to controls: the clamp, and a negative result on locality

The keystone dictates one algorithm change and invited a second. We built
the first and killed the second with data before building it.

### The KV budget clamp (implemented: `pipeline.py kv_budget_clamp`)

Redundancy becomes a budgeted purchase instead of a static config:
`granted = min(requested, (pool - 1.1 x working_set/replicas) / 4712)`.
Replayed against the measured dose-response it makes the right call at
every point: at the keystone workload it refuses all 32 slots (predicting
"overflow -> collapse" at requested - we measured 42.3% hits - and "edge"
at granted=0 - we measured 68.6%); at 900 tenants it grants 32 with
headroom for 102. The missing signal (working set) is exactly what the
llm-d prefix scorer already tracks; the clamp is the ~20-line coupling
between the two layers. Live validation on a second model: pending
(Qwen3-235B-A22B-FP8 rig).

### Origin-aware placement: offline negative result

The proposal: in DP+EP, keep per-rank load matrices instead of all-reduced
sums, and place experts near the ranks whose tokens use them (the router's
affinity creates exactly that structure). Tested offline on the measured
per-domain loads with vLLM's real placement policy (2 origin groups,
EP4/EP8, locality-greedy with a 10% balance cap):

| EP | placement | on-rank dispatch | balancedness | all-to-all cut |
| --- | --- | --- | --- | --- |
| 4 | stock (balance-only) | 0.259 | 1.000 | - |
| 4 | origin-aware, capped | 0.336 | 0.861 | 10.4% |
| 4 | locality ceiling | 0.375 | 0.458 | 15.7% |
| 8 | origin-aware, capped | 0.168 | 0.854 | 4.1% |

Verdict: not worth engine changes on this model family. The ceiling is set
by **bulk expert-usage overlap: 0.84** between domains - Figure-1's
divergence (JS 0.319, ~13% shared top-8) describes the head of the
distribution, but 84% of token volume flows to experts both domains use.
Locality can only capture the exclusive tail, diluted by ranks-per-group,
and pays for it in balance (stock is already at 1.000). A ~10% all-to-all
cut in a regime where all-to-all is 20-40% of step time is a low
single-digit end-to-end gain at a 14pp balance cost.

Falsifiable follow-up, not assumed: fine-grained-expert models
(DeepSeek-V3: 256 experts, Qwen3-Next: 512) train narrower experts and
should show lower bulk overlap; measure overlap there before revisiting.

### DeepSeek-V3 slot economics (arithmetic, not measured)

For scale: a DeepSeek-V3 redundant slot is 44 MB x 58 MoE layers = 2.55 GB
FP8. Under MLA the KV cache is only ~70 KB/token, so one slot displaces
~36,000 tokens of the hosting rank's pool - the reference config (32
redundant slots at EP16, 2/rank) spends roughly 18% of each rank's KV pool
on redundancy. The clamp question is not academic at DeepSeek scale; it is
the first thing to check.

## Second model: the clamp validated at 235B scale

Same protocol as the keystone, bigger stakes: Qwen3-235B-A22B-FP8 (94 MoE
layers, ~19 MB experts), 2x TP4+EP4 replicas, llm-d prefix-affinity, fixed
116-tenant workload (share 210,540 tokens/replica = 91% of the red=0
pool). The clamp's verdict was committed *before* the measurement
(pre-registered): requested 8 slots -> refuse all ("overflow, fill 135%,
expect cache collapse"); granted 0 -> "edge, fill 91%, partial eviction".

| redundant slots | pool/replica | WS fill | hit rate | goodput | p50 |
| --- | --- | --- | --- | --- | --- |
| 0 | 230,224 | 91% | **84.2%** | 1,248 tok/s | 3.34 s |
| 8 | 155,696 | 135% | **9.5%** | 527 tok/s | 7.34 s |

Eight redundant slots - 6% of the expert count - cost 58% of goodput.
Both predicted regimes matched: 84.2% is partial eviction at the edge;
9.5% is statistically the cold floor (warm pass: 1.5-1.8%).

**The slot-cost scaling law holds across models.** Predicted before boot
from `expert_bytes / kv_bytes_per_token_per_gpu`: ~9,220 tokens/slot.
Measured: (230,224 - 155,696) / 8 = **9,316** (1% error). With the 30B's
4,712 (exact at six points across DP4+EP4 and TP4+EP4), the KV price of
redundancy is now a *computable constant* per model - which is what lets
the clamp refuse a bad config before it ships. Per-slot pain scales with
model size: 0.2% of pool per slot at 30B, 4% at 235B - the bigger the
model, the more the balancer's memory appetite matters.

Ops appendix (each cost us a debugging round today):

- **InferencePool selectors pin fleets.** `qwen-pool` matched
  `llm-d.ai/model: qwen3-30b-a3b`; the 235B pods were invisible to EPP
  ("no pods available in datastore" -> gateway 500/503 on everything).
  Fleet swaps must update the pool selector or drop model labels from it.
- **EPP only datastores READY pods** - a gateway check 20s after EPP
  restart, mid-boot of a 20-minute model load, correctly fails. Order:
  fleet ready first, then verify the gateway.
- **The cluster reaper scales idle GPU pods to zero** (it did, during a
  token-expiry gap) and node taints move (disk-pressure took out the
  original target node). Rig scripts must re-verify placement on resume.
- The A/B driver hardcoded the served model name; the 235B server 404'd
  every request. Now auto-discovered from `/v1/models` (`--model` to
  override).

## DeepSeek-V3 closed-loop attempt: ops trail (in progress)

Goal: the closed loop (telemetry-driven redundancy decision vs DeepSeek's
reference config) on DeepSeek-V3-0324 FP8, 2 nodes x 8 H100, TP8+DP2+EP16.
Eight boot attempts over one night, each failing strictly deeper - recorded
because the ladder itself is reusable ops knowledge:

1. `--data-parallel-hybrid-lb` (per-node API servers): engine-ready
   handshake never completes cross-node -> revert to the proven
   headless-worker shape (single API on head).
2. DeepGEMM JIT: `ninja: build stopped` during FP8 grouped-GEMM warmup ->
   `VLLM_USE_DEEP_GEMM=0`.
3. flashinfer/TRT-LLM fp8_blockscale JIT: `fatal error: nvrtc.h` - the
   serving image ships CUDA 13 *runtime* libs but no dev headers.
4. Pip-wheel workaround trap: `nvidia-cuda-nvrtc-cu13` on PyPI is a 0.0.1
   stub; installing `-cu12` instead compiles the JIT against CUDA 12 on a
   CUDA 13 torch -> runtime illegal access surfacing in fused_add_rms_norm.
   The correct cu13 package is the *unified* name `nvidia-cuda-nvrtc==13.*`.
5. Linker: wheels ship only versioned `libnvrtc.so.13` - `-lnvrtc` needs an
   unversioned symlink.
6. `--enforce-eager` bypasses a capture-time flashinfer workspace sizing
   error (`Buffer 1146880, Required 3670016`).

Terminal state: engine-ready timeout persists on the FP8-blockscale JIT
path with the aligned toolchain; the clean fix is an image with the CUDA-13
dev toolchain baked (devel base), not runtime patching.

What the attempts DID establish on V3: EPLB initializes on DeepSeek-V3
(`TorchDistGlooStagedEplbCommunicator`), weights load at 44.5 GiB/GPU FP8,
and the **measured pool is 148,000 tokens/replica** under MLA - so the
pre-registered slot cost (36,330 tokens per slot-per-GPU = 2.554 GB
expert-set / 70.3 KB-per-token MLA cache) makes DeepSeek's own reference
convention (2 slots/GPU at EP16) cost **49% of the KV pool**. The clamp
question is sharpest exactly where EPLB was born.

Next move (not a boot retry): W4A16-quantized V3 uses precompiled Marlin
kernels - no flashinfer JIT, single-node, no cross-node DP.

### Two hard compatibility findings from the V3 pursuit

- **EPLB does not support quantized MoE.** vLLM raises
  `NotImplementedError: EPLB is not supported MoeWNA16Method`
  (`fused_moe/layer.py`) - the AWQ/W4A16 single-node route to DeepSeek-V3
  is closed by design, not by bug. Consequence worth stating on its own:
  today's quantized MoE deployments cannot use expert rebalancing at all,
  so the redundancy-vs-cache trade this study prices exists only for
  BF16/FP8-native fleets.
- Serving DeepSeek AWQ checkpoints at all requires
  `--quantization moe_wna16` (stock `awq_marlin` rejects the checkpoint's
  mixed-quantization layer map with a `ValueError` in `is_layer_skipped`).

### V3 FP8 pursuit: final status (9 attempts, hard stop)

With real toolkit headers (`apt cuda-nvrtc-dev-13-0`) the flashinfer
TRT-LLM fp8-blockscale JIT compiles and links - and its kernel then faults
at runtime (illegal access surfacing in `fused_add_rms_norm`) on every
rank, identically. The blocker is the kernel/environment combination
(branch-built vLLM + CUDA-13 torch + flashinfer wheel on H100), not
deployment configuration. Remedies, in order of preference: (a) serving
image built on a devel base with the matching flashinfer/vLLM pin, (b) a
vLLM build where the blockscale path can be steered to CUTLASS/Triton
end-to-end, (c) upstream flashinfer fix. All are image-build tasks, not
boot-time patches.

The V3 pursuit still banked: pool 148,000 tokens/replica measured under
MLA at TP8+DP2+EP16; EPLB confirmed initializing on DeepSeek-V3 (Gloo
staged communicator); slot-cost pre-registration (36,330 tokens/slot-per-
GPU => DeepSeek's own reference convention costs 49% of the pool); the
EPLB-unsupported-on-quantized-MoE finding; and a 9-step ops ladder future
deployments can skip.
