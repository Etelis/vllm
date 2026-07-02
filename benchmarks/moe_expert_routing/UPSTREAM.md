# Upstream plan: what this study supports proposing, and where

State after the full measurement campaign and audit (see `RESULTS.md`): every
claim below is scoped to what survived same-pod falsification.

## 1. vLLM PR — EPLB trigger control and observability

**Branch commits:** forced synchronized rearrangement + worker RPCs
(`trigger_eplb_rearrange`, `get_eplb_stats`), balancedness-threshold
auto-trigger with cooldown, rank-uniform-gating fix.

**Framing: operational control, not a throughput claim.**

- `EPLBConfig.rebalance_threshold` implements the balancedness-triggered
  rebalancing left as an unchecked to-do in the original EPLB PR (#18343):
  rearrange when (windowed) balancedness drops below a threshold instead of
  on a blind period. Verified live: detects a breach in one sampling
  interval and schedules a rearrangement that all EP ranks execute at the
  same step (in-band decision on all-reduced load, so synchronization is
  structural, not best-effort).
- `EplbState.schedule_forced_rearrangement()` + the worker RPC give
  orchestrators (llm-d EPP, Dynamo, autoscalers) an explicit rebalance hook
  for events only they can see (re-sharding, scale-up, tenant onboarding).
  `get_eplb_stats` exports per-logical-expert windowed load and the current
  placement — the engine-side half of llm-d RFC #30696's "load balancers
  have no visibility into EPLB state".
- Two defect fixes worth upstreaming regardless: (a) any monitor gated on
  rank-varying state (`is_dummy`) around a collective hangs workers — the
  fixed gate is rank-uniform; (b) naive threshold monitoring on
  `expert_load_pass` reads a cumulative average when the sliding window is
  not recording — the windowed-delta baseline keeps the signal fresh.

**Honest evidence to include in the PR:**

- Async rearrangement costs ~0.1% throughput per event (same-pod, two
  forced rearrangements mid-run: 8930 -> 8910 tok/s; p99 tail +3.9 s for
  in-flight requests). Threshold-gating is therefore cheap insurance, and
  the periodic default is not harmful either at this scale.
- Placement fit itself is worth ~1% on Qwen3-30B-A3B EP4/NVLink (same-pod
  fitted-vs-anti-fitted crux). The benefit regime for any EPLB policy is
  wide-EP / network-bound dispatch / many-expert models (consistent with
  DeepSeek deployment reports and CRAFT, MLSys'26). Do not claim mid-scale
  wins; the feature is control + observability.
- Cautionary methodology note (useful for reviewers reproducing):
  restart-per-arm A/Bs on shared clusters produced convincing false
  +/-15% effects that passed n=2 with sub-1% error bars; only same-pod
  comparisons were reliable.

## 2. llm-d RFC — the router side

**Standing on Figure 1 (solid, replicated):** the default
prefix-cache-affinity profile turns statistically interchangeable replicas
(cross-replica expert JS 0.001) into fully expert-divergent ones (JS 0.319 =
the intrinsic math<->code divergence) with the identical request set. The
router already *creates* expert specialization as a side effect of KV
affinity; today nothing consumes that fact.

Proposal sketch:

- Consume `get_eplb_stats` in the EPP as an optional per-endpoint signal
  (completes #30696 end-to-end), initially for observability dashboards.
- Fire the rebalance hook on re-shard/scale events (the events the engine
  cannot see). At mid-scale this is hygiene; at wide-EP scale it is expected
  to matter — gate the perf claim on the EP16 validation below.
- The grouping question (which domains should share a replica when
  domains > replicas) remains the open routing-layer decision no engine
  mechanism can make; its measurable payoff at scale is redundancy/KV
  memory economics, not step time.

## 3. Required before submitting perf-motivated pieces: EP16 validation

Cluster nodes expose `rdma/roce_gdr` (Mellanox RoCE + GPUDirect), so the
wide-EP regime is testable in-house: 2 nodes x 8xH100, llm-d wide-EP (LWS)
pattern, DP16+EP16, same-pod crux protocol (route-map flips on live pods)
for: placement fit magnitude, redundant-experts vs KV-pool trade, threshold
trigger value under drift. Outcomes decide whether the vLLM PR ships with a
perf section or as pure control/observability, and whether the llm-d
grouping RFC claims memory wins.
