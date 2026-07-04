# MoE routing/EPLB economics pipeline

One command turns a routed-experts capture into a deployment recommendation
that connects the two halves of MoE serving nobody was connecting: **what the
KV-cache-aware router does to expert load, and what fixing that costs in KV
cache.**

```bash
python -m benchmarks.moe_expert_routing.pipeline \
    --capture ./live_out/<run> --ep 16 --fabric rdma \
    --balance-target 0.95 --out ./pipeline_report
```

## What it computes (each stage validated on a live H100 cluster)

| stage | output | validation |
| --- | --- | --- |
| profiles | per-workload hot experts, concentration | top-8 sets reproduce across independent captures; bootstrap CI on JS +-0.003 |
| divergence | cross-workload JS, purity curve | live llm-d A/B: affinity routing JS 0.001 -> 0.319 (high-purity regime) |
| placement | balancedness default/fitted/anti across EP sizes, vLLM's real EPLB algorithm | model predicted 0.82 vs 0.78 measured at EP4; holds at batch size 4 |
| redundancy | slots needed for a balance target, pure vs mixed traffic | replica dispatch verified = engine's uniform token-hash |
| economics | KV cost of those slots | **measured: 4,712 KV tokens per redundant slot per rank, exactly linear (514,192 -> 495,344 -> 476,496 at 0/16/32 slots, EP4/H100)** |
| recommendation | EPLB trigger settings, redundancy count, router-affinity guidance, expected gain by fabric class | trigger machinery exercised live (threshold fired at 0.6975<0.70, all ranks synchronized) |

## The headline findings it encodes

1. **The cache term dominates when active.** With 96 tenants x 3.6K-token
   prefixes against 118K-token replica pools (working set > one pool, fits
   the fleet), prefix-affinity vs load-only routing = 98.5% vs 27.8% hits =
   **+77% goodput, p50 -48%** (measured, cold-start protocol both arms).
   The window is pure arithmetic the pipeline computes: working set vs
   per-replica pool - a pool that expert redundancy shrinks by a measured
   4,712 tokens/slot.

2. **EPLB redundancy is a cache-window knob (the coupling, live).** Same
   fleet, same routing, same 1,194-tenant workload; flipping only
   `num_redundant_experts` 0 -> 32 -> 64 shrank pools by an exact 4,712
   tokens/slot (topology-invariant: DP4+EP4 and TP4+EP4) and moved the
   router's cache outcome 68.6% -> 42.3% -> 4.5% hits (-46% goodput,
   2.5x p50). The EPLB decision and the router's cache win are one
   budget; the pipeline prices them jointly.

3. **The router decides who is hot.** llm-d's default prefix-affinity scorer
   turns interchangeable replicas into expert-divergent silos (math/code
   share ~13% of hot experts). High-purity effect: 80% affinity yields only
   ~30% of the divergence.
4. **Placement fit is regime-dependent.** ~1% on NVLink EP4; ~4% measured
   at network-bound cross-node EP8 (same-pod protocol); headroom grows with
   EP width (34pp balancedness at EP8, model-calibrated).
5. **Purity is not free.** At EP16 the hot expert of a pure workload exceeds
   rank capacity: 16 redundant slots required vs 0 for mixed traffic - a
   measured ~4,712 KV tokens/rank per slot. The router's affinity weight is
   therefore a *memory* knob, not just a latency knob.
6. **EPLB should be triggered, not periodic.** Async rearrangement costs
   ~0.1%; the balancedness-threshold auto-trigger (implemented on this
   branch, upstream TODO of the original EPLB PR) captures placement value
   without cadence tuning; an external trigger RPC lets routers/orchestrators
   fire it on re-shard events.

Companion tools: `live_router_ab.py` (capture + router A/B driver),
`live_analyze.py` (per-replica attribution), `make_charts.py` /
`step2_placement.py` / `redteam_analyses.py` (figures), `RESULTS.md`
(full measurement record including retractions and the adversarial
self-review).
