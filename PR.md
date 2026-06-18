## Purpose

Disable the parallel-agnostic fs-tier KV cache (#44733) when the V2 model runner
is active. The feature collapses tp/pp/pcp/dcp and rank out of the cache
namespace for a single full-attention group, assuming offloaded blocks are
parallelism-invariant. That invariant is not known to hold under V2, so sharing
a cache dir across layouts could alias distinct blocks. Gated on the canonical
`vllm_config.use_v2_model_runner` (not a raw env read), alongside the existing
MLA/multi-group exclusions.
