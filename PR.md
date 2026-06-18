# Purpose

Disable the parallel-agnostic fs-tier KV cache (#44733) when the V2 model runner
is active. The parallelism-invariant assumption it relies on does not hold under
V2.
