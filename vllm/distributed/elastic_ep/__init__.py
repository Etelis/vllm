# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import ParallelConfig


def can_reuse_fused_moe_kernel(parallel_config: "ParallelConfig") -> bool:
    """Whether fused MoE kernels can be reused across EP rank changes.

    The EP backend must preserve its CUDA graph entries when ranks are connected,
    disconnected, masked, or unmasked.
    """
    return parallel_config.all2all_backend == "nixl_ep"
