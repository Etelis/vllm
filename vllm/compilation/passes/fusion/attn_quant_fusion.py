# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ParamSpec

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
    kNvfp4Dynamic,
    kStaticTensorScale,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

from ..fx_utils import is_func
from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import MatcherQuantFP8
from .rms_quant_fusion import QUANT_OPS, empty_bf16, empty_fp32, empty_i32

logger = init_logger(__name__)
P = ParamSpec("P")
FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

ATTN_OP = torch.ops.vllm.unified_attention_with_output.default
RESHAPE_OP = torch.ops.aten.reshape.default


class AttentionQuantPattern(ABC):
    """
    The base class for Attn+Quant fusions.
    Should not be used directly.
    """

    def __init__(
        self,
        layer: Attention,
        quant_key: QuantKey,
        dtype: torch.dtype,
    ) -> None:
        self.layer = layer
        self.layer_name = layer.layer_name
        self.num_heads = layer.num_heads
        self.head_size = layer.head_size
        self.quant_key = quant_key
        self.quant_dtype = quant_key.dtype
        self.dtype = dtype

        assert self.quant_key in QUANT_OPS, (
            f"unsupported quantization scheme {self.quant_key}"
        )
        self.QUANT_OP = QUANT_OPS[self.quant_key]

    def empty(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs = {"dtype": self.dtype, "device": "cuda", **kwargs}
        return torch.empty(*args, **kwargs)

    def empty_quant(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs = {"dtype": self.quant_dtype, "device": "cuda", **kwargs}
        return torch.empty(*args, **kwargs)

    @staticmethod
    def wrap_trace_fn(
        trace_fn: Callable[P, fx.GraphModule],
        *process_fx_fns: Callable[[fx.GraphModule], None],
    ) -> Callable[P, fx.GraphModule]:
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> fx.GraphModule:
            gm = trace_fn(*args, **kwargs)
            for process_fx in process_fx_fns:
                process_fx(gm)

            return gm

        return wrapped

    @staticmethod
    def fx_view_to_reshape(gm: torch.fx.GraphModule) -> None:
        from torch._inductor.fx_passes.post_grad import view_to_reshape

        view_to_reshape(gm)

    @staticmethod
    def remove_noop_permutes(gm: torch.fx.GraphModule) -> None:
        for node in gm.graph.nodes:
            if not is_func(node, torch.ops.aten.permute.default):
                continue

            dims = node.args[1]
            if any(dim != i for i, dim in enumerate(dims)):
                continue

            # this is now an identity op, remove
            node.replace_all_uses_with(node.args[0])
            gm.graph.erase_node(node)

    def register_if_supported(self, pm_pass: PatternMatcherPass) -> None:
        if self.layer.impl.fused_output_quant_supported(self.quant_key):
            self._register(pm_pass)

    @abstractmethod
    def _register(self, pm_pass: PatternMatcherPass) -> None:
        raise NotImplementedError


class AttentionFp8StaticQuantPattern(AttentionQuantPattern):
    """
    Fusion for Attention+Fp8StaticQuant.

    Only triggers when the attention implementation returns True in
    `fused_output_quant_supported()`. If the pattern is found, the
    Fp8StaticQuant op will be removed from the graph, and its scale
    will be passed into Attention op as the `output_scale` argument.
    """

    def __init__(
        self,
        layer: Attention,
        dtype: torch.dtype,
        symmetric: bool = True,
    ) -> None:
        quant_key = QuantKey(
            dtype=FP8_DTYPE, scale=kStaticTensorScale, symmetric=symmetric
        )
        super().__init__(layer, quant_key, dtype)
        self.quant_matcher = MatcherQuantFP8(quant_key)

    def _register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            output_attn: torch.Tensor,
            scale: torch.Tensor,
            kv_cache_dummy_dep: torch.Tensor,
        ) -> torch.Tensor:
            at1 = auto_functionalized(
                ATTN_OP,
                query=q,
                key=k,
                value=v,
                output=output_attn,
                layer_name=self.layer_name,
                output_scale=None,
                output_block_scale=None,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            attn_out_view = RESHAPE_OP(
                at1[1], [q.shape[0], self.num_heads * self.head_size]
            )

            return self.quant_matcher(attn_out_view, scale)[0]

        def replacement(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            output_attn: torch.Tensor,
            scale: torch.Tensor,
            kv_cache_dummy_dep: torch.Tensor,
        ) -> torch.Tensor:
            # attn output in quant_dtype
            output_attn = torch.empty(
                [q.shape[0], self.num_heads, self.head_size],
                dtype=self.quant_dtype,
                device=q.device,
            )
            at1 = auto_functionalized(
                ATTN_OP,
                query=q,
                key=k,
                value=v,
                output=output_attn,
                layer_name=self.layer_name,
                output_scale=scale,
                output_block_scale=None,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            return RESHAPE_OP(at1[1], [-1, self.num_heads * self.head_size])

        inputs = [
            self.empty(5, self.num_heads, self.head_size),  # q
            self.empty(5, self.num_heads, self.head_size),  # k
            self.empty(5, self.num_heads, self.head_size),  # v
            self.empty(5, self.num_heads, self.head_size),  # attn_output
            empty_fp32(1, 1),  # scale
            self.empty(0),  # kv_cache_dummy_dep
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            AttentionQuantPattern.wrap_trace_fn(
                pm.fwd_only,
                AttentionQuantPattern.fx_view_to_reshape,
                AttentionQuantPattern.remove_noop_permutes,
            ),
            pm_pass,
        )


class AttentionNvfp4QuantPattern(AttentionQuantPattern):
    """
    Fusion for Attention+Nvfp4Quant.

    Only triggers when the attention implementation returns True in
    `fused_output_quant_supported()`. If the pattern is found, the
    Nvfp4Quant op will be removed from the graph, and its scale
    will be passed into Attention op as the `output_scale` argument.
    """

    def __init__(self, layer: Attention, dtype: torch.dtype) -> None:
        super().__init__(layer, kNvfp4Dynamic, dtype)

    def _register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            output_attn: torch.Tensor,
            output_quant: torch.Tensor,
            output_scale: torch.Tensor,
            input_scale: torch.Tensor,
            kv_cache_dummy_dep: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            at1 = auto_functionalized(
                ATTN_OP,
                query=q,
                key=k,
                value=v,
                output=output_attn,
                layer_name=self.layer_name,
                output_scale=None,
                output_block_scale=None,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            attn_out_view = RESHAPE_OP(
                at1[1], [q.shape[0], self.num_heads * self.head_size]
            )
            at2 = auto_functionalized(
                self.QUANT_OP,
                output=output_quant,
                input=attn_out_view,
                output_scale=output_scale,
                input_scale=input_scale,
                is_sf_swizzled_layout=True,
            )
            output_scale_view = torch.ops.aten.view.dtype(at2[2], FP8_DTYPE)
            return at2[1], output_scale_view

        def replacement(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            output_attn: torch.Tensor,
            output_quant: torch.Tensor,
            output_scale: torch.Tensor,
            input_scale: torch.Tensor,
            kv_cache_dummy_dep: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # attention output in quant_dtype
            output_attn = torch.empty(
                [q.shape[0], self.num_heads, self.head_size // 2],
                dtype=self.quant_dtype,
                device=q.device,
            )
            # attention output block scale
            output_scale_view = torch.ops.aten.view.dtype(output_scale, FP8_DTYPE)
            at2 = auto_functionalized(
                ATTN_OP,
                query=q,
                key=k,
                value=v,
                output=output_attn,
                layer_name=self.layer_name,
                output_scale=input_scale,
                output_block_scale=output_scale_view,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            output = RESHAPE_OP(at2[1], [-1, self.num_heads * self.head_size // 2])
            return output, at2[2]

        inputs = [
            empty_bf16(5, self.num_heads, self.head_size),  # q
            empty_bf16(5, self.num_heads, self.head_size),  # k
            empty_bf16(5, self.num_heads, self.head_size),  # v
            empty_bf16(5, self.num_heads, self.head_size),  # output_attn
            self.empty_quant(5, self.num_heads * self.head_size // 2),  # output_quant
            empty_i32(
                128, round_up(self.num_heads * self.head_size // 16, 4)
            ),  # output_scale
            empty_fp32(1, 1),  # input_scale
            self.empty(0),  # kv_cache_dummy_dep
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            AttentionQuantPattern.wrap_trace_fn(
                pm.fwd_only,
                AttentionQuantPattern.fx_view_to_reshape,
                AttentionQuantPattern.remove_noop_permutes,
            ),
            pm_pass,
        )


class AttentionFp8GroupQuantPattern(AttentionQuantPattern):
    """
    Fusion for Attention+Fp8DynamicGroupQuant.

    Only triggers when the attention implementation returns True in
    `fused_output_quant_supported()`. If the pattern is found, the
    per-group Fp8 quant op will be removed from the graph, and its
    scale output will be produced by the attention kernel directly
    via the `output_block_scale` argument.
    """

    def __init__(
        self,
        layer: Attention,
        dtype: torch.dtype,
        group_shape: GroupShape,
        has_col_major_scales: bool = False,
        is_e8m0: bool = False,
        is_tma_aligned: bool = False,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        quant_key = QuantKey(dtype=FP8_DTYPE, scale=scale, symmetric=True)
        super().__init__(layer, quant_key, dtype)
        self.group_shape = group_shape
        self.group_size = group_shape[1]
        self.has_col_major_scales = has_col_major_scales
        self.is_e8m0 = is_e8m0
        self.is_tma_aligned = is_tma_aligned
        self.quant_matcher = MatcherQuantFP8(
            quant_key,
            has_col_major_scales=has_col_major_scales,
            is_e8m0=is_e8m0,
            is_tma_aligned=is_tma_aligned,
        )

    def _register(self, pm_pass: PatternMatcherPass) -> None:
        hidden_size = self.num_heads * self.head_size
        num_groups = hidden_size // self.group_size

        def pattern(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            output_attn: torch.Tensor,
            output_scale: torch.Tensor,
            kv_cache_dummy_dep: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            at1 = auto_functionalized(
                ATTN_OP,
                query=q,
                key=k,
                value=v,
                output=output_attn,
                layer_name=self.layer_name,
                output_scale=None,
                output_block_scale=None,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            attn_out_view = RESHAPE_OP(
                at1[1], [q.shape[0], self.num_heads * self.head_size]
            )

            finfo = torch.finfo(FP8_DTYPE)
            result = torch.empty(
                attn_out_view.shape,
                device=attn_out_view.device,
                dtype=FP8_DTYPE,
            )
            _, result, output_scale = auto_functionalized(
                self.QUANT_OP,
                input=attn_out_view,
                output_q=result,
                output_s=output_scale,
                group_size=self.group_size,
                eps=1e-10,
                fp8_min=finfo.min,
                fp8_max=finfo.max,
                scale_ue8m0=self.is_e8m0,
                dummy_is_scale_transposed=self.has_col_major_scales,
                dummy_is_tma_aligned=self.is_tma_aligned,
            )
            return result, output_scale

        def replacement(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            output_attn: torch.Tensor,
            output_scale: torch.Tensor,
            kv_cache_dummy_dep: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Attention output directly in FP8
            output_attn = torch.empty(
                [q.shape[0], self.num_heads, self.head_size],
                dtype=self.quant_dtype,
                device=q.device,
            )
            at1 = auto_functionalized(
                ATTN_OP,
                query=q,
                key=k,
                value=v,
                output=output_attn,
                layer_name=self.layer_name,
                output_scale=None,
                output_block_scale=output_scale,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            output = RESHAPE_OP(at1[1], [-1, self.num_heads * self.head_size])
            return output, at1[2]

        # Construct scale tensor shape for pattern matching.
        # The scale shape depends on layout (col-major vs row-major)
        # but the pattern matcher only needs a representative tensor.
        if self.has_col_major_scales:
            scale_tensor = empty_fp32(num_groups, 5).permute(1, 0)
        else:
            scale_tensor = empty_fp32(5, num_groups)

        inputs = [
            self.empty(5, self.num_heads, self.head_size),  # q
            self.empty(5, self.num_heads, self.head_size),  # k
            self.empty(5, self.num_heads, self.head_size),  # v
            self.empty(5, self.num_heads, self.head_size),  # output_attn
            scale_tensor,  # output_scale
            self.empty(0),  # kv_cache_dummy_dep
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            AttentionQuantPattern.wrap_trace_fn(
                pm.fwd_only,
                AttentionQuantPattern.fx_view_to_reshape,
                AttentionQuantPattern.remove_noop_permutes,
            ),
            pm_pass,
        )


class AttnFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses post-attention quantization onto attention if supported.

    It uses the pattern matcher and matches each layer manually, as strings
    cannot be wildcarded. This also lets us check support on attention layers
    upon registration instead of during pattern matching.

    Supported fusions:
    - Static per-tensor FP8 (output_scale)
    - NVFP4 dynamic (output_scale + output_block_scale)
    - Dynamic per-group FP8 (output_block_scale, group_size 64/128)
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns = PatternMatcherPass(pass_name="attn_fusion_pass")

        attn_layers = get_layers_from_vllm_config(config, Attention)
        for layer_name, layer in attn_layers.items():
            pattern_fp8 = AttentionFp8StaticQuantPattern(
                layer, config.model_config.dtype
            )
            pattern_fp8.register_if_supported(self.patterns)

            if current_platform.is_cuda() and hasattr(torch.ops._C, "scaled_fp4_quant"):
                pattern_nvfp4 = AttentionNvfp4QuantPattern(
                    layer, config.model_config.dtype
                )
                pattern_nvfp4.register_if_supported(self.patterns)

            if current_platform.is_cuda():
                for group_shape in [GroupShape(1, 128), GroupShape(1, 64)]:
                    for has_col_major in [True, False]:
                        for is_e8m0 in [False, True]:
                            for is_tma_aligned in [False, True]:
                                pattern_group = AttentionFp8GroupQuantPattern(
                                    layer,
                                    config.model_config.dtype,
                                    group_shape=group_shape,
                                    has_col_major_scales=has_col_major,
                                    is_e8m0=is_e8m0,
                                    is_tma_aligned=is_tma_aligned,
                                )
                                pattern_group.register_if_supported(self.patterns)

        if len(attn_layers) == 0:
            logger.warning(
                "Attention + quant fusion is enabled, but no attention layers "
                "were found in CompilationConfig.static_forward_context "
                "so no fusion patterns were registered."
            )

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Fused quant onto %s attention nodes", self.matched_count)

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(
            self,
            AttentionQuantPattern,
            AttentionFp8StaticQuantPattern,
            AttentionNvfp4QuantPattern,
            AttentionFp8GroupQuantPattern,
        )
