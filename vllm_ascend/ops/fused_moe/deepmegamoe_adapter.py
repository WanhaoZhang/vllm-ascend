# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapter for DeepGEMM-Ascend's fused MegaMoE operator."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.profiler.moe_profile import moe_profile_range
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

_SYMM_BUFFERS: dict[tuple[int, int, int, int, int, int], Any] = {}
_SYMM_BUFFER_LOCK = threading.Lock()


def release_deepmegamoe_buffers() -> None:
    """Release borrowed views after pending launches, before HCCL teardown."""
    with _SYMM_BUFFER_LOCK:
        if not _SYMM_BUFFERS:
            return
        torch.npu.synchronize()
        for buffer in _SYMM_BUFFERS.values():
            buffer.destroy()
        _SYMM_BUFFERS.clear()


@dataclass(frozen=True)
class DeepMegaMoELayerCapability:
    supported: bool
    reason: str = ""


def deepmegamoe_backend_enabled() -> bool:
    config = get_ascend_config()
    return config.enable_fused_mc2 == 1 and getattr(config, "fused_mc2_backend", "auto") == "deepmegamoe"


def evaluate_deepmegamoe_layer(
    moe_config,
    quant_method: object,
    activation: Any,
    *,
    n_shared_experts: int = 0,
    expert_map_path: str | None = None,
    mix_placement: bool = False,
    log2phy: torch.Tensor | None = None,
    swiglu_limit: float = 0.0,
    apply_router_weight_on_input: bool = False,
    has_bias: bool = False,
    zero_expert_type: str | None = None,
    is_sequence_parallel: bool = False,
    pcp_size: int = 1,
    speculative_model: bool = False,
    multistream_overlap_gate: bool = False,
    device_type: AscendDeviceType | None = None,
) -> DeepMegaMoELayerCapability:
    """Check a routed-expert layer against DeepGEMM MegaMoE's current contract."""
    device_type = device_type or get_ascend_device_type()
    if device_type != AscendDeviceType.A5:
        return DeepMegaMoELayerCapability(False, f"DeepMegaMoE requires A5, got {device_type}")
    quant_scheme = getattr(quant_method, "quant_method", None)
    if quant_scheme is not None and hasattr(quant_scheme, "quant_type"):
        quant_method = quant_scheme
    quant_type = getattr(quant_method, "quant_type", QuantType.NONE)
    if quant_type != QuantType.W4A8MXFP:
        return DeepMegaMoELayerCapability(False, f"DeepMegaMoE requires W4A8 MXFP4 weights, got {quant_type}")
    if getattr(quant_method, "dynamic_eplb", False):
        return DeepMegaMoELayerCapability(False, "DeepMegaMoE does not support dynamic EPLB")
    if n_shared_experts > 1:
        return DeepMegaMoELayerCapability(False, "DeepMegaMoE currently supports at most one outer shared expert")
    if expert_map_path or mix_placement or log2phy is not None:
        return DeepMegaMoELayerCapability(False, "DeepMegaMoE requires contiguous expert placement")
    if not math.isfinite(swiglu_limit) or swiglu_limit < 0:
        return DeepMegaMoELayerCapability(False, f"swiglu_limit must be finite and nonnegative, got {swiglu_limit}")
    if apply_router_weight_on_input:
        return DeepMegaMoELayerCapability(False, "DeepMegaMoE requires router weights on the combined output")
    if has_bias:
        return DeepMegaMoELayerCapability(False, "DeepMegaMoE does not support expert bias")
    if zero_expert_type is not None:
        return DeepMegaMoELayerCapability(False, "DeepMegaMoE does not support zero experts")
    if is_sequence_parallel:
        return DeepMegaMoELayerCapability(False, "DeepMegaMoE does not support sequence parallelism")
    if pcp_size != 1:
        return DeepMegaMoELayerCapability(False, "DeepMegaMoE requires PCP size 1")
    if speculative_model:
        return DeepMegaMoELayerCapability(False, "DeepMegaMoE does not support speculative MoE")
    if multistream_overlap_gate:
        return DeepMegaMoELayerCapability(False, "DeepMegaMoE does not support multistream gate overlap")

    activation_name = str(getattr(activation, "value", activation)).lower()
    if activation_name not in {"silu", "swiglu", "moeactivation.silu"}:
        return DeepMegaMoELayerCapability(False, f"DeepMegaMoE requires SiLU, got {activation}")
    if getattr(moe_config, "in_dtype", torch.bfloat16) != torch.bfloat16:
        return DeepMegaMoELayerCapability(False, f"DeepMegaMoE requires BF16 input, got {moe_config.in_dtype}")

    hidden = int(moe_config.hidden_dim)
    intermediate = int(moe_config.intermediate_size_per_partition)
    if hidden <= 0 or hidden % 256:
        return DeepMegaMoELayerCapability(False, f"hidden size must be positive and divisible by 256, got {hidden}")
    if intermediate < 512 or intermediate % 256:
        return DeepMegaMoELayerCapability(
            False, f"intermediate size must be at least 512 and divisible by 256, got {intermediate}"
        )
    ep_size = int(moe_config.ep_size)
    num_experts = int(moe_config.num_experts)
    if ep_size < 2:
        return DeepMegaMoELayerCapability(False, "DeepMegaMoE requires expert parallelism")
    if num_experts % ep_size:
        return DeepMegaMoELayerCapability(False, f"experts must be divisible by EP size, got {num_experts}/{ep_size}")
    top_k = int(moe_config.experts_per_token)
    if top_k < 1 or top_k > min(32, num_experts):
        return DeepMegaMoELayerCapability(False, f"topK must be in [1, {min(32, num_experts)}], got {top_k}")
    return DeepMegaMoELayerCapability(True)


def _pack_mxfp4_scales(scale: torch.Tensor, weight: torch.Tensor, name: str) -> torch.Tensor:
    if scale.ndim != 3 or weight.ndim != 3 or scale.shape[:2] != weight.shape[:2]:
        raise ValueError(f"DeepMegaMoE {name} weight and scale shapes do not match: {weight.shape}, {scale.shape}")
    if scale.shape[-1] % 2 or weight.shape[-1] * 2 != scale.shape[-1] * 32:
        raise ValueError(f"DeepMegaMoE {name} requires two E8M0 scales per packed int16 scale")
    scale_bytes = scale.view(torch.uint8).contiguous()
    paired = scale_bytes.reshape(*scale.shape[:-1], scale.shape[-1] // 2, 2)
    return paired.view(torch.int16).squeeze(-1).contiguous()


def prepare_deepmegamoe_weights(layer: torch.nn.Module) -> None:
    """Convert raw MXFP4 experts to DeepGEMM's interleaved layout after loading."""
    if getattr(layer, "_deepmegamoe_weights_ready", False):
        return
    if get_ascend_device_type() != AscendDeviceType.A5:
        raise RuntimeError("DeepMegaMoE weights can only be prepared on A5")

    w1 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    if w1.dtype != torch.uint8 or w2.dtype != torch.uint8:
        raise TypeError(f"DeepMegaMoE expects packed MXFP4 uint8 weights, got {w1.dtype} and {w2.dtype}")
    if not w1.is_contiguous() or not w2.is_contiguous():
        raise ValueError("DeepMegaMoE expects contiguous checkpoint MXFP4 weights")
    w1_int8 = w1.view(torch.int8)
    w2_int8 = w2.view(torch.int8)
    w1_scale = _pack_mxfp4_scales(layer.w13_weight_scale.data, w1, "w13")
    w2_scale = _pack_mxfp4_scales(layer.w2_weight_scale.data, w2, "w2")

    import deep_gemm

    (w1_transformed, w1_scale_transformed), (w2_transformed, w2_scale_transformed) = (
        deep_gemm.transform_weights_for_mega_moe((w1_int8, w1_scale), (w2_int8, w2_scale))
    )
    layer.w13_weight.data = w1_transformed
    layer.w13_weight_scale.data = w1_scale_transformed
    layer.w2_weight.data = w2_transformed
    layer.w2_weight_scale.data = w2_scale_transformed
    layer._deepmegamoe_weights_ready = True
    logger.info_once(
        "Prepared DeepMegaMoE FP4 experts: w13=%s, w2=%s",
        tuple(w1_transformed.shape),
        tuple(w2_transformed.shape),
    )


def _get_symm_buffer(moe_config):
    from vllm.distributed import get_ep_group

    group = get_ep_group().device_group
    if group is None:
        raise RuntimeError("DeepMegaMoE requires an initialized HCCL EP group")
    max_tokens = int(get_ascend_config().deepmegamoe_max_tokens_per_rank)
    key = (
        id(group),
        int(moe_config.num_experts),
        max_tokens,
        int(moe_config.experts_per_token),
        int(moe_config.hidden_dim),
        int(moe_config.intermediate_size_per_partition),
    )
    with _SYMM_BUFFER_LOCK:
        buffer = _SYMM_BUFFERS.get(key)
        if buffer is None:
            import deep_gemm

            buffer = deep_gemm.SymmBuffer(
                group,
                int(moe_config.num_experts),
                max_tokens,
                int(moe_config.experts_per_token),
                int(moe_config.hidden_dim),
                int(moe_config.intermediate_size_per_partition),
                num_shared_experts=0,
            )
            _SYMM_BUFFERS[key] = buffer
            logger.info_once(
                "Created DeepMegaMoE EP scratch buffer: capacity=%d, EP=%d, H=%d, I=%d",
                max_tokens,
                int(moe_config.ep_size),
                int(moe_config.hidden_dim),
                int(moe_config.intermediate_size_per_partition),
            )
    return buffer


def _quantize_mxfp8_activations(hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    x, scale = torch_npu.npu_dynamic_mx_quant(
        hidden_states.contiguous(),
        dst_type=torch.float8_e4m3fn,
        scale_alg=0,
    )
    expected_scale_shape = (hidden_states.shape[0], hidden_states.shape[1] // 32)
    paired_scale_shape = (hidden_states.shape[0], hidden_states.shape[1] // 64, 2)
    if tuple(scale.shape) not in (expected_scale_shape, paired_scale_shape):
        raise ValueError(f"DeepMegaMoE activation scale must have shape {expected_scale_shape}, got {scale.shape}")
    scale_bytes = scale.view(torch.uint8).reshape(expected_scale_shape).contiguous()
    if scale_bytes.shape[1] % 2:
        raise ValueError("DeepMegaMoE requires an even number of E8M0 activation scales per token")
    scale_pairs = scale_bytes.reshape(hidden_states.shape[0], -1, 2).view(torch.int16).squeeze(-1).contiguous()
    return x.contiguous(), scale_pairs


def apply_deepmegamoe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    moe_config,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    if hidden_states.dtype != torch.bfloat16 or hidden_states.ndim != 2:
        raise ValueError("DeepMegaMoE requires a 2-D BF16 hidden-state tensor")
    if hidden_states.shape[1] != int(moe_config.hidden_dim):
        raise ValueError(
            f"DeepMegaMoE hidden size must match the MoE config: "
            f"{hidden_states.shape[1]} != {moe_config.hidden_dim}"
        )
    if not math.isfinite(swiglu_limit) or swiglu_limit < 0:
        raise ValueError(f"swiglu_limit must be finite and nonnegative, got {swiglu_limit}")

    num_tokens = hidden_states.shape[0]
    max_tokens = int(get_ascend_config().deepmegamoe_max_tokens_per_rank)
    if num_tokens > max_tokens:
        raise ValueError(
            "DeepMegaMoE token count exceeds deepmegamoe_max_tokens_per_rank: "
            f"{num_tokens} > {max_tokens}"
        )
    if topk_ids.shape != topk_weights.shape or topk_ids.shape != (
        num_tokens,
        int(moe_config.experts_per_token),
    ):
        raise ValueError("DeepMegaMoE top-k ids and weights must have shape [M, experts_per_token]")
    if any(not isinstance(t, torch.Tensor) for t in (w1, w1_scale, w2, w2_scale)):
        raise TypeError("DeepMegaMoE requires transformed expert weights and scales as tensors")
    if w1.dtype != torch.int8 or w2.dtype != torch.int8 or w1_scale.dtype != torch.int16 or w2_scale.dtype != torch.int16:
        raise TypeError("DeepMegaMoE requires transformed int8 FP4 weights and int16 E8M0 scales")

    if num_tokens == 0:
        x = torch.empty(
            (0, hidden_states.shape[1]), dtype=torch.float8_e4m3fn, device=hidden_states.device
        )
        x_scale = torch.empty((0, hidden_states.shape[1] // 64), dtype=torch.int16, device=hidden_states.device)
    else:
        x, x_scale = _quantize_mxfp8_activations(hidden_states)
    sym_buffer = _get_symm_buffer(moe_config)
    sym_buffer.x[:num_tokens].copy_(x)
    sym_buffer.x_sf[:num_tokens].copy_(x_scale)
    sym_buffer.topk_idx[:num_tokens].copy_(topk_ids.to(torch.int64))
    sym_buffer.topk_weights[:num_tokens].copy_(topk_weights.to(torch.float32))

    y = torch.empty(hidden_states.shape, dtype=torch.bfloat16, device=hidden_states.device)
    import deep_gemm

    with moe_profile_range("deepmegamoe", "kernel_enqueue", hidden_states, topk_ids):
        deep_gemm.fp8_fp4_mega_moe(
            y,
            (w1, w1_scale),
            (w2, w2_scale),
            sym_buffer,
            activation_clamp=float(swiglu_limit) if swiglu_limit > 0 else None,
        )
    return y
