# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Capability and runtime adapter for the external CatCCOS A5 operator."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import FusedMoEConfig


_LAYER_CAPABILITY_ATTR = "catccos_fused_mc2_capability"
_MODEL_CAPABILITY_ATTR = "_ascend_catccos_fused_mc2_capability"
_INITIALIZED_GROUP: tuple[int, int] | None = None
_INITIALIZE_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class CatccosLayerCapability:
    supported: bool
    reason: str = ""


def catccos_backend_enabled() -> bool:
    config = get_ascend_config()
    return config.enable_fused_mc2 == 1 and config.fused_mc2_backend == "catccos"


def _unsupported(reason: str) -> CatccosLayerCapability:
    return CatccosLayerCapability(False, reason)


def evaluate_catccos_layer(
    moe_config: FusedMoEConfig,
    quant_method: object,
    activation: Any,
    *,
    n_shared_experts: int = 0,
    device_type: AscendDeviceType | None = None,
    library_exists: bool | None = None,
) -> CatccosLayerCapability:
    """Check an instantiated routed-expert layer against the kernel contract."""
    if not catccos_backend_enabled():
        return _unsupported("CatCCOS is not selected")

    device_type = device_type or get_ascend_device_type()
    if device_type != AscendDeviceType.A5:
        return _unsupported(f"CatCCOS requires A5, got {device_type}")

    if library_exists is None:
        library_path = Path(get_ascend_config().catccos_library_path)
        library_exists = library_path.is_file()
    else:
        library_path = Path("<provided by caller>")
    if not library_exists:
        return _unsupported(f"CatCCOS extension does not exist: {library_path}")

    quant_type = getattr(quant_method, "quant_type", QuantType.NONE)
    if quant_type != QuantType.NONE:
        return _unsupported(
            "CatCCOS currently accepts an unquantized checkpoint and converts "
            f"its expert weights to MXFP8, got {quant_type}"
        )
    if getattr(quant_method, "dynamic_eplb", False):
        return _unsupported("CatCCOS does not support dynamic EPLB")
    if n_shared_experts:
        return _unsupported("CatCCOS does not support shared experts")

    activation_name = str(getattr(activation, "value", activation)).lower()
    if activation_name not in {"silu", "swiglu", "moeactivation.silu"}:
        return _unsupported(f"CatCCOS requires SiLU, got {activation}")

    in_dtype = getattr(moe_config, "in_dtype", torch.bfloat16)
    if in_dtype != torch.bfloat16:
        return _unsupported(f"CatCCOS requires BF16 input, got {in_dtype}")

    hidden = int(moe_config.hidden_dim)
    merged_intermediate = 2 * int(moe_config.intermediate_size_per_partition)
    if hidden <= 0 or hidden % 256:
        return _unsupported(f"hidden size must be divisible by 256, got {hidden}")
    if merged_intermediate <= 0 or merged_intermediate % 512:
        return _unsupported(f"merged intermediate size must be divisible by 512, got {merged_intermediate}")

    ep_size = int(moe_config.ep_size)
    num_experts = int(moe_config.num_experts)
    if ep_size < 2:
        return _unsupported("CatCCOS requires expert parallelism")
    if num_experts % ep_size:
        return _unsupported(f"experts must be divisible by EP size, got {num_experts}/{ep_size}")
    if int(moe_config.experts_per_token) <= 0:
        return _unsupported("top-k must be positive")
    return CatccosLayerCapability(True)


def get_model_catccos_capability(
    model_instance: torch.nn.Module | None,
) -> CatccosLayerCapability:
    """Aggregate immutable CatCCOS capabilities registered on MoE layers."""
    if model_instance is None:
        return _unsupported("model instance is unavailable")
    cached = getattr(model_instance, _MODEL_CAPABILITY_ATTR, None)
    if isinstance(cached, CatccosLayerCapability):
        return cached

    capabilities = [
        capability
        for module in model_instance.modules()
        if isinstance(
            (capability := getattr(module, _LAYER_CAPABILITY_ATTR, None)),
            CatccosLayerCapability,
        )
    ]
    if not capabilities:
        return _unsupported("model has no registered CatCCOS layers")
    result = next(
        (capability for capability in capabilities if not capability.supported),
        capabilities[0],
    )
    setattr(model_instance, _MODEL_CAPABILITY_ATTR, result)
    return result


def register_catccos_capability(layer: torch.nn.Module, capability: CatccosLayerCapability) -> None:
    setattr(layer, _LAYER_CAPABILITY_ATTR, capability)


@lru_cache(maxsize=1)
def load_catccos_library(path: str) -> None:
    library = Path(path)
    if not library.is_file():
        raise FileNotFoundError(f"CatCCOS extension does not exist: {library}")
    torch.ops.load_library(str(library))
    logger.info("Loaded CatCCOS A5 extension from %s", library)


def initialize_catccos() -> None:
    """Initialize the process-local CatCCOS runtime for the exact EP group."""
    global _INITIALIZED_GROUP
    from vllm.distributed import get_ep_group

    group = get_ep_group()
    rank = group.rank_in_group
    world_size = group.world_size
    group_key = (rank, world_size)
    if group_key == _INITIALIZED_GROUP:
        return
    if _INITIALIZED_GROUP is not None:
        raise RuntimeError(
            f"CatCCOS is already initialized for a different EP group: {_INITIALIZED_GROUP} != {group_key}"
        )

    with _INITIALIZE_LOCK:
        if group_key == _INITIALIZED_GROUP:
            return
        config = get_ascend_config()
        load_catccos_library(config.catccos_library_path)
        group.barrier()
        status = torch.ops.catccos.init(
            rank,
            world_size,
            config.catccos_local_mem_size,
            config.catccos_store_url,
        )
        if status != 0:
            raise RuntimeError(f"CatCCOS initialization failed with status {status} for EP rank {rank}/{world_size}")
        _INITIALIZED_GROUP = group_key
        logger.info(
            "Initialized CatCCOS fused MC2 backend: EP rank=%d/%d",
            rank,
            world_size,
        )


def _quantize_mxfp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    original_shape = weight.shape
    matrix = weight.reshape(-1, original_shape[-1]).contiguous()
    quantized, scale = torch_npu.npu_dynamic_mx_quant(
        matrix,
        dst_type=torch.float8_e4m3fn,
        scale_alg=0,
    )
    scale = scale.reshape(matrix.shape[0], -1).view(torch.float8_e8m0fnu)
    scale_shape = (*original_shape[:-1], scale.shape[-1])
    return (
        quantized.reshape(original_shape).contiguous(),
        scale.reshape(scale_shape).contiguous(),
    )


def validate_catccos_weight_shapes(w1: torch.Tensor, w2: torch.Tensor) -> None:
    if w1.ndim != 3 or w2.ndim != 3:
        raise ValueError(f"CatCCOS requires 3-D weights, got {w1.shape} and {w2.shape}")
    experts, merged_intermediate, hidden = w1.shape
    if w2.shape[:2] != (experts, hidden):
        raise ValueError(f"CatCCOS weight shapes do not match: w1={w1.shape}, w2={w2.shape}")
    if merged_intermediate != 2 * w2.shape[2]:
        raise ValueError(f"CatCCOS gate/up and down-projection dimensions do not match: w1={w1.shape}, w2={w2.shape}")
    if hidden % 256 or merged_intermediate % 512:
        raise ValueError("CatCCOS requires hidden % 256 == 0 and merged intermediate % 512 == 0")


def _set_runtime_buffer(layer: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    if name in layer._buffers:
        layer._buffers[name] = value
    else:
        layer.register_buffer(name, value, persistent=False)


def prepare_catccos_weights(layer: torch.nn.Module) -> None:
    """Build persistent CatCCOS MXFP8 weights during model weight processing."""
    if getattr(layer, "_catccos_weights_ready", False):
        return
    w1 = layer.w13_weight.data.transpose(1, 2).contiguous()
    w2 = layer.w2_weight.data.transpose(1, 2).contiguous()
    validate_catccos_weight_shapes(w1, w2)
    w1_quantized, w1_scale = _quantize_mxfp8(w1)
    w2_quantized, w2_scale = _quantize_mxfp8(w2)
    _set_runtime_buffer(layer, "catccos_w1", w1_quantized)
    _set_runtime_buffer(layer, "catccos_w1_scale", w1_scale)
    _set_runtime_buffer(layer, "catccos_w2", w2_quantized)
    _set_runtime_buffer(layer, "catccos_w2_scale", w2_scale)
    layer._catccos_weights_ready = True


def apply_catccos(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    initialize_catccos()
    # The current CatCCOS binding launches CCEC directly. Flush torch_npu's
    # queued producers before entering that launcher; this can be removed once
    # the binding participates in torch_npu's dependency scheduler.
    torch.npu.synchronize()
    output = torch.ops.catccos.ascend950_dispatch_ffn_combine(
        hidden_states.contiguous(),
        topk_ids.to(torch.int32).contiguous(),
        topk_weights.to(torch.float32).contiguous(),
        w1,
        w1_scale,
        w2,
        w2_scale,
    )
    if get_ascend_config().catccos_sync_after_launch:
        torch.npu.synchronize()
    return output
