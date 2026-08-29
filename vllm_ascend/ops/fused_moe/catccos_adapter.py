# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Capability checks and runtime adapter for the CatCCOS A5 operator."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

_INITIALIZED_GROUP: tuple[int, int] | None = None
_INITIALIZE_LOCK = threading.Lock()


@dataclass(frozen=True)
class CatccosLayerCapability:
    supported: bool
    reason: str = ""


def register_catccos_capability(layer: torch.nn.Module, capability: CatccosLayerCapability) -> None:
    layer.catccos_fused_mc2_capability = capability


def catccos_backend_enabled() -> bool:
    config = get_ascend_config()
    return config.enable_fused_mc2 == 1 and getattr(config, "fused_mc2_backend", "auto") == "catccos"


def _unsupported(reason: str) -> CatccosLayerCapability:
    return CatccosLayerCapability(False, reason)


def evaluate_catccos_layer(
    moe_config,
    quant_method: object,
    activation: Any,
    *,
    n_shared_experts: int = 0,
    device_type: AscendDeviceType | None = None,
    library_exists: bool | None = None,
) -> CatccosLayerCapability:
    """Validate the v0.23 legacy MoE layer against the CatCCOS contract."""
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
    if quant_type != QuantType.NONE or getattr(quant_method, "quant_method", None) is not None:
        return _unsupported("CatCCOS requires an unquantized checkpoint")
    if getattr(quant_method, "dynamic_eplb", False):
        return _unsupported("CatCCOS does not support dynamic EPLB")
    if n_shared_experts:
        return _unsupported("CatCCOS does not support shared experts")
    activation_name = str(getattr(activation, "value", activation)).lower()
    if activation_name not in {"silu", "swiglu", "moeactivation.silu"}:
        return _unsupported(f"CatCCOS requires SiLU, got {activation}")
    if getattr(moe_config, "in_dtype", torch.bfloat16) != torch.bfloat16:
        return _unsupported(f"CatCCOS requires BF16 input, got {moe_config.in_dtype}")
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
    return CatccosLayerCapability(True)


@lru_cache(maxsize=1)
def load_catccos_library(path: str) -> None:
    library = Path(path)
    if not library.is_file():
        raise FileNotFoundError(f"CatCCOS extension does not exist: {library}")
    torch.ops.load_library(str(library))
    logger.info("Loaded CatCCOS A5 extension from %s", library)


def initialize_catccos() -> None:
    """Initialize CatCCOS once for the vLLM expert-parallel group."""
    global _INITIALIZED_GROUP
    from vllm.distributed import get_ep_group

    group = get_ep_group()
    rank, world_size = group.rank_in_group, group.world_size
    group_key = (rank, world_size)
    if group_key == _INITIALIZED_GROUP:
        return
    if _INITIALIZED_GROUP is not None:
        raise RuntimeError(f"CatCCOS already initialized for {_INITIALIZED_GROUP}, got {group_key}")
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
            raise RuntimeError(f"CatCCOS initialization failed for EP rank {rank}/{world_size}: {status}")
        _INITIALIZED_GROUP = group_key
        logger.info("Initialized CatCCOS A5 backend: EP rank=%d/%d", rank, world_size)


def _quantize_mxfp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    original_shape = weight.shape
    matrix = weight.reshape(-1, original_shape[-1]).contiguous()
    quantized, scale = torch_npu.npu_dynamic_mx_quant(matrix, dst_type=torch.float8_e4m3fn, scale_alg=0)
    scale = scale.reshape(matrix.shape[0], -1).view(torch.float8_e8m0fnu)
    return (
        quantized.reshape(original_shape).contiguous(),
        scale.reshape((*original_shape[:-1], scale.shape[-1])).contiguous(),
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
    """Quantize v0.23 expert weights once after checkpoint loading."""
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
    logger.info_once("Converted CatCCOS expert weights to MXFP8: w1=%s, w2=%s", tuple(w1.shape), tuple(w2.shape))


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
    # CatCCOS launches CCEC directly, so flush torch_npu producers first.
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
