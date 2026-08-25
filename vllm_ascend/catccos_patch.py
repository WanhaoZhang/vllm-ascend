# SPDX-License-Identifier: Apache-2.0
"""Experimental CatCCOS A5 MegaMoE integration for vLLM-Ascend 0.23."""

from __future__ import annotations

import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend import envs as envs_ascend

_INITIALIZED_GROUP: tuple[int, int] | None = None
_INITIALIZE_LOCK = threading.Lock()
_ORIGINAL_FORWARD = None


@lru_cache(maxsize=1)
def _load_library(path: str) -> None:
    library = Path(path)
    if not library.is_file():
        raise FileNotFoundError(f"CatCCOS extension does not exist: {library}")
    torch.ops.load_library(str(library))
    logger.info("Loaded CatCCOS A5 extension from %s", library)


def _get_ep_rank_and_world_size() -> tuple[int, int, Any | None]:
    try:
        from vllm.distributed import get_ep_group

        group = get_ep_group()
        return group.rank_in_group, group.world_size, group
    except (AssertionError, RuntimeError):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return (
                torch.distributed.get_rank(),
                torch.distributed.get_world_size(),
                None,
            )
        return 0, 1, None


def _ensure_initialized() -> None:
    global _INITIALIZED_GROUP
    if _INITIALIZED_GROUP is not None:
        return
    with _INITIALIZE_LOCK:
        if _INITIALIZED_GROUP is not None:
            return
        _load_library(envs_ascend.VLLM_ASCEND_CATCCOS_LIBRARY_PATH)
        rank, world_size, group = _get_ep_rank_and_world_size()
        if group is not None and world_size > 1:
            group.barrier()
        status = torch.ops.catccos.init(
            rank,
            world_size,
            envs_ascend.VLLM_ASCEND_CATCCOS_MEM,
            envs_ascend.VLLM_ASCEND_CATCCOS_IPPORT,
        )
        if status != 0:
            raise RuntimeError(f"CatCCOS initialization failed with status {status} for EP rank {rank}/{world_size}")
        _INITIALIZED_GROUP = (rank, world_size)
        logger.info(
            "Initialized CatCCOS A5 backend: EP rank=%d/%d, store=%s, symmetric memory=%d bytes",
            rank,
            world_size,
            envs_ascend.VLLM_ASCEND_CATCCOS_IPPORT,
            envs_ascend.VLLM_ASCEND_CATCCOS_MEM,
        )


@lru_cache(maxsize=1)
def _get_cpu_quantizer(utils_path: str):
    path = Path(utils_path)
    module_path = path / "gen_mx_quant_allgather_data.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"CatCCOS MXFP8 quantizer does not exist: {module_path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from gen_mx_quant_allgather_data import _quantize_fp8

    return _quantize_fp8


def _quantize_cpu(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quantize_fp8 = _get_cpu_quantizer(envs_ascend.VLLM_ASCEND_CATCCOS_UTILS_PATH)
    device = weight.device
    original_shape = weight.shape
    matrix = weight.reshape(-1, original_shape[-1]).float().cpu()
    quantized, scale, _ = quantize_fp8(matrix, "E4M3", axis=1)
    scale_shape = (*original_shape[:-1], scale.shape[-1])
    return (
        quantized.reshape(original_shape).to(device),
        scale.reshape(scale_shape).to(torch.float8_e8m0fnu).to(device),
    )


def _quantize_npu(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    return quantized.reshape(original_shape).contiguous(), scale.reshape(scale_shape).contiguous()


def _quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    backend = envs_ascend.VLLM_ASCEND_CATCCOS_WEIGHT_QUANT_BACKEND
    if backend == "npu":
        return _quantize_npu(weight)
    if backend == "cpu":
        return _quantize_cpu(weight)
    raise ValueError(f"VLLM_ASCEND_CATCCOS_WEIGHT_QUANT_BACKEND must be 'npu' or 'cpu', got {backend!r}")


def _validate_weight_shapes(w1: torch.Tensor, w2: torch.Tensor) -> None:
    if w1.ndim != 3 or w2.ndim != 3:
        raise ValueError(f"CatCCOS requires 3-D weights, got {w1.shape} and {w2.shape}")
    experts, merged_intermediate, hidden_size = w1.shape
    if w2.shape[0] != experts or w2.shape[1] != hidden_size:
        raise ValueError(f"CatCCOS weight shapes do not match: w1={w1.shape}, w2={w2.shape}")
    if merged_intermediate != 2 * w2.shape[2]:
        raise ValueError(f"CatCCOS gate/up and down-projection dimensions do not match: w1={w1.shape}, w2={w2.shape}")
    if hidden_size % 256:
        raise ValueError(f"CatCCOS requires hidden size divisible by 256, got {hidden_size}")
    if merged_intermediate % 512:
        raise ValueError(f"CatCCOS requires merged gate/up size divisible by 512, got {merged_intermediate}")


def _get_or_build_weight_cache(layer) -> dict[str, torch.Tensor]:
    cache = getattr(layer, "_catccos_a5_weight_cache", None)
    if cache is not None:
        return cache

    import torch_npu

    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND

    w13 = torch_npu.npu_format_cast(layer.w13_weight.data, ACL_FORMAT_FRACTAL_ND)
    w2_native = torch_npu.npu_format_cast(layer.w2_weight.data, ACL_FORMAT_FRACTAL_ND)
    w1 = w13.transpose(1, 2).contiguous()
    w2 = w2_native.transpose(1, 2).contiguous()
    _validate_weight_shapes(w1, w2)

    w1_quantized, w1_scale = _quantize_weight(w1)
    w2_quantized, w2_scale = _quantize_weight(w2)
    cache = {
        "w1": w1_quantized,
        "w1_scale": w1_scale,
        "w2": w2_quantized,
        "w2_scale": w2_scale,
    }
    layer._catccos_a5_weight_cache = cache
    logger.info_once(
        "Converted CatCCOS A5 expert weights to MXFP8: w1=%s, w2=%s, backend=%s",
        tuple(w1_quantized.shape),
        tuple(w2_quantized.shape),
        envs_ascend.VLLM_ASCEND_CATCCOS_WEIGHT_QUANT_BACKEND,
    )
    return cache


def _should_use_native_path(layer, hidden_states: torch.Tensor, return_with_event: bool) -> bool:
    token_count = hidden_states.numel() // hidden_states.shape[-1]
    if token_count < envs_ascend.VLLM_ASCEND_CATCCOS_MINM:
        return True
    if return_with_event or getattr(layer, "_shared_experts", None) is not None:
        logger.warning_once("CatCCOS A5 does not yet support the shared-expert event path; using native MoE")
        return True
    if getattr(layer, "dynamic_eplb", False):
        raise ValueError("CatCCOS A5 does not support dynamic EPLB")
    if getattr(layer, "quant_config", None) is not None:
        raise ValueError("CatCCOS A5 currently requires an unquantized BF16 model")
    activation = getattr(layer, "activation", "silu")
    if getattr(activation, "value", activation) != "silu":
        raise ValueError(f"CatCCOS A5 requires SiLU activation, got {activation}")
    return False


def _catccos_forward_impl(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    return_with_event: bool = False,
):
    if _should_use_native_path(self, hidden_states, return_with_event):
        assert _ORIGINAL_FORWARD is not None
        return _ORIGINAL_FORWARD(self, hidden_states, router_logits, return_with_event)

    from vllm.forward_context import get_forward_context

    from vllm_ascend.ops.fused_moe.experts_selector import select_experts

    _ensure_initialized()
    cache = _get_or_build_weight_cache(self)
    input_ids = getattr(get_forward_context(), "input_ids", None)
    topk_weights, topk_ids = select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=self.top_k,
        use_grouped_topk=self.use_grouped_topk,
        renormalize=self.renormalize,
        topk_group=self.topk_group,
        num_expert_group=self.num_expert_group,
        custom_routing_function=self.custom_routing_function,
        scoring_func=self.scoring_func,
        routed_scaling_factor=self._original_routed_scaling_factor,
        e_score_correction_bias=self.e_score_correction_bias,
        num_experts=self.moe_config.num_experts,
        input_ids=input_ids,
        tid2eid=self.tid2eid,
    )

    original_shape = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, original_shape[-1]).contiguous()
    output = torch.ops.catccos.ascend950_dispatch_ffn_combine(
        hidden_states,
        topk_ids.to(torch.int32).contiguous(),
        topk_weights.to(torch.float32).contiguous(),
        cache["w1"],
        cache["w1_scale"],
        cache["w2"],
        cache["w2_scale"],
    )
    if envs_ascend.VLLM_ASCEND_CATCCOS_SYNC_DEVICE:
        torch.npu.synchronize()
    return output.reshape(original_shape)


def apply_catccos_patch() -> None:
    """Patch the vLLM-Ascend 0.23 routed MoE implementation."""
    global _ORIGINAL_FORWARD
    if not envs_ascend.VLLM_ASCEND_CATCCOS:
        return
    try:
        from vllm_ascend.ops.fused_moe.fused_moe_0_23_0 import AscendFusedMoE
    except ImportError as exc:
        raise RuntimeError("CatCCOS A5 integration currently requires vLLM-Ascend 0.23.0") from exc
    if getattr(AscendFusedMoE.forward_impl, "_catccos_a5_patched", False):
        return
    _ORIGINAL_FORWARD = AscendFusedMoE.forward_impl
    _catccos_forward_impl._catccos_a5_patched = True
    AscendFusedMoE.forward_impl = _catccos_forward_impl
    logger.info("Enabled CatCCOS A5 fused dispatch-FFN-combine backend")
