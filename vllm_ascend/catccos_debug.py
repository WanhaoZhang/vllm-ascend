# SPDX-License-Identifier: Apache-2.0
"""Correctness probes for the experimental CatCCOS A5 backend."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re
import torch

_VALID_ORDERS = {
    "native-catccos",
    "catccos-native",
    "native-native",
    "catccos-catccos",
}


@dataclass(frozen=True)
class CatccosProbeConfig:
    """Validated configuration for one process' CatCCOS probe."""

    output_dir: Path
    token_counts: frozenset[int]
    order: str
    max_calls_per_layer: int
    cosine_threshold: float
    relative_l2_threshold: float
    dump_tensors: bool
    dump_weights: bool

    def selects(self, token_count: int) -> bool:
        return token_count in self.token_counts


def _parse_token_counts(value: str) -> frozenset[int]:
    try:
        counts = frozenset(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("CatCCOS probe token counts must be comma-separated integers") from exc
    if not counts or any(count < 1 for count in counts):
        raise ValueError("CatCCOS probe token counts must contain positive integers")
    return counts


def load_probe_config() -> CatccosProbeConfig:
    """Load and validate the opt-in probe environment."""
    from vllm_ascend import envs as envs_ascend

    output_dir = envs_ascend.VLLM_ASCEND_CATCCOS_DEBUG_DIR
    if not output_dir:
        raise ValueError("VLLM_ASCEND_CATCCOS_DEBUG_DIR must be set when the probe is enabled")
    order = envs_ascend.VLLM_ASCEND_CATCCOS_DEBUG_ORDER
    if order not in _VALID_ORDERS:
        choices = ", ".join(sorted(_VALID_ORDERS))
        raise ValueError(f"CatCCOS probe order must be one of: {choices}")
    max_calls = envs_ascend.VLLM_ASCEND_CATCCOS_DEBUG_MAX_CALLS_PER_LAYER
    if max_calls < 1:
        raise ValueError("CatCCOS probe max calls per layer must be positive")
    cosine_threshold = envs_ascend.VLLM_ASCEND_CATCCOS_DEBUG_COSINE_THRESHOLD
    if not 0.0 <= cosine_threshold <= 1.0:
        raise ValueError("CatCCOS probe cosine threshold must be between 0 and 1")
    relative_l2_threshold = envs_ascend.VLLM_ASCEND_CATCCOS_DEBUG_RELATIVE_L2_THRESHOLD
    if relative_l2_threshold <= 0.0:
        raise ValueError("CatCCOS probe relative L2 threshold must be positive")
    return CatccosProbeConfig(
        output_dir=Path(output_dir),
        token_counts=_parse_token_counts(envs_ascend.VLLM_ASCEND_CATCCOS_DEBUG_TOKEN_COUNTS),
        order=order,
        max_calls_per_layer=max_calls,
        cosine_threshold=cosine_threshold,
        relative_l2_threshold=relative_l2_threshold,
        dump_tensors=envs_ascend.VLLM_ASCEND_CATCCOS_DEBUG_DUMP_TENSORS,
        dump_weights=envs_ascend.VLLM_ASCEND_CATCCOS_DEBUG_DUMP_WEIGHTS,
    )


def _cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().cpu()


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    """Return a bitwise SHA-256 fingerprint without dtype conversion."""
    byte_view = _cpu_tensor(tensor).view(torch.uint8).numpy()
    return hashlib.sha256(byte_view.tobytes()).hexdigest()


def tensor_layout(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "contiguous": tensor.is_contiguous(),
    }


def tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    return {**tensor_layout(tensor), "sha256": tensor_fingerprint(tensor)}


def route_metadata(expert_idx: torch.Tensor, gate_weight: torch.Tensor) -> dict[str, Any]:
    """Return compact routing values useful for rank-mapping diagnosis."""
    expert_idx_cpu = _cpu_tensor(expert_idx).long().reshape(-1)
    gate_weight_cpu = _cpu_tensor(gate_weight).float()
    unique_experts, counts = torch.unique(expert_idx_cpu, return_counts=True)
    return {
        "expert_min": int(expert_idx_cpu.min()),
        "expert_max": int(expert_idx_cpu.max()),
        "expert_token_counts": {
            str(expert): int(count) for expert, count in zip(unique_experts.tolist(), counts.tolist())
        },
        "gate_weight_min": float(gate_weight_cpu.min()),
        "gate_weight_max": float(gate_weight_cpu.max()),
        "gate_weight_row_sum_min": float(gate_weight_cpu.sum(dim=-1).min()),
        "gate_weight_row_sum_max": float(gate_weight_cpu.sum(dim=-1).max()),
    }


def compare_tensors(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    """Compare frozen outputs on CPU and return JSON-serializable metrics."""
    if reference.shape != candidate.shape:
        return {
            "shape_equal": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }

    reference_cpu = _cpu_tensor(reference)
    candidate_cpu = _cpu_tensor(candidate)
    result: dict[str, Any] = {
        "shape_equal": True,
        "exact_equal": torch.equal(reference_cpu, candidate_cpu),
        "reference_sha256": tensor_fingerprint(reference_cpu),
        "candidate_sha256": tensor_fingerprint(candidate_cpu),
    }
    reference_float = reference_cpu.float().reshape(-1)
    candidate_float = candidate_cpu.float().reshape(-1)
    finite = bool(torch.isfinite(reference_float).all().item() and torch.isfinite(candidate_float).all().item())
    result["finite"] = finite
    if not finite:
        return result

    difference = (reference_float - candidate_float).abs()
    reference_norm = torch.linalg.vector_norm(reference_float)
    candidate_norm = torch.linalg.vector_norm(candidate_float)
    denominator = reference_norm * candidate_norm
    if denominator.item() == 0:
        cosine = 1.0 if reference_norm.item() == candidate_norm.item() else 0.0
    else:
        cosine = torch.dot(reference_float, candidate_float) / denominator
    nonzero = (reference_float != 0) & (candidate_float != 0)
    sign_flip = ((reference_float < 0) != (candidate_float < 0)) & nonzero
    result.update(
        {
            "cosine_similarity": float(cosine),
            "max_abs_diff": float(difference.max()) if difference.numel() else 0.0,
            "mean_abs_diff": float(difference.mean()) if difference.numel() else 0.0,
            "relative_l2": float(torch.linalg.vector_norm(difference) / (reference_norm + 1e-12)),
            "norm_ratio": float(candidate_norm / (reference_norm + 1e-12)),
            "sign_flip_ratio": float(sign_flip.float().mean()) if sign_flip.numel() else 0.0,
        }
    )
    return result


def is_significant_mismatch(
    metrics: dict[str, Any],
    cosine_threshold: float,
    relative_l2_threshold: float,
) -> bool:
    if not metrics.get("shape_equal", False) or not metrics.get("finite", False):
        return True
    cosine = metrics.get("cosine_similarity")
    relative_l2 = metrics.get("relative_l2")
    return (
        cosine is None
        or not math.isfinite(cosine)
        or cosine < cosine_threshold
        or relative_l2 is None
        or not math.isfinite(relative_l2)
        or relative_l2 > relative_l2_threshold
    )


def _safe_layer_name(layer: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", layer)


def write_probe_result(
    config: CatccosProbeConfig,
    rank: int,
    record: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor],
) -> Path | None:
    """Append one summary and optionally dump the first mismatch per rank."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / f"probe-rank{rank:03d}.jsonl"
    with summary_path.open("a", encoding="utf-8") as summary:
        summary.write(json.dumps(record, sort_keys=True) + "\n")

    if not config.dump_tensors or not record["significant_mismatch"]:
        return None
    marker_path = config.output_dir / f"first-mismatch-rank{rank:03d}.json"
    if marker_path.exists():
        return None

    layer = _safe_layer_name(record["layer"])
    dump_path = config.output_dir / f"first-mismatch-rank{rank:03d}-{layer}.pt"
    payload: dict[str, Any] = {
        "metadata": record,
        "tensors": {name: _cpu_tensor(tensor) for name, tensor in tensors.items()},
    }
    if config.dump_weights:
        payload["weights"] = {name: _cpu_tensor(weight) for name, weight in weights.items()}
    torch.save(payload, dump_path)
    marker_path.write_text(
        json.dumps({"dump": dump_path.name, "record": record}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dump_path
