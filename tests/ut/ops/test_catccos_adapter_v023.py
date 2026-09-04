# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.ops.fused_moe.catccos_adapter import (
    CatccosLayerCapability,
    evaluate_catccos_layer,
    validate_catccos_weight_shapes,
)
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType


def _moe_config(**overrides):
    values = {
        "in_dtype": torch.bfloat16,
        "hidden_dim": 2048,
        "intermediate_size_per_partition": 768,
        "ep_size": 4,
        "num_experts": 128,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _quant_method(**overrides):
    values = {"quant_type": QuantType.NONE, "dynamic_eplb": False}
    values.update(overrides)
    return SimpleNamespace(**values)


@patch(
    "vllm_ascend.ops.fused_moe.catccos_adapter.catccos_backend_enabled",
    return_value=True,
)
def test_qwen3_shape_is_supported(_mock_enabled):
    capability = evaluate_catccos_layer(
        _moe_config(),
        _quant_method(),
        "silu",
        device_type=AscendDeviceType.A5,
        library_exists=True,
    )
    assert capability == CatccosLayerCapability(True)


@patch(
    "vllm_ascend.ops.fused_moe.catccos_adapter.catccos_backend_enabled",
    return_value=True,
)
def test_unsupported_layer_fails_closed(_mock_enabled):
    capability = evaluate_catccos_layer(
        _moe_config(num_experts=127),
        _quant_method(),
        "silu",
        device_type=AscendDeviceType.A5,
        library_exists=True,
    )
    assert not capability.supported
    assert "divisible by EP size" in capability.reason


def test_qwen3_operator_weight_shapes_are_supported():
    validate_catccos_weight_shapes(
        torch.empty(32, 1536, 2048, device="meta"),
        torch.empty(32, 2048, 768, device="meta"),
    )
