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

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.ops.fused_moe.catccos_adapter import (
    CatccosLayerCapability,
    evaluate_catccos_layer,
    get_model_catccos_capability,
    register_catccos_capability,
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
        "experts_per_token": 8,
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
def test_rejects_non_divisible_expert_layout(_mock_enabled):
    capability = evaluate_catccos_layer(
        _moe_config(num_experts=127),
        _quant_method(),
        "silu",
        device_type=AscendDeviceType.A5,
        library_exists=True,
    )

    assert not capability.supported
    assert "divisible by EP size" in capability.reason


@patch(
    "vllm_ascend.ops.fused_moe.catccos_adapter.catccos_backend_enabled",
    return_value=True,
)
def test_rejects_quantized_checkpoint_and_shared_experts(_mock_enabled):
    quantized = evaluate_catccos_layer(
        _moe_config(),
        _quant_method(quant_type=QuantType.W8A8MXFP),
        "silu",
        device_type=AscendDeviceType.A5,
        library_exists=True,
    )
    shared = evaluate_catccos_layer(
        _moe_config(),
        _quant_method(),
        "silu",
        n_shared_experts=1,
        device_type=AscendDeviceType.A5,
        library_exists=True,
    )

    assert not quantized.supported
    assert not shared.supported


def test_model_capability_fails_closed_when_any_layer_is_unsupported():
    supported_layer = torch.nn.Module()
    unsupported_layer = torch.nn.Module()
    register_catccos_capability(supported_layer, CatccosLayerCapability(True))
    register_catccos_capability(
        unsupported_layer,
        CatccosLayerCapability(False, "unsupported layer"),
    )
    model = torch.nn.Sequential(supported_layer, unsupported_layer)

    capability = get_model_catccos_capability(model)

    assert not capability.supported
    assert capability.reason == "unsupported layer"


def test_qwen3_operator_weight_shapes_are_supported():
    w1 = torch.empty(32, 1536, 2048, device="meta")
    w2 = torch.empty(32, 2048, 768, device="meta")

    validate_catccos_weight_shapes(w1, w2)


def test_rejects_mismatched_operator_weight_shapes():
    w1 = torch.empty(32, 1536, 2048, device="meta")
    w2 = torch.empty(32, 2048, 512, device="meta")

    try:
        validate_catccos_weight_shapes(w1, w2)
    except ValueError as exc:
        assert "dimensions do not match" in str(exc)
    else:
        raise AssertionError("mismatched weights must be rejected")
