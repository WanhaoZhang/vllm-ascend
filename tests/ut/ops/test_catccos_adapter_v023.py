# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

import vllm_ascend.ops.fused_moe.catccos_adapter as catccos_adapter
from vllm_ascend.ops.fused_moe.catccos_adapter import (
    CatccosLayerCapability,
    _dump_catccos_inputs,
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


def test_dump_catccos_inputs_writes_replay_files_once(tmp_path, monkeypatch):
    trigger_path = tmp_path / "capture.trigger"
    trigger_path.touch()
    monkeypatch.setattr(catccos_adapter, "_CATCCOS_DUMP_DIR", str(tmp_path))
    monkeypatch.setattr(catccos_adapter, "_CATCCOS_DUMP_TRIGGER", str(trigger_path))
    monkeypatch.setattr(catccos_adapter, "_INITIALIZED_GROUP", (1, 4))

    tensors = (
        torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        torch.full((2, 2), 0.5, dtype=torch.float32),
        torch.empty((1, 2, 4), dtype=torch.float8_e4m3fn),
        torch.empty((1, 2, 2), dtype=torch.float8_e8m0fnu),
        torch.empty((1, 4, 1), dtype=torch.float8_e4m3fn),
        torch.empty((1, 4, 2), dtype=torch.float8_e8m0fnu),
    )

    _dump_catccos_inputs(*tensors)

    manifest_path = tmp_path / "manifest_rank_1.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["ep_rank"] == 1
    assert manifest["ep_world_size"] == 4
    assert manifest["tensors"]["x"]["shape"] == [2, 4]
    for argument_name, file_name in manifest["files"].items():
        tensor = tensors[[name for name, _ in catccos_adapter._CATCCOS_DUMP_FILES].index(argument_name)]
        assert (tmp_path / file_name).stat().st_size == tensor.numel() * tensor.element_size()

    x_path = tmp_path / manifest["files"]["x"]
    original_bytes = x_path.read_bytes()
    tensors[0].fill_(0)
    _dump_catccos_inputs(*tensors)
    assert x_path.read_bytes() == original_bytes


def test_dump_catccos_inputs_waits_for_trigger(tmp_path, monkeypatch):
    monkeypatch.setattr(catccos_adapter, "_CATCCOS_DUMP_DIR", str(tmp_path))
    monkeypatch.setattr(catccos_adapter, "_CATCCOS_DUMP_TRIGGER", str(tmp_path / "missing.trigger"))
    monkeypatch.setattr(catccos_adapter, "_INITIALIZED_GROUP", (0, 4))
    tensors = (torch.empty(1),) * 7

    _dump_catccos_inputs(*tensors)

    assert not list(tmp_path.glob("*.bin"))
