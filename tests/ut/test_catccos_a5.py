# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum
from types import SimpleNamespace
from unittest import TestCase, mock

import torch

from vllm_ascend.catccos_patch import (
    _quantize_weight,
    _should_use_native_path,
    _validate_weight_shapes,
)


class TestCatccosA5(TestCase):
    class Activation(Enum):
        SILU = "silu"

    def test_qwen3_moe_weight_shapes_are_supported(self):
        w1 = torch.empty(8, 1536, 2048, device="meta")
        w2 = torch.empty(8, 2048, 768, device="meta")

        _validate_weight_shapes(w1, w2)

    def test_rejects_mismatched_weight_shapes(self):
        w1 = torch.empty(8, 1536, 2048, device="meta")
        w2 = torch.empty(8, 2048, 512, device="meta")

        with self.assertRaisesRegex(ValueError, "dimensions do not match"):
            _validate_weight_shapes(w1, w2)

    def test_small_batches_and_shared_experts_use_native_path(self):
        layer = SimpleNamespace(
            _shared_experts=None,
            dynamic_eplb=False,
            quant_config=None,
            activation="silu",
        )
        hidden_states = torch.empty(32, 2048)

        with mock.patch.dict(os.environ, {"VLLM_ASCEND_CATCCOS_MINM": "64"}):
            self.assertTrue(_should_use_native_path(layer, hidden_states, False))
            layer._shared_experts = object()
            self.assertTrue(_should_use_native_path(layer, torch.empty(64, 2048), False))

    def test_rejects_dynamic_eplb(self):
        layer = SimpleNamespace(
            _shared_experts=None,
            dynamic_eplb=True,
            quant_config=None,
            activation="silu",
        )

        with self.assertRaisesRegex(ValueError, "dynamic EPLB"):
            _should_use_native_path(layer, torch.empty(64, 2048), False)

    def test_accepts_silu_enum(self):
        layer = SimpleNamespace(
            _shared_experts=None,
            dynamic_eplb=False,
            quant_config=None,
            activation=self.Activation.SILU,
        )

        self.assertFalse(_should_use_native_path(layer, torch.empty(64, 2048), False))

    def test_rejects_unknown_weight_quantization_backend(self):
        with (
            mock.patch.dict(
                os.environ,
                {"VLLM_ASCEND_CATCCOS_WEIGHT_QUANT_BACKEND": "unknown"},
            ),
            self.assertRaisesRegex(ValueError, "must be 'npu' or 'cpu'"),
        ):
            _quantize_weight(torch.empty(1))
