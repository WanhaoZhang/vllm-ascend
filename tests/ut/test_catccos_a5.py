# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

import torch

from vllm_ascend.catccos_debug import (
    CatccosProbeConfig,
    compare_tensors,
    is_significant_mismatch,
    load_probe_config,
    route_metadata,
    write_probe_result,
)
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

    def test_single_token_uses_catccos_when_threshold_is_one(self):
        layer = SimpleNamespace(
            _shared_experts=None,
            dynamic_eplb=False,
            quant_config=None,
            activation="silu",
        )

        with mock.patch.dict(os.environ, {"VLLM_ASCEND_CATCCOS_MINM": "1"}):
            self.assertFalse(_should_use_native_path(layer, torch.empty(1, 2048), False))

    def test_rejects_non_positive_token_threshold(self):
        layer = SimpleNamespace(
            _shared_experts=None,
            dynamic_eplb=False,
            quant_config=None,
            activation="silu",
        )

        with (
            mock.patch.dict(os.environ, {"VLLM_ASCEND_CATCCOS_MINM": "0"}),
            self.assertRaisesRegex(ValueError, "positive integer"),
        ):
            _should_use_native_path(layer, torch.empty(1, 2048), False)

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

    def test_probe_config_selects_fixed_prompt_token_count(self):
        environment = {
            "VLLM_ASCEND_CATCCOS_DEBUG_DIR": "/tmp/catccos-probe",
            "VLLM_ASCEND_CATCCOS_DEBUG_TOKEN_COUNTS": "1,177",
            "VLLM_ASCEND_CATCCOS_DEBUG_MOE_INSTANCE_IDS": "0,2",
            "VLLM_ASCEND_CATCCOS_DEBUG_ORDER": "catccos-native",
            "VLLM_ASCEND_CATCCOS_DEBUG_MAX_CALLS_PER_LAYER": "2",
            "VLLM_ASCEND_CATCCOS_DEBUG_COSINE_THRESHOLD": "0.95",
            "VLLM_ASCEND_CATCCOS_DEBUG_RELATIVE_L2_THRESHOLD": "0.2",
            "VLLM_ASCEND_CATCCOS_DEBUG_DUMP_TENSORS": "1",
            "VLLM_ASCEND_CATCCOS_DEBUG_DUMP_SELECTED": "1",
            "VLLM_ASCEND_CATCCOS_DEBUG_DUMP_WEIGHTS": "0",
        }
        with mock.patch.dict(os.environ, environment):
            config = load_probe_config()

        self.assertTrue(config.selects(1, 0))
        self.assertTrue(config.selects(177, 2))
        self.assertFalse(config.selects(177, 1))
        self.assertFalse(config.selects(64, 0))
        self.assertEqual(config.moe_instance_ids, frozenset({0, 2}))
        self.assertEqual(config.order, "catccos-native")
        self.assertEqual(config.max_calls_per_layer, 2)
        self.assertEqual(config.relative_l2_threshold, 0.2)
        self.assertTrue(config.dump_selected)

    def test_probe_rejects_invalid_order(self):
        environment = {
            "VLLM_ASCEND_CATCCOS_DEBUG_DIR": "/tmp/catccos-probe",
            "VLLM_ASCEND_CATCCOS_DEBUG_TOKEN_COUNTS": "177",
            "VLLM_ASCEND_CATCCOS_DEBUG_ORDER": "invalid",
        }
        with (
            mock.patch.dict(os.environ, environment),
            self.assertRaisesRegex(ValueError, "order must be one of"),
        ):
            load_probe_config()

    def test_probe_rejects_negative_moe_instance_id(self):
        environment = {
            "VLLM_ASCEND_CATCCOS_DEBUG_DIR": "/tmp/catccos-probe",
            "VLLM_ASCEND_CATCCOS_DEBUG_TOKEN_COUNTS": "177",
            "VLLM_ASCEND_CATCCOS_DEBUG_MOE_INSTANCE_IDS": "-1",
        }
        with (
            mock.patch.dict(os.environ, environment),
            self.assertRaisesRegex(ValueError, "must be non-negative"),
        ):
            load_probe_config()

    def test_probe_compares_frozen_outputs(self):
        reference = torch.tensor([[1.0, -2.0, 3.0]])
        identical = compare_tensors(reference, reference.clone())
        different = compare_tensors(reference, -reference)

        self.assertTrue(identical["exact_equal"])
        self.assertAlmostEqual(identical["cosine_similarity"], 1.0, places=6)
        self.assertFalse(different["exact_equal"])
        self.assertAlmostEqual(different["cosine_similarity"], -1.0, places=6)
        self.assertEqual(different["sign_flip_ratio"], 1.0)

        scaled = compare_tensors(reference, reference * 2)
        self.assertGreater(scaled["cosine_similarity"], 0.99)
        self.assertTrue(is_significant_mismatch(scaled, 0.99, 0.1))

    def test_probe_summarizes_expert_routing(self):
        metadata = route_metadata(
            torch.tensor([[0, 2], [2, 3]], dtype=torch.int32),
            torch.tensor([[0.25, 0.75], [0.5, 0.5]]),
        )

        self.assertEqual(metadata["expert_min"], 0)
        self.assertEqual(metadata["expert_max"], 3)
        self.assertEqual(metadata["expert_token_counts"], {"0": 1, "2": 2, "3": 1})
        self.assertEqual(metadata["gate_weight_row_sum_min"], 1.0)
        self.assertEqual(metadata["gate_weight_row_sum_max"], 1.0)

    def test_probe_writes_summary_and_only_first_mismatch_dump(self):
        with tempfile.TemporaryDirectory() as directory:
            config = CatccosProbeConfig(
                output_dir=Path(directory),
                token_counts=frozenset({177}),
                moe_instance_ids=frozenset({0}),
                order="native-catccos",
                max_calls_per_layer=1,
                cosine_threshold=0.99,
                relative_l2_threshold=0.1,
                dump_tensors=True,
                dump_selected=False,
                dump_weights=False,
            )
            record = {
                "layer": "model.layers.0.mlp",
                "significant_mismatch": True,
            }
            tensors = {"hidden_states": torch.ones(2, 4)}

            first_dump = write_probe_result(config, 0, record, tensors, {})
            second_dump = write_probe_result(config, 0, record, tensors, {})

            self.assertIsNotNone(first_dump)
            assert first_dump is not None
            self.assertTrue(first_dump.is_file())
            self.assertIsNone(second_dump)
            summaries = (Path(directory) / "probe-rank000.jsonl").read_text().splitlines()
            self.assertEqual(len(summaries), 2)

    def test_targeted_probe_dumps_selected_call_without_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            config = CatccosProbeConfig(
                output_dir=Path(directory),
                token_counts=frozenset({177}),
                moe_instance_ids=frozenset({0}),
                order="native-catccos",
                max_calls_per_layer=1,
                cosine_threshold=0.99,
                relative_l2_threshold=0.1,
                dump_tensors=True,
                dump_selected=True,
                dump_weights=False,
            )
            record = {
                "layer": "model.layers.0.mlp",
                "significant_mismatch": False,
            }

            dump = write_probe_result(config, 0, record, {"hidden_states": torch.ones(1, 4)}, {})

            self.assertIsNotNone(dump)
            assert dump is not None
            self.assertIn("first-selected-rank000", dump.name)
            self.assertTrue((Path(directory) / "first-selected-rank000.json").is_file())
