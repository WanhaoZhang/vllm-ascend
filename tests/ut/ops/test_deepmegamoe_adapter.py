import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.fused_moe import deepmegamoe_adapter as adapter
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType


def _moe_config():
    return SimpleNamespace(
        hidden_dim=7168,
        intermediate_size_per_partition=2048,
        ep_size=4,
        num_experts=256,
        experts_per_token=8,
        in_dtype=torch.bfloat16,
    )


def test_deepmegamoe_accepts_deepseek_v4_layer():
    capability = adapter.evaluate_deepmegamoe_layer(
        _moe_config(),
        SimpleNamespace(quant_type=QuantType.W4A8MXFP),
        "silu",
        n_shared_experts=1,
        swiglu_limit=10.0,
        device_type=AscendDeviceType.A5,
    )

    assert capability.supported


def test_deepmegamoe_rejects_multiple_shared_experts():
    capability = adapter.evaluate_deepmegamoe_layer(
        _moe_config(),
        SimpleNamespace(quant_type=QuantType.W4A8MXFP),
        "silu",
        n_shared_experts=2,
        device_type=AscendDeviceType.A5,
    )

    assert not capability.supported
    assert "at most one" in capability.reason


def test_prepare_weights_pairs_mxfp4_scale_bytes(monkeypatch):
    captured = {}

    def transform(l1, l2):
        captured["l1"], captured["l2"] = l1, l2
        return l1, l2

    monkeypatch.setattr(adapter, "get_ascend_device_type", lambda: AscendDeviceType.A5)
    monkeypatch.setitem(sys.modules, "deep_gemm", SimpleNamespace(transform_weights_for_mega_moe=transform))
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.zeros((2, 256, 64), dtype=torch.uint8), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.zeros((2, 128, 64), dtype=torch.uint8), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.arange(2 * 256 * 4, dtype=torch.uint8).reshape(2, 256, 4), requires_grad=False
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.arange(2 * 128 * 4, dtype=torch.uint8).reshape(2, 128, 4), requires_grad=False
    )

    adapter.prepare_deepmegamoe_weights(layer)

    assert captured["l1"][0].dtype == torch.int8
    assert captured["l1"][1].dtype == torch.int16
    assert captured["l1"][1].shape == (2, 256, 2)
    assert torch.equal(captured["l1"][1].view(torch.uint8), layer.w13_weight_scale.view(torch.uint8))
    assert layer.w2_weight.dtype == torch.int8
    assert layer.w2_weight_scale.dtype == torch.int16
    assert layer._deepmegamoe_weights_ready


def test_apply_deepmegamoe_populates_ep_inputs_and_passes_clamp(monkeypatch):
    class Buffer:
        x = torch.zeros((4, 64), dtype=torch.float8_e4m3fn)
        x_sf = torch.zeros((4, 1), dtype=torch.int16)
        topk_idx = torch.zeros((4, 2), dtype=torch.int64)
        topk_weights = torch.zeros((4, 2), dtype=torch.float32)

    buffer = Buffer()
    monkeypatch.setattr(adapter, "get_ascend_config", lambda: SimpleNamespace(deepmegamoe_max_tokens_per_rank=4))
    monkeypatch.setattr(adapter, "_get_symm_buffer", lambda _: buffer)
    monkeypatch.setattr(
        adapter,
        "_quantize_mxfp8_activations",
        lambda x: (x.to(torch.float8_e4m3fn), torch.full((x.shape[0], 1), 7, dtype=torch.int16)),
    )
    launch = Mock(side_effect=lambda y, *_args, **_kwargs: y.fill_(1))
    monkeypatch.setitem(sys.modules, "deep_gemm", SimpleNamespace(fp8_fp4_mega_moe=launch))

    hidden = torch.arange(128, dtype=torch.bfloat16).reshape(2, 64)
    topk_ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.bfloat16)
    w1 = torch.empty((1,), dtype=torch.int8)
    w1_scale = torch.empty((1,), dtype=torch.int16)
    output = adapter.apply_deepmegamoe(
        hidden,
        topk_ids,
        topk_weights,
        w1,
        w1_scale,
        w1,
        w1_scale,
        SimpleNamespace(hidden_dim=64, experts_per_token=2),
        swiglu_limit=10.0,
    )

    assert torch.equal(buffer.x[:2], hidden.to(torch.float8_e4m3fn))
    assert torch.equal(buffer.x_sf[:2], torch.full((2, 1), 7, dtype=torch.int16))
    assert torch.equal(buffer.topk_idx[:2], topk_ids.to(torch.int64))
    assert torch.equal(buffer.topk_weights[:2], topk_weights.float())
    assert torch.equal(output, torch.ones_like(output))
    launch.assert_called_once()
    assert launch.call_args.kwargs["activation_clamp"] == 10.0


def test_apply_deepmegamoe_empty_batch_still_participates_in_ep_dispatch(monkeypatch):
    class Buffer:
        x = torch.empty((4, 64), dtype=torch.float8_e4m3fn)
        x_sf = torch.empty((4, 1), dtype=torch.int16)
        topk_idx = torch.empty((4, 2), dtype=torch.int64)
        topk_weights = torch.empty((4, 2), dtype=torch.float32)

    monkeypatch.setattr(adapter, "get_ascend_config", lambda: SimpleNamespace(deepmegamoe_max_tokens_per_rank=4))
    buffer = Buffer()
    get_buffer = Mock(return_value=buffer)
    monkeypatch.setattr(adapter, "_get_symm_buffer", get_buffer)
    launch = Mock()
    monkeypatch.setitem(sys.modules, "deep_gemm", SimpleNamespace(fp8_fp4_mega_moe=launch))

    hidden = torch.empty((0, 64), dtype=torch.bfloat16)
    topk_ids = torch.empty((0, 2), dtype=torch.int32)
    topk_weights = torch.empty((0, 2), dtype=torch.bfloat16)
    weight = torch.empty((1,), dtype=torch.int8)
    scale = torch.empty((1,), dtype=torch.int16)
    output = adapter.apply_deepmegamoe(
        hidden,
        topk_ids,
        topk_weights,
        weight,
        scale,
        weight,
        scale,
        SimpleNamespace(hidden_dim=64, experts_per_token=2),
        swiglu_limit=10.0,
    )

    assert output.shape == hidden.shape
    assert output.dtype == hidden.dtype
    get_buffer.assert_called_once()
    launch.assert_called_once()


def test_apply_deepmegamoe_rejects_hidden_size_mismatch(monkeypatch):
    monkeypatch.setattr(adapter, "get_ascend_config", lambda: SimpleNamespace(deepmegamoe_max_tokens_per_rank=4))
    hidden = torch.empty((1, 64), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="hidden size must match"):
        adapter.apply_deepmegamoe(
            hidden,
            torch.zeros((1, 2), dtype=torch.int32),
            torch.ones((1, 2), dtype=torch.bfloat16),
            torch.empty((1,), dtype=torch.int8),
            torch.empty((1,), dtype=torch.int16),
            torch.empty((1,), dtype=torch.int8),
            torch.empty((1,), dtype=torch.int16),
            SimpleNamespace(hidden_dim=128, experts_per_token=2),
        )


@pytest.mark.parametrize("scale_shape", [(2, 4), (2, 2, 2)])
def test_quantize_mxfp8_activations_packs_adjacent_scale_bytes(monkeypatch, scale_shape):
    scales = torch.arange(8, dtype=torch.uint8).reshape(scale_shape)
    quantized = torch.ones((2, 128), dtype=torch.float8_e4m3fn)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_dynamic_mx_quant=lambda *_args, **_kwargs: (quantized, scales)),
    )

    x, packed_scales = adapter._quantize_mxfp8_activations(torch.zeros((2, 128), dtype=torch.bfloat16))

    assert x.dtype == torch.float8_e4m3fn
    assert packed_scales.dtype == torch.int16
    assert packed_scales.shape == (2, 2)
    assert torch.equal(packed_scales.view(torch.uint8), scales.reshape(2, 4))


def test_deepmegamoe_accepts_wrapped_quant_method():
    capability = adapter.evaluate_deepmegamoe_layer(
        _moe_config(),
        SimpleNamespace(quant_method=SimpleNamespace(quant_type=QuantType.W4A8MXFP)),
        "silu",
        n_shared_experts=1,
        swiglu_limit=10.0,
        device_type=AscendDeviceType.A5,
    )
    assert capability.supported


def test_deepmegamoe_rejects_wrapped_dynamic_eplb():
    capability = adapter.evaluate_deepmegamoe_layer(
        _moe_config(),
        SimpleNamespace(quant_method=SimpleNamespace(quant_type=QuantType.W4A8MXFP, dynamic_eplb=True)),
        "silu",
        device_type=AscendDeviceType.A5,
    )
    assert not capability.supported
    assert "EPLB" in capability.reason


@pytest.mark.parametrize("intermediate", [256, 640])
def test_deepmegamoe_rejects_unsupported_intermediate(intermediate):
    config = _moe_config()
    config.intermediate_size_per_partition = intermediate
    capability = adapter.evaluate_deepmegamoe_layer(
        config, SimpleNamespace(quant_type=QuantType.W4A8MXFP), "silu", device_type=AscendDeviceType.A5
    )
    assert not capability.supported
    assert "at least 512 and divisible by 256" in capability.reason


def test_deepmegamoe_rejects_topk_over_kernel_limit():
    config = _moe_config()
    config.experts_per_token = 33
    capability = adapter.evaluate_deepmegamoe_layer(
        config, SimpleNamespace(quant_type=QuantType.W4A8MXFP), "silu", device_type=AscendDeviceType.A5
    )
    assert not capability.supported
    assert "[1, 32]" in capability.reason


def test_release_buffers_waits_before_destroy_and_is_idempotent(monkeypatch):
    events = []
    monkeypatch.setattr(torch.npu, "synchronize", lambda: events.append("sync"))
    first = SimpleNamespace(destroy=lambda: events.append("first"))
    second = SimpleNamespace(destroy=lambda: events.append("second"))
    monkeypatch.setattr(adapter, "_SYMM_BUFFERS", {1: first, 2: second})
    adapter.release_deepmegamoe_buffers()
    adapter.release_deepmegamoe_buffers()
    assert events == ["sync", "first", "second"]
    assert not adapter._SYMM_BUFFERS


def test_fusedmc2_accepts_standard_ep_expert_map(monkeypatch):
    from contextlib import nullcontext
    from vllm_ascend.ops.fused_moe import moe_comm_method as comm

    monkeypatch.setattr(comm, "deepmegamoe_backend_enabled", lambda: True)
    monkeypatch.setattr(comm, "moe_profile_range", lambda *_args: nullcontext())
    launch = Mock(return_value=torch.ones((1, 256), dtype=torch.bfloat16))
    monkeypatch.setattr(comm, "apply_deepmegamoe", launch)
    impl = comm.FusedMC2CommImpl.__new__(comm.FusedMC2CommImpl)
    impl.token_dispatcher = comm.TokenDispatcherWithMC2.__new__(comm.TokenDispatcherWithMC2)
    impl.deepmegamoe_capability = adapter.DeepMegaMoELayerCapability(True)
    impl.moe_config = _moe_config()
    weight = torch.empty(1, dtype=torch.int8)
    scale = torch.empty(1, dtype=torch.int16)
    inputs = SimpleNamespace(
        weights=SimpleNamespace(w1=weight, w2=weight, w1_scale=scale, w2_scale=scale,
                                w1_bias=None, w2_bias=None),
        routing=SimpleNamespace(expert_map=torch.tensor([0, 1, -1, -1]), log2phy=None,
                                global_redundant_expert_num=0, apply_router_weight_on_input=False),
        quant=SimpleNamespace(quant_type=QuantType.W4A8MXFP),
        dynamic_eplb=False, lora_context=None, swiglu_limit=10.,
        hidden_states=torch.zeros((1, 256), dtype=torch.bfloat16),
        topk_ids=torch.zeros((1, 6), dtype=torch.int32), topk_weights=torch.ones((1, 6)),
    )
    output = impl.fused_experts(inputs)
    launch.assert_called_once()
    assert output.routed_out.shape == (1, 256)


def test_prepare_profile_uses_deepmegamoe_name(monkeypatch):
    from vllm_ascend.ops.fused_moe import prepare_finalize as prepare

    monkeypatch.setattr(prepare, "catccos_backend_enabled", lambda: False)
    monkeypatch.setattr(prepare, "deepmegamoe_backend_enabled", lambda: True)
    monkeypatch.setattr(prepare, "_EXTRA_CTX", SimpleNamespace(moe_comm_type=prepare.MoECommType.FUSED_MC2))
    instance = prepare.PrepareAndFinalizeWithMC2.__new__(prepare.PrepareAndFinalizeWithMC2)
    assert instance._profile_backend() == "deepmegamoe"
