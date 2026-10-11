# deepMegaMoE: DeepSeek-V4-Flash integration

The `deepmegamoe` FusedMC2 backend calls DeepGEMM-Ascend MegaMoE on Ascend
950DT. Official packed MXFP4 expert weights and E8M0 scales are transformed
to the kernel layout without requantization. Activations use MXFP8. The
single shared expert runs outside the routed kernel with SwiGLU clamp=10.

## Configuration

Use TP1 with EP enabled, contiguous expert placement, BF16 hidden states,
SiLU, and no MTP. Dynamic EPLB, redundant experts, LoRA, expert bias,
sequence parallelism and multistream gate overlap are unsupported and
rejected. This integration was tested with DP4/TP1/EP4 on NPUs 0–3.

```json
{
  "enable_fused_mc2": 1,
  "fused_mc2_backend": "deepmegamoe",
  "deepmegamoe_max_tokens_per_rank": 4096,
  "multistream_dsv4_dsa_overlap": false
}
```

The DeepGEMM extension must be importable in the worker Python environment.
For this experiment it is built from the isolated
`codex/deepseek-v4-flash-deepmegamoe` branch based on `8491bbb4`. Its
`cann91_compat.hpp` covers the APIs used by MegaMoE on CANN 9.1. This is
not a compatibility claim for every DeepGEMM operator. The built extension
and its JIT compiler use GCC14 runtime libraries supplied by the experiment
directory; apply the same runtime to each comparison backend.

The EP scratch buffer is shared by layers with the same group and shape.
Worker shutdown waits for pending launches, destroys the buffers, then
tears down HCCL. A zero-token rank still participates in the collective.
Token counts above the configured per-rank capacity are rejected.

## Numerical behavior and validation

MegaMoE applies routing weights before intermediate MXFP8 quantization;
CatCCOS applies them after the second linear. Their results need not be
bitwise equal. The independent reference decodes E2M1/E8M0 on CPU and
models MegaMoE's FP32 SwiGLU, routing-weight placement, MXFP8 rounding,
BF16 second-linear output and FP32 combination.

Four-rank synthetic tests cover random and constant weights/scales,
M=1/32/4096, ragged and zero-token ranks, clamp=10, top-k=6 and repeated
calls. Maximum relative L2 in the small random suite was 6.12e-5; the
E256/H4096/I2048/M4096 constant model-size case matched exactly on all
ranks. The common-environment full model passed 140/140 requests, including
4096-token prefill and concurrency-32 long decode. These tests do not
establish task accuracy or provide formal performance scores.

Targeted unit tests:

```bash
pytest -q tests/ut/ops/test_deepmegamoe_adapter.py \
  tests/ut/ops/test_swiglu_mxfp8_quant.py \
  tests/ut/patch/platform/test_patch_dp_device_ids.py \
  tests/ut/test_ascend_config.py tests/ut/test_ascend_forward_context.py
```

These 85 tests passed in the common container. Full-model logs, synthetic
references, launch scripts and frozen dataset hashes are retained in
`work/deepseek-v4-test/results/full-20261011` on the experiment server.
