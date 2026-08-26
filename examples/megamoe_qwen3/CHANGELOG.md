# CatCCOS integration commit ledger

This ledger explains the integration history so that a server checkout can be
matched to the behavior being evaluated. Hashes are from the
vLLM-Ascend `codex/megamoe-vllm-v023` and CatCCOS
`codex/megamoe-vllm` branches unless stated otherwise.

## vLLM-Ascend repository

### `d20d8714` — Qwen3 MoE Docker launcher

Added `run_docker.sh`, `benchmark.sh`, and the initial Qwen3-30B-A3B deployment
guide. The launcher validates selected NPU devices and requires generated
HCCL/HiXLEP topology files before TP2/TP4 startup.

### `393930f5` — experimental CatCCOS A5 backend

Added opt-in environment variables and a patch for the vLLM-Ascend 0.23.0
`AscendFusedMoE` path. It initializes CatCCOS per EP rank, converts BF16 expert
weights to MXFP8, invokes `ascend950_dispatch_ffn_combine`, and falls back to
native MoE below the configured token threshold. Unit tests cover environment
parsing, shape constraints, unsupported modes, and native fallback.

### `00063cd7` — activation enum compatibility

Accepted both the string `"silu"` and the vLLM activation enum used by the
real Qwen3 MoE layer. This fixed model startup after the initial integration.

### `41cb5348` — direct-launch input synchronization

Added an explicit NPU synchronization before the CatCCOS direct CCEC launch
and retained route tensors through the operator call. Optional post-launch
synchronization remains enabled by default. This fixed a reproducible
single-NPU invalid-device-pointer failure in the direct launcher.

It does not prove that the previously observed EP4 GSM8K accuracy regression
is fixed; that requires the A/B evaluation documented in
[AISBENCH_GSM8K.md](AISBENCH_GSM8K.md).

## CatCCOS repository

### `1d6a7e56` — A5 MegaMoE PyTorch binding

Added the `torch.ops.catccos` extension and the
`ascend950_dispatch_ffn_combine` binding used by vLLM-Ascend, plus Python
operator validation utilities.

### `55f642d6` — A5 architecture build wrapper

Made the PyTorch extension build select `CATLASS_ARCH=3510`, matching Ascend
950/A5 compilation requirements.

### `9cdc8985` — single-rank A5 operator validation

Allowed the Python operator runner to initialize and verify rank size one so
the operator can be separated from TP/EP and topology issues.

## Current documentation and evaluation work

### `6cc1222a` — reproducible deployment and AISBench hand-off

Made `run_docker.sh` select `MODE=native` or
`MODE=catccos`, adds reproducible 950DT hand-off instructions, provides an
environment-driven AISBench model configuration and GSM8K runner, and records
single-NPU smoke results.

### Multi-NPU report and partial accuracy evidence

Added a Chinese TP2/TP4 execution report and recorded the intentionally
stopped 185-sample common-prefix accuracy comparison. The final hash is the
`9c9704b7`.

### 2026-08-26 runtime correspondence validation

Recorded the verified Docker 29.6.2 host and official v0.23.0 A5 image digest,
proved that the image Git base is the branch's exact `v0.23.0` merge base, and
matched all three bind-mounted runtime files by SHA-256. Re-ran the six
CatCCOS unit tests and a 320-token real-model request that crosses the CatCCOS
threshold. Documented the important limitation that changes outside the three
mounted files require another mount or a complete image rebuild.
