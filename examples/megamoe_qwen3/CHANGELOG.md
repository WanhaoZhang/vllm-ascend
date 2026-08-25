# CatCCOS integration commit ledger

This ledger explains the integration history so that a server checkout can be
matched to the behavior being evaluated. Hashes are from the
`codex/megamoe-vllm` branches unless stated otherwise.

## vLLM-Ascend repository

### `b3a041aa7` — Qwen3 MoE Docker launcher

Added `run_docker.sh`, `benchmark.sh`, and the initial Qwen3-30B-A3B deployment
guide. The launcher validates selected NPU devices and requires generated
HCCL/HiXLEP topology files before TP2/TP4 startup.

### `be61c2916` — experimental CatCCOS A5 backend

Added opt-in environment variables and a patch for the vLLM-Ascend 0.23.0
`AscendFusedMoE` path. It initializes CatCCOS per EP rank, converts BF16 expert
weights to MXFP8, invokes `ascend950_dispatch_ffn_combine`, and falls back to
native MoE below the configured token threshold. Unit tests cover environment
parsing, shape constraints, unsupported modes, and native fallback.

### `724296219` — activation enum compatibility

Accepted both the string `"silu"` and the vLLM activation enum used by the
real Qwen3 MoE layer. This fixed model startup after the initial integration.

### `826e9a360` — direct-launch input synchronization

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

The current change makes `run_docker.sh` select `MODE=native` or
`MODE=catccos`, adds reproducible 950DT hand-off instructions, provides an
environment-driven AISBench model configuration and GSM8K runner, and records
single-NPU test results. Its final commit hash is recorded in the Git history;
the files in this section are intentionally committed together so the guide,
scripts, and reported results cannot drift independently.
