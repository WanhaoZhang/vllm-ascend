# Qwen3-30B-A3B with the experimental CatCCOS backend

This directory is the hand-off entry point for running
`Qwen/Qwen3-30B-A3B` on Ascend 950. The same launcher supports the native
vLLM-Ascend MoE path and the experimental CatCCOS
`ascend950_dispatch_ffn_combine` path so that the two implementations can be
compared with identical serving arguments.

The integration uses the current vLLM-Ascend source tree and eager execution.
Set `IMAGE` explicitly to a development image built for the same vLLM and
vLLM-Ascend revisions; the older v0.23 image used by the monkey-patch branch
is not compatible with this native backend integration.

## Documents and scripts

- [CATCCOS_950DT_GUIDE.md](CATCCOS_950DT_GUIDE.md): clone, build, TP2/TP4,
  verification, and troubleshooting steps for another 950DT server.
- [AISBENCH_GSM8K.md](AISBENCH_GSM8K.md): controlled native-versus-CatCCOS
  GSM8K accuracy and performance evaluation.
- [MULTI_NPU_TEST_REPORT.md](MULTI_NPU_TEST_REPORT.md): Chinese execution report
  and acceptance checklist for TP2/EP2 and TP4/EP4 on another 950DT server.
- [CHANGELOG.md](CHANGELOG.md): purpose and validation status of every commit
  in this integration branch.
- `run_docker.sh`: launches either the native or CatCCOS service.
- `benchmark.sh`: runs a deterministic random-token serving benchmark.
- `run_aisbench_gsm8k.sh`: runs the documented AISBench GSM8K job.

## Quick start

Define a compatible development image and the paths used by both modes:

```bash
export IMAGE=<compatible-vllm-ascend-development-image>
export MODEL_PATH=/data/models/Qwen3-30B-A3B
export VLLM_ASCEND_SOURCE=/data/src/vllm-ascend
export CATCCOS_SOURCE=/data/src/catccos
```

Native single-NPU service:

```bash
MODE=native \
CONTAINER_NAME=megamoe-native \
PORT=18080 \
NPU_DEVICES=0 \
MODEL_PATH="$MODEL_PATH" \
bash examples/megamoe_qwen3/run_docker.sh
```

CatCCOS two-NPU service:

```bash
MODE=catccos \
CONTAINER_NAME=megamoe-catccos \
PORT=18081 \
NPU_DEVICES=0,1 \
VLLM_ASCEND_SOURCE="$VLLM_ASCEND_SOURCE" \
CATCCOS_SOURCE="$CATCCOS_SOURCE" \
MODEL_PATH="$MODEL_PATH" \
bash examples/megamoe_qwen3/run_docker.sh
```

The CatCCOS extension must already exist at
`$CATCCOS_SOURCE/build_torch_a5/lib/libcatccos_torch.so`. See the 950DT guide
for the reproducible build command.

For four NPUs, change `NPU_DEVICES` to `0,1,2,3`. The launcher
automatically sets tensor parallel size and enables expert parallelism. It
also refuses to start multi-NPU mode if the host topology files are absent.

## Basic verification

```bash
curl -fsS http://127.0.0.1:18081/health
curl -fsS http://127.0.0.1:18081/v1/models

curl http://127.0.0.1:18081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-30B-A3B",
    "messages": [{"role": "user", "content": "What is 17 * 23?"}],
    "temperature": 0,
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Confirm that CatCCOS was selected rather than silently running the native
path:

```bash
docker logs megamoe-catccos 2>&1 | grep -E \
  'Enabled CatCCOS|Initialized CatCCOS|Converted CatCCOS'
```

## Scope and known risk

The backend accepts unquantized BF16 MoE layers with SiLU activation and
converts expert weights to MXFP8 during model weight processing. Communication
selection uses the rank-invariant MC2 token capacity; batches above
`catccos_max_tokens_per_rank` use the native path on every EP rank. Dynamic
EPLB, quantized checkpoints, model runner v2, graph mode, LoRA, and shared
experts are rejected during startup.

Capability, routing-selection, shape, and adapter unit tests pass. Multi-NPU
startup and numerical correctness cannot be validated on a host without HCCL
and the generated 950 D2D topology. A prior EP4 run on the monkey-patch branch
also showed a large GSM8K accuracy regression; this integration refactor is
not evidence that the cross-rank issue is resolved. Always run the
native/CatCCOS A/B procedure in [AISBENCH_GSM8K.md](AISBENCH_GSM8K.md) before
using multi-NPU performance numbers.
