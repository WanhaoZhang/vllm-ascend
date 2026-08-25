# Qwen3-30B-A3B with the experimental CatCCOS backend

This directory is the hand-off entry point for running
`Qwen/Qwen3-30B-A3B` on Ascend 950. The same launcher supports the native
vLLM-Ascend MoE path and the experimental CatCCOS
`ascend950_dispatch_ffn_combine` path so that the two implementations can be
compared with identical serving arguments.

The integration currently targets the official
`quay.io/ascend/vllm-ascend:v0.23.0-a5` image and eager execution.

## Documents and scripts

- [CATCCOS_950DT_GUIDE.md](CATCCOS_950DT_GUIDE.md): clone, build, single-NPU,
  TP2/TP4, verification, and troubleshooting steps for another 950DT server.
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

Pull the image and define the paths used by both modes:

```bash
docker pull quay.io/ascend/vllm-ascend:v0.23.0-a5

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

CatCCOS single-NPU service:

```bash
MODE=catccos \
CONTAINER_NAME=megamoe-catccos \
PORT=18081 \
NPU_DEVICES=0 \
VLLM_ASCEND_SOURCE="$VLLM_ASCEND_SOURCE" \
CATCCOS_SOURCE="$CATCCOS_SOURCE" \
MODEL_PATH="$MODEL_PATH" \
bash examples/megamoe_qwen3/run_docker.sh
```

The CatCCOS extension must already exist at
`$CATCCOS_SOURCE/build_torch_a5/lib/libcatccos_torch.so`. See the 950DT guide
for the reproducible build command.

For two or four NPUs, change `NPU_DEVICES` to `0,1` or `0,1,2,3`. The launcher
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

The backend accepts unquantized BF16 MoE layers with SiLU activation. Batches
with fewer than `VLLM_ASCEND_CATCCOS_MINM` tokens use the native path. Dynamic
EPLB, quantized model weights, and the shared-expert event path are not
supported by this prototype.

Single-NPU serving and numerical smoke tests have passed. Multi-NPU startup
cannot be validated on a host without the generated 950 D2D topology. A prior
EP4 run also showed a large GSM8K accuracy regression. The synchronization
fix in this branch addresses direct-launch input readiness, but it is not
evidence that the cross-rank accuracy issue is resolved. Always run the
native/CatCCOS A/B procedure in [AISBENCH_GSM8K.md](AISBENCH_GSM8K.md) before
using multi-NPU performance numbers.
