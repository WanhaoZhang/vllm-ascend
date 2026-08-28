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
- [CATCCOS_PROBE.md](CATCCOS_PROBE.md): same-input native/CatCCOS layer probe,
  four-order validation matrix, JSONL output, and first-mismatch tensor dump.
- [CATCCOS_FIRST_LAYER_REDUCTION_PROBE.md](CATCCOS_FIRST_LAYER_REDUCTION_PROBE.md):
  first-layer native-local/reduced and CatCCOS pre/post-reduction probe.
- [CHANGELOG.md](CHANGELOG.md): purpose and validation status of every commit
  in this integration branch.
- `run_docker.sh`: launches either the native or CatCCOS service.
- `run_probe_in_container.sh`: launches normal A/B or one of the four probe
  orders from inside an existing four-card container.
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

The single-NPU launcher defaults `CATCCOS_MINM` to `1`, so both prefill and
single-token decode use CatCCOS. Multi-NPU runs retain the conservative value
`64` until TP2/TP4 decode correctness is accepted; set `CATCCOS_MINM=1`
explicitly when running that gate.

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
  'Enabled CatCCOS|Initialized CatCCOS|Converted CatCCOS|Executing CatCCOS'
```

For a decode-path check, require both a multi-token message and
`Executing CatCCOS A5 single-token decode path`. A healthy endpoint alone
does not prove that decode used CatCCOS.

## Scope and known risk

The backend accepts unquantized BF16 MoE layers with SiLU activation. Batches
with fewer than `VLLM_ASCEND_CATCCOS_MINM` token rows use the native path.
Dynamic EPLB, quantized model weights, and the shared-expert event path are
not supported by this prototype. The direct CatCCOS launcher is outside the
TorchNPU dependency scheduler, so synchronization before and after every MoE
call is mandatory in this correctness-first integration.

On `a5new`, a standalone single-rank replay passed after one M=64 call and ten
consecutive M=1 calls with bitwise-stable output. A real vLLM request also
logged both the multi-token prefill path and the M=1 decode path. That request
did not finish within the diagnostic timeout, so end-to-end CatCCOS decode
accuracy is not yet accepted. The earlier 185-sample result used the default
threshold 64 and therefore mostly measured CatCCOS prefill plus native decode.

Multi-NPU startup cannot be validated on a host without generated 950 D2D
topology. A prior EP4 run also showed a large GSM8K accuracy regression.
Always run the fixed-prompt native/CatCCOS comparison first, followed by the
full A/B procedure in [AISBENCH_GSM8K.md](AISBENCH_GSM8K.md), before using
accuracy or performance numbers.
