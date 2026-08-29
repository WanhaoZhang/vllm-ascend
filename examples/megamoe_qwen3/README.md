# Qwen3-30B-A3B with the formal CatCCOS FusedMC2 backend

This directory is the hand-off entry point for running
`Qwen/Qwen3-30B-A3B` on Ascend 950. The same launcher supports the native
vLLM-Ascend MoE path and the CatCCOS
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
- [CATCCOS_FORMAL_E2E_QUICKSTART.md](CATCCOS_FORMAL_E2E_QUICKSTART.md):
  correctness-first end-to-end validation on the formal FusedMC2 path, both
  from the host and inside an existing container.
- `CATCCOS_PROBE.md` and `CATCCOS_FIRST_LAYER_REDUCTION_PROBE.md` record the
  historical monkey-patch investigation; they are not production launch paths.
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

CatCCOS two-NPU service:

```bash
MODE=catccos \
CONTAINER_NAME=megamoe-catccos \
PORT=18081 \
NPU_DEVICES=0,1 \
MAX_NUM_SEQS=1 \
CATCCOS_MIN_TOKENS=64 \
VLLM_ASCEND_SOURCE="$VLLM_ASCEND_SOURCE" \
CATCCOS_SOURCE="$CATCCOS_SOURCE" \
MODEL_PATH="$MODEL_PATH" \
bash examples/megamoe_qwen3/run_docker.sh
```

The launcher enables post-launch device synchronization by default for the
first correctness run. Set `CATCCOS_SYNC_DEVICE=0` only after correctness is
accepted and before measuring performance.

It also defaults `CATCCOS_MIN_TOKENS=64`, so small decode batches use native
MC2 while a larger prefill uses CatCCOS. Keep `MAX_NUM_SEQS=1` for this
prefill-only validation. Set `CATCCOS_MIN_TOKENS=1` only when validating the
CatCCOS decode path.

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
    "messages": [{
      "role": "user",
      "content": "The dragon can reach 1000 feet. Polly throws 400 feet normally and three times as far with a gemstone. How far outside the dragon's reach can she stand and still hit it? Return only the number."
    }],
    "temperature": 0,
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Confirm that CatCCOS was selected rather than silently running the native
path:

```bash
docker logs megamoe-catccos 2>&1 | grep -E \
  'Initialized CatCCOS|Executed CatCCOS through the formal FusedMC2 backend|CatCCOS is disabled below M=64'
```

## Scope and known risk

The backend is integrated through v0.23's normal FusedMC2 communication
lifecycle. It accepts unquantized BF16 MoE layers with SiLU activation and
prepares persistent MXFP8 expert weights during model loading. Dynamic EPLB,
quantized model weights, shared experts, graph mode, and LoRA are rejected.
The direct CatCCOS launcher is outside the TorchNPU dependency scheduler, so
input-readiness synchronization is retained before each call. The example
launcher also enables post-launch synchronization for correctness validation.

Capability, shape, routing-selection, and v0.23 lifecycle tests pass. Multi-NPU
startup cannot be validated on a host without generated 950 D2D topology. A
prior EP4 run on the monkey-patch branch showed a large GSM8K regression.
Always run the fixed-prompt native/CatCCOS comparison first, followed by the
full A/B procedure in [AISBENCH_GSM8K.md](AISBENCH_GSM8K.md), before using
accuracy or performance numbers.
