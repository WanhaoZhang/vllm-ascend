# CatCCOS deployment guide for an Ascend 950DT server

This guide reproduces the experimental CatCCOS integration on a server that
already has a working Ascend 950 host driver, Docker, and D2D topology. It uses
two source repositories and the official vLLM-Ascend 0.23.0 A5 image:

- vLLM-Ascend integration and launch scripts:
  `WanhaoZhang/vllm-ascend`, branch `codex/megamoe-vllm-v023`
- CatCCOS operator and PyTorch extension:
  `zhangwanhao/catccos`, branch `codex/megamoe-vllm`

Run all commands from the host unless a command explicitly enters Docker.

## 1. Check the host

```bash
npu-smi info
docker version
test -e /dev/davinci_manager
test -e /dev/hisi_hdc
ls /dev/davinci*
```

For TP2 or TP4, all three host topology paths must exist:

```bash
test -f /lib/route.conf
test -f /etc/hccl_rootinfo.json
test -d /etc/hixlep
```

Generate missing topology files for this specific server. Do not copy them
from a different 950DT host. The launcher stops before model loading when a
multi-NPU topology path is missing.

## 2. Synchronize both repositories

New checkout:

```bash
mkdir -p /data/src

git clone --branch codex/megamoe-vllm-v023 \
  https://github.com/WanhaoZhang/vllm-ascend.git \
  /data/src/vllm-ascend

git clone --branch codex/megamoe-vllm --recurse-submodules \
  https://gitcode.com/zhangwanhao/catccos.git \
  /data/src/catccos
```

Update an existing checkout without discarding local work:

```bash
git -C /data/src/vllm-ascend switch codex/megamoe-vllm-v023
git -C /data/src/vllm-ascend pull --ff-only

git -C /data/src/catccos switch codex/megamoe-vllm
git -C /data/src/catccos pull --ff-only
git -C /data/src/catccos submodule update --init --recursive
```

Before an evaluation, record the exact revisions:

```bash
git -C /data/src/vllm-ascend rev-parse HEAD
git -C /data/src/catccos rev-parse HEAD
docker image inspect quay.io/ascend/vllm-ascend:v0.23.0-a5 \
  --format '{{.Id}}'
```

## 3. Build the CatCCOS PyTorch extension

The source directory is mounted read-write because the build output is kept
under `build_torch_a5`. Change `/dev/davinci0` if necessary.

```bash
export CATCCOS_SOURCE=/data/src/catccos
export IMAGE=quay.io/ascend/vllm-ascend:v0.23.0-a5

docker pull "$IMAGE"

docker run --rm -it \
  --ipc host \
  --network host \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$CATCCOS_SOURCE:/workspace/catccos" \
  "$IMAGE" \
  bash -lc '
    cd /workspace/catccos
    git config --global --add safe.directory /workspace/catccos
    bash examples/ascend950_dispatch_ffn_combine/scripts/build_python.sh
  '
```

The build is ready when both files exist:

```bash
test -f "$CATCCOS_SOURCE/build_torch_a5/lib/libcatccos_torch.so"
test -f \
  "$CATCCOS_SOURCE/build_torch_a5/lib/libascend950_dispatch_ffn_combine_kernel.so"
```

Optional operator-only validation should be completed before loading vLLM:

```bash
# Run inside the same image with the CatCCOS source mounted.
cd /workspace/catccos
bash examples/ascend950_dispatch_ffn_combine/scripts/run_python.sh 0
bash examples/ascend950_dispatch_ffn_combine/scripts/run_python.sh 0,1
bash examples/ascend950_dispatch_ffn_combine/scripts/run_python.sh 0,1,2,3
```

Use a different `IPPORT` for each concurrent CatCCOS job.

## 4. Launch the native baseline

```bash
cd /data/src/vllm-ascend

MODEL_PATH=/data/models/Qwen3-30B-A3B \
MODE=native \
CONTAINER_NAME=megamoe-native \
PORT=18080 \
NPU_DEVICES=0 \
bash examples/megamoe_qwen3/run_docker.sh
```

For a fair A/B test, use the same physical NPUs sequentially rather than
running native and CatCCOS simultaneously.

## 5. Launch CatCCOS

Single NPU:

```bash
cd /data/src/vllm-ascend

MODEL_PATH=/data/models/Qwen3-30B-A3B \
MODE=catccos \
CONTAINER_NAME=megamoe-catccos \
PORT=18081 \
NPU_DEVICES=0 \
VLLM_ASCEND_SOURCE=/data/src/vllm-ascend \
CATCCOS_SOURCE=/data/src/catccos \
CATCCOS_IPPORT=tcp://127.0.0.1:27020 \
bash examples/megamoe_qwen3/run_docker.sh
```

Two NPUs with TP2 and EP2:

```bash
MODEL_PATH=/data/models/Qwen3-30B-A3B \
MODE=catccos \
CONTAINER_NAME=megamoe-catccos-tp2 \
PORT=18081 \
NPU_DEVICES=0,1 \
GPU_MEMORY_UTILIZATION=0.80 \
VLLM_ASCEND_SOURCE=/data/src/vllm-ascend \
CATCCOS_SOURCE=/data/src/catccos \
CATCCOS_IPPORT=tcp://127.0.0.1:27021 \
bash examples/megamoe_qwen3/run_docker.sh
```

Four NPUs with TP4 and EP4:

```bash
MODEL_PATH=/data/models/Qwen3-30B-A3B \
MODE=catccos \
CONTAINER_NAME=megamoe-catccos-tp4 \
PORT=18081 \
NPU_DEVICES=0,1,2,3 \
GPU_MEMORY_UTILIZATION=0.80 \
VLLM_ASCEND_SOURCE=/data/src/vllm-ascend \
CATCCOS_SOURCE=/data/src/catccos \
CATCCOS_IPPORT=tcp://127.0.0.1:27022 \
bash examples/megamoe_qwen3/run_docker.sh
```

The launcher maps only the selected `/dev/davinci*` devices. It sets TP to the
number of devices and enables expert parallelism in multi-NPU mode.

## 6. Prove which path executed

```bash
curl -fsS http://127.0.0.1:18081/health

docker logs megamoe-catccos-tp4 2>&1 | grep -E \
  'Enabled CatCCOS|Initialized CatCCOS|Converted CatCCOS'
```

Expected CatCCOS messages include one enable message and initialization for
every EP rank. Absence of these messages means the result is not a CatCCOS
measurement.

Run a deterministic serving smoke test:

```bash
CONTAINER_NAME=megamoe-catccos-tp4 \
BASE_URL=http://127.0.0.1:18081 \
NUM_PROMPTS=8 \
INPUT_LEN=128 \
OUTPUT_LEN=64 \
MAX_CONCURRENCY=4 \
bash examples/megamoe_qwen3/benchmark.sh
```

Then run the GSM8K accuracy gate before collecting performance data. Follow
[AISBENCH_GSM8K.md](AISBENCH_GSM8K.md) exactly for both native and CatCCOS.

## 7. Correctness-first environment switches

The launcher defaults are suitable for initial correctness testing:

| Variable | Default | Purpose |
|---|---:|---|
| `CATCCOS_MINM` | `64` | Smaller batches fall back to native MoE. |
| `CATCCOS_WEIGHT_QUANT_BACKEND` | `npu` | Converts BF16 expert weights to MXFP8 on NPU. |
| `CATCCOS_SYNC_DEVICE` | `1` | Surfaces asynchronous custom-kernel errors at the MoE call. |
| `CATCCOS_MEM` | `1073741824` | Symmetric memory passed to CatCCOS initialization. |

For exact parity with the CatCCOS data generator, repeat the accuracy run with
`CATCCOS_WEIGHT_QUANT_BACKEND=cpu`. This is slower at startup. Set
`CATCCOS_SYNC_DEVICE=0` only after native/CatCCOS accuracy parity has passed;
otherwise asynchronous failures can be attributed to the wrong layer.

## 8. Troubleshooting

- `RootInfoDetect failed` or HCCL error code 4: the 950 D2D topology is absent
  or invalid. Regenerate it on the target server.
- `CatCCOS extension does not exist`: rebuild CatCCOS and check
  `build_torch_a5/lib/libcatccos_torch.so`.
- `libshmem...so: cannot open shared object file`: check
  `3rdparty/shmem/install/shmem/lib` and rebuild with the CatCCOS setup script.
- No CatCCOS log messages: confirm `MODE=catccos`, the source branch, and that
  `/vllm-workspace/vllm-ascend` is the expected read-only source mount in
  `docker inspect`.
- Accuracy differs strongly from native: stop performance testing, retain the
  AISBench predictions and all rank logs, repeat with CPU weight quantization,
  and reduce to single NPU to separate quantization from cross-rank behavior.
