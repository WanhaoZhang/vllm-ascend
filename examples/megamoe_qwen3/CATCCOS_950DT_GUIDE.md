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

## 7. Verified `a5new` reference environment

The following single-NPU configuration was revalidated on 2026-08-26:

| Item | Verified value |
|---|---|
| vLLM-Ascend branch | `codex/megamoe-vllm-v023@a2fa5c2b` |
| vLLM-Ascend base | `v0.23.0@5cb98caa` |
| Docker client/server | 29.6.2, API 1.55 |
| Host | Ubuntu 24.04.4 LTS, kernel 6.8.0-136-generic |
| Storage/cgroup | overlayfs, systemd, cgroup v2 |
| Image | `quay.io/ascend/vllm-ascend:v0.23.0-a5` |
| Image digest | `sha256:cc57064f119054904dc81360cd1105d211fa9b91bf726486926dd025c26f17b7` |
| vLLM-Ascend package | 0.23.0 |
| PyTorch/TorchNPU | 2.10.0+cpu / 2.10.0.post4 |

The image contains the exact `v0.23.0` Git commit `5cb98caa`. CatCCOS mode
does not rebuild the whole branch into the image. It bind-mounts these three
runtime files from the checkout:

```text
vllm_ascend/__init__.py
vllm_ascend/envs.py
vllm_ascend/catccos_patch.py
```

On `a5new`, the host and container SHA-256 values were identical:

| File | SHA-256 |
|---|---|
| `__init__.py` | `e6505ce2562054f74df176c35cee2f3da64a8acef524c1741265a6160f4dea29` |
| `envs.py` | `b6a5c5a420b31465dfc793bf3a7d9cee0f989daff4a6bd1536a2bb033aa93251` |
| `catccos_patch.py` | `e9384577a9c7a38b4ec1efb584e1d501ef4ed41fd473bc0b0bf424311942076b` |

Recheck the relationship after every source update:

```bash
git rev-parse HEAD
git merge-base HEAD v0.23.0

docker exec megamoe-catccos \
  git -C /vllm-workspace/vllm-ascend rev-parse HEAD

sha256sum \
  vllm_ascend/__init__.py \
  vllm_ascend/envs.py \
  vllm_ascend/catccos_patch.py

docker exec megamoe-catccos sha256sum \
  /vllm-workspace/vllm-ascend/vllm_ascend/__init__.py \
  /vllm-workspace/vllm-ascend/vllm_ascend/envs.py \
  /vllm-workspace/vllm-ascend/vllm_ascend/catccos_patch.py
```

The branch unit test passed with six tests. The isolated test container needs
the Ascend driver mounts and `TORCH_DEVICE_BACKEND_AUTOLOAD=0`; without that
variable, Torch backend auto-loading fails during test collection before any
CatCCOS test runs.

The real-model smoke used 320 prompt tokens, exceeding the default
`CATCCOS_MINM=64` threshold, and returned HTTP 200 with the correct answer
`42`. The serving container remained running with restart count zero and no
new error or traceback in the ten-minute validation window. This is a
single-NPU functional check, not a performance result or multi-NPU accuracy
acceptance result.

This source-to-container correspondence remains valid only while runtime
changes are limited to the three mounted files. If future development changes
any other `vllm_ascend` source file, extend `run_docker.sh` to mount it or build
a new image from the complete branch before testing.

## 8. Correctness-first environment switches

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

## 9. Troubleshooting

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
