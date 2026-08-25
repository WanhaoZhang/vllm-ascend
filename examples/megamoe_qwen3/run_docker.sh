#!/usr/bin/env bash

set -euo pipefail

IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.23.0-a5}"
CONTAINER_NAME="${CONTAINER_NAME:-megamoe-qwen3}"
MODEL_PATH="${MODEL_PATH:-}"
NPU_DEVICES="${NPU_DEVICES:-0}"
PORT="${PORT:-18080}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-auto}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"
RECREATE="${RECREATE:-0}"

usage() {
    cat <<'EOF'
Usage:
  MODEL_PATH=/path/to/Qwen3-30B-A3B NPU_DEVICES=0,1 bash run_docker.sh

Optional environment variables:
  IMAGE, CONTAINER_NAME, PORT, MAX_MODEL_LEN, MAX_NUM_BATCHED_TOKENS,
  MAX_NUM_SEQS, GPU_MEMORY_UTILIZATION, ENABLE_EXPERT_PARALLEL,
  HEALTH_TIMEOUT, RECREATE

ENABLE_EXPERT_PARALLEL accepts auto, 0, or 1. In auto mode it is enabled
when more than one NPU is selected. Set RECREATE=1 to replace an existing
container with the same name.
EOF
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -n "${MODEL_PATH}" ]] || {
    usage
    fail "MODEL_PATH is required"
}
[[ -d "${MODEL_PATH}" ]] || fail "model directory does not exist: ${MODEL_PATH}"
[[ -f "${MODEL_PATH}/config.json" ]] || fail "config.json is missing"
[[ -f "${MODEL_PATH}/model.safetensors.index.json" ]] || {
    fail "model.safetensors.index.json is missing"
}
command -v docker >/dev/null || fail "docker is not installed"
command -v curl >/dev/null || fail "curl is not installed"

IFS=',' read -r -a device_ids <<<"${NPU_DEVICES}"
(( ${#device_ids[@]} > 0 )) || fail "NPU_DEVICES is empty"

docker_devices=()
for device_id in "${device_ids[@]}"; do
    [[ "${device_id}" =~ ^[0-9]+$ ]] || {
        fail "invalid NPU device ID: ${device_id}"
    }
    device_path="/dev/davinci${device_id}"
    [[ -e "${device_path}" ]] || fail "device does not exist: ${device_path}"
    docker_devices+=(--device "${device_path}")
done

for device_path in /dev/davinci_manager /dev/hisi_hdc; do
    [[ -e "${device_path}" ]] || fail "device does not exist: ${device_path}"
    docker_devices+=(--device "${device_path}")
done

parallel_size="${#device_ids[@]}"
topology_mounts=()
if (( parallel_size > 1 )); then
    topology_paths=(/lib/route.conf /etc/hccl_rootinfo.json /etc/hixlep)
    missing_paths=()
    for topology_path in "${topology_paths[@]}"; do
        [[ -e "${topology_path}" ]] || missing_paths+=("${topology_path}")
    done
    if (( ${#missing_paths[@]} > 0 )); then
        printf 'ERROR: multi-NPU HCCL/HiXLEP topology is incomplete:\n' >&2
        printf '  missing %s\n' "${missing_paths[@]}" >&2
        printf 'Generate the Ascend 950 D2D topology before retrying.\n' >&2
        exit 1
    fi
    topology_mounts=(
        -v /lib/route.conf:/lib/route.conf:ro
        -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json:ro
        -v /etc/hixlep:/etc/hixlep:ro
    )
fi

case "${ENABLE_EXPERT_PARALLEL}" in
    auto)
        if (( parallel_size > 1 )); then
            enable_ep=1
        else
            enable_ep=0
        fi
        ;;
    0|1) enable_ep="${ENABLE_EXPERT_PARALLEL}" ;;
    *) fail "ENABLE_EXPERT_PARALLEL must be auto, 0, or 1" ;;
esac

serve_args=(
    vllm serve /model
    --served-model-name Qwen3-30B-A3B
    --trust-remote-code
    --dtype bfloat16
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --no-enable-prefix-caching
    --enforce-eager
    --host 0.0.0.0
    --port "${PORT}"
)

docker_env=()
if (( parallel_size > 1 )); then
    serve_args+=(
        --tensor-parallel-size "${parallel_size}"
        --distributed-executor-backend mp
    )
    docker_env+=(
        -e HCCL_OP_EXPANSION_MODE=AIV
        -e HCCL_BUFFSIZE=1024
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    )
fi
if (( enable_ep == 1 )); then
    serve_args+=(--enable-expert-parallel)
fi

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    if [[ "${RECREATE}" == "1" ]]; then
        docker rm -f "${CONTAINER_NAME}" >/dev/null
    else
        fail "container ${CONTAINER_NAME} already exists; set RECREATE=1 to replace it"
    fi
fi

echo "Starting ${CONTAINER_NAME} with TP=${parallel_size}, EP=${enable_ep}"
echo "Image: ${IMAGE}"
echo "NPUs: ${NPU_DEVICES}"

docker run -d \
    --name "${CONTAINER_NAME}" \
    --ipc host \
    --network host \
    "${docker_devices[@]}" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v "${MODEL_PATH}:/model:ro" \
    "${topology_mounts[@]}" \
    "${docker_env[@]}" \
    -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
    "${IMAGE}" \
    "${serve_args[@]}" >/dev/null

deadline=$((SECONDS + HEALTH_TIMEOUT))
until curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null; do
    if ! docker container inspect -f '{{.State.Running}}' \
        "${CONTAINER_NAME}" 2>/dev/null | grep -qx true; then
        docker logs --tail 200 "${CONTAINER_NAME}" >&2
        fail "container exited before becoming healthy"
    fi
    if (( SECONDS >= deadline )); then
        docker logs --tail 200 "${CONTAINER_NAME}" >&2
        fail "service did not become healthy within ${HEALTH_TIMEOUT}s"
    fi
    sleep 5
done

echo "Service is healthy: http://127.0.0.1:${PORT}"
echo "Model endpoint: http://127.0.0.1:${PORT}/v1/models"
