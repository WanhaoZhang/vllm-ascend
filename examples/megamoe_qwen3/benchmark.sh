#!/usr/bin/env bash

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-megamoe-qwen3}"
BASE_URL="${BASE_URL:-http://127.0.0.1:18080}"
MODEL="${MODEL:-Qwen3-30B-A3B}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
NUM_WARMUPS="${NUM_WARMUPS:-1}"
INPUT_LEN="${INPUT_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"

docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1 || {
    echo "ERROR: container does not exist: ${CONTAINER_NAME}" >&2
    exit 1
}

docker exec "${CONTAINER_NAME}" /bin/bash -lc \
    'source /usr/local/Ascend/ascend-toolkit/set_env.sh && exec "$@"' \
    -- vllm bench serve \
    --backend openai \
    --base-url "${BASE_URL}" \
    --endpoint /v1/completions \
    --model "${MODEL}" \
    --tokenizer /model \
    --dataset-name random \
    --num-prompts "${NUM_PROMPTS}" \
    --num-warmups "${NUM_WARMUPS}" \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --request-rate inf \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --ignore-eos \
    --temperature 0 \
    --seed 0
