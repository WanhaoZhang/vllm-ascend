#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AISBENCH_ROOT="${AISBENCH_ROOT:-}"
MODEL_PATH="${MODEL_PATH:-}"
MODE="${MODE:-accuracy}"
RUN_LABEL="${RUN_LABEL:-megamoe}"
NUM_PROMPTS="${NUM_PROMPTS:-}"
NUM_WARMUPS="${NUM_WARMUPS:-1}"
AISBENCH_HOST="${AISBENCH_HOST:-127.0.0.1}"
AISBENCH_PORT="${AISBENCH_PORT:-18080}"
AISBENCH_MAX_OUT_LEN="${AISBENCH_MAX_OUT_LEN:-1024}"
AISBENCH_BATCH_SIZE="${AISBENCH_BATCH_SIZE:-1}"
AISBENCH_REQUEST_RATE="${AISBENCH_REQUEST_RATE:-0}"
AISBENCH_STREAM="${AISBENCH_STREAM:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-30B-A3B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -n "${AISBENCH_ROOT}" ]] || fail "AISBENCH_ROOT is required"
[[ -n "${MODEL_PATH}" ]] || fail "MODEL_PATH is required"
if [[ -z "${AISBENCH_STREAM}" ]]; then
    if [[ "${MODE}" == "perf" ]]; then
        AISBENCH_STREAM=1
    else
        AISBENCH_STREAM=0
    fi
fi
[[ "${AISBENCH_STREAM}" =~ ^[01]$ ]] || {
    fail "AISBENCH_STREAM must be 0 or 1"
}
[[ -x "${AISBENCH_ROOT}/.venv/bin/ais_bench" ]] || {
    fail "AISBench executable is missing under ${AISBENCH_ROOT}/.venv"
}
[[ -d "${AISBENCH_ROOT}/ais_bench/datasets/gsm8k" ]] || {
    fail "GSM8K dataset is missing under ${AISBENCH_ROOT}/ais_bench/datasets"
}
command -v curl >/dev/null || fail "curl is not installed"
curl -fsS --max-time 5 \
    "http://${AISBENCH_HOST}:${AISBENCH_PORT}/health" >/dev/null || {
    fail "vLLM service is not healthy at ${AISBENCH_HOST}:${AISBENCH_PORT}"
}

if [[ -z "${OUTPUT_ROOT}" ]]; then
    OUTPUT_ROOT="${AISBENCH_ROOT}/outputs/megamoe"
fi
work_dir="${OUTPUT_ROOT}/${RUN_LABEL}/${MODE}"
mkdir -p "${work_dir}"

common_args=(
    --config-dir "${SCRIPT_DIR}/aisbench"
    --models vllm_api_megamoe
    --work-dir "${work_dir}"
    --num-warmups "${NUM_WARMUPS}"
)
if [[ -n "${NUM_PROMPTS}" ]]; then
    [[ "${NUM_PROMPTS}" =~ ^[1-9][0-9]*$ ]] || {
        fail "NUM_PROMPTS must be a positive integer"
    }
    common_args+=(--num-prompts "${NUM_PROMPTS}")
fi

export MODEL_PATH
export SERVED_MODEL_NAME
export AISBENCH_HOST
export AISBENCH_PORT
export AISBENCH_MAX_OUT_LEN
export AISBENCH_BATCH_SIZE
export AISBENCH_REQUEST_RATE
export AISBENCH_STREAM
export AISBENCH_MODEL_ABBR="${RUN_LABEL}"

cd "${AISBENCH_ROOT}"
case "${MODE}" in
    accuracy)
        exec .venv/bin/ais_bench \
            "${common_args[@]}" \
            --datasets gsm8k_gen_0_shot_cot_chat_prompt.py \
            --mode all \
            --dump-eval-details \
            --merge-ds
        ;;
    perf)
        exec .venv/bin/ais_bench \
            "${common_args[@]}" \
            --datasets gsm8k_gen_0_shot_cot_str_perf.py \
            --summarizer default_perf \
            --mode perf
        ;;
    *) fail "MODE must be accuracy or perf" ;;
esac
