#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/home/weights/Qwen3-30B-A3B-Instruct-2507}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-catccos}"
PORT="${PORT:-28001}"
NPU_DEVICES="${NPU_DEVICES:-4,5,6,7}"
CATCCOS_SOURCE="${CATCCOS_SOURCE:-/home/z00956592/catccos}"
CATCCOS_BUILD_DIR="${CATCCOS_BUILD_DIR:-${CATCCOS_SOURCE}/build_codex}"
CATCCOS_IPPORT="${CATCCOS_IPPORT:-tcp://127.0.0.1:27020}"
CATCCOS_MEM="${CATCCOS_MEM:-2147483648}"
CATCCOS_MINM="${CATCCOS_MINM:-64}"
CATCCOS_WEIGHT_QUANT_BACKEND="${CATCCOS_WEIGHT_QUANT_BACKEND:-cpu}"
PROBE_OUTPUT_ROOT="${PROBE_OUTPUT_ROOT:-/home/z00956592/catccos-probe-results}"
PROBE_RUN_ID="${PROBE_RUN_ID:-prompt177}"
PROBE_TOKEN_COUNTS="${PROBE_TOKEN_COUNTS:-177}"
PROBE_MAX_CALLS_PER_LAYER="${PROBE_MAX_CALLS_PER_LAYER:-1}"
PROBE_COSINE_THRESHOLD="${PROBE_COSINE_THRESHOLD:-0.99}"
PROBE_RELATIVE_L2_THRESHOLD="${PROBE_RELATIVE_L2_THRESHOLD:-0.1}"
PROBE_DUMP_TENSORS="${PROBE_DUMP_TENSORS:-1}"
PROBE_DUMP_WEIGHTS="${PROBE_DUMP_WEIGHTS:-0}"

usage() {
    cat <<'EOF'
Usage:
  bash run_probe_in_container.sh baseline
  bash run_probe_in_container.sh catccos
  bash run_probe_in_container.sh native-native
  bash run_probe_in_container.sh catccos-catccos
  bash run_probe_in_container.sh native-catccos
  bash run_probe_in_container.sh catccos-native

The first two modes run the normal A/B service. The other four modes enable
the same-input MoE probe. Run each probe mode in a fresh server process and
send the fixed prompt exactly once with greedy decoding and max_tokens=1.
EOF
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -n "${MODE}" ]] || {
    usage
    exit 2
}
unset VLLM_ASCEND_CATCCOS_DEBUG_DIR
unset VLLM_ASCEND_CATCCOS_DEBUG_TOKEN_COUNTS
unset VLLM_ASCEND_CATCCOS_DEBUG_ORDER
unset VLLM_ASCEND_CATCCOS_DEBUG_MAX_CALLS_PER_LAYER
unset VLLM_ASCEND_CATCCOS_DEBUG_COSINE_THRESHOLD
unset VLLM_ASCEND_CATCCOS_DEBUG_RELATIVE_L2_THRESHOLD
unset VLLM_ASCEND_CATCCOS_DEBUG_DUMP_TENSORS
unset VLLM_ASCEND_CATCCOS_DEBUG_DUMP_WEIGHTS

probe_order=""
case "${MODE}" in
    baseline)
        export VLLM_ASCEND_CATCCOS=0
        ;;
    catccos)
        export VLLM_ASCEND_CATCCOS=1
        ;;
    native-native|catccos-catccos|native-catccos|catccos-native)
        export VLLM_ASCEND_CATCCOS=1
        probe_order="${MODE}"
        ;;
    *)
        usage
        fail "unsupported mode: ${MODE}"
        ;;
esac

[[ -d "${MODEL_PATH}" ]] || fail "model directory does not exist: ${MODEL_PATH}"
command -v vllm >/dev/null || fail "vllm is not installed in this container"
if [[ "${VLLM_ASCEND_CATCCOS}" == "1" ]]; then
    [[ "${CATCCOS_MINM}" =~ ^[1-9][0-9]*$ ]] || fail "CATCCOS_MINM must be positive"
    [[ -f "${CATCCOS_BUILD_DIR}/lib/libcatccos_torch.so" ]] || {
        fail "CatCCOS extension does not exist under ${CATCCOS_BUILD_DIR}/lib"
    }
    [[ -d "${CATCCOS_SOURCE}/examples/utils" ]] || {
        fail "CatCCOS quantization utilities do not exist"
    }
fi

export VLLM_ASCEND_CATCCOS_IPPORT="${CATCCOS_IPPORT}"
export VLLM_ASCEND_CATCCOS_MEM="${CATCCOS_MEM}"
export VLLM_ASCEND_CATCCOS_MINM="${CATCCOS_MINM}"
export VLLM_ASCEND_CATCCOS_LIBRARY_PATH="${CATCCOS_BUILD_DIR}/lib/libcatccos_torch.so"
export VLLM_ASCEND_CATCCOS_UTILS_PATH="${CATCCOS_SOURCE}/examples/utils"
export VLLM_ASCEND_CATCCOS_WEIGHT_QUANT_BACKEND="${CATCCOS_WEIGHT_QUANT_BACKEND}"
export LD_LIBRARY_PATH="${CATCCOS_BUILD_DIR}/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-26000-26100}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${probe_order}" ]]; then
    command -v "${PYTHON_BIN}" >/dev/null || fail "Python executable does not exist: ${PYTHON_BIN}"
    [[ "${PROBE_RUN_ID}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
        fail "PROBE_RUN_ID may contain only letters, digits, dot, underscore, and dash"
    }
    [[ "${PROBE_TOKEN_COUNTS}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || {
        fail "PROBE_TOKEN_COUNTS must contain positive comma-separated integers"
    }
    [[ "${PROBE_MAX_CALLS_PER_LAYER}" =~ ^[1-9][0-9]*$ ]] || {
        fail "PROBE_MAX_CALLS_PER_LAYER must be positive"
    }
    [[ "${PROBE_COSINE_THRESHOLD}" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || {
        fail "PROBE_COSINE_THRESHOLD must be between 0 and 1"
    }
    [[ "${PROBE_RELATIVE_L2_THRESHOLD}" =~ ^(0\.[0-9]*[1-9][0-9]*|[1-9][0-9]*(\.[0-9]+)?)$ ]] || {
        fail "PROBE_RELATIVE_L2_THRESHOLD must be positive"
    }
    [[ "${PROBE_DUMP_TENSORS}" =~ ^[01]$ ]] || fail "PROBE_DUMP_TENSORS must be 0 or 1"
    [[ "${PROBE_DUMP_WEIGHTS}" =~ ^[01]$ ]] || fail "PROBE_DUMP_WEIGHTS must be 0 or 1"
    IFS=',' read -r -a probe_token_counts <<<"${PROBE_TOKEN_COUNTS}"
    for token_count in "${probe_token_counts[@]}"; do
        (( token_count >= CATCCOS_MINM )) || {
            fail "probe M=${token_count} is below CATCCOS_MINM=${CATCCOS_MINM}; CatCCOS would be bypassed"
        }
    done
    probe_output_dir="${PROBE_OUTPUT_ROOT}/${PROBE_RUN_ID}/${probe_order}"
    if [[ -d "${probe_output_dir}" ]] && find "${probe_output_dir}" -mindepth 1 -print -quit | grep -q .; then
        fail "probe output is not empty: ${probe_output_dir}; choose a new PROBE_RUN_ID"
    fi
    mkdir -p "${probe_output_dir}"

    probe_module="$("${PYTHON_BIN}" -c '
from pathlib import Path
import vllm_ascend.catccos_debug as debug
import vllm_ascend.catccos_patch as patch

patch_source = Path(patch.__file__).read_text(encoding="utf-8")
if "VLLM_ASCEND_CATCCOS_DEBUG_DIR" not in patch_source:
    raise SystemExit("the active catccos_patch.py does not contain the probe")
print(debug.__file__)
')" || fail "the active vLLM-Ascend does not contain commit 499c53661 or later"

    export VLLM_ASCEND_CATCCOS_DEBUG_DIR="${probe_output_dir}"
    export VLLM_ASCEND_CATCCOS_DEBUG_TOKEN_COUNTS="${PROBE_TOKEN_COUNTS}"
    export VLLM_ASCEND_CATCCOS_DEBUG_ORDER="${probe_order}"
    export VLLM_ASCEND_CATCCOS_DEBUG_MAX_CALLS_PER_LAYER="${PROBE_MAX_CALLS_PER_LAYER}"
    export VLLM_ASCEND_CATCCOS_DEBUG_COSINE_THRESHOLD="${PROBE_COSINE_THRESHOLD}"
    export VLLM_ASCEND_CATCCOS_DEBUG_RELATIVE_L2_THRESHOLD="${PROBE_RELATIVE_L2_THRESHOLD}"
    export VLLM_ASCEND_CATCCOS_DEBUG_DUMP_TENSORS="${PROBE_DUMP_TENSORS}"
    export VLLM_ASCEND_CATCCOS_DEBUG_DUMP_WEIGHTS="${PROBE_DUMP_WEIGHTS}"

    echo "[serve-probe] probe module=${probe_module}"
    echo "[serve-probe] output=${probe_output_dir} M=${PROBE_TOKEN_COUNTS} order=${probe_order}"
fi

echo "[serve-probe] mode=${MODE} CatCCOS=${VLLM_ASCEND_CATCCOS} MINM=${CATCCOS_MINM}"
echo "[serve-probe] devices=${NPU_DEVICES} port=${PORT} model=${MODEL_PATH}"

exec vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --port "${PORT}" \
    --trust-remote-code \
    --max-num-seqs 32 \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --distributed-executor-backend mp \
    --enforce-eager \
    --gpu-memory-utilization 0.75 \
    --host 0.0.0.0
