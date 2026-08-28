#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${PROBE_RUN_ID:?Set a unique PROBE_RUN_ID, for example stage1-20260828-01}"

export PROBE_TOKEN_COUNTS="${PROBE_TOKEN_COUNTS:-177}"
export PROBE_MOE_INSTANCE_IDS="${PROBE_MOE_INSTANCE_IDS:-0}"
export PROBE_MAX_CALLS_PER_LAYER="${PROBE_MAX_CALLS_PER_LAYER:-1}"
export PROBE_DUMP_TENSORS="${PROBE_DUMP_TENSORS:-1}"
export PROBE_DUMP_SELECTED="${PROBE_DUMP_SELECTED:-1}"
export PROBE_DUMP_WEIGHTS="${PROBE_DUMP_WEIGHTS:-1}"

echo "[stage1] run_id=${PROBE_RUN_ID} M=${PROBE_TOKEN_COUNTS} moe_instance_ids=${PROBE_MOE_INSTANCE_IDS}"
echo "[stage1] weights=${PROBE_DUMP_WEIGHTS}; four output stages and the selected sample are always dumped"

exec bash "${SCRIPT_DIR}/run_probe_in_container.sh" native-catccos
