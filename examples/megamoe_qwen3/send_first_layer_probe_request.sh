#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PORT="${PORT:-28001}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-catccos}"
PROBE_OUTPUT_ROOT="${PROBE_OUTPUT_ROOT:-/home/z00956592/catccos-probe-results}"
PROBE_TOKEN_COUNTS="${PROBE_TOKEN_COUNTS:-177}"
PROBE_EXPECTED_RANKS="${PROBE_EXPECTED_RANKS:-4}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"

: "${PROBE_RUN_ID:?Use the same PROBE_RUN_ID as the server terminal}"

probe_dir="${PROBE_OUTPUT_ROOT}/${PROBE_RUN_ID}/native-catccos"
response_path="${probe_dir}/chat-completion-response.json"
archive_path="${PROBE_OUTPUT_ROOT}/${PROBE_RUN_ID}.tar.gz"

deadline=$((SECONDS + HEALTH_TIMEOUT))
until curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; do
    (( SECONDS < deadline )) || {
        echo "ERROR: vLLM did not become healthy within ${HEALTH_TIMEOUT}s" >&2
        exit 1
    }
    sleep 2
done

mkdir -p "${probe_dir}"
curl --fail --silent --show-error "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data-binary @- >"${response_path}" <<JSON
{
  "model": "${SERVED_MODEL_NAME}",
  "messages": [{
    "role": "user",
    "content": "Answer the following question.The last line of the response should follow this format: \"answer:\$ANSWER\" (without quotes), where ANSWER is a number. Let's think step by step.\n\nQuestion: The great dragon, Perg, sat high atop mount Farbo, breathing fire upon anything within a distance of 1000 feet.  Polly could throw the gold javelin, the only known weapon that could sleigh the dragon, for a distance of 400 feet, well within the reach of the dragon's flames.  But when Polly held the sapphire gemstone, she could throw the javelin three times farther than when not holding the gemstone. If holding the gemstone, how far outside of the reach of the dragon's flames could Polly stand and still hit the dragon with the gold javelin?"
  }],
  "max_tokens": 1,
  "temperature": 0,
  "seed": 42
}
JSON

actual_prompt_tokens="$(${PYTHON_BIN} -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as response_file:
    response = json.load(response_file)
if "error" in response:
    raise SystemExit("request failed: {}".format(response["error"]))
print(response["usage"]["prompt_tokens"])
' "${response_path}")"

if [[ ",${PROBE_TOKEN_COUNTS}," != *",${actual_prompt_tokens},"* ]]; then
    echo "ERROR: request used M=${actual_prompt_tokens}, but PROBE_TOKEN_COUNTS=${PROBE_TOKEN_COUNTS}" >&2
    echo "No selected probe may have run; update PROBE_TOKEN_COUNTS and restart the server." >&2
    exit 1
fi

marker_deadline=$((SECONDS + 120))
while true; do
    marker_count="$(find "${probe_dir}" -maxdepth 1 -name 'first-selected-rank*.json' | wc -l | tr -d ' ')"
    if (( marker_count >= PROBE_EXPECTED_RANKS )); then
        break
    fi
    (( SECONDS < marker_deadline )) || {
        echo "ERROR: found ${marker_count}/${PROBE_EXPECTED_RANKS} rank markers under ${probe_dir}" >&2
        exit 1
    }
    sleep 1
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_first_layer_reduction_probe.py" "${probe_dir}"
tar -czf "${archive_path}" -C "${PROBE_OUTPUT_ROOT}" "${PROBE_RUN_ID}"

echo "[stage1] response=${response_path}"
echo "[stage1] results=${probe_dir}"
echo "[stage1] archive=${archive_path}"
