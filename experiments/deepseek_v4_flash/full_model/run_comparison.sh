#!/usr/bin/env bash
# The initial full-model validation of all three backends must already pass.
set -Eeuo pipefail
root=/home/z009565592/catccos-vllm/work/deepseek-v4-test
validation=$root/results/full-20261011
run_root=${1:?usage: run_comparison.sh OUTPUT_DIRECTORY [--resume]}
attempt=1
if [[ ${2:-} == --resume ]]; then
 test -f "$run_root/pipeline.exit-code"
 test "$(cat "$run_root/pipeline.exit-code")" != 0
 attempt=2
 while [[ -e "$run_root/services/catccos-attempt$attempt" ]]; do attempt=$((attempt+1)); done
 mv "$run_root/pipeline.exit-code" "$run_root/pipeline-attempt$((attempt-1)).exit-code"
else
 test ! -e "$run_root"
fi
mkdir -p "$run_root"
python3 - "$validation" <<'PY'
import json, sys
from pathlib import Path
r=Path(sys.argv[1])
for name in ('catccos-r17-hccl1536','deepmegamoe-r3-hccl1536','native-r2-hccl1536'):
    d=json.loads((r/name/'smoke-summary.json').read_text())
    assert d == dict(passed=True, requests=140), name
    env=json.loads((r/name/'environment.json').read_text())
    assert env['environment']['HCCL_BUFFSIZE'] == '1536', name
PY
active_service=''
cleanup() {
 rc=$?
 trap - EXIT
 if [[ -n "$active_service" ]]; then
  python3 "$root/stop_server.py" "$active_service" || true
  npu-smi info > "$active_service/npu-after.txt" || true
 fi
 printf '%s\n' "$rc" > "$run_root/pipeline.exit-code"
 exit "$rc"
}
trap cleanup EXIT
for backend in catccos deepmegamoe native; do
 service=$run_root/services/$backend-attempt$attempt
 mkdir -p "$service"
 active_service=$service
 RESULT_DIR="$service" nohup bash "$root/run_server.sh" "$backend" </dev/null >"$service/launcher.log" 2>&1 &
 echo $! > "$service/server.pid"
 python3 "$root/wait_for_service.py" "$service"
 python3 "$root/model_smoke.py" --datasets "$root/datasets-dsv4-20261011" --out "$service" >"$service/smoke-run.log" 2>&1
 bash "$root/run_aisbench_backend.sh" "$backend" "$service" "$run_root/aisbench/$backend"
 python3 "$root/stop_server.py" "$service"
 active_service=''
 npu-smi info > "$service/npu-after.txt"
done
/opt/aisbench-venv/bin/python "$root/summarize_aisbench.py" "$run_root/aisbench" --out "$run_root/summary"
date -Iseconds > "$run_root/completed.txt"
