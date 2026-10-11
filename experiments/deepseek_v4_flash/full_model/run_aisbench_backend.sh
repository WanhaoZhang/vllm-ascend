#!/usr/bin/env bash
# Run against an already validated service; keep the service identical across points.
set -Eeuo pipefail
backend=${1:?usage: run_aisbench_backend.sh BACKEND VALIDATED_SERVICE_DIR OUTPUT_DIR}
service_dir=${2:?validated service directory}
output_dir=${3:?new output directory}
root=/home/z009565592/catccos-vllm/work/deepseek-v4-test
datasets=$root/datasets-dsv4-20261011
test -s "$service_dir/smoke-summary.json"
python3 - "$service_dir/smoke-summary.json" "$datasets" <<'PY'
import hashlib, json, sys
from pathlib import Path
assert json.loads(Path(sys.argv[1]).read_text())['passed']
root = Path(sys.argv[2])
for name, info in json.loads((root / 'dataset-manifest.json').read_text())['files'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == info['sha256'], name
checkpoint = root.parent / 'results/full-20261011/checkpoint-manifest.json'
for name, expected in json.loads(checkpoint.read_text())['tokenizer_files_sha256'].items():
    assert hashlib.sha256(Path('/data/models/DeepSeek-V4-Flash', name).read_bytes()).hexdigest() == expected, name
PY
mkdir -p "$output_dir"
if [[ -e "$output_dir/dataset-manifest.json" ]]; then
 cmp "$datasets/dataset-manifest.json" "$output_dir/dataset-manifest.json"
 cmp "$root/results/full-20261011/checkpoint-manifest.json" "$output_dir/checkpoint-manifest.json"
fi
cp "$datasets/dataset-manifest.json" "$output_dir/"
cp "$root/results/full-20261011/checkpoint-manifest.json" "$output_dir/"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1
export PERF_PORT=28041
points=(
 'decode-c1 decode-128.jsonl 1 256'
 'decode-c8 decode-128.jsonl 8 256'
 'decode-c32 decode-128.jsonl 32 256'
 'prefill-2048 prefill-2048.jsonl 1 1'
 'prefill-4096 prefill-4096.jsonl 1 1'
 'mixed-c8 mixed-512-4096-80-20.jsonl 8 128'
 'mixed-c32 mixed-512-4096-80-20.jsonl 32 128'
)
for point in "${points[@]}"; do
 read -r name file concurrency osl <<< "$point"
 requests=200
 warmups=16
 if [[ "$name" == decode-c1 ]]; then
  requests=32
  warmups=4
 fi
 for round in 1 2 3; do
  run=$output_dir/$name/round$round
  mkdir -p "$run"
  if [[ -e "$run/status.json" ]]; then
   python3 - "$run/status.json" "$backend" "$name" "$round" "$concurrency" "$osl" "$file" "$requests" "$warmups" <<'PY'
import json, sys
from pathlib import Path
path, backend, point, round_, concurrency, osl, dataset, requests, warmups = sys.argv[1:]
expected=dict(backend=backend, point=point, round=int(round_), concurrency=int(concurrency),
 output_tokens=int(osl), requests=int(requests), warmups=int(warmups), aisbench_rc=0, dataset=dataset)
assert json.loads(Path(path).read_text()) == expected, 'Existing round protocol differs or CLI failed'
PY
   python3 "$root/verify_aisbench_run.py" "$run"
   echo "$backend $name round$round verified and retained"
   continue
  fi
  test ! -e "$run/aisbench"
  export PERF_DATASET=$datasets/$file PERF_BATCH=$concurrency PERF_OSL=$osl
  python3 - "$root/perf_config.py" "$run/perf_config.py" "$PERF_DATASET" "$concurrency" "$osl" <<'PY'
import json, sys
from pathlib import Path
source, target, dataset, concurrency, osl = sys.argv[1:]
text = Path(source).read_text().replace('__DATASET__', json.dumps(dataset))
text = text.replace('__BATCH__', concurrency).replace('__OSL__', osl)
Path(target).write_text(text)
PY
  curl -fsS --max-time 5 http://127.0.0.1:28041/health > "$run/health-before.txt"
  date -Iseconds > "$run/started.txt"
  set +e
  /opt/aisbench-venv/bin/ais_bench "$run/perf_config.py" --mode perf \
   --num-prompts "$requests" --num-warmups "$warmups" --work-dir "$run/aisbench" --debug > "$run/aisbench.log" 2>&1
  rc=$?
  set -e
  python3 - "$run" "$backend" "$name" "$round" "$concurrency" "$osl" "$rc" "$file" "$requests" "$warmups" <<'PY'
import json, sys
from pathlib import Path
run, backend, point, round_, concurrency, osl, rc, dataset, requests, warmups = sys.argv[1:]
Path(run, 'status.json').write_text(json.dumps(dict(backend=backend, point=point,
 round=int(round_), concurrency=int(concurrency), output_tokens=int(osl),
 requests=int(requests), warmups=int(warmups), aisbench_rc=int(rc), dataset=dataset), indent=2))
PY
  test "$rc" = 0
  python3 "$root/verify_aisbench_run.py" "$run"
  curl -fsS --max-time 5 http://127.0.0.1:28041/health > "$run/health-after.txt"
  echo "$backend $name round$round finished"
 done
done
