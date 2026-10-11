#!/usr/bin/env python3
"""Audit all 63 measured rounds against raw records and current source artifacts.

Run with the AISBench Python environment after the comparison pipeline exits.
This does not launch services, change results, or treat partial results as complete.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

from summarize_aisbench import summarize


BACKENDS = ('catccos', 'deepmegamoe', 'native')
POINTS = {
    'decode-c1': ('decode-128.jsonl', 1, 256, 32, 4),
    'decode-c8': ('decode-128.jsonl', 8, 256, 200, 16),
    'decode-c32': ('decode-128.jsonl', 32, 256, 200, 16),
    'prefill-2048': ('prefill-2048.jsonl', 1, 1, 200, 16),
    'prefill-4096': ('prefill-4096.jsonl', 1, 1, 200, 16),
    'mixed-c8': ('mixed-512-4096-80-20.jsonl', 8, 128, 200, 16),
    'mixed-c32': ('mixed-512-4096-80-20.jsonl', 32, 128, 200, 16),
}


def read_json(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def audit(run_root):
    require((run_root / 'pipeline.exit-code').read_text().strip() == '0',
            'Comparison pipeline did not exit successfully')
    require(bool((run_root / 'completed.txt').read_text().strip()),
            'Completion marker is empty')
    root = run_root.parent.parent
    dataset_root = root / 'datasets-dsv4-20261011'
    manifest = read_json(dataset_root / 'dataset-manifest.json')
    checkpoint_path = root / 'results/full-20261011/checkpoint-manifest.json'
    checkpoint = read_json(checkpoint_path)
    for name, info in manifest['files'].items():
        require(hashlib.sha256((dataset_root / name).read_bytes()).hexdigest() == info['sha256'],
                f'Dataset changed: {name}')
    for name, digest in checkpoint['tokenizer_files_sha256'].items():
        require(hashlib.sha256(Path('/data/models/DeepSeek-V4-Flash', name).read_bytes()).hexdigest() == digest,
                f'Tokenizer changed: {name}')

    runtime = read_json(run_root / 'common-runtime-audit.json')
    require(runtime['passed'] and set(runtime['formal_backends']) == set(BACKENDS)
            and not runtime['native_formal_pending'], 'Common formal runtime audit incomplete')
    sources = read_json(run_root / 'resume-source-audit.json')
    require(sources['passed'], 'Source resume audit failed')
    for record in sources['commits']:
        args = ['git', '-c', 'safe.directory=' + record['path'], '-C', record['path']]
        head = subprocess.check_output(args + ['rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(args + ['status', '--porcelain', '--untracked-files=no'], text=True).strip()
        require(head == record['commit'] and not dirty, f'Implementation changed: {record["path"]}')
    for name, digest in sources['libraries_sha256'].items():
        require(hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest,
                f'Compiled library changed: {name}')

    full_runs = ('catccos-r17-hccl1536', 'deepmegamoe-r3-hccl1536', 'native-r2-hccl1536')
    for backend, name in zip(BACKENDS, full_runs):
        require(read_json(root / 'results/full-20261011' / name / 'smoke-summary.json')
                == dict(passed=True, requests=140), f'Full-model validation failed: {backend}')
        service = run_root / 'services' / (backend + '-attempt2')
        require(read_json(service / 'smoke-summary.json') == dict(passed=True, requests=44),
                f'Formal service restart validation failed: {backend}')
        require((service / 'npu-after.txt').is_file(), f'Missing device shutdown snapshot: {backend}')
        pid = int((service / 'server.pid').read_text().strip())
        proc = Path('/proc') / str(pid)
        if proc.exists():
            command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            require('vllm serve /data/models/DeepSeek-V4-Flash' not in command,
                    f'Owned model service is still running: {backend}/{pid}')

    expected_dirs = {run_root / 'aisbench' / b / p / f'round{n}'
                     for b in BACKENDS for p in POINTS for n in (1, 2, 3)}
    require({p.parent for p in (run_root / 'aisbench').rglob('status.json')} == expected_dirs,
            'Formal matrix must contain exactly the 63 requested rounds')
    measured = collections.defaultdict(list)
    for backend in BACKENDS:
        backend_root = run_root / 'aisbench' / backend
        require(read_json(backend_root / 'dataset-manifest.json') == manifest,
                f'Dataset manifest differs: {backend}')
        require(read_json(backend_root / 'checkpoint-manifest.json') == checkpoint,
                f'Checkpoint manifest differs: {backend}')
        for point, (dataset, concurrency, output, count, warmups) in POINTS.items():
            for round_number in (1, 2, 3):
                run = backend_root / point / f'round{round_number}'
                expected = dict(backend=backend, point=point, round=round_number,
                                concurrency=concurrency, output_tokens=output, requests=count,
                                warmups=warmups, aisbench_rc=0, dataset=dataset)
                require(read_json(run / 'status.json') == expected, f'Protocol differs: {run}')
                validation = read_json(run / 'validation.json')
                require(validation['passed'] and validation['failed'] == 0,
                        f'Round validation failed: {run}')
                files = list((run / 'aisbench').rglob('*_details.jsonl'))
                require(len(files) == 1, f'Measured result file count differs: {run}')
                details = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
                require(len(details) == count and all(row['success'] for row in details),
                        f'Request count or success differs: {run}')
                require(collections.Counter(row['output_tokens'] for row in details) == {output: count},
                        f'Fixed output length differs: {run}')
                inputs = {512: 160, 4096: 40} if point.startswith('mixed') else {
                    int(dataset.split('-')[-1].split('.')[0]): count}
                require(collections.Counter(row['input_tokens'] for row in details) == inputs,
                        f'Input distribution differs: {run}')
                measured[backend, point].append(summarize(run))

    with (run_root / 'summary/medians.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 21 and {(r['backend'], r['point']) for r in rows} == set(measured),
            'Median table does not cover exactly 21 backend/point combinations')
    for row in rows:
        rounds = measured[row['backend'], row['point']]
        require(int(row['measured_rounds']) == 3 and int(row['errors']) == 0
                and int(row['requests']) == sum(r['requests'] for r in rounds),
                f'Median table counts differ: {row}')
        for field in ('ttft_p50_ms', 'ttft_p95_ms', 'tpot_p50_ms', 'tpot_p95_ms',
                      'e2e_p50_ms', 'request_s', 'input_token_s', 'output_token_s'):
            values = [r[field] for r in rounds if r[field] is not None]
            if not values:
                require(row[field] == '', f'Undefined TPOT must be empty: {row}')
            else:
                actual = float(row[field])
                require(math.isfinite(actual) and actual > 0
                        and math.isclose(actual, statistics.median(values), rel_tol=1e-10),
                        f'Metric differs from raw three-round median: {row}/{field}')
    return dict(passed=True, formal_rounds=63, backend_point_combinations=21,
                measured_requests=sum(r['requests'] for rounds in measured.values() for r in rounds),
                errors=0, source_commits=len(sources['commits']),
                compiled_hashes=len(sources['libraries_sha256']),
                note='Device availability, numerical test scope and local archive SHA are also reviewed separately.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_root', type=Path)
    args = parser.parse_args()
    result = audit(args.run_root)
    (args.run_root / 'completion-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
