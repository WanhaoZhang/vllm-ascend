#!/usr/bin/env python3
"""Reject failed requests or changed token workloads before accepting a run."""
import argparse
import collections
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    status = json.loads((args.run / 'status.json').read_text())
    files = list((args.run / 'aisbench').rglob('*_details.jsonl'))
    if len(files) != 1:
        raise AssertionError(f'Expected one measured result file, found {len(files)}')
    details = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
    failures = [row for row in details if not row.get('success')]
    lengths = collections.Counter(row.get('input_tokens') for row in details)
    outputs = collections.Counter(row.get('output_tokens') for row in details)
    errors = []
    if len(details) != status['requests']:
        errors.append(f'Request count mismatch: {len(details)}')
    if failures:
        errors.append(f'Failed requests: {len(failures)}')
    if outputs != {status['output_tokens']: status['requests']}:
        errors.append(f'Output length mismatch: {dict(outputs)}')
    filename = status['dataset']
    if filename == 'mixed-512-4096-80-20.jsonl':
        expected = {512: 160, 4096: 40}
    else:
        length = int(filename.split('-')[-1].split('.')[0])
        expected = {length: status['requests']}
    if lengths != expected:
        errors.append(f'Input length distribution mismatch: {dict(lengths)}')
    result = dict(passed=not errors, errors=errors, requests=len(details),
                  failed=len(failures), input_lengths=dict(lengths), output_lengths=dict(outputs))
    (args.run / 'validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
