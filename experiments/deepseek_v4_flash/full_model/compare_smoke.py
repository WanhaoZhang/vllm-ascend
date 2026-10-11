#!/usr/bin/env python3
"""Compare archived greedy generations; this is not a task accuracy metric."""
import argparse
import json
from pathlib import Path


def load(path):
    rows = json.loads((path / 'smoke-responses.json').read_text())
    assert len(rows) == 140 and all(row['passed'] for row in rows), path
    result = {row['name']: row for row in rows}
    assert len(result) == len(rows), path
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', required=True, type=Path)
    parser.add_argument('--catccos', required=True, type=Path)
    parser.add_argument('--deepmegamoe', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    native = load(args.native)
    report = {'definition': 'Same-position token agreement and exact sequence agreement '
                           'on archived greedy smoke requests. Concurrent batch scheduling '
                           'can change numerics; this is not task accuracy.', 'comparisons': {}}
    for backend in ('catccos', 'deepmegamoe'):
        candidate = load(getattr(args, backend))
        assert candidate.keys() == native.keys()
        details = []
        for name, reference in native.items():
            other = candidate[name]
            assert reference['request'] == other['request'], name
            expected = reference['response']['choices'][0]['token_ids']
            actual = other['response']['choices'][0]['token_ids']
            assert isinstance(expected, list) and isinstance(actual, list), name
            assert len(expected) == len(actual) > 0, name
            matched = sum(a == b for a, b in zip(expected, actual))
            prefix = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), len(expected))
            details.append(dict(name=name, tokens=len(expected), matched=matched,
                                common_prefix=prefix, exact=expected == actual))
        report['comparisons'][backend] = dict(requests=len(details),
            exact_requests=sum(row['exact'] for row in details),
            matched_tokens=sum(row['matched'] for row in details),
            total_tokens=sum(row['tokens'] for row in details), details=details)
    args.out.write_text(json.dumps(report, indent=2))
    for backend, result in report['comparisons'].items():
        print(backend, {key: value for key, value in result.items() if key != 'details'})


if __name__ == '__main__':
    main()
