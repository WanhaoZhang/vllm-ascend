#!/usr/bin/env python3
"""Archive correctness smoke responses before accepting performance results."""
import argparse
import concurrent.futures
import json
import math
import time
import urllib.request
from pathlib import Path


def check_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise AssertionError('Nonfinite value in model response')
    if isinstance(value, dict):
        for item in value.values():
            check_finite(item)
    if isinstance(value, list):
        for item in value:
            check_finite(item)


def request(base, name, prompt, output_length, fixed=True):
    body = dict(model='deepseek-v4-flash', messages=[dict(role='user', content=prompt)],
                temperature=0, seed=42, max_tokens=output_length, ignore_eos=fixed,
                logprobs=True, top_logprobs=1, return_token_ids=True,
                skip_special_tokens=False)
    start = time.monotonic()
    result = dict(name=name, request=body)
    try:
        req = urllib.request.Request(base + '/v1/chat/completions',
                                     json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=900) as response:
            data = json.load(response)
        check_finite(data)
        choice = data['choices'][0]
        content = choice['message'].get('content') or choice['message'].get('reasoning_content') or ''
        tokens = choice.get('token_ids')
        if tokens and all(token == 0 for token in tokens):
            raise AssertionError('Model returned only BOS token IDs')
        if not content.strip():
            raise AssertionError('Model returned empty text')
        if name == 'arithmetic' and content.strip() != '4':
            raise AssertionError(f'Arithmetic smoke expected 4, got {content!r}')
        logprobs = (choice.get('logprobs') or {}).get('content')
        if not logprobs or any(not isinstance(item.get('logprob'), (int, float))
                              or not math.isfinite(item['logprob']) for item in logprobs):
            raise AssertionError('Model did not return finite generated-token log probabilities')
        if fixed and data['usage']['completion_tokens'] != output_length:
            raise AssertionError('Fixed output token count mismatch')
        result.update(passed=True, response=data)
    except Exception as error:
        result.update(passed=False, error=str(error))
    result['elapsed_seconds'] = time.monotonic() - start
    print(json.dumps({k: v for k, v in result.items() if k not in ('request', 'response')}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='http://127.0.0.1:28041')
    parser.add_argument('--datasets', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--stress', action='store_true', help='Also exercise long prefill and decode at concurrency 32')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    def archive(items):
        results.extend(items)
        (args.out / 'smoke-responses.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
        if not all(item['passed'] for item in items):
            raise SystemExit(1)
    archive([request(args.base, 'arithmetic', 'What is 2 + 2? Answer with the number only.', 32, False)])
    for length in (128, 2048, 4096):
        file = args.datasets / ('decode-128.jsonl' if length == 128 else f'prefill-{length}.jsonl')
        prompt = json.loads(file.read_text().splitlines()[0])['question']
        archive([request(args.base, f'input-{length}', prompt, 16 if length == 128 else 1)])
        if results[-1]['response']['usage']['prompt_tokens'] != length:
            raise AssertionError(f'Server input length differs from frozen {length} token workload')
    for concurrency in (8, 32):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            archive(list(pool.map(lambda index: request(args.base, f'c{concurrency}-{index}',
                         f'Write the number {index} and a short greeting.', 8), range(concurrency))))
    if args.stress:
        for name, filename, output_length in (
            ('long32', 'prefill-4096.jsonl', 8),
            ('decode32', 'decode-128.jsonl', 256),
            ('mixed32', 'mixed-512-4096-80-20.jsonl', 128),
        ):
            rows = [json.loads(line) for line in (args.datasets / filename).read_text().splitlines()]
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
                archive(list(pool.map(lambda index: request(args.base, f'{name}-{index}',
                             rows[index]['question'], output_length), range(32))))
    (args.out / 'smoke-summary.json').write_text(json.dumps(dict(passed=True, requests=len(results))))


if __name__ == '__main__':
    main()
