#!/usr/bin/env python3
"""Summarize measured requests, then take medians across three rounds."""
import argparse
import csv
import io
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(run):
    status = json.loads((run / 'status.json').read_text())
    validation = json.loads((run / 'validation.json').read_text())
    if status['aisbench_rc'] or not validation['passed']:
        raise AssertionError(f'Cannot accept failed run: {run}')
    details_path, = (run / 'aisbench').rglob('*_details.jsonl')
    performance_dir = details_path.parent
    databases = {}
    for database in (performance_dir / 'db_data').glob('*.db'):
        with sqlite3.connect(database) as connection:
            databases[database.name] = {
                key: np.load(io.BytesIO(blob), allow_pickle=False).ravel()
                for key, blob in connection.execute('SELECT id, arr_blob FROM numpy_store')
            }
    ttft, tpot, e2e = [], [], []
    for line in details_path.read_text().splitlines():
        item = json.loads(line)
        if not item['success']:
            raise AssertionError(f'Failed request: {run}')
        points = item['time_points']
        if isinstance(points, dict):
            database = item.get('db_name')
            if database is None and len(databases) == 1:
                database = next(iter(databases))
            points = databases[database][points['__db_ref__']]
        if len(points) < 2 or not np.isfinite(points).all():
            raise AssertionError(f'Missing/invalid request timestamps: {run}')
        first = float(points[1] - points[0])
        total = float(points[-1] - points[0])
        ttft.append(first * 1000)
        e2e.append(total * 1000)
        if item['output_tokens'] > 1:
            tpot.append((total - first) / (item['output_tokens'] - 1) * 1000)
    summary = None
    for file in performance_dir.glob('*.json'):
        candidate = json.loads(file.read_text())
        if 'Benchmark Duration' in candidate:
            summary = candidate
            break
    if summary is None:
        raise AssertionError(f'Missing AISBench performance summary: {run}')
    def number(name):
        value = summary[name]['total']
        return float(str(value).split()[0])
    return dict(backend=status['backend'], point=status['point'], round=status['round'],
                concurrency=status['concurrency'], output_tokens=status['output_tokens'],
                requests=len(ttft), errors=validation['failed'],
                ttft_p50_ms=statistics.median(ttft), ttft_p95_ms=float(np.quantile(ttft, .95)),
                tpot_p50_ms=statistics.median(tpot) if tpot else None,
                tpot_p95_ms=float(np.quantile(tpot, .95)) if tpot else None,
                e2e_p50_ms=statistics.median(e2e), request_s=number('Request Throughput'),
                input_token_s=number('Input Token Throughput'), output_token_s=number('Output Token Throughput'))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_full_report(path, rows):
    backends = ('catccos', 'deepmegamoe', 'native')
    points = ('decode-c1', 'decode-c8', 'decode-c32', 'prefill-2048',
              'prefill-4096', 'mixed-c8', 'mixed-c32')
    indexed = {(row['backend'], row['point']): row for row in rows}
    expected = {(backend, point) for backend in backends for point in points}
    if set(indexed) != expected:
        return  # Partial summaries must not claim the full matrix completed.
    assert len(rows) == 21
    assert all(row['measured_rounds'] == 3 and row['errors'] == 0 for row in rows)
    assert all(row['requests'] == (96 if row['point'] == 'decode-c1' else 600) for row in rows)
    names = dict(catccos='CatCCOS', deepmegamoe='deepMegaMoE', native='native')
    lines = [
        '# DeepSeek-V4-Flash 三路径 AISBench 对比', '',
        f"共63轮，{sum(row['requests'] for row in rows)}个测量请求，0错误。预热不计入成绩。", '',
        '三轮中位数；TTFT/TPOT 是每轮请求P50的中位数。prefill输出1，TPOT无定义。', '',
        '| 负载 | 后端 | TTFT ms | TPOT ms | 输入 token/s | 输出 token/s | 相对native吞吐 | 请求数 |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for point in points:
        field = 'input_token_s' if point.startswith('prefill') else 'output_token_s'
        baseline = indexed['native', point][field]
        assert baseline > 0
        for backend in backends:
            row = indexed[backend, point]
            tpot = '-' if row['tpot_p50_ms'] is None else f"{row['tpot_p50_ms']:.2f}"
            lines.append(f"| {point} | {names[backend]} | {row['ttft_p50_ms']:.2f} | {tpot} | "
                         f"{row['input_token_s']:.4f} | {row['output_token_s']:.4f} | "
                         f"{row[field] / baseline:.3f}x | {row['requests']} |")
    lines += ['', '归一化吞吐：prefill使用输入吞吐，其余使用输出吞吐。', '',
              '统一设置：NPUs0–3、DP4/TP1/EP4、eager、MTP关闭、prefix cache关闭、',
              'max-model-len8192、batched-tokens4096、max-seqs32、GPU memory utilization0.80、',
              'HCCL_BUFFSIZE1536、DSV4 DSA attention overlap关闭。temperature0、seed42。', '',
              'decode-c1每轮32请求/4预热；其余每轮200请求/16预热。',
              'mixed严格160条512-token输入、40条4096-token输入，输出128。', '',
              'CatCCOS保留官方MXFP4/E8M0，每次调用精确展开FP4→FP8，展开开销计入成绩。',
              'deepMegaMoE在中间量化前应用路由权重；CatCCOS在第二个linear后应用。',
              '三路径各140请求整模smoke已通过，但长greedy输出不保证相同；这些结果不等于任务精度。', '',
              '逐轮数据见rounds.csv；三轮中位数见medians.csv；完整复现及数值检查见实验目录REPRODUCE.md。']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    rounds = [summarize(file.parent) for file in sorted(args.root.rglob('status.json'))]
    if not rounds:
        raise SystemExit('No measured AISBench rounds found')
    groups = defaultdict(list)
    for row in rounds:
        groups[row['backend'], row['point']].append(row)
    medians = []
    for (backend, point), rows in groups.items():
        if sorted(row['round'] for row in rows) != [1, 2, 3]:
            raise AssertionError(f'Expected three independent rounds: {backend}/{point}')
        result = dict(backend=backend, point=point, concurrency=rows[0]['concurrency'],
                      output_tokens=rows[0]['output_tokens'], measured_rounds=3,
                      requests=sum(row['requests'] for row in rows), errors=sum(row['errors'] for row in rows))
        for key in ('ttft_p50_ms', 'ttft_p95_ms', 'tpot_p50_ms', 'tpot_p95_ms', 'e2e_p50_ms',
                    'request_s', 'input_token_s', 'output_token_s'):
            values = [row[key] for row in rows if row[key] is not None]
            result[key] = statistics.median(values) if values else None
        medians.append(result)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'rounds.csv', rounds)
    write_csv(args.out / 'medians.csv', medians)
    write_full_report(args.out / 'report.md', medians)
    (args.out / 'metrics-definition.json').write_text(json.dumps(dict(
        units='Latency in milliseconds; throughput in requests/tokens per second',
        ttft='first response timestamp minus request start',
        tpot='(last timestamp - first timestamp)/(output tokens - 1); undefined for output=1',
        median='Median of three per-round statistics; requests/errors summed across rounds'), indent=2))


if __name__ == '__main__':
    main()
