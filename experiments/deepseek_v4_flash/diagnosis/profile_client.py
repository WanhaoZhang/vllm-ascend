import argparse
import concurrent.futures as cf
import json
import hashlib
import time
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--datasets', type=Path, required=True)
p.add_argument('--measure-only',action='store_true')
p.add_argument('--base', default='http://127.0.0.1:28041')
a = p.parse_args()
def rpc(path, body=None):
    req = urllib.request.Request(a.base+path, None if body is None else json.dumps(body).encode(),
                                 {'Content-Type':'application/json'}, method='POST')
    return urllib.request.urlopen(req, timeout=900)
def request(prompt, output):
    body = dict(model='deepseek-v4-flash', messages=[dict(role='user',content=prompt)], temperature=0, seed=42,
                max_tokens=output, ignore_eos=True, stream=True, stream_options=dict(include_usage=True))
    t0 = time.perf_counter(); arrivals = []; final_usage = None; text = ''
    with rpc('/v1/chat/completions', body) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            item = json.loads(line[6:])
            if item.get('usage'): final_usage = item['usage']
            for choice in item.get('choices', []):
                delta=choice.get('delta') or {}
                piece=delta.get('content') or delta.get('reasoning_content') or ''
                if piece:
                    arrivals.append(time.perf_counter()); text += piece
    assert text and arrivals
    return dict(elapsed_ms=(time.perf_counter()-t0)*1000, ttft_ms=(arrivals[0]-t0)*1000,
                mean_chunk_interval_ms=(arrivals[-1]-arrivals[0])*1000/max(len(arrivals)-1,1),
                chunks=len(arrivals), requested_output=output, usage=final_usage, text_sha256=hashlib.sha256(text.encode()).hexdigest())
records=[]
for name, isl, output, concurrency in [('decode-c1',128,32,1), ('decode-c8',128,32,8),
                                       ('prefill2048',2048,1,1), ('prefill4096',4096,1,1)]:
    dataset=a.datasets/('decode-128.jsonl' if isl==128 else f'prefill-{isl}.jsonl')
    prompts=[json.loads(z)['question'] for z in dataset.read_text().splitlines()[:16]]
    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for warm in range(2): list(pool.map(lambda s:request(s,output),prompts[:concurrency]))
        for rnd in range(3):
            measurements=list(pool.map(lambda s:request(s,output),prompts[:concurrency]))
            row=dict(point=name, round=rnd+1, measurements=measurements, profiled=False)
            records.append(row); print(json.dumps(row),flush=True)
        a.out.mkdir(parents=True,exist_ok=True)
        (a.out/'client-results.json').write_text(json.dumps(records,indent=2))
        if a.measure_only: continue
        rpc('/start_profile').close()
        if output == 1:
            for prompt in prompts[:6]: request(prompt,output)
        else:
            list(pool.map(lambda s:request(s,output),prompts[:concurrency]))
        rpc('/stop_profile').close()
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'client-results.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(dict(profile_finished=name)),flush=True)
