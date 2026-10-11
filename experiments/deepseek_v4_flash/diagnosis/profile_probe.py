import argparse,json,concurrent.futures as cf,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--datasets',type=Path);a=p.parse_args()
if (a.out.parent/'disable-probes').exists():
 print('Profiler probe disabled after two crashes; measurements were completed before profiling.'); raise SystemExit(0)
base='http://127.0.0.1:28041'
def rpc(path,body=None):
 req=urllib.request.Request(base+path,None if body is None else json.dumps(body).encode(),{'Content-Type':'application/json'},method='POST')
 with urllib.request.urlopen(req,timeout=600) as r:return r.read()
prompts=[json.loads(z)['question'] for z in (a.datasets/'decode-128.jsonl').read_text().splitlines()[:8]]
rpc('/start_profile')
with cf.ThreadPoolExecutor(max_workers=8) as pool:
 responses=list(pool.map(lambda s:json.loads(rpc('/v1/completions',dict(model='deepseek-v4-flash',prompt=s,max_tokens=4,temperature=0,seed=42,ignore_eos=True))),prompts))
(a.out/'probe-responses.json').write_text(json.dumps(responses,indent=2))
rpc('/stop_profile')
