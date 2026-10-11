import csv
import hashlib
import json
import statistics as st
from pathlib import Path

ROOT=Path(__file__).resolve().parent
data={}
for service in ['profile2-catccos','profile2-deepmegamoe','profile2-native','cache0-catccos','cache8-catccos']:
    p=ROOT/service/'client-results.json'
    if not p.exists(): continue
    rows=json.loads(p.read_text()); points={}
    for point in sorted({r['point'] for r in rows}):
        rounds=[r for r in rows if r['point']==point]
        assert len(rounds)==3
        measurements=[m for r in rounds for m in r['measurements']]
        isl={'decode-c1':128,'decode-c8':128,'prefill2048':2048,'prefill4096':4096}[point]
        osl=32 if point.startswith('decode') else 1
        for m in measurements:
            assert m['usage']['prompt_tokens']==isl and m['usage']['completion_tokens']==osl
            assert m['chunks']==osl, (service,point,m)
        points[point]=dict(requests=len(measurements),
            ttft_ms=st.median(st.median(m['ttft_ms'] for m in r['measurements']) for r in rounds),
            mean_token_interval_ms=st.median(st.median(m['mean_chunk_interval_ms'] for m in r['measurements']) for r in rounds),
            ttft_round_medians=[st.median(m['ttft_ms'] for m in r['measurements']) for r in rounds],
            interval_round_medians=[st.median(m['mean_chunk_interval_ms'] for m in r['measurements']) for r in rounds])
    data[service]=points
micro={}
for b in ['catccos','deepmegamoe']:
    rows=json.loads((ROOT/f'micro-{b}'/'timings.json').read_text())
    micro[b]=[]
    for route in ['spread','concentrated']:
        for m in [1,2,8,32,512,2048]:
            modes=sorted({r['mode'] for r in rows})
            for mode in modes:
                rounds=[r for r in rows if r['routing']==route and r['tokens_per_rank']==m and r['mode']==mode]
                assert len(rounds)==3
                micro[b].append(dict(routing=route,tokens_per_rank=m,mode=mode,
                                      wall_ms=st.median(r['max_rank_wall_ms'] for r in rounds),
                                      stream_ms=st.median(r['max_rank_stream_ms'] for r in rounds)))
equality=[]
if all(s in data for s in ['cache0-catccos','cache8-catccos']):
    base=json.loads((ROOT/'cache0-catccos/client-results.json').read_text())
    cache=json.loads((ROOT/'cache8-catccos/client-results.json').read_text())
    for x,y in zip(base,cache,strict=True):
        assert x['point']==y['point'] and x['round']==y['round']
        for i,(a,b) in enumerate(zip(x['measurements'],y['measurements'],strict=True)):
            equality.append(dict(point=x['point'],round=x['round'],request_index=i,
                                 equal_text_sha256=a['text_sha256']==b['text_sha256']))
    for c in [0,8]:
        audit=json.loads((ROOT/f'cache{c}-catccos/ablation-status.json').read_text())
        assert audit['client_rc']==0 and audit['original_unchanged']
    logs=(ROOT/'cache8-catccos/server.log').read_text(errors='replace')
    parsed=[json.loads(line.split('DIAGNOSTIC_EXACT_CACHE ',1)[1]) for line in logs.splitlines() if 'DIAGNOSTIC_EXACT_CACHE ' in line]
    assert len(parsed)==32 and all(r['exact_mapping'] for r in parsed),len(parsed)
    for device in ['npu:0','npu:1','npu:2','npu:3']:
        assert sorted(r['layer'] for r in parsed if r['device']==device)==list(range(1,9))
summary=dict(services=data,micro=micro,cache_text_checks=equality,
             cache_equal_text_requests=sum(e['equal_text_sha256'] for e in equality),
             cache_text_requests=len(equality),
             profiler_usable=False,
             profiler_limitation='Worker segmentation faults at profiler stop; offline export lacks TASK table. No device trace timing is used.',
             notes=['profile2 services enable MoE ranges; use cache0/cache8 (ranges disabled) for causal timing.',
                    'Streaming interval is an average over exactly 32 output-token chunks, not AISBench pooled ITL P50.',
                    'Synthetic weights are constant, routes are controlled, and repeated weights may be hot in cache.',
                    'DeepMegaMoE synthetic self-comparison is only a finiteness check; it is not cross-backend equality.'])
real=[row for p in (ROOT/'real-weight-equivalence').glob('rank-*.json') for row in json.loads(p.read_text())]
assert len(real)==16 and all(z['passed'] for z in real),real
summary['real_weight_equivalence']=dict(checks=len(real),passed=True,max_abs=max(z['packed_vs_cached_max_abs'] for z in real))
summary['concurrent_text_variability']={}
for name in ['cache0-catccos','cache8-catccos']:
 rows=[z for z in json.loads((ROOT/name/'client-results.json').read_text()) if z['point']=='decode-c8']
 summary['concurrent_text_variability'][name]=sum(len({z['measurements'][i]['text_sha256'] for z in rows})>1 for i in range(8))
(ROOT/'diagnosis-summary.json').write_text(json.dumps(summary,indent=2))
files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
       for p in ROOT.rglob('*') if p.is_file() and 'cached-vllm' not in p.parts and 'traces' not in p.parts and '__pycache__' not in p.parts
       and p.name not in ['artifact-sha256.json','diagnosis-evidence.tar.gz','archive-manifest.json']}
(ROOT/'artifact-sha256.json').write_text(json.dumps(files,indent=2))
print(json.dumps(summary,indent=2))
