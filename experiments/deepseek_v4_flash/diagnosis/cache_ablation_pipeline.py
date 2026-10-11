import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path('/home/z009565592/catccos-vllm/work')
DIAG=ROOT/'deepseek-v4-diagnosis-20261011'
TEST=ROOT/'deepseek-v4-test'
deadline=time.monotonic()+1800
while True:
    p=DIAG/'profile2-status.json'
    if p.exists():
        status=json.loads(p.read_text())
        if len(status)==3 and all(x['client_rc']==0 for x in status): break
    if time.monotonic()>deadline: raise RuntimeError('Profiling pipeline did not complete')
    time.sleep(5)

# Hardlink unchanged files, then unlink ONLY the adapter in this diagnostic
# copy before modifying it. The original Git worktree is never written.
src=ROOT/'v4-catccos-vllm'
copy=DIAG/'cached-vllm'
if not copy.exists(): subprocess.run(['cp','-al',str(src),str(copy)],check=True)
adapter=copy/'vllm_ascend/ops/fused_moe/catccos_adapter.py'
original=(src/'vllm_ascend/ops/fused_moe/catccos_adapter.py').read_text()
needle='    with moe_profile_range("catccos", "adapter_host_scope", hidden_states, topk_ids):'
assert original.count(needle)==1
patched=original.replace(needle,
    '    from catccos_cache_diagnostic import exact_cache\n'
    '    w1, w1_scale, w2, w2_scale = exact_cache(w1, w1_scale, w2, w2_scale)\n'+needle)
adapter.unlink()
adapter.write_text(patched)
assert (src/'vllm_ascend/ops/fused_moe/catccos_adapter.py').read_text()==original
runner=(DIAG/'run_profiled.sh').read_text()
runner=runner.replace('$inside/work/v4-${backend}-vllm:/vllm-workspace/vllm',
                      str(copy)+':'+str(DIAG)+':/vllm-workspace/vllm')
(DIAG/'run_cached.sh').write_text(runner)
for cache_layers in [0,8]:
    service=DIAG/f'cache{cache_layers}-catccos'
    service.mkdir(exist_ok=True)
    env=dict(os.environ,RESULT_DIR=str(service),PROFILE_DIR=str(service/'traces'),
             DIAGNOSTIC_CACHE_LAYERS=str(cache_layers),VLLM_ASCEND_MOE_PROFILE_RANGES='0',MSMONITOR_USE_DAEMON='0')
    child=subprocess.Popen(['bash',str(DIAG/'run_cached.sh'),'catccos'],env=env)
    (service/'server.pid').write_text(str(child.pid))
    time.sleep(10)
    rc=None
    try:
        subprocess.run(['python3',str(DIAG/'wait_for_service.py'),str(service),'--timeout','600'],check=True)
        with (service/'client.log').open('w') as log:
            rc=subprocess.run(['python3',str(DIAG/'profile_client.py'),'--out',str(service),
                               '--datasets',str(TEST/'datasets-dsv4-20261011'),'--measure-only'],stdout=log,stderr=subprocess.STDOUT).returncode
    finally:
        subprocess.run(['python3',str(TEST/'stop_server.py'),str(service)],check=True)
        child.wait(timeout=40)
        audit=dict(client_rc=rc, server_rc=child.returncode,
                   original_adapter_sha256=hashlib.sha256(original.encode()).hexdigest(),
                   diagnostic_adapter_sha256=hashlib.sha256(patched.encode()).hexdigest(),
                   original_unchanged=(src/'vllm_ascend/ops/fused_moe/catccos_adapter.py').read_text()==original)
        (service/'ablation-status.json').write_text(json.dumps(audit,indent=2))
    if rc: raise RuntimeError(f'Cache ablation client failed: {rc}')
