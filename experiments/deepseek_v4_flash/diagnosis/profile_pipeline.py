"""Sequential owned-service profiler; always releases each service."""
import json
import os
import subprocess
import time
from pathlib import Path
ROOT=Path('/home/z009565592/catccos-vllm/work')
TEST=ROOT/'deepseek-v4-test'
DIAG=ROOT/'deepseek-v4-diagnosis-20261011'
status=[]
for backend in ['catccos','deepmegamoe','native']:
    service=DIAG/f'profile2-{backend}'
    service.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,RESULT_DIR=str(service),PROFILE_DIR=str(service/'traces'),
             VLLM_ASCEND_MOE_PROFILE_RANGES='1',MSMONITOR_USE_DAEMON='0')
    if backend=='catccos' and (service/'server.pid').exists() and Path('/proc/'+(service/'server.pid').read_text().strip()).exists():
        child=None
    else:
        child=subprocess.Popen(['bash',str(DIAG/'run_profiled.sh'),backend],env=env)
        (service/'server.pid').write_text(str(child.pid))
        time.sleep(10)
    rc=None
    try:
        subprocess.run(['python3',str(DIAG/'wait_for_service.py'),str(service),'--timeout','600'],check=True)
        with (service/'client.log').open('w') as output:
            proc=subprocess.run(['python3',str(DIAG/'profile_client.py'),'--out',str(service),
                                  '--datasets',str(TEST/'datasets-dsv4-20261011'),'--measure-only'],stdout=output,stderr=subprocess.STDOUT)
        rc=proc.returncode
        if rc==0:
            with (service/'probe.log').open('w') as probe:
                probe_rc=subprocess.run(['python3',str(DIAG/'profile_probe.py'),'--out',str(service),'--datasets',str(TEST/'datasets-dsv4-20261011')],stdout=probe,stderr=subprocess.STDOUT).returncode
            (service/'probe.exit-code').write_text(str(probe_rc))
    finally:
        subprocess.run(['python3',str(TEST/'stop_server.py'),str(service)],check=True)
        if child is not None: child.wait(timeout=40)
        status.append(dict(backend=backend,client_rc=rc,server_rc=child.returncode if child is not None else None))
        (DIAG/'profile2-status.json').write_text(json.dumps(status,indent=2))
    if rc!=0: raise RuntimeError(status[-1])
