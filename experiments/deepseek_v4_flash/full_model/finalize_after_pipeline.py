#!/usr/bin/env python3
"""Finish server-side evidence collection after one identified pipeline exits."""
import argparse
import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path


def identity(pid):
    proc = Path('/proc') / str(pid)
    try:
        stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        return stat[19], command, stat[0]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_root', type=Path)
    parser.add_argument('pipeline_pid', type=int)
    args = parser.parse_args()
    run = args.run_root.resolve()
    root = run.parent.parent
    prefix = run.parent / (run.name + '-finalize')
    initial = identity(args.pipeline_pid)
    if initial is None or 'run_comparison.sh' not in initial[1] or str(run) not in initial[1]:
        raise RuntimeError('The specified comparison pipeline is not running')
    (Path(str(prefix) + '.identity.json')).write_text(json.dumps(dict(
        pipeline_pid=args.pipeline_pid, start_ticks=initial[0], command=initial[1]), indent=2))
    while True:
        current = identity(args.pipeline_pid)
        if current is None or current[0] != initial[0] or current[2] in ('Z', 'X'):
            break
        print('Waiting for identified pipeline', args.pipeline_pid, flush=True)
        time.sleep(55)
    try:
        subprocess.run(['/opt/aisbench-venv/bin/python', str(root / 'audit_completion.py'), str(run)],
                       check=True)
        npu = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=True)
        (run / 'final-npu-smi.txt').write_text(npu.stdout)
        (run / 'goal-checkpoint.json').write_text(json.dumps(dict(
            timestamp=time.time(), goal_status='active', formal_rounds=63,
            remaining='Review final device/numerical evidence and verify local archive delivery',
            pipeline_pid=args.pipeline_pid, pipeline_terminal=True), indent=2))
        archive = run.parent / (run.name + '-complete.tar.gz')
        # The archive lives outside the measured result tree.
        with tarfile.open(archive, 'w:gz') as bundle:
            bundle.add(run, arcname=run.name)
        metadata = dict(archive=str(archive), size=archive.stat().st_size,
                        sha256=hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest(),
                        formal_audit=read_audit(run),
                        note='Server evidence is collected; goal completion requires local delivery and final review.')
        Path(str(prefix) + '.archive.json').write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata), flush=True)
    except Exception:
        Path(str(prefix) + '.exit-code').write_text('1\n')
        raise
    Path(str(prefix) + '.exit-code').write_text('0\n')


def read_audit(run):
    return json.loads((run / 'completion-audit.json').read_text())


if __name__ == '__main__':
    main()
