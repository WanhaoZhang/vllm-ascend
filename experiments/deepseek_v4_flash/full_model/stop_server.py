#!/usr/bin/env python3
"""Stop only the recorded test service and its captured process tree."""
import argparse
import json
import os
import re
import signal
import time
from pathlib import Path


def process(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        return dict(pid=pid, state=fields[0], parent=int(fields[1]), start=fields[19], command=command)
    except (FileNotFoundError, ProcessLookupError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('service_dir', type=Path)
    args = parser.parse_args()
    pid = int((args.service_dir / 'server.pid').read_text())
    root = process(pid)
    root_alive = root is not None and root['state'] != 'Z'
    if root_alive and ('vllm' not in root['command'] or 'DeepSeek-V4-Flash' not in root['command']):
        raise SystemExit(f'Recorded PID no longer matches the test service: {root}')
    inventory = {int(path.name): process(int(path.name)) for path in Path('/proc').iterdir() if path.name.isdigit()}
    owned = {pid: root} if root_alive else {}
    # Fatal engine errors can reparent workers before the launcher exits.
    # Include only worker PIDs recorded in this service's own log, and verify
    # their current process title before signaling any reparented process.
    log = (args.service_dir / 'server.log').read_text(errors='replace')
    recorded_workers = {int(match) for match in re.findall(r'\(Worker(?:_[A-Za-z0-9]+)* pid=(\d+)\)', log)}
    for worker_pid in recorded_workers:
        current = inventory.get(worker_pid)
        if current and current['state'] != 'Z' and current['command'].startswith('VLLM::Worker_DP'):
            owned[worker_pid] = current
    while True:
        children = {key: value for key, value in inventory.items()
                    if value and value['parent'] in owned and key not in owned}
        if not children:
            break
        owned.update(children)
    (args.service_dir / 'stopped-processes.json').write_text(json.dumps(list(owned.values()), indent=2))
    if root_alive:
        os.kill(pid, signal.SIGTERM)
    else:
        for worker_pid in owned:
            try:
                os.kill(worker_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        snapshot = [process(key) for key in owned]
        if all(current is None or current['state'] == 'Z' for current in snapshot):
            break
        time.sleep(1)
    for key, saved in owned.items():
        current = process(key)
        if current and current['state'] != 'Z' and current['start'] == saved['start']:
            print(f'Killing remaining test process {key}: {saved["command"]}', flush=True)
            try:
                os.kill(key, signal.SIGKILL)
            except ProcessLookupError:
                pass


if __name__ == '__main__':
    main()
