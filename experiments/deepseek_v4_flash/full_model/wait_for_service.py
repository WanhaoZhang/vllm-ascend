#!/usr/bin/env python3
"""Wait for health only while the recorded model service is alive."""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('service', type=Path)
    parser.add_argument('--base', default='http://127.0.0.1:28041')
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args()
    pid = int((args.service / 'server.pid').read_text())
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            proc = Path(f'/proc/{pid}')
            state = (proc / 'stat').read_text().split(') ', 1)[1].split()[0]
            command = (proc / 'cmdline').read_bytes()
            if state == 'Z' or not (b'DeepSeek-V4-Flash' in command or b'/run_server.sh' in command):
                raise RuntimeError(f'Recorded model service {pid} exited')
        except FileNotFoundError:
            raise RuntimeError(f'Recorded model service {pid} exited') from None
        try:
            with urllib.request.urlopen(args.base + '/health', timeout=2) as response:
                if response.status == 200:
                    (args.service / 'service-ready.json').write_text(json.dumps(dict(pid=pid, timestamp=time.time())))
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(3)
    raise TimeoutError(f'Service did not become ready within {args.timeout}s; inspect {args.service}/server.log')


if __name__ == '__main__':
    main()
