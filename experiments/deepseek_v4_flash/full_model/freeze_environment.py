#!/usr/bin/env python3
"""Record the resolved environment and dirty source used by a service."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
from pathlib import Path


def command(args):
    result = subprocess.run(args, capture_output=True, text=True)
    return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path('/home/z00956592/catccos-vllm/work')
    sources = dict(vllm_ascend=root / f'v4-{args.backend}-vllm',
                   catccos=root / 'v4-catccos-kernel')
    if args.backend == 'deepmegamoe':
        sources['deepgemm'] = Path(os.environ['DEEPMEGAMOE_SOURCE'])
    repos = {}
    for name, source in sources.items():
        git = ['git', '-c', 'safe.directory=*', '-C', str(source)]
        repos[name] = dict(path=str(source), head=command(git + ['rev-parse', 'HEAD']),
                          branch=command(git + ['branch', '--show-current']),
                          status=command(git + ['status', '--short', '--ignore-submodules=all']))
        patch = command(git + ['diff', '--binary', '--no-ext-diff', '--ignore-submodules=all'])
        if patch['returncode']:
            raise RuntimeError(f'Cannot freeze {name}: {patch}')
        (args.out / f'{name}-dirty.patch').write_text(patch['stdout'])
        untracked = command(git + ['ls-files', '--others', '--exclude-standard'])
        repos[name]['untracked_sha256'] = {
            path: digest(source / path) for path in untracked['stdout'].splitlines()
            if (source / path).is_file()
        }
    versions = {}
    for package in ('torch', 'torch-npu', 'vllm', 'vllm-ascend', 'transformers'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    modules = {}
    for name in ('torch', 'torch_npu', 'vllm', 'vllm_ascend'):
        spec = importlib.util.find_spec(name)
        modules[name] = None if spec is None else spec.origin
    libraries = list((root / 'deepseek-v4-test/runtime').glob('*.so.*'))
    libraries += list((root / 'v4-catccos-kernel/build_torch_a5/lib').glob('*.so'))
    custom_opp = Path(os.environ['ASCEND_CUSTOM_OPP_PATH'].split(':')[0])
    libraries += list((custom_opp / 'op_api/lib').glob('*.so'))
    if args.backend == 'deepmegamoe':
        libraries += list((sources['deepgemm'] / 'deep_gemm').glob('_C*.so'))
    build_dependencies = {}
    if args.backend == 'deepmegamoe':
        for dependency in ('deep-jit', 'magic_enum'):
            include = sources['deepgemm'] / 'third-party' / dependency / 'include'
            build_dependencies[dependency] = {
                str(p.relative_to(include)): digest(p) for p in include.rglob('*') if p.is_file()
            }
    metadata = dict(backend=args.backend, versions=versions, modules=modules, sources=repos,
                    libraries_sha256={str(p): digest(p) for p in libraries},
                    build_dependencies_sha256=build_dependencies,
                    custom_opp_sha256={str(p.relative_to(custom_opp)): digest(p)
                                       for p in (custom_opp / 'op_impl/ai_core/tbe/kernel').rglob('*')
                                       if p.is_file() and p.suffix in ('.json', '.o')},
                    environment={name: os.environ.get(name) for name in (
                        'ASCEND_HOME_PATH', 'ASCEND_RT_VISIBLE_DEVICES', 'PYTHONPATH', 'LD_LIBRARY_PATH',
                        'TASK_QUEUE_ENABLE', 'HCCL_BUFFSIZE', 'HCCL_OP_EXPANSION_MODE')},
                    npu=command(['npu-smi', 'info']),
                    compiler=command(['/usr/local/Ascend/ascend-toolkit/latest/bin/bisheng', '--version']))
    (args.out / 'environment.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
