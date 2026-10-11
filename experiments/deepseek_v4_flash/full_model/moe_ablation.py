"""Independent four-rank diagnostic; never changes accepted backend sources."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu

p = argparse.ArgumentParser()
p.add_argument('--backend', choices=['catccos', 'deepmegamoe'], required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--library')
p.add_argument('--iterations', type=int, default=20)
p.add_argument('--hidden', type=int, default=4096)
p.add_argument('--intermediate', type=int, default=2048)
p.add_argument('--experts', type=int, default=256)
p.add_argument('--topk', type=int, default=6)
p.add_argument('--tokens', default='1,2,8,32,512,2048')
a = p.parse_args()
rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
torch.set_num_threads(1)
torch.npu.set_device(rank)
dev = torch.device(f'npu:{rank}')
dist.init_process_group('hccl' if a.backend == 'deepmegamoe' else 'gloo')
H, I, E, K = a.hidden, a.intermediate, a.experts // world, a.topk
# Constant packed weights keep the FP4/FP8 comparison bit-exact and avoid a
# multi-gigabyte CPU unpack. Scales are preserved; no re-quantization occurs.
w1 = torch.full((E, 2*I, H//2), 0x22, dtype=torch.uint8, device=dev)
w2 = torch.full((E, H, I//2), 0x22, dtype=torch.uint8, device=dev)
s1 = torch.full((E, 2*I, H//32), 120, dtype=torch.uint8, device=dev)
s2 = torch.full((E, H, I//32), 120, dtype=torch.uint8, device=dev)
if a.backend == 'catccos':
    torch.ops.load_library(a.library)
    assert torch.ops.catccos.init(rank, world, 1 << 30, 'tcp://127.0.0.1:27061') == 0
    fp4 = (w1.transpose(1, 2), s1.reshape(E, 2*I, H//64, 2).transpose(1, 2),
           w2.transpose(1, 2), s2.reshape(E, H, I//64, 2).transpose(1, 2))
    fp8 = (torch.full((E, 2*I, H), 0x38, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn), s1,
           torch.full((E, H, I), 0x38, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn), s2)
    modes = ['packed_fp4', 'preexpanded_fp8']
else:
    from vllm_ascend.ops.fused_moe import deepmegamoe_adapter as adapter
    import vllm.distributed
    vllm.distributed.get_ep_group = lambda: SimpleNamespace(device_group=dist.group.WORLD)
    adapter.get_ascend_config = lambda: SimpleNamespace(deepmegamoe_max_tokens_per_rank=4096)
    layer = torch.nn.Module()
    for key, value in [('w13_weight', w1), ('w2_weight', w2), ('w13_weight_scale', s1), ('w2_weight_scale', s2)]:
        setattr(layer, key, torch.nn.Parameter(value, requires_grad=False))
    adapter.prepare_deepmegamoe_weights(layer)
    cfg = SimpleNamespace(num_experts=a.experts, ep_size=world, hidden_dim=H,
                          intermediate_size_per_partition=I, experts_per_token=K)
    modes = ['direct_fp4']
results = []
try:
    for routing in ['spread', 'concentrated']:
        for M in map(int, a.tokens.split(',')):
            x = torch.full((M, H), .125, dtype=torch.bfloat16, device=dev)
            ids_cpu = ((torch.arange(M)[:, None]*K + rank*E + torch.arange(K)) % a.experts)
            if routing == 'concentrated':
                ids_cpu = (torch.arange(K) + rank*E).expand(M, -1).contiguous()
            ids = ids_cpu.to(dev, dtype=torch.int32)
            probs = torch.full((M, K), 1/K, dtype=torch.float32, device=dev)
            def launch(mode):
                if a.backend == 'catccos':
                    return torch.ops.catccos.ascend950_dispatch_ffn_combine(
                        x, ids, probs, *(fp4 if mode == 'packed_fp4' else fp8), 10.)
                return adapter.apply_deepmegamoe(x, ids, probs, layer.w13_weight,
                    layer.w13_weight_scale, layer.w2_weight, layer.w2_weight_scale, cfg, swiglu_limit=10)
            outputs = {}
            for mode in modes:
                for _ in range(5):
                    y = launch(mode)
                torch.npu.synchronize()
                outputs[mode] = y.cpu().float()
            for round_idx in range(3):
                # Alternate order to reduce a fixed-order clock/cache bias.
                for mode in modes if round_idx % 2 == 0 else list(reversed(modes)):
                    dist.barrier(); torch.npu.synchronize()
                    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    started = time.perf_counter(); begin.record()
                    for _ in range(a.iterations):
                        y = launch(mode)
                    end.record(); torch.npu.synchronize()
                    wall = (time.perf_counter()-started)*1000/a.iterations
                    device = begin.elapsed_time(end)/a.iterations
                    gathered = [None]*world
                    dist.all_gather_object(gathered, dict(rank=rank, wall_ms=wall, stream_ms=device))
                    if rank == 0:
                        row = dict(backend=a.backend, mode=mode, routing=routing, tokens_per_rank=M,
                                   round=round_idx+1, iterations=a.iterations, ranks=gathered,
                                   max_rank_wall_ms=max(z['wall_ms'] for z in gathered),
                                   max_rank_stream_ms=max(z['stream_ms'] for z in gathered))
                        results.append(row); print(json.dumps(row), flush=True)
            finite = all(bool(torch.isfinite(z).all()) for z in outputs.values())
            delta = float((outputs[modes[0]]-outputs[modes[-1]]).abs().max())
            assert finite and delta == 0, (rank, routing, M, finite, delta)
            if rank == 0:
                print(json.dumps(dict(numerical_check=True, routing=routing, M=M,
                                      fp4_vs_fp8_max_abs=delta, finite=finite)), flush=True)
                a.out.mkdir(parents=True, exist_ok=True)
                (a.out/'timings.json').write_text(json.dumps(results, indent=2))
finally:
    dist.barrier()
    if a.backend == 'catccos':
        torch.ops.catccos.finalize()
    else:
        adapter.release_deepmegamoe_buffers()
    dist.destroy_process_group()
