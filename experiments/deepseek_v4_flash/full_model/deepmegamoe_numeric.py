#!/usr/bin/env python3
"""Four-rank adapter tests with independently decoded checkpoint weights.

MegaMoE applies routing weights before the intermediate MXFP8 quantization;
the reference follows that order, including BF16 rounding and clamp=10.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu

from catccos_mxfp4_numeric import dequantize_fp4, round_mxfp8
from vllm_ascend.ops.fused_moe import deepmegamoe_adapter as adapter


def round_intermediate_mxfp8(matrix):
    """MegaMoE quantizes FP32 SwiGLU directly, without a BF16 intermediate."""
    groups = matrix.float().reshape(matrix.shape[0], -1, 32)
    amax = groups.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448)))
    values = (groups / scale).to(torch.float8_e4m3fn).float() * scale
    return values.reshape(matrix.shape)


def run_case(rank, world, device, name, hidden, intermediate, experts, capacity):
    random = name not in ('constant', 'zero')
    m = {'decode': 1, 'batch': 32, 'prefill': capacity, 'constant': capacity,
         'zero': 0, 'ragged': (0, 1, 17, 32)[rank]}[name]
    topk = min(6, experts)
    gen = torch.Generator().manual_seed(42)
    local = experts // world
    section = slice(rank * local, (rank + 1) * local)
    shape1 = (experts, 2 * intermediate, hidden // 2)
    shape2 = (experts, hidden, intermediate // 2)
    if random:
        w1 = torch.randint(0, 256, shape1, generator=gen, dtype=torch.uint8)
        w2 = torch.randint(0, 256, shape2, generator=gen, dtype=torch.uint8)
        s1 = torch.randint(119, 123, (experts, 2 * intermediate, hidden // 32), generator=gen, dtype=torch.uint8)
        s2 = torch.randint(119, 123, (experts, hidden, intermediate // 32), generator=gen, dtype=torch.uint8)
        x = torch.randn((m, hidden), generator=gen).mul_(.15).to(torch.bfloat16)
    else:
        w1 = torch.full(shape1, 0x22, dtype=torch.uint8)
        w2 = torch.full(shape2, 0x22, dtype=torch.uint8)
        s1 = torch.full((experts, 2 * intermediate, hidden // 32), 126, dtype=torch.uint8)
        s2 = torch.full((experts, hidden, intermediate // 32), 120, dtype=torch.uint8)
        x = torch.full((m, hidden), .125, dtype=torch.bfloat16)
    ids = (torch.arange(m)[:, None] + rank * local + torch.arange(topk)) % experts
    probs = torch.arange(1, topk + 1).float().expand(m, -1).contiguous()
    probs /= probs.sum(-1, keepdim=True)
    reference = torch.zeros((m, hidden))
    if m:
        rounded_x = round_mxfp8(x, device)
        if random:
            for expert in ids.unique().tolist():
                rows, slots = (ids == expert).nonzero(as_tuple=True)
                gate_up = (rounded_x[rows] @ dequantize_fp4(w1[expert], s1[expert]).T).to(torch.bfloat16).float()
                gate, up = gate_up.chunk(2, -1)
                act = torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
                act = act * probs[rows, slots, None]
                rounded_act = round_intermediate_mxfp8(act)
                result = (rounded_act @ dequantize_fp4(w2[expert], s2[expert]).T).to(torch.bfloat16).float()
                reference.index_add_(0, rows, result)
        else:
            gate_up = (rounded_x[:1] @ dequantize_fp4(w1[0], s1[0]).T).to(torch.bfloat16).float()
            gate, up = gate_up.chunk(2, -1)
            act = torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
            for slot in range(topk):
                weighted = act * ((slot + 1) / (topk * (topk + 1) / 2))
                result = (round_intermediate_mxfp8(weighted) @ dequantize_fp4(w2[0], s2[0]).T).to(torch.bfloat16).float()
                reference += result
    reference = reference.to(torch.bfloat16).float()
    layer = torch.nn.Module()
    for key, value in [('w13_weight', w1[section]), ('w2_weight', w2[section]),
                       ('w13_weight_scale', s1[section]), ('w2_weight_scale', s2[section])]:
        setattr(layer, key, torch.nn.Parameter(value.to(device), requires_grad=False))
    adapter.prepare_deepmegamoe_weights(layer)
    config = SimpleNamespace(num_experts=experts, ep_size=world, hidden_dim=hidden,
                             intermediate_size_per_partition=intermediate, experts_per_token=topk)
    x_device = x.to(device)
    ids_device, probs_device = ids.to(device, dtype=torch.int32), probs.to(device)
    outputs = []
    for _ in range(4):
        outputs.append(adapter.apply_deepmegamoe(x_device, ids_device, probs_device,
                       layer.w13_weight, layer.w13_weight_scale, layer.w2_weight, layer.w2_weight_scale,
                       config, swiglu_limit=10))
    torch.npu.synchronize()
    metrics = []
    for output in outputs:
        value = output.cpu().float()
        finite = bool(torch.isfinite(value).all())
        diff = value - reference
        relative = float(diff.norm() / reference.norm().clamp_min(1e-12))
        cosine = float(torch.nn.functional.cosine_similarity(value.double().flatten(), reference.double().flatten(), dim=0)) if m else 1.
        metrics.append(dict(finite=finite, max_abs=float(diff.abs().max()) if m else 0., relative_l2=relative, cosine=cosine))
    passed = all(x['finite'] and x['relative_l2'] < .001 and x['cosine'] > .999999 for x in metrics)
    diagnostics = {}
    if not passed:
        buffer = adapter._get_symm_buffer(config)
        for key in ('x', 'x_sf', 'l1_acts', 'l1_acts_sf', 'l2_acts', 'l2_acts_sf'):
            tensor = getattr(buffer, key).cpu()
            raw = tensor.view(torch.uint8).flatten()
            diagnostics[key] = dict(shape=list(tensor.shape), bytes_first=raw[:64].tolist())
        diagnostics['output_first'] = outputs[-1].cpu().float().flatten()[:32].tolist()
        diagnostics['reference_first'] = reference.flatten()[:32].tolist()
    return dict(case=name, rank=rank, tokens=m, hidden=hidden, intermediate=intermediate,
                experts=experts, topk=topk, calls=metrics, passed=passed, diagnostics=diagnostics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='all', choices=['all', 'decode', 'batch', 'prefill', 'constant', 'zero', 'ragged'])
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--intermediate', type=int, default=512)
    parser.add_argument('--experts', type=int, default=8)
    parser.add_argument('--capacity', type=int, default=4096)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    torch.npu.set_device(rank)
    device = torch.device(f'npu:{rank}')
    dist.init_process_group('hccl')
    import vllm.distributed
    vllm.distributed.get_ep_group = lambda: SimpleNamespace(device_group=dist.group.WORLD)
    adapter.get_ascend_config = lambda: SimpleNamespace(deepmegamoe_max_tokens_per_rank=args.capacity)
    names = ['decode', 'batch', 'prefill', 'zero', 'ragged'] if args.case == 'all' else [args.case]
    results = []
    try:
        for name in names:
            result = run_case(rank, world, device, name, args.hidden, args.intermediate, args.experts, args.capacity)
            results.append(result)
            print(json.dumps(result), flush=True)
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f'rank-{rank}.json').write_text(json.dumps(results, indent=2))
            flags = [None] * world
            dist.all_gather_object(flags, result['passed'])
            if not all(flags):
                raise AssertionError(f'Numerical check failed on one or more ranks: {flags}')
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f'rank-{rank}.json').write_text(json.dumps(results, indent=2))
    finally:
        adapter.release_deepmegamoe_buffers()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
