#!/usr/bin/env python3
"""Four-rank MXFP4 CatCCOS checks against an independent unpacked reference.

Run with torchrun after loading CANN and the CatCCOS SHMEM environment.
The reference decodes FP4 nibbles and E8M0 bytes explicitly on CPU. Only
activation MXFP8 quantization uses torch_npu, to match the hardware rounding.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu


def dequantize_fp4(weight, scale):
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2).long()
    factors = torch.exp2(scale.float() - 127).repeat_interleave(32, dim=-1)
    return lut[codes] * factors


def round_mxfp8(matrix, device):
    quantized, scales = torch_npu.npu_dynamic_mx_quant(
        matrix.to(device=device, dtype=torch.bfloat16).contiguous(),
        dst_type=torch.float8_e4m3fn,
        scale_alg=0,
    )
    factors = torch.exp2(scales.view(torch.uint8).cpu().float().reshape(matrix.shape[0], -1) - 127)
    return quantized.cpu().float() * factors.repeat_interleave(32, dim=-1)[:, : matrix.shape[-1]]


def run_case(rank, world_size, device, case, hidden=256, intermediate=256, routing='uniform', burst=1,
             interleave_linear=False, linear_output_multiplier=1):
    experts, m, random_weights, limit, top_k = case
    generator = torch.Generator().manual_seed(42)
    if random_weights:
        w1 = torch.randint(0, 256, (experts, 2 * intermediate, hidden // 2), generator=generator, dtype=torch.uint8)
        w2 = torch.randint(0, 256, (experts, hidden, intermediate // 2), generator=generator, dtype=torch.uint8)
        x = torch.randn((m, hidden), generator=generator).mul_(0.15).to(torch.bfloat16)
        s1 = torch.randint(119, 123, (experts, 2 * intermediate, hidden // 32), generator=generator, dtype=torch.uint8)
        s2 = torch.randint(119, 123, (experts, hidden, intermediate // 32), generator=generator, dtype=torch.uint8)
    else:
        w1 = torch.full((experts, 2 * intermediate, hidden // 2), 0x22, dtype=torch.uint8)
        w2 = torch.full((experts, hidden, intermediate // 2), 0x22, dtype=torch.uint8)
        x = torch.full((m, hidden), 0.125, dtype=torch.bfloat16)
        exponent = 126 if limit else 120
        s1 = torch.full((experts, 2 * intermediate, hidden // 32), exponent, dtype=torch.uint8)
        s2 = torch.full((experts, hidden, intermediate // 32), 120, dtype=torch.uint8)
    ids = ((torch.arange(m)[:, None] + rank * (experts // world_size) + torch.arange(top_k)) % experts)
    if routing == 'sparse':
        ids = torch.arange(top_k).expand(m, -1).contiguous()
    elif routing == 'skewed':
        routing_generator = torch.Generator().manual_seed(4200 + rank)
        distribution = torch.arange(1, experts + 1).float().reciprocal().expand(m, -1)
        ids = torch.multinomial(distribution, top_k, replacement=False, generator=routing_generator)
    probs = torch.arange(1, top_k + 1).float().expand(m, -1).contiguous()
    probs /= probs.sum(-1, keepdim=True)
    x_ref = round_mxfp8(x, device)
    reference = torch.zeros((m, hidden))
    for expert in (ids.unique().tolist() if random_weights else [0]):
        selected = (ids == expert).any(-1) if random_weights else torch.ones(m, dtype=torch.bool)
        reference_x = x_ref[selected] if random_weights else x_ref[:1]
        gate_up = (reference_x @ dequantize_fp4(w1[expert], s1[expert]).T).to(torch.bfloat16).float()
        gate, up = gate_up.chunk(2, dim=-1)
        if limit:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        activated = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        rounded = round_mxfp8(activated, device)
        weight = ((ids == expert) * probs).sum(-1)[selected, None] if random_weights else torch.ones((m, 1))
        reference[selected] += (rounded @ dequantize_fp4(w2[expert], s2[expert]).T).to(torch.bfloat16).float() * weight
    reference = reference.to(torch.bfloat16).float()

    local_experts = experts // world_size
    section = slice(rank * local_experts, (rank + 1) * local_experts)
    w1_device = w1[section].to(device).transpose(1, 2)
    w2_device = w2[section].to(device).transpose(1, 2)
    s1_device = s1[section].to(device).reshape(local_experts, 2 * intermediate, hidden // 64, 2).transpose(1, 2)
    s2_device = s2[section].to(device).reshape(local_experts, hidden, intermediate // 64, 2).transpose(1, 2)
    x_device, ids_device, probs_device = x.to(device), ids.to(device, dtype=torch.int32), probs.to(device)
    if interleave_linear:
        if random_weights or limit <= 0:
            raise ValueError('Interleaved identity linear requires constant weights with saturated clamp')
        linear_weight = torch.eye(hidden).repeat(1, linear_output_multiplier).to(torch.float8_e4m3fn).to(device)
        linear_scale = torch.full((hidden // 64, hidden * linear_output_multiplier, 2), 127, dtype=torch.uint8, device=device)
        norm_weight = torch.ones(512, dtype=torch.bfloat16, device=device)
    dist.barrier()
    outputs = []
    for iteration in range(burst):
        if rank == 3 and burst > 1:
            time.sleep(0.001)
        outputs.append(torch.ops.catccos.ascend950_dispatch_ffn_combine(
            x_device.contiguous(), ids_device, probs_device, w1_device, s1_device, w2_device, s2_device, float(limit)
        ))
        if interleave_linear:
            quantized, scale = torch_npu.npu_dynamic_mx_quant(outputs[-1], dst_type=torch.float8_e4m3fn)
            x_device = torch_npu.npu_quant_matmul(
                quantized, linear_weight, linear_scale, scale_dtype=torch_npu.float8_e8m0fnu,
                pertoken_scale=scale, pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                output_dtype=torch.bfloat16, group_sizes=[1, 1, 32])
            if linear_output_multiplier > 1:
                normalized, _ = torch_npu.npu_rms_norm(x_device.unflatten(-1, (-1, 512)), norm_weight, 1e-6)
                x_device = normalized.flatten(1)[:, :hidden]
    torch.npu.synchronize()
    output = outputs[-1]
    actual = output.cpu().float()
    # Run the same fused kernel with independently unpacked FP8 weights.
    # This distinguishes the new nibble decoder from existing layout/routing.
    lut_fp8 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
    def unpack_device(weight):
        codes = torch.stack((weight & 15, weight >> 4), -1).flatten(-2).long()
        return lut_fp8[codes].to(torch.float8_e4m3fn).to(device)
    output_fp8 = torch.ops.catccos.ascend950_dispatch_ffn_combine(
        x_device.contiguous(), ids_device, probs_device, unpack_device(w1[section]), s1[section].to(device),
        unpack_device(w2[section]), s2[section].to(device), float(limit)
    )
    torch.npu.synchronize()
    actual_fp8 = output_fp8.cpu().float()
    finite = bool(torch.isfinite(actual).all())
    delta = actual - reference
    rel_l2 = float(delta.norm() / reference.norm().clamp_min(1e-8))
    cosine = float(torch.nn.functional.cosine_similarity(actual.double().flatten(), reference.double().flatten(), dim=0))
    burst_max_error = max(float((item.cpu().float() - reference).abs().max()) for item in outputs)
    burst_relative_l2 = max(float((item.cpu().float() - reference).norm() / reference.norm().clamp_min(1e-8))
                            for item in outputs)
    passed = finite and burst_relative_l2 <= 0.08 and cosine >= 0.99 and all(bool(torch.isfinite(item).all().cpu()) for item in outputs)
    result = dict(rank=rank, experts=experts, local_experts=local_experts, m=m, top_k=top_k,
                  random_weights=random_weights, swiglu_limit=limit, finite=finite,
                  relative_l2=rel_l2, cosine=cosine, max_abs_error=float(delta.abs().max()),
                  actual_first=float(actual[0, 0]), expected_first=float(reference[0, 0]), passed=passed)
    result['fp4_vs_fp8_max_error'] = float((actual - actual_fp8).abs().max())
    result['fp8_relative_l2'] = float((actual_fp8 - reference).norm() / reference.norm().clamp_min(1e-8))
    result.update(burst=burst, burst_max_abs_error=burst_max_error, burst_relative_l2=burst_relative_l2,
                  interleave_linear=interleave_linear)
    result['linear_output_multiplier'] = linear_output_multiplier
    print(json.dumps(result), flush=True)
    flags = [None] * world_size
    dist.all_gather_object(flags, passed)
    return result, all(flags)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--store", default="tcp://127.0.0.1:27051")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--experts", type=int)
    parser.add_argument("--routing", choices=('uniform', 'sparse', 'skewed'), default='uniform')
    parser.add_argument("--burst", type=int, default=1)
    parser.add_argument("--interleave-linear", action='store_true')
    parser.add_argument("--linear-output-multiplier", type=int, default=1)
    parser.add_argument("--case", default="all", choices=("all", "single", "multi", "random", "clamp", "decode6", "batch6", "prefill6", "prefill-constant"))
    args = parser.parse_args()
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.set_num_threads(1)
    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")
    dist.init_process_group("gloo")
    torch.ops.load_library(args.library)
    status = torch.ops.catccos.init(rank, world_size, 1 << 30, args.store)
    if status:
        raise RuntimeError(f"SHMEM init returned {status}")
    cases = {"single": (world_size, 4, False, 0, 1), "multi": (world_size * 2, 4, False, 0, 1),
             "random": (world_size * 2, 17, True, 10, 1), "clamp": (world_size * 2, 1, False, 10, 1),
             "decode6": (world_size * 2, 1, True, 10, 6), "batch6": (world_size * 2, 32, True, 10, 6),
             "prefill6": (world_size * 2, 4096, True, 10, 6),
             "prefill-constant": (world_size * 64, 4096, False, 10, 6)}
    results = []
    all_passed = True
    try:
        for name, case in cases.items():
            if args.case != "all" and args.case != name:
                continue
            if args.experts is not None:
                case = (args.experts, *case[1:])
            result, passed = run_case(rank, world_size, device, case, args.hidden_size, args.intermediate_size,
                                     args.routing, args.burst, args.interleave_linear, args.linear_output_multiplier)
            result.update(hidden_size=args.hidden_size, intermediate_size=args.intermediate_size)
            result['routing'] = args.routing
            result["case"] = name
            results.append(result)
            all_passed &= passed
            if not passed:
                break
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"rank-{rank}.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        dist.barrier()
        torch.ops.catccos.finalize()
        dist.destroy_process_group()
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
