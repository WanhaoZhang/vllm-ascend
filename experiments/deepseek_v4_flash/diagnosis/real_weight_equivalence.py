"""Packed/device unpack versus independent CPU exact cache on real layer 0."""
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
import torch_npu
from safetensors import safe_open
from catccos_cache_diagnostic import exact_cache

r=Path(__file__).resolve().parent
rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
torch.set_num_threads(1);torch.npu.set_device(rank);device=f'npu:{rank}'
dist.init_process_group('gloo')
root=Path('/data/models/DeepSeek-V4-Flash')
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
arrays={n:[] for n in ['w1.weight','w3.weight','w2.weight','w1.scale','w3.scale','w2.scale']}
for e in range(rank*64,(rank+1)*64):
    for n in arrays:
        key=f'layers.0.ffn.experts.{e}.{n}'
        with safe_open(str(root/index[key]),framework='pt',device='cpu') as f:
            arrays[n].append(f.get_tensor(key).view(torch.uint8))
raw1=torch.stack([torch.cat([x,y],dim=0) for x,y in zip(arrays['w1.weight'],arrays['w3.weight'])])
raw2=torch.stack(arrays['w2.weight'])
sc1=torch.stack([torch.cat([x,y],dim=0) for x,y in zip(arrays['w1.scale'],arrays['w3.scale'])])
sc2=torch.stack(arrays['w2.scale'])
w1=raw1.to(device).transpose(1,2);w2=raw2.to(device).transpose(1,2)
s1=sc1.to(device).reshape(64,4096,64,2).transpose(1,2)
s2=sc2.to(device).reshape(64,4096,32,2).transpose(1,2)
cached=exact_cache(w1,s1,w2,s2)
torch.ops.load_library(str(r.parent/'v4-catccos-kernel/build_torch_a5/lib/libcatccos_torch.so'))
assert torch.ops.catccos.init(rank,world,1<<30,'tcp://127.0.0.1:27061')==0
rows=[]
try:
    for M in [1,2,8,32]:
        g=torch.Generator().manual_seed(4200+rank)
        x=torch.randn((M,4096),generator=g).mul_(.15).to(torch.bfloat16).to(device)
        ids=torch.stack([torch.randperm(256,generator=g)[:6] for _ in range(M)]).to(device,dtype=torch.int32)
        probs=torch.softmax(torch.randn((M,6),generator=g),-1).to(device)
        outputs=[]
        for weights in [(w1,s1,w2,s2),cached,(w1,s1,w2,s2)]:
            y=torch.ops.catccos.ascend950_dispatch_ffn_combine(x,ids,probs,*weights,10.)
            torch.npu.synchronize();outputs.append(y.cpu().float())
        row=dict(rank=rank,M=M,finite=all(bool(torch.isfinite(y).all()) for y in outputs),
                 packed_vs_cached_max_abs=float((outputs[0]-outputs[1]).abs().max()),
                 repeated_packed_max_abs=float((outputs[0]-outputs[2]).abs().max()),
                 relative_l2=float((outputs[0]-outputs[1]).norm()/outputs[0].norm().clamp_min(1e-8)))
        row['passed']=row['finite'] and row['packed_vs_cached_max_abs']==0 and row['repeated_packed_max_abs']==0
        rows.append(row);print(json.dumps(row),flush=True)
    out=r/'real-weight-equivalence';out.mkdir(exist_ok=True)
    (out/f'rank-{rank}.json').write_text(json.dumps(rows,indent=2))
    flags=[None]*world;dist.all_gather_object(flags,all(x['passed'] for x in rows));assert all(flags),flags
finally:
    dist.barrier();torch.ops.catccos.finalize();dist.destroy_process_group()
