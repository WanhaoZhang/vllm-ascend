import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

r=Path(__file__).resolve().parent
s=json.loads((r/'diagnosis-summary.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(13,5))
ax=axes[0]; ms=[1,2,8,32,512,2048]; x=np.arange(len(ms)); w=.25
groups=[('catccos','packed_fp4','CatCCOS: per-call unpack','#e76f51'),
        ('catccos','preexpanded_fp8','CatCCOS: exact FP8 pre-unpack','#457b9d'),
        ('deepmegamoe','direct_fp4','deepMegaMoE: direct FP4','#2a9d8f')]
for j,(backend,mode,label,color) in enumerate(groups):
    ys=[next(z['wall_ms'] for z in s['micro'][backend] if z['mode']==mode and z['routing']=='spread' and z['tokens_per_rank']==m) for m in ms]
    ax.bar(x+(j-1)*w,ys,w,label=label,color=color)
ax.set_yscale('log');ax.set_ylim(.08,8);ax.set_xticks(x,ms)
ax.set_xlabel('Tokens per rank (controlled spread routing)')
ax.set_ylabel('Average fused-call wall time (ms, log scale)')
ax.set_title('Four-rank operator ablation',loc='left',weight='bold')
ax.legend(fontsize=8,loc='upper left')
ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
ax=axes[1]; points=['decode-c1','decode-c8'];x=np.arange(2)
for j,(service,label,color) in enumerate([('cache0-catccos','No persistent cache','#e76f51'),('cache8-catccos','8 layers cached exactly','#457b9d')]):
    ys=[s['services'][service][p]['mean_token_interval_ms'] for p in points]
    bars=ax.bar(x+(j-.5)*.32,ys,.32,label=label,color=color)
    ax.bar_label(bars,labels=[f'{y:.2f}' for y in ys],padding=4)
ax.set_ylim(0,165);ax.set_xticks(x,['Concurrency 1','Concurrency 8'])
ax.set_ylabel('Mean output-token interval (ms)')
ax.set_title('Real model: cache only 8 of 43 layers',loc='left',weight='bold')
ax.legend(fontsize=9,loc='lower left');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
for i,p in enumerate(points):
    a=s['services']['cache0-catccos'][p]['mean_token_interval_ms']
    b=s['services']['cache8-catccos'][p]['mean_token_interval_ms']
    ax.text(i,145,f'{100*(1-b/a):.2f}% lower',ha='center',weight='bold')
fig.suptitle('DeepSeek-V4-Flash: repeated FP4 unpack has a measured end-to-end cost',weight='bold',y=.99)
fig.text(.06,.015,'Left: constant synthetic weights; 3 rounds x 20 calls after warm-up. Right: official weights, ISL128/OSL32; 3 rounds; profiler off.\nThe operator ablation is not a prediction of full-model latency. The 8-layer cache adds 12 GiB per rank.',fontsize=9)
fig.tight_layout(rect=(0,.085,1,.95))
fig.savefig(r/'diagnosis.png',dpi=180)
fig.savefig(r/'diagnosis.svg')
