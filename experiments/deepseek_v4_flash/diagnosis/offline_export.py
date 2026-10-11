from pathlib import Path
from torch_npu.profiler.profiler import analyse
r=Path('/home/z009565592/catccos-vllm/work/deepseek-v4-diagnosis-20261011')
for p in sorted((r/'profile2-catccos/traces').glob('*_ascend_pt')):
 print('OFFLINE_ANALYSE',p,flush=True)
 analyse(str(p))
