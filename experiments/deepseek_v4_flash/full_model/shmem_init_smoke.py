#!/usr/bin/env python3
"""Four-rank SHMEM init probe using the same CatCCOS extension as vLLM."""

import os

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
if os.environ.get("SIMULATE_VLLM_DP") == "1":
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(rank)
    local_rank = 0
else:
    local_rank = int(os.environ["LOCAL_RANK"])

import torch
import torch_npu
import torch.distributed as dist


store_url = os.environ.get("CATCCOS_STORE_URL", "tcp://127.0.0.1:27042")
library = os.environ.get(
    "CATCCOS_LIBRARY",
    "/home/z009565592/catccos-vllm/work/v4-catccos-kernel/build_torch_a5/lib/libcatccos_torch.so",
)

torch_npu.npu.set_device(local_rank)
dist.init_process_group(backend="hccl")
dist.barrier()
if os.environ.get("RESTORE_SHMEM_VISIBLE") == "1":
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3"
torch.ops.load_library(library)
status = torch.ops.catccos.init(rank, world_size, 1 << 30, store_url)
print(
    f"rank={rank} local_rank={local_rank} world_size={world_size} "
    f"device={torch_npu.npu.current_device()} store={store_url} init_status={status}",
    flush=True,
)
dist.destroy_process_group()
