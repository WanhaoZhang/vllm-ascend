# CatCCOS formal FusedMC2 首次端到端验证

本文验证 `codex/megamoe-vllm-v023-formal` 的正式 FusedMC2 接入能否完成一个
Qwen3-30B-A3B 请求。第一次只做 correctness smoke，不采集性能数据。

固定约束：

- vLLM/vLLM-Ascend `0.23.0`；
- Ascend 950，至少两张 NPU；
- TP 与 EP 大小相同；
- BF16 未量化模型；
- eager 模式；
- 首次运行启用 `catccos_sync_after_launch=true`。

## 路径一：从宿主机启动新容器（推荐）

在宿主机更新代码并检查目标提交：

```bash
export VLLM_ASCEND_SOURCE=/data/src/vllm-ascend
export CATCCOS_SOURCE=/data/src/catccos
export MODEL_PATH=/data/models/Qwen3-30B-A3B

cd "${VLLM_ASCEND_SOURCE}"
git status --short --branch
git remote get-url megamoe >/dev/null 2>&1 || \
  git remote add megamoe https://github.com/WanhaoZhang/vllm-ascend.git
git fetch megamoe codex/megamoe-vllm-v023-formal
git switch codex/megamoe-vllm-v023-formal
git pull --ff-only megamoe codex/megamoe-vllm-v023-formal
git log -1 --oneline

test -f "${CATCCOS_SOURCE}/build_torch_a5/lib/libcatccos_torch.so"
test -f "${MODEL_PATH}/config.json"
```

第一次建议 TP4/EP4；机器只有两张空闲卡时改为 `0,1`：

```bash
MODE=catccos \
CONTAINER_NAME=megamoe-catccos-formal \
PORT=18081 \
NPU_DEVICES=0,1,2,3 \
CATCCOS_SYNC_DEVICE=1 \
VLLM_ASCEND_SOURCE="${VLLM_ASCEND_SOURCE}" \
CATCCOS_SOURCE="${CATCCOS_SOURCE}" \
MODEL_PATH="${MODEL_PATH}" \
bash examples/megamoe_qwen3/run_docker.sh
```

脚本会等待 `/health` 成功。如果已有同名容器，确认可以替换后增加
`RECREATE=1`；不要删除名称不同的现有容器。

## 路径二：在已经运行的容器内启动

只有当容器已经挂载 NPU、模型、CatCCOS 动态库和多卡拓扑时使用此路径。
下面假设代码、模型和 CatCCOS 分别位于：

```text
/vllm-workspace/vllm-ascend
/model
/workspace/catccos
```

进入容器后先确认源码确实来自 formal 分支：

```bash
cd /vllm-workspace/vllm-ascend
git status --short --branch
git log -1 --oneline

test -f vllm_ascend/ops/fused_moe/catccos_adapter.py
test -f /workspace/catccos/build_torch_a5/lib/libcatccos_torch.so

export PYTHONPATH=/vllm-workspace/vllm-ascend:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/workspace/catccos/build_torch_a5/lib:/workspace/catccos/3rdparty/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1024
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

确保没有另一个 vLLM 进程占用相同 NPU 或端口，然后在当前终端前台启动：

```bash
vllm serve /model \
  --served-model-name Qwen3-30B-A3B \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.70 \
  --no-enable-prefix-caching \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 18081 \
  --additional-config \
  '{"enable_fused_mc2":1,"fused_mc2_backend":"catccos","catccos_library_path":"/workspace/catccos/build_torch_a5/lib/libcatccos_torch.so","catccos_store_url":"tcp://127.0.0.1:27020","catccos_local_mem_size":1073741824,"catccos_max_tokens_per_rank":512,"catccos_sync_after_launch":true}'
```

使用两张卡时，把 `ASCEND_RT_VISIBLE_DEVICES` 和
`--tensor-parallel-size` 同时改成 `0,1` 与 `2`。

## 发送第一个端到端请求

在宿主机或容器的第二个终端执行：

```bash
curl -fsS http://127.0.0.1:18081/health

curl --fail --silent --show-error \
  http://127.0.0.1:18081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-30B-A3B",
    "messages": [{
      "role": "user",
      "content": "What is 17 * 23? Return only the number."
    }],
    "temperature": 0,
    "seed": 42,
    "max_tokens": 32,
    "chat_template_kwargs": {"enable_thinking": false}
  }' | tee /tmp/catccos-formal-e2e.json
```

预期返回 `391`，并且没有超时、空输出或 worker 退出。

## 确认实际走了 CatCCOS

宿主机 launcher 路径：

```bash
docker logs megamoe-catccos-formal 2>&1 | grep -E \
  'Initialized CatCCOS|Converted CatCCOS|Executed CatCCOS through the formal FusedMC2 backend'
```

已有容器路径直接检查服务终端。必须至少看到每个 rank 初始化 CatCCOS 和转换
MXFP8 权重；请求后还必须出现下面的执行日志：

```text
Executed CatCCOS through the formal FusedMC2 backend
```

这些信息使用 `logger.info_once`，每个 worker 最多打印一次。

如果日志没有 CatCCOS 初始化信息，不能把 HTTP 成功算作通过；它可能走了 native
fallback。若服务失败，先保存：

```bash
docker logs --tail 500 megamoe-catccos-formal \
  > /tmp/megamoe-catccos-formal.log 2>&1
```

## 本次 smoke 的通过条件

| 项目 | 通过条件 |
| --- | --- |
| 启动 | 所有 worker 完成 CatCCOS 初始化，服务健康 |
| 接线 | 日志确认请求执行了 formal FusedMC2 CatCCOS 路径 |
| prefill/decode | `max_tokens=32` 请求完成，无卡死或 worker 退出 |
| 输出 | 固定贪心请求得到 `391` |

这只证明一个端到端 smoke。通过后再分别运行 native/CatCCOS 固定请求 A/B、
GSM8K smoke 和 layer0 正式路径 tensor dump。性能测试前应关闭后同步：

```bash
CATCCOS_SYNC_DEVICE=0
```

关闭同步后必须重新做固定请求和准确率 smoke，确认异步执行的依赖关系正确。
