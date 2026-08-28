# CatCCOS 第一层 MoE 四阶段归约探针

## 目的

这个探针用同一份 `hidden_states`、`router_logits` 和路由结果，在第一层
MoE 内依次运行 native 与 CatCCOS，然后在副本上调用生产路径完全相同的
`torch.ops.vllm.maybe_all_reduce_tensor_model_parallel`。真实返回给模型的
CatCCOS 输出不被副本上的诊断归约修改。

每个 rank 保存四个关键输出：

| 名称 | 含义 |
| --- | --- |
| `native_local` | `AscendFusedMoE.forward_impl` 的 native 返回值 |
| `native_reduced` | 对 `native_local` 的副本执行外层 MoE 归约后的值 |
| `catccos_pre_reduce` | CatCCOS fused dispatch-FFN-combine 直接返回值 |
| `catccos_post_reduce` | 对 CatCCOS 返回值的副本执行外层 MoE 归约后的值 |

同时保存输入、路由、native `w13/w2`、CatCCOS MXFP8 `w1/w2/scale`，并记录
`moe_comm_type`、TP/EP rank、world size、FlashComm 状态和实际外层归约动作。

> 这是重型正确性探针，只能在固定 prompt、eager 模式、所有 rank 使用相同
> 环境变量时运行。不要用于性能测试或并发请求。

## 在另一台机器的已有 Docker 中运行

以下命令假设容器名为 `megamoe-vllm`，仓库位于
`/workspace/vllm-ascend`，模型和 CatCCOS 路径与 A5 现场默认值一致。

先在第一个终端拉取分支并启动服务：

```bash
ssh <目标机器>
docker exec -it megamoe-vllm bash
cd /workspace/vllm-ascend
git fetch origin codex/megamoe-vllm-v023
git switch codex/megamoe-vllm-v023
git pull --ff-only origin codex/megamoe-vllm-v023

export PROBE_RUN_ID=stage1-$(date +%Y%m%d-%H%M%S)
echo "${PROBE_RUN_ID}" >/tmp/catccos-stage1-run-id

bash examples/megamoe_qwen3/run_first_layer_reduction_probe.sh
```

如果目标机路径或卡号不同，在最后一条命令前覆盖相应变量：

```bash
export MODEL_PATH=/home/weights/Qwen3-30B-A3B-Instruct-2507
export CATCCOS_SOURCE=/home/z00956592/catccos
export CATCCOS_BUILD_DIR=${CATCCOS_SOURCE}/build_codex
export NPU_DEVICES=4,5,6,7
export PORT=28001
```

脚本也会自动识别 a5new 风格的
`/workspace/catccos/build_torch_a5/lib/libcatccos_torch.so`。模型必须由目标
Docker 自己挂载；启动前若 `MODEL_PATH` 不存在，脚本会直接报错而不会误跑。

在第二个终端进入同一容器，发送一次固定请求：

```bash
ssh <目标机器>
docker exec -it megamoe-vllm bash
cd /workspace/vllm-ascend

export PROBE_RUN_ID=$(cat /tmp/catccos-stage1-run-id)
bash examples/megamoe_qwen3/send_first_layer_probe_request.sh
```

客户端会执行以下检查：

1. 等待 `/health` 就绪；
2. 发送记录中相同的 dragon/Perg prompt，`temperature=0`、`seed=42`、
   `max_tokens=1`；
3. 确认 chat template 后的 `prompt_tokens=177`；
4. 等到 4 个 rank 都保存完成；
5. 打印关键指标并生成 `${PROBE_RUN_ID}.tar.gz`。

运行结束后在第一个终端按 `Ctrl-C` 停止服务。每次重跑必须换一个新的
`PROBE_RUN_ID`，避免新旧结果混在一起。

## 输出位置

默认目录：

```text
/home/z00956592/catccos-probe-results/<PROBE_RUN_ID>/native-catccos/
```

关键文件：

- `probe-rankNNN.jsonl`：各 rank 的指标与通信上下文；
- `first-selected-rankNNN-*.pt`：输入、路由、四阶段输出和权重；
- `first-selected-rankNNN.json`：对应 dump 的可读索引；
- `chat-completion-response.json`：固定请求返回；
- 上一级 `<PROBE_RUN_ID>.tar.gz`：可直接复制回分析机器的结果包。

查看一个 dump 的键：

```bash
python - <<'PY'
from pathlib import Path
import torch

path = next(Path("/home/z00956592/catccos-probe-results").glob(
    "*/native-catccos/first-selected-rank*.pt"
))
payload = torch.load(path, map_location="cpu", weights_only=False)
print(payload["metadata"]["parallel_context"])
print("tensors:", sorted(payload["tensors"]))
print("weights:", sorted(payload.get("weights", {})))
PY
```

## 如何判读

优先看三个比较：

1. `native_reduced_vs_catccos_pre_reduce`：判断 CatCCOS 直接输出是否已经等价
   于 native 完整输出；
2. `native_reduced_vs_catccos_post_reduce`：判断生产路径继续执行外层归约后
   是否引入额外偏差；
3. `native_local_vs_catccos_pre_reduce`：仅用于确认旧探针是否比较了不同阶段，
   不能单独作为算子错误证据。

典型结论：

- 若第 1 项 cosine 接近 1、relative L2 很小，而第 2 项明显变差，且
  `outer_reduction=tp-all-reduce`，则 CatCCOS 很可能已经返回完整量，生产路径
  又归约了一次；
- 若第 1 项仍明显不一致，则“阶段错位”不能解释差异，下一步应使用保存的
  第一层输入、路由和权重检查 CatCCOS 的 DP+EP/SP+EP 输入分片契约；
- `native-local` 与 CatCCOS 不一致本身不是错误，因为两者可能不是同一阶段。

如果只想先验证四阶段输出、不保存大权重，可在启动服务前设置：

```bash
export PROBE_DUMP_WEIGHTS=0
```

## 非默认参数

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PROBE_TOKEN_COUNTS` | `177` | 选择的 token-row 数；改变后必须使用匹配请求 |
| `PROBE_MOE_INSTANCE_IDS` | `0` | 只探第一层；逗号分隔可选择多层 |
| `PROBE_EXPECTED_RANKS` | `4` | 客户端等待的 rank dump 数量 |
| `PROBE_DUMP_SELECTED` | `1` | 即使低于 mismatch 阈值也保存所选调用 |
| `PROBE_DUMP_WEIGHTS` | `1` | 保存 native 与 CatCCOS 权重，磁盘和主机内存开销较大 |
| `PROBE_OUTPUT_ROOT` | `/home/z00956592/catccos-probe-results` | 容器内输出根目录 |
