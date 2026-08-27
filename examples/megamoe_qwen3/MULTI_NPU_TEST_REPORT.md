# CatCCOS 接入 vLLM-Ascend 多卡测试报告与执行清单

## 1. 测试目标

本报告用于在具备完整 D2D topology 的 Ascend 950DT 机器上验证：

1. CatCCOS `ascend950_dispatch_ffn_combine` 能否在 TP2/EP2 和 TP4/EP4
   正常启动、完成请求且无 rank hang。
2. 接入 CatCCOS 后，Qwen3-30B-A3B 的完整 GSM8K 精度是否与原生
   vLLM-Ascend 基线一致。
3. 只有精度通过后，再决定是否进行多卡性能测试。

本次多卡测试的首要目标是正确性和精度，不要求用单卡性能数据推断多卡收益。

## 2. 固定版本

| 项目 | 版本 |
|---|---|
| vLLM-Ascend | GitHub `codex/megamoe-vllm-v023`，至少包含 `9c9704b7` |
| CatCCOS | GitHub/GitCode `codex/megamoe-vllm`，`9cdc8985` |
| Docker image | `quay.io/ascend/vllm-ascend:v0.23.0-a5` |
| Model | `Qwen/Qwen3-30B-A3B`，本地 BF16 checkpoint |
| AISBench | source install，已验证版本 3.1.0 |
| Dataset | AISBench/OpenCompass GSM8K test split，1319 条 |

### 2.1 `a5new` 单卡参考环境

2026-08-26 已在 `a5new` 重新确认当前 0.23.0 分支与运行容器的
对应关系：

- vLLM-Ascend 分支为 `codex/megamoe-vllm-v023@a2fa5c2b`，基线为
  `v0.23.0@5cb98caa`。
- Docker client/server 均为 29.6.2，API 1.55；宿主机为 Ubuntu
  24.04.4 LTS、kernel 6.8.0-136-generic。
- 运行镜像为 `quay.io/ascend/vllm-ascend:v0.23.0-a5`，digest 为
  `sha256:cc57064f119054904dc81360cd1105d211fa9b91bf726486926dd025c26f17b7`。
- 镜像内 Git HEAD 正是 `v0.23.0@5cb98caa`；分支的 `__init__.py`、
  `envs.py`、`catccos_patch.py` 和 `catccos_debug.py` 通过 bind mount 覆盖镜像文件，
  宿主机与容器内 SHA-256 逐文件一致。
- 320-token 真实请求超过当时的 `CATCCOS_MINM=64` 并返回 HTTP
  200/正确答案 `42`；容器 restart count 为 0，本轮日志无新
  ERROR/Traceback。该配置的单 token decode 会回退原生路径，因此这个
  结果只证明 CatCCOS prefill，不证明 CatCCOS decode。

这是“官方 v0.23.0 镜像 + 四个运行时补丁文件 + CatCCOS 动态库”
的组合，不是把整个分支重新构建成镜像。后续若修改其他
`vllm_ascend` 源码，必须增加挂载或重新构建完整镜像，否则容器
不会使用新修改。详细校验命令见
[CATCCOS_950DT_GUIDE.md](CATCCOS_950DT_GUIDE.md)。

测试前记录实际 commit 和镜像 digest：

```bash
git -C /data/src/vllm-ascend rev-parse HEAD
git -C /data/src/catccos rev-parse HEAD
docker image inspect quay.io/ascend/vllm-ascend:v0.23.0-a5 \
  --format '{{.Id}}'
```

## 3. 同步代码

首次拉取：

```bash
mkdir -p /data/src

git clone --branch codex/megamoe-vllm-v023 \
  https://github.com/WanhaoZhang/vllm-ascend.git \
  /data/src/vllm-ascend

git clone --branch codex/megamoe-vllm --recurse-submodules \
  https://gitcode.com/zhangwanhao/catccos.git \
  /data/src/catccos
```

已有仓库只允许 fast-forward 更新，避免覆盖机器上的本地修改：

```bash
git -C /data/src/vllm-ascend switch codex/megamoe-vllm-v023
git -C /data/src/vllm-ascend pull --ff-only

git -C /data/src/catccos switch codex/megamoe-vllm
git -C /data/src/catccos pull --ff-only
git -C /data/src/catccos submodule update --init --recursive
```

## 4. 机器和 topology 门禁

```bash
npu-smi info
test -f /lib/route.conf
test -f /etc/hccl_rootinfo.json
test -d /etc/hixlep
```

以上 topology 必须在目标机器生成，不能从其他服务器复制。缺少任一项时不要进行
TP2/TP4 测试。完整构建命令见
[CATCCOS_950DT_GUIDE.md](CATCCOS_950DT_GUIDE.md)。

## 5. 先验证 standalone 算子

CatCCOS extension 构建完成后，在相同 Docker image 内依次执行：

```bash
cd /workspace/catccos

bash examples/ascend950_dispatch_ffn_combine/scripts/run_python.sh 0
bash examples/ascend950_dispatch_ffn_combine/scripts/run_python.sh 0,1
bash examples/ascend950_dispatch_ffn_combine/scripts/run_python.sh 0,1,2,3
```

验收要求：

- 每个 rank 均退出成功并显示 `PASS`。
- 无 hang、device pointer、SHMEM 和 kernel launch 错误。
- TP2、TP4 分别使用不同的 `IPPORT`，避免多个 SHMEM job 冲突。

standalone 不通过时不要启动 vLLM。

## 6. TP2/EP2 完整精度 A/B

### 6.1 启动原生基线

```bash
cd /data/src/vllm-ascend

MODEL_PATH=/data/models/Qwen3-30B-A3B \
SERVED_MODEL_NAME=Qwen3-30B-A3B \
MODE=native \
CONTAINER_NAME=megamoe-native-tp2 \
PORT=18080 \
NPU_DEVICES=0,1 \
GPU_MEMORY_UTILIZATION=0.80 \
bash examples/megamoe_qwen3/run_docker.sh
```

### 6.2 跑完整 GSM8K

不要设置 `NUM_PROMPTS`，否则只是子集：

```bash
unset NUM_PROMPTS

AISBENCH_ROOT=/data/tools/benchmark \
MODEL_PATH=/data/models/Qwen3-30B-A3B \
SERVED_MODEL_NAME=Qwen3-30B-A3B \
AISBENCH_PORT=18080 \
MODE=accuracy \
RUN_LABEL=native-tp2-full \
NUM_WARMUPS=1 \
AISBENCH_MAX_OUT_LEN=1024 \
AISBENCH_BATCH_SIZE=8 \
bash examples/megamoe_qwen3/run_aisbench_gsm8k.sh
```

### 6.3 使用相同物理卡启动 CatCCOS

停止基线后再启动 CatCCOS，避免两个服务同时争用同一组 NPU：

```bash
docker stop megamoe-native-tp2

MODEL_PATH=/data/models/Qwen3-30B-A3B \
SERVED_MODEL_NAME=Qwen3-30B-A3B \
MODE=catccos \
CONTAINER_NAME=megamoe-catccos-tp2 \
PORT=18081 \
NPU_DEVICES=0,1 \
GPU_MEMORY_UTILIZATION=0.80 \
VLLM_ASCEND_SOURCE=/data/src/vllm-ascend \
CATCCOS_SOURCE=/data/src/catccos \
CATCCOS_IPPORT=tcp://127.0.0.1:27021 \
CATCCOS_MINM=1 \
bash examples/megamoe_qwen3/run_docker.sh
```

确认两个 EP rank 都初始化了 CatCCOS：

```bash
docker logs megamoe-catccos-tp2 2>&1 | grep -E \
  'Enabled CatCCOS|Initialized CatCCOS|Converted CatCCOS'
```

然后运行相同 GSM8K 配置，只修改端口和标签：

```bash
unset NUM_PROMPTS

AISBENCH_ROOT=/data/tools/benchmark \
MODEL_PATH=/data/models/Qwen3-30B-A3B \
SERVED_MODEL_NAME=Qwen3-30B-A3B \
AISBENCH_PORT=18081 \
MODE=accuracy \
RUN_LABEL=catccos-tp2-full \
NUM_WARMUPS=1 \
AISBENCH_MAX_OUT_LEN=1024 \
AISBENCH_BATCH_SIZE=8 \
bash examples/megamoe_qwen3/run_aisbench_gsm8k.sh
```

## 7. TP4/EP4 完整精度 A/B

TP2 通过后，按相同顺序运行 TP4。启动参数仅调整为：

```bash
NPU_DEVICES=0,1,2,3
CONTAINER_NAME=megamoe-native-tp4
```

CatCCOS 端使用：

```bash
NPU_DEVICES=0,1,2,3
CONTAINER_NAME=megamoe-catccos-tp4
CATCCOS_IPPORT=tcp://127.0.0.1:27022
```

AISBench 标签分别改为 `native-tp4-full` 和 `catccos-tp4-full`。其他模型、
服务和 AISBench 参数不得变化。

## 8. 精度验收标准

每个 TP size 单独验收：

| 检查项 | 要求 |
|---|---|
| GSM8K 样本数 | 1319/1319 |
| Failed requests | 0 |
| 空输出、NaN、服务退出 | 0 |
| CatCCOS rank 初始化 | TP2 为 2 rank，TP4 为 4 rank |
| CatCCOS 相对 native 精度差 | 建议不低于 -1.0 个百分点 |
| 逐样本差异 | 保存并检查，不只保留总分 |

如果团队已有正式精度阈值，以正式阈值为准。无论采用什么阈值，都应在看到结果前
固定，不能根据结果临时放宽。

AISBench 结果默认位于：

```text
/data/tools/benchmark/outputs/megamoe/<RUN_LABEL>/accuracy/<timestamp>/
```

需要归档：`configs/`、`logs/`、`predictions/`、`results/` 和 `summary/`。

## 9. 单卡已有证据

`a5new` 上使用 AISBench 自带 GSM8K evaluator 比较了共同前缀 185 条：

| 指标 | Native | CatCCOS |
|---|---:|---:|
| 正确数 | 155/185 | 158/185 |
| Accuracy | 83.78% | 85.41% |
| Failed requests | 0 | 0 |

CatCCOS 相对 native 为 +1.62 个百分点，185 条中有 151 条提取后的最终答案一致。
该作业由用户主动停止，因此不是完整 GSM8K 分数。它只说明单卡没有观察到类似历史
EP4 报告中的大幅精度下降，不能替代 TP2/TP4 完整验证。该评测运行时
`CATCCOS_MINM=64`，生成阶段通常是 M=1，所以它也不能作为 CatCCOS decode
精度证据。

2026-08-27 在 `a5new` 单卡补充验证：同一进程先运行 M=64，再连续运行十次
M=1，算子均完成且十次 M=1 输出逐 bit 一致；正式 vLLM 服务日志也确认真实请求
进入了 M=1 CatCCOS 路径。但该请求未在 180 秒诊断超时内返回，因此完整 decode
输出和时延仍是待验收项。

## 10. 出现多卡精度回退时

按以下顺序缩小问题范围：

1. 保留失败 run 的 predictions、所有 rank 日志、环境变量、commit 和镜像 digest。
2. 用同一批差异样本从 TP4 降到 TP2，再降到 TP1。
3. 将 `CATCCOS_WEIGHT_QUANT_BACKEND=cpu` 重跑差异样本，区分权重量化差异。
4. 保持接入层调用前后的强制同步；直发算子尚未接入 TorchNPU 的 stream
   dependency tracking，当前不能关闭同步。
5. 在相同 rank size 上重跑 standalone 算子。
6. 对照 `/lib/route.conf`、`/etc/hccl_rootinfo.json` 和 `/etc/hixlep`，确认没有
   使用其他机器的 topology。

## 11. 性能测试说明

本轮优先完成多卡正确性和精度。只有 TP2/TP4 精度均通过后，才按
[AISBENCH_GSM8K.md](AISBENCH_GSM8K.md) 的 streaming performance 步骤测试
TTFT、TPOT 和 throughput。单卡 performance 不作为该多卡算子收益的验收依据。

还需注意：单卡默认 `CATCCOS_MINM=1`，prefill 和 decode 都会进入 CatCCOS；
多卡在正式验收前默认保守值 64。多卡 decode 验收必须显式设置
`CATCCOS_MINM=1`，并在日志中确认 single-token decode 消息。评测报告必须保留
该值，并区分“服务整体性能”和“融合算子实际覆盖率”。
