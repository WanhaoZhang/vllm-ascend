# CatCCOS 在已有 Docker 中的分层验收 Runbook

本文用于验证当前 `vLLM-Ascend 0.23.0 + CatCCOS` 接入是否正确。假设 Docker
已经运行，模型、CatCCOS 动态库、NPU 设备和多卡拓扑已经在容器中可用。

本文中的 `layer0` 指 `moe_instance_id=0`，即模型实际执行顺序中的第一个 MoE
实例；如果模型前面存在 dense 层，它不一定等于 `model.layers.0`。

验证分为三个独立的 gate：

| Gate | 证明什么 | 不能证明什么 |
| --- | --- | --- |
| A：接线路径 | CatCCOS 输出没有再次执行外层 TP all-reduce | CatCCOS 数值精度和最终模型输出 |
| B：layer0 数值 | 同一输入、路由下，CatCCOS layer0 输出接近 native 完整输出 | 后续层、LM head 和 decode 都正确 |
| C：端到端 | native/CatCCOS 的 token、准确率和稳定性在约定范围内 | 未测试 shape、并发和并行配置的泛化性 |

必须依次通过 A、B、C，才能开始正式性能测试。HTTP 200、健康检查通过或日志中
出现 `Executing CatCCOS`，都不能单独证明接入正确。

## 1. 设置容器和仓库路径

以下命令在宿主机执行。按实际情况修改前两项：

```bash
export CONTAINER=megamoe-vllm
export REPO=/workspace/vllm-ascend

docker ps --filter "name=^/${CONTAINER}$"
docker exec -it "${CONTAINER}" bash
```

后续若没有特别说明，命令都在这个 Docker shell 中执行：

```bash
cd /workspace/vllm-ascend
export REPO=/workspace/vllm-ascend
```

如果仓库不是 `/workspace/vllm-ascend`，替换为真实路径。

## 2. 确认目标版本和代码已经生效

当前实验接入只接受 vLLM `0.23.0` 的 eager 路径。先检查版本、提交和 patch：

```bash
cd "${REPO}"

git status --short --branch
git log -3 --oneline

if [[ ! -x .venv/bin/python ]]; then
  command -v uv >/dev/null || {
    echo 'ERROR: install uv in the container before continuing' >&2
    exit 1
  }
  uv venv --system-site-packages --python 3.12 .venv
fi
export PYTHON_BIN="${REPO}/.venv/bin/python"

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata

print("vllm:", importlib.metadata.version("vllm"))
print("vllm-ascend:", importlib.metadata.version("vllm-ascend"))
PY

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import vllm_ascend.catccos_patch as patch

path = Path(patch.__file__).resolve()
source = path.read_text(encoding="utf-8")
print("active patch:", path)
for marker in (
    "_catccos_a5_output_is_reduced",
    "_catccos_maybe_reduce_final_output",
    "production_outer_reduction",
):
    print(marker, "OK" if marker in source else "MISSING")
PY
```

期望：

- vLLM 版本为 `0.23.0`；
- 分支包含提交 `b0c8ff6e0` 和 `3059ddd2f`，或它们的后续提交；
- `active patch` 指向准备测试的仓库，而不是另一个 site-packages；
- 三个 marker 都是 `OK`；
- 服务启动参数包含 `--enforce-eager`。

这里的 `.venv` 继承官方镜像已经安装好的 Torch/vLLM 依赖，仅用于检查和单测；
所有 Python 命令仍通过 `.venv/bin/python` 执行。运行服务的环境与检查环境还
必须导入同一个 `vllm_ascend.catccos_patch`。

如果容器里的 vLLM 比 `0.23.0` 更新，不要继续使用当前 monkey patch。较新
MoERunner 的归约接口不同，应迁移到该版本的 `output_is_reduced`/MoE kernel
接口后重新验收。

## 3. 更新仓库并运行控制流单测

确认工作区没有需要保留的本地修改后，拉取实验分支：

```bash
cd "${REPO}"
git fetch origin codex/megamoe-vllm-v023
git switch codex/megamoe-vllm-v023
git pull --ff-only origin codex/megamoe-vllm-v023
```

如果 `git status` 不干净，不要用 reset 或 checkout 覆盖；先保存或提交自己的
修改。

使用项目虚拟环境运行单测：

```bash
cd "${REPO}"
.venv/bin/python -m unittest discover \
  -s tests/ut -p test_catccos_a5.py -v
```

期望 `Ran 19 tests` 和 `OK`。其中四个关键测试分别保证：

1. CatCCOS 输出跳过外层 TP all-reduce；
2. native 输出仍执行原来的归约；
3. CatCCOS forward 成功后设置“已经归约”；
4. 小 M 等 native fallback 会清除上一次的标志。

如果容器没有项目 `.venv`，应使用镜像已有的受控 Python 环境补齐测试依赖，
不要因此跳过真实 NPU probe。单测只证明控制流，不证明算子数值。

## 4. 重启为 layer0 probe 服务

环境变量只在进程启动时读取，因此更新源码后必须停止旧 vLLM 服务并启动新
进程。只停止当前用户自己的服务进程。

先确认当前容器里的服务状态：

```bash
ps -ef | grep '[v]llm serve' || true
curl -fsS http://127.0.0.1:28001/health || true
```

如果旧服务运行在原终端，回到该终端按 `Ctrl-C` 并等待 worker 退出。不要使用
模糊的 `pkill python`，也不要停止不属于自己的 NPU 进程。

在容器的服务终端中：

```bash
cd "${REPO}"

export MODEL_PATH=/model
export CATCCOS_SOURCE=/workspace/catccos
export CATCCOS_BUILD_DIR=/workspace/catccos/build_torch_a5
export NPU_DEVICES=4,5,6,7
export PORT=28001
export SERVED_MODEL_NAME=qwen3-catccos

export PROBE_RUN_ID=layer0-fix-$(date +%Y%m%d-%H%M%S)
echo "${PROBE_RUN_ID}" >/tmp/catccos-layer0-run-id

export PROBE_TOKEN_COUNTS=177
export PROBE_MOE_INSTANCE_IDS=0
export PROBE_DUMP_WEIGHTS=0
export PROBE_OUTPUT_ROOT=/home/z00956592/catccos-probe-results
export CATCCOS_MINM=64

set -o pipefail
bash examples/megamoe_qwen3/run_first_layer_reduction_probe.sh \
  2>&1 | tee /tmp/catccos-layer0-service.log
```

说明：

- `PROBE_MOE_INSTANCE_IDS=0` 只比较第一个 MoE 实例；
- `M=177` 对应仓库内固定 dragon/Perg prompt；
- `PROBE_DUMP_WEIGHTS=0` 先做轻量验收，失败后再设为 `1` 保存权重；
- 这个进程会一直占用当前终端，保持它运行。

启动阶段必须看到 CatCCOS enable、每个 EP rank 的初始化和 MXFP8 权重转换日志。
若没有这些日志，说明实际走的不是 CatCCOS。

## 5. 发送固定请求并分析 layer0

在宿主机打开第二个终端，再进入同一容器：

```bash
docker exec -it megamoe-vllm bash
cd /workspace/vllm-ascend

export PYTHON_BIN=/workspace/vllm-ascend/.venv/bin/python
export PROBE_RUN_ID=$(cat /tmp/catccos-layer0-run-id)
export PORT=28001
export SERVED_MODEL_NAME=qwen3-catccos
export PROBE_EXPECTED_PROMPT_TOKENS=177
export PROBE_MAX_TOKENS=1
export PROBE_OUTPUT_ROOT=/home/z00956592/catccos-probe-results

bash examples/megamoe_qwen3/send_first_layer_probe_request.sh
```

客户端会等待四个 rank、打印汇总并生成 tar.gz。也可以再次手动分析：

```bash
export RESULT_DIR=/home/z00956592/catccos-probe-results/${PROBE_RUN_ID}/native-catccos

"${PYTHON_BIN}" examples/megamoe_qwen3/analyze_first_layer_reduction_probe.py \
  "${RESULT_DIR}"
```

### Gate A：接线路径判定

每个 rank 的摘要必须同时包含：

```text
comm=ALLGATHER
default_outer=tp-all-reduce
production_outer=identity
```

服务日志必须出现：

```text
Skipping outer TP all-reduce for the already-reduced CatCCOS A5 output
```

可直接检查：

```bash
grep -E \
  'Enabled CatCCOS|Initialized CatCCOS|Executing CatCCOS|Skipping outer TP' \
  /tmp/catccos-layer0-service.log
```

含义是：当前 ALLGATHER native 默认需要外层归约，但本次返回的是 CatCCOS 已归约
输出，所以生产路径跳过该动作。如果仍显示 `production_outer=tp-all-reduce`，
或者日志没有 skip 消息，Gate A 失败。

### Gate B：layer0 数值判定

重点只看以下比较：

```text
native_reduced_vs_catccos_pre_reduce
```

各指标含义：

| 指标 | 含义 | 当前参考值 |
| --- | --- | ---: |
| cosine | 向量方向是否一致，越接近 1 越好 | 约 `0.99929` |
| relative L2 | 总误差相对 native 输出的比例，越小越好 | 约 `0.04435` |
| norm ratio | CatCCOS/native 输出模长比例，越接近 1 越好 | `0.9668–0.9673` |

当前结果足以说明“比较阶段对了”和“二次归约已定位”，但约 4.4% relative L2
是否可接受仍必须由后续端到端准确率决定。建议分两级 gate：

- 接线调试门槛：cosine `>= 0.99`、norm ratio 在 `[0.9, 1.1]`，且四个 rank
  的 CatCCOS 输出一致；
- 数值收敛目标：建立 native 波动/量化基线后，再收紧 relative L2。不要未经
  模型评估就把 `0.044` 宣布为正常误差。

以下两项只用于诊断：

- `native_local_vs_catccos_pre_reduce` 不要求相等，因为一个是 rank 局部贡献，
  一个是完整输出；
- `catccos_post_reduce` 仍会接近 CatCCOS 的 4 倍，因为 probe 故意在副本上
  执行旧动作。真实生产返回值是 `catccos_pre_reduce`。

用保存的张量直接检查四个 rank 的完整输出是否一致：

```bash
"${PYTHON_BIN}" - "${RESULT_DIR}" <<'PY'
import sys
from pathlib import Path

import torch

paths = sorted(Path(sys.argv[1]).glob("first-selected-rank*.pt"))
if len(paths) != 4:
    raise SystemExit(f"expected 4 rank dumps, found {len(paths)}")

payloads = [
    torch.load(path, map_location="cpu", weights_only=False)["tensors"]
    for path in paths
]
for name in ("native_reduced", "catccos_pre_reduce"):
    reference = payloads[0][name].float().reshape(-1)
    for rank, payload in enumerate(payloads[1:], start=1):
        current = payload[name].float().reshape(-1)
        delta = current - reference
        relative_l2 = delta.norm() / reference.norm().clamp_min(1e-12)
        cosine = torch.nn.functional.cosine_similarity(
            reference,
            current,
            dim=0,
        )
        print(
            name,
            f"rank0-vs-rank{rank}",
            f"cosine={cosine.item():.9f}",
            f"relative_l2={relative_l2.item():.9g}",
            f"max_abs={delta.abs().max().item():.9g}",
        )
PY
```

完整输出在各 rank 上应相同或只有约定范围内的极小数值误差。如果 CatCCOS
跨 rank 不一致，即使每个 rank 各自对 native 的 cosine 很高，也不能通过。

如果 Gate B 失败，停止端到端和性能测试，重新以
`PROBE_DUMP_WEIGHTS=1` 运行，并检查路由 ID/权重、gate weight、MXFP8 scale、
权重布局和所有 rank 的输入分片。

## 6. 固定请求端到端 A/B

同一个服务进程里运行 native 后再运行 CatCCOS probe 会增加诊断副作用。端到端
A/B 应使用两个全新服务进程，并保持同一镜像、模型、物理 NPU、TP/EP、服务
参数和请求 JSON。推荐顺序运行，避免两套服务争抢 NPU。

### 6.1 保存固定请求

在容器中创建请求文件：

```bash
export MODEL=qwen3-catccos

cat >/tmp/catccos-e2e-request.json <<'JSON'
{
  "model": "qwen3-catccos",
  "messages": [{
    "role": "user",
    "content": "What is 17 * 23? Return only the number."
  }],
  "temperature": 0,
  "seed": 42,
  "max_tokens": 32,
  "chat_template_kwargs": {"enable_thinking": false}
}
JSON
```

这里使用 `cat` 只是终端内创建临时请求文件，不修改仓库。

### 6.2 native 基线

停止 probe 服务后，在第一个终端启动 native：

```bash
cd "${REPO}"

export MODEL_PATH=/model
export PORT=28001
export SERVED_MODEL_NAME=qwen3-catccos
export NPU_DEVICES=4,5,6,7

set -o pipefail
bash examples/megamoe_qwen3/run_probe_in_container.sh baseline \
  2>&1 | tee /tmp/native-e2e-service.log
```

第二个终端请求并保存结果：

```bash
for max_tokens in 1 2 32; do
  sed "s/\"max_tokens\": 32/\"max_tokens\": ${max_tokens}/" \
    /tmp/catccos-e2e-request.json | \
    curl --fail --silent --show-error \
      http://127.0.0.1:28001/v1/chat/completions \
      -H 'Content-Type: application/json' \
      --data-binary @- \
      >"/tmp/native-e2e-${max_tokens}.json"
done
```

### 6.3 CatCCOS prefill + decode

停止 native 服务，重新启动 CatCCOS。必须设置 `CATCCOS_MINM=1`，否则 decode
的 `M=1` 会回退 native，不能证明 CatCCOS decode：

```bash
cd "${REPO}"

export MODEL_PATH=/model
export PORT=28001
export SERVED_MODEL_NAME=qwen3-catccos
export NPU_DEVICES=4,5,6,7
export CATCCOS_MINM=1
export CATCCOS_SOURCE=/workspace/catccos
export CATCCOS_BUILD_DIR=/workspace/catccos/build_torch_a5

set -o pipefail
bash examples/megamoe_qwen3/run_probe_in_container.sh catccos \
  2>&1 | tee /tmp/catccos-e2e-service.log
```

第二个终端发送完全相同的请求：

```bash
for max_tokens in 1 2 32; do
  sed "s/\"max_tokens\": 32/\"max_tokens\": ${max_tokens}/" \
    /tmp/catccos-e2e-request.json | \
    curl --fail --silent --show-error \
      http://127.0.0.1:28001/v1/chat/completions \
      -H 'Content-Type: application/json' \
      --data-binary @- \
      >"/tmp/catccos-e2e-${max_tokens}.json"
done
```

确认日志同时出现多 token 与单 token 路径：

```bash
grep -E \
  'Executing CatCCOS A5 multi-token|Executing CatCCOS A5 single-token|Skipping outer TP' \
  /tmp/catccos-e2e-service.log
```

如果服务日志只显示 multi-token，没有 single-token，则只验证了 prefill。

### 6.4 比较响应

```bash
"${PYTHON_BIN}" - <<'PY'
import json

def load(path):
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    if "error" in data:
        raise SystemExit(f"{path}: {data['error']}")
    choice = data["choices"][0]
    return choice["message"]["content"], choice["finish_reason"], data["usage"]

for max_tokens in (1, 2, 32):
    native = load(f"/tmp/native-e2e-{max_tokens}.json")
    catccos = load(f"/tmp/catccos-e2e-{max_tokens}.json")
    print(f"max_tokens={max_tokens}")
    print("  native :", native)
    print("  catccos:", catccos)
    print("  same text:", native[0] == catccos[0])
    print("  same finish reason:", native[1] == catccos[1])
PY
```

固定贪心请求的第一道 gate 是输出文本、finish reason 和生成 token 数一致。更强
的比较应开启 API logprobs，逐步检查 top-1 token 是否一致以及 native token 在
CatCCOS 分布中的 logprob 变化。不要只比较最后一句自然语言是否“意思差不多”。

### 6.5 区分 prefill 和 decode 故障

对同一个 prompt 依次请求：

1. `max_tokens=1`：采样第一个 token，只依赖 prefill logits；
2. `max_tokens=2`：产生第二个 token前至少执行一次 `M=1` decode。

判定：

| 现象 | 首个怀疑边界 |
| --- | --- |
| 第一个 token 就不同 | prefill、layer0/其他层数值或 LM head 放大误差 |
| 第一个相同、第二个首次不同 | `M=1` decode 路径 |
| token 相同但 logprob 差异持续扩大 | 数值误差在后续层累积 |
| 请求超时或 rank 卡住 | collective 顺序、CatCCOS 状态或资源问题 |

## 7. 全模型准确率 Gate C

固定请求通过后，再用 AISBench 跑相同的 GSM8K 子集：

```bash
cd "${REPO}"

AISBENCH_ROOT=/data/tools/benchmark \
MODEL_PATH=/model \
SERVED_MODEL_NAME=qwen3-catccos \
AISBENCH_PORT=28001 \
MODE=accuracy \
RUN_LABEL=catccos-tp4-smoke \
NUM_PROMPTS=32 \
NUM_WARMUPS=1 \
AISBENCH_MAX_OUT_LEN=512 \
bash examples/megamoe_qwen3/run_aisbench_gsm8k.sh
```

先对 native 跑相同命令，只修改 `RUN_LABEL`，再顺序运行 CatCCOS。必须比较：

- failed request 数；
- 每个样本的 prediction，而不只是最终总分；
- 准确率差异；
- 超时、乱码、空输出和异常结束；
- CatCCOS 日志是否覆盖预期的 prefill/decode shape。

32 条 smoke 全部正常后，再跑完整 GSM8K。Gate C 的精确容差需要在评测前确定；
最低要求是零失败请求、没有系统性输出异常、没有无法解释的大幅分数下降。

## 8. 泛化矩阵

端到端通过只代表当前样本。至少补齐以下矩阵：

| 维度 | 建议取值 |
| --- | --- |
| token rows M | `1, 2, 8, 32, 64, 177, 256, 512, 1024` |
| MoE 层 | 第一个、中间、最后一个 MoE 实例 |
| TP/EP | `1/1, 2/2, 4/4` |
| weight quant backend | `npu, cpu` |
| 并发 | `1, 4, 8, 16, 32` |
| 请求类型 | prefill 主导、decode 主导、长上下文、混合 batch |

对于每个新配置，至少重复 Gate A/B 的轻量 probe 和固定请求。不要把 TP4 layer0
结果直接推广到 TP1、TP2、所有层或所有 M。

## 9. 性能测试前清理

probe 会额外运行 native、clone 大张量、同步设备并写磁盘，绝对不能用于性能
数据。停止 probe 服务，清除所有 debug 变量并启动新的 CatCCOS 进程：

```bash
unset VLLM_ASCEND_CATCCOS_DEBUG_DIR
unset VLLM_ASCEND_CATCCOS_DEBUG_TOKEN_COUNTS
unset VLLM_ASCEND_CATCCOS_DEBUG_MOE_INSTANCE_IDS
unset VLLM_ASCEND_CATCCOS_DEBUG_ORDER
unset VLLM_ASCEND_CATCCOS_DEBUG_MAX_CALLS_PER_LAYER
unset VLLM_ASCEND_CATCCOS_DEBUG_COSINE_THRESHOLD
unset VLLM_ASCEND_CATCCOS_DEBUG_RELATIVE_L2_THRESHOLD
unset VLLM_ASCEND_CATCCOS_DEBUG_DUMP_TENSORS
unset VLLM_ASCEND_CATCCOS_DEBUG_DUMP_SELECTED
unset VLLM_ASCEND_CATCCOS_DEBUG_DUMP_WEIGHTS
```

性能 A/B 必须使用相同物理 NPU、镜像、模型、TP/EP、服务参数、输入/输出长度、
并发、warmup 和请求数。每组至少重复三次取中位数，记录 TTFT、TPOT/ITL、吞吐、
p50/p99 和 HBM。

特别注意：若 `CATCCOS_MINM>1`，单 token decode 会走 native fallback，此时不能
把 decode 性能归因于 CatCCOS。

## 10. 最终验收表

| 项目 | 通过条件 | 状态 |
| --- | --- | --- |
| 目标版本 | vLLM `0.23.0`、active patch 指向目标仓库 | `[ ]` |
| 控制流单测 | 19/19 通过 | `[ ]` |
| Gate A | 四个 rank 均为 `production_outer=identity` | `[ ]` |
| layer0 阶段 | `native_reduced ≈ catccos_pre_reduce` | `[ ]` |
| layer0 数值 | 达到预先约定的 cosine/L2/norm 阈值 | `[ ]` |
| prefill E2E | 固定请求第一个 token/输出与 native 一致 | `[ ]` |
| decode E2E | `CATCCOS_MINM=1` 且 M=1 路径/输出通过 | `[ ]` |
| GSM8K smoke | 零失败、无明显准确率回退 | `[ ]` |
| 完整准确率 | 达到预先约定的质量容差 | `[ ]` |
| 泛化矩阵 | 目标 M、层、TP/EP、并发已覆盖 | `[ ]` |
| 性能环境 | debug 全关闭，A/B 参数完全一致 | `[ ]` |

只有前九项达到目标后，性能数字才具有解释意义。
