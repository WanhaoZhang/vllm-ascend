# CatCCOS same-input MoE probe

For the targeted first-layer four-stage reduction experiment, use
[CATCCOS_FIRST_LAYER_REDUCTION_PROBE.md](CATCCOS_FIRST_LAYER_REDUCTION_PROBE.md).
It records native local/reduced and CatCCOS pre/post-reduction outputs without
changing the real output returned to the model.

This opt-in diagnostic runs two MoE implementations on independent clones of
the same real vLLM input. It synchronizes and freezes the first output before
starting the second implementation, compares the outputs, writes one JSONL
record per layer and rank, and saves the first significant mismatch as a
PyTorch file.

The probe is for correctness diagnosis only. It adds synchronization, device
copies, a second MoE execution, hashing, and file I/O. Never use its latency or
throughput as a performance result.

## What it separates

For the known 177-token prompt with `CATCCOS_MINM=64`:

- prefill, M=177: CatCCOS is eligible;
- decode, normally M=1: native fallback is used.

Selecting only M=177 therefore isolates the prefill MoE calls and skips model
profile calls with other token counts. A `max_tokens=1` request is sufficient
to run the complete prefill probe.

The in-process native result is a comparison path, not an independent golden.
Always validate the probe itself with repeated-backend and reversed-order
runs before attributing a mismatch to CatCCOS.

## 现有四卡容器现场执行手册

本节适用于容器已经安装好 vLLM、vLLM-Ascend、模型和 CatCCOS 的场景。
`run_probe_in_container.sh` 默认沿用已知问题现场的配置：物理卡 4-7、端口
28001、CPU 权重量化、`CATCCOS_MINM=64`、模型路径
`/home/weights/Qwen3-30B-A3B-Instruct-2507`。

### 0. 执行原则

- 四个 order 必须分别冷启动服务，不能在同一进程里连续切换。
- 每个服务只发送一次完全相同的错误 prompt。
- prefill 实验使用 `temperature=0`、`max_tokens=1`、`stream=false`。
- 探针会让每个被选中的 MoE 调用执行两次，结果不能用于性能分析。
- 交叉对比只证明 native 与 CatCCOS 不一致，不能单独决定谁正确。

先确认容器实际导入的 vLLM-Ascend 已包含脚本 commit `681cbfb6f` 或更新版本：

```bash
cd /path/to/vllm-ascend
git rev-parse HEAD

python -c '
import vllm_ascend
import vllm_ascend.catccos_debug as debug
print(vllm_ascend.__file__)
print(debug.__file__)
'
```

启动脚本还会再次检查实际导入的 `catccos_patch.py` 是否包含探针，避免更新了
checkout、但 `vllm` 实际仍从另一个安装目录导入旧代码。

### 1. 第一次先用 M 窗口，避免 token 数变化导致白跑

`M=177` 是历史请求的已知值，不是探针的硬编码。第一轮建议监听一个小窗口：

```bash
cd /path/to/vllm-ascend

export PROBE_RUN_ID=prompt-window-20260827
export CATCCOS_MINM=64
export PROBE_TOKEN_COUNTS="$(seq -s, 160 192)"
export PROBE_MAX_CALLS_PER_LAYER=1
```

不要直接监听所有 M。模型启动时也会执行 profile/dummy forward；范围过宽可能
在真实请求前触发双路探针，增加启动时间并污染结果。

### 2. 启动第一轮 native-native

在第一个容器终端运行：

```bash
bash examples/megamoe_qwen3/run_probe_in_container.sh native-native
```

在第二个容器终端等待服务健康：

```bash
curl -fsS http://127.0.0.1:28001/health
```

发送请求前先确认窗口没有命中启动阶段的 profile：

```bash
find \
  "/home/z00956592/catccos-probe-results/${PROBE_RUN_ID}/native-native" \
  -name 'probe-rank*.jsonl' -print
```

此时正常应没有输出。如果已经出现 JSONL，停止服务，缩小 M 窗口并换一个新的
`PROBE_RUN_ID` 后重启。

将固定错误 prompt 原样保存为一个文本文件后发送一次：

```bash
jq -n --rawfile prompt /path/to/fixed_bad_prompt.txt '
  {
    model: "qwen3-catccos",
    messages: [{role: "user", content: $prompt}],
    temperature: 0,
    max_tokens: 1,
    stream: false
  }
' | curl -fsS http://127.0.0.1:28001/v1/chat/completions \
    -H 'Content-Type: application/json' \
    --data-binary @-
```

如果原问题使用的不是 `/v1/chat/completions`，必须继续使用原来的请求体和接口，
因为 chat template、system message 和 generation prompt 都可能改变实际 M。

请求结束后查看实际命中的 M：

```bash
jq -r '.token_count' \
  "/home/z00956592/catccos-probe-results/${PROBE_RUN_ID}/native-native/probe-rank000.jsonl" \
  | sort -nu
```

记下真实 M，然后在启动服务的终端按 Ctrl-C 停止服务。假设实际输出为 175，
后续三轮固定：

```bash
export PROBE_TOKEN_COUNTS=175
```

### 3. 分别冷启动剩余三个 order

以下三条命令每次只运行一条。每轮都等待健康、发送同一个请求一次、等待结果
落盘，然后 Ctrl-C 停止服务：

```bash
bash examples/megamoe_qwen3/run_probe_in_container.sh catccos-catccos
```

```bash
bash examples/megamoe_qwen3/run_probe_in_container.sh native-catccos
```

```bash
bash examples/megamoe_qwen3/run_probe_in_container.sh catccos-native
```

推荐的完整顺序为：

```text
native-native
catccos-catccos
native-catccos
catccos-native
```

同一个 `PROBE_RUN_ID` 下，每个 order 只能写入一次。脚本发现目标目录非空时会
拒绝启动，防止新旧 JSONL 被追加到一起。重跑时应更换 `PROBE_RUN_ID`。

### 4. 结果保存在哪里

默认目录结构为：

```text
/home/z00956592/catccos-probe-results/
└── <PROBE_RUN_ID>/
    ├── native-native/
    │   ├── probe-rank000.jsonl
    │   ├── probe-rank001.jsonl
    │   ├── probe-rank002.jsonl
    │   ├── probe-rank003.jsonl
    │   ├── first-mismatch-rank000.json
    │   └── first-mismatch-rank000-<layer>.pt
    ├── catccos-catccos/
    ├── native-catccos/
    └── catccos-native/
```

| 文件 | 保存内容 |
| --- | --- |
| `probe-rankXXX.jsonl` | 该 rank 每个被选中 MoE 层/调用的完整摘要；即使没有明显异常也会写。 |
| `first-mismatch-rankXXX.json` | 该 rank 首个超过阈值的层、指标和对应 `.pt` 文件名。 |
| `first-mismatch-rankXXX-<layer>.pt` | 首个明显异常层的真实输入、路由和两份冻结输出。 |

`.pt` 默认包含：

```text
metadata
tensors.hidden_states
tensors.router_logits
tensors.expert_idx
tensors.gate_weight
tensors.first_output
tensors.second_output
```

四个 order 中两份输出的含义为：

| Order | `first_output` | `second_output` |
| --- | --- | --- |
| `native-native` | native 第一次 | native 第二次 |
| `catccos-catccos` | CatCCOS 第一次 | CatCCOS 第二次 |
| `native-catccos` | native | CatCCOS |
| `catccos-native` | CatCCOS | native |

默认不保存完整 MXFP8 权重。只有设置 `PROBE_DUMP_WEIGHTS=1` 时，首次异常
`.pt` 才会额外包含 `weights`；四卡可能占用数 GiB，只应在已经定位坏层后重跑。

结果状态的含义：

- 有 JSONL、没有 `.pt`：命中了 M，但没有指标超过异常阈值。
- 有 JSONL 和 `.pt`：命中了 M，并保存了该 rank 的首个明显异常层。
- order 目录为空：实际 M 不在过滤范围、请求没有执行完成，或服务没有使用探针。

### 5. 先检查结果是否完整

确认四轮都有四个 rank 文件，并比较各 rank 的记录行数：

```bash
RESULT_ROOT="/home/z00956592/catccos-probe-results/${PROBE_RUN_ID}"

for order in \
  native-native \
  catccos-catccos \
  native-catccos \
  catccos-native
do
  echo "===== ${order} ====="
  wc -l "${RESULT_ROOT}/${order}"/probe-rank*.jsonl
done
```

同一个 order 下四个 rank 的行数和 token count 应一致。缺 rank、行数不同或
请求挂起时，先按不完整运行处理，不进入数值归因。

### 6. 找四轮各自的第一个异常层

先看 rank 0：

```bash
for order in \
  native-native \
  catccos-catccos \
  native-catccos \
  catccos-native
do
  echo "===== ${order} ====="
  jq -s '
    sort_by(.moe_instance_id)
    | map(select(.significant_mismatch))
    | first
    | {
        layer,
        moe_instance_id,
        token_count,
        order,
        cosine: .metrics.cosine_similarity,
        relative_l2: .metrics.relative_l2,
        max_abs: .metrics.max_abs_diff,
        norm_ratio: .metrics.norm_ratio,
        sign_flip: .metrics.sign_flip_ratio
      }
  ' "${RESULT_ROOT}/${order}/probe-rank000.jsonl"
done
```

输出为 `null` 表示该 rank 没有超过当前阈值的层，不代表两份输出 bitwise 完全
相等；仍可查看 JSONL 中的 `exact_equal`、hash 和连续数值指标。

对某个 order 比较四个 rank 的首错位置：

```bash
for file in "${RESULT_ROOT}/native-catccos"/probe-rank*.jsonl; do
  jq -s --arg file "$(basename "${file}")" '
    sort_by(.moe_instance_id)
    | map(select(.significant_mismatch))
    | first as $bad
    | {
        file: $file,
        layer: $bad.layer,
        moe_instance_id: $bad.moe_instance_id,
        cosine: $bad.metrics.cosine_similarity,
        relative_l2: $bad.metrics.relative_l2
      }
  ' "${file}"
done
```

### 7. 用四个 order 做归因

先看两个同后端控制组，再看两个交叉组：

| 观察 | 优先结论与下一步 |
| --- | --- |
| `native-native` 已发散 | native 路径存在重入、状态、通信或输出生命周期问题；不能把本轮 native 当黄金。 |
| `catccos-catccos` 已发散 | CatCCOS 重复执行、同步完成或 runtime 生命周期不稳定；先查非确定性。 |
| 两个同后端组稳定，两个交叉组在相同层稳定发散 | 存在稳定的接口契约或数值差异；继续用独立 BF16/MXFP8 reference 裁决谁正确。 |
| `native-catccos` 与 `catccos-native` 首错层或误差差异很大 | 调用顺序、stream、buffer 复用、collective 状态或 runtime re-entry 问题。 |
| 只有部分 rank 发散或首错层不同 | EP 通信、global/local expert id、expert placement、rank offset 或同步问题。 |
| 四个 rank 在同一层以近似指标发散 | 更像系统性的权重/scale/layout、路由权重、FFN 或 combine 契约问题。 |

不能仅凭 `native-catccos` 不一致就认定 CatCCOS 算子错误。独立 BF16 全专家重算
或同一 MXFP8 契约的 standalone reference 具有更高裁决优先级。

### 8. 打开首错 `.pt`

```bash
CASE_PATH="${RESULT_ROOT}/native-catccos/first-mismatch-rank000-<layer>.pt"

python - "${CASE_PATH}" <<'PY'
import sys

import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
metadata = payload["metadata"]

print("layer:", metadata["layer"])
print("order:", metadata["order"])
print("M:", metadata["token_count"])
print("metrics:", metadata["metrics"])
for name, tensor in payload["tensors"].items():
    print(name, tuple(tensor.shape), tensor.dtype)
print("has_weights:", "weights" in payload)
PY
```

这份文件可以用于 standalone replay。`metadata["first_backend"]` 和
`metadata["second_backend"]` 决定两份输出的后端含义，不能只根据 tensor
名称猜测。

### 9. 把结果保存在容器外

Ctrl-C 或 `docker stop` 不会删除同一个容器的文件；`docker rm` 会删除未挂载的
容器文件系统。若结果目录不是宿主机 bind mount，应在删除容器前从宿主机执行：

```bash
docker cp \
  <container-name>:/home/z00956592/catccos-probe-results/${PROBE_RUN_ID} \
  ./catccos-probe-${PROBE_RUN_ID}
```

也可以启动实验前将 `PROBE_OUTPUT_ROOT` 设置为容器中的宿主机挂载目录。结果中
包含模型 hidden states、router logits 和路由信息，应按模型调试数据管理。

### 10. 后续同时抓 prefill 和 decode

复现原始 `MINM=64` prefill 问题时，不要混入 decode 变量。完成 prefill 四轮后，
使用新的 `PROBE_RUN_ID` 单独运行：

```bash
export PROBE_RUN_ID=prompt-prefill-decode-20260827
export CATCCOS_MINM=1
export PROBE_TOKEN_COUNTS="1,$(seq -s, 160 192)"
```

此时请求必须设置 `max_tokens>=2`，否则通常只会从 prefill logits 采出第一个输出
token，不一定真正执行 M=1 decode forward。每层、每个被选中 M 默认只记录第一
次调用。

路径不同时可以覆盖默认值：

```bash
MODEL_PATH=/other/model \
CATCCOS_SOURCE=/other/catccos \
CATCCOS_BUILD_DIR=/other/catccos/build \
PROBE_OUTPUT_ROOT=/other/results \
bash examples/megamoe_qwen3/run_probe_in_container.sh native-native
```

## Four-card launch

Use a new output directory for every run. Start with `native-native`:

```bash
MODE=catccos \
CONTAINER_NAME=megamoe-probe-native-native \
PORT=28001 \
NPU_DEVICES=0,1,2,3 \
MODEL_PATH=/data/models/Qwen3-30B-A3B \
VLLM_ASCEND_SOURCE=/data/src/vllm-ascend \
CATCCOS_SOURCE=/data/src/catccos \
CATCCOS_MINM=64 \
CATCCOS_DEBUG_HOST_DIR=/data/catccos-probe/native-native \
CATCCOS_DEBUG_TOKEN_COUNTS=177 \
CATCCOS_DEBUG_ORDER=native-native \
CATCCOS_DEBUG_MAX_CALLS_PER_LAYER=1 \
CATCCOS_DEBUG_COSINE_THRESHOLD=0.99 \
CATCCOS_DEBUG_RELATIVE_L2_THRESHOLD=0.1 \
bash examples/megamoe_qwen3/run_docker.sh
```

Send the fixed prompt once with greedy decoding and `max_tokens=1`. Then stop
only that probe container and repeat with three new container names, ports,
output directories, and orders:

```text
catccos-catccos
native-catccos
catccos-native
```

All ranks must use the same order and token-count filter. Do not enable the
probe on only one rank because both implementations contain collectives.

## Output

The host output directory contains:

```text
probe-rank000.jsonl
probe-rank001.jsonl
probe-rank002.jsonl
probe-rank003.jsonl
first-mismatch-rank000.json
first-mismatch-rank000-<layer>.pt
...
```

Each JSONL record includes:

- global and EP rank;
- layer name, MoE instance, call ID, phase hint, and M;
- comparison order, cosine threshold, and relative-L2 threshold;
- exact equality, SHA-256, cosine, max/mean absolute error, relative L2,
  norm ratio, and sign-flip ratio;
- shape, dtype, stride, contiguity, and SHA-256 for input, router logits,
  expert IDs, and gate weights;
- selected-expert range and token histogram, plus gate-weight range and row
  sums, for spotting global/local expert mapping or combine-weight problems;
- MXFP8 weight and scale layout metadata.

The `.pt` file contains the real hidden states, router logits, expert IDs,
gate weights, and both frozen outputs for the first layer below the configured
cosine threshold or above the relative-L2 threshold. Full MXFP8 weights are
excluded by default. Set
`CATCCOS_DEBUG_DUMP_WEIGHTS=1` only for a targeted rerun after finding the bad
layer; it can consume several GiB across four ranks.

## Reading the result

Inspect the first mismatch on rank 0:

```bash
jq -s '
  sort_by(.moe_instance_id)
  | map(select(.significant_mismatch))
  | first
  | {
      layer,
      token_count,
      order,
      cosine: .metrics.cosine_similarity,
      max_abs: .metrics.max_abs_diff,
      relative_l2: .metrics.relative_l2,
      sign_flip: .metrics.sign_flip_ratio
    }
' /data/catccos-probe/native-catccos/probe-rank000.jsonl
```

Interpret the four runs in this order:

| Result | Interpretation |
| --- | --- |
| `native-native` diverges | Native reference is stateful, aliased, or non-reentrant; do not use it as golden. |
| `catccos-catccos` diverges | CatCCOS repeated execution or runtime completion is non-deterministic. |
| Same-backend runs pass, cross-backend runs agree across order | Stable native/CatCCOS numerical or contract difference. |
| `native-catccos` and `catccos-native` disagree strongly | Order, stream, buffer lifetime, or runtime re-entry problem. |
| Only TP4 fails after TP1 passes | EP communication, expert placement, or global/local expert mapping. |

After locating the first stable bad layer, rerun only that prompt with weight
dumping enabled and replay its `.pt` case outside vLLM against an independent
BF16 or same-MXFP8 reference. A native/CatCCOS difference alone does not decide
which implementation is correct.

## Probe environment variables

| Launcher variable | Default | Meaning |
| --- | ---: | --- |
| `CATCCOS_DEBUG_HOST_DIR` | empty | Absolute host output path; empty disables the probe. |
| `CATCCOS_DEBUG_TOKEN_COUNTS` | required | Comma-separated M values, for example `177` or `1,177`. |
| `CATCCOS_DEBUG_ORDER` | `native-catccos` | One of the four comparison orders above. |
| `CATCCOS_DEBUG_MAX_CALLS_PER_LAYER` | `1` | Calls recorded per selected M in each layer. |
| `CATCCOS_DEBUG_COSINE_THRESHOLD` | `0.99` | First result below this value is dumped per rank. |
| `CATCCOS_DEBUG_RELATIVE_L2_THRESHOLD` | `0.1` | First result above this value is dumped per rank. |
| `CATCCOS_DEBUG_DUMP_TENSORS` | `1` | Save the first mismatch's inputs, routes, and outputs. |
| `CATCCOS_DEBUG_DUMP_WEIGHTS` | `0` | Also save MXFP8 weights and scales; high disk and host-memory cost. |
