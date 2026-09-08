# CatCCOS 与 native MC2 服务内 A/B profiling

## 1. 分支和实现范围

本 profiling 分支基于 CatCCOS A5 正式接入提交：

```text
0a9b67667 feat(moe): integrate CatCCOS A5 backend
```

分支名称：

```text
codex/megamoe-a5-vllm-v023-service-profiling
```

该分支不包含后续 CatCCOS 输入 dump 提交。它复用 vLLM 自带的
`--profiler-config`、`/start_profile` 和 `/stop_profile`，仅增加以下
shape-qualified profiler ranges：

```text
vllm_ascend.moe.catccos.host_moe_body[M=2,H=2048,topK=8]
└── vllm_ascend.moe.catccos.adapter_host_scope[...]
    ├── vllm_ascend.moe.catccos.pre_sync_host_wait[...]
    ├── vllm_ascend.moe.catccos.input_prepare_host[...]
    ├── vllm_ascend.moe.catccos.kernel_enqueue[...]
    └── vllm_ascend.moe.catccos.post_sync_host_wait[...]  # sync=true 时存在

vllm_ascend.moe.native_mc2.host_moe_body[M=2,H=2048,topK=8]
└── vllm_ascend.moe.native_mc2.pipeline_host_scope[...]
    ├── vllm_ascend.moe.native_mc2.dispatch_enqueue[...]
    ├── vllm_ascend.moe.native_mc2.mlp_enqueue[...]
    └── vllm_ascend.moe.native_mc2.combine_enqueue[...]
```

环境变量不开时，这些 profiler ranges 不会建立：

```bash
export VLLM_ASCEND_MOE_PROFILE_RANGES=1
export VLLM_ASCEND_MOE_PROFILE_SYNC_BOUNDARIES=0
```

默认的 `host_moe_body` 从两条路径共同的 `fused_experts` 调用点开始和结束，
代码边界一致，但它是 Host range。CatCCOS 内有显式同步，而 native 默认是异步
launch，因此两个 `host_moe_body` 的 duration 不能直接相减。

本次脚本中的真实路径是：

| 轮次 | `moe_comm_type` | `fused_experts` 内部实现 |
|---|---|---|
| CatCCOS | `FUSED_MC2` | `apply_catccos` → `ascend950_dispatch_ffn_combine` |
| native baseline | `MC2` | `token_dispatch` → `_apply_mlp` → `token_combine` |

两边共同的 `fused_experts` range 对应同一个 routed-expert 替换接口，输入均为
router 选出的 `hidden_states/topk_ids/topk_weights` 和本层专家权重，输出均为
合并后的 routed expert hidden states。它们在语义上对应，但内部工作不是逐行等价：

- CatCCOS 包含 adapter 校验、初始化、显式 pre-sync、输入 dtype/layout 转换、
  融合算子 launch，以及配置开启时的 post-sync。
- native MC2 包含 dispatch 元数据构造与 launch、专家 MLP launch、combine
  元数据构造与 launch。
- 两边都不包含上游的 MC2 prepare/padding、router top-k，也不包含下游
  finalize；`build_fused_experts_input` 也在共同 range 之外。

因此要比较“服务采用这套后端付出的完整 routed-expert 成本”，看对齐模式的
`synchronized_moe_body`；要比较“CatCCOS 融合 Device kernel 本身”，则在
NPU timeline 中将它与 native dispatch、MLP、combine 的 Device critical-path
span 对齐。native 各 Device kernel 可能重叠，不能直接把 CSV 中每个 kernel 的
duration 无条件相加。

需要直接比较隔离后的 routed-expert 墙钟耗时时，再单独运行一轮：

```bash
export VLLM_ASCEND_MOE_PROFILE_SYNC_BOUNDARIES=1
```

该模式在进入共同 range 前执行一次 `torch.npu.synchronize()`，清空此前工作；
在 range 内的 MoE 调用后再次同步，再结束 range。range 名称变为
`synchronized_moe_body`。两边的开始和结束 Device 状态相同，因此该 range
才能直接比较。额外同步会改变 native 的正常 overlap，只用于诊断，不能用它
产生正式服务吞吐、TTFT 或 TPOT 数据。

名称中的 `M` 是 MC2 padding 和 TP 切分之后的真实 rank-local token 数。
在 TP4 下：

```text
全局 decode T=8     → rank-local M=2
全局 prefill T=552  → rank-local M=138
全局 prefill T=2048 → rank-local M=512
```

## 2. 在 A5 测试机器拉取分支

```bash
cd /home/z00956592/vllm-ascend-catccos

git fetch git@github.com:WanhaoZhang/vllm-ascend.git \
  codex/megamoe-a5-vllm-v023-service-profiling

git switch -C codex/megamoe-a5-vllm-v023-service-profiling FETCH_HEAD

git rev-parse HEAD
```

如果使用 editable install 或者启动脚本已经设置：

```bash
export PYTHONPATH=/home/z00956592/vllm-ascend-catccos:${PYTHONPATH:-}
```

切换分支后重启服务即可，不需要重新编译 CatCCOS 动态库。

## 3. 修改 CatCCOS 和 native 启动脚本

在原来的 `/home/z00956592/start_m1test.sh` 和
`/home/z00956592/start_m1test_native.sh` 中，在 `vllm serve` 之前加入：

```bash
export MSMONITOR_USE_DAEMON=0
export VLLM_ASCEND_MOE_PROFILE_RANGES=1
export VLLM_ASCEND_MOE_PROFILE_SYNC_BOUNDARIES=0

PROFILE_TAG="${PROFILE_TAG:?set PROFILE_TAG}"
PROFILE_ITERS="${PROFILE_ITERS:-20}"
PROFILE_DIR="/home/z00956592/profiles/${PROFILE_TAG}"

rm -rf "$PROFILE_DIR"
mkdir -p "$PROFILE_DIR"

PROFILER_CONFIG="{
  \"profiler\":\"torch\",
  \"torch_profiler_dir\":\"${PROFILE_DIR}\",
  \"torch_profiler_with_stack\":false,
  \"torch_profiler_with_memory\":false,
  \"ignore_frontend\":true,
  \"max_iterations\":${PROFILE_ITERS}
}"
```

在两个脚本的 `vllm serve` 参数中都加入：

```bash
--profiler-config "$PROFILER_CONFIG"
```

CatCCOS 启动脚本继续保留原有的：

```bash
--additional-config "$ADDCONF"
```

native 启动脚本仍然不传 `--additional-config`。因此 native 对照走的是：

```text
npu_moe_distribute_dispatch[_v2]
→ GMM1/SwiGLU
→ GMM2
→ npu_moe_distribute_combine[_v2]
```

它不是 CANN fused `_C_ascend.dispatch_ffn_combine`。

服务启动后先正常预热。只有调用 `/start_profile` 后才开始采集。不要在
profiling 同一轮开启输入 dump，否则 device-to-host 拷贝会污染时间线。

## 4. 使用原 AISBench 参数抓 CatCCOS

按照原复测报告的参数启动 CatCCOS 服务：

```bash
docker exec -d megamoe-vllm bash -lc '
  PROFILE_TAG=catccos_load_r1 PROFILE_ITERS=20 \
  bash /home/z00956592/start_m1test.sh \
  > /home/z00956592/serve_catccos_profile.log 2>&1
'
```

等待服务 `/health` 正常，然后后台启动原来的并发 8、200 题 perf 压测：

```bash
docker exec megamoe-vllm bash -lc '
  bash /home/z00956592/ab_test/run_aisbench_ab.sh \
    catccos perf 28001 200 load \
  > /home/z00956592/catccos_profile_load.log 2>&1 &
'
```

等待运行请求数达到 8：

```bash
docker exec megamoe-vllm bash -lc '
while true; do
  running=$(curl -s http://127.0.0.1:28001/metrics |
    awk "/^vllm:num_requests_running/ {sum += \$NF}
         END {print int(sum)}")
  echo "running=${running}"
  [ "$running" -ge 8 ] && break
  sleep 0.2
done
'
```

开始采集：

```bash
docker exec megamoe-vllm \
  curl -fsS -X POST http://127.0.0.1:28001/start_profile
```

`max_iterations=20` 会把 worker 采集限制为 20 个模型 iteration。等待数秒
后调用 stop，确保已经结束并落盘：

```bash
sleep 5

docker exec megamoe-vllm \
  curl -fsS -X POST http://127.0.0.1:28001/stop_profile
```

结果目录：

```text
/home/z00956592/profiles/catccos_load_r1
```

## 5. 使用相同请求抓 native

先停止 CatCCOS 服务并清理残留：

```bash
docker exec megamoe-vllm bash /home/z00956592/kill_vllm.sh
```

启动 native 服务：

```bash
docker exec -d megamoe-vllm bash -lc '
  PROFILE_TAG=native_load_r1 PROFILE_ITERS=20 \
  bash /home/z00956592/start_m1test_native.sh \
  > /home/z00956592/serve_native_profile.log 2>&1
'
```

发送相同的并发 8、200 题请求：

```bash
docker exec megamoe-vllm bash -lc '
  bash /home/z00956592/ab_test/run_aisbench_ab.sh \
    baseline perf 28001 200 load \
  > /home/z00956592/native_profile_load.log 2>&1 &
'
```

同样等 `vllm:num_requests_running` 达到 8，然后执行：

```bash
docker exec megamoe-vllm \
  curl -fsS -X POST http://127.0.0.1:28001/start_profile

sleep 5

docker exec megamoe-vllm \
  curl -fsS -X POST http://127.0.0.1:28001/stop_profile
```

结果目录：

```text
/home/z00956592/profiles/native_load_r1
```

## 6. 受控 decode 和 prefill profile

原 AISBench 请求可以得到真实负载下的混合时间线。profiler range 名称中的
`M` 可以区分纯 decode 与 prefill/decode 混批：

- `M=2`：TP4、并发 8 下的纯 decode。
- `M>2`：prefill 或 prefill/decode 混批。

如果需要干净的纯 decode trace，可以启动恰好 8 个长输出请求并设置
`ignore_eos=true`。等首 token 已经返回、8 个请求全部进入 decode 后，再调用
`/start_profile`，用 `max_iterations=10` 抓 10 个 iteration。

如果需要受控 prefill，重启服务并设置：

```text
PROFILE_ITERS=1
```

在服务空闲时先调用 `/start_profile`，再发送一个精确 552-token prompt：

```bash
curl -fsS -X POST http://127.0.0.1:28001/start_profile
python3 /home/z00956592/bigm_perf_test.py 1 552 1 1
curl -fsS -X POST http://127.0.0.1:28001/stop_profile
```

该请求在 TP4 下应显示 `M=138`。使用 2048-token prompt 可以测容量边界
`M=512`。不要用 4096-token prefill 对比 CatCCOS，因为当前配置的 CatCCOS
全局容量是 `512 × TP4 = 2048`，更大的 batch 可能回退到其他路径。

## 7. 解析 profiling 文件

每个 worker 会生成独立的 `*_ascend_pt` 目录。容器内执行：

```python
from pathlib import Path

from torch_npu.profiler.profiler import analyse

for tag in ("catccos_load_r1", "native_load_r1"):
    root = Path("/home/z00956592/profiles") / tag
    for profile in root.rglob("*_ascend_pt"):
        print("analyse:", profile)
        analyse(str(profile))
```

主要查看：

```text
ASCEND_PROFILER_OUTPUT/trace_view.json
ASCEND_PROFILER_OUTPUT/operator_details.csv
ASCEND_PROFILER_OUTPUT/kernel_details.csv
ASCEND_PROFILER_OUTPUT/op_statistic.csv
ASCEND_PROFILER_OUTPUT/step_trace_time.csv
```

`trace_view.json` 可以使用 MindStudio Insight 打开。

## 8. trace 中搜索的名称

CatCCOS：

```text
vllm_ascend.moe.catccos.host_moe_body
vllm_ascend.moe.catccos.synchronized_moe_body
vllm_ascend.moe.catccos.adapter_host_scope
vllm_ascend.moe.catccos.pre_sync_host_wait
vllm_ascend.moe.catccos.input_prepare_host
vllm_ascend.moe.catccos.kernel_enqueue
vllm_ascend.moe.catccos.post_sync_host_wait
```

native：

```text
vllm_ascend.moe.native_mc2.host_moe_body
vllm_ascend.moe.native_mc2.synchronized_moe_body
vllm_ascend.moe.native_mc2.pipeline_host_scope
vllm_ascend.moe.native_mc2.dispatch_enqueue
vllm_ascend.moe.native_mc2.mlp_enqueue
vllm_ascend.moe.native_mc2.combine_enqueue
```

完整名称会带真实输入 shape：

```text
vllm_ascend.moe.catccos.kernel_enqueue[M=2,H=2048,topK=8]
vllm_ascend.moe.native_mc2.dispatch_enqueue[M=2,H=2048,topK=8]
```

## 9. A/B 判读方式

同一 workload、同一 `M` 下比较：

1. 自然服务轮比较整个 worker iteration、Device critical-path span 和 NPU 空洞；
   不直接比较两个 `host_moe_body` duration。
2. 对齐边界轮比较 CatCCOS 与 native MC2 的 `synchronized_moe_body`。
3. CatCCOS `pre_sync_host_wait`、`post_sync_host_wait` 的 Host 等待时间。
4. CatCCOS 融合 kernel 与 native dispatch、MLP、combine 所关联的 Device
   Kernel 总时间。
5. NPU 时间线中的空闲区间和算子 launch 间隔。
6. 四个 rank 的最大耗时。集合通信路径由最慢 rank 决定，不能只看平均值。
7. 多个 iteration 的中位数和 P95，避免由单个 iteration 下结论。

若 CatCCOS kernel 本身明显长于 native 整条链，应继续使用真实输入进行 EP4
standalone replay，并在 CatCCOS Device Kernel 内增加 routing、MXFP8、GMM、
通信和 unpermute 的阶段时间戳。普通服务内 profile 只能看到融合 kernel 的总
Device 时间，无法自动拆开这些内部阶段。

profiling 会影响服务速度，因此 Output Token Throughput、TTFT 和 TPOT 的正式
A/B 数值仍以关闭 profiler 的原始 200 题复测为准。服务内 profiling 用来解释
性能差距来自 kernel、同步、launch、通信还是 rank 长尾。
