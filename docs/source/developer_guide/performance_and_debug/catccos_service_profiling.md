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
vllm_ascend.moe.catccos.total[M=2,H=2048,topK=8]
├── vllm_ascend.moe.catccos.pre_sync[...]
├── vllm_ascend.moe.catccos.input_prepare[...]
├── vllm_ascend.moe.catccos.kernel[...]
└── vllm_ascend.moe.catccos.post_sync[...]  # sync_after_launch=true 时存在

vllm_ascend.moe.native_mc2.total[M=2,H=2048,topK=8]
├── vllm_ascend.moe.native_mc2.dispatch[...]
├── vllm_ascend.moe.native_mc2.mlp[...]
└── vllm_ascend.moe.native_mc2.combine[...]
```

环境变量不开时，这些 profiler ranges 不会建立：

```bash
export VLLM_ASCEND_MOE_PROFILE_RANGES=1
```

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
vllm_ascend.moe.catccos.total
vllm_ascend.moe.catccos.pre_sync
vllm_ascend.moe.catccos.input_prepare
vllm_ascend.moe.catccos.kernel
vllm_ascend.moe.catccos.post_sync
```

native：

```text
vllm_ascend.moe.native_mc2.total
vllm_ascend.moe.native_mc2.dispatch
vllm_ascend.moe.native_mc2.mlp
vllm_ascend.moe.native_mc2.combine
```

完整名称会带真实输入 shape：

```text
vllm_ascend.moe.catccos.kernel[M=2,H=2048,topK=8]
vllm_ascend.moe.native_mc2.dispatch[M=2,H=2048,topK=8]
```

## 9. A/B 判读方式

同一 workload、同一 `M` 下比较：

1. CatCCOS `total` 与 native MC2 `total`。
2. CatCCOS `pre_sync`、`post_sync` 的 Host 等待时间。
3. CatCCOS 融合 kernel 与 native dispatch、MLP、combine 所关联的 Device
   Kernel 总时间。
4. NPU 时间线中的空闲区间和算子 launch 间隔。
5. 四个 rank 的最大耗时。集合通信路径由最慢 rank 决定，不能只看平均值。
6. 多个 iteration 的中位数和 P95，避免由单个 iteration 下结论。

若 CatCCOS kernel 本身明显长于 native 整条链，应继续使用真实输入进行 EP4
standalone replay，并在 CatCCOS Device Kernel 内增加 routing、MXFP8、GMM、
通信和 unpermute 的阶段时间戳。普通服务内 profile 只能看到融合 kernel 的总
Device 时间，无法自动拆开这些内部阶段。

profiling 会影响服务速度，因此 Output Token Throughput、TTFT 和 TPOT 的正式
A/B 数值仍以关闭 profiler 的原始 200 题复测为准。服务内 profiling 用来解释
性能差距来自 kernel、同步、launch、通信还是 rank 长尾。
