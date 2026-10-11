# DeepSeek-V4-Flash 三路径实验

代码与测试基线在 `pod1-4-z` 的 `catccos-deepseek-v4` 容器。
模型为 `/data/models/DeepSeek-V4-Flash`，测试目录为
`/home/z009565592/catccos-vllm/work/deepseek-v4-test`。
容器中该路径与 `/home/z00956592/catccos-vllm/work` 指向相同挂载。

## 代码分支

| 路径 | 分支 | 已提交版本 |
|---|---|---|
| v4-catccos-vllm | codex/deepseek-v4-flash-catccos | dd0b308e |
| v4-catccos-kernel | codex/deepseek-v4-flash-catccos | e0fb097 |
| v4-deepmegamoe-vllm | codex/deepseek-v4-flash-deepmegamoe | 832850b0 |
| v4-native-vllm | codex/deepseek-v4-flash-native | 55e6eb75 |
| /home/z00956592/DeepGEMM-Ascend-deepmegamoe | codex/deepseek-v4-flash-deepmegamoe | c07023c |

vLLM-Ascend 三分支的共同基线为 `3005f23d`；DeepGEMM 的基线为
`8491bbb4`。主 DeepGEMM 工作区的用户修改与 `.repro/` 保留。
完整提交号见 `results/full-20261011/implementation-commits.json`。

## 统一配置

NPUs 0–3、DP4/TP1/EP4、eager、BF16、不启用 MTP、关闭 prefix cache、
max-model-len=8192、max-num-batched-tokens=4096、max-num-seqs=32、seed42。
GPU memory utilization=0.80、关闭 DSV4 DSA attention multistream overlap、
HCCL_BUFFSIZE=1536，实际通信空间为3072MB。native 的启动 profiling
shape 要求至少2225MB，因此原1024配置的实际2048MB不足。

DP4/TP1/EP 的指南来源是共同基线中的
`docs/source/tutorials/models/DeepSeek-V4-Flash.md`，Ascend950DT mixed
deployment 示例。其余参数是本次对比设置，不能称为官方指南配置。

三条路径使用同一容器、Torch2.10.0、torch-npu2.10.0.post4、vLLM0.23.0、
Transformers5.5.4、CANN9.1.0 和同一 GCC14 runtime。runtime 位于本目录
`runtime/`。DeepGEMM 扩展使用容器内 GCC14 构建；JIT 使用 CANN Bisheng。
源码、扩展/运行库 SHA、OPP metadata 和启动命令逐次留在 service 目录。

## 启停服务与模型检查

进入容器后可复用脚本，必须先释放上一服务，避免同时使用四张卡：

```bash
root=/home/z009565592/catccos-vllm/work/deepseek-v4-test
out=$root/results/manual-catccos
mkdir -p "$out"
RESULT_DIR="$out" nohup bash "$root/run_server.sh" catccos \
  </dev/null >"$out/launcher.log" 2>&1 &
echo $! > "$out/server.pid"
python3 "$root/wait_for_service.py" "$out"
python3 "$root/model_smoke.py" --datasets "$root/datasets-dsv4-20261011" \
  --out "$out" --stress
python3 "$root/stop_server.py" "$out"
npu-smi info
```

把 backend 改为 `deepmegamoe` 或 `native` 即可测试对应分支。
`stop_server.py` 只处理记录的服务及其 worker；不要用全局 pkill。
`model_smoke.py --stress` 共140请求，覆盖短请求、长输入、并发解码、
混合负载、实际 token 数、有限 log probability 和算术 sanity check。

checkpoint 清单、tensor header/index 对应关系及 tokenizer SHA
保存在 `results/full-20261011/checkpoint-manifest.json`。这些是本地文件
完整性检查，不是上游每个分片 digest 的认证。

## 合成算子测试

先确认0–3无模型进程，设置相同 CANN/runtime 环境。CatCCOS 额外 source
`v4-catccos-kernel/3rdparty/shmem/install/set_env.sh`。核心命令：

```bash
root=/home/z009565592/catccos-vllm/work/deepseek-v4-test
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source "$root/../v4-catccos-kernel/3rdparty/shmem/install/set_env.sh"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_BUFFSIZE=1536
export LD_LIBRARY_PATH="$root/runtime:$LD_LIBRARY_PATH"
torchrun --nproc_per_node=4 --master_port=27050 \
  "$root/catccos_mxfp4_numeric.py" \
  --library "$root/../v4-catccos-kernel/build_torch_a5/lib/libcatccos_torch.so" \
  --out "$root/results/manual-catccos-numeric"

export PYTHONPATH="/home/z009565592/catccos-vllm/work/v4-deepmegamoe-vllm:/home/z00956592/DeepGEMM-Ascend-deepmegamoe:/vllm-workspace/vllm"
torchrun --nproc_per_node=4 --master_port=27050 \
  "$root/deepmegamoe_numeric.py" --out "$root/results/manual-deep-numeric"
torchrun --nproc_per_node=4 --master_port=27050 \
  "$root/deepmegamoe_numeric.py" --case constant --hidden 4096 \
  --intermediate 2048 --experts 256 --capacity 4096 \
  --out "$root/results/manual-deep-modelshape"
```

CatCCOS 保留官方 MXFP4/E8M0，在每次调用中精确展开 FP4→FP8，展开
开销计入成绩；没有常驻逐层 FP8 权重缓存。deepMegaMoE 使用 FP8×FP4，
路由权重在中间量化前应用，与 CatCCOS 在第二个 linear 后应用的顺序不同。
因此结果不保证 bitwise 相同。CANN9.1 兼容范围限于经过测试的 MegaMoE
路径，不代表 DeepGEMM 所有算子都兼容。

已归档9组四卡数值测试，52份证据文件的SHA均已核验；汇总见
`results/numerical-evidence-audit.json`。CatCCOS随机top-k6的最大相对L2为
0.000291，deepMegaMoE小规模随机测试为0.0000612，模型尺寸常量测试
相对L2为0。旧CatCCOS日志中float32 cosine有超出1的值，不作为有效
余弦证据；这些测试以另行核验的有限输出和相对L2为依据。

## AISBench 协议与结果

所有三路径完成140请求验收后，`run_comparison.sh NEW_OUTPUT_DIR`
依次重启 CatCCOS、deepMegaMoE、native，每次先做44请求重启检查，
再执行正式跑分。任意步骤失败会终止实验并清理本次服务。

| 负载 | 输入/输出 | 并发 | 每轮测量/预热 |
|---|---|---|---|
| decode | 128/256 | 1 | 32/4 |
| decode | 128/256 | 8、32 | 200/16 |
| prefill | 2048或4096/1 | 1 | 200/16 |
| mixed | 512、4096按80:20/128 | 8、32 | 200/16 |

每点3轮；temperature0、seed42、ignore_eos、固定输出长度、无重试。
C1 固定取相同32条数据前缀；其余取200条，mixed 为160/40条。
数据 SHA 在 `datasets-dsv4-20261011/dataset-manifest.json`，不重新生成。
每轮核验请求成功、输入分布和输出长度，不接受失败轮的成绩。

本次正式结果目录：`results/aisbench-final-20261011`。
当前管线日志：`results/aisbench-final-20261011-pipeline2.log`；首次启动的日志保留在`results/aisbench-final-20261011-pipeline.log`。
`pipeline.exit-code` 为0且存在 `completed.txt` 才代表全部结束；有日志
或某个 backend 启动成功不能视为整套测试完成。
最终 `summary/medians.csv` 为三轮中位数，`rounds.csv` 保留逐轮数据。
TTFT 为首 token 时间减请求开始时间；TPOT 为首至末 token 时间除以
输出长度减1，输出1时无 TPOT。另报告输出/输入吞吐和错误数。

管线成功退出后，使用 AISBench Python 执行最终原始数据审计：

```bash
/opt/aisbench-venv/bin/python "$root/audit_completion.py" \
  "$root/results/aisbench-final-20261011"
```

审计必须覆盖63轮、21组中位数、固定负载、三个140请求整模检查和
正式服务44请求重启检查，并核对当前提交、运行库、数据及tokenizer。
成功时生成 `completion-audit.json`；仍在运行的实验不会通过此审计。
设备释放、合成数值测试范围及本地归档SHA另行核验。

本次实验另有 `finalize_after_pipeline.py` 等待已识别的管线进程结束，
随后执行上述审计、保存设备快照并生成服务器完整结果压缩包及SHA256。
记录在 `results/aisbench-final-20261011-finalize.*`；此收集步骤成功
仍需本地传输校验和最终审阅后才能宣告整体任务完成。

三路径统一配置的140请求验收均通过。与 native 对比，CatCCOS 有
33/140、deepMegaMoE 有34/140请求完整 token 序列相同，两者各有
139/140首 token 相同。详细数据见 `smoke-token-comparison.json`。
这是稳定性与 greedy 输出一致性检查，不能代替任务精度评测。
