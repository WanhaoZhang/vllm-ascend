# CatCCOS and native MC2 service profiling

This branch adds shape-qualified profiler ranges around the CatCCOS and native
MC2 MoE paths. It uses vLLM's built-in online profiler; the ranges do not dump
tensor values and are disabled by default.

## Trace ranges

Set `VLLM_ASCEND_MOE_PROFILE_RANGES=1` before starting every worker. The
following ranges will then appear in the Ascend PyTorch Profiler trace:

```text
vllm_ascend.moe.catccos.total[M=2,H=2048,topK=8]
├── vllm_ascend.moe.catccos.pre_sync[...]
├── vllm_ascend.moe.catccos.input_prepare[...]
├── vllm_ascend.moe.catccos.kernel[...]
└── vllm_ascend.moe.catccos.post_sync[...]  # sync_after_launch=true only

vllm_ascend.moe.native_mc2.total[M=2,H=2048,topK=8]
├── vllm_ascend.moe.native_mc2.dispatch[...]
├── vllm_ascend.moe.native_mc2.mlp[...]
└── vllm_ascend.moe.native_mc2.combine[...]
```

`M` is the rank-local token count after MC2 padding and TP splitting. For TP4,
a steady decode batch of eight requests is recorded as `M=2`. A global prefill
batch of 552 tokens is recorded as `M=138`.

## Start the service

Keep the model, cards, TP/EP configuration, scheduler limits, and requests the
same between runs. Use a different absolute output directory for every run.

```bash
export MSMONITOR_USE_DAEMON=0
export VLLM_ASCEND_MOE_PROFILE_RANGES=1

PROFILE_CONFIG='{
  "profiler":"torch",
  "torch_profiler_dir":"/home/z00956592/profiles/cat_decode_r1",
  "torch_profiler_with_stack":false,
  "torch_profiler_with_memory":false,
  "ignore_frontend":true,
  "max_iterations":10
}'

vllm serve /home/weights/Qwen3-30B-A3B-Instruct-2507 \
  --served-model-name qwen3-catccos \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --distributed-executor-backend mp \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.80 \
  --no-enable-prefix-caching \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 28001 \
  --profiler-config "$PROFILE_CONFIG" \
  --additional-config "$ADDCONF"
```

The CatCCOS run uses the existing CatCCOS `ADDCONF`. For the native baseline,
change only `torch_profiler_dir` and omit `--additional-config`. The unconfigured
A5 baseline uses the native MC2 dispatch, MLP, and combine path rather than the
CANN fused `dispatch_ffn_combine` operator.

Warm up the service before starting a profile. Do not enable input dumping in
the same run because device-to-host copies will contaminate the timeline.

## Capture steady decode

Start exactly eight long-running requests with `ignore_eos=true`, or use the
existing fixed-prompt load generator. Wait until prefill has completed and all
eight requests are decoding, then collect ten model iterations:

```bash
curl -fsS -X POST http://127.0.0.1:28001/start_profile
sleep 3
curl -fsS -X POST http://127.0.0.1:28001/stop_profile
```

`max_iterations=10` limits worker collection even if the stop request is late.
Repeat the same request sequence after restarting the native service with a
new trace directory.

## Capture one prefill

Restart the service with `max_iterations=1` and a new trace directory. Start
profiling while the service is idle, then send one request containing exactly
552 prompt tokens and one output token:

```bash
curl -fsS -X POST http://127.0.0.1:28001/start_profile
python3 /home/z00956592/bigm_perf_test.py 1 552 1 1
curl -fsS -X POST http://127.0.0.1:28001/stop_profile
```

Use 2048 prompt tokens to measure the configured CatCCOS capacity boundary
(`M=512` per rank). Do not use a 4096-token prefill for this comparison because
the current CatCCOS capacity is 2048 global tokens and larger batches can take
a fallback path.

## Analyze the traces

Each worker writes a separate `*_ascend_pt` directory. Analyze every rank:

```python
from pathlib import Path

from torch_npu.profiler.profiler import analyse

root = Path("/home/z00956592/profiles/cat_decode_r1")
for profile in root.rglob("*_ascend_pt"):
    analyse(str(profile))
```

Open `ASCEND_PROFILER_OUTPUT/trace_view.json` in MindStudio Insight. Use
`operator_details.csv`, `kernel_details.csv`, `op_statistic.csv`, and
`step_trace_time.csv` for aggregate comparisons.

Compare CatCCOS `total` with native MC2 `total` at the same `M`, and then
inspect the child ranges. Use the slowest rank for collective-path comparisons.
The CatCCOS device implementation is one fused kernel, so service profiling can
measure its total device time but cannot split its internal routing, MXFP8,
GMM, communication, and unpermute phases. Use real-input standalone replay and
device-side timestamps if the fused kernel itself is the remaining bottleneck.
