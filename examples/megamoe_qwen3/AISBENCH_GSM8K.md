# AISBench GSM8K native/CatCCOS evaluation

This procedure measures three different questions:

1. Generalization: does the CatCCOS path run the same GSM8K workload on
   single NPU, TP2, and TP4 without failed requests or invalid outputs?
2. Accuracy: does the full GSM8K score remain within the agreed tolerance of
   the native vLLM-Ascend baseline?
3. Performance: after accuracy passes, how do TTFT, TPOT, E2EL, request
   throughput, and token throughput change?

Do not accept a faster result if accuracy has regressed. Keep the native and
CatCCOS predictions as well as the aggregate summaries.

## 1. Install AISBench in an isolated environment

AISBench supports Python 3.10 through 3.12. The commands below use `uv` and do
not modify the system Python environment.

```bash
mkdir -p /data/tools
cd /data/tools

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/AISBench/benchmark.git
cd benchmark

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python \
  -r requirements/api.txt \
  -r requirements/extra.txt

.venv/bin/ais_bench -h
```

If `pypi.org` is inaccessible, repeat the two install commands with a reachable
mirror, for example:

```bash
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
export UV_HTTP_TIMEOUT=120
```

Record the AISBench revision:

```bash
git rev-parse HEAD
.venv/bin/python -c \
  'import importlib.metadata; print(importlib.metadata.version("ais-bench-benchmark"))'
```

## 2. Download GSM8K

Use the dataset URL from the vLLM-Ascend AISBench guide:

```bash
cd /data/tools/benchmark/ais_bench/datasets
curl -fL --retry 3 \
  -o gsm8k.zip \
  https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip
unzip gsm8k.zip
rm gsm8k.zip

test -f gsm8k/test.jsonl
test -f gsm8k/train.jsonl
```

## 3. Smoke test one endpoint

Use three prompts only to validate the service URL, model name, dataset,
scorer, and output permissions. `SERVED_MODEL_NAME` must exactly match the ID
returned by `/v1/models`.

```bash
curl -fsS http://127.0.0.1:18080/v1/models

cd /data/src/vllm-ascend
AISBENCH_ROOT=/data/tools/benchmark \
MODEL_PATH=/data/models/Qwen3-30B-A3B \
SERVED_MODEL_NAME=Qwen3-30B-A3B \
AISBENCH_PORT=18080 \
MODE=accuracy \
RUN_LABEL=native-tp4-smoke \
NUM_PROMPTS=3 \
NUM_WARMUPS=0 \
bash examples/megamoe_qwen3/run_aisbench_gsm8k.sh
```

Do not continue if any request failed, the summary contains `-`, or the model
ID was not found.

## 4. Full accuracy A/B test

Run the native baseline first. Leave `NUM_PROMPTS` unset to evaluate the full
GSM8K test split.

```bash
cd /data/src/vllm-ascend

AISBENCH_ROOT=/data/tools/benchmark \
MODEL_PATH=/data/models/Qwen3-30B-A3B \
SERVED_MODEL_NAME=Qwen3-30B-A3B \
AISBENCH_PORT=18080 \
MODE=accuracy \
RUN_LABEL=native-tp4-full \
NUM_WARMUPS=1 \
AISBENCH_MAX_OUT_LEN=1024 \
bash examples/megamoe_qwen3/run_aisbench_gsm8k.sh
```

Stop the baseline, launch CatCCOS on the same physical NPUs with the same
vLLM serving arguments, and repeat:

```bash
AISBENCH_ROOT=/data/tools/benchmark \
MODEL_PATH=/data/models/Qwen3-30B-A3B \
SERVED_MODEL_NAME=Qwen3-30B-A3B \
AISBENCH_PORT=18081 \
MODE=accuracy \
RUN_LABEL=catccos-tp4-full \
NUM_WARMUPS=1 \
AISBENCH_MAX_OUT_LEN=1024 \
bash examples/megamoe_qwen3/run_aisbench_gsm8k.sh
```

The output root defaults to:

```text
/data/tools/benchmark/outputs/megamoe/<RUN_LABEL>/accuracy/<timestamp>/
```

Archive at least `configs/`, `logs/`, `predictions/`, `results/`, and
`summary/`. Compare per-sample prediction files in addition to the final
score. A useful first gate is zero failed requests and no large score drop;
set the exact acceptance tolerance before the production evaluation.

## 5. Performance A/B test

Only run performance after the accuracy gate passes. The wrapper enables
streaming in `MODE=perf`, allowing AISBench to report TTFT, TPOT, and ITL.

Native:

```bash
AISBENCH_ROOT=/data/tools/benchmark \
MODEL_PATH=/data/models/Qwen3-30B-A3B \
SERVED_MODEL_NAME=Qwen3-30B-A3B \
AISBENCH_PORT=18080 \
MODE=perf \
RUN_LABEL=native-tp4-perf \
NUM_WARMUPS=3 \
AISBENCH_MAX_OUT_LEN=512 \
AISBENCH_BATCH_SIZE=1 \
bash examples/megamoe_qwen3/run_aisbench_gsm8k.sh
```

CatCCOS:

```bash
AISBENCH_ROOT=/data/tools/benchmark \
MODEL_PATH=/data/models/Qwen3-30B-A3B \
SERVED_MODEL_NAME=Qwen3-30B-A3B \
AISBENCH_PORT=18081 \
MODE=perf \
RUN_LABEL=catccos-tp4-perf \
NUM_WARMUPS=3 \
AISBENCH_MAX_OUT_LEN=512 \
AISBENCH_BATCH_SIZE=1 \
bash examples/megamoe_qwen3/run_aisbench_gsm8k.sh
```

Repeat each performance job at least three times after warmup. Use the median
of runs, not the best run. Keep these variables identical:

- physical NPU IDs and TP/EP size;
- image digest and both source revisions;
- vLLM serving flags, model length, memory utilization, and model weights;
- AISBench dataset revision, prompt count, output length, batch size, request
  rate, streaming mode, and warmup count;
- CatCCOS correctness switches, especially weight quantization and device
  synchronization.

For load testing, increase `AISBENCH_BATCH_SIZE` or set a nonzero
`AISBENCH_REQUEST_RATE`, but treat it as a separate experiment. Do not mix its
results with the concurrency-one latency run.

## 6. TP1, TP2, and TP4 matrix

Complete the following matrix on the 950DT server. Test the standalone CatCCOS
operator for each rank size before the end-to-end vLLM job.

| NPU mode | Native accuracy | CatCCOS accuracy | Native perf | CatCCOS perf |
|---|---:|---:|---:|---:|
| TP1/EP1 | required | required | required | required |
| TP2/EP2 | required | required | required | required |
| TP4/EP4 | required | required | required | required |

For a generalization claim, also vary prompt batch shape and concurrency after
the full GSM8K accuracy comparison. The current CatCCOS threshold means short
batches may fall back to the native MoE path; retain the CatCCOS initialization
logs and use sufficiently large prompts/concurrency when measuring operator
coverage.

## 7. Single-NPU smoke result on `a5new`

Date: 2026-08-25. This was a pipeline validation, not a statistically useful
benchmark. Both services used the same image digest
`sha256:cc57064f119054904dc81360cd1105d211fa9b91bf726486926dd025c26f17b7`.
AISBench was version 3.1.0 at revision
`29c363e38e9d1560e6f19eff582f6117943b6a77`.

The CatCCOS service logs confirmed EP rank `0/1`, NPU weight quantization, and
MXFP8 weight shapes `w1=(128,1536,2048)` and `w2=(128,2048,768)`.

| Metric, 3 prompts | Native | CatCCOS |
|---|---:|---:|
| GSM8K accuracy | 66.67% | 66.67% |
| Failed accuracy requests | 0 | 0 |
| E2EL average, 128 output tokens | 4112.2 ms | 4239.3 ms |
| TTFT average | 81.2 ms | 96.0 ms |
| TPOT average | 31.7 ms | 32.6 ms |
| Output throughput | 31.1248 token/s | 30.1918 token/s |
| Failed performance requests | 0 | 0 |

The performance run used streaming, concurrency one, no warmup, an average of
57.33 input tokens, and exactly 128 output tokens. Three prompts are too few
to infer a performance advantage or accuracy parity. The result only proves
that the repository runner and both single-NPU endpoints work end to end.

## 8. Failure triage order

1. Confirm `/health`, `/v1/models`, and the exact served model ID.
2. Confirm zero AISBench failed requests before reading aggregate scores.
3. Confirm CatCCOS enable, initialization, weight conversion, and per-rank
   messages in container logs.
4. Repeat CatCCOS with `CATCCOS_WEIGHT_QUANT_BACKEND=cpu`.
5. Reduce TP4 to TP2, then TP1, using the same first failing GSM8K samples.
6. Run the standalone CatCCOS operator at the failing rank size and shape.
7. Preserve predictions, rank logs, topology files' checksums, source hashes,
   image digest, and all environment variables for the bug report.
