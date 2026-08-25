# Qwen3-30B-A3B Docker deployment

These scripts launch and benchmark `Qwen/Qwen3-30B-A3B` with the official
vLLM Ascend A5 image. They support single-NPU regression checks and TP2/TP4
deployment on Ascend 950 servers.

## Prerequisites

- Docker and an Ascend 950 host driver are installed.
- The model is downloaded to a local directory containing `config.json`,
  `model.safetensors.index.json`, and all Safetensors shards.
- The image is available locally, or the machine can access Quay:

  ```bash
  docker pull quay.io/ascend/vllm-ascend:v0.23.0-a5
  ```

- For multi-NPU deployment, the host has valid D2D topology configuration at
  `/lib/route.conf`, `/etc/hccl_rootinfo.json`, and `/etc/hixlep`.

The launcher fails before loading weights when a required multi-NPU topology
path is missing. Do not copy topology files from another server: generate them
for the target host according to the Ascend 950 HiXLEP instructions.

## Launch

Single NPU:

```bash
MODEL_PATH=/data/models/Qwen3-30B-A3B \
NPU_DEVICES=0 \
bash examples/megamoe_qwen3/run_docker.sh
```

Two NPUs with TP2 and expert parallelism:

```bash
MODEL_PATH=/data/models/Qwen3-30B-A3B \
NPU_DEVICES=0,1 \
GPU_MEMORY_UTILIZATION=0.80 \
bash examples/megamoe_qwen3/run_docker.sh
```

Four NPUs with TP4 and expert parallelism:

```bash
MODEL_PATH=/data/models/Qwen3-30B-A3B \
NPU_DEVICES=0,1,2,3 \
GPU_MEMORY_UTILIZATION=0.80 \
bash examples/megamoe_qwen3/run_docker.sh
```

The default service endpoint is `http://127.0.0.1:18080`. The launcher maps
only the selected `/dev/davinci*` nodes and intentionally does not set
`ASCEND_RT_VISIBLE_DEVICES`.

To replace a previous container or change the port:

```bash
RECREATE=1 PORT=18081 \
MODEL_PATH=/data/models/Qwen3-30B-A3B \
NPU_DEVICES=4,5 \
bash examples/megamoe_qwen3/run_docker.sh
```

## Verify

```bash
curl http://127.0.0.1:18080/health
curl http://127.0.0.1:18080/v1/models
```

Non-thinking chat request:

```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-30B-A3B",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0,
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

## Benchmark

Run a deterministic random-token serving benchmark from the server container:

```bash
MAX_CONCURRENCY=4 \
NUM_PROMPTS=8 \
INPUT_LEN=128 \
OUTPUT_LEN=64 \
bash examples/megamoe_qwen3/benchmark.sh
```

Change `MAX_CONCURRENCY`, `NUM_PROMPTS`, `INPUT_LEN`, and `OUTPUT_LEN` together
when comparing TP2 and TP4. Ensure no unrelated jobs are using the selected
NPUs during a comparison.

## Known failure mode

If startup reports `RootInfoDetect failed` or HCCL error code 4, the target
host's multi-NPU HCCL/HiXLEP topology is missing or invalid. Single-NPU
inference may still work, but TP and expert-parallel collectives will not.
