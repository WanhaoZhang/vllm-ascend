# CatCCOS same-input MoE probe

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
|---|---|
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
|---|---:|---|
| `CATCCOS_DEBUG_HOST_DIR` | empty | Absolute host output path; empty disables the probe. |
| `CATCCOS_DEBUG_TOKEN_COUNTS` | required | Comma-separated M values, for example `177` or `1,177`. |
| `CATCCOS_DEBUG_ORDER` | `native-catccos` | One of the four comparison orders above. |
| `CATCCOS_DEBUG_MAX_CALLS_PER_LAYER` | `1` | Calls recorded per selected M in each layer. |
| `CATCCOS_DEBUG_COSINE_THRESHOLD` | `0.99` | First result below this value is dumped per rank. |
| `CATCCOS_DEBUG_RELATIVE_L2_THRESHOLD` | `0.1` | First result above this value is dumped per rank. |
| `CATCCOS_DEBUG_DUMP_TENSORS` | `1` | Save the first mismatch's inputs, routes, and outputs. |
| `CATCCOS_DEBUG_DUMP_WEIGHTS` | `0` | Also save MXFP8 weights and scales; high disk and host-memory cost. |
