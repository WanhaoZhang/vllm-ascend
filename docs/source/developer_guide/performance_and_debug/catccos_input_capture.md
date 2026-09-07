# Capture CatCCOS inputs for standalone replay

The CatCCOS input capture records the seven tensors passed to
`torch.ops.catccos.ascend950_dispatch_ffn_combine`. It is disabled by default
and only runs in eager mode when both environment variables are set and the
trigger file exists.

Set the variables before starting vLLM. Do not create the trigger yet:

```bash
export VLLM_ASCEND_CATCCOS_DUMP_DIR=/data/catccos_capture/run_001
export VLLM_ASCEND_CATCCOS_DUMP_TRIGGER=/tmp/catccos_capture.trigger
rm -f "${VLLM_ASCEND_CATCCOS_DUMP_TRIGGER}"
rm -rf "${VLLM_ASCEND_CATCCOS_DUMP_DIR}"
```

Start the CatCCOS-backed vLLM service and wait until model loading and warmup
have completed. Create the trigger immediately before sending the request to
capture:

```bash
touch "${VLLM_ASCEND_CATCCOS_DUMP_TRIGGER}"
```

Each EP worker captures its first CatCCOS invocation after the trigger appears.
A completed EP4 capture contains `manifest_rank_0.json` through
`manifest_rank_3.json` and seven binary files per rank. A manifest is written
only after all seven files for that rank have been written successfully.

The binary names match the CatCCOS
`ascend950_dispatch_ffn_combine.py` standalone runner:

| Operator argument | Captured file |
|---|---|
| `x` | `in_routing_matrix_a_<rank>.bin` |
| `expert_idx` | `in_routing_expert_idx_<rank>.bin` |
| `gate_weight` | `in_gather_gate_weight_<rank>.bin` |
| `w1` | `in_gmm_matrix_b_<rank>.bin` |
| `w1_scale` | `in_gmm_matrix_b_scale_<rank>.bin` |
| `w2` | `in_gmm_matrix_b2_<rank>.bin` |
| `w2_scale` | `in_gmm_matrix_b2_scale_<rank>.bin` |

Tensor storage is copied byte-for-byte after the adapter has converted
`expert_idx` to int32, converted `gate_weight` to float32, and made the three
dynamic inputs contiguous. The manifest records shape, dtype, stride, and byte
size. Weight tensors retain their MXFP8 E4M3 and E8M0 bit patterns.

Use a new empty dump directory for each service run. Capturing copies all seven
inputs from NPU to CPU and writes about 150 MiB per Qwen3-30B-A3B EP rank. It is
for correctness and standalone profiling only; do not collect end-to-end
performance numbers from the captured request.
