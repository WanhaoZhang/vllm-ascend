#!/usr/bin/env bash
set -Eeo pipefail

backend=${1:?usage: run_server.sh catccos|deepmegamoe|native}
root=/home/z00956592/catccos-vllm
inside=/home/z009565592/catccos-vllm
test_root=$root/work/deepseek-v4-test
model=/data/models/DeepSeek-V4-Flash
port=${PORT:-28041}
store_port=${STORE_PORT:-27041}
result_root=${RESULT_DIR:-$test_root/results/$backend}
mkdir -p "$result_root"
exec >>"$result_root/server.log" 2>&1
printf '[%s] starting backend=%s port=%s\n' "$(date -Iseconds)" "$backend" "$port"

npu_snapshot=$(npu-smi info)
printf '%s\n' "$npu_snapshot" > "$result_root/npu-before.txt"
if [[ $(printf '%s\n' "$npu_snapshot" | grep -c 'No running processes found in NPU [0-3]') != 4 ]]; then
  echo 'NPU 0-3 still have running processes; release the previous service before starting.' >&2
  exit 1
fi

unset PYTHONPATH TORCH_DEVICE_BACKEND_AUTOLOAD
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

custom_op_package="$inside/work/v4-catccos-vllm/vllm_ascend/_cann_ops_custom/vendors/custom_transformer_transformer"
custom_vendor_alias="$inside/work/v4-catccos-vllm/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
test -f "$custom_op_package/bin/set_env.bash" || { echo "Hc custom-op environment not found: $custom_op_package" >&2; exit 1; }
if [[ ! -e "$custom_vendor_alias" ]]; then
  ln -s "$custom_op_package" "$custom_vendor_alias"
fi
test -e "$custom_vendor_alias/op_api/lib/libcust_opapi.so" || { echo "Hc custom-op alias not found: $custom_vendor_alias" >&2; exit 1; }
source "$custom_op_package/bin/set_env.bash"

runtime_custom_op_package=${CANN_CUSTOM_OP_PACKAGE:-$custom_op_package}
test -s "$runtime_custom_op_package/op_impl/ai_core/tbe/kernel/config/ascend950/binary_info_config.json" || {
  echo "Ascend950 custom-op metadata missing: $runtime_custom_op_package" >&2
  exit 1
}
grep -q 'InplacePartialRotaryMul' "$runtime_custom_op_package/op_impl/ai_core/tbe/kernel/config/ascend950/binary_info_config.json" || {
  echo "InplacePartialRotaryMul registration missing: $runtime_custom_op_package" >&2
  exit 1
}
export ASCEND_CUSTOM_OPP_PATH="$runtime_custom_op_package${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$runtime_custom_op_package/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
common_runtime="$test_root/runtime"
if [[ -f "$common_runtime/libstdc++.so.6" ]]; then
  export LD_LIBRARY_PATH="$common_runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
custom_opapi_library="$runtime_custom_op_package/op_api/lib/libcust_opapi.so"
test -f "$custom_opapi_library" || { echo "Hc custom-op API library not found: $custom_opapi_library" >&2; exit 1; }
unset LD_PRELOAD

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export HCCL_BUFFSIZE=1536
export HCCL_OP_EXPANSION_MODE=AIV
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="/home/z009565592/catccos-vllm/work/deepseek-v4-diagnosis-20261011/cached-vllm:/home/z009565592/catccos-vllm/work/deepseek-v4-diagnosis-20261011:/vllm-workspace/vllm${PYTHONPATH:+:$PYTHONPATH}"

# Keep the attention execution policy common across all three MoE backends.
# The base branch's DSV4 attention multistream path fails on 4096-token
# prefill (QuantBatchMatmulV3/RmsNorm device address faults, R7/R9 logs).
additional_config='{"multistream_dsv4_dsa_overlap":false}'
case "$backend" in
  catccos)
    source "$inside/work/v4-catccos-kernel/3rdparty/shmem/install/set_env.sh"
    catccos_library="$inside/work/v4-catccos-kernel/build_torch_a5/lib/libcatccos_torch.so"
    additional_config=$(python3 -c 'import json,sys; print(json.dumps({"enable_fused_mc2":1,"fused_mc2_backend":"catccos","catccos_library_path":sys.argv[1],"catccos_store_url":sys.argv[2],"catccos_local_mem_size":1073741824,"catccos_min_tokens":1,"catccos_max_tokens_per_rank":4096,"catccos_sync_after_launch":False,"enable_prefill_mc2":True}))' "$catccos_library" "tcp://127.0.0.1:$store_port")
    ;;
  deepmegamoe)
    export DEEPMEGAMOE_SOURCE=${DEEPMEGAMOE_SOURCE:-$root/../DeepGEMM-Ascend-deepmegamoe}
    deepmegamoe_source=$DEEPMEGAMOE_SOURCE
    export PYTHONPATH="$inside/work/v4-deepmegamoe-vllm:$deepmegamoe_source${PYTHONPATH:+:$PYTHONPATH}"
    deepmegamoe_runtime=${DEEPMEGAMOE_RUNTIME:-$test_root/runtime}
    export LD_LIBRARY_PATH="$deepmegamoe_runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    additional_config=$(python3 -c 'import json; print(json.dumps({"enable_fused_mc2":1,"fused_mc2_backend":"deepmegamoe","deepmegamoe_max_tokens_per_rank":4096}))')
    ;;
  native)
    additional_config='{"enable_fused_mc2":0,"fused_mc2_backend":"auto","enable_prefill_mc2":true}'
    ;;
  *)
    echo "unknown backend: $backend" >&2
    exit 2
    ;;
esac

additional_config=$(python3 -c 'import json,sys; c=json.loads(sys.argv[1]); c["multistream_dsv4_dsa_overlap"]=False; print(json.dumps(c))' "$additional_config")
if [[ "$backend" == catccos && ${CATCCOS_DIAGNOSTIC_SYNC:-0} == 1 ]]; then
  additional_config=$(python3 -c 'import json,sys; c=json.loads(sys.argv[1]); c["catccos_sync_after_launch"]=True; print(json.dumps(c))' "$additional_config")
fi

args=(
  "$model"
  --served-model-name deepseek-v4-flash
  --tokenizer-mode deepseek_v4
  --tensor-parallel-size 1
  --data-parallel-size 4
  --enable-expert-parallel
  --distributed-executor-backend mp
  --dtype bfloat16
  --max-model-len 8192
  --max-num-batched-tokens 4096
  --max-num-seqs 32
  # The custom EP scratch and long-prefill workspaces need headroom.
  # Apply the same memory budget to every backend in the comparison.
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}"
  --enforce-eager
  --no-enable-prefix-caching
  --seed 42
  --host 0.0.0.0
  --port "$port"
)
if [[ -n "$additional_config" ]]; then
  args+=(--additional-config "$additional_config")
fi

args+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"ignore_frontend\":true,\"max_iterations\":0}")
python3 "$test_root/freeze_environment.py" --backend "$backend" --out "$result_root"
printf '%q ' vllm serve "${args[@]}" > "$result_root/command.sh"
printf '\n' >> "$result_root/command.sh"
exec vllm serve "${args[@]}"
