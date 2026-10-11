#!/usr/bin/env bash
set -eo pipefail

inside=/home/z009565592/catccos-vllm
repo="$inside/work/v4-catccos-vllm"
custom_op_package="$repo/vllm_ascend/_cann_ops_custom/vendors/custom_transformer_transformer"

unset PYTHONPATH TORCH_DEVICE_BACKEND_AUTOLOAD
unset ASCEND_CUSTOM_OPP_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source "$custom_op_package/bin/set_env.bash"
test -s "$custom_op_package/op_impl/ai_core/tbe/kernel/config/ascend950/binary_info_config.json"
compgen -G "$custom_op_package/op_impl/ai_core/tbe/kernel/ascend950/inplace_partial_rotary_mul/InplacePartialRotaryMul_*.o" >/dev/null
export ASCEND_CUSTOM_OPP_PATH="$custom_op_package"
export LD_LIBRARY_PATH="$custom_op_package/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ASCEND_RT_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONPATH="$repo:/vllm-workspace/vllm${PYTHONPATH:+:$PYTHONPATH}"

exec python3 "$inside/work/deepseek-v4-test/test_dsa_rope_package.py"
