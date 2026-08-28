#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from collections.abc import Callable
from typing import Any

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

env_variables: dict[str, Callable[[], Any]] = {
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled.
    # This configuration option should only be set to False when running UT
    # scenarios in an environment without an NPU. Do not set it to False in
    # other scenarios.
    "COMPILE_CUSTOM_KERNELS": lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to enable MatmulAllReduce fusion kernel when tensor parallel is enabled.
    # this feature is supported in A2, and eager mode will get better performance.
    "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", "0"))),
    # Whether to enable FlashComm optimization when tensor parallel is enabled.
    # This feature will get better performance when concurrency is large.
    # DEPRECATED: use additional_config.enable_flashcomm1 instead.
    "VLLM_ASCEND_ENABLE_FLASHCOMM1": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0"))),
    # Whether to enable FLASHCOMM2. Setting it to 0 disables the feature, while setting it to 1 or above enables it.
    # The specific value set will be used as the O-matrix TP group size for flashcomm2.
    # For a detailed introduction to the parameters and the differences and applicable scenarios
    # between this feature and FLASHCOMM1, please refer to the feature guide in the documentation.
    "VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": lambda: int(os.getenv("VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE", 0)),
    # Whether to enable msMonitor tool to monitor the performance of vllm-ascend.
    "MSMONITOR_USE_DAEMON": lambda: bool(int(os.getenv("MSMONITOR_USE_DAEMON", "0"))),
    # Whether to enable MLAPO optimization for DeepSeek W8A8 series models.
    # This option is enabled by default. MLAPO can improve performance, but
    # it will consume more NPU memory. If reducing NPU memory usage is a higher priority
    # for your DeepSeek W8A8 scene, then disable it.
    "VLLM_ASCEND_ENABLE_MLAPO": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MLAPO", "1"))),
    # Whether to enable weight cast format to FRACTAL_NZ.
    # 0: close nz;
    # 1: only quant case enable nz;
    # 2: enable nz as long as possible.
    "VLLM_ASCEND_ENABLE_NZ": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_NZ", 1)),
    # Whether to anbale dynamic EPLB
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Whether to enable fused MC2 (`dispatch_gmm_combine_decode` / `dispatch_ffn_combine`).
    # 0, or not set: default ALLTOALL and MC2 will be used.
    # 1: ALLTOALL and MC2 might be replaced by `dispatch_ffn_combine` operator.
    # `dispatch_ffn_combine` can be used only for moe layer with W8A8, EP<=32, non-mtp, non-dynamic-eplb.
    # 2: MC2 might be replaced by `dispatch_gmm_combine_decode` operator.
    # `dispatch_gmm_combine_decode` can be used only for **decode node** moe layer
    # with W8A8. And MTP layer must be W8A8.
    "VLLM_ASCEND_ENABLE_FUSED_MC2": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_FUSED_MC2", "0")),
    # Experimental CatCCOS Ascend 950 fused dispatch + FFN + combine backend.
    # This prototype is available only with vLLM/vLLM-Ascend 0.23.0 and eager
    # execution. Set MINM above one only to opt small batches into native MoE.
    "VLLM_ASCEND_CATCCOS": lambda: bool(int(os.getenv("VLLM_ASCEND_CATCCOS", "0"))),
    "VLLM_ASCEND_CATCCOS_LIBRARY_PATH": lambda: os.getenv(
        "VLLM_ASCEND_CATCCOS_LIBRARY_PATH",
        "/workspace/catccos/build_torch_a5/lib/libcatccos_torch.so",
    ),
    "VLLM_ASCEND_CATCCOS_UTILS_PATH": lambda: os.getenv(
        "VLLM_ASCEND_CATCCOS_UTILS_PATH", "/workspace/catccos/examples/utils"
    ),
    "VLLM_ASCEND_CATCCOS_IPPORT": lambda: os.getenv("VLLM_ASCEND_CATCCOS_IPPORT", "tcp://127.0.0.1:27020"),
    # The A5 binding allocates a fixed 1004 MiB symmetric buffer.
    "VLLM_ASCEND_CATCCOS_MEM": lambda: int(os.getenv("VLLM_ASCEND_CATCCOS_MEM", str(1024 * 1024 * 1024))),
    "VLLM_ASCEND_CATCCOS_MINM": lambda: int(os.getenv("VLLM_ASCEND_CATCCOS_MINM", "1")),
    # NPU quantization avoids copying every expert weight through the host.
    # Set to "cpu" only when exact parity with CatCCOS' data generator is needed.
    "VLLM_ASCEND_CATCCOS_WEIGHT_QUANT_BACKEND": lambda: os.getenv(
        "VLLM_ASCEND_CATCCOS_WEIGHT_QUANT_BACKEND", "npu"
    ).lower(),
    # Optional same-input native/CatCCOS correctness probe. Empty directory
    # disables it. Output contains no credentials but can contain model inputs.
    "VLLM_ASCEND_CATCCOS_DEBUG_DIR": lambda: os.getenv("VLLM_ASCEND_CATCCOS_DEBUG_DIR", ""),
    # Required when DEBUG_DIR is set. Comma-separated token-row counts, for
    # example "177" for one fixed prompt or "1,177" for decode plus prefill.
    "VLLM_ASCEND_CATCCOS_DEBUG_TOKEN_COUNTS": lambda: os.getenv("VLLM_ASCEND_CATCCOS_DEBUG_TOKEN_COUNTS", ""),
    # Optional comma-separated zero-based MoE instance IDs. Empty selects all
    # MoE layers. Use "0" to limit a probe to the first routed MoE layer.
    "VLLM_ASCEND_CATCCOS_DEBUG_MOE_INSTANCE_IDS": lambda: os.getenv(
        "VLLM_ASCEND_CATCCOS_DEBUG_MOE_INSTANCE_IDS", ""
    ),
    # Valid values: native-catccos, catccos-native, native-native,
    # catccos-catccos. All ranks must use the same order.
    "VLLM_ASCEND_CATCCOS_DEBUG_ORDER": lambda: os.getenv("VLLM_ASCEND_CATCCOS_DEBUG_ORDER", "native-catccos").lower(),
    # Positive limit for each selected token count in each MoE layer.
    "VLLM_ASCEND_CATCCOS_DEBUG_MAX_CALLS_PER_LAYER": lambda: int(
        os.getenv("VLLM_ASCEND_CATCCOS_DEBUG_MAX_CALLS_PER_LAYER", "1")
    ),
    # [0, 1]. The first output pair below this cosine is dumped per rank.
    "VLLM_ASCEND_CATCCOS_DEBUG_COSINE_THRESHOLD": lambda: float(
        os.getenv("VLLM_ASCEND_CATCCOS_DEBUG_COSINE_THRESHOLD", "0.99")
    ),
    # The first output pair above this relative L2 error is dumped per rank.
    "VLLM_ASCEND_CATCCOS_DEBUG_RELATIVE_L2_THRESHOLD": lambda: float(
        os.getenv("VLLM_ASCEND_CATCCOS_DEBUG_RELATIVE_L2_THRESHOLD", "0.1")
    ),
    # Save inputs, routes, and frozen outputs for the first mismatch per rank.
    "VLLM_ASCEND_CATCCOS_DEBUG_DUMP_TENSORS": lambda: bool(
        int(os.getenv("VLLM_ASCEND_CATCCOS_DEBUG_DUMP_TENSORS", "1"))
    ),
    # Save the first selected call even when it is below mismatch thresholds.
    # This is intended for targeted, one-layer contract probes only.
    "VLLM_ASCEND_CATCCOS_DEBUG_DUMP_SELECTED": lambda: bool(
        int(os.getenv("VLLM_ASCEND_CATCCOS_DEBUG_DUMP_SELECTED", "0"))
    ),
    # Also save the first mismatch's MXFP8 weights and scales. This can use
    # several GiB across ranks, so it is disabled by default.
    "VLLM_ASCEND_CATCCOS_DEBUG_DUMP_WEIGHTS": lambda: bool(
        int(os.getenv("VLLM_ASCEND_CATCCOS_DEBUG_DUMP_WEIGHTS", "0"))
    ),
    # DEPRECATED: VLLM_ASCEND_BALANCE_SCHEDULING env var will be removed in a future release.
    # Use --additional-config '{"enable_balance_scheduling": true}' instead.
    "VLLM_ASCEND_BALANCE_SCHEDULING": lambda: bool(int(os.getenv("VLLM_ASCEND_BALANCE_SCHEDULING", "0"))),
    # use fused op transpose_kv_cache_by_block, default is True
    "VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK": lambda: bool(
        int(os.getenv("VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK", "1"))
    ),
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    # "1": force enable, "0": force disable, None: auto-detect from CANN headers.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
