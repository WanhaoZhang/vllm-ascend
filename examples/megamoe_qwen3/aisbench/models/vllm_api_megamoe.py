# SPDX-License-Identifier: Apache-2.0

from ais_bench.benchmark.models import VLLMCustomAPIChat

_env = __import__("os").environ

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr=_env.get("AISBENCH_MODEL_ABBR", "megamoe-vllm"),
        path=_env["MODEL_PATH"],
        model=_env.get("SERVED_MODEL_NAME", "Qwen3-30B-A3B"),
        stream=_env.get("AISBENCH_STREAM", "0") == "1",
        request_rate=float(_env.get("AISBENCH_REQUEST_RATE", "0")),
        use_timestamp=False,
        retry=2,
        host_ip=_env.get("AISBENCH_HOST", "127.0.0.1"),
        host_port=int(_env.get("AISBENCH_PORT", "18080")),
        max_out_len=int(_env.get("AISBENCH_MAX_OUT_LEN", "1024")),
        batch_size=int(_env.get("AISBENCH_BATCH_SIZE", "1")),
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=0,
            seed=0,
        ),
    )
]
