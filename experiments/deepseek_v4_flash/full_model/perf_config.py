"""AISBench template; render workload constants before mmengine lazy imports."""

from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer

from ais_bench.benchmark.datasets.custom import CustomDataset
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer

MODEL_PATH = "/data/models/DeepSeek-V4-Flash"
DATASET_PATH = __DATASET__
BATCH = __BATCH__
OUTPUT_LENGTH = __OSL__
PORT = 28041

models[0].update(
    abbr="deepseek-v4-flash",
    path=MODEL_PATH,
    model="deepseek-v4-flash",
    host_ip="127.0.0.1",
    host_port=PORT,
    stream=True,
    batch_size=BATCH,
    request_rate=0,
    max_out_len=OUTPUT_LENGTH,
    retry=1,  # AISBench interprets this as total attempts; disable retries.
    generation_kwargs=dict(temperature=0, seed=42, ignore_eos=True),
)

datasets = [
    dict(
        abbr="deepseek-v4-fixed",
        type=CustomDataset,
        path=DATASET_PATH,
        reader_cfg=dict(input_columns=["question"], output_column="answer"),
        infer_cfg=dict(
            prompt_template=dict(type=PromptTemplate, template="{question}"),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
    ),
]
