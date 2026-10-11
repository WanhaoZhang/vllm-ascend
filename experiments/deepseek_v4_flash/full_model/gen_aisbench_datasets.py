#!/usr/bin/env python3
"""Generate deterministic, exact-length DeepSeek-V4 chat prompts for AISBench."""

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

from vllm.tokenizers import get_tokenizer


CORPUS = [
    "The experiment records how the system behaves under a fixed workload.",
    "A reproducible benchmark keeps the request order and sampling settings constant.",
    "Researchers compare the latency, throughput, and correctness of each implementation.",
    "The service receives requests from several clients and schedules them together.",
    "The report includes the measured input length and the generated output length.",
    "Engineers verify the software environment before starting a performance run.",
    "The model processes the prompt and returns a concise answer to the user.",
    "Long inputs and short inputs exercise different parts of the inference pipeline.",
    "每次实验都记录软件版本、请求顺序、输入长度和输出长度。",
    "性能结果需要同时记录吞吐、首字延迟和生成阶段的延迟。",
    "多个请求可以并发提交，但每个后端都使用完全相同的数据。",
    "测试开始前确认服务健康，结束后确认设备进程已经释放。",
]

FILLERS = (
    " a", " the", " of", " to", " and", " in", " is", " on", " for", " with",
    " data", " model", " token", " system", " service", " request", " test", " text",
    " x", " y", " 1", " 2", ".", ",", ";", " because", " example", " result",
    " latency", " throughput", " client", " server", " input", " output", " batch",
)


def make_counter(tokenizer):
    def count(question: str) -> int:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(encoded)

    return count


def build_question(
    target: int,
    rng: random.Random,
    count,
    overhead: int,
    prefix_tokens: int,
    sentence_tokens: dict[str, int],
) -> str:
    prefix = "Please analyze the following material and identify its main ideas. "
    parts = [prefix]
    estimated_tokens = prefix_tokens
    while estimated_tokens < target - overhead + 64:
        sentence = rng.choice(CORPUS)
        parts.append(sentence)
        estimated_tokens += sentence_tokens[sentence]
    text = " ".join(parts)

    low, high = 0, len(text)
    best = ""
    best_count = overhead
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid].rstrip()
        candidate_count = count(candidate)
        if candidate_count <= target:
            if candidate_count > best_count:
                best, best_count = candidate, candidate_count
            low = mid + 1
        else:
            high = mid - 1

    while best_count < target:
        remaining = target - best_count
        candidates = []
        for suffix in FILLERS:
            candidate = best + suffix
            candidate_count = count(candidate)
            delta = candidate_count - best_count
            if 0 < delta <= remaining:
                candidates.append((delta, candidate, candidate_count))
        if not candidates:
            raise RuntimeError(
                f"Could not construct an exact {target}-token prompt; gap={remaining}"
            )
        _, best, best_count = max(candidates, key=lambda item: item[0])

    if best_count != target:
        raise AssertionError(f"expected {target} input tokens, got {best_count}")
    return best


def write_dataset(path: Path, rows) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/models/DeepSeek-V4-Flash")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples < 40:
        raise ValueError("--samples must be at least 40 for the 80:20 mixed dataset")

    tokenizer = get_tokenizer(
        args.model,
        tokenizer_mode="deepseek_v4",
        trust_remote_code=True,
    )
    count = make_counter(tokenizer)
    overhead = count("")
    prefix = "Please analyze the following material and identify its main ideas. "
    prefix_tokens = max(0, count(prefix) - overhead)
    sentence_tokens = {
        sentence: max(0, count(sentence) - overhead) for sentence in CORPUS
    }
    manifest = {
        "model": str(args.model),
        "tokenizer_mode": "deepseek_v4",
        "seed": args.seed,
        "samples_per_scenario": args.samples,
        "length_definition": "vLLM DeepseekV4Tokenizer apply_chat_template token count",
        "files": {},
    }

    def make_rows(target: int, n: int, seed_offset: int):
        rng = random.Random(args.seed + seed_offset)
        rows = []
        lengths = []
        for _ in range(n):
            question = build_question(
                target,
                rng,
                count,
                overhead,
                prefix_tokens,
                sentence_tokens,
            )
            lengths.append(count(question))
            rows.append({"question": question, "answer": ""})
        if set(lengths) != {target}:
            raise AssertionError(f"unexpected input lengths for {target}: {set(lengths)}")
        return rows, lengths

    datasets = [
        ("decode-128.jsonl", [(128, args.samples, 1)]),
        ("prefill-2048.jsonl", [(2048, args.samples, 2)]),
        ("prefill-4096.jsonl", [(4096, args.samples, 3)]),
    ]
    for filename, groups in datasets:
        rows, lengths = make_rows(*groups[0])
        path = args.out / filename
        digest = write_dataset(path, rows)
        manifest["files"][filename] = {
            "sha256": digest,
            "rows": len(rows),
            "input_tokens": {"min": min(lengths), "median": statistics.median(lengths), "max": max(lengths)},
        }
        print(f"{filename}: {len(rows)} rows, {digest}", flush=True)

    short_count = args.samples * 4 // 5
    long_count = args.samples - short_count
    short_rows, _ = make_rows(512, short_count, 4)
    long_rows, _ = make_rows(4096, long_count, 5)
    mixed_rows = short_rows + long_rows
    random.Random(args.seed + 6).shuffle(mixed_rows)
    mixed_path = args.out / "mixed-512-4096-80-20.jsonl"
    mixed_digest = write_dataset(mixed_path, mixed_rows)
    mixed_lengths = [count(row["question"]) for row in mixed_rows]
    manifest["files"][mixed_path.name] = {
        "sha256": mixed_digest,
        "rows": len(mixed_rows),
        "input_tokens": {"min": min(mixed_lengths), "median": statistics.median(mixed_lengths), "max": max(mixed_lengths)},
        "mix_counts": {"512": short_count, "4096": long_count},
    }
    print(f"{mixed_path.name}: {len(mixed_rows)} rows, {mixed_digest}", flush=True)

    manifest_path = args.out / "dataset-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
