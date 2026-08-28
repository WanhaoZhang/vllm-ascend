#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_NAMES = (
    "native_local_vs_native_reduced",
    "native_local_vs_catccos_pre_reduce",
    "native_reduced_vs_catccos_pre_reduce",
    "native_reduced_vs_catccos_post_reduce",
    "catccos_pre_reduce_vs_catccos_post_reduce",
)


def _format_number(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.6g}"


def _load_records(output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("probe-rank*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("schema_version", 0) >= 2:
                    records.append(record)
    if not records:
        raise SystemExit(f"No schema-v2 probe records found under {output_dir}")
    return sorted(records, key=lambda record: (record["rank"], record["moe_instance_id"], record["call_id"]))


def _print_record(record: dict[str, Any]) -> None:
    context = record["parallel_context"]
    print(
        f"rank={record['rank']} ep={record['ep_rank']}/{record['ep_world_size']} "
        f"tp={context['tp_rank']}/{context['tp_world_size']} layer={record['moe_instance_id']} "
        f"M={record['token_count']} comm={context['moe_comm_type']} outer={context['outer_reduction']}"
    )
    print("  comparison                                      cosine       rel_l2    norm_ratio")
    for name in METRIC_NAMES:
        metrics = record.get("stage_metrics", {}).get(name)
        if metrics is None:
            continue
        print(
            f"  {name:<47}"
            f"{_format_number(metrics.get('cosine_similarity')):>10} "
            f"{_format_number(metrics.get('relative_l2')):>12} "
            f"{_format_number(metrics.get('norm_ratio')):>13}"
        )


def _print_diagnosis(records: list[dict[str, Any]]) -> None:
    aligned = [
        record["stage_metrics"]["native_reduced_vs_catccos_pre_reduce"]
        for record in records
        if "native_reduced_vs_catccos_pre_reduce" in record.get("stage_metrics", {})
    ]
    final = [
        record["stage_metrics"]["native_reduced_vs_catccos_post_reduce"]
        for record in records
        if "native_reduced_vs_catccos_post_reduce" in record.get("stage_metrics", {})
    ]
    if not aligned or not final:
        return

    aligned_cosine = sum(float(metric["cosine_similarity"]) for metric in aligned) / len(aligned)
    final_cosine = sum(float(metric["cosine_similarity"]) for metric in final) / len(final)
    aligned_l2 = sum(float(metric["relative_l2"]) for metric in aligned) / len(aligned)
    final_l2 = sum(float(metric["relative_l2"]) for metric in final) / len(final)
    print("\nAggregate diagnostic (all ranks):")
    print(f"  native_reduced vs catccos_pre : mean cosine={aligned_cosine:.6g}, mean rel_l2={aligned_l2:.6g}")
    print(f"  native_reduced vs catccos_post: mean cosine={final_cosine:.6g}, mean rel_l2={final_l2:.6g}")

    contexts = {record["parallel_context"]["outer_reduction"] for record in records}
    if aligned_cosine > 0.99 and aligned_l2 < final_l2 and "tp-all-reduce" in contexts:
        print("  RESULT: CatCCOS pre-reduce is already aligned with native's complete output;")
        print("          applying the normal outer TP all-reduce again is a double-reduction candidate.")
    elif aligned_cosine < 0.9:
        print("  RESULT: stage alignment does not explain the discrepancy; inspect CatCCOS's")
        print("          input-sharding/routing/weight contract using the saved first-layer tensors.")
    else:
        print("  RESULT: the outcome is mixed; inspect per-rank values before changing the runtime path.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the CatCCOS first-layer four-stage reduction probe")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    records = _load_records(args.output_dir)
    for record in records:
        _print_record(record)
    _print_diagnosis(records)


if __name__ == "__main__":
    main()
