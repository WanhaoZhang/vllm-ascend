#!/usr/bin/env python3
"""Verify checkpoint shard headers against the index without reading NPU data."""

import argparse
import hashlib
import json
import struct
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    index_path = args.model / "model.safetensors.index.json"
    index_bytes = index_path.read_bytes()
    weight_map = json.loads(index_bytes)["weight_map"]
    shards = []
    errors = []
    found = {}
    for name in sorted(set(weight_map.values())):
        path = args.model / name
        if not path.is_file():
            errors.append(f"missing shard: {name}")
            continue
        size = path.stat().st_size
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < header_size <= min(size - 8, 100_000_000):
                errors.append(f"invalid header length: {name}")
                continue
            header_bytes = stream.read(header_size)
            header = json.loads(header_bytes)
        tensors = {key: value for key, value in header.items() if key != "__metadata__"}
        offsets = []
        for key, metadata in tensors.items():
            if key in found:
                errors.append(f"duplicate tensor: {key}")
            found[key] = name
            begin, end = metadata["data_offsets"]
            if not 0 <= begin <= end <= size - 8 - header_size:
                errors.append(f"invalid tensor bounds: {key}")
            offsets.append((begin, end))
        offsets.sort()
        cursor = 0
        for begin, end in offsets:
            if begin != cursor:
                errors.append(f"non-contiguous tensor payload: {name}, {cursor}->{begin}")
            cursor = end
        if cursor + 8 + header_size != size:
            errors.append(f"shard payload length mismatch: {name}")
        shards.append(dict(file=name, bytes=size, tensors=len(tensors),
                           header_sha256=hashlib.sha256(header_bytes).hexdigest()))
    if found != weight_map:
        errors.append(f"index/header mapping mismatch: missing={len(set(weight_map) - set(found))}, "
                      f"unexpected={len(set(found) - set(weight_map))}, "
                      f"wrong_shard={sum(found.get(key) != value for key, value in weight_map.items())}")
    config_path = args.model / "config.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    tokenizer_files = sorted({*args.model.glob('*token*'), *args.model.glob('*template*'),
                              args.model / 'generation_config.json'})
    tokenizer_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in tokenizer_files if path.is_file()}
    result = dict(model=str(args.model), passed=not errors, errors=errors,
                  shard_count=len(shards), tensor_count=len(found),
                  total_bytes=sum(shard["bytes"] for shard in shards),
                  index_sha256=hashlib.sha256(index_bytes).hexdigest(),
                  config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                  tokenizer_files_sha256=tokenizer_hashes,
                  config={key: config.get(key) for key in
                          ("model_type", "architectures", "hidden_size", "moe_intermediate_size",
                           "n_routed_experts", "n_shared_experts", "num_experts_per_tok", "swiglu_limit")},
                  integrity_scope="Index, tensor names, header hash, offsets and complete payload length; "
                                  "no upstream full-file digest is available for comparison.",
                  shards=shards)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "shards"}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
