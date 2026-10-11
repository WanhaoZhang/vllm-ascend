# DeepSeek V4 Flash experiment code

This branch preserves the launch, numerical validation, AISBench and causal
diagnosis scripts used on 2026-10-11. The five accepted implementation commits
remain unchanged. This harness branch is based on their common vLLM-Ascend
baseline, `3005f23d8ba67bff1f94bb898f7681c3000d6d94`.

## Implementation branches

| Repository | Branch | Tested commit |
| --- | --- | --- |
| vllm-ascend | codex/deepseek-v4-flash-catccos | dd0b308e3a2903186a204131cd5e94088ccfe9cd |
| CATCCOS | codex/deepseek-v4-flash-catccos | e0fb097423c4195c33fd38d2b7caf4fdd1608751 |
| vllm-ascend | codex/deepseek-v4-flash-deepmegamoe | 832850b0631e6ff435bd2f4e67e0654e2d9bad90 |
| DeepGEMM-Ascend | codex/deepseek-v4-flash-deepmegamoe | c07023c31900b3d3f6cb43daefa77f4efe953883 |
| vllm-ascend | codex/deepseek-v4-flash-native | 55e6eb7504e81e8dccf76f5ff06b645b21979a5d |

The repositories are under `WanhaoZhang` on GitHub and `zhangwanhao` on GitCode.
GitCode uses the lower-case repository name `catccos`.

## Reproduce the accepted measurements

Use [the full-model instructions](full_model/REPRODUCE.md). The scripts retain
the original container paths for reproducibility. Deploy `full_model/` into
`/home/z009565592/catccos-vllm/work/deepseek-v4-test/`, with the implementation
checkouts alongside it as described in the instructions. The container also
mounts `/home/z00956592/catccos-vllm` as an alias of the same checkout.

Obtain the official checkpoint separately at `/data/models/DeepSeek-V4-Flash`.
Build the matching custom operators, CatCCOS/SHMEM and DeepGEMM extensions;
the original GCC 14 runtime directory and container packages are prerequisites.
Use the same four NPUs and one backend at a time, in this order: CatCCOS,
deepMegaMoE, native. All use DP4/TP1/EP4, eager, no MTP or prefix cache,
max-model-len 8192, max-num-batched-tokens 4096 and max-num-seqs 32.

The accepted AISBench matrix has 63 measured rounds, 11,088 requests and zero
request errors. [Medians](evidence/medians.csv) and the
[dataset manifest](evidence/dataset-manifest.json) provide compact provenance.
Full logs and raw traces remain in the local experiment archive.

## Performance diagnosis

Deploy `diagnosis/` into the sibling `deepseek-v4-diagnosis-20261011/` directory.
The cache diagnostic requires a separate source copy with the probe installed;
it is a causal experiment, not an accepted implementation or benchmark backend.
Review the scripts before running them on a different checkout layout.

The [diagnosis summary](evidence/diagnosis-summary.json) records the four-rank
packed versus exactly expanded weight ablation and the bounded eight-layer
cache measurements. Profiler attempts failed and their traces must not be used
as valid device critical-path measurements. Greedy text agreement checks do
not establish downstream task accuracy.

`source-manifest.json` records hashes of the published original scripts and
compact evidence. Model weights, compiled binaries, credentials and large raw
results are excluded from this code branch.
