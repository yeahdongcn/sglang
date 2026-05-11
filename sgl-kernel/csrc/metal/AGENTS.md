# AGENTS.md

Scope: instructions for AI agents working with Metal kernel code and the
benchmark/investigation markdown files under `sgl-kernel/csrc/metal/`.

This file is the first stop for agents. Use `METAL_DOC_INDEX.md` as the
secondary routing table after these instructions.

If a broader `AGENTS.md` is added higher in the repository, follow both files.
For conflicts while editing files in this directory, this closer file controls
unless the user gives a direct instruction in chat.

## Start Here

- Final accepted serving result: `METAL_FINAL_PERF_RESULTS_2026_05_11.md`.
- Formal completion audit: `METAL_COMPLETION_AUDIT_2026_05_10.md`.
- Routing/search index: `METAL_DOC_INDEX.md`.
- Test-record taxonomy and writeup rules: `METAL_TEST_RECORDS_GUIDE.md`.
- Architecture and remaining raw-kernel work:
  `FULL_FLASH_ATTENTION_METAL_PLAN.md`.
- Chronological regression history:
  `MLX_METAL_REGRESSION_INVESTIGATION.md`.
- Historical proof-of-concept material:
  `FLASH_ATTENTION_METAL_PLAN.md` and `BENCHMARK_RESULTS.md`.

## Current Accepted State

- The 2026-05-11 serving-baseline objective is met by the retained hybrid
  MLX/Metal runtime.
- The accepted state is not a universal raw direct-Metal replacement. Dense MLX
  fast attention still wins several raw contiguous decode/prefill rows.
- Correctness and performance are a joint gate. Do not claim a performance win
  unless the matching correctness command is recorded.
- Final focused correctness gate:

```text
PYTHONPATH=sgl-kernel/python:python ./sglang-mlx/bin/python -m pytest \
  sgl-kernel/tests/test_metal.py \
  test/registered/unit/hardware_backend/mlx/kv_cache/test_attention_wrapper.py \
  test/registered/unit/hardware_backend/mlx/kv_cache/test_paged_cache.py \
  test/registered/unit/hardware_backend/mlx/test_model_runner.py -q

156 passed, 6 warnings
```

- Original baseline commit: `c4903e67bf50f50286406273dedc52190fe9011f`.
- Final compacted target commit:
  `5b674a5a7d71f386f074396dfab3974ec93055fc`.
- Older notes may describe the same target state as
  `88eb26ca540572507a47b039be43093b75a4f838` plus local changes before the
  squash.

## Working Rules

- Preserve the working tree. Do not reset, checkout, or discard local changes
  while using these notes.
- Prefer paired same-window benchmark comparisons. Do not rely on a lone saved
  run when machine load or warmup variance can explain the result.
- Record benchmark command lines, environment variables, output paths, baseline
  source, current source, and percent deltas.
- Classify every new test or benchmark record with
  `METAL_TEST_RECORDS_GUIDE.md` before choosing where to write it.
- When a probe is rejected and reverted, write down why and add it to the most
  specific result file plus the `Do Not Retry Without New Evidence` section in
  `METAL_DOC_INDEX.md`.
- Keep old artifacts marked as historical or superseded. Do not silently rewrite
  history; append dated continuation notes when preserving provenance matters.

## Update Matrix

| Change type | Update these docs |
|---|---|
| Final serving result changes | `METAL_FINAL_PERF_RESULTS_2026_05_11.md`, `METAL_COMPLETION_AUDIT_2026_05_10.md`, `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md`, `METAL_DOC_INDEX.md`, this file |
| Raw direct-kernel gate changes | `FULL_FLASH_ATTENTION_METAL_PLAN.md`, the specific `METAL_MICROBENCH_RESULTS_*.md` artifact, `METAL_DOC_INDEX.md` |
| New test or benchmark record | Classify with `METAL_TEST_RECORDS_GUIDE.md`, write the specific result artifact, update `METAL_DOC_INDEX.md` if a new artifact or rejected idea is added |
| New rejected experiment | The specific result artifact, `METAL_DOC_INDEX.md` |
| Public Metal API or wrapper limit changes | `README.md`, tests under `sgl-kernel/tests/`, relevant plan/result docs |
| Environment or benchmark command changes | `BENCHMARK_RESULTS.md` or the active result artifact, then `METAL_DOC_INDEX.md` if discoverability changes |

## Retrieval Keywords

Use these terms when searching this directory:

`mlx-metal`, `apple-silicon`, `flashattention`, `paged-kv`, `radix-cache`,
`p1`, `bf16`, `serving-baseline`, `raw-kernel-gap`, `accepted-serving`,
`serving-benchmark`, `raw-microbench`, `cache-behavior`, `rejected`,
`5b674a5a`, `c4903e67`, `88eb26ca`.
