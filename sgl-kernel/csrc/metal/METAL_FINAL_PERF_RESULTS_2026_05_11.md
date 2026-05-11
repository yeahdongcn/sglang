# MLX/Metal Final Commit Benchmark Comparison

Date: 2026-05-11

## Agent Retrieval Note

- Status: canonical final serving-baseline comparison for the 2026-05-11
  audit after the work was compacted into commit
  `5b674a5a7d71f386f074396dfab3974ec93055fc`.
- Use when: answering "what was the final perf change?", comparing radix on/off
  workloads, comparing `5b674a5a` against `c4903e67`, or checking the accepted
  correctness gate.
- Do not use as: proof that direct raw Metal kernels universally replace dense
  MLX fast attention; that caveat remains open.
- Read next: `METAL_DOC_INDEX.md`,
  `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md`,
  `METAL_COMPLETION_AUDIT_2026_05_10.md`.
- Search tags: `final-results`, `accepted-serving`, `no-radix`, `radix-cache`,
  `correctness-gate`, `5b674a5a`, `c4903e67`.

This records the final accepted serving-baseline result for the retained
hybrid MLX/Metal FlashAttention path at commit
`5b674a5a7d71f386f074396dfab3974ec93055fc`, compared with the previous MLX
baseline `c4903e67bf50f50286406273dedc52190fe9011f`.

The current commit is the squashed form of the earlier audit state that some
older notes call `88eb26ca...` plus local changes.

## Metric Convention

- Radix off uses overall throughput. Percentage is
  `(current / baseline - 1) * 100`.
- Radix on uses end-to-end request latency for a BS=1 prefix-cache serving
  sequence. Percentage is latency reduction:
  `(baseline - current) / baseline * 100`.
- Positive percentage means commit `5b674a5a` is better than the `c4903e67`
  baseline for that row.

## Summary

| Radix cache | Batch / probe | Metric | `c4903e67` baseline | `5b674a5a` current | Performance change |
|---|---:|---|---:|---:|---:|
| Off | BS=1 | Overall throughput | 218.12 tok/s | 219.26 tok/s | +0.52% |
| Off | BS=4 | Overall throughput | 329.47 tok/s | 345.65 tok/s | +4.91% |
| Off | BS=8 | Overall throughput | 460.29 tok/s | 492.00 tok/s | +6.89% |
| On | BS=1 `short_first` | Request latency | 1.153 s | 1.133 s | +1.78% |
| On | BS=1 `short_hit1` | Request latency | 2.227 s | 0.686 s | +69.20% |
| On | BS=1 `short_hit2` | Request latency | 0.635 s | 0.629 s | +0.91% |
| On | BS=1 `long_partial` | Request latency | 2.601 s | 2.389 s | +8.13% |
| On | BS=1 `long_full` | Request latency | 0.862 s | 0.771 s | +10.55% |
| On | BS=1 `delayed_short` | Request latency | 1.992 s | 1.861 s | +6.59% |

Radix-on BS=4/BS=8 rows were not part of this benchmark set. The radix cache
gate is the fixed single-request prefix-hit sequence above because it exercises
cache reuse behavior that standalone `bench_one_batch` does not model.

## Evidence Files

No-radix current, `5b674a5a`:

```text
/tmp/sglang_current_no_radix_final_audit_20260511.jsonl
/tmp/sglang_current_no_radix_b1_final_reps_{1,2,3}_20260511.jsonl
```

No-radix baseline, `c4903e67`:

```text
/tmp/sglang_baseline_no_radix_samewindow_012839_20260511.jsonl
/tmp/sglang_baseline_no_radix_b1_currentwindow_reps_{1,2,3}_20260511.jsonl
```

Radix current, `5b674a5a`:

```text
/tmp/sglang_current_radix_default_b1ctxfast_norepatch_noprofile_41843_20260511.jsonl
/tmp/sglang_current_radix_final384_41753_014051_20260511.jsonl
/tmp/sglang_current_radix_final384_rerun_44212_014155_20260511.jsonl
/tmp/sglang_current_radix_final384_rerun2_47585_014856_20260511.jsonl
/tmp/sglang_current_radix_profile_36867_20260511.jsonl
/tmp/sglang_current_radix_revertclean_36604_20260511.jsonl
/tmp/sglang_current_radix_revertclean2_33598_20260511.jsonl
```

Radix baseline, `c4903e67`:

```text
/tmp/sglang_baseline_radix_samewindow_b1smallfull_45978_20260511.jsonl
/tmp/sglang_baseline_radix_rerun_44069_015602_20260511.jsonl
```

## Reproduction Commands

No-radix throughput sweep from the current worktree:

```text
PYTHONPATH=sgl-kernel/python:python SGLANG_USE_MLX=1 \
  ./sglang-mlx/bin/python -m sglang.bench_one_batch \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --trust-remote-code \
  --device mps \
  --batch-size 1 4 8 \
  --input-len 256 \
  --output-len 32 \
  --result-filename /tmp/sglang_current_no_radix_final_audit_20260511.jsonl
```

No-radix baseline sweep from the safe baseline worktree:

```text
cd /private/tmp/sglang-c490
PYTHONPATH=sgl-kernel/python:python SGLANG_USE_MLX=1 \
  /Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx/bin/python \
  -m sglang.bench_one_batch \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --trust-remote-code \
  --device mps \
  --batch-size 1 4 8 \
  --input-len 256 \
  --output-len 32 \
  --result-filename /tmp/sglang_baseline_no_radix_samewindow_012839_20260511.jsonl
```

Radix serving probe for current or baseline, changing `--root`, `--port`,
`--output`, and `--log` per run:

```text
PYTHONPATH=sgl-kernel/python:python \
  ./sglang-mlx/bin/python sgl-kernel/csrc/metal/bench_mlx_radix_serving.py \
  --root /Users/yexiaodong/go/src/github.com/yeahdongcn/sglang \
  --python /Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx/bin/python \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --port 41753 \
  --output /tmp/sglang_current_radix_final384_41753_014051_20260511.jsonl \
  --log /tmp/sglang_current_radix_final384_41753_014051_20260511.jsonl.log
```

## Correctness Gate

```text
PYTHONPATH=sgl-kernel/python:python ./sglang-mlx/bin/python -m pytest \
  sgl-kernel/tests/test_metal.py \
  test/registered/unit/hardware_backend/mlx/kv_cache/test_attention_wrapper.py \
  test/registered/unit/hardware_backend/mlx/kv_cache/test_paged_cache.py \
  test/registered/unit/hardware_backend/mlx/test_model_runner.py -q

156 passed, 6 warnings
```

## Decision

The serving-native no-radix and radix-cache paths beat the previous MLX
baseline on the recorded paired workloads. The accepted production route
remains hybrid: direct raw Metal kernels are still not a universal replacement
for dense MLX fast attention on every raw contiguous decode/prefill
microbenchmark row.
