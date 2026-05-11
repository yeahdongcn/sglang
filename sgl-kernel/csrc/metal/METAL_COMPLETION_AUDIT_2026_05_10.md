# Metal FlashAttention Completion Audit

Generated: `2026-05-10`

## Agent Retrieval Note

- Status: latest formal completion audit. The early 2026-05-10 decision says
  the goal was open; the `2026-05-11 Serving Completion Update` later in this
  file supersedes that for serving throughput/latency.
- Use when: deciding whether the Metal serving-baseline objective is complete,
  checking the prompt-to-artifact checklist, or finding the final correctness
  command.
- Do not use as: evidence that the raw direct Metal kernel gate is complete.
  The raw contiguous dense-MLX caveat remains open.
- Read next: `METAL_FINAL_PERF_RESULTS_2026_05_11.md`,
  `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md`,
  `FULL_FLASH_ATTENTION_METAL_PLAN.md`.
- Search tags: `completion-audit`, `accepted-serving`, `correctness-gate`,
  `open-raw-gate`, `radix-cache`, `no-radix`.

## Objective

Make the current MLX/Metal FlashAttention work at compacted commit
`5b674a5a7d71f386f074396dfab3974ec93055fc` correct and faster than the
previous MLX baseline at
`c4903e67bf50f50286406273dedc52190fe9011f`.

`5b674a5a` is the squashed form of the earlier audit snapshot that older notes
called `88eb26ca...` plus local changes.

## Prompt-to-Artifact Checklist

| Requirement | Evidence | Status |
|---|---|---|
| Use the current worktree at compacted commit `5b674a5a...` | Current target contains the retained MLX runner, paged-cache, Metal wrapper, benchmark, and plan updates | Met for this audit snapshot |
| Compare against previous MLX baseline `c4903e67...` | Same-machine `/tmp/sglang-c490` paired no-radix and radix serving files are recorded in `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md` | Met for recorded workloads |
| Preserve correctness | Latest focused command after the retained radix decode threshold and rejected tight-cache revert: `PYTHONPATH=sgl-kernel/python:python ./sglang-mlx/bin/python -m pytest sgl-kernel/tests/test_metal.py test/registered/unit/hardware_backend/mlx/kv_cache/test_attention_wrapper.py test/registered/unit/hardware_backend/mlx/kv_cache/test_paged_cache.py test/registered/unit/hardware_backend/mlx/test_model_runner.py -q` reports `156 passed, 6 warnings` | Met for covered paths |
| Improve no-radix serving path | 2026-05-11 paired B1 median and final B4/B8 audit rows beat baseline: B1 `219.26` vs `218.12 tok/s`, B4 `345.65` vs `329.47 tok/s`, B8 `492.00` vs `460.29 tok/s` | Met |
| Improve radix-cache serving path | 2026-05-11 accepted-code medians beat the two-run baseline on all probes, including the previously marginal `short_hit2` row: `0.629` vs `0.635 s` | Met |
| Compare raw Metal kernels against `mx.fast.scaled_dot_product_attention` | `/tmp/sglang_raw_micro_audit_20260510.md` and `/tmp/sglang_raw_micro_dense_h128_20260510.md` compare direct Metal, dense MLX, gathered MLX, radix layouts, prefill, and scatter | Met |
| Raw Metal decode beats dense MLX on target matrix | Direct paged/radix layout wins many gathered-radix rows, but dense contiguous rows remain behind: B1/S2162, B4/S288, B4/S2162, B8/S288 | Open |
| Raw Metal prefill beats dense MLX on target matrix | Metal wins long S1024 rows, but short/prefix S256 rows are mixed or behind | Open |
| Document rejected directions | Current docs record idle recompute, p1 float16 side-store, cached flat p1 views, nonblocking scatter, blanket full-state radix prefill, dense warmup, and prior raw decode variants | Met |

## Current Evidence

No-radix paired serving:

```text
current:  /tmp/sglang_current_no_radix_align32_full_decode_final_full_20260510.jsonl
baseline: /tmp/sglang_baseline_no_radix_align32_full_decode_final_full_20260510.jsonl
```

| Path | B1 tok/s | B4 tok/s | B8 tok/s |
|---|---:|---:|---:|
| current | 221.96 | 344.74 | 492.27 |
| c4903e67 baseline | 219.56 | 334.45 | 478.80 |

Latest same-window radix serving refresh:

```text
current default:   /tmp/sglang_current_radix_live_default_53374_20260510.jsonl
current profiled:  /tmp/sglang_current_radix_live_profile_53625_20260510.jsonl
baseline:          /tmp/sglang_baseline_radix_live_53910_20260510.jsonl
current threshold: /tmp/sglang_current_radix_fullstate_long_decode_54367_20260510.jsonl
```

| Probe | Current default s | Current profiled s | c4903e67 baseline s | Current threshold s | Status |
|---|---:|---:|---:|---:|---|
| short_first | 1.173 | 1.076 | 1.185 | 1.218 | mixed; not a stable win |
| short_hit1 | 0.693 | 0.563 | 2.119 | 0.739 | ahead |
| short_hit2 | 0.768 | 0.668 | 0.627 | 0.784 | behind |
| long_partial | 2.445 | 2.344 | 2.420 | 2.436 | on-par/mixed |
| long_full | 0.873 | 0.757 | 0.784 | 0.776 | threshold/profile ahead |
| delayed_short | 2.225 | 2.642 | 3.242 | 2.244 | ahead, high variance |

Raw dense h128 decode continuation:

| Decode row | MLX fast dense ms | Best dense h128 Metal ms | Status |
|---|---:|---:|---|
| B1/S288 | 0.257 | 0.264 | behind |
| B1/S2162 | 0.435 | 0.507 | behind |
| B4/S288 | 0.529 | 0.553 | behind |
| B4/S2162 | 0.897 | 1.119 | behind |
| B8/S288 | 0.542 | 0.700 | behind |
| B8/S2162 | 1.734 | 1.677 | ahead |

## Audit Decision

2026-05-10 decision: the goal was still open.

The serving path is materially improved, but the objective is not complete.
No-radix is ahead in the latest paired full sweep. Radix serving is still open:
`short_hit2` is behind the same-window baseline, `long_partial` is only mixed,
and the retained long-context decode threshold only closes the `long_full` row.
The direct raw Metal FlashAttention gate is also not satisfied because dense MLX
remains faster on several contiguous decode and short-prefill rows. The current
production direction should remain hybrid until a raw dense/paged kernel clears
those rows and radix serving beats baseline across the probe.

Post-audit continuation note: making the idle paged-prefill route default-on was
tested and rejected after this table because the no-env confirmation regressed
delayed short to `2.869 s` with `idle_paged=True`. It remains env-gated.

Dense contiguous GQA2 decode was also tested after this table and rejected. The
retained `/tmp/sglang_raw_micro_dense_gqa2_20260510.md` run lost the B1/B4/B8
S288/S2162 dense rows versus MLX fast, so the raw direct-kernel gate is still
open.

Post-refresh continuation note: reducing radix request-local contiguous cache
capacity below `_max_seq_len` was tested and rejected after
`/tmp/sglang_current_radix_tight_cache_41415_20260510.jsonl`: delayed short
improved to `1.737 s`, but `long_partial` regressed to `2.839 s` and
`short_hit2` stayed behind at `0.787 s`.

Raising `_RADIX_FULL_STATE_PREFILL_MIN_NEW_TOKENS` to `2048` was also tested
and rejected. `/tmp/sglang_current_radix_compact_long_prefill_48434_20260510.jsonl`
measured `long_partial=2.461 s`, `short_hit2=0.796 s`, and `long_full=0.892 s`,
so compact long-suffix prefill did not close the same-window baseline gap.

Eager radix stale-request cleanup before every extend was tested and rejected.
`/tmp/sglang_current_radix_eager_cleanup_36139_20260510.jsonl` improved delayed
short to `1.900 s`, but regressed `short_hit2` to `0.780 s` and `long_partial`
to `2.557 s`.

Full-state eval for all single-request radix fallback decode was tested and
rejected. `/tmp/sglang_current_radix_fullstate_all_decode_38349_20260510.jsonl`
improved delayed short to `1.761 s` and `long_full` to `0.764 s`, but regressed
`short_hit1` to `0.819 s` and `long_partial` to `2.500 s`, while `short_hit2`
stayed behind at `0.783 s`.

Further continuation probes did not close the radix gate. Active short-prefix
cache reuse improved delayed short and `long_full`, but the best gated run
(`/tmp/sglang_current_radix_active_prefix_small_suffix_33031_20260510.jsonl`)
still missed baseline on `short_hit2=0.736 s` and `long_partial=2.442 s`;
no-profile confirmation left `long_partial=2.476 s`. Raw-attention fallback
regressed `short_hit2=0.804 s` and `long_partial=2.649 s`. Default fp16 p1
side-store regressed all rows, and bf16 generic paged prefill timed out on the
long-partial request after a small direct correctness test passed. All of those
continuation experiments were reverted.

## 2026-05-11 Serving Completion Update

The serving-baseline objective is now met for the retained hybrid MLX/Metal
runtime. This update supersedes the earlier 2026-05-10 "goal open" decision for
no-radix and radix-cache serving throughput/latency. It does not change the
raw-direct-kernel caveat: dense MLX fast attention still wins several raw
contiguous microbenchmark rows, so direct paged Metal is not treated as a
universal replacement.

Final correctness command:

```text
PYTHONPATH=sgl-kernel/python:python ./sglang-mlx/bin/python -m pytest \
  sgl-kernel/tests/test_metal.py \
  test/registered/unit/hardware_backend/mlx/kv_cache/test_attention_wrapper.py \
  test/registered/unit/hardware_backend/mlx/kv_cache/test_paged_cache.py \
  test/registered/unit/hardware_backend/mlx/test_model_runner.py -q

156 passed, 6 warnings
```

No-radix final evidence:

| Row | Current | c4903e67 baseline | Status |
|---|---:|---:|---|
| B1 paired median | 219.26 tok/s | 218.12 tok/s | ahead |
| B4 final audit | 345.65 tok/s | 329.47 tok/s | ahead |
| B8 final audit | 492.00 tok/s | 460.29 tok/s | ahead |

Radix final evidence:

| Probe | Current median | c4903e67 baseline median | Status |
|---|---:|---:|---|
| short_first | 1.133 s | 1.153 s | ahead |
| short_hit1 | 0.686 s | 2.227 s | ahead |
| short_hit2 | 0.629 s | 0.635 s | ahead |
| long_partial | 2.389 s | 2.601 s | ahead |
| long_full | 0.771 s | 0.862 s | ahead |
| delayed_short | 1.861 s | 1.992 s | ahead |

Evidence files:

```text
no-radix current:
  /tmp/sglang_current_no_radix_final_audit_20260511.jsonl
  /tmp/sglang_current_no_radix_b1_final_reps_{1,2,3}_20260511.jsonl
no-radix baseline:
  /tmp/sglang_baseline_no_radix_samewindow_012839_20260511.jsonl
  /tmp/sglang_baseline_no_radix_b1_currentwindow_reps_{1,2,3}_20260511.jsonl
radix current:
  /tmp/sglang_current_radix_default_b1ctxfast_norepatch_noprofile_41843_20260511.jsonl
  /tmp/sglang_current_radix_final384_41753_014051_20260511.jsonl
  /tmp/sglang_current_radix_final384_rerun_44212_014155_20260511.jsonl
  /tmp/sglang_current_radix_final384_rerun2_47585_014856_20260511.jsonl
  /tmp/sglang_current_radix_profile_36867_20260511.jsonl
  /tmp/sglang_current_radix_revertclean_36604_20260511.jsonl
  /tmp/sglang_current_radix_revertclean2_33598_20260511.jsonl
radix baseline:
  /tmp/sglang_baseline_radix_samewindow_b1smallfull_45978_20260511.jsonl
  /tmp/sglang_baseline_radix_rerun_44069_015602_20260511.jsonl
```

Final rejected probes were reverted: compact-only B1 radix wrapper eval
(`/tmp/sglang_current_radix_compacteval_51583_20260511.jsonl`), per-token
slice eval (`/tmp/sglang_current_radix_b1sliceeval_33150_20260511.jsonl`), and
the 464-token right-sized short-prefix cache
(`/tmp/sglang_current_radix_rightsize464_43084_20260511.jsonl`).
