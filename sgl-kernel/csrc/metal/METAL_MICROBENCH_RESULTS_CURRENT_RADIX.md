# Metal Attention Microbenchmark Results

Generated: `2026-05-09 18:51:40`

## Agent Retrieval Note

- Status: consolidated benchmark ledger for the current radix/no-radix
  investigation. Later sections append serving refreshes and rejected probes
  through 2026-05-11.
- Use when: checking raw microbenchmark rows, same-window serving evidence,
  radix cache behavior, p1 lazy decode probes, or rejected continuation ideas.
- Do not use as: a single flat benchmark table. The earliest table is not the
  final answer; later dated sections supersede earlier serving decisions.
- Read next: `METAL_FINAL_PERF_RESULTS_2026_05_11.md`,
  `METAL_COMPLETION_AUDIT_2026_05_10.md`,
  `FULL_FLASH_ATTENTION_METAL_PLAN.md`.
- Search tags: `microbench`, `radix-cache`, `no-radix`, `p1`, `bf16`,
  `raw-kernel-gap`, `rejected`.

Command context:

```text
dtype=float16, heads=16, kv_heads=8, head_dim=128, block_size=16
warmups=3, decode/scatter iters=10, prefill iters=5
prefill_prefix_lens=[256], sync_layers=28
```

| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| decode | metal_paged | 1 | 288 | 0.3734 | 0.3827 | 0.3457 | 0.4375 |
| decode | mlx_fast_dense | 1 | 288 | 0.2260 | 0.2668 | 0.2152 | 0.5797 |
| decode | mlx_gather_paged_kv | 1 | 288 | 0.2694 | 0.2753 | 0.2397 | 0.3392 |
| decode | mlx_fast_gathered_paged | 1 | 288 | 0.3178 | 0.3342 | 0.2875 | 0.4615 |
| decode | metal_paged_radix_shuffled | 1 | 288 | 0.4423 | 0.4845 | 0.3843 | 0.7408 |
| decode | mlx_gather_radix_paged_kv | 1 | 288 | 0.3046 | 0.3375 | 0.2849 | 0.4450 |
| decode | mlx_fast_radix_gathered_paged | 1 | 288 | 0.3883 | 0.4606 | 0.3298 | 0.9996 |
| decode | metal_dense_wrapper | 1 | 288 | 0.8645 | 1.0771 | 0.7215 | 1.9485 |
| decode | metal_paged | 1 | 2162 | 0.6654 | 0.6823 | 0.5315 | 0.9176 |
| decode | mlx_fast_dense | 1 | 2162 | 0.4673 | 0.4985 | 0.4321 | 0.8238 |
| decode | mlx_gather_paged_kv | 1 | 2162 | 0.9958 | 1.0107 | 0.8805 | 1.2586 |
| decode | mlx_fast_gathered_paged | 1 | 2162 | 1.1508 | 1.1968 | 1.0805 | 1.4004 |
| decode | metal_paged_radix_shuffled | 1 | 2162 | 0.6572 | 0.6607 | 0.5873 | 0.8189 |
| decode | mlx_gather_radix_paged_kv | 1 | 2162 | 1.0091 | 1.0570 | 0.8739 | 1.6362 |
| decode | mlx_fast_radix_gathered_paged | 1 | 2162 | 1.7799 | 1.8639 | 1.2470 | 2.6450 |
| decode | metal_dense_wrapper | 1 | 2162 | 4.2976 | 4.6517 | 2.8930 | 6.9706 |
| decode | metal_paged | 4 | 288 | 0.7513 | 1.3152 | 0.5134 | 4.0293 |
| decode | mlx_fast_dense | 4 | 288 | 0.5471 | 0.5990 | 0.3644 | 1.1218 |
| decode | mlx_gather_paged_kv | 4 | 288 | 0.9188 | 0.9256 | 0.6476 | 1.3226 |
| decode | mlx_fast_gathered_paged | 4 | 288 | 0.9175 | 0.9612 | 0.8382 | 1.2433 |
| decode | metal_paged_radix_shuffled | 4 | 288 | 0.5300 | 0.5570 | 0.4750 | 0.8632 |
| decode | mlx_gather_radix_paged_kv | 4 | 288 | 0.7945 | 0.8415 | 0.6819 | 1.2666 |
| decode | mlx_fast_radix_gathered_paged | 4 | 288 | 0.9006 | 0.9356 | 0.8181 | 1.1790 |
| decode | metal_dense_wrapper | 4 | 288 | 1.2240 | 1.3453 | 0.9544 | 2.2665 |
| decode | metal_paged | 4 | 2162 | 1.2013 | 1.1877 | 0.9736 | 1.3649 |
| decode | mlx_fast_dense | 4 | 2162 | 1.0269 | 1.1164 | 0.8653 | 2.0961 |
| decode | mlx_gather_paged_kv | 4 | 2162 | 3.3852 | 3.4177 | 3.0582 | 3.8816 |
| decode | mlx_fast_gathered_paged | 4 | 2162 | 4.0770 | 4.1047 | 3.6713 | 5.0426 |
| decode | metal_paged_radix_shuffled | 4 | 2162 | 1.0986 | 1.1902 | 1.0658 | 1.4790 |
| decode | mlx_gather_radix_paged_kv | 4 | 2162 | 3.0966 | 3.1634 | 2.8843 | 3.5854 |
| decode | mlx_fast_radix_gathered_paged | 4 | 2162 | 3.8598 | 3.9250 | 3.5730 | 4.3808 |
| decode | metal_dense_wrapper | 4 | 2162 | 5.1029 | 5.1530 | 5.0014 | 5.3748 |
| decode | metal_paged | 8 | 288 | 0.6555 | 0.8913 | 0.6133 | 3.0325 |
| decode | mlx_fast_dense | 8 | 288 | 0.4645 | 0.5284 | 0.4147 | 1.1726 |
| decode | mlx_gather_paged_kv | 8 | 288 | 0.9921 | 1.0008 | 0.9290 | 1.1443 |
| decode | mlx_fast_gathered_paged | 8 | 288 | 1.2133 | 1.2202 | 1.1722 | 1.3096 |
| decode | metal_paged_radix_shuffled | 8 | 288 | 0.6497 | 0.6459 | 0.6068 | 0.6701 |
| decode | mlx_gather_radix_paged_kv | 8 | 288 | 1.1588 | 1.2037 | 0.9433 | 1.6875 |
| decode | mlx_fast_radix_gathered_paged | 8 | 288 | 1.2200 | 1.3584 | 1.1363 | 2.4195 |
| decode | metal_dense_wrapper | 8 | 288 | 1.2361 | 1.3717 | 1.1384 | 2.3030 |
| decode | metal_paged | 8 | 2162 | 1.7513 | 1.9993 | 1.6239 | 3.3520 |
| decode | mlx_fast_dense | 8 | 2162 | 1.6542 | 1.7121 | 1.4209 | 2.3449 |
| decode | mlx_gather_paged_kv | 8 | 2162 | 6.9336 | 9.7047 | 5.7631 | 24.2705 |
| decode | mlx_fast_gathered_paged | 8 | 2162 | 8.3736 | 9.5818 | 7.0881 | 18.1481 |
| decode | metal_paged_radix_shuffled | 8 | 2162 | 1.8210 | 1.8783 | 1.6411 | 2.4513 |
| decode | mlx_gather_radix_paged_kv | 8 | 2162 | 5.8314 | 5.8600 | 5.4932 | 6.2615 |
| decode | mlx_fast_radix_gathered_paged | 8 | 2162 | 6.9880 | 6.9780 | 6.8622 | 7.0836 |
| decode | metal_dense_wrapper | 8 | 2162 | 10.1461 | 10.1425 | 10.0468 | 10.2379 |
| prefill | metal_prefill_paged_no_prefix | 1 | 256 | 0.8933 | 0.9346 | 0.8120 | 1.2103 |
| prefill | metal_flash_varlen | 1 | 256 | 0.8403 | 0.8586 | 0.8282 | 0.9428 |
| prefill | mlx_fast_dense | 1 | 256 | 0.6700 | 0.6668 | 0.6546 | 0.6798 |
| prefill | metal_prefill_paged_prefix_256 | 1 | 256 | 1.4950 | 1.5106 | 1.4574 | 1.6138 |
| prefill | mlx_fast_dense_prefix_256 | 1 | 256 | 0.9243 | 0.9736 | 0.9193 | 1.1374 |
| prefill | metal_prefill_paged_no_prefix | 1 | 1024 | 3.3995 | 3.3759 | 3.3034 | 3.4138 |
| prefill | metal_flash_varlen | 1 | 1024 | 3.2203 | 3.2581 | 3.2118 | 3.3368 |
| prefill | mlx_fast_dense | 1 | 1024 | 5.0964 | 5.0569 | 4.9541 | 5.1213 |
| prefill | metal_prefill_paged_prefix_256 | 1 | 1024 | 4.9720 | 5.1437 | 4.9298 | 5.8515 |
| prefill | mlx_fast_dense_prefix_256 | 1 | 1024 | 6.1152 | 6.1462 | 6.0328 | 6.3677 |
| prefill | metal_prefill_paged_no_prefix | 4 | 256 | 1.6463 | 1.7729 | 1.5884 | 2.3427 |
| prefill | metal_flash_varlen | 4 | 256 | 1.6801 | 1.7858 | 1.5359 | 2.4280 |
| prefill | mlx_fast_dense | 4 | 256 | 1.5858 | 2.0382 | 1.5566 | 3.5040 |
| prefill | metal_prefill_paged_prefix_256 | 4 | 256 | 3.3611 | 3.3482 | 3.2510 | 3.4130 |
| prefill | mlx_fast_dense_prefix_256 | 4 | 256 | 2.7206 | 2.7552 | 2.7129 | 2.8969 |
| prefill | metal_prefill_paged_no_prefix | 4 | 1024 | 10.9897 | 11.0961 | 10.9175 | 11.6242 |
| prefill | metal_flash_varlen | 4 | 1024 | 10.6398 | 10.6445 | 10.5543 | 10.7434 |
| prefill | mlx_fast_dense | 4 | 1024 | 18.5656 | 19.0314 | 18.3239 | 20.9734 |
| prefill | metal_prefill_paged_prefix_256 | 4 | 1024 | 15.9813 | 16.0317 | 15.9574 | 16.1720 |
| prefill | mlx_fast_dense_prefix_256 | 4 | 1024 | 23.1030 | 23.1106 | 22.9333 | 23.3994 |
| scatter | metal_paged_kv_scatter | 1 | 1 | 0.3775 | 0.3975 | 0.3569 | 0.4746 |
| scatter | metal_paged_kv_scatter | 1 | 4 | 0.3322 | 0.3317 | 0.3185 | 0.3459 |
| scatter | metal_paged_kv_scatter | 1 | 256 | 0.4563 | 0.4898 | 0.4366 | 0.7748 |
| radix_sync | mlx_set_kv_all_layers_stacked | 1 | 28 | 2.0597 | 2.0802 | 1.9452 | 2.2395 |
| radix_sync | mlx_set_kv_all_layers_list | 1 | 28 | 1.3882 | 1.4340 | 1.3393 | 1.6613 |
| radix_sync | mlx_set_kv_all_layers_stacked | 4 | 28 | 1.7405 | 1.8085 | 1.6040 | 2.5615 |
| radix_sync | mlx_set_kv_all_layers_list | 4 | 28 | 1.5058 | 1.5810 | 1.4073 | 2.1472 |
| radix_sync | mlx_set_kv_all_layers_stacked | 256 | 28 | 8.0237 | 7.9744 | 7.5016 | 8.1085 |
| radix_sync | mlx_set_kv_all_layers_list | 256 | 28 | 6.4931 | 6.6768 | 5.9536 | 8.1535 |

## Serving Refresh: 2026-05-10

The raw table above explains the routing decision: short contiguous B1 still
belongs on MLX fast attention, while direct paged Metal is useful mainly as
radix/paged infrastructure and for longer or shuffled radix-style decode rows.
The current accepted serving path is therefore hybrid rather than direct-paged
only.

No-radix paired serving benchmark:

```text
current:  /tmp/sglang_current_no_radix_align32_full_decode_final_full_20260510.jsonl
baseline: /tmp/sglang_baseline_no_radix_align32_full_decode_final_full_20260510.jsonl
command:  python -m sglang.bench_one_batch --disable-radix-cache --batch-size 1 4 8 --input-len 256 --output-len 32
```

| Path | B1 tok/s | B4 tok/s | B8 tok/s |
|---|---:|---:|---:|
| current | 221.96 | 344.74 | 492.27 |
| c4903e67 baseline | 219.56 | 334.45 | 478.80 |

Earlier paired radix sweeps looked broadly positive, but the latest same-window
live refresh is more mixed and is the current decision point:

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
| short_hit2 | 0.768 | 0.668 | 0.627 | 0.784 | behind in this live window |
| long_partial | 2.445 | 2.344 | 2.420 | 2.436 | on-par/mixed |
| long_full | 0.873 | 0.757 | 0.784 | 0.776 | threshold/profile ahead |
| delayed_short | 2.225 | 2.642 | 3.242 | 2.244 | ahead, but still high variance |

The accepted conclusion is narrower than the earlier median table: the current
radix path is improved on key cache-hit rows, especially `short_hit1` and the
delayed short hit versus this baseline run, but it is not robustly beyond the
baseline across the full probe. `short_hit2` and `long_partial` remain the
active serving gaps.

The latest retained routing tweak is `_RADIX_FULL_STATE_DECODE_MIN_TOKENS=1024`
for single-request radix fallback decode. It keeps compact cache eval for short
contexts and evaluates full cache state for long contexts. The confirmation run
improved `long_full` from the current default `0.873 s` to `0.776 s` and edged
the same-window baseline `0.784 s`, without claiming a broad radix completion.

Rejected experiments in this final pass:

- Idle recompute and wider small-prefix recompute: sequence-sensitive and
  regressed delayed or long-partial serving rows.
- Cached flat p1 views: unit-testable but caused Metal OOM during serving
  side-store sync.
- p1 float16 side-store storage: off-by-default diagnostic run regressed long
  partial (`3.221 s`) and delayed short (`2.970 s`), so the env knob was removed.
- Full-state prefill blanket routing: helped some immediate rows but regressed
  delayed short.
- Tightening radix request-local contiguous cache capacity to the prompt plus
  decode slack: improved delayed short in one run (`1.737 s`) but regressed
  `long_partial` to `2.839 s`, left `short_hit2` behind (`0.787 s`), and did
  not beat the retained long-full threshold path. The experiment is recorded in
  `/tmp/sglang_current_radix_tight_cache_41415_20260510.jsonl` and reverted.
- Raising `_RADIX_FULL_STATE_PREFILL_MIN_NEW_TOKENS` from `384` to `2048` so
  the `prefix=434/new=1728` long-partial row used compact eval. The profiled
  run `/tmp/sglang_current_radix_compact_long_prefill_48434_20260510.jsonl`
  measured `long_partial=2.461 s`, behind the same-window baseline `2.420 s`,
  and also regressed `short_hit2=0.796 s` and `long_full=0.892 s`; reverted.
- Eager radix stale-request cleanup before every extend. The profiled run
  `/tmp/sglang_current_radix_eager_cleanup_36139_20260510.jsonl` improved
  delayed short (`1.900 s`) and preserved `long_full` (`0.773 s`), but regressed
  `short_hit2=0.780 s` and `long_partial=2.557 s` versus baseline; reverted.
- Full-state eval for all single-request radix fallback decode
  (`_RADIX_FULL_STATE_DECODE_MIN_TOKENS=0`). The run
  `/tmp/sglang_current_radix_fullstate_all_decode_38349_20260510.jsonl`
  improved delayed short (`1.761 s`) and long full (`0.764 s`), but regressed
  `short_hit1=0.819 s` and `long_partial=2.500 s`, while `short_hit2=0.783 s`
  remained behind baseline. Keep the retained `1024`-token threshold.
- Active short-prefix contiguous-cache reuse. The ungated run
  `/tmp/sglang_current_radix_active_prefix_reuse_38353_20260510.jsonl`
  improved `long_full=0.743 s` but regressed `long_partial=2.741 s`. Adding a
  small-suffix gate recovered `long_partial` to `2.442 s` and improved
  delayed short to `1.763 s` in
  `/tmp/sglang_current_radix_active_prefix_small_suffix_33031_20260510.jsonl`,
  but `short_hit2=0.736 s` and `long_partial=2.442 s` still missed the
  same-window baseline (`0.627 s` and `2.420 s`). The no-profile confirmation
  `/tmp/sglang_current_radix_active_prefix_small_suffix_noprofile_40127_20260510.jsonl`
  also left `short_hit2=0.733 s` and `long_partial=2.476 s` behind; reverted.
- Raw-attention fallback for radix single-request contiguous decode/prefill.
  `/tmp/sglang_current_radix_raw_fallback_active_prefix_profile_41598_20260510.jsonl`
  regressed `short_hit2=0.804 s` and `long_partial=2.649 s`; reverted.
- Default fp16 page-size-1 radix side-store. This enabled the existing fp16 p1
  paged decode route but badly regressed every row in
  `/tmp/sglang_current_radix_p1_fp16_active_prefix_profile_49201_20260510.jsonl`
  (`short_hit1/2=2.925/2.880 s`, `long_partial=5.103 s`); reverted.
- bf16 generic paged-prefill support. A direct small-shape reference test passed,
  but the serving probe
  `/tmp/sglang_current_radix_bf16_prefill_active_prefix_profile_46506_20260510.jsonl`
  timed out entering the long-partial row after completing only
  `short_first/hit1/hit2=1.199/0.723/0.737 s`. The source and rebuilt metallib
  were reverted.

## Raw Dense h128 Decode Audit: 2026-05-10

New harness output:

```text
/tmp/sglang_raw_micro_dense_h128_20260510.md
command: bench_metal_attention_micro.py --dtype float16 --num-heads 16 --num-kv-heads 8 --head-dim 128 --block-size 16 --warmups 3 --iters 7 --prefill-iters 3 --decode-batches 1 4 8 --decode-seq-lens 288 2162 --prefill-batches 1 4 --prefill-seq-lens 256 1024 --prefill-prefix-lens 256
```

The new lazy dense h128 decode kernel is correct and benchmarkable, but it does
not clear the direct raw Metal gate. It is retained as a microbench variant, not
routed as the serving default.

| Decode row | MLX fast dense ms | Best dense h128 Metal ms | Status |
|---|---:|---:|---|
| B1/S288 | 0.257 | 0.264 | behind |
| B1/S2162 | 0.435 | 0.507 | behind |
| B4/S288 | 0.529 | 0.553 | behind |
| B4/S2162 | 0.897 | 1.119 | behind |
| B8/S288 | 0.542 | 0.700 | behind |
| B8/S2162 | 1.734 | 1.677 | ahead |

The same run still confirms the radix/paged-layout value of the Metal path:
shuffled radix lazy decode is ahead of gathered radix MLX fast on all B1/B4/B8
S288 and S2162 rows. Short prefill remains mixed: Metal wins long S1024 rows
but not the B4/S256 and prefix256/S256 rows.

Rejected in this continuation:

- Dense radix startup prefill warmup for the 434-token cold row. It moved
  `short_first` near baseline in one run (`1.182 s`) but regressed
  `long_partial` (`2.750 s`) and delayed short (`1.918 s`) versus the accepted
  current median, so it was reverted.
- Making the idle paged-prefill guard default-on. The opt-in profiled run
  measured delayed short at `1.828 s` with `idle_paged=True`, but the no-env
  default-on confirmation regressed delayed short to `2.869 s` with
  `idle_paged=True`. Keep this route behind
  `SGLANG_MLX_ENABLE_IDLE_PAGED_PREFILL=1`.

## Raw Dense GQA2 Decode Probe: 2026-05-10

New harness output:

```text
/tmp/sglang_raw_micro_dense_gqa2_20260510.md
command: bench_metal_attention_micro.py --dtype float16 --num-heads 16 --num-kv-heads 8 --head-dim 128 --block-size 16 --warmups 3 --iters 9 --prefill-iters 3 --decode-batches 1 4 8 --decode-seq-lens 288 2162 --prefill-batches 1 4 --prefill-seq-lens 256 1024 --prefill-prefix-lens 256
```

This probe computes the two query heads that share one KV head in a single
threadgroup for dense contiguous decode. It is correct and benchmarkable, but
not accepted as a runtime route. The temporary direct probe looked promising on
B1/S288 and B4/S2162, but the retained harness run did not hold the win:

| Decode row | MLX fast dense ms | Best dense GQA2 Metal ms | Status |
|---|---:|---:|---|
| B1/S288 | 0.266 | 0.359 | behind |
| B1/S2162 | 0.489 | 0.561 | behind |
| B4/S288 | 0.427 | 0.637 | behind |
| B4/S2162 | 0.999 | 1.249 | behind |
| B8/S288 | 0.545 | 0.750 | behind |
| B8/S2162 | 1.999 | 2.022 | behind |

Decision: keep dense GQA2 as a documented benchmark variant only. It does not
change the raw direct-kernel gate; dense MLX remains faster on the contiguous
decode matrix in the retained run.

## p1 Lazy Decode and Same-Window Guard: 2026-05-10

New detailed artifact:

```text
sgl-kernel/csrc/metal/METAL_MICROBENCH_RESULTS_P1_LAZY_DECODE.md
```

The added lazy p1 paged decode kernel improves the direct p1/radix block-table
path against p1 gather+MLX on the broad decode matrix:

| Row | p1 lazy radix ms | p1 gather+MLX radix ms | Status |
|---|---:|---:|---|
| B1/S2162 | 0.4783 | 1.0557 | ahead |
| B4/S2162 | 1.0888 | 4.7432 | ahead |
| B8/S2162 | 1.5313 | 12.3570 | ahead |

Short rows are thresholded: B1/S16 stays on gather+MLX, while B1 contexts above
16 tokens and all measured B4/B8 rows use the p1 lazy route when a paged decode
context is active.

Latest same-window no-radix guard after this change:

```text
current:  /tmp/sglang_current_no_radix_p1_lazy_guard_20260510.jsonl
baseline: /tmp/sglang_baseline_no_radix_p1_lazy_guard_20260510.jsonl
command:  python -m sglang.bench_one_batch --disable-radix-cache --batch-size 1 4 8 --input-len 256 --output-len 32 --warmups 1
```

| Path | B1 tok/s | B4 tok/s | B8 tok/s |
|---|---:|---:|---:|
| current | 217.60 | 343.21 | 492.57 |
| c4903e67 baseline | 213.28 | 329.58 | 465.45 |

Latest radix E2E probe after the p1 lazy decode addition:

```text
current: /tmp/sglang_current_radix_p1_lazy_decode_43940_20260510.jsonl
```

| Probe | Latency s | cached_tokens |
|---|---:|---:|
| short_first | 1.273 | 0 |
| short_hit1 | 0.817 | 433 |
| short_hit2 | 0.831 | 433 |
| long_partial | 3.020 | 434 |
| long_full | 0.815 | 2161 |
| delayed_short | 2.168 | 433 |

Decision: retain the p1 lazy direct-paged decode improvement and the no-radix
guard. The default radix serving path remains hybrid/contiguous-cache compute,
so this does not close the direct-paged-kernel or full radix serving gate.

## bf16 p1 Lazy Decode and Radix Serving Guard: 2026-05-10

New detailed artifact:

```text
sgl-kernel/csrc/metal/METAL_MICROBENCH_RESULTS_P1_BF16_LAZY_DECODE.md
```

The scalar p1 lazy decode kernel adds `bfloat16` coverage for the Qwen serving
dtype. Raw direct p1/radix decode is much faster than p1 gather+MLX on the
long rows:

| Row | bf16 p1 lazy radix ms | bf16 p1 gather+MLX radix ms | Status |
|---|---:|---:|---|
| B1/S2162 | 0.5074 | 1.0749 | ahead |
| B4/S2162 | 0.8785 | 5.0674 | ahead |
| B8/S2162 | 1.6477 | 10.0106 | ahead |

Serving did not accept the raw win as a default route. Forcing bf16 p1 paged
decode in radix serving entered the `radix_paged` route but regressed E2E:

```text
forced paged: /tmp/sglang_current_radix_bf16_p1_paged_decode_43983_20260510.jsonl
default guard: /tmp/sglang_current_radix_bf16_guard_default_43984_20260510.jsonl
```

| Probe | Forced paged s | Default guarded hybrid s |
|---|---:|---:|
| short_first | 1.369 | 1.189 |
| short_hit1 | 0.916 | 0.743 |
| short_hit2 | 1.020 | 0.835 |
| long_partial | 2.830 | 2.554 |
| long_full | 0.923 | 0.765 |
| delayed_short | 2.046 | 1.918 |

The accepted runtime therefore keeps bf16 p1 paged decode behind
`SGLANG_MLX_ENABLE_BF16_PAGED_DECODE=1`; the default radix path logs
`MLX decode timing radix`, not `radix_paged`.

Rejected follow-ups in this continuation:

- Reusing the p1 decode kernel for one-token radix prefix-hit prefill. Normal
  short-hit prefill eval dropped to roughly `41-48 ms`, but long partial
  regressed to `2.864 s` and delayed short regressed to `2.197 s`.
- Materializing p1 prefixes into a contiguous cache before the one-token
  prefix-hit model call. After bookkeeping fixes it remained coherent, but
  delayed short regressed to `5.203 s`.
- Switching bf16 p1 decode from 256 to 128 threads. A scratch kernel helped
  selected batch-heavy rows, but hurt the serving-relevant B1/S434 and
  B1/S2162 rows, so it is not a default-route candidate.

Focused correctness after reverting those rejected serving experiments:

```text
PYTHONPATH=sgl-kernel/python:python ./sglang-mlx/bin/python -m pytest \
  sgl-kernel/tests/test_metal.py \
  test/registered/unit/hardware_backend/mlx/kv_cache/test_attention_wrapper.py \
  test/registered/unit/hardware_backend/mlx/kv_cache/test_paged_cache.py \
  test/registered/unit/hardware_backend/mlx/test_model_runner.py -q

156 passed, 6 warnings
```

## Serving Completion Refresh: 2026-05-11

This refresh supersedes the earlier 2026-05-10 "radix serving gate open" note
for the serving-baseline objective. The runtime remains a hybrid MLX/Metal path:
direct paged Metal kernels are still not a universal replacement for dense
`mx.fast.scaled_dot_product_attention` on every raw microbenchmark row.

Final focused correctness:

```text
PYTHONPATH=sgl-kernel/python:python ./sglang-mlx/bin/python -m pytest \
  sgl-kernel/tests/test_metal.py \
  test/registered/unit/hardware_backend/mlx/kv_cache/test_attention_wrapper.py \
  test/registered/unit/hardware_backend/mlx/kv_cache/test_paged_cache.py \
  test/registered/unit/hardware_backend/mlx/test_model_runner.py -q

156 passed, 6 warnings
```

No-radix serving evidence:

```text
final current full sweep: /tmp/sglang_current_no_radix_final_audit_20260511.jsonl
same-window B4/B8 baseline: /tmp/sglang_baseline_no_radix_samewindow_012839_20260511.jsonl
paired B1 current repeats: /tmp/sglang_current_no_radix_b1_final_reps_{1,2,3}_20260511.jsonl
paired B1 baseline repeats: /tmp/sglang_baseline_no_radix_b1_currentwindow_reps_{1,2,3}_20260511.jsonl
```

| Row | Current tok/s | c4903e67 baseline tok/s | Status |
|---|---:|---:|---|
| B1 paired median | 219.26 | 218.12 | ahead by 0.5% |
| B4 final audit | 345.65 | 329.47 | ahead by 4.9% |
| B8 final audit | 492.00 | 460.29 | ahead by 6.9% |

The B1 row is intentionally reported from paired current-window repeats because
the single final full sweep measured `209.48 tok/s`, while the paired baseline
also moved down in the same load window. The paired medians clear the previous
MLX baseline without relying on the older saved B1 snapshot.

Radix serving evidence:

```text
baseline:
  /tmp/sglang_baseline_radix_samewindow_b1smallfull_45978_20260511.jsonl
  /tmp/sglang_baseline_radix_rerun_44069_015602_20260511.jsonl
accepted current:
  /tmp/sglang_current_radix_default_b1ctxfast_norepatch_noprofile_41843_20260511.jsonl
  /tmp/sglang_current_radix_final384_41753_014051_20260511.jsonl
  /tmp/sglang_current_radix_final384_rerun_44212_014155_20260511.jsonl
  /tmp/sglang_current_radix_final384_rerun2_47585_014856_20260511.jsonl
  /tmp/sglang_current_radix_profile_36867_20260511.jsonl
  /tmp/sglang_current_radix_revertclean_36604_20260511.jsonl
  /tmp/sglang_current_radix_revertclean2_33598_20260511.jsonl
```

| Probe | Current median s | c4903e67 baseline median s | Status |
|---|---:|---:|---|
| short_first | 1.133 | 1.153 | ahead by 1.8% |
| short_hit1 | 0.686 | 2.227 | ahead by 69.2% |
| short_hit2 | 0.629 | 0.635 | ahead by 0.9% |
| long_partial | 2.389 | 2.601 | ahead by 8.1% |
| long_full | 0.771 | 0.862 | ahead by 10.6% |
| delayed_short | 1.861 | 1.992 | ahead by 6.6% |

Rejected in this final continuation:

- Forcing compact eval for radix B1 batched-wrapper decode by setting
  `_RADIX_BATCHED_WRAPPER_B1_FULL_STATE_MAX_CAPACITY=0`. The run
  `/tmp/sglang_current_radix_compacteval_51583_20260511.jsonl` did not provide
  a useful `short_hit2` win and regressed delayed short to `2.570 s`; reverted.
- Evaluating only the newly written KV slice for the short B1 radix wrapper
  decode. `/tmp/sglang_current_radix_b1sliceeval_33150_20260511.jsonl`
  measured `short_hit2=0.653 s`, still behind baseline; reverted.
- Tightening the right-sized short-prefix materialized cache to a 464-token
  capacity. `/tmp/sglang_current_radix_rightsize464_43084_20260511.jsonl`
  regressed delayed short to `5.615 s` and `long_full` to `0.867 s`; reverted.

Decision: the serving-native no-radix and radix-cache paths now beat the
previous MLX baseline on the recorded paired workloads. The remaining open item
is narrower: direct raw Metal kernels still do not beat dense MLX fast
attention on every contiguous decode/prefill microbenchmark row, so the
accepted production route stays hybrid.
