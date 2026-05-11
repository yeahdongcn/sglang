# Metal FlashAttention Completion Audit

Generated: `2026-05-09`

Superseded by `METAL_COMPLETION_AUDIT_2026_05_10.md` for the latest
no-radix/radix serving evidence and raw dense h128 decode audit.

## Agent Retrieval Note

- Status: superseded audit snapshot. Keep it for provenance and for the
  rejected-direction list, not for the final serving answer.
- Use when: reconstructing why the 2026-05-09 state was still incomplete or
  checking early radix/no-radix serving probes.
- Do not use as: the current completion decision. The final serving decision
  lives in `METAL_COMPLETION_AUDIT_2026_05_10.md` and
  `METAL_FINAL_PERF_RESULTS_2026_05_11.md`.
- Read next: `METAL_DOC_INDEX.md`,
  `METAL_COMPLETION_AUDIT_2026_05_10.md`,
  `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md`.
- Search tags: `superseded-audit`, `partially-met`, `rejected`, `radix`,
  `no-radix`, `raw-kernel-gap`.

## Objective

Make the current MLX/Metal FlashAttention work at commit
`88eb26ca540572507a47b039be43093b75a4f838` plus local changes both correct and
faster than the previous MLX baseline at
`c4903e67bf50f50286406273dedc52190fe9011f`.

## Prompt-to-Artifact Checklist

| Requirement | Evidence | Status |
|---|---|---|
| Use the current worktree at `88eb26ca...` plus local changes | `MLX_METAL_REGRESSION_INVESTIGATION.md` tracks the current target; `git status` remains dirty with local Metal/MLX changes preserved | In progress |
| Compare against previous MLX baseline `c4903e67...` | `/tmp/sglang-c490` baseline runs are recorded in `METAL_SERVING_BENCH_RESULTS_2026_05_09.md`; latest no-radix same-window rows include baseline and current B1/B4/B8 | Partially met |
| Preserve correctness | `sgl-kernel/tests/test_metal.py` now includes fused decode cache-write and mixed-dtype all-layer scatter coverage; focused Metal tests and MLX cache/model tests pass after the native all-layer scatter dtype fix and batched-contiguous no-radix route | Met for covered unit/kernel paths |
| Coherent E2E generation | Current radix-enabled serving probes on ports 43465 and 43471 stayed coherent for short hits, long partial hit, and long full hit after the dtype fix | Met for probed cases |
| Improve no-radix path | The accepted wrapper-based contiguous-cache route is variance-sensitive but ahead on the latest B1/B8 full sweep and isolated B4 rerun; the full B4 sweep remains slightly below the nearest same-window baseline | Partially met |
| Improve radix-cache path | Page-size-1 radix serving plus deferred decode sync and the idle-gap paged short-hit guard improves short hits, long full hit, and delayed short hit; cold first and long partial remain behind the refreshed p1 baseline | Partially met |
| Raw Metal decode beats `mx.fast.scaled_dot_product_attention` for target decode matrix | `METAL_MICROBENCH_RESULTS_DECODE_LAZY_MLX_KERNEL.md` shows radix-style direct decode wins against gather+MLX, but contiguous no-radix rows remain mixed vs dense MLX fast | Open |
| Reproducible microbench harness covers lazy direct paged decode | `bench_metal_attention_micro.py` now emits accepted lazy, register-score lazy, and native fused-KV h128/block16 rows for contiguous and shuffled radix decode layouts | Met |
| Raw Metal prefill beats MLX fast for target prefill/radix-prefix matrix | Steel/paged prefill route wins selected large radix-prefix cases, but short prefix hits remain on dense MLX and raw prefill is not a universal replacement | Open |
| Remove hybrid/contiguous guard | Plan explicitly says to keep the guard until paged kernels and serving integration beat refreshed MLX baseline | Open |
| Document rejected kernel directions | Rejection docs exist for q-shared, two-pass long decode, contiguous block specialization, dense lazy decode, paged sdpa-vector, and paged two-pass vector decode | Met |
| No per-call old environment flag dependency | Runtime no longer depends on old per-call `SGLANG_MLX_USE_METAL_ATTENTION` gating; backend startup checks required Metal APIs | Met |

## Current Evidence Summary

Correctness evidence:

```text
PYTHONPATH=sgl-kernel/python:python ./sglang-mlx/bin/python -m pytest sgl-kernel/tests/test_metal.py -q
37 passed

PYTHONPATH=sgl-kernel/python:python ./sglang-mlx/bin/python -m pytest \
  test/registered/unit/hardware_backend/mlx/kv_cache/test_attention_wrapper.py \
  test/registered/unit/hardware_backend/mlx/kv_cache/test_paged_cache.py \
  test/registered/unit/hardware_backend/mlx/test_model_runner.py -q
85 passed

PYTHONPATH=sgl-kernel/python:python ./sglang-mlx/bin/python -m unittest \
  test.registered.unit.hardware_backend.mlx.test_model_runner \
  test.registered.unit.hardware_backend.mlx.kv_cache.test_attention_wrapper \
  test.registered.unit.hardware_backend.mlx.kv_cache.test_paged_cache \
  test.registered.unit.hardware_backend.mlx.kv_cache.test_paged_context
88 passed
```

Serving evidence:

- No-radix same-window full sweep after the native scatter dtype fix:
  - Baseline B1/B4/B8 total tok/s: `160.66 / 255.71 / 360.24`
  - Current B1/B4/B8 total tok/s: `161.27 / 258.43 / 354.95`
- No-radix isolated B8 rerun after the dtype fix:
  - Baseline total tok/s: `361.71`
  - Current total tok/s: `363.97`
- Current radix-enabled probe on port 43465 after the dtype fix:
  - Short first/hit1/hit2: `2.58 / 1.17 / 0.78 s`, coherent
  - Long partial/full hit: `2.88 / 0.95 s`, coherent
  - Short hit after 10 s idle: `3.09 s`, coherent but still a variance watch
- Final no-radix rerun after the equal-length batched contiguous-cache route:
  - Baseline B1/B4/B8 total tok/s: `197.93 / 305.29 / 421.64`
  - Current B1/B4/B8 total tok/s: `184.92 / 375.76 / 496.65`
  - Accepted improvement: B4/B8. Open: B1.
- Same-window radix rerun:
  - Baseline `c4903e67` short first/hit1/hit2:
    `1.40 / 2.55 / 0.65 s`
  - Current default p16 short first/hit1/hit2:
    `1.79 / 1.28 / 1.17 s`
  - Baseline long partial/full and delayed short:
    `2.41 / 0.81 / 1.77 s`
  - Current default p16 long partial/full and delayed short:
    `3.16 / 1.01 / 2.98 s`
  - Current `--page-size 1` restores cached-token counts but still does not
    broadly beat baseline.
- Post-resume accepted serving rerun:
  - No-radix wrapper route full sweep B1/B4/B8 totals:
    `189.37 / 298.17 / 413.02 tok/s`; isolated B4 rerun: `304.27 tok/s`.
  - Latest accepted radix p1 probe:
    short first/hit1/hit2 `1.71 / 0.93 / 0.68 s`, long partial/full
    `2.77 / 0.89 s`, delayed short `1.31 s`, all coherent.
  - The idle-gap paged short-hit guard reduced the delayed row from the
    pre-guard `4.21 s` outlier to `1.31 s`.

Raw-kernel evidence:

- Accepted lazy paged decode:
  - S2162 shuffled radix B1/B4/B8: `0.4773 / 1.2585 / 2.7048 ms`
  - Gather+MLX radix B1/B4/B8: `1.3844 / 7.1466 / 14.0863 ms`
- Harness-generated rerun:
  - `METAL_MICROBENCH_RESULTS_DECODE_LAZY_HARNESS_RERUN.md` is generated by
    `bench_metal_attention_micro.py` and includes both lazy contiguous and lazy
    shuffled-radix rows.
- B1 short-context threshold rerun:
  - `METAL_MICROBENCH_RESULTS_B1_SHORT_DECODE_THRESHOLD_RERUN.md` keeps B1/S16
    on gather+MLX (`0.2873 ms` versus `0.3097 ms` lazy radix), but routes
    B1/S24+ to lazy direct decode (`0.2754 ms` at S24 and `0.2786 ms` at S32).
- Fused-KV decode experiment:
  - `METAL_MICROBENCH_RESULTS_FUSED_KV_DECODE.md` shows the native fused
    scatter+decode kernel is correct but slower than accepted lazy decode on
    almost every tested contiguous and shuffled radix row, so it is not routed.
- All-layer native scatter sync:
  - `METAL_MICROBENCH_RESULTS_RADIX_SYNC_METAL_SCATTER.md` shows 28-layer
    radix side-store sync improving from `1.4097` to `0.8832 ms` for one token,
    from `1.4123` to `1.1346 ms` for four tokens, and from `11.1655` to
    `9.5193 ms` for 256 tokens versus the previous per-layer list assignment.
  - The same result file now includes a serving-shaped `bfloat16 -> float16`
    spot check after the dtype fix: one-token sync improves from `1.3566` to
    `1.2174 ms`, four-token sync from `1.6968` to `1.5727 ms`, and 256-token
    sync from `14.6236` to `12.6477 ms`.
- Remaining raw contiguous gap:
  - Dense MLX remains competitive or faster on multiple contiguous rows.
  - Several Metal alternatives were correct but rejected because they did not
    clear the full target matrix.

## Rejected Directions

| Direction | Evidence |
|---|---|
| q-shared/GQA2 decode dispatch | `METAL_MICROBENCH_RESULTS_DECODE_Q_SHARED.md`, GQA2 rerun docs |
| Two-pass long paged decode | `METAL_MICROBENCH_RESULTS_DECODE_2PASS.md` |
| Contiguous block-table shortcut and reshape-to-MLX | `METAL_MICROBENCH_RESULTS_CONTIGUOUS_DECODE_SPECIALIZATION.md` |
| Dense-cache lazy Metal decode | `METAL_MICROBENCH_RESULTS_DENSE_DECODE_LAZY.md` |
| Paged MLX `sdpa_vector` adaptation | `METAL_MICROBENCH_RESULTS_PAGED_SDPA_VECTOR_DECODE.md` |
| Paged two-pass vector split | `METAL_MICROBENCH_RESULTS_PAGED_2PASS_VECTOR_DECODE.md` |
| Register-score lazy decode as universal replacement | `METAL_MICROBENCH_RESULTS_DECODE_REGSCORE_SWEEP.md`, `METAL_MICROBENCH_RESULTS_DECODE_FAST_EXP_COMPARE.md` |
| Native fused scatter+decode h128/block16 decode | `METAL_MICROBENCH_RESULTS_FUSED_KV_DECODE.md` |
| Idle-gap materialized-prefix heuristic for short radix hits | `METAL_SERVING_BENCH_RESULTS_2026_05_09.md` |
| Pure-MLX block-size-1 side-store/dtype route | `METAL_SERVING_BENCH_RESULTS_2026_05_09.md` |
| Decoupled scheduler p1 plus MLX side-store block16 route | `METAL_SERVING_BENCH_RESULTS_2026_05_09.md` |
| Prompt-side side-store sync deferral | `METAL_SERVING_BENCH_RESULTS_2026_05_09.md` |
| Unchecked full-block pool-backed gather | `METAL_SERVING_BENCH_RESULTS_2026_05_09.md` |
| Paged-precedence and pool-backed long-partial routes | `METAL_SERVING_BENCH_RESULTS_2026_05_09.md` |

## Audit Decision

The investigation is still open.

The current branch has restored correctness in the covered tests and improved
both serving paths, but the objective is not complete. The no-radix guard is
variance-sensitive rather than stably ahead on every row, and radix serving is
still behind baseline for cold first request and long partial-prefix hit. Direct
raw Metal decode/prefill also does not beat the previous MLX route across the
full targeted matrix. The hybrid contiguous/no-radix guard must remain until a
future kernel clears that gate.
