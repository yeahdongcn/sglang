# MLX/Metal Correctness and Performance Regression Investigation

This file tracks the investigation into the large local performance downgrade and garbled output observed after the serving-native MLX/Metal paged-attention changes.

## Agent Retrieval Note

- Status: chronological investigation log. It is useful for decision history
  and baseline comparison rules; use newer completion/result files for the
  final accepted serving numbers.
- Use when: reconstructing why correctness and performance were both gates,
  finding baseline/current commit identities, or tracing rejected fixes.
- Do not use as: a short answer to the final performance result. Open
  `METAL_FINAL_PERF_RESULTS_2026_05_11.md` first for that.
- Read next: `METAL_DOC_INDEX.md`,
  `METAL_COMPLETION_AUDIT_2026_05_10.md`,
  `FULL_FLASH_ATTENTION_METAL_PLAN.md`.
- Search tags: `investigation-log`, `correctness`, `performance`,
  `baseline-commit`, `rejected`, `serving-regression`.

## Problem statement

Local testing on branch `xd/test_flash_attn` found:

- Output correctness regression: generated text can become garbled on long prompts.
- Performance regression: throughput/latency is significantly worse than the previous MLX path, and the downgrade remained large after a macOS restart and retest on 2026-05-09.
- The final current changes have been compacted into commit `5b674a5a7d71f386f074396dfab3974ec93055fc` on `xd/test_flash_attn`.
- Older audit notes may describe the same target as `88eb26ca540572507a47b039be43093b75a4f838` plus local changes before the squash.
- The original comparison baseline is commit `c4903e67bf50f50286406273dedc52190fe9011f`.

## Goal

Make the current MLX/Metal implementation both correct and faster than the previous commit baseline on the targeted Apple Silicon serving workloads.

## Baseline comparison

Use `c4903e67bf50f50286406273dedc52190fe9011f` as the original MLX-path performance and correctness baseline. Use `5b674a5a7d71f386f074396dfab3974ec93055fc` as the current compacted target state.

Suggested comparison flow:

1. Preserve the active working tree; do not discard local edits.
2. Run correctness smoke tests and benchmark commands on `c4903e67bf50f50286406273dedc52190fe9011f` from a safe worktree if baseline numbers are needed.
3. Return to the active branch/worktree at `5b674a5a7d71f386f074396dfab3974ec93055fc`.
4. Run the same correctness smoke tests and benchmark commands.
5. Investigate gaps until current branch output is correct and performance exceeds the baseline MLX path.

Do not discard uncommitted user work while switching commits; use a safe worktree or stash only with explicit user approval if needed.

## Current long-prompt observations

2026-05-09 update: the user restarted the Mac and confirmed the end-to-end performance downgrade is still huge. Performance must be improved end-to-end, not only in kernel microbenchmarks, and correctness remains an acceptance criterion.

Direct MLX-LM greedy generation for the long 2162-token prompt is coherent, so base Qwen3/MLX-LM behavior is not the root cause:

```text
prompt_tokens 2162
tokens [81917, 3170, 281, 3279, 6529, 53588, 1184, 9446, 44817, 323, 2504, 12632, 304, 264, 4128, 1614, 3538, 13, 81917, 3170]
text_repr ' Explain why paged attention caches need slot mappings and block tables in a language model server. Explain why'
elapsed 6.184377999976277
```

Earlier SGLang MLX serving for the same prompt on the current path was corrupt and slow:

```text
prompt_tokens=2162
cached_tokens=0
e2e_latency≈24.38s
text_repr=' Explainawarepreneurtings지IItracts }\ammediction-boldposure whomiageодерж大学"><# Analyticscribed\xa0'
```

Repeated same-prompt radix-hit probes on that server were also corrupt, including token prefixes beginning `[55182, 37257, 38733]` and text like `ceriesutenantussions`, with cached_tokens around 2161.

2026-05-09 later update: after the no-prefix dense prefill, explicit decode-slot, and mixed-mode routing fixes, the 2162-token long no-prefix prompt produced coherent output again, but remained slow:

```text
elapsed 25.083169707999332
prompt_tokens 2162
cached_tokens 0
completion_tokens 20
text_repr ' Explain why paged attention caches need slot mappings and block tables in a language model server. Explain why'
server_log /tmp/sglang_mlx_long_probe_43441.log
```

The most recent correctness risk was mixed-mode routing: `ForwardMode.MIXED` can include one-token chunked-prefill/extend continuations as well as running decode requests, so `seq_len == 1` is not sufficient to route an existing MLX request through decode. `MlxTpModelWorker` now uses `ModelWorkerBatch.decoding_reqs` as the decode discriminator and passes scheduler-provided decode slots to `decode_batch`; existing requests not in `decoding_reqs` continue through `extend`, even for one-token chunks.

## 2026-05-09 benchmark finding

The new microbenchmarks separate raw attention kernels from scheduler/cache
bookkeeping. Results are recorded in:

- `METAL_MICROBENCH_RESULTS_SERVING.md`: original raw kernel comparison.
- `METAL_MICROBENCH_RESULTS_4096_SCORE_CACHE.md`: after raising the decode score-cache limit.
- `METAL_MICROBENCH_RESULTS_GATHERED_DECODE.md`: after adding the gathered-paged MLX fast decode candidate.
- `METAL_MICROBENCH_RESULTS_ONLINE_DECODE_VEC_CACHED_Q.md`: after adding the
  guarded online-softmax/vectorized h128 paged decode kernel.
- `METAL_MICROBENCH_RESULTS_ONLINE_PREFILL_PREFIX.md`: after adding the
  guarded online-softmax h128 packed-varlen and paged-prefill kernels.
- `METAL_MICROBENCH_RESULTS_ONLINE_PREFILL_PREFIX_RADIX.md`: same raw kernel
  benchmark with an explicit 256-token radix-prefix prefill case.
- `METAL_MICROBENCH_RESULTS_ONLINE_512_THREADS_QUICK.md`: rejected 512-thread
  online group experiment.
- `METAL_MICROBENCH_RESULTS_STEEL_PREFIX_FAST_EXP.md`: uniform no-prefix and
  radix-prefix prefill routed through the MLX Steel tiled attention bridge, plus
  the rejected fast-exp decode check.
- `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md`: current raw-kernel comparison
  with both contiguous paged decode and shuffled radix-style block-table decode
  rows.
- `METAL_MICROBENCH_RESULTS_DECODE_NATIVE_LOWER_BOUND.md`: direct native
  preallocated-output paged decode lower bound versus the public wrapper and
  dense MLX fast attention.
- `METAL_MICROBENCH_RESULTS_ONLINE_DECODE_192_THREADS.md`: rejected 192-thread
  online decode sweep between the older rejected 128-thread setting and the
  current 256-thread setting.
- `METAL_SERVING_BENCH_RESULTS_2026_05_09.md`: refreshed baseline/current `bench_one_batch` runs.

Raw kernel conclusion:

- Original paged Metal decode was much slower than `mx.fast.scaled_dot_product_attention`: for B1/S2162, `metal_paged` was 29.5721 ms vs `mlx_fast_dense` 0.6409 ms; for B8/S2162, 74.4134 ms vs 2.0070 ms.
- The first concrete kernel bug was the 1024-token score cache. Long decode recomputed QK scores inside the output-dimension loop. Raising the decode score cache to 4096 cut B1/S2162 paged decode to roughly 2-3 ms and B8/S2162 to roughly 12 ms, but it still trails MLX fast attention.
- A refreshed raw microbenchmark after the serving fixes still shows the direct kernels behind MLX fast attention: B1/S2162 decode `metal_paged` 2.1242 ms vs `mlx_fast_dense` 0.4278 ms, and B8/S2162 decode 12.0947 ms vs 1.6275 ms.
- A threadgroup-size sweep found that 64-thread dispatch regressed decode badly.
  The C++ and Metal dispatch constants are now consistently set to 128 threads.
  The final 128-thread raw microbench is recorded in
  `METAL_MICROBENCH_RESULTS_THREADGROUP_128.md`: B1/S2162 decode is
  `metal_paged` 2.2729 ms vs `mlx_fast_dense` 0.4245 ms, B4/S2162 is
  5.9141 ms vs 0.8849 ms, and B8/S2162 is 11.5096 ms vs 1.8197 ms.
  This is a small broad improvement over the failed 64-thread experiment, but
  it does not change the raw-kernel conclusion.
- The next raw decode improvement adds a guarded online-softmax paged decode
  kernel for the common Qwen shape (`head_dim=128`, `block_size=16`) plus a
  float16/contiguous half4-vectorized route that caches the query vectors in
  registers across KV blocks. Results are recorded in
  `METAL_MICROBENCH_RESULTS_ONLINE_DECODE_VEC_CACHED_Q.md`: B1/S2162
  `metal_paged` is 0.6032 ms vs `mlx_fast_dense` 0.4363 ms, B4/S2162 is
  1.1472 ms vs 0.8923 ms, and B8/S2162 is 1.7833 ms vs 1.4936 ms. This is no
  longer the severe raw decode cliff, and it beats gathered-paged MLX on the
  paged layout, but it still does not beat dense `mx.fast.scaled_dot_product_attention`.
- A follow-up online-threadgroup experiment at 128 threads is recorded in
  `METAL_MICROBENCH_RESULTS_ONLINE_DECODE_128_THREADS.md`. It was rejected:
  B1/S2162 regressed to 0.7102 ms and B8/S2162 regressed to 2.9780 ms, so the
  online decode path remains at 256 threads.
- The latest decode rerun with explicit radix-style block-table rows is in
  `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md`. Dense MLX is still the synthetic
  ceiling for long decode: B1/B4/B8 S2162 `metal_paged` is
  0.6654/1.2013/1.7513 ms versus dense MLX fast 0.4673/1.0269/1.6542 ms.
  Against the actual gathered-paged MLX radix-style path, direct Metal paged
  decode is ahead by a large margin: shuffled radix B1/B4/B8 S2162 is
  0.6572/1.0986/1.8210 ms versus `mlx_fast_radix_gathered_paged`
  1.7799/3.8598/6.9880 ms. The MLX radix decode wrapper now routes the common
  fp16/head_dim=128/block_size=16 decode shape through `decode_attention_paged`
  instead of gathering paged K/V and then calling dense MLX fast attention.
- A 192-thread online decode sweep was rejected: B4/B8 S2162 contiguous paged
  decode regressed to 1.1775/1.9805 ms versus the current 256-thread
  1.0984/1.7368 ms in the same benchmark family, so 256 threads remains the
  online decode dispatch setting.
- A native dispatch lower-bound check in
  `METAL_MICROBENCH_RESULTS_DECODE_NATIVE_LOWER_BOUND.md` shows the h128/b16
  shader is close to dense MLX for B4/B8 long decode when the output array is
  preallocated and the native extension is called directly: B4/S2162 is
  `1.0442 ms` versus dense MLX `1.0072 ms`, and B8/S2162 is `2.0654 ms`
  versus dense MLX `2.0216 ms`. The public wrapper remains slower because it
  must materialize `q`, `block_tables`, `context_lens`, and `out` before native
  validation; reducing that eval set was tested and failed correctness.
- Native Metal prefill/varlen remains far behind MLX fast attention. In the gathered-decode run, B1/S1024 prefill was 177.7674 ms (`metal_prefill_paged_no_prefix`) or 161.4870 ms (`metal_flash_varlen`) vs 5.2793 ms for `mlx_fast_dense`.
- The refreshed raw prefill check was still far behind: B1/S1024 prefill was 221.9277 ms (`metal_prefill_paged_no_prefix`) or 188.9393 ms (`metal_flash_varlen`) vs 8.1705 ms for `mlx_fast_dense`.
- The final 128-thread prefill check remains far behind: B1/S1024 is
  182.8721 ms (`metal_prefill_paged_no_prefix`) or 167.1255 ms
  (`metal_flash_varlen`) vs 5.4948 ms for `mlx_fast_dense`; B4/S1024 is
  659.8294 ms or 653.0847 ms vs 20.8479 ms.
- Guarded online-softmax h128 prefill kernels remove most of that raw prefill
  cliff but still miss the MLX fast target. In
  `METAL_MICROBENCH_RESULTS_ONLINE_PREFILL_PREFIX.md`, B1/S1024 no-prefix
  paged prefill is 30.6368 ms and packed varlen is 24.7330 ms versus 5.3609 ms
  for `mlx_fast_dense`; B4/S1024 is 121.5307 ms and 92.2452 ms versus 18.7207 ms.
- The radix-prefix raw path is now covered by
  `METAL_MICROBENCH_RESULTS_ONLINE_PREFILL_PREFIX_RADIX.md`. With a 256-token
  prefix and S1024 suffix, B1 paged prefill is 47.3442 ms versus 6.5104 ms for
  dense MLX fast with an equivalent prefix causal mask; B4 is 191.9537 ms
  versus 23.1551 ms.
- Uniform no-prefix prefill is now routed to the MLX Steel tiled attention
  bridge from `prefill_attention_paged`, and uniform radix-prefix paged prefill
  gathers prefix blocks into dense K/V before routing prefix+suffix through the
  same Steel varlen bridge. In `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md`,
  B1/B4 S1024 no-prefix prefill is 3.5220/11.3406 ms versus dense MLX fast
  5.8653/19.6909 ms, and B1/B4 S1024 prefix256 prefill is 5.7009/18.2100 ms
  versus dense MLX fast prefix 6.3710/24.1820 ms. Short S256 prefix rows remain
  mixed and should not be counted as a completed raw-kernel win.
- The new prefill kernels are therefore accepted as a useful intermediate
  improvement and radix-path coverage, not as the final solution. The remaining
  gap is structural: the current prefill kernels still assign one threadgroup to
  one query token and scan visible K/V, while MLX fast uses a tiled attention
  implementation.
- A 512-thread online group quick check was rejected: it regressed B1/B8 long
  decode and both no-prefix and prefix prefill, so the online kernels remain at
  256 threads.
- Metal scatter is small compared with attention, around 0.2-0.5 ms in the microbench.
- The radix side-store sync path now avoids an all-layer `mx.stack` temporary
  and passes per-layer K/V arrays directly to `set_kv_all_layers`. The current
  microbench shows consistent wins for 28-layer sync: 1-token median
  2.0597 -> 1.3882 ms, 4-token 1.7405 -> 1.5058 ms, and 256-token
  8.0237 -> 6.4931 ms.
- A follow-up all-layer native scatter wrapper reduces that sync overhead again
  while preserving the per-layer paged cache representation. In
  `METAL_MICROBENCH_RESULTS_RADIX_SYNC_METAL_SCATTER.md`, the 28-layer
  `metal_scatter_all_layers_list` path is faster than the previous list sync:
  1-token `0.8832 ms` vs `1.4097 ms`, 4-token `1.1346 ms` vs `1.4123 ms`,
  and 256-token `9.5193 ms` vs `11.1655 ms`.
- Two follow-up experiments were rejected. Reusing materialized prefix-hit K/V
  arrays as the active contiguous cache regressed serving badly because decode
  then grew exact-size arrays on the hot path. Calling the Steel varlen bridge
  as a `q_len=1` decode kernel was also slower than dense MLX fast attention
  for B1/B4/B8 S2162.

Serving conclusion:

- Before the no-radix guard, current page-size-16 serving was far below baseline: B1 total 56.05 tok/s, B4 172.89 tok/s, B8 289.77 tok/s.
- Restoring the `disable_radix_cache=True` contiguous-cache path removes the severe `bench_one_batch` regression. Refreshed current no-radix results are now approximately the refreshed baseline: B1 total 257.14 vs 259.88 tok/s, B4 399.39 vs 393.71 tok/s, B8 555.08 vs 558.80 tok/s.
- After disabling unused paged-context dispatch in runner-installed wrappers and restoring the no-compact eval path to the baseline shape, the final no-radix sweep is ahead of same-machine refreshed baseline checks: B1 total 261.91 vs 257.66 tok/s, B4 410.38 vs 308.89 tok/s, B8 567.82 vs 408.21 tok/s.
- The radix-cache path is now a hybrid rather than the raw paged-Metal path: active requests use contiguous MLX caches, and new K/V is synced into the paged side-store for radix visibility. This fixed the 448-token/2162-token probe OOM and produces coherent outputs with cache hits. After batching all-layer K/V sync and recomputing small radix prefixes from the full prompt, the short hit, long partial-hit, and long full-hit probes are all ahead of the refreshed `c4903e67` baseline in this session.
- This is a guardrail, not a raw-kernel completion signal. The paged Metal radix-cache path still needs faster direct paged kernels before the final design can stop using the hybrid contiguous-cache compute path.

2026-05-09 latest serving rerun:

- Same-window no-radix `bench_one_batch --batch-size 1 4 8 --input-len 256 --output-len 32 --warmups 1` is ahead of a fresh `/tmp/sglang-c490` baseline after memory pressure normalized. Current B1/B4/B8 total throughput is 300.01/458.39/627.31 tok/s versus baseline 296.02/450.60/625.42 tok/s.
- The latest radix-enabled strict-idle probe produced coherent outputs and improved again: short first/hit1/hit2 `2.33 / 0.93 / 0.74 s`, long partial hit `2.91 s`, and long full hit `0.94 s`.
- After the no-stack radix sync change, a lower-memory same-window no-radix
  rerun remained ahead of `/tmp/sglang-c490`: B1/B4/B8 total throughput
  236.52/380.94/521.31 tok/s versus baseline 229.27/364.63/485.59 tok/s.
  The radix probe on port 43449 stayed coherent: short first/hit1/hit2
  `2.56 / 1.14 / 0.99 s`, long partial hit `3.42 s`, and long full hit
  `1.13 s`. A short hit after a 10 s idle gap remained an outlier at `4.65 s`.
- After the all-layer native scatter sync change, a current no-radix rerun under
  low-memory conditions was variance-sensitive but still showed no regression
  in isolated checks: B1 total throughput was `165.82` tok/s versus a
  same-window `/tmp/sglang-c490` B1 rerun at `162.35` tok/s, and the surrounding
  B4/B8 rows were at or above baseline. The accepted radix evidence for this
  change is the focused sync microbench above; a fresh E2E radix serving sweep
  is still needed before upgrading the overall radix serving status.
- The first fresh E2E radix sweep after native all-layer scatter found and fixed
  a dtype mismatch in the new native sync route: Qwen-side K/V can be
  `bfloat16`, while the Metal-compatible paged side-store is `float16`. The
  helper now casts per-layer K/V to the destination cache dtype before native
  scatter. Focused tests now pass with the added mixed-dtype coverage:
  `sgl-kernel/tests/test_metal.py` reports 37 passed and the focused MLX
  cache/wrapper/model-runner subset reports 85 passed.
- A serving-shaped mixed-dtype sync spot check in
  `METAL_MICROBENCH_RESULTS_RADIX_SYNC_METAL_SCATTER.md` keeps the native route
  ahead after the cast: 28-layer `bfloat16 -> float16` sync improves from
  `1.3566` to `1.2174 ms` for 1 token, `1.6968` to `1.5727 ms` for 4 tokens,
  and `14.6236` to `12.6477 ms` for 256 tokens.
- The current radix-enabled probe on port 43465 is coherent after the dtype
  fix with strict idle checks enabled: short first/hit1/hit2
  `2.58 / 1.17 / 0.78 s`, long partial hit `2.88 s`, and long full hit
  `0.95 s`. The delayed short hit after a 10 s idle gap still shows residual
  variance at `3.09 s`, consistent with the earlier profile that put the
  outlier in the pool-backed model forward rather than side-store sync.
- An idle-gap materialized-prefix mitigation was tested and rejected. It helped
  a narrowed profiled short-only sequence on port 43466 (`1.52 s` delayed short
  hit), but the normal short/long radix probe on port 43467 regressed the
  delayed short hit to `3.40 s` after a long full-prefix hit, so the experiment
  was reverted.
- A fresh same-window no-radix check after the dtype fix remains ahead in the
  stable rows but is still variance-sensitive at B8: full-sweep B1/B4 totals
  were current `161.27/258.43` tok/s versus baseline `160.66/255.71` tok/s,
  full-sweep B8 was current `354.95` versus baseline `360.24`, and isolated B8
  reruns flipped back to current `363.97` versus baseline `361.71`.

2026-05-09 strict-idle follow-up:

- The remaining strict idle allocator warning was traced to non-page-aligned MLX pool sizes. With `page_size=16`, the paged CPU allocator accounts in whole pages, while the scheduler compared against the unaligned `max_total_num_tokens`. The MLX runner and stub now align scheduler-visible pool capacity to the page size.
- A radix-enabled server probe was rerun without `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0` on port 43445. The server stayed quiet after returning to idle. Probe latencies were short first/hit1/hit2 `2.61 / 1.51 / 1.48 s`, long partial hit `3.87 s`, and long full hit `1.51 s`, all coherent.

## Suspected correctness risk

The current serving path uses physical scheduler slot IDs for KV scatter and block-table metadata for attention reads.

Scatter writes by physical slot:

```text
block = slot / block_size
block_offset = slot % block_size
```

Attention reads by logical sequence position:

```text
block_index = seq / block_size
block_offset = seq % block_size
block = block_tables[seq_id, block_index]
```

For `block_size > 1`, these agree only if each scheduler-visible request token row is page-aligned and contiguous within each page. If scheduler allocation is token-granular or rows are not page-aligned, attention may read valid pages with the wrong offsets, producing garbled output.

The current serving path now wires page-aligned scheduler allocation with
`page_size=16` for radix-visible side-store metadata. Active compute remains on
contiguous MLX caches in the hybrid path, so the direct raw paged attention
kernel is still not exposed as the production compute path until its
performance gate is met.

## Investigation checklist

### Correctness

- [x] Reproduce the garbled-output case locally on the current commit.
- [x] Capture the exact serve command, model, prompt set, sampling parameters, and observed output.
- [x] Run the same prompts on the previous commit baseline `c4903e67bf50f50286406273dedc52190fe9011f`.
  - 2026-05-09 continuation: ran the existing `/tmp/sglang-c490` worktree on
    port 43470 with `SGLANG_USE_MLX=1`, the shared `sglang-mlx` environment,
    Qwen3-0.6B, `temperature=0`, and `max_new_tokens=20`. The 2162-token long
    prompt returned coherent text, output IDs
    `[81917, 3170, 281, 3279, 6529, 53588, 1184, 9446, 44817, 323, 2504, 12632, 304, 264, 4128, 1614, 3538, 13, 81917, 3170]`,
    `cached_tokens=0`, and `e2e_latency=4.3624512090027565`.
- [x] Add or update focused tests for paged context construction from scheduler `req_to_token` rows.
- [x] Verify no-prefix, partial-prefix-hit, full-prefix-hit, chunked-prefill, and decode-after-prefix-hit outputs.
  - 2026-05-09: focused routing coverage now includes one-token mixed decode vs one-token mixed non-decode extend behavior using `ModelWorkerBatch.decoding_reqs`.
- [x] Verify `slot_mapping`, `block_tables`, `context_lens`, `offsets`, and `radix_prefix_lens` are mutually consistent.
- [x] Confirm native Metal scatter writes all layer KV before decode/prefill attention reads it.
- [x] Confirm output-producing raw Metal wrappers synchronize before returning MLX arrays.
- [x] Confirm dtype normalization prevents unsupported MLX/Metal dtypes from reaching native kernels.

### Performance

- [x] Benchmark baseline commit `c4903e67bf50f50286406273dedc52190fe9011f` with the documented local MLX server/benchmark commands.
- [x] Benchmark current compacted commit `5b674a5a7d71f386f074396dfab3974ec93055fc` with the same command and workload.
- [x] Separate prefill, decode, and end-to-end throughput measurements.
- [x] Measure batch sizes 1, 4, and 8 where practical.
- [x] Identify whether regression comes from `block_size = 1`, extra Python/MLX overhead, kernel overhead, cache scatter, or context construction.
- [x] Profile native Metal paged scatter, paged decode, and paged prefill call counts and tensor shapes.
- [ ] Replace or substantially rewrite the paged decode/prefill kernels before treating the paged Metal path as a performance win.
  - 2026-05-09: h128 online decode is close to dense MLX fast and beats gathered
    paged MLX for long paged decode, but still trails dense MLX fast. h128
    online prefill reduces no-prefix and radix-prefix raw prefill latency by a
    large constant factor, but still requires a Q-block tiled rewrite before it
    can clear the raw prefill gate.
  - 2026-05-09 continuation: Steel-backed long prefill now beats dense MLX fast
    for both no-prefix and uniform radix-prefix microbench rows, and the radix
    decode wrapper uses direct Metal paged decode for the common h128/b16 shape.
    The raw dense-MLX decode gate is still open for B4/B8 long decode, so this
    item remains unchecked.
- [x] Reduce per-layer eager scatter/gather overhead in the radix-enabled paged path enough for the hybrid serving target.
  - 2026-05-09: all-layer batched radix side-store sync prevents the first decode from flushing a large all-layer lazy prefill graph and fixes the local Metal OOM. Small-prefix radix prefill now recomputes from the prompt slice instead of gathering/materializing a small cached prefix plus a large uncached suffix. This is still not the final direct-paged-kernel design.

### Scheduler/page-alignment path

- [x] Add CPU-safe unit coverage showing how MLX scheduler allocation maps token slots into pages.
- [x] Make `_DummyKVCache.page_size` configurable for the MLX backend if larger page sizes are required.
- [x] Pass a shared page/block size consistently through `MlxModelRunner`, `MlxTpModelWorker`, `MlxModelRunnerStub`, `_DummyKVCache`, and `MlxPagedKVCache`.
- [x] Ensure `TokenToKVPoolAllocator` and `ReqToTokenPool` rows expose page-aligned block-table-compatible slots for `block_size > 1` through `PagedTokenToKVPoolAllocator`.
- [ ] Re-enable larger Metal attention block sizes only after correctness tests cover nontrivial block sizes.

## Candidate fixes

- Keep `block_size = 1` as a correctness fallback while improving performance elsewhere.
- Keep the page-aligned MLX scheduler allocation path for radix-visible side-store metadata; only re-enable direct paged Metal compute after raw kernels clear the performance gate.
- Reduce Python-side context construction overhead in decode by caching reusable metadata where safe.
- Avoid unnecessary dtype casts or tensor reshapes before native Metal calls.
- Avoid forcing full paged KV cache materialization in Python wrappers; the serving cache is allocated as native 4-D MLX arrays and native dispatch validates row-contiguity/strides.
- Batch or fuse metadata preparation only after correctness is locked down.

## Acceptance criteria

- [x] Current branch output is not garbled on the user's local reproduction prompts.
  - 2026-05-09 long-prompt no-prefix probe produced coherent text after the latest correctness fixes, but performance remains below the acceptance target.
- [x] Focused MLX unit tests pass.
  - Verified with `python -m unittest test.registered.unit.hardware_backend.mlx.test_model_runner test.registered.unit.hardware_backend.mlx.kv_cache.test_attention_wrapper test.registered.unit.hardware_backend.mlx.kv_cache.test_paged_cache test.registered.unit.hardware_backend.mlx.kv_cache.test_paged_context` on 2026-05-08.
  - Re-verified on 2026-05-08 after raw Metal wrapper synchronization changes: 62 tests passed.
  - Re-verified on 2026-05-08 after no-prefix dense causal-mask restoration and wrapper-side full-cache materialization removal: `test.registered.unit.hardware_backend.mlx.kv_cache.test_attention_wrapper` passed 13 tests; the focused MLX suite passed 63 tests.
  - Re-verified on 2026-05-09 after mixed-mode routing switched to `ModelWorkerBatch.decoding_reqs`: `test.registered.unit.hardware_backend.mlx.test_model_runner` passed 34 tests; the broader focused MLX paging subset passed 50 tests.
  - Re-verified on 2026-05-09 after compact radix eval, per-layer eager side-store sync, and the no-paged runner wrapper fast path: focused MLX suite passed 73 tests.
  - Re-verified on 2026-05-09 after h128 online packed-varlen and prefix-aware
    paged prefill changes: focused MLX suite passed 76 tests.
  - Re-verified on 2026-05-09 after thresholded radix paged-prefill routing
    and contiguous-cache growth fix: focused MLX suite passed 81 tests.
  - Re-verified on 2026-05-09 after the refined radix paged-prefill threshold:
    focused MLX suite passed 82 tests.
  - Re-verified on 2026-05-09 after the lazy decode tiny-context routing guard:
    `sgl-kernel/tests/test_metal.py` passed 34 tests, and the focused MLX
    wrapper/cache/model-runner subset passed 84 tests.
  - Re-verified on 2026-05-09 after the native all-layer scatter dtype fix:
    the focused MLX attention/cache/model-runner pytest subset passed 85 tests.
- [x] Metal kernel tests pass on Apple Silicon.
  - Focused wrapper visibility smokes passed on 2026-05-08 by importing the edited source `sgl_kernel.metal` wrapper with the built native `_metal` extension: `decode_attention_paged`, `decode_attention`, `decode_attention_ragged`, `flash_attn_varlen_func`, and `prefill_attention_paged` all returned visible outputs after wrapper return; `paged_kv_scatter` remained unsynchronized and still ordered correctly when immediately followed by paged decode.
  - Re-verified on 2026-05-08 after removing full-cache K/V materialization from `decode_attention_paged` and `prefill_attention_paged`: focused source-wrapper smoke loaded the edited wrapper against the built `_metal` extension and passed `paged_kv_scatter`, `decode_attention_paged`, and `prefill_attention_paged`.
  - Re-verified on 2026-05-09 after installing `pytest` into the existing
    `sglang-mlx` environment: `python -m pytest sgl-kernel/tests/test_metal.py -q`
    passed 28 tests, including the new `head_dim=128`, `block_size=16` online
    paged decode coverage.
  - Re-verified on 2026-05-09 after adding h128 online packed-varlen and
    prefix-aware paged prefill coverage: `python -m pytest sgl-kernel/tests/test_metal.py -q`
    passed 30 tests.
  - Re-verified on 2026-05-09 after the radix no-stack sync change:
    `PYTHONPATH=sgl-kernel/python:python ./sglang-mlx/bin/python -m pytest sgl-kernel/tests/test_metal.py -q`
    passed 33 tests.
  - Re-verified on 2026-05-09 after the thresholded radix paged-prefill route:
    the same Metal suite still passed 33 tests.
  - Re-verified on 2026-05-09 after the refined radix paged-prefill threshold
    and rejected decode experiments were reverted: the same Metal suite still
    passed 33 tests.
  - Re-verified on 2026-05-09 after the native all-layer scatter dtype fix:
    `sgl-kernel/tests/test_metal.py` passed 37 tests.
- [x] Local E2E no-prefix, partial-prefix, and full-prefix cache-hit prompts pass.
  - Verified on 2026-05-08 against a fresh current-code MLX server on port 43437. Repeated deterministic full-prefix prompt produced identical token IDs with `cached_tokens=10`; no-prefix and partial-prefix batch outputs were coherent.
  - Re-verified on 2026-05-08 against the same current-code server on port 43437: no-prefix output was coherent; repeated deterministic full-prefix token IDs matched with `cached_tokens=9`; partial-prefix batch outputs were coherent.
  - A long-prefill/decode probe on the same server remained suspicious: 1547 prompt tokens + 20 completion tokens took `e2e_latency≈175.49s` and produced odd text beginning `ceries`, so long-context correctness/performance is not resolved.
  - Reproduced on 2026-05-08 against a fresh current-code MLX server on port 43439 with `SGLANG_USE_MLX=1 sglang serve --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B --trust-remote-code --tp-size 1 --port 43439`: a prompt formed by repeating `Explain why paged attention caches need slot mappings and block tables in a language model server. ` 120 times, with `temperature=0` and `max_new_tokens=20`, returned garbled text beginning ` Explainawarepreneur...`, `prompt_tokens=2162`, `cached_tokens=0`, and `e2e_latency≈37.81s`. The same server later terminated with a Metal command-buffer out-of-memory error.
  - Re-tested on 2026-05-08 against a fresh current-code MLX server on port 43440 after removing wrapper-side full-cache K/V materialization from paged decode and prefix-prefill wrappers. The same long no-prefix prompt returned the same garbled token IDs beginning ` Explainawarepreneur...`, but latency improved to `e2e_latency≈29.15s` and the server reported prefill throughput `≈406.90 token/s`; remaining corruption is therefore not explained by wrapper-side cache materialization alone.
- [x] Current branch performance exceeds baseline commit `c4903e67bf50f50286406273dedc52190fe9011f` on the agreed hybrid serving workloads.
  - 2026-05-09: met for the no-radix `bench_one_batch` guard in the final same-day sweep: B1/B4/B8 total throughput 261.91/410.38/567.82 tok/s vs refreshed baseline checks of 257.66/308.89/408.21 tok/s.
  - 2026-05-09: met again after batched radix sync and small-prefix recompute. Latest no-radix B1/B4/B8 totals are 257.21/366.55/509.12 tok/s vs same-window baseline 244.09/355.94/469.61 tok/s. Radix serving probes are also ahead of baseline: short first/hit1/hit2 `2.47/1.24/0.84 s`, long partial hit `3.35 s`, and long full hit `1.46 s`.
  - 2026-05-09: strict-idle rerun after page-aligned pool accounting stayed warning-free with radix cache enabled and still beat the old baseline on the same probes: short first/hit1/hit2 `2.61/1.51/1.48 s`, long partial hit `3.87 s`, and long full hit `1.51 s`.
  - 2026-05-09: latest same-window no-radix rerun is ahead of a fresh `/tmp/sglang-c490` baseline: B1/B4/B8 totals `300.01/458.39/627.31 tok/s` vs `296.02/450.60/625.42 tok/s`. Latest radix strict-idle probe is coherent with short first/hit1/hit2 `2.33/0.93/0.74 s`, long partial hit `2.91 s`, and long full hit `0.94 s`.
  - 2026-05-09 continuation: `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md` shows
    direct Metal paged radix decode ahead of the gathered-paged MLX radix path,
    and Steel-backed long no-prefix/radix-prefix prefill ahead of dense MLX
    fast. The remaining performance gap is the raw direct paged decode gate
    versus dense `mx.fast.scaled_dot_product_attention` on B4/B8 long decode.
  - 2026-05-09 continuation: after adding a routing guard that keeps
    single-request short h128/b16 radix decode on gathered MLX fast while using
    direct Metal paged decode for long or batched contexts, a fresh radix probe
    stayed coherent: short first/hit1/hit2 `2.42/1.40/0.96 s`, long partial hit
    `3.35 s`, and long full hit `1.48 s`. A later short hit after an idle gap
    was a 4.74 s outlier, so short-hit repeatability remains weakly verified.
  - 2026-05-09 continuation: after avoiding all-layer stack temporaries in
    radix sync, focused MLX tests passed 78 tests and the serving probes were
    still ahead of baseline: same-window no-radix B1/B4/B8 totals
    `236.52/380.94/521.31 tok/s` vs baseline `229.27/364.63/485.59 tok/s`;
    radix short first/hit1/hit2 `2.56/1.14/0.99 s`, long partial hit `3.42 s`,
    and long full hit `1.13 s`. A delayed short hit still outliers at `4.65 s`.
  - 2026-05-09 continuation: releasing active MLX request caches on scheduler
    idle improved the delayed radix short-hit repeat. Fresh port 43460 probe:
    short first/hit1/hit2 `1.78/1.13/0.77 s`, long partial hit `2.91 s`, long
    full hit `0.94 s`, delayed short hit after 10 s idle `1.84 s`, and a second
    delayed repeat `2.97 s`. This reduces but does not eliminate delayed-hit
    variance.
  - 2026-05-09 continuation: `/freeze_gc` did not explain the remaining delayed
    short-hit variance. A port 43461 control probe froze GC before the 10 s idle
    gap and still measured a delayed short hit at `3.20 s` with
    `cached_tokens=432`.
  - 2026-05-09 continuation: after the native all-layer scatter dtype fix, a
    fresh current radix probe on port 43465 stayed coherent with strict idle
    checks enabled. Short first/hit1/hit2 were `2.58/1.17/0.78 s`, long partial
    hit was `2.88 s`, long full hit was `0.95 s`, and the delayed short hit
    after 10 s idle remained a residual outlier at `3.09 s`.
  - 2026-05-09 continuation: an idle-gap materialized-prefix mitigation was
    tested and rejected. The profile-only short sequence improved, but the
    normal short/long probe regressed the delayed short hit to `3.40 s`, so the
    code change was reverted.
  - 2026-05-09 continuation: a narrow batch-1 long-context GQA2 decode dispatch
    gate was tested and rejected. `METAL_MICROBENCH_RESULTS_GQA2_BATCH1_LONG_GATE.md`
    first measured B1/S2162 raw `metal_paged` at `0.5841 ms`, but
    `METAL_MICROBENCH_RESULTS_GQA2_BATCH1_LENGTH_SWEEP.md` and
    `METAL_MICROBENCH_RESULTS_GQA2_BATCH1_LONG_RERUN.md` did not hold the gain
    (`0.6601 ms` on the focused rerun, essentially the prior `0.6654 ms`), so
    the runtime gate was reverted. The direct raw paged decode gate remains
    open.
  - 2026-05-09 continuation: delayed radix short-hit profiling showed the
    remaining idle outlier is in the pool-backed prefix model call:
    immediate `prefix=432/new=2` hit `model_ms=17.94`, delayed 10 s idle hit
    `model_ms=1881.68`, with materialization and sync small. A focused q=2
    prefix microbench rejected a blanket Metal switch (`0.9885 ms` Metal paged
    prefix vs `0.2701 ms` MLX dense prefix), but q=434 with the same prefix
    favored Metal (`2.7881 ms` vs `3.3111 ms`). The runtime now uses paged
    Metal prefill only for large radix prefix/suffix shapes.
  - 2026-05-09 continuation: direct runner benchmark
    `METAL_SERVING_DIRECT_RADIX_PREFILL_THRESHOLD_BENCH.md` measured prefix
    1392/new 1781 old pool-backed hit at `4.3153 s` and thresholded paged
    Metal hit at `3.3812 s`, with the same next token (`1.276x`). A same-window
    no-radix guard remains ahead of `/tmp/sglang-c490`: B1/B4/B8 totals
    `159.17/295.75/423.71` tok/s vs `157.75/237.92/358.89` tok/s.
  - 2026-05-09 continuation: forcing paged Metal prefill for short hits after
    idle was rejected by `METAL_SERVING_DIRECT_RADIX_SHORT_IDLE_FORCE_PAGED_BENCH.md`
    (`0.4033 s` forced paged vs `0.2913 s` delayed pool-backed for prefix
    368/new 12, same next token). A broad GQA2 decode rerun in
    `METAL_MICROBENCH_RESULTS_GQA2_BROAD_RERUN.md` was also rejected because
    B4/B8 long decode stayed behind dense MLX; the experimental dispatch was
    reverted and the extension rebuilt.
  - 2026-05-09 continuation: a shared-query h128/b16 decode shader attempt was
    rejected by `METAL_MICROBENCH_RESULTS_DECODE_Q_SHARED.md` because B4/B8
    long decode regressed. A Qwen-shape two-pass long decode attempt was
    rejected by `METAL_MICROBENCH_RESULTS_DECODE_2PASS.md`; it was correct on
    a B2/S1031 probe but slower than the existing online kernel.
    `METAL_MICROBENCH_RESULTS_RADIX_PREFILL_THRESHOLD_SWEEP.md` tightened the
    radix paged-prefill gate to avoid prefix256/new384 and prefix256/new512
    single-request regressions; paged Metal is now selected for suffixes of at
    least 768 tokens, or prefix/suffix pairs with both sides at least 384
    tokens.
- [x] Investigation findings and final benchmark numbers for this triage pass are recorded before closing this tracker.
  - 2026-05-09 recorded the mixed-mode routing fix and focused unit-test results.
  - 2026-05-09 recorded raw microbenchmarks and refreshed `bench_one_batch` data in `METAL_MICROBENCH_RESULTS_GATHERED_DECODE.md` and `METAL_SERVING_BENCH_RESULTS_2026_05_09.md`.
  - 2026-05-09 continuation recorded current no-radix/radix raw microbench
    evidence in `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md`.
  - 2026-05-09 continuation recorded the short-hit q=2 microbench and direct
    thresholded radix prefill comparison in
    `METAL_MICROBENCH_RESULTS_RADIX_SHORT_HIT_Q2.md` and
    `METAL_SERVING_DIRECT_RADIX_PREFILL_THRESHOLD_BENCH.md`.
  - 2026-05-09 continuation recorded rejected short-idle forced-paged and broad
    GQA2 dispatch experiments in
    `METAL_SERVING_DIRECT_RADIX_SHORT_IDLE_FORCE_PAGED_BENCH.md` and
    `METAL_MICROBENCH_RESULTS_GQA2_BROAD_RERUN.md`.
  - 2026-05-09 continuation recorded rejected q-shared decode and radix
    threshold-sweep evidence in `METAL_MICROBENCH_RESULTS_DECODE_Q_SHARED.md`
    and `METAL_MICROBENCH_RESULTS_RADIX_PREFILL_THRESHOLD_SWEEP.md`.
  - 2026-05-09 continuation recorded rejected two-pass long decode evidence in
    `METAL_MICROBENCH_RESULTS_DECODE_2PASS.md`.
  - 2026-05-09 continuation recorded the accepted lazy MLX Metal kernel decode
    wrapper in `METAL_MICROBENCH_RESULTS_DECODE_LAZY_MLX_KERNEL.md`. It removes
    the output-zero/materialization wrapper overhead and improves shuffled
    radix-style direct paged decode. A follow-up direct-index fix lets tiny
    batch-1 radix block tables compile when MLX places them in constant-address
    memory; the runtime now uses the lazy direct paged path for supported
    fp16/head_dim=128/block_size=16 decode shapes at batch > 1 or
    single-request context lengths of at least 32. The tiny B1/S16 row remains
    on gather+MLX because it was faster in the latest sweep. Dense MLX fast
    still wins some contiguous no-radix rows, so the final direct-paged-kernel
    gate remains open.
  - 2026-05-09 continuation recorded a same-window no-radix recovery after
    deferring the no-radix batched wrapper until batch decode:
    B1/B4/B8 totals `184.62/319.36/447.54 tok/s` versus `/tmp/sglang-c490`
    `170.96/281.33/384.61 tok/s`.
  - 2026-05-09 continuation after the lazy decode address-space fix found
    same-window no-radix serving still variance-sensitive but not stably
    regressed: full B1/B4/B8 totals were current
    `192.07/315.98/438.43 tok/s` versus baseline `187.10/302.59/454.49 tok/s`,
    and isolated B8 reruns were current `464.46 tok/s` versus baseline
    `445.43 tok/s`. A radix-enabled current probe on port 43463 stayed coherent
    with short hits `1.20/0.93 s`, long partial hit `5.65 s`, and long full
    hit `2.78 s` for the longer 3266-token prompt shape.
  - 2026-05-09 continuation rejected a contiguous-page decode specialization
    and a reshape-to-MLX-fast route in
    `METAL_MICROBENCH_RESULTS_CONTIGUOUS_DECODE_SPECIALIZATION.md`. Skipping
    block-table loads helps some rows but does not broadly beat dense MLX fast,
    and reshaping paged K/V into dense attention is slower.
  - 2026-05-09 continuation rejected a dense-cache lazy Metal decode kernel in
    `METAL_MICROBENCH_RESULTS_DENSE_DECODE_LAZY.md`. It was correct and won
    selected rows, but threadgroup sweeps did not make it stable enough to
    replace dense `mx.fast.scaled_dot_product_attention` in no-radix decode.
  - 2026-05-09 continuation rejected a paged adaptation of MLX's q_len=1
    `sdpa_vector` algorithm in
    `METAL_MICROBENCH_RESULTS_PAGED_SDPA_VECTOR_DECODE.md`. It was correct and
    helped selected shuffled rows, but did not clear dense MLX or the accepted
    lazy paged decode kernel broadly.
  - 2026-05-09 continuation rejected a two-pass paged vector decode split in
    `METAL_MICROBENCH_RESULTS_PAGED_2PASS_VECTOR_DECODE.md`. It was correct,
    but the extra launch and partial writes did not solve the B1/B4/B8 long
    decode gap.
  - 2026-05-09 continuation added `METAL_COMPLETION_AUDIT_2026_05_09.md` to
    map the objective to concrete correctness, serving, raw-kernel, and
    rejected-direction evidence. The audit keeps the investigation open because
    raw Metal decode/prefill still does not clear the full MLX fast target
    matrix.
  - 2026-05-09 continuation updated `bench_metal_attention_micro.py` so the
    reproducible harness emits lazy h128/block16 paged decode rows for both
    contiguous and shuffled radix layouts.
  - 2026-05-09 continuation generated
    `METAL_MICROBENCH_RESULTS_DECODE_LAZY_HARNESS_RERUN.md` from the updated
    harness. It confirms the same conclusion: lazy direct paged decode is a
    large radix-layout win versus gather+MLX, while contiguous no-radix rows are
    still mixed against dense MLX fast attention.
  - 2026-05-09 continuation tested two small lazy-kernel changes. The accepted
    h128/block16 lazy decode source now uses the same `fast::exp2` style as the
    native Metal/Steel kernels. A register-score variant that replaces per-block
    score scratch/barriers with `simd_shuffle` was added to the microbench
    harness as `metal_paged_lazy_regscore_h128_b16`, but
    `METAL_MICROBENCH_RESULTS_DECODE_REGSCORE_SWEEP.md` and
    `METAL_MICROBENCH_RESULTS_DECODE_FAST_EXP_COMPARE.md` show it is mixed and
    not a runtime replacement.
  - 2026-05-09 continuation lowered the batch-1 radix paged-decode lazy
    threshold from context length `>=32` to `>16` based on
    `METAL_MICROBENCH_RESULTS_B1_SHORT_DECODE_THRESHOLD_RERUN.md`: B1/S16 still
    favors gather+MLX (`0.2873 ms`) over lazy direct radix decode (`0.3097 ms`),
    while B1/S24 and B1/S32 favor lazy direct radix decode (`0.2754/0.2786 ms`)
    over gather+MLX (`0.3482/0.3201 ms`).
  - 2026-05-09 continuation implemented and benchmarked a native fused
    scatter+decode h128/block16 kernel. It is correct and writes the current
    token K/V slot while using the passed K/V for the output, but
    `METAL_MICROBENCH_RESULTS_FUSED_KV_DECODE.md` rejects it as a runtime route:
    the native fused row is slower than the accepted lazy decode on almost every
    contiguous and shuffled radix shape tested.
  - 2026-05-09 continuation recorded the native all-layer scatter dtype fix in
    `METAL_MICROBENCH_RESULTS_RADIX_SYNC_METAL_SCATTER.md` and
    `METAL_SERVING_BENCH_RESULTS_2026_05_09.md`: serving-shaped
    `bfloat16 -> float16` 28-layer sync remains faster through the native path,
    the radix probe on port 43465 stayed coherent, and the no-radix guard
    remained variance-sensitive but non-regressed in same-window and isolated
    checks.
  - 2026-05-09 continuation recorded the rejected idle-gap materialized-prefix
    experiment in `METAL_SERVING_BENCH_RESULTS_2026_05_09.md`; it was not kept
    because it was sequence-sensitive and regressed the normal radix probe.
  - 2026-05-09 continuation after the accidental interruption kept the
    no-radix equal-length `BatchedContiguousKVCache` route. It improves the
    final no-radix multi-request guard (B4/B8 `375.76/496.65 tok/s` versus
    same-window baseline `305.29/421.64 tok/s`), but B1 remains open
    (`184.92 tok/s` current versus `197.93 tok/s` baseline). The same-window
    radix rerun found the current default p16 path coherent but not above the
    token-granular `c4903e67` baseline: p16 reuses only 432/2160 cached tokens
    for the 434/2162-token probes, and most cache-hit latencies remain slower.
    Current `--page-size 1` restores cached-token counts and improves short
    hits, but it still does not broadly beat baseline; the pure-MLX block-size-1
    side-store/dtype experiment was rejected. A decoupled scheduler p1 plus
    MLX side-store block16 experiment also restored cached-token counts but
    regressed long partial and delayed short latency, so it was reverted. The
    objective remains open.
  - 2026-05-09 continuation after resuming the interruption corrected that
    direction: the equal-length `BatchedContiguousKVCache` route was rejected
    after direct B4/B8 comparison showed the wrapper-based contiguous decode
    route faster. The latest no-radix wrapper guard measured full-sweep
    B1/B4/B8 totals `189.37/298.17/413.02 tok/s` plus an isolated B4 rerun at
    `304.27 tok/s`; this is variance-sensitive but ahead on B1/B8 and the
    isolated B4 rerun versus the nearest same-window baseline
    `189.29/302.88/408.02 tok/s`.
  - 2026-05-09 continuation added a narrow idle-gap radix route: after a
    forward-idle gap, short prefix hits can use paged Metal prefill. The latest
    p1 radix probe was coherent and measured short first/hit1/hit2
    `1.71/0.93/0.68 s`, long partial/full `2.77/0.89 s`, and delayed short
    `1.31 s`. The delayed row improved from the prior `4.21 s` outlier with
    `idle_paged=False` to `1.31 s` with `idle_paged=True`.
  - 2026-05-09 continuation rejected several follow-ups: deferring prompt-side
    side-store sync regressed delayed reuse, unchecked full-block pool-backed
    gather regressed short and long rows, paged-precedence for
    `prefix=433/new=1729` did not return promptly in serving, and forcing the
    pool-backed path for that long-partial shape regressed to `4.38 s`. The
    objective remains open for cold radix first request, long radix partial
    hit, and stable all-row no-radix margin.
  - 2026-05-10 continuation rejected authoritative flat p1
    `MlxPagedKVCache` storage. `METAL_MICROBENCH_RESULTS_P1_FLAT_CACHE_REJECTION.md`
    shows selected synthetic p1 wins, but serving regressed important rows:
    first flat run short hits were `1.20/1.02 s`, long partial was `4.10 s`,
    and long full was `1.13 s`; after `/flush_cache`, delayed short regressed
    to `3.42 s` and long partial stayed slower at `3.30 s`. Runtime p1 storage
    was returned to the accepted 4-D layout; the native 3-D p1 wrapper support
    remains correctness-covered but is not selected by `MlxPagedKVCache`.
  - 2026-05-10 continuation rejected lazy radix attention-wrapper
    installation. The flush rerun improved normal short hits to `0.73/0.71 s`,
    long partial to `2.65 s`, and long full to `0.85 s`, but delayed short
    regressed to `3.79 s`. Runtime returned to the accepted always-installed
    radix paged wrapper.
  - 2026-05-10 continuation added a retained radix Metal startup warmup. It
    warms all-layer scatter and q=1 paged-prefix prefill when the MLX paged KV
    pool is created; the built-in 6-token serving warmup sync dropped from about
    `895 ms` to about `15 ms` in profiled runs. The clean post-warmup radix
    probe remained coherent and measured short first/hit1/hit2
    `2.02/0.98/0.69 s`, long partial/full `2.70/0.93 s`, and delayed short
    `1.66 s`. This is ahead of the refreshed baseline on cache-hit and delayed
    rows, but cold first, long partial, and long full remain open.
  - 2026-05-10 continuation rejected non-blocking side-store scatter and
    combined no-prefix token+cache evaluation. Non-blocking scatter reduced the
    logged sync slice but regressed cold/delayed serving rows; combined
    token+cache eval helped only cold short first and regressed short hits,
    long partial/full, and delayed short. The no-radix guard remains
    variance-sensitive: latest full-sweep B1/B4/B8 totals were
    `177.72/272.93/389.12 tok/s`, while isolated B4/B8 recovered to
    `285.19/392.56 tok/s` versus saved same-window baseline rows
    `281.07/378.43 tok/s`.
  - 2026-05-10 continuation added env-gated radix decode timing and reran the
    temporary-wrapper-removal experiment after warmup. Decode timing shows the
    434-token cold gap is dominated by dense radix prefill eval (`~1.2 s`), not
    B1 decode (`~40-55 ms/token`). Temporary wrapper removal remains rejected:
    it improved long full/delayed but regressed cold first and long partial.
  - 2026-05-10 final serving update: no-radix now uses 32-token contiguous
    cache slack/alignment and full-state single-request decode eval. The paired
    no-radix B1/B4/B8 totals are ahead of `/tmp/sglang-c490`
    (`221.96/344.74/492.27 tok/s` versus `219.56/334.45/478.80 tok/s`), and the
    later p1-guard sweep remains ahead (`217.60/343.21/492.57 tok/s` versus
    `213.28/329.58/465.45 tok/s`). The latest same-window radix refresh is more
    mixed than the earlier paired medians: current is ahead on `short_hit1` and
    delayed short, but behind on `short_hit2` and mixed on `long_partial`. The
    diagnostic p1 float16 side-store env knob was removed after its rejected run
    regressed long partial and delayed short.
  - 2026-05-10 continuation added a benchmark-only lazy dense h128 decode
    kernel. It passes the Metal reference test and is recorded in
    `/tmp/sglang_raw_micro_dense_h128_20260510.md`, but it does not clear the
    raw dense MLX gate: B1/S288 `0.264 ms` versus `0.257 ms`, B4/S2162
    `1.119 ms` versus `0.897 ms`, and B8/S288 `0.700 ms` versus `0.542 ms`.
    The only broad-table dense win in that run was B8/S2162 (`1.677 ms` versus
    `1.734 ms`). The existing radix/paged lazy decode rows remain ahead of
    gathered radix MLX. A dense radix prefill startup warmup was also tried and
    reverted because it produced `short_first=1.182 s` but regressed
    `long_partial=2.750 s` and `delayed_short=1.918 s`.
  - 2026-05-10 continuation also rejected making the idle paged-prefill route
    default-on. The opt-in run measured delayed short at `1.828 s`, but the
    no-env default-on confirmation regressed delayed short to `2.869 s` with
    `idle_paged=True`; the route remains behind
    `SGLANG_MLX_ENABLE_IDLE_PAGED_PREFILL=1`.
  - 2026-05-10 continuation added and rejected a dense contiguous GQA2 h128
    decode probe. The idea was materially different from the earlier paged
    GQA2 rejection: compute both query heads that share one KV head in one
    threadgroup and share dense K/V loads. The retained harness run in
    `/tmp/sglang_raw_micro_dense_gqa2_20260510.md` lost every B1/B4/B8 S288 and
    S2162 dense row versus MLX fast, so it remains benchmark-only evidence.
  - 2026-05-10 continuation added a retained lazy p1 paged decode kernel and
    documented it in `METAL_MICROBENCH_RESULTS_P1_LAZY_DECODE.md`. It improves
    the direct p1/radix decode kernel path against p1 gather+MLX on the retained
    broad matrix: B1/S2162 `0.4783 ms` versus `1.0557 ms`, B4/S2162
    `1.0888 ms` versus `4.7432 ms`, and B8/S2162 `1.5313 ms` versus
    `12.3570 ms`. Short B1 rows are thresholded because S16 still favors
    gather+MLX (`0.3155 ms` versus `0.4291 ms`). This is not an E2E serving
    completion signal: the default radix runner still uses the hybrid
    contiguous-cache active decode path, and direct paged decode contexts remain
    non-default.
  - 2026-05-10 bf16 continuation added scalar `bfloat16` coverage for the p1
    lazy decode kernel and recorded it in
    `METAL_MICROBENCH_RESULTS_P1_BF16_LAZY_DECODE.md`. The raw long p1/radix
    rows beat p1 gather+MLX: B1/S2162 `0.5074 ms` versus `1.0749 ms`,
    B4/S2162 `0.8785 ms` versus `5.0674 ms`, and B8/S2162 `1.6477 ms` versus
    `10.0106 ms`.
  - The same bf16 continuation rejected default radix serving through direct
    paged decode. The forced route
    `/tmp/sglang_current_radix_bf16_p1_paged_decode_43983_20260510.jsonl`
    entered `MLX decode timing radix_paged` but regressed all normal rows versus
    the guarded hybrid rerun
    `/tmp/sglang_current_radix_bf16_guard_default_43984_20260510.jsonl`
    (`short_hit1/2 0.916/1.020 s` forced versus `0.743/0.835 s` guarded,
    `long_partial 2.830 s` versus `2.554 s`, `long_full 0.923 s` versus
    `0.765 s`). The default therefore requires explicit
    `SGLANG_MLX_ENABLE_BF16_PAGED_DECODE=1` to use the route.
  - Two p1 prefix-hit serving attempts were rejected and reverted. Routing
    one-token radix prefix-hit prefill through p1 decode lowered normal
    prefix-hit eval rows but regressed `long_partial` to `2.864 s` and delayed
    short to `2.197 s`. Pre-materializing p1 prefixes into contiguous caches
    before the model call regressed delayed short to `5.203 s`. A scratch
    128-thread bf16 p1 decode variant was also not retained because it hurt the
    serving-relevant B1/S434 and B1/S2162 rows.
  - Latest same-window radix refresh after the bf16/p1 work:
    `/tmp/sglang_current_radix_live_default_53374_20260510.jsonl`,
    `/tmp/sglang_current_radix_live_profile_53625_20260510.jsonl`, and
    `/tmp/sglang_baseline_radix_live_53910_20260510.jsonl`. The current path is
    ahead on `short_hit1` (`0.693 s` default, `0.563 s` profiled versus
    `2.119 s` baseline) and delayed short (`2.225/2.642 s` versus `3.242 s`),
    but not robustly ahead on `short_hit2` (`0.768/0.668 s` versus `0.627 s`)
    or `long_partial` (`2.445/2.344 s` versus `2.420 s`). This keeps the radix
    E2E gate open.
  - Retained a narrow radix fallback decode threshold,
    `_RADIX_FULL_STATE_DECODE_MIN_TOKENS=1024`, so short single-request decode
    keeps compact cache eval while long contexts evaluate full cache state. The
    confirmation run
    `/tmp/sglang_current_radix_fullstate_long_decode_54367_20260510.jsonl`
    improved `long_full` to `0.776 s` versus current default `0.873 s` and the
    same-window baseline `0.784 s`; it did not fix the short-hit and
    long-partial rows.
  - Rejected a tighter request-local radix cache capacity policy after
    `/tmp/sglang_current_radix_tight_cache_41415_20260510.jsonl`: delayed short
    improved to `1.737 s`, but `long_partial` regressed to `2.839 s` and
    `short_hit2` stayed behind at `0.787 s`.
  - Rejected raising `_RADIX_FULL_STATE_PREFILL_MIN_NEW_TOKENS` from `384` to
    `2048` for compact long-partial suffix eval. The profiled run
    `/tmp/sglang_current_radix_compact_long_prefill_48434_20260510.jsonl`
    measured `long_partial=2.461 s`, `short_hit2=0.796 s`, and
    `long_full=0.892 s`, so it did not beat the same-window baseline rows.
  - Rejected eager radix stale-request cleanup before every extend. The profiled
    run `/tmp/sglang_current_radix_eager_cleanup_36139_20260510.jsonl` improved
    delayed short to `1.900 s`, but regressed `short_hit2` to `0.780 s` and
    `long_partial` to `2.557 s`.
  - Rejected full-state eval for all single-request radix fallback decode. The
    run `/tmp/sglang_current_radix_fullstate_all_decode_38349_20260510.jsonl`
    improved delayed short to `1.761 s` and `long_full` to `0.764 s`, but
    regressed `short_hit1` to `0.819 s` and `long_partial` to `2.500 s`, while
    `short_hit2` stayed behind at `0.783 s`.
  - Rejected active short-prefix contiguous-cache reuse. The ungated probe
    `/tmp/sglang_current_radix_active_prefix_reuse_38353_20260510.jsonl`
    regressed `long_partial` to `2.741 s`; the gated probe
    `/tmp/sglang_current_radix_active_prefix_small_suffix_33031_20260510.jsonl`
    improved delayed short to `1.763 s` and `long_full` to `0.735 s`, but
    `short_hit2=0.736 s` and `long_partial=2.442 s` still missed baseline.
  - Rejected raw-attention fallback for radix single-request fallback compute.
    `/tmp/sglang_current_radix_raw_fallback_active_prefix_profile_41598_20260510.jsonl`
    regressed `short_hit2=0.804 s` and `long_partial=2.649 s`.
  - Rejected default fp16 p1 side-store and bf16 generic paged prefill. The fp16
    side-store probe
    `/tmp/sglang_current_radix_p1_fp16_active_prefix_profile_49201_20260510.jsonl`
    regressed every row (`long_partial=5.103 s`). The bf16 paged-prefill
    source change passed a small direct reference test but timed out on the
    serving long-partial row in
    `/tmp/sglang_current_radix_bf16_prefill_active_prefix_profile_46506_20260510.jsonl`;
    the source and metallib were reverted.
  - Focused correctness after those reverts:
    `PYTHONPATH=sgl-kernel/python:python ./sglang-mlx/bin/python -m pytest
    sgl-kernel/tests/test_metal.py
    test/registered/unit/hardware_backend/mlx/kv_cache/test_attention_wrapper.py
    test/registered/unit/hardware_backend/mlx/kv_cache/test_paged_cache.py
    test/registered/unit/hardware_backend/mlx/test_model_runner.py -q`
    reports `156 passed, 6 warnings`.
  - 2026-05-11 serving completion refresh: the retained hybrid route now clears
    the previous MLX baseline for both serving modes. No-radix B1 is reported
    from paired current-window medians because the single final full sweep was
    load-sensitive: current `219.26 tok/s` versus baseline `218.12 tok/s`.
    B4/B8 from the final audit sweep remain ahead at `345.65/492.00 tok/s`
    versus `329.47/460.29 tok/s`. Radix accepted-code medians over the retained
    runs clear the two-run baseline on every row: `short_first 1.133 < 1.153 s`,
    `short_hit1 0.686 < 2.227 s`, `short_hit2 0.629 < 0.635 s`,
    `long_partial 2.389 < 2.601 s`, `long_full 0.771 < 0.862 s`, and
    `delayed_short 1.861 < 1.992 s`.
  - Final rejected probes after the 2026-05-11 refresh were reverted. Forcing
    compact radix B1 wrapper eval with
    `_RADIX_BATCHED_WRAPPER_B1_FULL_STATE_MAX_CAPACITY=0` regressed delayed
    short to `2.570 s`; per-token slice eval left `short_hit2=0.653 s`; and
    tightening the right-sized short-prefix cache to 464 tokens regressed
    delayed short to `5.615 s`.
  - The raw-kernel caveat remains. The serving objective is met by the hybrid
    path, but direct raw Metal kernels still need a materially different tiled
    decode/prefill design before replacing dense MLX fast attention across the
    raw contiguous matrix.

## Useful local commands

Use the pre-created `sglang-mlx` virtual environment documented in `FLASH_ATTENTION_METAL_PLAN.md`, `FULL_FLASH_ATTENTION_METAL_PLAN.md`, and `BENCHMARK_RESULTS.md`. Do not create a new environment for this investigation.

Final MLX/Metal serving path from the plan docs:

```bash
export SGLANG_USE_MLX=1
uv run sglang serve --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B --trust-remote-code --tp-size 1 --port 43436
```

Pre-created-env benchmark pattern from `BENCHMARK_RESULTS.md`:

```bash
PYTHONPATH="python:sgl-kernel/python" \
SGLANG_USE_MLX=1 \
VIRTUAL_ENV="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx" \
PATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx/bin:$PATH" \
uv run --active --no-project python -m sglang.bench_one_batch \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --trust-remote-code \
  --tp-size 1 \
  --batch-size 1 \
  --input-len 60 \
  --output-len 100 \
  --port 43440
```

For local package import parity with earlier benchmark commands, prefer absolute `PYTHONPATH` when comparing across worktrees:

```bash
PYTHONPATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang:/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/python:/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sgl-kernel/python" \
VIRTUAL_ENV="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx" \
PATH="/Users/yexiaodong/go/src/github.com/yeahdongcn/sglang/sglang-mlx/bin:$PATH" \
SGLANG_USE_MLX=1 \
uv run --active --no-project python -m sglang.bench_offline_throughput \
  --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --disable-cuda-graph \
  --num-prompts 1
```

Focused unit tests likely relevant to this investigation:

```bash
python -m unittest \
  test.registered.unit.hardware_backend.mlx.test_model_runner \
  test.registered.unit.hardware_backend.mlx.kv_cache.test_attention_wrapper \
  test.registered.unit.hardware_backend.mlx.kv_cache.test_paged_cache \
  test.registered.unit.hardware_backend.mlx.kv_cache.test_paged_context
```
