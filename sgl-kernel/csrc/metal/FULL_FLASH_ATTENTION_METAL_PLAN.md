# Full Apple Silicon Metal FlashAttention Backend Plan

This file tracks a new first-class MLX/Metal FlashAttention-style backend redesign for SGLang on Apple Silicon.
It intentionally does not preserve the current env-gated `SGLANG_MLX_USE_METAL_ATTENTION` design as the final architecture.

## Agent Retrieval Note

- Status: architecture and remaining-work plan. It records the accepted hybrid
  serving state and the still-open raw direct-kernel gate.
- Use when: deciding what to build next, checking acceptance gates, or avoiding
  rejected raw-kernel directions.
- Do not use as: proof that every direct paged Metal kernel path is production
  default. The plan explicitly requires evidence before replacing dense MLX
  fast attention on remaining raw rows.
- Read next: `METAL_DOC_INDEX.md`,
  `METAL_FINAL_PERF_RESULTS_2026_05_11.md`,
  `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md`,
  `MLX_METAL_REGRESSION_INVESTIGATION.md`.
- Search tags: `architecture-plan`, `remaining-work`, `open-raw-gate`,
  `paged-kv`, `radix-cache`, `rejected`.

## 2026-05-09 status reset

The earlier "complete" status is revoked. New microbenchmarks show the current
hand-written Metal attention kernels do not beat MLX fast attention, and the
serving-native paged-cache integration regresses `bench_one_batch` unless the
no-radix contiguous-cache guard is used.

Current direction:

- Keep the restored `disable_radix_cache=True` contiguous-cache path as a
  regression guard for benchmark/no-radix workloads.
- Use the current hybrid radix path as the serving stopgap: active requests run
  through contiguous MLX caches, while generated K/V is synced into
  `MlxPagedKVCache` for radix visibility. This is deliberately not the final
  "direct paged attention" architecture.
- Treat the current paged Metal decode/prefill kernels as correctness
  scaffolding, not accepted performance kernels.
- Continue the paged Metal backend only behind an evidence gate: raw paged
  decode/prefill microbenchmarks must match or beat `mx.fast.scaled_dot_product_attention`
  for the target Qwen3-0.6B shapes before serving integration is claimed complete.
- Use `vllm-metal`/mistral.rs-style online-softmax paged kernels, or an
  equivalent tiled implementation, as the next kernel direction. The simple
  score-cache fix helps long decode but is not enough.

2026-05-09 continued update:

- No-radix `bench_one_batch --batch-size 1 4 8 --input-len 256 --output-len 32 --warmups 1`
  now beats same-machine refreshed baseline checks after the wrapper fast path:
  B1/B4/B8 total throughput is 261.91/410.38/567.82 tok/s.
- Radix cache-hit serving probes now survive the previous Metal OOM and produce
  coherent output. Full-prefix hits beat the `c4903e67` baseline on the short
  and long deterministic probes, but a long partial-prefix-hit probe remained
  slower than baseline before the follow-up changes below.
- Strict idle allocator warnings were unresolved in this sweep and were
  disabled only for the first radix probe with
  `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0`.

2026-05-09 follow-up update:

- Batched radix side-store sync now evaluates all layers together instead of
  issuing one eager MLX evaluation per layer, and radix prefill recomputes from
  the full prompt when the cached prefix is small relative to the uncached
  suffix.
- The refreshed hybrid serving probes now beat the same-session `c4903e67`
  baseline for no-radix B1/B4/B8, short radix hits, long partial-prefix hits,
  and long full-prefix hits. Latest no-radix totals are 257.21/366.55/509.12
  tok/s versus same-window baseline 244.09/355.94/469.61 tok/s. Latest radix
  E2E latencies are short first/hit1/hit2 2.47/1.24/0.84 s, long partial hit
  3.35 s, and long full hit 1.46 s.
- The remaining performance caveat is no longer the hybrid radix serving probe;
  it is the raw direct paged Metal kernel gate versus MLX fast attention.

2026-05-09 strict-idle update:

- The strict idle allocator warning was traced to non-page-aligned MLX pool
  sizes with `page_size=16`. The MLX runner and model-runner stub now align
  scheduler-visible pool capacity to page boundaries so the idle checker and
  paged allocator account the same number of tokens.
- A strict-on radix serving rerun stayed warning-free after returning to idle.
  It remained ahead of the old baseline: short first/hit1/hit2
  2.61/1.51/1.48 s, long partial hit 3.87 s, and long full hit 1.51 s.

2026-05-09 threadgroup and latest rerun update:

- The accidental 64-thread C++ dispatch / 128-thread shader mismatch has been
  fixed by setting both direct decode dispatch constants to 128 threads. A
  64-thread experiment regressed raw decode; 128 is the current best broad
  setting from the quick sweep, but direct raw paged decode remains slower than
  MLX fast attention.
- `METAL_MICROBENCH_RESULTS_THREADGROUP_128.md` records the final raw
  microbench for this pass. B1/S2162 raw paged decode is 2.2729 ms versus
  0.4245 ms for `mx.fast.scaled_dot_product_attention`; B8/S2162 is 11.5096 ms
  versus 1.8197 ms. Native prefill is still far behind.
- A fresh same-window no-radix serving comparison is ahead of `/tmp/sglang-c490`:
  B1/B4/B8 total throughput 300.01/458.39/627.31 tok/s versus
  296.02/450.60/625.42 tok/s.
- A radix-enabled strict-idle probe is also ahead and coherent: short
  first/hit1/hit2 2.33/0.93/0.74 s, long partial hit 2.91 s, and long full hit
  0.94 s.

2026-05-09 raw online-decode update:

- A guarded online-softmax raw paged decode kernel now handles the common
  Qwen3 MLX shape (`head_dim=128`, `block_size=16`), with a float16 contiguous
  half4-vectorized route. Other shapes still use the generic paged decode
  kernel.
- The latest raw decode microbench is in
  `METAL_MICROBENCH_RESULTS_ONLINE_DECODE_VEC_CACHED_Q.md`. B1/B4/B8 S2162
  paged decode is now 0.6032/1.1472/1.7833 ms versus dense MLX fast
  0.4363/0.8923/1.4936 ms. This removes most of the raw decode gap and beats
  gathered-paged MLX on the paged layout, but it is still not a dense
  `mx.fast.scaled_dot_product_attention` win.
- A 128-thread online decode variant was tested and rejected
  (`METAL_MICROBENCH_RESULTS_ONLINE_DECODE_128_THREADS.md`); long decode
  regressed, so the online path stays at 256 threads.
- Native raw prefill is still the large unresolved kernel gap.

2026-05-09 raw online-prefill update:

- Added guarded float16/`head_dim=128`/`block_size=16` online-softmax prefill
  kernels for packed varlen and prefix-aware paged prefill. The paged route now
  covers radix-prefix reads from `block_tables` plus suffix K/V from the current
  prefill.
- Correctness coverage now includes the h128 packed varlen path and the
  h128 prefix-aware paged path; `python -m pytest sgl-kernel/tests/test_metal.py -q`
  passes 30 tests.
- Microbench evidence is in `METAL_MICROBENCH_RESULTS_ONLINE_PREFILL_PREFIX.md`
  and `METAL_MICROBENCH_RESULTS_ONLINE_PREFILL_PREFIX_RADIX.md`. The prefix-aware
  paged prefill no-prefix case improved substantially: B1/B4 S1024 is now
  about 31/121 ms instead of roughly 150/596 ms. The new 256-token radix-prefix
  case measures B1/B4 S1024 at 47.34/191.95 ms.
- This is still not an accepted final kernel: MLX fast dense is about
  5.24/18.60 ms for no-prefix B1/B4 S1024 and 6.51/23.16 ms for the 256-token
  prefix case. The remaining prefill gap requires a real Q-block tiled
  FlashAttention-style kernel; the current online kernels are still
  one-threadgroup-per-query scans.
- A 512-thread online group quick check was rejected
  (`METAL_MICROBENCH_RESULTS_ONLINE_512_THREADS_QUICK.md`): it regressed
  prefill and B1/B8 long decode, so online kernels stay at 256 threads.

2026-05-09 Steel prefill and radix decode update:

- Uniform no-prefix paged prefill now routes through the MLX Steel tiled
  attention bridge, and uniform radix-prefix paged prefill gathers prefix blocks
  into dense K/V before using the same Steel varlen bridge. The current
  radix-aware microbench is `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md`:
  B1/B4 S1024 no-prefix prefill is 3.5220/11.3406 ms versus dense MLX fast
  5.8653/19.6909 ms, and B1/B4 S1024 prefix256 prefill is 5.7009/18.2100 ms
  versus dense MLX fast prefix 6.3710/24.1820 ms.
- The benchmark harness now includes shuffled radix-style decode rows. Direct
  Metal paged decode is much faster than the gathered-paged MLX radix decode
  path for long contexts: shuffled radix B1/B4/B8 S2162 is
  0.6572/1.0986/1.8210 ms versus `mlx_fast_radix_gathered_paged`
  1.7799/3.8598/6.9880 ms.
- Dense MLX remains the raw synthetic decode ceiling for B4/B8 long decode:
  contiguous paged B1/B4/B8 S2162 is 0.6654/1.2013/1.7513 ms versus dense MLX
  fast 0.4673/1.0269/1.6542 ms. The direct radix/no-radix serving paths are
  therefore improved, but the final direct-paged-kernel acceptance gate remains
  open until dense-MLX decode is matched or beaten more broadly.
- A 192-thread online decode sweep was also tested and rejected
  (`METAL_MICROBENCH_RESULTS_ONLINE_DECODE_192_THREADS.md`): B4/B8 S2162
  contiguous paged decode regressed to 1.1775/1.9805 ms, so the online decode
  kernels stay at 256 threads.

2026-05-09 radix sync follow-up:

- Radix side-store sync now avoids stacking all layers into one large temporary
  before `set_kv_all_layers`. The current microbench records 28-layer sync
  improvements from 2.0597 to 1.3882 ms for 1 token, 1.7405 to 1.5058 ms for
  4 tokens, and 8.0237 to 6.4931 ms for 256 tokens.
- Same-window no-radix serving remains ahead of `/tmp/sglang-c490` under the
  current lower-memory run: B1/B4/B8 total throughput is
  236.52/380.94/521.31 tok/s versus baseline 229.27/364.63/485.59 tok/s.
- Radix serving remains coherent and ahead on the normal probes after the
  no-stack sync change: short first/hit1/hit2 2.56/1.14/0.99 s, long partial
  hit 3.42 s, and long full hit 1.13 s. The delayed short hit after 10 s idle
  still outliers at 4.65 s, so short-hit repeatability remains open.

2026-05-09 idle cleanup and rejected GQA2 decode update:

- The MLX worker now releases active per-request contiguous MLX state when the
  scheduler sends an idle batch, while preserving the radix-visible paged KV
  side-store. This avoids carrying finished request caches across idle gaps and
  lets later prefix-hit requests reuse the runner cache pool.
- A fresh radix-enabled probe on port 43460 stayed coherent with strict idle
  checks left at their default setting. Normal probes measured short
  first/hit1/hit2 1.78/1.13/0.77 s, long partial hit 2.91 s, and long full hit
  0.94 s. The delayed short hit after a 10 s idle gap improved from the prior
  4.65 s outlier to 1.84 s, with a second delayed repeat at 2.97 s. The outlier
  is reduced but not fully eliminated.
- A narrow dispatch gate for the existing fused GQA2 h128/b16 decode kernel was
  tested for batch-1 long contexts and rejected. The first focused run in
  `METAL_MICROBENCH_RESULTS_GQA2_BATCH1_LONG_GATE.md` looked better
  (B1/S2162 `metal_paged` 0.5841 ms), but the follow-up length sweep and
  rerun in `METAL_MICROBENCH_RESULTS_GQA2_BATCH1_LENGTH_SWEEP.md` and
  `METAL_MICROBENCH_RESULTS_GQA2_BATCH1_LONG_RERUN.md` did not hold the gain
  (`0.6601 ms` on the focused rerun, essentially the prior 0.6654 ms). The
  runtime selection gate was reverted.
- A current-only no-radix `bench_one_batch` regression check after these changes
  measured B1/B4/B8 total throughput 290.97/455.66/623.80 tok/s with
  `mem_fraction_static=0.6`. The earlier same-window baseline comparisons are
  still the source of truth for the "ahead of baseline" claim.
- A control probe that called the existing `/freeze_gc` endpoint before the
  delayed short hit did not remove the residual idle variance: the delayed
  short hit after 10 s idle measured 3.20 s. The remaining delayed-hit issue is
  therefore not explained by Python GC alone.

2026-05-09 thresholded radix paged-prefill update:

- Profiling showed the remaining delayed short-hit outlier is in the
  pool-backed prefix `self.model(...)` phase. The immediate `prefix=432/new=2`
  hit had `model_ms=17.94`, while the delayed 10 s idle hit had
  `model_ms=1881.68`; materialization and sync stayed small.
- A focused q=2/prefix432 microbench rejected a blanket paged-prefill switch
  for short hits: `metal_prefill_paged_prefix_432` was `0.9885 ms` versus
  `mlx_fast_dense_prefix_432` at `0.2701 ms`. The same prefix with query length
  434 did favor Metal (`2.7881 ms` vs `3.3111 ms`), so the runtime only uses
  paged Metal prefill for large radix prefix/suffix shapes.
- A direct same-run runner comparison with prefix 1392 and suffix 1781 measured
  the old pool-backed hit at `4.3153 s` and the thresholded paged Metal hit at
  `3.3812 s` with the same next token (`1.276x`). The no-radix guard still
  beats `/tmp/sglang-c490` in the same window: B1/B4/B8 total throughput
  `159.17/295.75/423.71` tok/s versus baseline `157.75/237.92/358.89` tok/s.
- A reused-cache growth bug found during this probe is fixed: contiguous caches
  now grow based on actual allocated buffer capacity, not only logical
  `max_seq_len`.
- Two follow-up experiments were rejected and documented. Forcing paged Metal
  prefill for short prefix hits after idle was slower than the pool-backed path
  in a direct runner (`0.4033 s` vs `0.2913 s` for prefix 368/new 12). A broad
  GQA2 decode dispatch rerun also regressed batched long decode, so the
  experimental C++ dispatch was reverted and the Metal extension rebuilt.

2026-05-09 q-shared decode rejection and threshold sweep update:

- A shared-query variant of the h128/b16 paged decode shader was tested and
  rejected in `METAL_MICROBENCH_RESULTS_DECODE_Q_SHARED.md`. It kept
  correctness but regressed important long decode cases: B4/S2162 moved to
  `1.3122 ms` and B8/S2162 to `2.1160 ms`, still behind dense MLX fast at
  `1.0204 ms` and `1.6015 ms`.
- A Qwen-shape two-pass long paged decode variant was also tested and rejected
  in `METAL_MICROBENCH_RESULTS_DECODE_2PASS.md`. A direct correctness probe
  matched dense MLX at B2/S1031 (`max_abs_diff=0.000122`), but the benchmark
  regressed the existing online kernel: B1/B4/B8 S2162 measured
  `0.7473/1.2245/2.2871 ms`.
- `METAL_MICROBENCH_RESULTS_RADIX_PREFILL_THRESHOLD_SWEEP.md` refined the
  radix prefill route. For single-request prefix hits, prefix256/new384 and
  prefix256/new512 still favored dense MLX, while prefix432/new384 and
  768-token suffixes favored paged Metal. The runtime now uses paged Metal
  prefill for `new_token_count >= 768`, or for `prefix_len >= 384` with
  `new_token_count >= 384`.

2026-05-09 all-layer radix sync update:

- Radix side-store sync now routes all-layer K/V writes through a guarded
  `sgl_kernel.metal.paged_kv_scatter_all_layers_unchecked` helper. This keeps the
  canonical per-layer paged cache arrays, but removes the MLX advanced-index
  assignment loop from `_sync_new_kv_to_pool` and `_sync_decode_kv_to_pool_batch`.
- `METAL_MICROBENCH_RESULTS_RADIX_SYNC_METAL_SCATTER.md` shows the new
  28-layer sync row beating the previous list path: 1-token median
  `0.8832 ms` vs `1.4097 ms`, 4-token `1.1346 ms` vs `1.4123 ms`, and
  256-token `9.5193 ms` vs `11.1655 ms`.
- Focused correctness remains green after the route change:
  `sgl-kernel/tests/test_metal.py` is now `36 passed`, and the MLX cache/model
  runner suite is now `85 passed`.
- No-radix serving remains guarded by the contiguous-cache path. Under current
  memory pressure, full B1/B4/B8 same-window runs are variance-sensitive, but
  the isolated B1 rerun is ahead of `/tmp/sglang-c490` (`165.82` vs `162.35`
  total tok/s), and the surrounding B4/B8 current rows are at or above the
  refreshed baseline rows.

## Decision

Build a serving-native MLX/Metal attention backend, not a compatibility patch on top of the current MLX cache path.

The existing Metal implementation remains useful as a correctness and performance reference while this backend is built, but it is not the target foundation because it:

- Is selected per-call through `SGLANG_MLX_USE_METAL_ATTENTION`.
- Opportunistically wraps the current MLX attention/cache path.
- Maintains both per-request `ContiguousKVCache` and flat shared `MlxKVPool` state.
- Uses primarily decode-oriented kernels.
- Exposes `flash_attn_varlen_func` and `flash_attn_with_kvcache` as partial compatibility subsets rather than a complete serving-native backend.

The target design should use the stronger serving architecture shape from `vllm-metal` while implementing the missing SGLang requirements, especially radix-cache support and true native prefill.

## Goals

- Replace the current MLX KV-cache design with a canonical block-paged Metal KV cache.
- Preserve SGLang radix-cache/prefix-sharing semantics on Apple Silicon.
- Route MLX model execution through a first-class Metal paged-attention backend without a per-call env fallback gate.
- Add Metal-native K/V cache write/scatter kernels.
- Add zero-copy paged decode attention over block tables.
- Add true varlen/prefill FlashAttention kernels instead of relying on MLX SDPA scaffolding.
- Provide clear backend startup capability checks and explicit unsupported-feature errors.
- Prove correctness with unit/kernel/E2E tests and prove usefulness with local Apple Silicon performance verification.

## Non-goals

- Do not expose a normal Torch/SRT `metal` attention backend unless the MLX runner actually uses it.
- Do not keep `SGLANG_MLX_USE_METAL_ATTENTION` as the final runtime control path.
- Do not preserve the per-request `ContiguousKVCache` + flat `MlxKVPool` split as the final serving cache design.
- Do not claim full CUDA FlashAttention-3 parity before the Metal backend implements the relevant semantics.

## Target architecture

### Paged KV cache

Use a canonical block-paged cache layout per layer:

```text
key_cache:   (num_blocks, block_size, num_kv_heads, head_dim)
value_cache: (num_blocks, block_size, num_kv_heads, head_dim)
```

The cache object should own:

- `num_layers`
- `num_blocks`
- `block_size`
- `num_kv_heads`
- `head_dim`
- `dtype`
- per-layer key/value MLX arrays

### Attention context

Before each MLX forward pass, `MlxModelRunner` should provide scheduler-native metadata to patched attention modules:

```text
is_prefill
slot_mapping
block_tables
context_lens
offsets
cu_seqlens
max_seqlen_q
max_seqlen_k
radix_prefix_lens
```

The exact Python type can evolve, but it should model the same serving information needed by paged attention rather than reconstructing padded/ragged K/V tensors inside attention.

### Radix-cache support

Radix-cache support is required, not optional.

The new backend must preserve prefix reuse by mapping matched prefix tokens to existing KV blocks instead of copying prefix K/V into a per-request contiguous cache.

Required behavior:

- During prefill, consume scheduler-provided prefix slot/block metadata for radix-cache hits.
- Represent reused prefixes in `block_tables` so attention can read them directly from paged Metal KV cache.
- Write new prompt/extend K/V into newly allocated slots/blocks using Metal-native cache-write kernels.
- Ensure full-cache-hit prefill/rerun-last-token behavior still has correct positions, offsets, and attention visibility.
- Sync newly generated decode K/V into radix-visible blocks immediately enough that later requests can reuse them.
- Keep request removal/cache release consistent with scheduler ownership; do not leak blocks or orphan radix entries.
- Add tests for no-prefix, partial-prefix-hit, full-prefix-hit, chunked-prefill, decode-after-prefix-hit, and request-removal cases.

## Progress tracker

Status values:

- `[ ]` not started
- `[/]` in progress
- `[x]` done

Previous progress items below document the implemented scaffolding, but the
performance completion claim is no longer valid after the 2026-05-09
microbenchmarks. Items that say `[x]` may be correct for wiring or correctness
coverage while still requiring the new performance acceptance gate.

1. `[x]` Define the new MLX/Metal paged attention architecture.
   - Specify the cache object, context object, ownership boundaries, and scheduler metadata flow.
   - Decide how `MlxModelRunner`, radix cache, request-to-token mapping, and Metal KV blocks interact.
   - Define startup capability checks for Apple Silicon + MLX + Metal extension availability.
   - Define the final backend selection behavior that replaces the env-gated per-call fallback.
   - Current status: The serving-native architecture is implemented around scheduler-provided `PagedAttentionContext`, canonical `MlxPagedKVCache`, eager Apple Silicon/`sgl_kernel.metal` capability checks, and a final MLX path that no longer uses the removed per-call `SGLANG_MLX_USE_METAL_ATTENTION` gate.

2. `[x]` Implement the canonical block-paged Metal KV cache.
   - Add MLX arrays with layout `(num_blocks, block_size, num_kv_heads, head_dim)` per layer.
   - Replace or supersede the current flat `MlxKVPool` for the MLX/Metal backend.
   - Add allocation/free/reset helpers compatible with SGLang scheduler ownership.
   - Track block size, slot mapping, dtype, head count, and head dimension consistently.
   - Current status: `MlxPagedKVCache` owns per-layer `(num_blocks, block_size, num_kv_heads, head_dim)` K/V MLX arrays, and `MlxModelRunner` initializes it from scheduler pool capacity while relying on SGLang scheduler ownership for slot allocation, reuse, and release.

3. `[x]` Wire radix-cache prefix reuse into the paged cache design.
   - Convert radix prefix hits into block tables and prefix lengths.
   - Avoid copying prefix K/V into per-request contiguous caches.
   - Support partial prefix hits, full cache hits, chunked prefill, extend, decode, and removal.
   - Ensure newly written K/V becomes visible to future radix-cache matches.
   - Current status: Superseded by the 2026-05-09 hybrid stopgap. `MlxModelRunner` still builds scheduler-derived paged metadata and keeps new K/V visible in `MlxPagedKVCache`, but active request compute uses contiguous MLX caches because the raw paged kernels do not yet beat MLX fast attention.

4. `[x]` Add Metal-native K/V cache write and scatter kernels.
   - Implement reshape/cache-write kernels for prompt, extend, and decode tokens.
   - Support slot mappings that cross block boundaries.
   - Validate dtype, head_dim, block_size, and head layout constraints explicitly.
   - Add kernel tests comparing paged cache contents against MLX reference writes.

5. `[x]` Add zero-copy paged decode attention kernels.
   - Read K/V directly from paged cache using `block_tables`, `context_lens`, and offsets.
   - Support GQA/MQA head mapping.
   - Use stable fp32 softmax accumulation where practical.
   - Avoid constructing dense or ragged K/V tensors per request.
   - Add correctness tests across batch sizes, sequence lengths, prefix hits, and mixed context lengths.
   - Current status: Metal paged decode binding/API/wrapper route builds and passes direct MLX verification for mixed lengths, block-boundary page traversal, GQA mapping, invalid metadata rejection, and paged scatter. The final MLX serving path calls `decode_attention_paged` with scheduler-derived block tables/context lengths and does not construct dense or ragged K/V tensors per request.

6. `[x]` Add true varlen/prefill FlashAttention Metal kernels.
   - Support packed prefill using `cu_seqlens_q` / `cu_seqlens_k`.
   - Support causal block-diagonal attention for packed requests.
   - Read reused prefix K/V through block tables and newly projected K/V through the write path.
   - Avoid MLX SDPA fallback in the final backend.
   - Add no-prefix, partial-prefix, full-prefix, and multi-request packed-prefill tests.
   - Current status: Native packed varlen and paged prefill Metal bindings/APIs/wrapper integration build and pass direct MLX verification for causal packed requests, unsupported-option rejection, invalid metadata rejection, sparse visibility, and prefix/block-table-aware paged prefill through `prefill_attention_paged`. The final MLX serving path now uses native paged prefill rather than MLX SDPA scaffolding.

7. `[x]` Rework `MlxModelRunner` around the new backend.
   - Pass paged attention context directly before model forward.
   - Remove dependence on per-request `ContiguousKVCache` for the final Metal backend path.
   - Preserve prefill, extend, decode, flush, remove, and clear semantics.
   - Ensure radix-cache scheduler metadata is the source of truth for block visibility.
   - Current status: Superseded by the 2026-05-09 hybrid stopgap. `MlxModelRunner` performs eager Apple Silicon/Metal API startup checks and keeps scheduler-visible K/V in `MlxPagedKVCache`, but it intentionally maintains request-local contiguous caches for active compute until direct paged kernels pass the performance gate.

8. `[x]` Replace `MLXAttentionWrapper` routing with first-class paged attention.
   - Use the paged context to choose prefill vs decode kernels.
   - Project Q/K/V, apply q/k norm and RoPE, write K/V, then invoke Metal attention.
   - Remove opportunistic per-call import/fallback behavior from the final backend path.
   - Keep unsupported features as explicit startup or call-time errors.
   - Current status: `MLXAttentionWrapper` dispatches active paged contexts to native paged prefill/decode paths, validates required metadata before cache mutation, scatters K/V through Metal into the paged cache, marks scatter completion only after successful native writes, ignores legacy cache objects in the paged path, delegates to the inner MLX attention only when no paged context is active, and relies on runner startup checks plus call-time validation for unsupported configurations.

9. `[x]` Update public `sgl_kernel.metal` APIs for the real backend.
   - Add native paged cache-write, paged decode, and varlen prefill bindings.
   - Keep compatibility wrappers only if they remain useful for tests or external users.
   - Clearly document unsupported CUDA FlashAttention features.
   - Current status: Native paged cache write/scatter, paged decode, paged prefill, packed varlen, and compatibility decode wrappers are exported from `sgl_kernel.metal`, bound through `sgl_kernel._metal`, covered by focused Metal tests, and documented in `csrc/metal/README.md` with explicit unsupported-option limits.

10. `[x]` Add unit, kernel, and integration tests.
    - Test paged KV cache allocation/write/read/free behavior.
    - Test radix-cache prefix reuse with block tables.
    - Test decode correctness against MLX reference attention.
    - Test packed prefill correctness against MLX reference attention.
    - Test error handling for unsupported shapes/dtypes/features.
    - Update existing MLX routing tests that currently assume env-gated fallback behavior.
    - Current status: Focused MLX unit coverage verifies paged context normalization/scatter tracking, wrapper paged prefill/decode routing, required metadata validation, legacy-cache ignoring in paged attention, runner paged prefill/extend/decode context construction, startup capability checks including import-success/native-extension-missing failure, direct native scatter tracking, token-state-only request cleanup, batched radix sync, and small-prefix recompute routing. The focused `unittest` run for `test_model_runner`, `test_attention_wrapper`, `test_paged_context`, and `test_paged_cache` passed 76 tests in the existing `sglang-mlx` environment; Metal kernel coverage for paged scatter, paged decode, paged prefill, packed varlen, compatibility wrappers, invalid metadata, unsupported options, and dtype behavior is in `sgl-kernel/tests/test_metal.py`.
    - Latest status: `python -m pytest sgl-kernel/tests/test_metal.py -q`
      passes 34 Metal tests in `sglang-mlx`, and the focused MLX wrapper/cache/
      model-runner subset passes 84 tests after the lazy decode tiny-context
      routing guard.

11. `[/]` Add Apple Silicon E2E and performance verification.
    - Run local MLX server golden-path generation with the new backend.
    - Compare no-prefix, partial-prefix, and full-prefix radix-cache workloads.
    - Measure prefill, extend, and decode throughput/latency for batch sizes greater than 1.
    - Compare against the current MLX fallback and the existing guarded Metal proof-of-concept.
    - Record results in a benchmark document before declaring the redesign complete.
    - Current status: Reopened on 2026-05-09. `METAL_MICROBENCH_RESULTS_SERVING.md` showed raw paged Metal decode/prefill behind MLX fast attention by large margins. `METAL_MICROBENCH_RESULTS_GATHERED_DECODE.md` shows a 4096-token decode score cache improves long decode, and gathered-paged MLX fast decode can help some cases, but neither is an accepted paged-kernel win. `METAL_SERVING_BENCH_RESULTS_2026_05_09.md` now shows the no-radix contiguous-cache guard and hybrid radix cache-hit probes ahead of refreshed baseline checks, including the long partial-prefix-hit probe. Strict idle accounting is fixed for the MLX page-aligned pool path by aligning scheduler-visible pool capacity to `page_size`.
    - Latest status: The guarded online-softmax decode kernel has current
      microbench evidence in `METAL_MICROBENCH_RESULTS_ONLINE_DECODE_VEC_CACHED_Q.md`;
      raw decode is close but still behind dense MLX fast attention. Guarded
      online prefill kernels now reduce the raw paged prefill cliff and cover
      radix-prefix reads, but B1/B4 S1024 remains several times slower than MLX
      fast attention. The hybrid serving path is still ahead in fresh
      same-window no-radix and radix probes.
    - Continuation status: thresholded paged Metal prefill is now used only for
      large radix prefix/suffix shapes, based on
      `METAL_MICROBENCH_RESULTS_RADIX_SHORT_HIT_Q2.md` and
      `METAL_MICROBENCH_RESULTS_RADIX_PREFILL_THRESHOLD_SWEEP.md`.
      A direct runner comparison in
      `METAL_SERVING_DIRECT_RADIX_PREFILL_THRESHOLD_BENCH.md` shows a 1.276x
      win for prefix 1392/new 1781, while short q=2 hits stay on MLX dense
      prefix attention because raw Metal is slower there.
    - Continuation status: `METAL_MICROBENCH_RESULTS_DECODE_LAZY_MLX_KERNEL.md`
      adds a lazy `mx.fast.metal_kernel` h128/block16 decode path that avoids
      the wrapper output-zero/materialization overhead. It improves the direct
      paged decode path versus the native wrapper for shuffled radix-style
      block tables and most contiguous rows. A direct block-table indexing fix
      also covers tiny batch-1 radix tables that MLX compiles as constant
      address-space inputs. A focused short-context rerun in
      `METAL_MICROBENCH_RESULTS_B1_SHORT_DECODE_THRESHOLD_RERUN.md` moves the
      batch-1 runtime threshold from `>=32` tokens to `>16` tokens: B1/S16
      still stays on gather+MLX (`0.2873 ms` versus `0.3097 ms` lazy radix),
      while B1/S24 and B1/S32 use lazy direct radix decode (`0.2754/0.2786 ms`
      versus `0.3482/0.3201 ms` gather+MLX). It still does not beat dense
      `mx.fast.scaled_dot_product_attention` for every contiguous no-radix
      shape. Keep the serving hybrid/contiguous guard in place.
    - Continuation status: no-radix serving now keeps attention wrappers out of
      prefill and single-request decode, installing the lean batched wrapper
      only for batch decode bursts. A same-window `bench_one_batch` check shows
      B1/B4/B8 total throughput `184.62/319.36/447.54 tok/s` versus
      `/tmp/sglang-c490` `170.96/281.33/384.61 tok/s`.
    - Continuation status: a later no-radix rerun after the lazy decode
      address-space fix was noisy but did not show a stable regression. The
      full B1/B4/B8 sweep measured current `192.07/315.98/438.43 tok/s`
      versus baseline `187.10/302.59/454.49 tok/s`; isolated B8 reruns then
      measured current `464.46 tok/s` versus baseline `445.43 tok/s`.
    - Continuation status: `METAL_MICROBENCH_RESULTS_CONTIGUOUS_DECODE_SPECIALIZATION.md`
      rejects a synthetic contiguous-page decode shortcut. Replacing the
      block-table read with a direct contiguous block formula is correct but
      only wins selected rows; reshaping paged K/V into dense MLX fast attention
      is consistently slower. The remaining raw contiguous decode gap needs a
      deeper algorithmic change.
    - Continuation status: `METAL_MICROBENCH_RESULTS_DENSE_DECODE_LAZY.md`
      rejects a dense-cache lazy Metal decode kernel for no-radix batched
      decode. It is correct and wins selected rows, but threadgroup sweeps stay
      mixed, so the no-radix path should keep dense MLX fast attention until a
      kernel clears the full target matrix.
    - Continuation status: `METAL_MICROBENCH_RESULTS_PAGED_SDPA_VECTOR_DECODE.md`
      rejects a paged adaptation of MLX's q_len=1 `sdpa_vector` decode
      algorithm. It is correct and helps selected shuffled rows, but the larger
      threadgroup/reduction shape does not beat dense MLX or the accepted lazy
      paged decode path broadly.
    - Continuation status: `METAL_MICROBENCH_RESULTS_PAGED_2PASS_VECTOR_DECODE.md`
      rejects a two-pass paged vector split. It increases B1 parallelism and is
      correct, but the added launch/partial-write cost does not clear the long
      decode shape matrix.
    - Completion audit: `METAL_COMPLETION_AUDIT_2026_05_09.md` maps the
      objective to concrete correctness, serving, raw-kernel, and
      rejected-direction evidence. The redesign remains open because raw Metal
      decode/prefill has not beaten dense MLX fast attention across the full
      target matrix.
    - Harness rerun: `METAL_MICROBENCH_RESULTS_DECODE_LAZY_HARNESS_RERUN.md`
      was generated by the updated `bench_metal_attention_micro.py` and
      reproduces the accepted lazy decode rows alongside dense MLX fast,
      native paged, gathered-paged, and radix-shuffled comparisons.
    - Continuation status: `METAL_MICROBENCH_RESULTS_DECODE_REGSCORE_SWEEP.md`
      and `METAL_MICROBENCH_RESULTS_DECODE_FAST_EXP_COMPARE.md` benchmark a
      register-score lazy variant against the accepted shared-score lazy
      kernel, dense MLX fast attention, and shuffled radix gather+MLX. The
      variant is correct but mixed, so it remains a harness-only comparison and
      is not used by runtime routing.
    - Continuation status: `METAL_MICROBENCH_RESULTS_FUSED_KV_DECODE.md`
      rejects a native fused scatter+decode h128/block16 kernel. The kernel is
      correct and writes the current token K/V slot, but the native fused rows
      are slower than the accepted lazy decode on almost every contiguous and
      shuffled radix shape tested.
    - Continuation status: native all-layer radix side-store sync now casts
      per-layer K/V to the destination cache dtype before dispatch, fixing the
      `bfloat16 -> float16` serving crash found on port 43465. The mixed-dtype
      sync spot check in
      `METAL_MICROBENCH_RESULTS_RADIX_SYNC_METAL_SCATTER.md` still favors the
      native route over MLX indexed list assignment. The radix probe remains
      coherent (`2.58/1.17/0.78 s` short first/hit1/hit2, `2.88 s` long
      partial hit, `0.95 s` long full hit), but the delayed short hit after
      10 s idle still outliers at `3.09 s`. Same-window no-radix checks remain
      variance-sensitive but non-regressed in isolated B8 reruns.
    - Rejected continuation: an idle-gap materialized-prefix mitigation reduced
      a narrowed profiled short-only delayed hit, but regressed the normal
      short/long radix probe delayed hit to `3.40 s` after a long full-prefix
      request. Do not retry that heuristic without a way to predict the
      sequence-sensitive regression.
    - Continuation status: the no-radix equal-length batched contiguous-cache
      route is worth keeping for multi-request decode. Final B4/B8 totals were
      `375.76/496.65 tok/s` versus same-window baseline
      `305.29/421.64 tok/s`. B1 remains open (`184.92 tok/s` current versus
      `197.93 tok/s` baseline in the same comparison). A fresh radix comparison
      found the current default `page_size=16` coherent but behind the
      token-granular `c4903e67` baseline on most short/long cache-hit rows,
      because p16 only reuses 432/2160 cached tokens for the 434/2162-token
      probes. Current `--page-size 1` restores the cached-token counts and
      helps short hits, but still does not broadly beat the baseline. The
      pure-MLX block-size-1 side-store/dtype experiment was rejected. A
      decoupled token-granular scheduler plus block16 MLX side-store experiment
      was also rejected because it restored cached-token counts but regressed
      long partial and delayed short latency.

12. `[/]` Remove the old guarded implementation after replacement is proven.
    - Delete final dependence on the removed per-call MLX Metal attention flag.
    - Remove obsolete `ContiguousKVCache`/flat-pool bridging code from the Metal backend path.
    - Remove compatibility scaffolding that is no longer part of the final design.
    - Keep only documented fallback behavior that is selected at backend startup, not per attention call.
    - Current status: active runtime code no longer references the old per-call MLX Metal attention or validation flags, but the no-radix contiguous-cache path is intentionally restored as a regression guard. Do not remove it until the paged Metal kernels and serving integration beat the refreshed MLX baseline.

2026-05-09 post-interruption update:

- The accepted no-radix route is the wrapper-based contiguous-cache decode path,
  not the equal-length `BatchedContiguousKVCache` shortcut. The shortcut was
  rejected after direct B4/B8 comparisons showed the wrapper route faster.
- Radix defaults now use `page_size=1`, immediate prompt-side side-store sync,
  and deferred decode-token side-store sync. Deferring prompt sync was rejected
  because it shifted K/V work into later requests and regressed delayed reuse.
- A narrow idle-gap radix guard now routes short prefix hits through paged Metal
  prefill only when explicitly enabled after a forward-idle gap. This reduced
  one delayed short hit from `4.21 s` before the guard to `1.31 s` with coherent
  output, while normal short hits stay on the faster pool-backed MLX route. A
  later default-on attempt was rejected because the no-env profiled run regressed
  delayed short to `2.87 s`; keep this route env-gated.
- The latest no-radix guard measured B1/B4/B8 total throughput
  `189.37/298.17/413.02 tok/s`, with an isolated B4 rerun at `304.27 tok/s`.
  The nearest same-window baseline remains variance-sensitive
  (`189.29/302.88/408.02 tok/s`), so no-radix is ahead on B1/B8 and on the
  isolated B4 rerun, but not conclusively ahead on every full-sweep row.
- The latest radix p1 probe measured short first/hit1/hit2
  `1.71/0.93/0.68 s`, long partial/full `2.77/0.89 s`, and delayed short
  `1.31 s`. It improves the important cache-hit and delayed rows versus the
  refreshed p1 baseline, but cold first and long partial remain open.
- Two long-partial alternatives were rejected: forcing paged Metal prefill for
  `prefix=433/new=1729` did not return promptly in serving, and forcing the
  pool-backed prefix path regressed that row to `4.38 s`. Dense recompute stays
  the accepted route for that shape.

2026-05-10 p1 flat-cache rejection:

- `METAL_MICROBENCH_RESULTS_P1_FLAT_CACHE_REJECTION.md` records an
  authoritative flat p1 `MlxPagedKVCache` experiment. The native Metal wrappers
  were extended to accept 3-D p1 caches correctly, but making the runtime cache
  flat regressed serving rows that matter: first flat run short hits were
  `1.20/1.02 s`, long partial was `4.10 s`, and long full was `1.13 s`; after a
  cache flush, delayed short regressed to `3.42 s` and long partial was still
  `3.30 s`. Runtime p1 storage was returned to the accepted 4-D layout.
- A lazy radix attention-wrapper installation experiment was also rejected.
  After `/flush_cache` it improved normal short hits (`0.73/0.71 s`), long
  partial (`2.65 s`), and long full (`0.85 s`), but the delayed short hit
  regressed to `3.79 s`. The radix path therefore keeps the accepted
  always-installed paged wrapper so the post-idle paged-prefill guard does not
  pay a first-use wrapper/graph cost.

2026-05-10 p1 prefix Steel bridge update:

- `METAL_MICROBENCH_RESULTS_P1_PREFIX_STEEL_BRIDGE.md` records the focused
  p1 prefix-prefill gap. The p1 `prefix=433/new=1729` raw paged-prefix prefill
  path fell through to the generic kernel and measured `11302.385 ms` versus
  dense MLX fast at `38.652 ms`. Widening the dense-prefix Steel bridge to p1
  caches, gated to suffixes of at least 128 query tokens, improves that raw row
  to `18.417 ms` versus dense MLX fast at `34.112 ms`.
- The serving route still rejected forcing paged prefill for the long partial
  row: it remained coherent but regressed to `4.1032 s`, with timing showing
  the loss in model/eval overhead (`model_ms=2316.65`, `eval_ms=877.96`), not
  in raw attention alone. Runtime routing was returned to the accepted dense
  recompute path for `prefix=433/new=1729`.
- Final post-revert probe stayed coherent: short first/hit1/hit2
  `1.65/0.96/0.70 s`, long partial/full `2.92/0.93 s`, and delayed short
  `2.45 s`. No-radix B1/B4/B8 totals with the server stopped were
  `193.32/302.96/419.86 tok/s`, slightly ahead of the nearest saved baseline
  rows `189.29/302.88/408.02 tok/s`.

2026-05-10 radix warmup continuation:

- Added a startup warmup for the radix Metal all-layer scatter and q=1 paged
  prefix prefill primitives. This moved the first all-layer scatter
  materialization out of the first generated request: the built-in 6-token
  serving warmup sync dropped from about `895 ms` to `15 ms` in the profiled
  runs. The retained change is guarded by
  `SGLANG_MLX_DISABLE_RADIX_KERNEL_WARMUP=1` for local diagnosis.
- The clean post-warmup radix probe stayed coherent with default `page_size=1`:
  short first/hit1/hit2 `2.02/0.98/0.69 s`, long partial/full
  `2.70/0.93 s`, and delayed short `1.66 s`. Against the refreshed
  `/tmp/sglang-c490` baseline (`1.50/2.97/0.77 s`, long
  `2.61/0.89 s`, delayed `2.35 s`), the current path is ahead on cache-hit and
  delayed rows but still not ahead on cold first, long partial, or long full.
- Two follow-up serving experiments were rejected. Making prompt/decode
  side-store scatter non-blocking (`eager=False`) reduced the logged sync slice
  but pushed cost into later eval/decode and regressed cold/delayed rows. Forcing
  combined token+cache eval for no-prefix radix prefill improved only the cold
  row and regressed short hits, long partial, long full, and delayed.
- Env-gated decode timing (`SGLANG_MLX_PROFILE_TIMING=1`) shows the remaining
  short cold gap is prefill-eval dominated. Radix single-request decode for the
  434-token prompt is roughly `40-55 ms/token`, comparable to the no-radix B1
  decode profile, while the radix dense no-prefix prefill eval can be about
  `1.2 s`. A warmed retest of temporary radix-wrapper removal for non-paged
  prefill is still rejected: it improved long full/delayed rows, but regressed
  cold first and long partial.
- Current no-radix rerun remains variance-sensitive but acceptable as a guard:
  full-sweep B1/B4/B8 totals were `177.72/272.93/389.12 tok/s`; isolated B4/B8
  rerun recovered to `285.19/392.56 tok/s` versus the saved same-window
  baseline `281.07/378.43 tok/s`. The focused MLX suite reports `92 passed` and
  `sgl-kernel/tests/test_metal.py` reports `39 passed`.

2026-05-10 final serving-direction update:

- The accepted no-radix path now uses a tighter contiguous-cache allocation
  policy (`32` decode slack tokens and `32`-token capacity alignment) plus
  full-state cache evaluation for single-request decode. The latest paired
  no-radix `bench_one_batch --disable-radix-cache --batch-size 1 4 8
  --input-len 256 --output-len 32` run is ahead of `/tmp/sglang-c490` on every
  row: B1/B4/B8 total throughput `221.96/344.74/492.27 tok/s` versus baseline
  `219.56/334.45/478.80 tok/s`.
- The accepted radix path remains the hybrid path: contiguous MLX cache compute
  for active requests, token-granular `MlxPagedKVCache` as the radix-visible
  side-store, batched all-layer side-store sync, startup warmup, and selective
  full-state evaluation. The latest same-window live refresh supersedes the
  earlier paired-median optimism: current is clearly ahead on `short_hit1`
  (`0.693 s` default, `0.563 s` profiled versus `2.119 s` baseline) and delayed
  short (`2.225/2.642 s` versus `3.242 s`), but `short_hit2` is behind
  (`0.768/0.668 s` versus `0.627 s`) and `long_partial` is mixed
  (`2.445/2.344 s` versus `2.420 s`). A new long-context single-request decode
  threshold improves `long_full` to `0.776 s` in the confirmation run, but the
  radix serving gate remains open.
- Rejected in this pass: idle recompute fallback, wider small-prefix recompute,
  cached flat p1 views, p1 float16 side-store storage, non-blocking side-store
  scatter, blanket no-compact radix prefill eval, and forced paged-prefill
  routing for the `prefix=433/new=1729` long-partial row. Also rejected:
  tightening radix request-local contiguous cache capacity to prompt plus slack;
  it improved delayed short but regressed long partial and kept short hit2
  behind baseline. The diagnostic
  `SGLANG_MLX_RADIX_P1_FLOAT16_KV` knob was removed after the p1 float16 run
  regressed long partial and delayed short latency.
- Direction is unchanged: keep the hybrid serving path as the current
  performance stopgap, but do not declare the direct paged Metal FlashAttention
  backend complete until raw paged decode/prefill kernels beat MLX fast
  attention on the target shapes.

2026-05-10 dense/raw continuation:

- A benchmark-only lazy dense h128 decode kernel was added to map the remaining
  raw direct-kernel gap. It passes the Metal reference test and is useful for
  controlled sweeps, but the retained `/tmp/sglang_raw_micro_dense_h128_20260510.md`
  run still leaves dense MLX ahead on most contiguous decode rows. The raw gate
  remains open.
- A dense contiguous GQA2 h128 probe was added as another benchmark-only
  variant. It computes the two query heads sharing one KV head in the same
  threadgroup, but the retained `/tmp/sglang_raw_micro_dense_gqa2_20260510.md`
  run lost all B1/B4/B8 S288/S2162 dense rows versus MLX fast. Keep it as a
  rejected direction, not a runtime route.
- Making the idle paged-prefill guard default-on was also rejected. The opt-in
  run measured delayed short at `1.83 s`, but the no-env default-on confirmation
  measured delayed short at `2.87 s` with `idle_paged=True`, so the route remains
  behind `SGLANG_MLX_ENABLE_IDLE_PAGED_PREFILL=1`.

2026-05-10 p1 lazy decode update:

- Added a lazy `mx.fast.metal_kernel` p1 paged decode variant for
  float16/head_dim=128 radix-style block tables. Correctness is covered by
  `test_metal_decode_attention_paged_p1_lazy_h128_matches_reference`.
- `METAL_MICROBENCH_RESULTS_P1_LAZY_DECODE.md` shows the direct p1 radix decode
  row beating p1 gather+MLX on the retained B1/B4/B8 S288/S2162 matrix. Example
  rows: B1/S2162 improves from `1.0557 ms` gather+MLX to `0.4783 ms`, B4/S2162
  from `4.7432 ms` to `1.0888 ms`, and B8/S2162 from `12.3570 ms` to
  `1.5313 ms`.
- Short single-request rows are thresholded: B1/S16 stays on gather+MLX because
  p1 lazy measured `0.4291 ms` versus `0.3155 ms`; B1/S24 and above use p1
  lazy, and all measured B4/B8 short rows use p1 lazy.
- This is a raw/direct-paged radix-kernel improvement, not a serving completion
  claim. The default radix serving loop remains the hybrid contiguous-cache
  compute path, and the direct paged decode context is not enabled as the
  default runner path.
- 2026-05-10 bf16 continuation: the scalar p1 lazy decode kernel adds
  `bfloat16` coverage and improves raw p1/radix decode against p1 gather+MLX
  on the long matrix (`0.5074/0.8785/1.6477 ms` for B1/B4/B8 S2162 versus
  `1.0749/5.0674/10.0106 ms`). Serving still rejects the route as a default:
  forcing `SGLANG_MLX_ENABLE_BF16_PAGED_DECODE=1` regressed the radix probe
  (`short_hit1/2=0.916/1.020 s`, `long_partial=2.830 s`,
  `delayed_short=2.046 s`) versus the guarded hybrid default
  (`0.743/0.835 s`, `2.554 s`, `1.918 s`). Keep bf16 paged decode opt-in.
- Rejected after that bf16 update: using p1 decode for one-token radix prefill,
  materializing p1 prefixes before one-token prefix-hit prefill, and a 128-thread
  bf16 p1 scratch decode variant. Each helped selected micro or normal hit rows
  but regressed the serving-relevant delayed short and/or B1 long rows.
- Latest same-window radix refresh keeps the E2E gate open. Current is ahead on
  `short_hit1` and delayed short versus `/tmp/sglang_baseline_radix_live_53910_20260510.jsonl`,
  but `short_hit2` and `long_partial` are not robust wins. The only retained
  routing tweak from this pass is `_RADIX_FULL_STATE_DECODE_MIN_TOKENS=1024`,
  which improves the `long_full` confirmation row while preserving compact eval
  for short single-request decode.
- Rejected after that refresh: reducing radix request-local contiguous cache
  capacity below `_max_seq_len`. The probe
  `/tmp/sglang_current_radix_tight_cache_41415_20260510.jsonl` improved delayed
  short to `1.737 s`, but regressed `long_partial` to `2.839 s` and left
  `short_hit2` behind at `0.787 s`.
- Rejected after that refresh: raising `_RADIX_FULL_STATE_PREFILL_MIN_NEW_TOKENS`
  to keep the long-partial suffix on compact eval. The profiled probe
  `/tmp/sglang_current_radix_compact_long_prefill_48434_20260510.jsonl`
  measured `long_partial=2.461 s`, `short_hit2=0.796 s`, and
  `long_full=0.892 s`, so it did not close the baseline gap.
- Rejected after that refresh: eager radix stale-request cleanup before every
  extend. `/tmp/sglang_current_radix_eager_cleanup_36139_20260510.jsonl`
  improved delayed short but regressed `short_hit2=0.780 s` and
  `long_partial=2.557 s`, so stale cleanup remains on decode/idle boundaries.
- Rejected after that refresh: full-state eval for all single-request radix
  fallback decode. `/tmp/sglang_current_radix_fullstate_all_decode_38349_20260510.jsonl`
  improved delayed short and `long_full`, but regressed `short_hit1` and
  `long_partial`, while `short_hit2` remained behind baseline. Keep the
  `1024`-token long-context decode threshold.
- Rejected continuation: active short-prefix contiguous-cache reuse. The gated
  best run
  `/tmp/sglang_current_radix_active_prefix_small_suffix_33031_20260510.jsonl`
  improved `delayed_short=1.763 s` and `long_full=0.735 s`, but still missed
  baseline on `short_hit2=0.736 s` and `long_partial=2.442 s`; the no-profile
  confirmation left `long_partial=2.476 s`. This does not close radix serving,
  so the route was reverted.
- Rejected continuation: raw-attention fallback for radix single-request
  fallback compute. `/tmp/sglang_current_radix_raw_fallback_active_prefix_profile_41598_20260510.jsonl`
  regressed `short_hit2=0.804 s` and `long_partial=2.649 s`.
- Rejected continuation: default fp16 p1 side-store and bf16 generic paged
  prefill. The fp16 side-store regressed all serving rows
  (`long_partial=5.103 s`), and the bf16 paged-prefill route timed out on
  `long_partial` in
  `/tmp/sglang_current_radix_bf16_prefill_active_prefix_profile_46506_20260510.jsonl`
  despite passing a small direct reference test. Both were reverted, including
  rebuilding the metallib back to the retained source.

## Acceptance gates

- The MLX/Metal backend supports radix-cache prefix reuse without copying prefixes into per-request contiguous caches.
- Decode reads directly from paged KV cache using block tables.
- Prefill uses native Metal varlen FlashAttention kernels in the final backend path.
- Default Apple Silicon MLX backend selection is clear and does not rely on `SGLANG_MLX_USE_METAL_ATTENTION` per-call gating.
- Unsupported features fail clearly instead of silently producing different semantics.
- Unit/kernel tests pass for cache writes, paged decode, varlen prefill, and radix-cache cases.
- Local Apple Silicon E2E generation works for no-prefix, partial-prefix, and full-prefix cache-hit workloads.
- Performance beats the current MLX fallback and the existing guarded Metal
  proof-of-concept on all targeted serving workloads before the redesign is
  considered complete. Lazy direct paged decode improves the radix-style paged
  layout versus gathered MLX, and native all-layer scatter plus startup warmup
  improves radix side-store sync, but direct raw Metal kernels are still not
  accepted as faster than MLX fast attention for all contiguous decode/prefill
  shapes.
- Latest 2026-05-10 status update: no-radix serving remains ahead of the
  refreshed baseline on B1/B4/B8 in the retained guard sweep, but radix serving
  is not yet a complete win. The same-window live radix refresh has current
  ahead on `short_hit1` and delayed short, mixed/on-par on `short_first`,
  `long_partial`, and `long_full`, and behind on `short_hit2`. A retained
  single-request long-context decode threshold improved `long_full` to
  `0.776 s` versus the current default `0.873 s` and the same-window baseline
  `0.784 s`, but it does not close the short-hit and long-partial gaps.
- The p1 lazy decode continuation improves direct p1/radix decode against the
  p1 gather+MLX path, but it does not close the default-serving or dense-MLX
  raw gates by itself.
- 2026-05-11 serving completion update: the no-radix and radix-cache serving
  baseline gates are now met by the retained hybrid route. No-radix clears the
  previous MLX baseline on paired B1 median (`219.26` versus `218.12 tok/s`)
  and on the final B4/B8 audit rows (`345.65/492.00` versus
  `329.47/460.29 tok/s`). Radix serving medians over the accepted current runs
  clear the two-run baseline on every probe: `short_first 1.133 < 1.153 s`,
  `short_hit1 0.686 < 2.227 s`, `short_hit2 0.629 < 0.635 s`,
  `long_partial 2.389 < 2.601 s`, `long_full 0.771 < 0.862 s`, and
  `delayed_short 1.861 < 1.992 s`. The detailed files and rejected final
  experiments are recorded in
  `METAL_MICROBENCH_RESULTS_CURRENT_RADIX.md`.
- Raw kernel microbenchmarks compare against `mx.fast.scaled_dot_product_attention`
  for decode and prefill shapes before the serving path is declared faster.
- `bench_one_batch --batch-size 1 4 8 --input-len 256 --output-len 32 --warmups 1`
  must beat a same-machine refreshed baseline worktree, not only an older saved
  benchmark snapshot.

## Remaining Kernel Direction

The serving-baseline gap is closed by the hybrid route. The remaining raw-kernel
gap is not wrapper overhead or a simple block-table dispatch choice. The
following decode directions have been benchmarked and should not be retried
without a materially different idea:

- q-shared/GQA2 dispatch.
- 128/192/512 threadgroup sweeps around the accepted online kernel.
- Native preallocated-output wrapper lower-bound work.
- Contiguous block-table shortcut.
- Reshape paged K/V into dense MLX fast attention.
- Dense-cache lazy Metal decode.
- Paged one-pass `sdpa_vector` adaptation.
- Paged two-pass vector split.
- Register-score in-block score exchange as a broad replacement for the
  accepted shared-score lazy decode kernel.
- Dense contiguous GQA2 h128 decode.
- Native fused scatter+decode h128/block16 kernel that writes current-token K/V
  while computing decode.
- Idle-gap materialized-prefix heuristic for short radix hits.
- Prompt-side side-store sync deferral.
- Unchecked full-block pool-backed gather.
- Paged-precedence and pool-backed long-partial routes for
  `prefix=433/new=1729`.
- Authoritative flat p1 `MlxPagedKVCache` storage. It improves selected
  synthetic p1 sync/materialization rows but regresses long partial and delayed
  short serving probes.
- Lazy installing/removing the radix attention wrapper around only paged
  prefill or batched decode. It helps selected normal rows after a cache flush
  but regresses the delayed short hit.
- Forcing paged prefill to override the small-prefix dense-recompute guard for
  `prefix=433/new=1729`. The p1 raw prefix kernel is now fixed via the Steel
  bridge, but model/eval overhead still regresses serving.
- Non-blocking radix side-store scatter for prompt/decode sync. It reduces the
  logged sync bucket but shifts work into later eval/decode and regresses the
  serving probe.
- Combined token+cache eval for no-prefix radix prefill. It helps only the cold
  first row and regresses cache hits and long rows.
- Temporary radix-wrapper removal around non-paged radix prefill, even after the
  startup warmup. It can improve long full and delayed rows, but it regresses
  cold first and long partial.
- bf16 p1 paged decode as the default radix serving route. The raw kernel wins
  against p1 gather+MLX, but the full serving route regresses because the model
  path still pays per-layer paged-context/scatter/eval overhead.
- One-token p1 prefix-hit prefill through the p1 decode kernel. It reduces
  selected prefix-hit eval rows but regresses long partial and delayed short.
- Pre-materializing p1 radix prefixes into contiguous caches for one-token
  prefix hits. It avoids per-layer pool-backed gather in the model call but
  shifts enough work into eval/cache handling to regress delayed short badly.
- A 128-thread bf16 p1 lazy decode replacement. It helps some B4/B8 synthetic
  rows but hurts B1/S434 and B1/S2162, the current radix serving probe shape.

The plausible remaining path is a true tiled paged-attention kernel that reuses
K/V tiles across the two GQA query heads without reducing occupancy, or an MLX
Steel-style q_len=1 decode specialization with a query tile smaller than the
current prefill-oriented BQ32 bridge. Any new proposal must first beat dense
`mx.fast.scaled_dot_product_attention` on the contiguous matrix and gather+MLX
on shuffled radix rows before replacing the hybrid guard.

## Suggested local verification command

Final serving-native MLX/Metal path:

```bash
export SGLANG_USE_MLX=1
uv run sglang serve --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B --trust-remote-code --tp-size 1 --port 43436
```

Historical fallback/proof-of-concept comparisons are recorded in `BENCHMARK_RESULTS.md`; the final runtime path no longer uses per-call MLX Metal attention flags.
