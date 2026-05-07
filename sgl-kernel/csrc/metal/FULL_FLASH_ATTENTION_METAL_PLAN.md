# Full Apple Silicon Metal FlashAttention Backend Plan

This file tracks a new first-class MLX/Metal FlashAttention-style backend redesign for SGLang on Apple Silicon.
It intentionally does not preserve the current env-gated `SGLANG_MLX_USE_METAL_ATTENTION` design as the final architecture.

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

All progress items are complete as of the serving-native MLX/Metal paged-attention integration, eager startup guard, focused unit/kernel verification, and local Apple Silicon E2E/performance verification.

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
   - Current status: `MlxModelRunner` builds prefill, full-prefix-hit, extend, and decode contexts from scheduler slot rows/block tables, writes new K/V directly into paged cache through Metal scatter, tracks request state without request-local contiguous cache mirrors, and leaves slot/block ownership to the scheduler and radix cache.

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
   - Current status: `MlxModelRunner` now performs eager Apple Silicon/Metal API startup checks, constructs native paged prefill/decode contexts from scheduler metadata, requires paged KV cache for serving prefill/extend/decode, uses lightweight offset-only cache shims for model compatibility, tracks request state through token IDs plus scheduler pool indices, and no longer maintains request-local `ContiguousKVCache`/`PoolBackedCache` mirrors.

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
    - Current status: Focused MLX unit coverage verifies paged context normalization/scatter tracking, wrapper paged prefill/decode routing, required metadata validation, legacy-cache ignoring in paged attention, runner paged prefill/extend/decode context construction, startup capability checks including import-success/native-extension-missing failure, direct native scatter tracking, and token-state-only request cleanup. The focused `unittest` run for `test_model_runner`, `test_attention_wrapper`, and `test_paged_context` passed 39 tests in the existing `sglang-mlx` environment; Metal kernel coverage for paged scatter, paged decode, paged prefill, packed varlen, compatibility wrappers, invalid metadata, unsupported options, and dtype behavior is in `sgl-kernel/tests/test_metal.py`.

11. `[x]` Add Apple Silicon E2E and performance verification.
    - Run local MLX server golden-path generation with the new backend.
    - Compare no-prefix, partial-prefix, and full-prefix radix-cache workloads.
    - Measure prefill, extend, and decode throughput/latency for batch sizes greater than 1.
    - Compare against the current MLX fallback and the existing guarded Metal proof-of-concept.
    - Record results in a benchmark document before declaring the redesign complete.
    - Current status: Post-dtype-fix local MLX/Metal server smoke completed successfully for no-prefix, partial-prefix, and full-prefix prompt cases on Qwen3-0.6B. Focused results are recorded in `BENCHMARK_RESULTS.md`. The repeat `bench_one_batch --batch-size 1 4 8 --input-len 256 --output-len 32` comparison shows Metal ahead on total throughput for batch sizes 1, 4, and 8; median decode throughput is ahead for batch sizes 4 and 8 and slightly behind for batch size 1. This closes the local E2E/performance verification gate and leaves guarded-path removal as the next step.

12. `[x]` Remove the old guarded implementation after replacement is proven.
    - Delete final dependence on the removed per-call MLX Metal attention flag.
    - Remove obsolete `ContiguousKVCache`/flat-pool bridging code from the Metal backend path.
    - Remove compatibility scaffolding that is no longer part of the final design.
    - Keep only documented fallback behavior that is selected at backend startup, not per attention call.
    - Current status: active runtime code no longer references the old per-call MLX Metal attention or validation flags, the obsolete flat-slot `decode_attention_slots` public API/kernel/tests/docs have been removed, and the serving-native MLX runner path now requires scheduler-owned paged contexts instead of request-local contiguous-cache mirrors. Remaining `ContiguousKVCache`/`PoolBackedCache` symbols are isolated to the legacy cache module and are no longer exported through the MLX KV-cache package or used by focused serving-path tests.

## Acceptance gates

- The MLX/Metal backend supports radix-cache prefix reuse without copying prefixes into per-request contiguous caches.
- Decode reads directly from paged KV cache using block tables.
- Prefill uses native Metal varlen FlashAttention kernels in the final backend path.
- Default Apple Silicon MLX backend selection is clear and does not rely on `SGLANG_MLX_USE_METAL_ATTENTION` per-call gating.
- Unsupported features fail clearly instead of silently producing different semantics.
- Unit/kernel tests pass for cache writes, paged decode, varlen prefill, and radix-cache cases.
- Local Apple Silicon E2E generation works for no-prefix, partial-prefix, and full-prefix cache-hit workloads.
- Performance beats the current MLX fallback and the existing guarded Metal proof-of-concept on targeted serving workloads before the redesign is considered complete.

## Suggested local verification command

Final serving-native MLX/Metal path:

```bash
export SGLANG_USE_MLX=1
uv run sglang serve --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B --trust-remote-code --tp-size 1 --port 43436
```

Historical fallback/proof-of-concept comparisons are recorded in `BENCHMARK_RESULTS.md`; the final runtime path no longer uses per-call MLX Metal attention flags.
