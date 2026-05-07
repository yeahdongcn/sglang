# Apple Silicon Metal Attention Plan

This file is the historical staged proof-of-concept plan for FlashAttention-like Metal attention on Apple Silicon. The serving-native design is tracked in `FULL_FLASH_ATTENTION_METAL_PLAN.md`; the old per-call MLX Metal attention flags described below are no longer the target runtime architecture.

## Goals

- Add native Metal attention kernels in `sgl-kernel` using MLX arrays and nanobind.
- Improve the current `SGLANG_USE_MLX=1` batched decode path by removing the per-request K/V padding and `mx.fast.scaled_dot_product_attention` bottleneck.
- Expose Metal-compatible subsets of `flash_attn_with_kvcache` and `flash_attn_varlen_func` without claiming full CUDA FA3 parity too early.
- Historical proof-of-concept behavior was guarded by explicit feature flags; the final serving path is selected by the MLX backend startup path.
- Prove correctness with unit tests and prove usefulness with local Apple Silicon E2E/performance verification.

## Historical feature flags

- `SGLANG_USE_MLX=1`: existing flag that selects the MLX worker path.
- The old proof-of-concept used `SGLANG_MLX_USE_METAL_ATTENTION=1` and `SGLANG_MLX_METAL_ATTENTION_VALIDATE=1` for per-call Metal routing and local fallback validation. These flags were removed from the final serving-native path.

## Progress tracker

Status values:

- `[ ]` not started
- `[/]` in progress
- `[x]` done

1. `[x]` Establish the Metal extension foundation.
   - Add real attention source files under `sgl-kernel/csrc/metal/`.
   - Register the source files in `sgl-kernel/setup_metal.py`.
   - Provide C++/nanobind entry points that accept MLX arrays.
   - Keep placeholder kernels intact until the new smoke path builds.
   - Add Python wrappers in `sgl-kernel/python/sgl_kernel/metal.py` with import guards and shape/dtype validation.
   - Add attention smoke/unit tests that verify the Metal extension loads and the real attention wrappers execute.

2. `[x]` Add decode-only ragged attention for the MLX path.
   - Target shape family used by `MLXAttentionWrapper._batched_decode`: Q/K/V as `(B, H, 1, D)` for queries and per-request contiguous K/V caches.
   - Support GQA/MQA by mapping query heads to KV heads.
   - Support per-request sequence lengths without padding to the maximum sequence length.
   - Compute stable softmax with max/subtract/sum in fp32 accumulation where practical.
   - Return output in a shape that drops into the existing MLX attention wrapper.

3. `[x]` Integrate Metal decode attention into SGLang MLX batched decode.
   - Historical proof-of-concept: route batched decode to the Metal wrapper only when the temporary Metal flag is enabled and the inputs are supported.
   - Final backend: route active paged attention contexts directly through the serving-native MLX/Metal path without the temporary per-call flags.

4. `[x]` Add a Metal-compatible `flash_attn_with_kvcache` subset.
   - Start with read-only contiguous KV cache decode and then in-place append/update.
   - Match CUDA wrapper semantics where implemented: GQA/MQA, `cache_seqlens`, `cache_batch_idx` when feasible, bottom-right causal behavior for multi-query cases.
   - Reject unsupported features explicitly: rotary, descale tensors, sinks, softcap, `qv`, score modifiers, paged cache until implemented.

5. `[x]` Add paged KV cache and varlen prefill support.
   - Add paged KV lookup using `page_table` and block offsets.
   - Add packed varlen prefill for `flash_attn_varlen_func` using `cu_seqlens_q` and `cu_seqlens_k`.
   - Add causal and non-causal varlen tests.
   - Expand supported shape/dtype matrix only after decode correctness and performance are stable.

6. `[x]` Wire Metal attention through the MLX worker path only.
   - Route `MlxModelRunner` prefill/decode batches through scheduler-native paged attention contexts.
   - Use `MLXAttentionWrapper` paged prefill/decode dispatch for native Metal attention.
   - Do not expose a normal SRT `metal` attention backend or CLI choice, because `SGLANG_USE_MLX=1` runs through `MlxModelRunner`.

7. `[x]` Run correctness, E2E, and performance verification.
   - Unit/smoke tests for `sgl-kernel` Metal wrappers.
   - Local MLX/Metal server golden path:
     ```bash
     export SGLANG_USE_MLX=1
     uv run sglang serve --model-path /Users/yexiaodong/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B --trust-remote-code --tp-size 1 --port 43436
     ```
   - Historical fallback/proof-of-concept comparisons are recorded in `BENCHMARK_RESULTS.md`.
   - Measure decode throughput and latency at batch sizes greater than 1, because the first target is the current batched decode bottleneck.

## Acceptance gates

- MLX backend startup behavior is explicit and does not depend on removed per-call Metal attention flags.
- Unsupported API options fail with clear errors instead of silently returning different semantics.
- Metal decode output matches the existing MLX fallback within dtype-appropriate tolerance.
- The serving-native Metal path is faster than the current MLX fallback on targeted batched decode workloads before considering the MLX integration complete.
