# Metal p1 Lazy Decode Microbenchmark

Generated on 2026-05-10 after adding `decode_attention_paged_p1_lazy_unchecked`.

This is a direct-paged/radix raw-kernel result, not an end-to-end serving
completion signal. The default radix serving path is still the hybrid path that
uses request-local contiguous MLX caches for active decode and the p1
`MlxPagedKVCache` as a radix-visible side-store. The new kernel is used when a
paged decode context is active; it is not evidence by itself that the hybrid
serving path got faster.

## Commands

```bash
PYTHONPATH=sgl-kernel/python:python ./sglang-mlx/bin/python \
  sgl-kernel/csrc/metal/bench_metal_attention_micro.py \
  --dtype float16 --num-heads 16 --num-kv-heads 8 --head-dim 128 \
  --block-size 16 --warmups 2 --iters 5 --prefill-iters 1 \
  --decode-batches 1 4 8 --decode-seq-lens 288 2162 \
  --prefill-batches 1 --prefill-seq-lens 256 --prefill-prefix-lens 256 \
  --output /tmp/sglang_micro_p1_lazy_decode_20260510.md

PYTHONPATH=sgl-kernel/python:python ./sglang-mlx/bin/python \
  sgl-kernel/csrc/metal/bench_metal_attention_micro.py \
  --dtype float16 --num-heads 16 --num-kv-heads 8 --head-dim 128 \
  --block-size 16 --warmups 2 --iters 5 --prefill-iters 1 \
  --decode-batches 1 --decode-seq-lens 8 16 24 32 64 128 \
  --prefill-batches 1 --prefill-seq-lens 16 --prefill-prefix-lens 16 \
  --output /tmp/sglang_micro_p1_lazy_decode_short_20260510.md

PYTHONPATH=sgl-kernel/python:python ./sglang-mlx/bin/python \
  sgl-kernel/csrc/metal/bench_metal_attention_micro.py \
  --dtype float16 --num-heads 16 --num-kv-heads 8 --head-dim 128 \
  --block-size 16 --warmups 2 --iters 7 --prefill-iters 1 \
  --decode-batches 4 8 --decode-seq-lens 8 16 24 32 64 \
  --prefill-batches 1 --prefill-seq-lens 16 --prefill-prefix-lens 16 \
  --output /tmp/sglang_micro_p1_lazy_decode_short_batched_20260510.md
```

## Broad Decode Rows

All timings are median milliseconds.

| Row | Dense MLX fast | p1 lazy radix | p1 gather+MLX radix | Decision |
|---|---:|---:|---:|---|
| B1/S288 | 0.2790 | 0.2774 | 0.3286 | p1 lazy ahead of p1 gather+MLX |
| B1/S2162 | 0.4184 | 0.4783 | 1.0557 | p1 lazy ahead of p1 gather+MLX, still behind dense MLX |
| B4/S288 | 0.3289 | 0.3151 | 0.8232 | p1 lazy ahead |
| B4/S2162 | 0.9079 | 1.0888 | 4.7432 | p1 lazy ahead of p1 gather+MLX, still behind dense MLX |
| B8/S288 | 0.5001 | 0.4425 | 1.2690 | p1 lazy ahead |
| B8/S2162 | 1.6087 | 1.5313 | 12.3570 | p1 lazy ahead |

## Short Single-Request Rows

| Row | p1 lazy radix | p1 gather+MLX radix | Decision |
|---|---:|---:|---|
| B1/S8 | 0.2605 | 0.2613 | tie/no route change needed |
| B1/S16 | 0.4291 | 0.3155 | keep gather+MLX |
| B1/S24 | 0.3662 | 0.4718 | use p1 lazy |
| B1/S32 | 0.3194 | 0.3639 | use p1 lazy |
| B1/S64 | 0.2758 | 0.3060 | use p1 lazy |
| B1/S128 | 0.3005 | 0.3187 | use p1 lazy |

## Short Batched Rows

| Row | p1 lazy radix | p1 gather+MLX radix | Decision |
|---|---:|---:|---|
| B4/S8 | 0.2763 | 0.3743 | use p1 lazy |
| B4/S16 | 0.3110 | 0.3950 | use p1 lazy |
| B4/S24 | 0.3402 | 0.4707 | use p1 lazy |
| B4/S32 | 0.4612 | 0.5643 | use p1 lazy |
| B4/S64 | 0.4809 | 0.6065 | use p1 lazy |
| B8/S8 | 0.4820 | 0.5766 | use p1 lazy |
| B8/S16 | 0.3790 | 0.5793 | use p1 lazy |
| B8/S24 | 0.3796 | 0.7360 | use p1 lazy |
| B8/S32 | 0.4801 | 0.5959 | use p1 lazy |
| B8/S64 | 0.2777 | 0.5562 | use p1 lazy |

## Decision

Keep the p1 lazy decode kernel and route it in the paged-attention wrapper for
float16/head_dim=128/page_size=1 decode when `batch > 1` or single-request
`context_len > 16`. Keep B1 contexts up to 16 tokens on the existing gather+MLX
path.

The raw direct-paged radix decode path is now materially better for p1
block-table layouts. The broader completion gate remains open because default
serving decode is still hybrid/contiguous-cache based and dense
`mx.fast.scaled_dot_product_attention` is still faster on some synthetic
contiguous rows.
