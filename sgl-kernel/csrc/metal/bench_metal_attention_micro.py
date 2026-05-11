"""Microbenchmarks for MLX/Metal attention kernels on Apple Silicon.

This script intentionally bypasses SGLang serving/model code.  It measures raw
kernel time for the current ``sgl_kernel.metal`` wrappers and compares the same
tensor shapes against MLX fast scaled dot-product attention where possible.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
import statistics
import time
from pathlib import Path

from sgl_kernel.metal import (
    decode_attention,
    decode_attention_dense_gqa2_h128_unchecked,
    decode_attention_dense_h128_unchecked,
    decode_attention_paged_lazy_regscore_unchecked,
    decode_attention_paged_lazy_unchecked,
    decode_attention_paged_p1_lazy_unchecked,
    decode_attention_paged_unchecked,
    decode_attention_paged_with_kv_unchecked,
    flash_attn_varlen_func,
    paged_kv_scatter,
    prefill_attention_paged,
)

import mlx.core as mx
from sglang.srt.hardware_backend.mlx.kv_cache import MlxPagedKVCache


def _eval_contiguous(x):
    out = mx.contiguous(x)
    mx.eval(out)
    return out


def _randn(shape, dtype):
    return _eval_contiguous(mx.random.normal(shape).astype(dtype))


def _time_call(fn, *, warmups: int, iters: int):
    for _ in range(warmups):
        out = fn()
        if out is not None:
            if isinstance(out, (list, tuple)):
                mx.eval(*out)
            else:
                mx.eval(out)
    mx.synchronize()

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        out = fn()
        if out is not None:
            if isinstance(out, (list, tuple)):
                mx.eval(*out)
            else:
                mx.eval(out)
        mx.synchronize()
        samples.append(time.perf_counter() - start)
    return {
        "median_ms": statistics.median(samples) * 1000.0,
        "mean_ms": statistics.fmean(samples) * 1000.0,
        "min_ms": min(samples) * 1000.0,
        "max_ms": max(samples) * 1000.0,
    }


def _dense_decode_case(batch, seq_len, num_heads, num_kv_heads, head_dim, dtype):
    q = _randn((batch, num_heads, 1, head_dim), dtype)
    k = _randn((batch, num_kv_heads, seq_len, head_dim), dtype)
    v = _randn((batch, num_kv_heads, seq_len, head_dim), dtype)
    scale = 1.0 / math.sqrt(head_dim)
    return q, k, v, scale


def _paged_from_dense(k, v, block_size, *, physical_layout="contiguous"):
    batch, num_kv_heads, seq_len, head_dim = k.shape
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = batch * blocks_per_seq
    k_cache = mx.zeros(
        (total_blocks + 1, block_size, num_kv_heads, head_dim),
        dtype=k.dtype,
    )
    v_cache = mx.zeros(k_cache.shape, dtype=v.dtype)

    def physical_block(batch_idx, block_idx):
        linear = batch_idx * blocks_per_seq + block_idx
        if physical_layout == "contiguous":
            return linear + 1
        if physical_layout == "shuffled":
            return total_blocks - linear
        raise ValueError(f"unknown paged physical layout: {physical_layout}")

    block_rows = []
    for batch_idx in range(batch):
        row = []
        for block_idx in range(blocks_per_seq):
            block = physical_block(batch_idx, block_idx)
            row.append(block)
            start = block_idx * block_size
            end = min(start + block_size, seq_len)
            k_cache[block : block + 1, : end - start, :, :] = k[
                batch_idx : batch_idx + 1, :, start:end, :
            ].transpose(0, 2, 1, 3)
            v_cache[block : block + 1, : end - start, :, :] = v[
                batch_idx : batch_idx + 1, :, start:end, :
            ].transpose(0, 2, 1, 3)
        block_rows.append(row)
    block_tables = mx.array(block_rows, dtype=mx.int32)
    context_lens = mx.array([seq_len] * batch, dtype=mx.int32)
    mx.eval(k_cache, v_cache, block_tables, context_lens)
    return k_cache, v_cache, block_tables, context_lens


def _gather_paged_dense(k_cache, v_cache, block_tables):
    k_blocks = k_cache[block_tables]
    v_blocks = v_cache[block_tables]
    batch, max_blocks, block_size, num_kv_heads, head_dim = k_blocks.shape
    max_tokens = max_blocks * block_size
    k_dense = mx.contiguous(
        k_blocks.reshape(batch, max_tokens, num_kv_heads, head_dim).transpose(
            0, 2, 1, 3
        )
    )
    v_dense = mx.contiguous(
        v_blocks.reshape(batch, max_tokens, num_kv_heads, head_dim).transpose(
            0, 2, 1, 3
        )
    )
    return k_dense, v_dense


def _causal_mask(length, dtype):
    positions = mx.arange(length)
    mask_bool = positions[:, None] < positions[None, :]
    return mx.where(
        mask_bool[None, None, :, :],
        mx.array(mx.finfo(dtype).min, dtype=dtype),
        mx.array(0.0, dtype=dtype),
    )


def _prefix_causal_mask(prefix_len, seq_len, dtype):
    q_positions = mx.arange(seq_len, dtype=mx.int32)
    k_positions = mx.arange(prefix_len + seq_len, dtype=mx.int32)
    invalid = k_positions[None, :] > (prefix_len + q_positions[:, None])
    return mx.where(
        invalid[None, None, :, :],
        mx.array(mx.finfo(dtype).min, dtype=dtype),
        mx.array(0.0, dtype=dtype),
    )


def _decode_padding_mask(context_lens, max_tokens, dtype):
    positions = mx.arange(max_tokens, dtype=mx.int32)
    invalid = positions[None, None, None, :] >= context_lens[:, None, None, None]
    return mx.where(
        invalid,
        mx.array(mx.finfo(dtype).min, dtype=dtype),
        mx.array(0.0, dtype=dtype),
    )


def _prefill_case(batch, seq_len, num_heads, num_kv_heads, head_dim, dtype):
    q = _randn((batch, num_heads, seq_len, head_dim), dtype)
    k = _randn((batch, num_kv_heads, seq_len, head_dim), dtype)
    v = _randn((batch, num_kv_heads, seq_len, head_dim), dtype)
    scale = 1.0 / math.sqrt(head_dim)
    q_packed = _eval_contiguous(
        q.transpose(0, 2, 1, 3).reshape(batch * seq_len, num_heads, head_dim)
    )
    k_packed = _eval_contiguous(
        k.transpose(0, 2, 1, 3).reshape(batch * seq_len, num_kv_heads, head_dim)
    )
    v_packed = _eval_contiguous(
        v.transpose(0, 2, 1, 3).reshape(batch * seq_len, num_kv_heads, head_dim)
    )
    cu = mx.array([i * seq_len for i in range(batch + 1)], dtype=mx.int32)
    mx.eval(cu)
    return q, k, v, q_packed, k_packed, v_packed, cu, scale


def _prefill_prefix_case(
    batch, seq_len, prefix_len, num_heads, num_kv_heads, head_dim, block_size, dtype
):
    q = _randn((batch, num_heads, seq_len, head_dim), dtype)
    k_suffix = _randn((batch, num_kv_heads, seq_len, head_dim), dtype)
    v_suffix = _randn((batch, num_kv_heads, seq_len, head_dim), dtype)
    k_prefix = _randn((batch, num_kv_heads, prefix_len, head_dim), dtype)
    v_prefix = _randn((batch, num_kv_heads, prefix_len, head_dim), dtype)
    scale = 1.0 / math.sqrt(head_dim)

    q_packed = _eval_contiguous(
        q.transpose(0, 2, 1, 3).reshape(batch * seq_len, num_heads, head_dim)
    )
    k_packed = _eval_contiguous(
        k_suffix.transpose(0, 2, 1, 3).reshape(batch * seq_len, num_kv_heads, head_dim)
    )
    v_packed = _eval_contiguous(
        v_suffix.transpose(0, 2, 1, 3).reshape(batch * seq_len, num_kv_heads, head_dim)
    )
    cu = mx.array([i * seq_len for i in range(batch + 1)], dtype=mx.int32)

    blocks_per_seq = (prefix_len + block_size - 1) // block_size
    k_cache = mx.zeros(
        (batch * blocks_per_seq + 1, block_size, num_kv_heads, head_dim),
        dtype=dtype,
    )
    v_cache = mx.zeros(k_cache.shape, dtype=dtype)
    block_rows = []
    for batch_idx in range(batch):
        row = []
        for block_idx in range(blocks_per_seq):
            block = batch_idx * blocks_per_seq + block_idx + 1
            row.append(block)
            start = block_idx * block_size
            end = min(start + block_size, prefix_len)
            k_cache[block : block + 1, : end - start, :, :] = k_prefix[
                batch_idx : batch_idx + 1, :, start:end, :
            ].transpose(0, 2, 1, 3)
            v_cache[block : block + 1, : end - start, :, :] = v_prefix[
                batch_idx : batch_idx + 1, :, start:end, :
            ].transpose(0, 2, 1, 3)
        block_rows.append(row)
    block_tables = mx.array(block_rows, dtype=mx.int32)
    prefix_lens = mx.array([prefix_len] * batch, dtype=mx.int32)
    k_full = mx.concatenate([k_prefix, k_suffix], axis=2)
    v_full = mx.concatenate([v_prefix, v_suffix], axis=2)
    mask = _prefix_causal_mask(prefix_len, seq_len, dtype)
    mx.eval(
        q,
        k_suffix,
        v_suffix,
        k_prefix,
        v_prefix,
        q_packed,
        k_packed,
        v_packed,
        cu,
        k_cache,
        v_cache,
        block_tables,
        prefix_lens,
        k_full,
        v_full,
        mask,
    )
    return (
        q,
        k_full,
        v_full,
        q_packed,
        k_packed,
        v_packed,
        k_cache,
        v_cache,
        block_tables,
        prefix_lens,
        cu,
        mask,
        scale,
    )


def _empty_paged_metadata(batch, seq_len, block_size):
    max_blocks = (seq_len + block_size - 1) // block_size
    return (
        mx.zeros((batch, max_blocks), dtype=mx.int32),
        mx.zeros((batch,), dtype=mx.int32),
    )


def run_benchmarks(args):
    dtype = getattr(mx, args.dtype)
    mx.random.seed(args.seed)
    rows = []

    for batch in ([] if args.only_scatter else args.decode_batches):
        for seq_len in args.decode_seq_lens:
            q, k, v, scale = _dense_decode_case(
                batch,
                seq_len,
                args.num_heads,
                args.num_kv_heads,
                args.head_dim,
                dtype,
            )
            k_cache, v_cache, block_tables, context_lens = _paged_from_dense(
                k, v, args.block_size
            )
            (
                k_cache_radix,
                v_cache_radix,
                block_tables_radix,
                context_lens_radix,
            ) = _paged_from_dense(k, v, args.block_size, physical_layout="shuffled")
            max_tokens = block_tables.shape[1] * args.block_size
            decode_mask = None
            if max_tokens != seq_len:
                decode_mask = _decode_padding_mask(context_lens, max_tokens, dtype)
                mx.eval(decode_mask)

            metal = _time_call(
                lambda: decode_attention_paged_unchecked(
                    q, k_cache, v_cache, block_tables, context_lens, scale
                ),
                warmups=args.warmups,
                iters=args.iters,
            )
            lazy_paged = None
            regscore_paged = None
            fused_paged = None
            if dtype == mx.float16 and args.head_dim == 128 and args.block_size == 16:
                k_new = mx.contiguous(k[:, :, seq_len - 1, :])
                v_new = mx.contiguous(v[:, :, seq_len - 1, :])
                last_block_idx = (seq_len - 1) // args.block_size
                last_block_offset = (seq_len - 1) % args.block_size
                slot_mapping = mx.contiguous(
                    block_tables[:, last_block_idx] * args.block_size
                    + last_block_offset
                )
                lazy_paged = _time_call(
                    lambda: decode_attention_paged_lazy_unchecked(
                        q, k_cache, v_cache, block_tables, context_lens, scale
                    ),
                    warmups=args.warmups,
                    iters=args.iters,
                )
                fused_paged = _time_call(
                    lambda: decode_attention_paged_with_kv_unchecked(
                        q,
                        k_new,
                        v_new,
                        k_cache,
                        v_cache,
                        block_tables,
                        context_lens,
                        slot_mapping,
                        scale,
                    ),
                    warmups=args.warmups,
                    iters=args.iters,
                )
                regscore_paged = _time_call(
                    lambda: decode_attention_paged_lazy_regscore_unchecked(
                        q, k_cache, v_cache, block_tables, context_lens, scale
                    ),
                    warmups=args.warmups,
                    iters=args.iters,
                )
            mlx_fast = _time_call(
                lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale),
                warmups=args.warmups,
                iters=args.iters,
            )
            gather_paged = _time_call(
                lambda: _gather_paged_dense(k_cache, v_cache, block_tables),
                warmups=args.warmups,
                iters=args.iters,
            )
            mlx_fast_gathered = _time_call(
                lambda: mx.fast.scaled_dot_product_attention(
                    q,
                    *_gather_paged_dense(k_cache, v_cache, block_tables),
                    scale=scale,
                    mask=decode_mask,
                ),
                warmups=args.warmups,
                iters=args.iters,
            )
            metal_radix = _time_call(
                lambda: decode_attention_paged_unchecked(
                    q,
                    k_cache_radix,
                    v_cache_radix,
                    block_tables_radix,
                    context_lens_radix,
                    scale,
                ),
                warmups=args.warmups,
                iters=args.iters,
            )
            lazy_radix = None
            regscore_radix = None
            fused_radix = None
            if dtype == mx.float16 and args.head_dim == 128 and args.block_size == 16:
                k_new = mx.contiguous(k[:, :, seq_len - 1, :])
                v_new = mx.contiguous(v[:, :, seq_len - 1, :])
                last_block_idx = (seq_len - 1) // args.block_size
                last_block_offset = (seq_len - 1) % args.block_size
                slot_mapping_radix = mx.contiguous(
                    block_tables_radix[:, last_block_idx] * args.block_size
                    + last_block_offset
                )
                lazy_radix = _time_call(
                    lambda: decode_attention_paged_lazy_unchecked(
                        q,
                        k_cache_radix,
                        v_cache_radix,
                        block_tables_radix,
                        context_lens_radix,
                        scale,
                    ),
                    warmups=args.warmups,
                    iters=args.iters,
                )
                fused_radix = _time_call(
                    lambda: decode_attention_paged_with_kv_unchecked(
                        q,
                        k_new,
                        v_new,
                        k_cache_radix,
                        v_cache_radix,
                        block_tables_radix,
                        context_lens_radix,
                        slot_mapping_radix,
                        scale,
                    ),
                    warmups=args.warmups,
                    iters=args.iters,
                )
                regscore_radix = _time_call(
                    lambda: decode_attention_paged_lazy_regscore_unchecked(
                        q,
                        k_cache_radix,
                        v_cache_radix,
                        block_tables_radix,
                        context_lens_radix,
                        scale,
                    ),
                    warmups=args.warmups,
                    iters=args.iters,
                )
            gather_radix = _time_call(
                lambda: _gather_paged_dense(
                    k_cache_radix, v_cache_radix, block_tables_radix
                ),
                warmups=args.warmups,
                iters=args.iters,
            )
            mlx_fast_radix = _time_call(
                lambda: mx.fast.scaled_dot_product_attention(
                    q,
                    *_gather_paged_dense(
                        k_cache_radix, v_cache_radix, block_tables_radix
                    ),
                    scale=scale,
                    mask=decode_mask,
                ),
                warmups=args.warmups,
                iters=args.iters,
            )
            dense_lazy_variants = []
            if dtype == mx.float16 and args.head_dim == 128:
                for mode, threads, label in (
                    ("regscore", 128, "metal_dense_h128_regscore_t128"),
                    ("sharedscore", 128, "metal_dense_h128_sharedscore_t128"),
                    ("regscore", 256, "metal_dense_h128_regscore_t256"),
                    ("sharedscore", 256, "metal_dense_h128_sharedscore_t256"),
                ):
                    result = _time_call(
                        (
                            lambda mode=mode, threads=threads: (
                                decode_attention_dense_h128_unchecked(
                                    q,
                                    k,
                                    v,
                                    scale,
                                    mode=mode,
                                    threads=threads,
                                )
                            )
                        ),
                        warmups=args.warmups,
                        iters=args.iters,
                    )
                    dense_lazy_variants.append((label, result))
                if args.num_heads == 2 * args.num_kv_heads:
                    for threads, label in (
                        (128, "metal_dense_gqa2_h128_t128"),
                        (256, "metal_dense_gqa2_h128_t256"),
                    ):
                        result = _time_call(
                            (
                                lambda threads=threads: (
                                    decode_attention_dense_gqa2_h128_unchecked(
                                        q,
                                        k,
                                        v,
                                        scale,
                                        threads=threads,
                                    )
                                )
                            ),
                            warmups=args.warmups,
                            iters=args.iters,
                        )
                        dense_lazy_variants.append((label, result))
            p1_lazy_variants = []
            if dtype == mx.float16 and args.head_dim == 128:
                k_cache_p1, v_cache_p1, block_tables_p1, context_lens_p1 = (
                    _paged_from_dense(k, v, 1)
                )
                (
                    k_cache_p1_radix,
                    v_cache_p1_radix,
                    block_tables_p1_radix,
                    context_lens_p1_radix,
                ) = _paged_from_dense(k, v, 1, physical_layout="shuffled")
                p1_lazy = _time_call(
                    lambda: decode_attention_paged_p1_lazy_unchecked(
                        q,
                        k_cache_p1,
                        v_cache_p1,
                        block_tables_p1,
                        context_lens_p1,
                        scale,
                    ),
                    warmups=args.warmups,
                    iters=args.iters,
                )
                p1_lazy_radix = _time_call(
                    lambda: decode_attention_paged_p1_lazy_unchecked(
                        q,
                        k_cache_p1_radix,
                        v_cache_p1_radix,
                        block_tables_p1_radix,
                        context_lens_p1_radix,
                        scale,
                    ),
                    warmups=args.warmups,
                    iters=args.iters,
                )
                p1_gather_radix = _time_call(
                    lambda: _gather_paged_dense(
                        k_cache_p1_radix, v_cache_p1_radix, block_tables_p1_radix
                    ),
                    warmups=args.warmups,
                    iters=args.iters,
                )
                p1_mlx_fast_radix = _time_call(
                    lambda: mx.fast.scaled_dot_product_attention(
                        q,
                        *_gather_paged_dense(
                            k_cache_p1_radix, v_cache_p1_radix, block_tables_p1_radix
                        ),
                        scale=scale,
                    ),
                    warmups=args.warmups,
                    iters=args.iters,
                )
                p1_lazy_variants.extend(
                    [
                        ("metal_paged_p1_lazy_h128", p1_lazy),
                        ("metal_paged_p1_lazy_h128_radix_shuffled", p1_lazy_radix),
                        ("mlx_gather_p1_radix_paged_kv", p1_gather_radix),
                        ("mlx_fast_p1_radix_gathered_paged", p1_mlx_fast_radix),
                    ]
                )
            if args.bf16_p1_decode and args.head_dim == 128:
                q_bf16 = _eval_contiguous(q.astype(mx.bfloat16))
                k_bf16 = _eval_contiguous(k.astype(mx.bfloat16))
                v_bf16 = _eval_contiguous(v.astype(mx.bfloat16))
                k_cache_p1, v_cache_p1, block_tables_p1, context_lens_p1 = (
                    _paged_from_dense(k_bf16, v_bf16, 1)
                )
                (
                    k_cache_p1_radix,
                    v_cache_p1_radix,
                    block_tables_p1_radix,
                    context_lens_p1_radix,
                ) = _paged_from_dense(k_bf16, v_bf16, 1, physical_layout="shuffled")
                p1_lazy = _time_call(
                    lambda: decode_attention_paged_p1_lazy_unchecked(
                        q_bf16,
                        k_cache_p1,
                        v_cache_p1,
                        block_tables_p1,
                        context_lens_p1,
                        scale,
                    ),
                    warmups=args.warmups,
                    iters=args.iters,
                )
                p1_lazy_radix = _time_call(
                    lambda: decode_attention_paged_p1_lazy_unchecked(
                        q_bf16,
                        k_cache_p1_radix,
                        v_cache_p1_radix,
                        block_tables_p1_radix,
                        context_lens_p1_radix,
                        scale,
                    ),
                    warmups=args.warmups,
                    iters=args.iters,
                )
                p1_gather_radix = _time_call(
                    lambda: _gather_paged_dense(
                        k_cache_p1_radix, v_cache_p1_radix, block_tables_p1_radix
                    ),
                    warmups=args.warmups,
                    iters=args.iters,
                )
                p1_mlx_fast_radix = _time_call(
                    lambda: mx.fast.scaled_dot_product_attention(
                        q_bf16,
                        *_gather_paged_dense(
                            k_cache_p1_radix, v_cache_p1_radix, block_tables_p1_radix
                        ),
                        scale=scale,
                    ),
                    warmups=args.warmups,
                    iters=args.iters,
                )
                p1_lazy_variants.extend(
                    [
                        ("metal_paged_p1_lazy_bf16_h128", p1_lazy),
                        (
                            "metal_paged_p1_lazy_bf16_h128_radix_shuffled",
                            p1_lazy_radix,
                        ),
                        ("mlx_gather_p1_bf16_radix_paged_kv", p1_gather_radix),
                        (
                            "mlx_fast_p1_bf16_radix_gathered_paged",
                            p1_mlx_fast_radix,
                        ),
                    ]
                )
            dense_metal = _time_call(
                lambda: decode_attention(q, k, v, scale),
                warmups=args.warmups,
                iters=args.iters,
            )
            rows.extend(
                [
                    {
                        "case": "decode",
                        "variant": "metal_paged",
                        "batch": batch,
                        "seq_len": seq_len,
                        **metal,
                    },
                    {
                        "case": "decode",
                        "variant": "mlx_fast_dense",
                        "batch": batch,
                        "seq_len": seq_len,
                        **mlx_fast,
                    },
                    *(
                        [
                            {
                                "case": "decode",
                                "variant": "metal_paged_lazy_h128_b16",
                                "batch": batch,
                                "seq_len": seq_len,
                                **lazy_paged,
                            }
                        ]
                        if lazy_paged is not None
                        else []
                    ),
                    *(
                        [
                            {
                                "case": "decode",
                                "variant": "metal_paged_lazy_regscore_h128_b16",
                                "batch": batch,
                                "seq_len": seq_len,
                                **regscore_paged,
                            }
                        ]
                        if regscore_paged is not None
                        else []
                    ),
                    *(
                        [
                            {
                                "case": "decode",
                                "variant": "metal_paged_fused_kv_h128_b16",
                                "batch": batch,
                                "seq_len": seq_len,
                                **fused_paged,
                            }
                        ]
                        if fused_paged is not None
                        else []
                    ),
                    {
                        "case": "decode",
                        "variant": "mlx_gather_paged_kv",
                        "batch": batch,
                        "seq_len": seq_len,
                        **gather_paged,
                    },
                    {
                        "case": "decode",
                        "variant": "mlx_fast_gathered_paged",
                        "batch": batch,
                        "seq_len": seq_len,
                        **mlx_fast_gathered,
                    },
                    {
                        "case": "decode",
                        "variant": "metal_paged_radix_shuffled",
                        "batch": batch,
                        "seq_len": seq_len,
                        **metal_radix,
                    },
                    *(
                        [
                            {
                                "case": "decode",
                                "variant": "metal_paged_lazy_h128_b16_radix_shuffled",
                                "batch": batch,
                                "seq_len": seq_len,
                                **lazy_radix,
                            }
                        ]
                        if lazy_radix is not None
                        else []
                    ),
                    *(
                        [
                            {
                                "case": "decode",
                                "variant": "metal_paged_lazy_regscore_h128_b16_radix_shuffled",
                                "batch": batch,
                                "seq_len": seq_len,
                                **regscore_radix,
                            }
                        ]
                        if regscore_radix is not None
                        else []
                    ),
                    *(
                        [
                            {
                                "case": "decode",
                                "variant": "metal_paged_fused_kv_h128_b16_radix_shuffled",
                                "batch": batch,
                                "seq_len": seq_len,
                                **fused_radix,
                            }
                        ]
                        if fused_radix is not None
                        else []
                    ),
                    {
                        "case": "decode",
                        "variant": "mlx_gather_radix_paged_kv",
                        "batch": batch,
                        "seq_len": seq_len,
                        **gather_radix,
                    },
                    {
                        "case": "decode",
                        "variant": "mlx_fast_radix_gathered_paged",
                        "batch": batch,
                        "seq_len": seq_len,
                        **mlx_fast_radix,
                    },
                    *[
                        {
                            "case": "decode",
                            "variant": label,
                            "batch": batch,
                            "seq_len": seq_len,
                            **result,
                        }
                        for label, result in dense_lazy_variants
                    ],
                    *[
                        {
                            "case": "decode",
                            "variant": label,
                            "batch": batch,
                            "seq_len": seq_len,
                            **result,
                        }
                        for label, result in p1_lazy_variants
                    ],
                    {
                        "case": "decode",
                        "variant": "metal_dense_wrapper",
                        "batch": batch,
                        "seq_len": seq_len,
                        **dense_metal,
                    },
                ]
            )

    for batch in ([] if args.only_scatter else args.prefill_batches):
        for seq_len in args.prefill_seq_lens:
            q, k, v, q_packed, k_packed, v_packed, cu, scale = _prefill_case(
                batch,
                seq_len,
                args.num_heads,
                args.num_kv_heads,
                args.head_dim,
                dtype,
            )
            block_tables, prefix_lens = _empty_paged_metadata(
                batch, seq_len, args.block_size
            )
            k_cache = mx.zeros(
                (1, args.block_size, args.num_kv_heads, args.head_dim),
                dtype=dtype,
            )
            v_cache = mx.zeros(k_cache.shape, dtype=dtype)
            mx.eval(k_cache, v_cache, block_tables, prefix_lens)
            mask = _causal_mask(seq_len, dtype)
            mx.eval(mask)

            metal_prefill = _time_call(
                lambda: prefill_attention_paged(
                    q_packed,
                    k_packed,
                    v_packed,
                    k_cache,
                    v_cache,
                    block_tables,
                    prefix_lens,
                    cu,
                    scale,
                    causal=True,
                ),
                warmups=args.warmups,
                iters=args.prefill_iters,
            )
            metal_varlen = _time_call(
                lambda: flash_attn_varlen_func(
                    q_packed,
                    k_packed,
                    v_packed,
                    cu,
                    cu,
                    max_seqlen_q=seq_len,
                    max_seqlen_k=seq_len,
                    softmax_scale=scale,
                    causal=True,
                ),
                warmups=args.warmups,
                iters=args.prefill_iters,
            )
            mlx_fast = _time_call(
                lambda: mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask
                ),
                warmups=args.warmups,
                iters=args.prefill_iters,
            )
            rows.extend(
                [
                    {
                        "case": "prefill",
                        "variant": "metal_prefill_paged_no_prefix",
                        "batch": batch,
                        "seq_len": seq_len,
                        **metal_prefill,
                    },
                    {
                        "case": "prefill",
                        "variant": "metal_flash_varlen",
                        "batch": batch,
                        "seq_len": seq_len,
                        **metal_varlen,
                    },
                    {
                        "case": "prefill",
                        "variant": "mlx_fast_dense",
                        "batch": batch,
                        "seq_len": seq_len,
                        **mlx_fast,
                    },
                ]
            )

            for prefix_len in args.prefill_prefix_lens:
                if prefix_len <= 0:
                    continue
                (
                    q_prefix,
                    k_full,
                    v_full,
                    q_prefix_packed,
                    k_prefix_packed,
                    v_prefix_packed,
                    k_prefix_cache,
                    v_prefix_cache,
                    prefix_block_tables,
                    prefix_lens,
                    prefix_cu,
                    prefix_mask,
                    prefix_scale,
                ) = _prefill_prefix_case(
                    batch,
                    seq_len,
                    prefix_len,
                    args.num_heads,
                    args.num_kv_heads,
                    args.head_dim,
                    args.block_size,
                    dtype,
                )
                metal_prefill_prefix = _time_call(
                    lambda: prefill_attention_paged(
                        q_prefix_packed,
                        k_prefix_packed,
                        v_prefix_packed,
                        k_prefix_cache,
                        v_prefix_cache,
                        prefix_block_tables,
                        prefix_lens,
                        prefix_cu,
                        prefix_scale,
                        causal=True,
                    ),
                    warmups=args.warmups,
                    iters=args.prefill_iters,
                )
                mlx_fast_prefix = _time_call(
                    lambda: mx.fast.scaled_dot_product_attention(
                        q_prefix,
                        k_full,
                        v_full,
                        scale=prefix_scale,
                        mask=prefix_mask,
                    ),
                    warmups=args.warmups,
                    iters=args.prefill_iters,
                )
                rows.extend(
                    [
                        {
                            "case": "prefill",
                            "variant": f"metal_prefill_paged_prefix_{prefix_len}",
                            "batch": batch,
                            "seq_len": seq_len,
                            **metal_prefill_prefix,
                        },
                        {
                            "case": "prefill",
                            "variant": f"mlx_fast_dense_prefix_{prefix_len}",
                            "batch": batch,
                            "seq_len": seq_len,
                            **mlx_fast_prefix,
                        },
                    ]
                )

    for batch in args.scatter_batches:
        for tokens in args.scatter_tokens:
            k_tokens = _randn((tokens, args.num_kv_heads, args.head_dim), dtype)
            v_tokens = _randn((tokens, args.num_kv_heads, args.head_dim), dtype)
            num_blocks = max(
                (tokens + args.block_size - 1) // args.block_size + 1, batch + 1
            )
            k_cache = mx.zeros(
                (num_blocks, args.block_size, args.num_kv_heads, args.head_dim),
                dtype=dtype,
            )
            v_cache = mx.zeros(k_cache.shape, dtype=dtype)
            slots = mx.array(
                list(range(args.block_size, args.block_size + tokens)), dtype=mx.int32
            )
            mx.eval(k_cache, v_cache, slots)
            scatter = _time_call(
                lambda: paged_kv_scatter(k_tokens, v_tokens, k_cache, v_cache, slots),
                warmups=args.warmups,
                iters=args.iters,
            )
            rows.append(
                {
                    "case": "scatter",
                    "variant": "metal_paged_kv_scatter",
                    "batch": batch,
                    "seq_len": tokens,
                    **scatter,
                }
            )

    for tokens in args.scatter_tokens:
        cache_stacked = MlxPagedKVCache(
            num_layers=args.sync_layers,
            num_blocks=max(1, math.ceil(tokens / args.block_size) + 1),
            block_size=args.block_size,
            n_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
        )
        cache_list = MlxPagedKVCache(
            num_layers=args.sync_layers,
            num_blocks=max(1, math.ceil(tokens / args.block_size) + 1),
            block_size=args.block_size,
            n_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
        )
        cache_metal = MlxPagedKVCache(
            num_layers=args.sync_layers,
            num_blocks=max(1, math.ceil(tokens / args.block_size) + 1),
            block_size=args.block_size,
            n_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
        )
        slots = mx.arange(tokens, dtype=mx.int32)
        k_layers = [
            _randn((tokens, args.num_kv_heads, args.head_dim), dtype)
            for _ in range(args.sync_layers)
        ]
        v_layers = [
            _randn((tokens, args.num_kv_heads, args.head_dim), dtype)
            for _ in range(args.sync_layers)
        ]
        k_stacked = mx.stack(k_layers)
        v_stacked = mx.stack(v_layers)
        mx.eval(slots, k_stacked, v_stacked, *k_layers, *v_layers)
        sync_stacked = _time_call(
            lambda: cache_stacked.set_kv_all_layers(
                slots, k_stacked, v_stacked, eager=True
            ),
            warmups=args.warmups,
            iters=args.iters,
        )
        sync_list = _time_call(
            lambda: cache_list.set_kv_all_layers(slots, k_layers, v_layers, eager=True),
            warmups=args.warmups,
            iters=args.iters,
        )
        sync_metal = _time_call(
            lambda: cache_metal.set_kv_all_layers(
                slots, k_layers, v_layers, eager=True, use_metal=True
            ),
            warmups=args.warmups,
            iters=args.iters,
        )
        rows.extend(
            [
                {
                    "case": "radix_sync",
                    "variant": "mlx_set_kv_all_layers_stacked",
                    "batch": tokens,
                    "seq_len": args.sync_layers,
                    **sync_stacked,
                },
                {
                    "case": "radix_sync",
                    "variant": "mlx_set_kv_all_layers_list",
                    "batch": tokens,
                    "seq_len": args.sync_layers,
                    **sync_list,
                },
                {
                    "case": "radix_sync",
                    "variant": "metal_scatter_all_layers_list",
                    "batch": tokens,
                    "seq_len": args.sync_layers,
                    **sync_metal,
                },
            ]
        )

    return rows


def format_markdown(rows, args):
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Metal Attention Microbenchmark Results",
        "",
        f"Generated: `{now}`",
        "",
        "Command context:",
        "",
        "```text",
        f"dtype={args.dtype}, heads={args.num_heads}, kv_heads={args.num_kv_heads}, head_dim={args.head_dim}, block_size={args.block_size}",
        f"warmups={args.warmups}, decode/scatter iters={args.iters}, prefill iters={args.prefill_iters}",
        f"prefill_prefix_lens={args.prefill_prefix_lens}, sync_layers={args.sync_layers}, bf16_p1_decode={args.bf16_p1_decode}",
        "```",
        "",
        "| Case | Variant | Batch/Tokens | Seq Len | Median ms | Mean ms | Min ms | Max ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {variant} | {batch} | {seq_len} | {median_ms:.4f} | {mean_ms:.4f} | {min_ms:.4f} | {max_ms:.4f} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--prefill-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decode-batches", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--decode-seq-lens", type=int, nargs="+", default=[256, 2048])
    parser.add_argument("--prefill-batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--prefill-seq-lens", type=int, nargs="+", default=[256])
    parser.add_argument("--prefill-prefix-lens", type=int, nargs="+", default=[])
    parser.add_argument("--scatter-batches", type=int, nargs="+", default=[1])
    parser.add_argument("--scatter-tokens", type=int, nargs="+", default=[1, 4, 256])
    parser.add_argument("--sync-layers", type=int, default=28)
    parser.add_argument("--bf16-p1-decode", action="store_true")
    parser.add_argument("--only-scatter", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sgl-kernel/csrc/metal/METAL_MICROBENCH_RESULTS.md"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = run_benchmarks(args)
    text = format_markdown(rows, args)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
