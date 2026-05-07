import math
import platform
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Metal extension smoke test requires Apple Silicon",
)


def _decode_reference(q, k, v, scale):
    import mlx.core as mx

    outputs = []
    groups = q.shape[1] // k.shape[1]
    for batch in range(q.shape[0]):
        batch_heads = []
        for head in range(q.shape[1]):
            kv_head = head // groups
            scores = mx.sum(q[batch, head, 0, :] * k[batch, kv_head, :, :], axis=-1)
            weights = mx.softmax(scores * scale, axis=-1)
            batch_heads.append(
                mx.sum(weights[:, None] * v[batch, kv_head, :, :], axis=0)
            )
        outputs.append(mx.stack(batch_heads, axis=0)[:, None, :])
    return mx.stack(outputs, axis=0)


def _decode_ragged_reference(q, k_list, v_list, scale):
    import mlx.core as mx

    outputs = []
    groups = q.shape[1] // k_list[0].shape[1]
    for batch, (k, v) in enumerate(zip(k_list, v_list, strict=True)):
        batch_heads = []
        for head in range(q.shape[1]):
            kv_head = head // groups
            scores = mx.sum(q[batch, head, 0, :] * k[0, kv_head, :, :], axis=-1)
            weights = mx.softmax(scores * scale, axis=-1)
            batch_heads.append(mx.sum(weights[:, None] * v[0, kv_head, :, :], axis=0))
        outputs.append(mx.stack(batch_heads, axis=0)[:, None, :])
    return mx.stack(outputs, axis=0)


def _flash_attn_reference(q, k_cache, v_cache, seq_lens, batch_indices, scale):
    import mlx.core as mx

    metal_q = q.transpose(0, 2, 1, 3)
    k_list = [
        k_cache[cache_batch : cache_batch + 1, :seq_len, :, :].transpose(0, 2, 1, 3)
        for cache_batch, seq_len in zip(batch_indices, seq_lens, strict=True)
    ]
    v_list = [
        v_cache[cache_batch : cache_batch + 1, :seq_len, :, :].transpose(0, 2, 1, 3)
        for cache_batch, seq_len in zip(batch_indices, seq_lens, strict=True)
    ]
    return _decode_ragged_reference(metal_q, k_list, v_list, scale).transpose(
        0, 2, 1, 3
    )


def _decode_paged_reference(q, k_cache, v_cache, page_rows, seq_lens, scale):
    import mlx.core as mx

    page_block_size = k_cache.shape[1]
    k_list = []
    v_list = []
    for blocks, seq_len in zip(page_rows, seq_lens, strict=True):
        needed_blocks = (seq_len + page_block_size - 1) // page_block_size
        k_pages = mx.concatenate(
            [k_cache[block : block + 1] for block in blocks[:needed_blocks]], axis=0
        )
        v_pages = mx.concatenate(
            [v_cache[block : block + 1] for block in blocks[:needed_blocks]], axis=0
        )
        k_flat = k_pages.reshape(
            1, needed_blocks * page_block_size, k_cache.shape[2], k_cache.shape[3]
        )[:, :seq_len, :, :]
        v_flat = v_pages.reshape(
            1, needed_blocks * page_block_size, v_cache.shape[2], v_cache.shape[3]
        )[:, :seq_len, :, :]
        k_list.append(k_flat.transpose(0, 2, 1, 3))
        v_list.append(v_flat.transpose(0, 2, 1, 3))
    return _decode_ragged_reference(q, k_list, v_list, scale)


def _flash_attn_paged_reference(q, k_cache, v_cache, page_rows, seq_lens, scale):
    return _decode_paged_reference(
        q.transpose(0, 2, 1, 3), k_cache, v_cache, page_rows, seq_lens, scale
    ).transpose(0, 2, 1, 3)


def _flash_attn_varlen_reference(q, k, v, cu_q, cu_k, scale, causal):
    import mlx.core as mx

    outputs = []
    groups = q.shape[1] // k.shape[1]
    for q_start, q_end, k_start, k_end in zip(cu_q, cu_q[1:], cu_k, cu_k[1:]):
        seq_q_len = q_end - q_start
        seq_k_len = k_end - k_start
        for q_offset in range(seq_q_len):
            visible_len = seq_k_len
            if causal:
                visible_len = min(
                    seq_k_len, max(0, q_offset + seq_k_len - seq_q_len + 1)
                )
            if visible_len == 0:
                outputs.append(mx.zeros((q.shape[1], q.shape[2]), dtype=q.dtype))
                continue
            batch_heads = []
            for head in range(q.shape[1]):
                kv_head = head // groups
                scores = mx.sum(
                    q[q_start + q_offset, head, :]
                    * k[k_start : k_start + visible_len, kv_head, :],
                    axis=-1,
                )
                weights = mx.softmax(scores * scale, axis=-1)
                batch_heads.append(
                    mx.sum(
                        weights[:, None]
                        * v[k_start : k_start + visible_len, kv_head, :],
                        axis=0,
                    )
                )
            outputs.append(mx.stack(batch_heads, axis=0))
    if not outputs:
        return mx.zeros(q.shape, dtype=q.dtype)
    return mx.stack(outputs, axis=0)


def _prefill_attention_paged_reference(
    q, k, v, k_cache, v_cache, page_rows, prefix_lens, cu_q, scale, causal
):
    import mlx.core as mx

    outputs = []
    groups = q.shape[1] // k.shape[1]
    block_size = k_cache.shape[1]
    for seq_id, (q_start, q_end, blocks, prefix_len) in enumerate(
        zip(cu_q[:-1], cu_q[1:], page_rows, prefix_lens, strict=True)
    ):
        del seq_id
        needed_blocks = (prefix_len + block_size - 1) // block_size
        if prefix_len:
            k_prefix = mx.concatenate(
                [k_cache[block : block + 1] for block in blocks[:needed_blocks]], axis=0
            )
            v_prefix = mx.concatenate(
                [v_cache[block : block + 1] for block in blocks[:needed_blocks]], axis=0
            )
            k_prefix = k_prefix.reshape(
                needed_blocks * block_size, k_cache.shape[2], k_cache.shape[3]
            )[:prefix_len]
            v_prefix = v_prefix.reshape(
                needed_blocks * block_size, v_cache.shape[2], v_cache.shape[3]
            )[:prefix_len]
            seq_k = mx.concatenate([k_prefix, k[q_start:q_end]], axis=0)
            seq_v = mx.concatenate([v_prefix, v[q_start:q_end]], axis=0)
        else:
            seq_k = k[q_start:q_end]
            seq_v = v[q_start:q_end]
        seq_q_len = q_end - q_start
        for q_offset in range(seq_q_len):
            visible_len = prefix_len + (q_offset + 1 if causal else seq_q_len)
            batch_heads = []
            for head in range(q.shape[1]):
                kv_head = head // groups
                scores = mx.sum(
                    q[q_start + q_offset, head, :] * seq_k[:visible_len, kv_head, :],
                    axis=-1,
                )
                weights = mx.softmax(scores * scale, axis=-1)
                batch_heads.append(
                    mx.sum(weights[:, None] * seq_v[:visible_len, kv_head, :], axis=0)
                )
            outputs.append(mx.stack(batch_heads, axis=0))
    if not outputs:
        return mx.zeros(q.shape, dtype=q.dtype)
    return mx.stack(outputs, axis=0)


def _scaled_arange(mx, shape, offset, divisor):
    values = mx.array(list(range(math.prod(shape))), dtype=mx.float32)
    return (values.reshape(*shape) + offset) / divisor


def _flash_attn_inputs(mx):
    q = _scaled_arange(mx, (2, 1, 4, 4), -7.0, 10.0)
    k_cache = _scaled_arange(mx, (3, 4, 2, 4), -11.0, 13.0)
    v_cache = _scaled_arange(mx, (3, 4, 2, 4), 5.0, 17.0)
    return q, k_cache, v_cache


def _flash_attn_varlen_inputs(mx):
    q = _scaled_arange(mx, (5, 4, 4), -3.0, 11.0)
    k = _scaled_arange(mx, (7, 2, 4), 2.0, 13.0)
    v = _scaled_arange(mx, (7, 2, 4), -5.0, 17.0)
    cu_q = mx.array([0, 2, 5], dtype=mx.int32)
    cu_k = mx.array([0, 3, 7], dtype=mx.int32)
    return q, k, v, cu_q, cu_k


@pytest.mark.parametrize("dtype,atol", [("float32", 1e-5), ("float16", 2e-3)])
def test_metal_decode_attention_matches_reference(dtype, atol):
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import decode_attention

    dtype = getattr(mx, dtype)
    q = mx.array(
        [
            [
                [[0.2, -0.4, 0.6, 0.8]],
                [[-0.3, 0.5, 0.7, -0.2]],
                [[0.9, -0.1, 0.4, 0.3]],
                [[-0.5, 0.2, 0.1, 0.6]],
            ],
            [
                [[0.1, 0.4, -0.6, 0.7]],
                [[-0.8, 0.3, 0.2, 0.5]],
                [[0.6, 0.1, -0.2, 0.4]],
                [[-0.1, -0.7, 0.5, 0.2]],
            ],
        ],
        dtype=dtype,
    )
    k = mx.array(
        [
            [
                [[0.2, 0.1, -0.3, 0.5], [0.4, -0.2, 0.6, 0.1], [-0.1, 0.3, 0.2, -0.4]],
                [[-0.5, 0.4, 0.1, 0.2], [0.3, 0.2, -0.6, 0.7], [0.8, -0.1, 0.5, -0.3]],
            ],
            [
                [[0.7, -0.3, 0.1, 0.2], [-0.4, 0.6, 0.5, -0.2], [0.3, 0.2, -0.1, 0.4]],
                [[-0.2, 0.8, -0.5, 0.1], [0.5, -0.4, 0.2, 0.6], [0.1, 0.3, 0.7, -0.6]],
            ],
        ],
        dtype=dtype,
    )
    v = mx.array(
        [
            [
                [[0.6, -0.2, 0.3, 0.1], [0.5, 0.4, -0.1, 0.2], [-0.3, 0.7, 0.2, 0.4]],
                [[0.1, 0.2, 0.8, -0.5], [-0.4, 0.6, 0.3, 0.7], [0.2, -0.1, 0.5, 0.9]],
            ],
            [
                [[-0.2, 0.5, 0.4, 0.3], [0.7, -0.6, 0.1, 0.2], [0.3, 0.8, -0.4, 0.6]],
                [[0.5, 0.1, -0.3, 0.7], [-0.1, 0.4, 0.9, -0.2], [0.6, -0.5, 0.2, 0.8]],
            ],
        ],
        dtype=dtype,
    )
    scale = 1 / math.sqrt(q.shape[-1])

    out = decode_attention(q, k, v, scale)
    ref = _decode_reference(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), scale
    )

    mx.eval(out, ref)
    assert out.shape == q.shape
    assert out.dtype == dtype
    assert mx.allclose(out.astype(mx.float32), ref, atol=atol, rtol=atol)


def test_metal_decode_attention_ragged_matches_reference():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import decode_attention_ragged

    q = mx.array(
        [
            [
                [[0.2, -0.4, 0.6, 0.8]],
                [[-0.3, 0.5, 0.7, -0.2]],
                [[0.9, -0.1, 0.4, 0.3]],
                [[-0.5, 0.2, 0.1, 0.6]],
            ],
            [
                [[0.1, 0.4, -0.6, 0.7]],
                [[-0.8, 0.3, 0.2, 0.5]],
                [[0.6, 0.1, -0.2, 0.4]],
                [[-0.1, -0.7, 0.5, 0.2]],
            ],
        ],
        dtype=mx.float32,
    )
    k_list = [
        mx.array(
            [
                [
                    [[0.2, 0.1, -0.3, 0.5], [0.4, -0.2, 0.6, 0.1]],
                    [[-0.5, 0.4, 0.1, 0.2], [0.3, 0.2, -0.6, 0.7]],
                ]
            ],
            dtype=mx.float32,
        ),
        mx.array(
            [
                [
                    [
                        [0.7, -0.3, 0.1, 0.2],
                        [-0.4, 0.6, 0.5, -0.2],
                        [0.3, 0.2, -0.1, 0.4],
                    ],
                    [
                        [-0.2, 0.8, -0.5, 0.1],
                        [0.5, -0.4, 0.2, 0.6],
                        [0.1, 0.3, 0.7, -0.6],
                    ],
                ]
            ],
            dtype=mx.float32,
        ),
    ]
    v_list = [
        mx.array(
            [
                [
                    [[0.6, -0.2, 0.3, 0.1], [0.5, 0.4, -0.1, 0.2]],
                    [[0.1, 0.2, 0.8, -0.5], [-0.4, 0.6, 0.3, 0.7]],
                ]
            ],
            dtype=mx.float32,
        ),
        mx.array(
            [
                [
                    [
                        [-0.2, 0.5, 0.4, 0.3],
                        [0.7, -0.6, 0.1, 0.2],
                        [0.3, 0.8, -0.4, 0.6],
                    ],
                    [
                        [0.5, 0.1, -0.3, 0.7],
                        [-0.1, 0.4, 0.9, -0.2],
                        [0.6, -0.5, 0.2, 0.8],
                    ],
                ]
            ],
            dtype=mx.float32,
        ),
    ]
    scale = 1 / math.sqrt(q.shape[-1])

    out = decode_attention_ragged(q, k_list, v_list, scale)
    ref = _decode_ragged_reference(q, k_list, v_list, scale)

    mx.eval(out, ref)
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert mx.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_metal_decode_attention_rejects_unsupported_shape():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import decode_attention

    q = mx.zeros((1, 2, 2, 4), dtype=mx.float32)
    k = mx.zeros((1, 1, 3, 4), dtype=mx.float32)
    v = mx.zeros((1, 1, 3, 4), dtype=mx.float32)

    with pytest.raises(ValueError, match="query must have shape"):
        decode_attention(q, k, v, 0.5)


def test_metal_decode_attention_ragged_rejects_mismatched_list_length():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import decode_attention_ragged

    q = mx.zeros((2, 2, 1, 4), dtype=mx.float32)
    k = mx.zeros((1, 1, 3, 4), dtype=mx.float32)
    v = mx.zeros((1, 1, 3, 4), dtype=mx.float32)

    with pytest.raises(ValueError, match="list lengths must match"):
        decode_attention_ragged(q, [k], [v], 0.5)


def _paged_kv_scatter_reference(k, v, k_cache, v_cache, slot_mapping):
    import mlx.core as mx

    expected_k = mx.array(k_cache)
    expected_v = mx.array(v_cache)
    mx.eval(slot_mapping)
    for token, slot in enumerate(slot_mapping.tolist()):
        block = slot // k_cache.shape[1]
        block_offset = slot % k_cache.shape[1]
        expected_k[block : block + 1, block_offset : block_offset + 1, :, :] = k[
            token : token + 1
        ].reshape(1, 1, k.shape[1], k.shape[2])
        expected_v[block : block + 1, block_offset : block_offset + 1, :, :] = v[
            token : token + 1
        ].reshape(1, 1, v.shape[1], v.shape[2])
    return expected_k, expected_v


def test_metal_paged_kv_scatter_matches_reference_across_block_boundaries():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import paged_kv_scatter

    k = _scaled_arange(mx, (5, 2, 4), -3.0, 11.0)
    v = _scaled_arange(mx, (5, 2, 4), 7.0, 13.0)
    k_cache = _scaled_arange(mx, (4, 2, 2, 4), 2.0, 17.0)
    v_cache = _scaled_arange(mx, (4, 2, 2, 4), -5.0, 19.0)
    slot_mapping = mx.array([1, 2, 3, 4, 7], dtype=mx.int32)
    expected_k, expected_v = _paged_kv_scatter_reference(
        k, v, k_cache, v_cache, slot_mapping
    )

    result = paged_kv_scatter(k, v, k_cache, v_cache, slot_mapping)
    mx.eval(k_cache, v_cache, expected_k, expected_v)

    assert result is None
    assert mx.allclose(k_cache, expected_k, atol=1e-5, rtol=1e-5)
    assert mx.allclose(v_cache, expected_v, atol=1e-5, rtol=1e-5)


def test_metal_paged_kv_scatter_rejects_invalid_metadata():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import paged_kv_scatter

    k = mx.zeros((2, 1, 4), dtype=mx.float32)
    v = mx.zeros((2, 1, 4), dtype=mx.float32)
    k_cache = mx.zeros((2, 2, 1, 4), dtype=mx.float32)
    v_cache = mx.zeros((2, 2, 1, 4), dtype=mx.float32)
    slot_mapping = mx.array([1, 2], dtype=mx.int32)

    with pytest.raises(ValueError, match="slot_mapping must have shape"):
        paged_kv_scatter(k, v, k_cache, v_cache, mx.zeros((2, 1), dtype=mx.int32))
    with pytest.raises(TypeError, match="slot_mapping must be an int32 array"):
        paged_kv_scatter(k, v, k_cache, v_cache, slot_mapping.astype(mx.float32))
    with pytest.raises(ValueError, match="head dimensions must match"):
        paged_kv_scatter(
            k,
            v,
            mx.zeros((2, 2, 1, 8), dtype=mx.float32),
            mx.zeros((2, 2, 1, 8), dtype=mx.float32),
            slot_mapping,
        )
    with pytest.raises(
        ValueError, match="slot_mapping entries must be in cache slot range"
    ):
        paged_kv_scatter(k, v, k_cache, v_cache, mx.array([0, 4], dtype=mx.int32))


def test_metal_decode_attention_paged_matches_reference_with_mixed_lengths_and_gqa():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import decode_attention_paged

    q = _scaled_arange(mx, (3, 4, 1, 4), -7.0, 10.0)
    k_cache = _scaled_arange(mx, (6, 2, 2, 4), -11.0, 13.0)
    v_cache = _scaled_arange(mx, (6, 2, 2, 4), 5.0, 17.0)
    block_tables = mx.array([[2, 0, 4], [1, 3, 5], [4, 2, 1]], dtype=mx.int32)
    context_lens = mx.array([3, 5, 4], dtype=mx.int32)
    scale = 1 / math.sqrt(q.shape[-1])

    out = decode_attention_paged(q, k_cache, v_cache, block_tables, context_lens, scale)
    ref = _decode_paged_reference(
        q, k_cache, v_cache, [[2, 0, 4], [1, 3, 5], [4, 2, 1]], [3, 5, 4], scale
    )

    mx.eval(out, ref)
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert mx.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_metal_decode_attention_paged_rejects_invalid_metadata():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import decode_attention_paged

    q = mx.zeros((2, 4, 1, 4), dtype=mx.float32)
    k_cache = mx.zeros((4, 2, 2, 4), dtype=mx.float32)
    v_cache = mx.zeros((4, 2, 2, 4), dtype=mx.float32)
    block_tables = mx.array([[0, 1], [2, 3]], dtype=mx.int32)
    context_lens = mx.array([2, 4], dtype=mx.int32)

    with pytest.raises(ValueError, match="block_tables must have shape"):
        decode_attention_paged(
            q, k_cache, v_cache, mx.zeros((2,), dtype=mx.int32), context_lens, 0.5
        )
    with pytest.raises(ValueError, match="context_lens must have shape"):
        decode_attention_paged(
            q, k_cache, v_cache, block_tables, mx.ones((2, 1), dtype=mx.int32), 0.5
        )
    with pytest.raises(TypeError, match="int32 arrays"):
        decode_attention_paged(
            q, k_cache, v_cache, block_tables.astype(mx.float32), context_lens, 0.5
        )
    with pytest.raises(ValueError, match="context_lens entries must be in"):
        decode_attention_paged(
            q, k_cache, v_cache, block_tables, mx.array([0, 4], dtype=mx.int32), 0.5
        )
    with pytest.raises(ValueError, match="context_lens entries must be in"):
        decode_attention_paged(
            q, k_cache, v_cache, block_tables, mx.array([2, 5], dtype=mx.int32), 0.5
        )
    with pytest.raises(ValueError, match="block_tables entries must index"):
        decode_attention_paged(
            q,
            k_cache,
            v_cache,
            mx.array([[0, 4], [2, 3]], dtype=mx.int32),
            context_lens,
            0.5,
        )
    with pytest.raises(ValueError, match="head dimensions must match"):
        decode_attention_paged(
            q,
            mx.zeros((4, 2, 2, 8), dtype=mx.float32),
            mx.zeros((4, 2, 2, 8), dtype=mx.float32),
            block_tables,
            context_lens,
            0.5,
        )
    with pytest.raises(ValueError, match="dtypes must match"):
        decode_attention_paged(
            q,
            k_cache.astype(mx.float16),
            v_cache.astype(mx.float16),
            block_tables,
            context_lens,
            0.5,
        )


def test_metal_flash_attn_with_kvcache_dense_matches_reference():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import flash_attn_with_kvcache

    q, k_cache, v_cache = _flash_attn_inputs(mx)
    k_cache = k_cache[: q.shape[0]]
    v_cache = v_cache[: q.shape[0]]
    scale = 1 / math.sqrt(q.shape[-1])

    out = flash_attn_with_kvcache(q, k_cache, v_cache, softmax_scale=scale)
    ref = _flash_attn_reference(q, k_cache, v_cache, [4, 4], [0, 1], scale)

    mx.eval(out, ref)
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert mx.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_metal_flash_attn_with_kvcache_ragged_matches_reference():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import flash_attn_with_kvcache

    q, k_cache, v_cache = _flash_attn_inputs(mx)
    seq_lens = mx.array([2, 3], dtype=mx.int32)
    batch_indices = mx.array([2, 0], dtype=mx.int32)
    scale = 1 / math.sqrt(q.shape[-1])

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=seq_lens,
        cache_batch_idx=batch_indices,
        softmax_scale=scale,
    )
    ref = _flash_attn_reference(q, k_cache, v_cache, [2, 3], [2, 0], scale)

    mx.eval(out, ref)
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert mx.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_metal_flash_attn_with_kvcache_appends_new_kv():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import flash_attn_with_kvcache

    q, k_cache, v_cache = _flash_attn_inputs(mx)
    k_cache_before = mx.array(k_cache)
    v_cache_before = mx.array(v_cache)
    k_new = _scaled_arange(mx, (2, 1, 2, 4), 3.0, 19.0)
    v_new = _scaled_arange(mx, (2, 1, 2, 4), -5.0, 23.0)
    scale = 1 / math.sqrt(q.shape[-1])

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k_new,
        v=v_new,
        cache_seqlens=mx.array([1, 2], dtype=mx.int32),
        cache_batch_idx=mx.array([2, 0], dtype=mx.int32),
        softmax_scale=scale,
    )
    mx.eval(k_cache, v_cache)

    expected_k = mx.array(k_cache_before)
    expected_v = mx.array(v_cache_before)
    expected_k[2:3, 1:2, :, :] = k_new[0:1]
    expected_v[2:3, 1:2, :, :] = v_new[0:1]
    expected_k[0:1, 2:3, :, :] = k_new[1:2]
    expected_v[0:1, 2:3, :, :] = v_new[1:2]
    ref = _flash_attn_reference(q, expected_k, expected_v, [2, 3], [2, 0], scale)

    mx.eval(out, ref, expected_k, expected_v)
    assert mx.allclose(k_cache, expected_k)
    assert mx.allclose(v_cache, expected_v)
    assert mx.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_metal_flash_attn_with_kvcache_scalar_cache_seqlens_matches_reference():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import flash_attn_with_kvcache

    q, k_cache, v_cache = _flash_attn_inputs(mx)
    scale = 1 / math.sqrt(q.shape[-1])

    out = flash_attn_with_kvcache(
        q, k_cache, v_cache, cache_seqlens=3, softmax_scale=scale
    )
    ref = _flash_attn_reference(q, k_cache, v_cache, [3, 3], [0, 1], scale)

    mx.eval(out, ref)
    assert mx.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_metal_flash_attn_with_kvcache_paged_matches_reference():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import flash_attn_with_kvcache

    q = _scaled_arange(mx, (2, 1, 4, 4), -7.0, 10.0)
    k_cache = _scaled_arange(mx, (4, 2, 2, 4), -11.0, 13.0)
    v_cache = _scaled_arange(mx, (4, 2, 2, 4), 5.0, 17.0)
    page_table = mx.array([[2, 0], [1, 3]], dtype=mx.int32)
    seq_lens = mx.array([3, 4], dtype=mx.int32)
    scale = 1 / math.sqrt(q.shape[-1])

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=seq_lens,
        page_table=page_table,
        softmax_scale=scale,
    )
    ref = _flash_attn_paged_reference(
        q, k_cache, v_cache, [[2, 0], [1, 3]], [3, 4], scale
    )

    mx.eval(out, ref)
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert mx.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_metal_flash_attn_with_kvcache_rejects_unsupported_options():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import flash_attn_with_kvcache

    q, k_cache, v_cache = _flash_attn_inputs(mx)

    with pytest.raises(NotImplementedError, match="rotary_cos"):
        flash_attn_with_kvcache(
            q, k_cache, v_cache, rotary_cos=mx.ones((1,), dtype=mx.float32)
        )
    with pytest.raises(NotImplementedError, match="qv"):
        flash_attn_with_kvcache(q, k_cache, v_cache, qv=mx.ones(q.shape, dtype=q.dtype))
    with pytest.raises(NotImplementedError, match="softcap"):
        flash_attn_with_kvcache(q, k_cache, v_cache, softcap=1.0)
    with pytest.raises(NotImplementedError, match="score_mod"):
        flash_attn_with_kvcache(q, k_cache, v_cache, score_mod=lambda score, *_: score)


def test_metal_flash_attn_with_kvcache_rejects_invalid_page_table():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import flash_attn_with_kvcache

    q, _, _ = _flash_attn_inputs(mx)
    paged_k = _scaled_arange(mx, (4, 2, 2, 4), -11.0, 13.0)
    paged_v = _scaled_arange(mx, (4, 2, 2, 4), 5.0, 17.0)

    with pytest.raises(ValueError, match="2-D"):
        flash_attn_with_kvcache(
            q, paged_k, paged_v, page_table=mx.array([0, 1], dtype=mx.int32)
        )
    with pytest.raises(ValueError, match="shape"):
        flash_attn_with_kvcache(
            q, paged_k, paged_v, page_table=mx.array([[0, 1]], dtype=mx.int32)
        )
    with pytest.raises(ValueError, match="at least one block"):
        flash_attn_with_kvcache(
            q, paged_k, paged_v, page_table=mx.zeros((2, 0), dtype=mx.int32)
        )
    with pytest.raises(ValueError, match="entries must be in"):
        flash_attn_with_kvcache(
            q,
            paged_k,
            paged_v,
            cache_seqlens=mx.array([5, 1], dtype=mx.int32),
            page_table=mx.array([[0, 1], [2, 3]], dtype=mx.int32),
        )
    with pytest.raises(ValueError, match="entries must index"):
        flash_attn_with_kvcache(
            q, paged_k, paged_v, page_table=mx.array([[0, 4], [1, 2]], dtype=mx.int32)
        )
    with pytest.raises(NotImplementedError, match="cache_batch_idx"):
        flash_attn_with_kvcache(
            q,
            paged_k,
            paged_v,
            cache_batch_idx=mx.array([0, 1], dtype=mx.int32),
            page_table=mx.array([[0, 1], [2, 3]], dtype=mx.int32),
        )
    with pytest.raises(NotImplementedError, match="append"):
        flash_attn_with_kvcache(
            q,
            paged_k,
            paged_v,
            k=mx.zeros((2, 1, 2, 4), dtype=mx.float32),
            v=mx.zeros((2, 1, 2, 4), dtype=mx.float32),
            page_table=mx.array([[0, 1], [2, 3]], dtype=mx.int32),
        )


def test_metal_flash_attn_with_kvcache_rejects_invalid_cache_metadata():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import flash_attn_with_kvcache

    q, k_cache, v_cache = _flash_attn_inputs(mx)

    with pytest.raises(ValueError, match="cache_seqlens must have shape"):
        flash_attn_with_kvcache(
            q, k_cache, v_cache, cache_seqlens=mx.array([1], dtype=mx.int32)
        )
    with pytest.raises(ValueError, match="cache_batch_idx must have shape"):
        flash_attn_with_kvcache(
            q, k_cache, v_cache, cache_batch_idx=mx.array([0], dtype=mx.int32)
        )
    with pytest.raises(ValueError, match="entries must index"):
        flash_attn_with_kvcache(
            q, k_cache, v_cache, cache_batch_idx=mx.array([0, 3], dtype=mx.int32)
        )
    with pytest.raises(ValueError, match="entries must be positive"):
        flash_attn_with_kvcache(
            q, k_cache, v_cache, cache_seqlens=mx.array([0, 1], dtype=mx.int32)
        )
    with pytest.raises(ValueError, match="provided together"):
        flash_attn_with_kvcache(
            q, k_cache, v_cache, k=mx.zeros((2, 1, 2, 4), dtype=mx.float32)
        )


def test_metal_flash_attn_varlen_func_noncausal_matches_reference():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import flash_attn_varlen_func

    q, k, v, cu_q, cu_k = _flash_attn_varlen_inputs(mx)
    scale = 1 / math.sqrt(q.shape[-1])

    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=3,
        max_seqlen_k=4,
        softmax_scale=scale,
    )
    ref = _flash_attn_varlen_reference(
        q, k, v, [0, 2, 5], [0, 3, 7], scale, causal=False
    )

    mx.eval(out, ref)
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert mx.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_metal_flash_attn_varlen_func_causal_matches_reference():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import flash_attn_varlen_func

    q, k, v, cu_q, cu_k = _flash_attn_varlen_inputs(mx)
    scale = 1 / math.sqrt(q.shape[-1])

    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=3,
        max_seqlen_k=4,
        softmax_scale=scale,
        causal=True,
    )
    ref = _flash_attn_varlen_reference(
        q, k, v, [0, 2, 5], [0, 3, 7], scale, causal=True
    )

    mx.eval(out, ref)
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert mx.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_metal_flash_attn_varlen_func_rejects_unsupported_options():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import flash_attn_varlen_func

    q, k, v, cu_q, cu_k = _flash_attn_varlen_inputs(mx)

    with pytest.raises(NotImplementedError, match="page_table"):
        flash_attn_varlen_func(
            q, k, v, cu_q, cu_k, page_table=mx.ones((2, 1), dtype=mx.int32)
        )
    with pytest.raises(NotImplementedError, match="qv"):
        flash_attn_varlen_func(q, k, v, cu_q, cu_k, qv=mx.ones(q.shape, dtype=q.dtype))
    with pytest.raises(NotImplementedError, match="window_size"):
        flash_attn_varlen_func(q, k, v, cu_q, cu_k, window_size=(0, 0))
    with pytest.raises(NotImplementedError, match="softcap"):
        flash_attn_varlen_func(q, k, v, cu_q, cu_k, softcap=1.0)
    with pytest.raises(NotImplementedError, match="return_softmax_lse"):
        flash_attn_varlen_func(q, k, v, cu_q, cu_k, return_softmax_lse=True)


def test_metal_flash_attn_varlen_func_rejects_invalid_metadata():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import flash_attn_varlen_func

    q, k, v, cu_q, cu_k = _flash_attn_varlen_inputs(mx)

    with pytest.raises(ValueError, match="same length"):
        flash_attn_varlen_func(q, k, v, mx.array([0, 5], dtype=mx.int32), cu_k)
    with pytest.raises(ValueError, match="start at 0 and end"):
        flash_attn_varlen_func(q, k, v, mx.array([1, 2, 5], dtype=mx.int32), cu_k)
    with pytest.raises(ValueError, match="nondecreasing"):
        flash_attn_varlen_func(q, k, v, mx.array([0, 6, 5], dtype=mx.int32), cu_k)
    with pytest.raises(ValueError, match="max_seqlen_q"):
        flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_seqlen_q=2)
    with pytest.raises(ValueError, match="max_seqlen_k"):
        flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_seqlen_k=3)


def test_metal_prefill_attention_paged_no_prefix_matches_reference():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import prefill_attention_paged

    q = _scaled_arange(mx, (5, 4, 4), -3.0, 11.0)
    k = _scaled_arange(mx, (5, 2, 4), 2.0, 13.0)
    v = _scaled_arange(mx, (5, 2, 4), -5.0, 17.0)
    k_cache = _scaled_arange(mx, (4, 2, 2, 4), -11.0, 19.0)
    v_cache = _scaled_arange(mx, (4, 2, 2, 4), 7.0, 23.0)
    block_tables = mx.array([[2, 0], [1, 3]], dtype=mx.int32)
    prefix_lens = mx.array([0, 0], dtype=mx.int32)
    cu_q = mx.array([0, 2, 5], dtype=mx.int32)
    scale = 1 / math.sqrt(q.shape[-1])

    out = prefill_attention_paged(
        q, k, v, k_cache, v_cache, block_tables, prefix_lens, cu_q, scale, causal=True
    )
    ref = _prefill_attention_paged_reference(
        q, k, v, k_cache, v_cache, [[2, 0], [1, 3]], [0, 0], [0, 2, 5], scale, True
    )

    mx.eval(out, ref)
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert mx.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_metal_prefill_attention_paged_partial_prefix_matches_reference():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import prefill_attention_paged

    q = _scaled_arange(mx, (5, 4, 4), -3.0, 11.0)
    k = _scaled_arange(mx, (5, 2, 4), 2.0, 13.0)
    v = _scaled_arange(mx, (5, 2, 4), -5.0, 17.0)
    k_cache = _scaled_arange(mx, (6, 2, 2, 4), -11.0, 19.0)
    v_cache = _scaled_arange(mx, (6, 2, 2, 4), 7.0, 23.0)
    block_tables = mx.array([[2, 0, 4], [1, 3, 5]], dtype=mx.int32)
    prefix_lens = mx.array([3, 5], dtype=mx.int32)
    cu_q = mx.array([0, 2, 5], dtype=mx.int32)
    scale = 1 / math.sqrt(q.shape[-1])

    out = prefill_attention_paged(
        q, k, v, k_cache, v_cache, block_tables, prefix_lens, cu_q, scale, causal=True
    )
    ref = _prefill_attention_paged_reference(
        q,
        k,
        v,
        k_cache,
        v_cache,
        [[2, 0, 4], [1, 3, 5]],
        [3, 5],
        [0, 2, 5],
        scale,
        True,
    )

    mx.eval(out, ref)
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert mx.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_metal_prefill_attention_paged_full_prefix_noncausal_matches_reference():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import prefill_attention_paged

    q = _scaled_arange(mx, (3, 4, 4), -3.0, 11.0)
    k = _scaled_arange(mx, (3, 2, 4), 2.0, 13.0)
    v = _scaled_arange(mx, (3, 2, 4), -5.0, 17.0)
    k_cache = _scaled_arange(mx, (6, 2, 2, 4), -11.0, 19.0)
    v_cache = _scaled_arange(mx, (6, 2, 2, 4), 7.0, 23.0)
    block_tables = mx.array([[2, 0, 4], [1, 3, 5]], dtype=mx.int32)
    prefix_lens = mx.array([6, 4], dtype=mx.int32)
    cu_q = mx.array([0, 1, 3], dtype=mx.int32)
    scale = 1 / math.sqrt(q.shape[-1])

    out = prefill_attention_paged(
        q, k, v, k_cache, v_cache, block_tables, prefix_lens, cu_q, scale, causal=False
    )
    ref = _prefill_attention_paged_reference(
        q,
        k,
        v,
        k_cache,
        v_cache,
        [[2, 0, 4], [1, 3, 5]],
        [6, 4],
        [0, 1, 3],
        scale,
        False,
    )

    mx.eval(out, ref)
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert mx.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_metal_prefill_attention_paged_float16_partial_prefix_matches_reference():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import prefill_attention_paged

    q = _scaled_arange(mx, (5, 4, 4), -3.0, 11.0).astype(mx.float16)
    k = _scaled_arange(mx, (5, 2, 4), 2.0, 13.0).astype(mx.float16)
    v = _scaled_arange(mx, (5, 2, 4), -5.0, 17.0).astype(mx.float16)
    k_cache = _scaled_arange(mx, (6, 2, 2, 4), -11.0, 19.0).astype(mx.float16)
    v_cache = _scaled_arange(mx, (6, 2, 2, 4), 7.0, 23.0).astype(mx.float16)
    block_tables = mx.array([[2, 0, 4], [1, 3, 5]], dtype=mx.int32)
    prefix_lens = mx.array([3, 5], dtype=mx.int32)
    cu_q = mx.array([0, 2, 5], dtype=mx.int32)
    scale = 1 / math.sqrt(q.shape[-1])

    out = prefill_attention_paged(
        q, k, v, k_cache, v_cache, block_tables, prefix_lens, cu_q, scale, causal=True
    )
    ref = _prefill_attention_paged_reference(
        q,
        k,
        v,
        k_cache,
        v_cache,
        [[2, 0, 4], [1, 3, 5]],
        [3, 5],
        [0, 2, 5],
        scale,
        True,
    )

    mx.eval(out, ref)
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert mx.allclose(out, ref, atol=4e-3, rtol=4e-3)


def test_metal_prefill_attention_paged_native_rejects_invalid_metadata():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel import _metal

    q = mx.contiguous(_scaled_arange(mx, (5, 4, 4), -3.0, 11.0))
    k = mx.contiguous(_scaled_arange(mx, (5, 2, 4), 2.0, 13.0))
    v = mx.contiguous(_scaled_arange(mx, (5, 2, 4), -5.0, 17.0))
    k_cache = mx.contiguous(_scaled_arange(mx, (4, 2, 2, 4), -11.0, 19.0))
    v_cache = mx.contiguous(_scaled_arange(mx, (4, 2, 2, 4), 7.0, 23.0))
    out = mx.zeros(q.shape, dtype=q.dtype)
    block_tables = mx.contiguous(mx.array([[2, 0], [1, 3]], dtype=mx.int32))
    prefix_lens = mx.contiguous(mx.array([1, 2], dtype=mx.int32))
    cu_q = mx.contiguous(mx.array([0, 2, 5], dtype=mx.int32))
    mx.eval(q, k, v, k_cache, v_cache, out, block_tables, prefix_lens, cu_q)
    scale = 1 / math.sqrt(q.shape[-1])

    def call(block_tables_arg=block_tables, prefix_lens_arg=prefix_lens, cu_q_arg=cu_q):
        block_tables_arg = mx.contiguous(block_tables_arg)
        prefix_lens_arg = mx.contiguous(prefix_lens_arg)
        cu_q_arg = mx.contiguous(cu_q_arg)
        mx.eval(block_tables_arg, prefix_lens_arg, cu_q_arg)
        return _metal.prefill_attention_paged(
            out,
            q,
            k,
            v,
            k_cache,
            v_cache,
            block_tables_arg,
            prefix_lens_arg,
            cu_q_arg,
            scale,
            True,
        )

    with pytest.raises(RuntimeError, match="cu_seqlens_q must start at 0"):
        call(cu_q_arg=mx.array([1, 2, 5], dtype=mx.int32))
    with pytest.raises(RuntimeError, match="cu_seqlens_q must end at total_q"):
        call(cu_q_arg=mx.array([0, 2, 4], dtype=mx.int32))
    with pytest.raises(RuntimeError, match="cu_seqlens_q must be nondecreasing"):
        call(
            block_tables_arg=mx.array([[2, 0], [1, 3], [0, 2]], dtype=mx.int32),
            prefix_lens_arg=mx.array([1, 2, 1], dtype=mx.int32),
            cu_q_arg=mx.array([0, 4, 3, 5], dtype=mx.int32),
        )
    with pytest.raises(RuntimeError, match="prefix_lens entries"):
        call(prefix_lens_arg=mx.array([1, 5], dtype=mx.int32))
    with pytest.raises(RuntimeError, match="block_tables entries"):
        call(block_tables_arg=mx.array([[4, 0], [1, 3]], dtype=mx.int32))


def test_metal_prefill_attention_paged_rejects_invalid_metadata():
    mx = pytest.importorskip("mlx.core")
    from sgl_kernel.metal import prefill_attention_paged

    q = _scaled_arange(mx, (5, 4, 4), -3.0, 11.0)
    k = _scaled_arange(mx, (5, 2, 4), 2.0, 13.0)
    v = _scaled_arange(mx, (5, 2, 4), -5.0, 17.0)
    k_cache = _scaled_arange(mx, (4, 2, 2, 4), -11.0, 19.0)
    v_cache = _scaled_arange(mx, (4, 2, 2, 4), 7.0, 23.0)
    block_tables = mx.array([[2, 0], [1, 3]], dtype=mx.int32)
    prefix_lens = mx.array([1, 2], dtype=mx.int32)
    cu_q = mx.array([0, 2, 5], dtype=mx.int32)
    scale = 1 / math.sqrt(q.shape[-1])

    with pytest.raises(ValueError, match="block_tables must have shape"):
        prefill_attention_paged(
            q,
            k,
            v,
            k_cache,
            v_cache,
            mx.array([0, 1], dtype=mx.int32),
            prefix_lens,
            cu_q,
            scale,
        )
    with pytest.raises(ValueError, match="prefix_lens must have shape"):
        prefill_attention_paged(
            q,
            k,
            v,
            k_cache,
            v_cache,
            block_tables,
            mx.array([1], dtype=mx.int32),
            cu_q,
            scale,
        )
    with pytest.raises(ValueError, match="cu_seqlens_q must have shape"):
        prefill_attention_paged(
            q,
            k,
            v,
            k_cache,
            v_cache,
            block_tables,
            prefix_lens,
            mx.array([0, 5], dtype=mx.int32),
            scale,
        )
    with pytest.raises(TypeError, match="must be int32"):
        prefill_attention_paged(
            q,
            k,
            v,
            k_cache,
            v_cache,
            block_tables.astype(mx.float32),
            prefix_lens,
            cu_q,
            scale,
        )
    with pytest.raises(ValueError, match="prefix_lens entries"):
        prefill_attention_paged(
            q,
            k,
            v,
            k_cache,
            v_cache,
            block_tables,
            mx.array([1, 5], dtype=mx.int32),
            cu_q,
            scale,
        )
    with pytest.raises(ValueError, match="block_tables entries"):
        prefill_attention_paged(
            q,
            k,
            v,
            k_cache,
            v_cache,
            mx.array([[4, 0], [1, 3]], dtype=mx.int32),
            prefix_lens,
            cu_q,
            scale,
        )
