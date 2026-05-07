"""Python entry points for the sgl_kernel Metal extension."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx

_METALLIB_NAME = "sgl_metal_kernels.metallib"
_MAX_DECODE_HEAD_DIM = 256

try:
    from . import _metal

    _metallib_path = Path(_metal.__file__).resolve().parent / _METALLIB_NAME
    if not _metallib_path.is_file():
        raise ImportError(
            f"{_METALLIB_NAME} not found next to sgl_kernel._metal at {_metallib_path}"
        )
    _metal.register_library(str(_metallib_path))
except ImportError as _exc:  # pragma: no cover - import guarded at call time
    _metal = None
    _IMPORT_ERROR: Exception | None = _exc
else:
    _IMPORT_ERROR = None


def _require_metal_extension():
    if _metal is None:
        raise ImportError("sgl_kernel._metal is not available") from _IMPORT_ERROR
    return _metal


def _require_array(x, name: str, mx):
    if not isinstance(x, mx.array):
        raise TypeError(f"{name} must be an MLX array")


def _require_float_dtype(x: "mx.array", op_name: str, mx) -> None:
    if x.dtype not in (mx.float16, mx.float32):
        raise TypeError(f"{op_name} supports only float16 and float32 arrays")


def _scale_value(scale: float, op_name: str) -> float:
    scale = float(scale)
    if not scale > 0.0:
        raise ValueError(f"{op_name} scale must be positive")
    return scale


def _validate_decode_tensor_family(
    q: "mx.array",
    k: "mx.array",
    v: "mx.array",
    op_name: str,
    mx,
    *,
    require_dense_batch: bool,
) -> None:
    for name, x in (("query", q), ("key cache", k), ("value cache", v)):
        _require_array(x, f"{op_name} {name}", mx)

    if q.ndim != 4 or q.shape[2] != 1:
        raise ValueError(f"{op_name} query must have shape (B, H, 1, D)")
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"{op_name} K/V caches must have shape (B, KVH, S, D)")
    if k.shape != v.shape:
        raise ValueError(f"{op_name} K/V cache shapes must match")
    if require_dense_batch and q.shape[0] != k.shape[0]:
        raise ValueError(f"{op_name} query and K/V batch dimensions must match")
    if not require_dense_batch and k.shape[0] != 1:
        raise ValueError(f"{op_name} ragged K/V entries must have batch dimension 1")

    for name, x in (("query", q), ("key cache", k), ("value cache", v)):
        _require_float_dtype(x, op_name, mx)
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"{op_name} query and K/V dtypes must match")
    if q.shape[3] != k.shape[3]:
        raise ValueError(f"{op_name} head dimensions must match")
    if k.shape[1] == 0 or q.shape[1] % k.shape[1] != 0:
        raise ValueError(f"{op_name} query heads must be divisible by KV heads")
    if k.shape[2] == 0:
        raise ValueError(f"{op_name} K/V sequence length must be positive")
    if q.shape[3] == 0 or q.shape[3] > _MAX_DECODE_HEAD_DIM:
        raise ValueError(f"{op_name} head dimension must be in the range [1, 256]")


def _normalize_cache_seqlens(
    cache_seqlens, batch_size: int, max_len: int, mx
) -> list[int]:
    if cache_seqlens is None:
        return [max_len] * batch_size
    if isinstance(cache_seqlens, Integral):
        value = int(cache_seqlens)
        if value < 0 or value > max_len:
            raise ValueError(
                "flash_attn_with_kvcache cache_seqlens entries must be in [0, seqlen_cache]"
            )
        return [value] * batch_size

    _require_array(cache_seqlens, "flash_attn_with_kvcache cache_seqlens", mx)
    if cache_seqlens.ndim != 1 or cache_seqlens.shape[0] != batch_size:
        raise ValueError("flash_attn_with_kvcache cache_seqlens must have shape (B,)")
    mx.eval(cache_seqlens)
    values = [int(x) for x in cache_seqlens.tolist()]
    if any(x < 0 or x > max_len for x in values):
        raise ValueError(
            "flash_attn_with_kvcache cache_seqlens entries must be in [0, seqlen_cache]"
        )
    return values


def _normalize_cache_batch_idx(
    cache_batch_idx, batch_size: int, cache_batch: int, mx
) -> list[int]:
    if cache_batch_idx is None:
        indices = list(range(batch_size))
    else:
        _require_array(cache_batch_idx, "flash_attn_with_kvcache cache_batch_idx", mx)
        if cache_batch_idx.ndim != 1 or cache_batch_idx.shape[0] != batch_size:
            raise ValueError(
                "flash_attn_with_kvcache cache_batch_idx must have shape (B,)"
            )
        mx.eval(cache_batch_idx)
        indices = [int(x) for x in cache_batch_idx.tolist()]
    if any(x < 0 or x >= cache_batch for x in indices):
        raise ValueError(
            "flash_attn_with_kvcache cache_batch_idx entries must index the KV cache batch"
        )
    return indices


def _require_int_vector(x, name: str, length: int | None, mx) -> list[int]:
    _require_array(x, name, mx)
    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1-D MLX array")
    if length is not None and x.shape[0] != length:
        raise ValueError(f"{name} must have shape ({length},)")
    mx.eval(x)
    return [int(value) for value in x.tolist()]


def _validate_cu_seqlens(values: list[int], total: int, name: str) -> None:
    if len(values) < 2:
        raise ValueError(f"{name} must contain at least two entries")
    if values[0] != 0 or values[-1] != total:
        raise ValueError(f"{name} must start at 0 and end at the packed tensor length")
    if any(curr < prev for prev, curr in zip(values, values[1:])):
        raise ValueError(f"{name} entries must be nondecreasing")


def _normalize_page_table(page_table, batch_size: int | None, mx) -> list[list[int]]:
    _require_array(page_table, "flash_attn_with_kvcache page_table", mx)
    if page_table.ndim != 2:
        raise ValueError("flash_attn_with_kvcache page_table must be a 2-D MLX array")
    if page_table.dtype != mx.int32:
        raise TypeError("flash_attn_with_kvcache page_table must be an int32 array")
    if batch_size is not None and page_table.shape[0] != batch_size:
        raise ValueError(
            "flash_attn_with_kvcache page_table must have shape (B, max_blocks)"
        )
    if page_table.shape[1] == 0:
        raise ValueError(
            "flash_attn_with_kvcache page_table must contain at least one block column"
        )
    mx.eval(page_table)
    return [[int(block) for block in row] for row in page_table.tolist()]


def _validate_paged_metadata(
    block_tables: "mx.array",
    context_lens: "mx.array",
    batch_size: int,
    num_blocks: int,
    block_size: int,
    op_name: str,
    mx,
) -> list[list[int]]:
    _require_array(block_tables, f"{op_name} block_tables", mx)
    _require_array(context_lens, f"{op_name} context_lens", mx)
    if block_tables.ndim != 2 or block_tables.shape[0] != batch_size:
        raise ValueError(f"{op_name} block_tables must have shape (B, max_blocks)")
    if context_lens.ndim != 1 or context_lens.shape[0] != batch_size:
        raise ValueError(f"{op_name} context_lens must have shape (B,)")
    if block_tables.dtype != mx.int32 or context_lens.dtype != mx.int32:
        raise TypeError(f"{op_name} block_tables and context_lens must be int32 arrays")
    if block_tables.shape[1] == 0:
        raise ValueError(f"{op_name} max block count must be positive")
    max_cache_len = block_tables.shape[1] * block_size
    mx.eval(block_tables, context_lens)
    block_rows = [[int(block) for block in row] for row in block_tables.tolist()]
    lens = [int(length) for length in context_lens.tolist()]
    if any(length <= 0 or length > max_cache_len for length in lens):
        raise ValueError(
            f"{op_name} context_lens entries must be in [1, max_blocks * block_size]"
        )
    for blocks in block_rows:
        if any(block < 0 or block >= num_blocks for block in blocks):
            raise ValueError(
                f"{op_name} block_tables entries must index KV cache blocks"
            )
    return block_rows


def _validate_prefill_paged_metadata(
    block_tables: "mx.array",
    prefix_lens: "mx.array",
    cu_seqlens_q: "mx.array",
    batch_size: int,
    total_q: int,
    num_blocks: int,
    block_size: int,
    op_name: str,
    mx,
) -> None:
    _require_array(block_tables, f"{op_name} block_tables", mx)
    _require_array(prefix_lens, f"{op_name} prefix_lens", mx)
    _require_array(cu_seqlens_q, f"{op_name} cu_seqlens_q", mx)
    if block_tables.ndim != 2 or block_tables.shape[0] != batch_size:
        raise ValueError(f"{op_name} block_tables must have shape (B, max_blocks)")
    if prefix_lens.ndim != 1 or prefix_lens.shape[0] != batch_size:
        raise ValueError(f"{op_name} prefix_lens must have shape (B,)")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_q.shape[0] != batch_size + 1:
        raise ValueError(f"{op_name} cu_seqlens_q must have shape (B + 1,)")
    if (
        block_tables.dtype != mx.int32
        or prefix_lens.dtype != mx.int32
        or cu_seqlens_q.dtype != mx.int32
    ):
        raise TypeError(
            f"{op_name} block_tables, prefix_lens, and cu_seqlens_q must be int32 arrays"
        )
    if block_tables.shape[1] == 0:
        raise ValueError(f"{op_name} max block count must be positive")
    max_prefix_len = block_tables.shape[1] * block_size
    mx.eval(block_tables, prefix_lens, cu_seqlens_q)
    block_rows = [[int(block) for block in row] for row in block_tables.tolist()]
    prefix_values = [int(length) for length in prefix_lens.tolist()]
    cu_q = [int(value) for value in cu_seqlens_q.tolist()]
    _validate_cu_seqlens(cu_q, total_q, f"{op_name} cu_seqlens_q")
    if any(length < 0 or length > max_prefix_len for length in prefix_values):
        raise ValueError(
            f"{op_name} prefix_lens entries must be in [0, max_blocks * block_size]"
        )
    for blocks, prefix_len in zip(block_rows, prefix_values, strict=True):
        needed_blocks = (prefix_len + block_size - 1) // block_size
        if any(block < 0 or block >= num_blocks for block in blocks[:needed_blocks]):
            raise ValueError(
                f"{op_name} block_tables entries must index KV cache blocks"
            )


def _reject_flash_attn_with_kvcache_unsupported(
    *,
    qv,
    rotary_cos,
    rotary_sin,
    cache_leftpad,
    cu_seqlens_q,
    cu_seqlens_k_new,
    max_seqlen_q,
    rotary_seqlens,
    q_descale,
    k_descale,
    v_descale,
    window_size,
    attention_chunk,
    softcap,
    scheduler_metadata,
    num_splits,
    pack_gqa,
    sm_margin,
    return_softmax_lse,
    sinks,
    score_mod,
    aux_tensors,
    ver,
    out,
) -> None:
    unsupported = {
        "qv": qv,
        "rotary_cos": rotary_cos,
        "rotary_sin": rotary_sin,
        "cache_leftpad": cache_leftpad,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k_new": cu_seqlens_k_new,
        "max_seqlen_q": max_seqlen_q,
        "rotary_seqlens": rotary_seqlens,
        "q_descale": q_descale,
        "k_descale": k_descale,
        "v_descale": v_descale,
        "scheduler_metadata": scheduler_metadata,
        "pack_gqa": pack_gqa,
        "sinks": sinks,
        "score_mod": score_mod,
        "aux_tensors": aux_tensors,
        "out": out,
    }
    for name, value in unsupported.items():
        if value is not None:
            raise NotImplementedError(
                f"flash_attn_with_kvcache Metal does not support {name}"
            )
    if window_size != (-1, -1):
        raise NotImplementedError(
            "flash_attn_with_kvcache Metal does not support window_size"
        )
    if attention_chunk not in (None, 0):
        raise NotImplementedError(
            "flash_attn_with_kvcache Metal does not support attention_chunk"
        )
    if softcap != 0.0:
        raise NotImplementedError(
            "flash_attn_with_kvcache Metal does not support softcap"
        )
    if num_splits not in (0, 1):
        raise NotImplementedError(
            "flash_attn_with_kvcache Metal does not support num_splits"
        )
    if sm_margin != 0:
        raise NotImplementedError(
            "flash_attn_with_kvcache Metal does not support sm_margin"
        )
    if return_softmax_lse:
        raise NotImplementedError(
            "flash_attn_with_kvcache Metal does not support return_softmax_lse"
        )
    if ver != 3:
        raise NotImplementedError("flash_attn_with_kvcache Metal supports only ver=3")


def _reject_flash_attn_varlen_unsupported(
    *,
    seqused_q,
    seqused_k,
    page_table,
    qv,
    q_descale,
    k_descale,
    v_descale,
    window_size,
    attention_chunk,
    softcap,
    num_splits,
    pack_gqa,
    sm_margin,
    return_softmax_lse,
    sinks,
    score_mod,
    aux_tensors,
    ver,
    out,
) -> None:
    unsupported = {
        "seqused_q": seqused_q,
        "seqused_k": seqused_k,
        "page_table": page_table,
        "qv": qv,
        "q_descale": q_descale,
        "k_descale": k_descale,
        "v_descale": v_descale,
        "pack_gqa": pack_gqa,
        "sinks": sinks,
        "score_mod": score_mod,
        "aux_tensors": aux_tensors,
        "out": out,
    }
    for name, value in unsupported.items():
        if value is not None:
            raise NotImplementedError(
                f"flash_attn_varlen_func Metal does not support {name}"
            )
    if window_size != (-1, -1):
        raise NotImplementedError(
            "flash_attn_varlen_func Metal does not support window_size"
        )
    if attention_chunk not in (None, 0):
        raise NotImplementedError(
            "flash_attn_varlen_func Metal does not support attention_chunk"
        )
    if softcap != 0.0:
        raise NotImplementedError(
            "flash_attn_varlen_func Metal does not support softcap"
        )
    if num_splits not in (0, 1):
        raise NotImplementedError(
            "flash_attn_varlen_func Metal does not support num_splits"
        )
    if sm_margin != 0:
        raise NotImplementedError(
            "flash_attn_varlen_func Metal does not support sm_margin"
        )
    if return_softmax_lse:
        raise NotImplementedError(
            "flash_attn_varlen_func Metal does not support return_softmax_lse"
        )
    if ver != 3:
        raise NotImplementedError("flash_attn_varlen_func Metal supports only ver=3")


def flash_attn_varlen_func(
    q: "mx.array",
    k: "mx.array",
    v: "mx.array",
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    seqused_q=None,
    seqused_k=None,
    page_table=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
    ver=3,
    out=None,
) -> "mx.array":
    _reject_flash_attn_varlen_unsupported(
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        page_table=page_table,
        qv=qv,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=window_size,
        attention_chunk=attention_chunk,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
        return_softmax_lse=return_softmax_lse,
        sinks=sinks,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        ver=ver,
        out=out,
    )
    _require_metal_extension()

    import mlx.core as mx

    for name, x in (("query", q), ("key", k), ("value", v)):
        _require_array(x, f"flash_attn_varlen_func {name}", mx)
        _require_float_dtype(x, "flash_attn_varlen_func", mx)
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("flash_attn_varlen_func query and K/V dtypes must match")
    if q.ndim != 3:
        raise ValueError("flash_attn_varlen_func query must have shape (total_q, H, D)")
    if k.ndim != 3 or v.ndim != 3:
        raise ValueError("flash_attn_varlen_func K/V must have shape (total_k, KVH, D)")
    if k.shape != v.shape:
        raise ValueError("flash_attn_varlen_func K/V shapes must match")
    if q.shape[2] != k.shape[2]:
        raise ValueError("flash_attn_varlen_func head dimensions must match")
    if k.shape[1] == 0 or q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            "flash_attn_varlen_func query heads must be divisible by KV heads"
        )
    if q.shape[2] == 0 or q.shape[2] > _MAX_DECODE_HEAD_DIM:
        raise ValueError(
            "flash_attn_varlen_func head dimension must be in the range [1, 256]"
        )

    cu_q = _require_int_vector(
        cu_seqlens_q, "flash_attn_varlen_func cu_seqlens_q", None, mx
    )
    cu_k = _require_int_vector(
        cu_seqlens_k, "flash_attn_varlen_func cu_seqlens_k", None, mx
    )
    if len(cu_q) != len(cu_k):
        raise ValueError(
            "flash_attn_varlen_func cu_seqlens_q and cu_seqlens_k must have the same length"
        )
    _validate_cu_seqlens(cu_q, q.shape[0], "flash_attn_varlen_func cu_seqlens_q")
    _validate_cu_seqlens(cu_k, k.shape[0], "flash_attn_varlen_func cu_seqlens_k")

    q_lens = [end - start for start, end in zip(cu_q, cu_q[1:])]
    k_lens = [end - start for start, end in zip(cu_k, cu_k[1:])]
    actual_max_q = max(q_lens, default=0)
    actual_max_k = max(k_lens, default=0)
    if max_seqlen_q is not None and actual_max_q > int(max_seqlen_q):
        raise ValueError(
            "flash_attn_varlen_func max_seqlen_q is smaller than cu_seqlens_q"
        )
    if max_seqlen_k is not None and actual_max_k > int(max_seqlen_k):
        raise ValueError(
            "flash_attn_varlen_func max_seqlen_k is smaller than cu_seqlens_k"
        )

    scale = _scale_value(
        softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5,
        "flash_attn_varlen_func",
    )
    q = mx.contiguous(q)
    k = mx.contiguous(k)
    v = mx.contiguous(v)
    cu_seqlens_q = mx.contiguous(cu_seqlens_q)
    cu_seqlens_k = mx.contiguous(cu_seqlens_k)
    out = mx.zeros(q.shape, dtype=q.dtype)
    mx.eval(q, k, v, cu_seqlens_q, cu_seqlens_k, out)
    _metal.flash_attn_varlen(
        out, q, k, v, cu_seqlens_q, cu_seqlens_k, scale, bool(causal)
    )
    return out


def flash_attn_with_kvcache(
    q: "mx.array",
    k_cache: "mx.array",
    v_cache: "mx.array",
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    cache_leftpad=None,
    page_table=None,
    cu_seqlens_q=None,
    cu_seqlens_k_new=None,
    max_seqlen_q=None,
    rotary_seqlens=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    attention_chunk=None,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
    ver=3,
    out=None,
) -> "mx.array":
    _reject_flash_attn_with_kvcache_unsupported(
        qv=qv,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_leftpad=cache_leftpad,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k_new,
        max_seqlen_q=max_seqlen_q,
        rotary_seqlens=rotary_seqlens,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=window_size,
        attention_chunk=attention_chunk,
        softcap=softcap,
        scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
        return_softmax_lse=return_softmax_lse,
        sinks=sinks,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        ver=ver,
        out=out,
    )

    import mlx.core as mx

    for name, x in (("query", q), ("key cache", k_cache), ("value cache", v_cache)):
        _require_array(x, f"flash_attn_with_kvcache {name}", mx)
        _require_float_dtype(x, "flash_attn_with_kvcache", mx)
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("flash_attn_with_kvcache query and K/V dtypes must match")
    if q.ndim != 4 or q.shape[1] != 1:
        raise ValueError("flash_attn_with_kvcache query must have shape (B, 1, H, D)")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "flash_attn_with_kvcache K/V caches must have shape (B_cache, S, KVH, D)"
        )
    if k_cache.shape != v_cache.shape:
        raise ValueError("flash_attn_with_kvcache K/V cache shapes must match")
    if q.shape[3] != k_cache.shape[3]:
        raise ValueError("flash_attn_with_kvcache head dimensions must match")
    if k_cache.shape[2] == 0 or q.shape[2] % k_cache.shape[2] != 0:
        raise ValueError(
            "flash_attn_with_kvcache query heads must be divisible by KV heads"
        )
    if q.shape[3] == 0 or q.shape[3] > _MAX_DECODE_HEAD_DIM:
        raise ValueError(
            "flash_attn_with_kvcache head dimension must be in the range [1, 256]"
        )
    if k_cache.shape[1] == 0:
        raise ValueError("flash_attn_with_kvcache K/V sequence length must be positive")
    if causal is not False and q.shape[1] != 1:
        raise NotImplementedError(
            "flash_attn_with_kvcache Metal supports causal masking only for decode seqlen 1"
        )
    if rotary_interleaved is not True:
        raise NotImplementedError(
            "flash_attn_with_kvcache Metal does not support rotary_interleaved=False"
        )
    if (k is None) != (v is None):
        raise ValueError("flash_attn_with_kvcache k and v must be provided together")

    batch_size = q.shape[0]
    paged = page_table is not None
    if paged:
        if cache_batch_idx is not None:
            raise NotImplementedError(
                "flash_attn_with_kvcache Metal does not support cache_batch_idx with page_table"
            )
        if k is not None:
            raise NotImplementedError(
                "flash_attn_with_kvcache Metal does not support K/V append with page_table"
            )
        _normalize_page_table(page_table, batch_size, mx)
        max_cache_len = page_table.shape[1] * k_cache.shape[1]
        batch_indices = list(range(batch_size))
    else:
        max_cache_len = k_cache.shape[1]
        batch_indices = _normalize_cache_batch_idx(
            cache_batch_idx, batch_size, k_cache.shape[0], mx
        )
    seq_lens = _normalize_cache_seqlens(cache_seqlens, batch_size, max_cache_len, mx)

    if k is not None:
        for name, x in (("key", k), ("value", v)):
            _require_array(x, f"flash_attn_with_kvcache {name}", mx)
            _require_float_dtype(x, "flash_attn_with_kvcache", mx)
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError(
                "flash_attn_with_kvcache new K/V dtypes must match query dtype"
            )
        if (
            k.shape != (batch_size, q.shape[1], k_cache.shape[2], q.shape[3])
            or v.shape != k.shape
        ):
            raise ValueError(
                "flash_attn_with_kvcache new K/V must have shape (B, 1, KVH, D)"
            )
        for batch, (cache_batch, seq_len) in enumerate(
            zip(batch_indices, seq_lens, strict=True)
        ):
            if seq_len >= k_cache.shape[1]:
                raise ValueError(
                    "flash_attn_with_kvcache cache is too short for K/V append"
                )
            k_cache[cache_batch : cache_batch + 1, seq_len : seq_len + 1, :, :] = k[
                batch : batch + 1
            ]
            v_cache[cache_batch : cache_batch + 1, seq_len : seq_len + 1, :, :] = v[
                batch : batch + 1
            ]
            seq_lens[batch] = seq_len + 1

    metal_q = mx.contiguous(q.transpose(0, 2, 1, 3))
    scale = _scale_value(
        softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5,
        "flash_attn_with_kvcache",
    )

    full_dense = (
        not paged
        and batch_size == k_cache.shape[0]
        and batch_indices == list(range(batch_size))
        and all(seq_len == k_cache.shape[1] for seq_len in seq_lens)
    )
    if full_dense:
        metal_k = mx.contiguous(k_cache.transpose(0, 2, 1, 3))
        metal_v = mx.contiguous(v_cache.transpose(0, 2, 1, 3))
        return decode_attention(metal_q, metal_k, metal_v, scale).transpose(0, 2, 1, 3)

    if paged:
        context_lens = mx.array(seq_lens, dtype=mx.int32)
        return decode_attention_paged(
            metal_q, k_cache, v_cache, page_table, context_lens, scale
        ).transpose(0, 2, 1, 3)

    k_list = []
    v_list = []
    for cache_batch, seq_len in zip(batch_indices, seq_lens, strict=True):
        if seq_len == 0:
            raise ValueError(
                "flash_attn_with_kvcache cache_seqlens entries must be positive for attention"
            )
        k_list.append(
            mx.contiguous(
                k_cache[cache_batch : cache_batch + 1, :seq_len, :, :].transpose(
                    0, 2, 1, 3
                )
            )
        )
        v_list.append(
            mx.contiguous(
                v_cache[cache_batch : cache_batch + 1, :seq_len, :, :].transpose(
                    0, 2, 1, 3
                )
            )
        )
    return decode_attention_ragged(metal_q, k_list, v_list, scale).transpose(0, 2, 1, 3)


def decode_attention(
    q: "mx.array", k: "mx.array", v: "mx.array", scale: float
) -> "mx.array":
    metal = _require_metal_extension()

    import mlx.core as mx

    scale = _scale_value(scale, "decode_attention")
    _validate_decode_tensor_family(
        q, k, v, "decode_attention", mx, require_dense_batch=True
    )

    q = mx.contiguous(q)
    k = mx.contiguous(k)
    v = mx.contiguous(v)
    out = mx.zeros(q.shape, dtype=q.dtype)
    mx.eval(q, k, v, out)
    metal.decode_attention(out, q, k, v, scale)
    return out


def decode_attention_ragged(
    q: "mx.array",
    k_list: Sequence["mx.array"],
    v_list: Sequence["mx.array"],
    scale: float,
) -> "mx.array":
    metal = _require_metal_extension()

    import mlx.core as mx

    _require_array(q, "decode_attention_ragged query", mx)
    _require_float_dtype(q, "decode_attention_ragged", mx)
    if q.ndim != 4 or q.shape[2] != 1:
        raise ValueError("decode_attention_ragged query must have shape (B, H, 1, D)")
    if q.shape[3] == 0 or q.shape[3] > _MAX_DECODE_HEAD_DIM:
        raise ValueError(
            "decode_attention_ragged head dimension must be in the range [1, 256]"
        )
    if not isinstance(k_list, Sequence) or not isinstance(v_list, Sequence):
        raise TypeError("decode_attention_ragged K/V caches must be sequences")
    if len(k_list) != q.shape[0] or len(v_list) != q.shape[0]:
        raise ValueError(
            "decode_attention_ragged K/V list lengths must match query batch"
        )

    scale = _scale_value(scale, "decode_attention_ragged")
    q = mx.contiguous(q)
    contiguous_k = []
    contiguous_v = []
    for i, (k, v) in enumerate(zip(k_list, v_list, strict=True)):
        _validate_decode_tensor_family(
            q,
            k,
            v,
            f"decode_attention_ragged entry {i}",
            mx,
            require_dense_batch=False,
        )
        contiguous_k.append(mx.contiguous(k))
        contiguous_v.append(mx.contiguous(v))

    out = mx.zeros(q.shape, dtype=q.dtype)
    mx.eval(q, out, *contiguous_k, *contiguous_v)
    metal.decode_attention_ragged(out, q, contiguous_k, contiguous_v, scale)
    return out


def decode_attention_paged(
    q: "mx.array",
    k_cache: "mx.array",
    v_cache: "mx.array",
    block_tables: "mx.array",
    context_lens: "mx.array",
    scale: float,
) -> "mx.array":
    metal = _require_metal_extension()

    import mlx.core as mx

    _require_array(q, "decode_attention_paged query", mx)
    _require_array(k_cache, "decode_attention_paged key cache", mx)
    _require_array(v_cache, "decode_attention_paged value cache", mx)
    _require_float_dtype(q, "decode_attention_paged", mx)
    _require_float_dtype(k_cache, "decode_attention_paged", mx)
    _require_float_dtype(v_cache, "decode_attention_paged", mx)
    if q.ndim != 4 or q.shape[2] != 1:
        raise ValueError("decode_attention_paged query must have shape (B, H, 1, D)")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "decode_attention_paged K/V caches must have shape (num_blocks, block_size, KVH, D)"
        )
    if k_cache.shape != v_cache.shape:
        raise ValueError("decode_attention_paged K/V cache shapes must match")
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("decode_attention_paged query and K/V cache dtypes must match")
    if q.shape[3] != k_cache.shape[3]:
        raise ValueError("decode_attention_paged head dimensions must match")
    if k_cache.shape[2] == 0 or q.shape[1] % k_cache.shape[2] != 0:
        raise ValueError(
            "decode_attention_paged query heads must be divisible by KV heads"
        )
    if q.shape[3] == 0 or q.shape[3] > _MAX_DECODE_HEAD_DIM:
        raise ValueError(
            "decode_attention_paged head dimension must be in the range [1, 256]"
        )
    if k_cache.shape[1] == 0:
        raise ValueError("decode_attention_paged block size must be positive")
    if q.shape[0] > 0 and k_cache.shape[0] == 0:
        raise ValueError(
            "decode_attention_paged KV cache must contain at least one block"
        )

    scale = _scale_value(scale, "decode_attention_paged")
    _validate_paged_metadata(
        block_tables,
        context_lens,
        q.shape[0],
        k_cache.shape[0],
        k_cache.shape[1],
        "decode_attention_paged",
        mx,
    )
    q = mx.contiguous(q)
    k_cache = mx.contiguous(k_cache)
    v_cache = mx.contiguous(v_cache)
    block_tables = mx.contiguous(block_tables)
    context_lens = mx.contiguous(context_lens)
    out = mx.zeros(q.shape, dtype=q.dtype)
    mx.eval(q, k_cache, v_cache, block_tables, context_lens, out)
    metal.decode_attention_paged(
        out, q, k_cache, v_cache, block_tables, context_lens, scale
    )
    return out


def prefill_attention_paged(
    q: "mx.array",
    k: "mx.array",
    v: "mx.array",
    k_cache: "mx.array",
    v_cache: "mx.array",
    block_tables: "mx.array",
    prefix_lens: "mx.array",
    cu_seqlens_q: "mx.array",
    scale: float,
    causal: bool = True,
) -> "mx.array":
    metal = _require_metal_extension()

    import mlx.core as mx

    for name, x in (
        ("query", q),
        ("key", k),
        ("value", v),
        ("key cache", k_cache),
        ("value cache", v_cache),
        ("block_tables", block_tables),
        ("prefix_lens", prefix_lens),
        ("cu_seqlens_q", cu_seqlens_q),
    ):
        _require_array(x, f"prefill_attention_paged {name}", mx)
    for x in (q, k, v, k_cache, v_cache):
        _require_float_dtype(x, "prefill_attention_paged", mx)
    if q.ndim != 3:
        raise ValueError(
            "prefill_attention_paged query must have shape (total_q, H, D)"
        )
    if k.ndim != 3 or v.ndim != 3:
        raise ValueError(
            "prefill_attention_paged K/V must have shape (total_q, KVH, D)"
        )
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "prefill_attention_paged K/V caches must have shape (num_blocks, block_size, KVH, D)"
        )
    if k.shape != v.shape:
        raise ValueError("prefill_attention_paged K/V shapes must match")
    if k_cache.shape != v_cache.shape:
        raise ValueError("prefill_attention_paged K/V cache shapes must match")
    if k.shape[0] != q.shape[0]:
        raise ValueError(
            "prefill_attention_paged K/V token count must match query token count"
        )
    if (
        q.dtype != k.dtype
        or q.dtype != v.dtype
        or q.dtype != k_cache.dtype
        or q.dtype != v_cache.dtype
    ):
        raise ValueError(
            "prefill_attention_paged query, K/V, and cache dtypes must match"
        )
    if q.shape[2] != k.shape[2] or q.shape[2] != k_cache.shape[3]:
        raise ValueError("prefill_attention_paged head dimensions must match")
    if k.shape[1] != k_cache.shape[2]:
        raise ValueError("prefill_attention_paged KV head counts must match")
    if k.shape[1] == 0 or q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            "prefill_attention_paged query heads must be divisible by KV heads"
        )
    if q.shape[2] == 0 or q.shape[2] > _MAX_DECODE_HEAD_DIM:
        raise ValueError(
            "prefill_attention_paged head dimension must be in the range [1, 256]"
        )
    if k_cache.shape[1] == 0:
        raise ValueError("prefill_attention_paged block size must be positive")
    if q.shape[0] > 0 and k_cache.shape[0] == 0:
        raise ValueError(
            "prefill_attention_paged KV cache must contain at least one block"
        )
    if block_tables.ndim != 2:
        raise ValueError(
            "prefill_attention_paged block_tables must have shape (B, max_blocks)"
        )

    scale = _scale_value(scale, "prefill_attention_paged")
    _validate_prefill_paged_metadata(
        block_tables,
        prefix_lens,
        cu_seqlens_q,
        block_tables.shape[0],
        q.shape[0],
        k_cache.shape[0],
        k_cache.shape[1],
        "prefill_attention_paged",
        mx,
    )
    q = mx.contiguous(q)
    k = mx.contiguous(k)
    v = mx.contiguous(v)
    k_cache = mx.contiguous(k_cache)
    v_cache = mx.contiguous(v_cache)
    block_tables = mx.contiguous(block_tables)
    prefix_lens = mx.contiguous(prefix_lens)
    cu_seqlens_q = mx.contiguous(cu_seqlens_q)
    out = mx.zeros(q.shape, dtype=q.dtype)
    mx.eval(q, k, v, k_cache, v_cache, block_tables, prefix_lens, cu_seqlens_q, out)
    metal.prefill_attention_paged(
        out,
        q,
        k,
        v,
        k_cache,
        v_cache,
        block_tables,
        prefix_lens,
        cu_seqlens_q,
        scale,
        bool(causal),
    )
    return out


def paged_kv_scatter(
    k: "mx.array",
    v: "mx.array",
    k_cache: "mx.array",
    v_cache: "mx.array",
    slot_mapping: "mx.array",
) -> None:
    metal = _require_metal_extension()

    import mlx.core as mx

    _require_array(k, "paged_kv_scatter key tensor", mx)
    _require_array(v, "paged_kv_scatter value tensor", mx)
    _require_array(k_cache, "paged_kv_scatter key cache", mx)
    _require_array(v_cache, "paged_kv_scatter value cache", mx)
    _require_array(slot_mapping, "paged_kv_scatter slot_mapping", mx)
    _require_float_dtype(k, "paged_kv_scatter", mx)
    _require_float_dtype(v, "paged_kv_scatter", mx)
    _require_float_dtype(k_cache, "paged_kv_scatter", mx)
    _require_float_dtype(v_cache, "paged_kv_scatter", mx)
    if k.ndim != 3 or v.ndim != 3:
        raise ValueError(
            "paged_kv_scatter K/V tensors must have shape (num_tokens, KVH, D)"
        )
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "paged_kv_scatter K/V caches must have shape (num_blocks, block_size, KVH, D)"
        )
    if k.shape != v.shape:
        raise ValueError("paged_kv_scatter K/V tensor shapes must match")
    if k_cache.shape != v_cache.shape:
        raise ValueError("paged_kv_scatter K/V cache shapes must match")
    if slot_mapping.ndim != 1 or slot_mapping.shape[0] != k.shape[0]:
        raise ValueError("paged_kv_scatter slot_mapping must have shape (num_tokens,)")
    if slot_mapping.dtype != mx.int32:
        raise TypeError("paged_kv_scatter slot_mapping must be an int32 array")
    if k.dtype != v.dtype or k.dtype != k_cache.dtype or k.dtype != v_cache.dtype:
        raise ValueError(
            "paged_kv_scatter K/V tensors and caches must have matching dtypes"
        )
    if k.shape[1] != k_cache.shape[2]:
        raise ValueError("paged_kv_scatter KV head counts must match")
    if k.shape[2] != k_cache.shape[3]:
        raise ValueError("paged_kv_scatter head dimensions must match")
    if k.shape[1] == 0:
        raise ValueError("paged_kv_scatter KV head count must be positive")
    if k.shape[2] == 0 or k.shape[2] > _MAX_DECODE_HEAD_DIM:
        raise ValueError(
            "paged_kv_scatter head dimension must be in the range [1, 256]"
        )
    if k_cache.shape[1] == 0:
        raise ValueError("paged_kv_scatter block size must be positive")
    if k.shape[0] > 0 and k_cache.shape[0] == 0:
        raise ValueError("paged_kv_scatter KV cache must contain at least one block")

    cache_slot_count = k_cache.shape[0] * k_cache.shape[1]
    mx.eval(slot_mapping)
    slot_values = slot_mapping.tolist()
    if any(slot < 0 or slot >= cache_slot_count for slot in slot_values):
        raise ValueError(
            "paged_kv_scatter slot_mapping entries must be in cache slot range"
        )

    k = mx.contiguous(k)
    v = mx.contiguous(v)
    slot_mapping = mx.contiguous(slot_mapping)
    mx.eval(k, v, k_cache, v_cache, slot_mapping)
    metal.paged_kv_scatter(k, v, k_cache, v_cache, slot_mapping)


__all__ = [
    "decode_attention",
    "decode_attention_paged",
    "decode_attention_ragged",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "paged_kv_scatter",
    "prefill_attention_paged",
]
