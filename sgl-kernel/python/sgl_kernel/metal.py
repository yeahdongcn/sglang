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


def _require_float_or_bfloat_dtype(x: "mx.array", op_name: str, mx) -> None:
    if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise TypeError(
            f"{op_name} supports only float16, bfloat16, and float32 arrays"
        )


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


def _paged_cache_shape(cache: "mx.array", op_name: str) -> tuple[int, int, int, int]:
    if cache.ndim == 4:
        return cache.shape[0], cache.shape[1], cache.shape[2], cache.shape[3]
    if cache.ndim == 3:
        return cache.shape[0], 1, cache.shape[1], cache.shape[2]
    raise ValueError(
        f"{op_name} K/V caches must have shape (num_blocks, block_size, KVH, D) "
        "or p1 shape (num_blocks, KVH, D)"
    )


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
    mx.synchronize()
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
    mx.synchronize()
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
    mx.synchronize()
    return out


def decode_attention_paged(
    q: "mx.array",
    k_cache: "mx.array",
    v_cache: "mx.array",
    block_tables: "mx.array",
    context_lens: "mx.array",
    scale: float,
) -> "mx.array":
    import mlx.core as mx

    _require_array(q, "decode_attention_paged query", mx)
    _require_array(k_cache, "decode_attention_paged key cache", mx)
    _require_array(v_cache, "decode_attention_paged value cache", mx)
    _require_float_dtype(q, "decode_attention_paged", mx)
    _require_float_dtype(k_cache, "decode_attention_paged", mx)
    _require_float_dtype(v_cache, "decode_attention_paged", mx)
    if q.ndim != 4 or q.shape[2] != 1:
        raise ValueError("decode_attention_paged query must have shape (B, H, 1, D)")
    num_blocks, block_size, num_kv_heads, head_dim = _paged_cache_shape(
        k_cache, "decode_attention_paged"
    )
    _paged_cache_shape(v_cache, "decode_attention_paged")
    if k_cache.shape != v_cache.shape:
        raise ValueError("decode_attention_paged K/V cache shapes must match")
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("decode_attention_paged query and K/V cache dtypes must match")
    if q.shape[3] != head_dim:
        raise ValueError("decode_attention_paged head dimensions must match")
    if num_kv_heads == 0 or q.shape[1] % num_kv_heads != 0:
        raise ValueError(
            "decode_attention_paged query heads must be divisible by KV heads"
        )
    if q.shape[3] == 0 or q.shape[3] > _MAX_DECODE_HEAD_DIM:
        raise ValueError(
            "decode_attention_paged head dimension must be in the range [1, 256]"
        )
    if block_size == 0:
        raise ValueError("decode_attention_paged block size must be positive")
    if q.shape[0] > 0 and num_blocks == 0:
        raise ValueError(
            "decode_attention_paged KV cache must contain at least one block"
        )

    scale = _scale_value(scale, "decode_attention_paged")
    _validate_paged_metadata(
        block_tables,
        context_lens,
        q.shape[0],
        num_blocks,
        block_size,
        "decode_attention_paged",
        mx,
    )
    return decode_attention_paged_unchecked(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale,
    )


def decode_attention_paged_unchecked(
    q: "mx.array",
    k_cache: "mx.array",
    v_cache: "mx.array",
    block_tables: "mx.array",
    context_lens: "mx.array",
    scale: float,
) -> "mx.array":
    metal = _require_metal_extension()

    import mlx.core as mx

    q = mx.contiguous(q)
    block_tables = mx.contiguous(block_tables)
    context_lens = mx.contiguous(context_lens)
    out = mx.zeros(q.shape, dtype=q.dtype)
    mx.eval(q, block_tables, context_lens, out)
    metal.decode_attention_paged(
        out, q, k_cache, v_cache, block_tables, context_lens, float(scale)
    )
    return out


def decode_attention_paged_with_kv(
    q: "mx.array",
    k: "mx.array",
    v: "mx.array",
    k_cache: "mx.array",
    v_cache: "mx.array",
    block_tables: "mx.array",
    context_lens: "mx.array",
    slot_mapping: "mx.array",
    scale: float,
) -> "mx.array":
    import mlx.core as mx

    for name, x in (
        ("query", q),
        ("key tensor", k),
        ("value tensor", v),
        ("key cache", k_cache),
        ("value cache", v_cache),
        ("block_tables", block_tables),
        ("context_lens", context_lens),
        ("slot_mapping", slot_mapping),
    ):
        _require_array(x, f"decode_attention_paged_with_kv {name}", mx)
    if q.dtype != mx.float16 or k.dtype != mx.float16 or v.dtype != mx.float16:
        raise ValueError("decode_attention_paged_with_kv supports only float16 tensors")
    if k_cache.dtype != mx.float16 or v_cache.dtype != mx.float16:
        raise ValueError("decode_attention_paged_with_kv supports only float16 caches")
    if q.ndim != 4 or q.shape[2] != 1 or q.shape[3] != 128:
        raise ValueError(
            "decode_attention_paged_with_kv query must have shape (B, H, 1, 128)"
        )
    if k.ndim != 3 or v.ndim != 3 or k.shape != v.shape:
        raise ValueError(
            "decode_attention_paged_with_kv K/V tensors must have shape (B, KVH, 128)"
        )
    if k.shape[0] != q.shape[0] or k.shape[2] != q.shape[3]:
        raise ValueError("decode_attention_paged_with_kv K/V tensors must match query")
    if k_cache.ndim != 4 or v_cache.ndim != 4 or k_cache.shape != v_cache.shape:
        raise ValueError(
            "decode_attention_paged_with_kv K/V caches must have matching 4-D shapes"
        )
    if (
        k_cache.shape[1] != 16
        or k_cache.shape[2] != k.shape[1]
        or k_cache.shape[3] != 128
    ):
        raise ValueError(
            "decode_attention_paged_with_kv requires block_size=16 and head_dim=128"
        )
    if q.shape[1] % k_cache.shape[2] != 0:
        raise ValueError(
            "decode_attention_paged_with_kv query heads must be divisible by KV heads"
        )
    if slot_mapping.ndim != 1 or slot_mapping.shape[0] != q.shape[0]:
        raise ValueError(
            "decode_attention_paged_with_kv slot_mapping must have shape (B,)"
        )
    if slot_mapping.dtype != mx.int32:
        raise TypeError("decode_attention_paged_with_kv slot_mapping must be int32")

    scale = _scale_value(scale, "decode_attention_paged_with_kv")
    _validate_paged_metadata(
        block_tables,
        context_lens,
        q.shape[0],
        k_cache.shape[0],
        k_cache.shape[1],
        "decode_attention_paged_with_kv",
        mx,
    )
    if slot_mapping.size:
        mx.eval(slot_mapping)
        if (
            int(mx.min(slot_mapping).item()) < 0
            or int(mx.max(slot_mapping).item()) >= k_cache.shape[0] * k_cache.shape[1]
        ):
            raise ValueError(
                "decode_attention_paged_with_kv slot_mapping entries out of range"
            )
    return decode_attention_paged_with_kv_unchecked(
        q,
        k,
        v,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        slot_mapping,
        scale,
    )


def decode_attention_paged_with_kv_unchecked(
    q: "mx.array",
    k: "mx.array",
    v: "mx.array",
    k_cache: "mx.array",
    v_cache: "mx.array",
    block_tables: "mx.array",
    context_lens: "mx.array",
    slot_mapping: "mx.array",
    scale: float,
) -> "mx.array":
    metal = _require_metal_extension()

    import mlx.core as mx

    q = mx.contiguous(q)
    k = mx.contiguous(k)
    v = mx.contiguous(v)
    block_tables = mx.contiguous(block_tables)
    context_lens = mx.contiguous(context_lens)
    slot_mapping = mx.contiguous(slot_mapping)
    out = mx.zeros(q.shape, dtype=q.dtype)
    mx.eval(q, k, v, block_tables, context_lens, slot_mapping, out)
    metal.decode_attention_paged_with_kv(
        out,
        q,
        k,
        v,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        slot_mapping,
        float(scale),
    )
    return out


_DECODE_PAGED_H128_B16_HEADER = """
inline float sgl_fast_exp(float x) {
    return fast::exp2(x * 1.4426950408889634f);
}
"""

_DECODE_DENSE_H128_SOURCE = """
constexpr int kHeadDim = 128;
constexpr int kBlockSize = 16;
constexpr int kNumThreads = SGL_DENSE_DECODE_THREADS;
constexpr int kNumSimdLanes = 32;
constexpr int kNumWarps = kNumThreads / kNumSimdLanes;
constexpr int kVElemsPerLane = kHeadDim / kNumSimdLanes;

const uint local = thread_position_in_threadgroup.x;
const uint group = thread_position_in_grid.x / kNumThreads;
const uint lane = thread_index_in_simdgroup;
const uint warp_idx = local / kNumSimdLanes;
const int head = int(group);
const int batch_idx = int(thread_position_in_grid.y);
if (local >= kNumThreads || batch_idx >= batch || head >= num_heads) {
    return;
}
if (seq_len <= 0) {
    return;
}

const int num_context_blocks = (seq_len + kBlockSize - 1) / kBlockSize;
const int kv_head = head / (num_heads / num_kv_heads);
const int q_base = (batch_idx * num_heads + head) * kHeadDim;
const int out_base = (batch_idx * num_heads + head) * kHeadDim;
const int kv_base = (batch_idx * num_kv_heads + kv_head) * seq_len * kHeadDim;
const int token_lane = int(lane >> 1);
const int dim_lane = int(lane & 1);

threadgroup float scratch[kNumWarps * kHeadDim + 2 * kNumWarps];
float running_max = -INFINITY;
float running_sum = 0.0f;
float v_acc[kVElemsPerLane];
for (int i = 0; i < kVElemsPerLane; ++i) {
    v_acc[i] = 0.0f;
}

half4 q_vecs[kHeadDim / 8];
for (int i = 0; i < kHeadDim / 8; ++i) {
    const int dim = dim_lane * 4 + i * 8;
    q_vecs[i] = *reinterpret_cast<const device half4*>(q + q_base + dim);
}

for (
    int block_idx = int(warp_idx);
    block_idx < num_context_blocks;
    block_idx += kNumWarps
) {
    const int token_idx = block_idx * kBlockSize + token_lane;
    float partial_score = 0.0f;
    if (token_idx < seq_len) {
        const int k_base = kv_base + token_idx * kHeadDim;
        for (int i = 0; i < kHeadDim / 8; ++i) {
            const int dim = dim_lane * 4 + i * 8;
            const half4 k_vec =
                *reinterpret_cast<const device half4*>(k + k_base + dim);
            partial_score += dot(
                static_cast<float4>(q_vecs[i]), static_cast<float4>(k_vec)
            );
        }
    }
    const float paired_score = partial_score + simd_shuffle_xor(partial_score, 1);
    SGL_DENSE_DECODE_SCORE_SETUP

    float new_max = max(running_max, block_max);
    if (new_max == -INFINITY) {
        new_max = 0.0f;
    }
    float old_correction =
        running_max == -INFINITY ? 0.0f : sgl_fast_exp(running_max - new_max);
    for (int i = 0; i < kVElemsPerLane; ++i) {
        v_acc[i] *= old_correction;
    }
    running_sum *= old_correction;
    running_max = new_max;

    for (int token = 0; token < block_valid_tokens; ++token) {
        SGL_DENSE_DECODE_WEIGHT
        running_sum += weight;
        const int v_base = kv_base + (block_idx * kBlockSize + token) * kHeadDim;
        for (int i = 0; i < kVElemsPerLane; ++i) {
            const int dim = int(lane) + i * kNumSimdLanes;
            v_acc[i] += weight * static_cast<float>(v[v_base + dim]);
        }
    }
    SGL_DENSE_DECODE_BLOCK_END
}

threadgroup_barrier(mem_flags::mem_threadgroup);
threadgroup float* merge_max = scratch;
threadgroup float* merge_sum = merge_max + kNumWarps;
threadgroup float* merge_out = merge_sum + kNumWarps;
if (lane == 0) {
    merge_max[warp_idx] = running_max;
    merge_sum[warp_idx] = running_sum;
}
threadgroup float* this_out = merge_out + warp_idx * kHeadDim;
for (int i = 0; i < kVElemsPerLane; ++i) {
    this_out[int(lane) + i * kNumSimdLanes] = v_acc[i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (warp_idx == 0) {
    for (int warp = 1; warp < kNumWarps; ++warp) {
        const float other_max = merge_max[warp];
        const float other_sum = merge_sum[warp];
        if (other_max == -INFINITY && other_sum == 0.0f) {
            continue;
        }
        float new_max = max(running_max, other_max);
        if (new_max == -INFINITY) {
            new_max = 0.0f;
        }
        const float this_correction =
            running_max == -INFINITY ? 0.0f : sgl_fast_exp(running_max - new_max);
        const float other_correction =
            other_max == -INFINITY ? 0.0f : sgl_fast_exp(other_max - new_max);
        const threadgroup float* other_out = merge_out + warp * kHeadDim;
        for (int i = 0; i < kVElemsPerLane; ++i) {
            const int dim = int(lane) + i * kNumSimdLanes;
            v_acc[i] =
                v_acc[i] * this_correction + other_out[dim] * other_correction;
        }
        running_sum = running_sum * this_correction + other_sum * other_correction;
        running_max = new_max;
    }
    const float inv_sum = 1.0f / (running_sum + 1e-6f);
    for (int i = 0; i < kVElemsPerLane; ++i) {
        const int dim = int(lane) + i * kNumSimdLanes;
        out[out_base + dim] = static_cast<half>(v_acc[i] * inv_sum);
    }
}
"""

_DECODE_DENSE_REGSCORE_SETUP = """
const float lane_score = token_idx < seq_len ? paired_score * scale : -INFINITY;
const int block_valid_tokens = min(kBlockSize, seq_len - block_idx * kBlockSize);
float block_max = lane_score;
for (int mask = kNumSimdLanes / 2; mask >= 1; mask >>= 1) {
    block_max = max(block_max, simd_shuffle_xor(block_max, mask));
}
"""

_DECODE_DENSE_REGSCORE_WEIGHT = """
const float token_score = simd_shuffle(lane_score, static_cast<ushort>(token * 2));
const float weight = sgl_fast_exp(token_score - running_max);
"""

_DECODE_DENSE_SHAREDSCORE_SETUP = """
threadgroup float* warp_scores = scratch + warp_idx * kBlockSize;
if (dim_lane == 0) {
    warp_scores[token_lane] = token_idx < seq_len ? paired_score * scale : -INFINITY;
}
simdgroup_barrier(mem_flags::mem_threadgroup);
const int block_valid_tokens = min(kBlockSize, seq_len - block_idx * kBlockSize);
float block_max = -INFINITY;
for (int token = int(lane); token < block_valid_tokens; token += kNumSimdLanes) {
    block_max = max(block_max, warp_scores[token]);
}
for (int mask = kNumSimdLanes / 2; mask >= 1; mask >>= 1) {
    block_max = max(block_max, simd_shuffle_xor(block_max, mask));
}
"""

_DECODE_DENSE_SHAREDSCORE_WEIGHT = """
const float weight = sgl_fast_exp(warp_scores[token] - running_max);
"""


def _dense_decode_source(*, mode: str, threads: int) -> str:
    if mode == "regscore":
        score_setup = _DECODE_DENSE_REGSCORE_SETUP
        weight = _DECODE_DENSE_REGSCORE_WEIGHT
        block_end = ""
    elif mode == "sharedscore":
        score_setup = _DECODE_DENSE_SHAREDSCORE_SETUP
        weight = _DECODE_DENSE_SHAREDSCORE_WEIGHT
        block_end = "simdgroup_barrier(mem_flags::mem_threadgroup);"
    else:
        raise ValueError(f"unknown dense decode mode: {mode}")
    return (
        _DECODE_DENSE_H128_SOURCE.replace("SGL_DENSE_DECODE_THREADS", str(threads))
        .replace("SGL_DENSE_DECODE_SCORE_SETUP", score_setup)
        .replace("SGL_DENSE_DECODE_WEIGHT", weight)
        .replace("SGL_DENSE_DECODE_BLOCK_END", block_end)
    )


_DECODE_DENSE_GQA2_H128_SOURCE = """
constexpr int kHeadDim = 128;
constexpr int kBlockSize = 16;
constexpr int kNumThreads = SGL_DENSE_GQA2_DECODE_THREADS;
constexpr int kNumSimdLanes = 32;
constexpr int kNumWarps = kNumThreads / kNumSimdLanes;
constexpr int kVElemsPerLane = kHeadDim / kNumSimdLanes;

const uint local = thread_position_in_threadgroup.x;
const uint group = thread_position_in_grid.x / kNumThreads;
const uint lane = thread_index_in_simdgroup;
const uint warp_idx = local / kNumSimdLanes;
const int kv_head = int(group);
const int batch_idx = int(thread_position_in_grid.y);
if (
    local >= kNumThreads || batch_idx >= batch || kv_head >= num_kv_heads
    || seq_len <= 0
) {
    return;
}

const int head0 = kv_head * 2;
const int head1 = head0 + 1;
const int num_context_blocks = (seq_len + kBlockSize - 1) / kBlockSize;
const int q_base0 = (batch_idx * num_heads + head0) * kHeadDim;
const int q_base1 = (batch_idx * num_heads + head1) * kHeadDim;
const int kv_base = (batch_idx * num_kv_heads + kv_head) * seq_len * kHeadDim;
const int token_lane = int(lane >> 1);
const int dim_lane = int(lane & 1);

threadgroup float scratch[kNumWarps * kHeadDim * 2 + 4 * kNumWarps];
float running_max0 = -INFINITY;
float running_max1 = -INFINITY;
float running_sum0 = 0.0f;
float running_sum1 = 0.0f;
float v_acc0[kVElemsPerLane];
float v_acc1[kVElemsPerLane];
for (int i = 0; i < kVElemsPerLane; ++i) {
    v_acc0[i] = 0.0f;
    v_acc1[i] = 0.0f;
}

half4 q0[kHeadDim / 8];
half4 q1[kHeadDim / 8];
for (int i = 0; i < kHeadDim / 8; ++i) {
    const int dim = dim_lane * 4 + i * 8;
    q0[i] = *reinterpret_cast<const device half4*>(q + q_base0 + dim);
    q1[i] = *reinterpret_cast<const device half4*>(q + q_base1 + dim);
}

for (
    int block_idx = int(warp_idx);
    block_idx < num_context_blocks;
    block_idx += kNumWarps
) {
    const int token_idx = block_idx * kBlockSize + token_lane;
    float partial0 = 0.0f;
    float partial1 = 0.0f;
    if (token_idx < seq_len) {
        const int k_base = kv_base + token_idx * kHeadDim;
        for (int i = 0; i < kHeadDim / 8; ++i) {
            const int dim = dim_lane * 4 + i * 8;
            const half4 k_vec =
                *reinterpret_cast<const device half4*>(k + k_base + dim);
            const float4 k_float = static_cast<float4>(k_vec);
            partial0 += dot(static_cast<float4>(q0[i]), k_float);
            partial1 += dot(static_cast<float4>(q1[i]), k_float);
        }
    }
    const float score0 =
        token_idx < seq_len
        ? (partial0 + simd_shuffle_xor(partial0, 1)) * scale
        : -INFINITY;
    const float score1 =
        token_idx < seq_len
        ? (partial1 + simd_shuffle_xor(partial1, 1)) * scale
        : -INFINITY;

    const int block_valid_tokens =
        min(kBlockSize, seq_len - block_idx * kBlockSize);
    float block_max0 = score0;
    float block_max1 = score1;
    for (int mask = kNumSimdLanes / 2; mask >= 1; mask >>= 1) {
        block_max0 = max(block_max0, simd_shuffle_xor(block_max0, mask));
        block_max1 = max(block_max1, simd_shuffle_xor(block_max1, mask));
    }

    float new_max0 = max(running_max0, block_max0);
    float new_max1 = max(running_max1, block_max1);
    if (new_max0 == -INFINITY) {
        new_max0 = 0.0f;
    }
    if (new_max1 == -INFINITY) {
        new_max1 = 0.0f;
    }
    const float correction0 =
        running_max0 == -INFINITY ? 0.0f : sgl_fast_exp(running_max0 - new_max0);
    const float correction1 =
        running_max1 == -INFINITY ? 0.0f : sgl_fast_exp(running_max1 - new_max1);
    for (int i = 0; i < kVElemsPerLane; ++i) {
        v_acc0[i] *= correction0;
        v_acc1[i] *= correction1;
    }
    running_sum0 *= correction0;
    running_sum1 *= correction1;
    running_max0 = new_max0;
    running_max1 = new_max1;

    for (int token = 0; token < block_valid_tokens; ++token) {
        const float token_score0 =
            simd_shuffle(score0, static_cast<ushort>(token * 2));
        const float token_score1 =
            simd_shuffle(score1, static_cast<ushort>(token * 2));
        const float weight0 = sgl_fast_exp(token_score0 - running_max0);
        const float weight1 = sgl_fast_exp(token_score1 - running_max1);
        running_sum0 += weight0;
        running_sum1 += weight1;
        const int v_base = kv_base + (block_idx * kBlockSize + token) * kHeadDim;
        for (int i = 0; i < kVElemsPerLane; ++i) {
            const int dim = int(lane) + i * kNumSimdLanes;
            const float value = static_cast<float>(v[v_base + dim]);
            v_acc0[i] += weight0 * value;
            v_acc1[i] += weight1 * value;
        }
    }
}

threadgroup_barrier(mem_flags::mem_threadgroup);
threadgroup float* merge_max0 = scratch;
threadgroup float* merge_sum0 = merge_max0 + kNumWarps;
threadgroup float* merge_max1 = merge_sum0 + kNumWarps;
threadgroup float* merge_sum1 = merge_max1 + kNumWarps;
threadgroup float* merge_out0 = merge_sum1 + kNumWarps;
threadgroup float* merge_out1 = merge_out0 + kNumWarps * kHeadDim;
if (lane == 0) {
    merge_max0[warp_idx] = running_max0;
    merge_sum0[warp_idx] = running_sum0;
    merge_max1[warp_idx] = running_max1;
    merge_sum1[warp_idx] = running_sum1;
}
threadgroup float* this_out0 = merge_out0 + warp_idx * kHeadDim;
threadgroup float* this_out1 = merge_out1 + warp_idx * kHeadDim;
for (int i = 0; i < kVElemsPerLane; ++i) {
    const int dim = int(lane) + i * kNumSimdLanes;
    this_out0[dim] = v_acc0[i];
    this_out1[dim] = v_acc1[i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (warp_idx == 0) {
    for (int warp = 1; warp < kNumWarps; ++warp) {
        const float other_max0 = merge_max0[warp];
        const float other_sum0 = merge_sum0[warp];
        const float other_max1 = merge_max1[warp];
        const float other_sum1 = merge_sum1[warp];
        float new_max0 = max(running_max0, other_max0);
        float new_max1 = max(running_max1, other_max1);
        if (new_max0 == -INFINITY) {
            new_max0 = 0.0f;
        }
        if (new_max1 == -INFINITY) {
            new_max1 = 0.0f;
        }
        const float this_correction0 =
            running_max0 == -INFINITY ? 0.0f : sgl_fast_exp(running_max0 - new_max0);
        const float other_correction0 =
            other_max0 == -INFINITY ? 0.0f : sgl_fast_exp(other_max0 - new_max0);
        const float this_correction1 =
            running_max1 == -INFINITY ? 0.0f : sgl_fast_exp(running_max1 - new_max1);
        const float other_correction1 =
            other_max1 == -INFINITY ? 0.0f : sgl_fast_exp(other_max1 - new_max1);
        const threadgroup float* other_out0 = merge_out0 + warp * kHeadDim;
        const threadgroup float* other_out1 = merge_out1 + warp * kHeadDim;
        for (int i = 0; i < kVElemsPerLane; ++i) {
            const int dim = int(lane) + i * kNumSimdLanes;
            v_acc0[i] =
                v_acc0[i] * this_correction0
                + other_out0[dim] * other_correction0;
            v_acc1[i] =
                v_acc1[i] * this_correction1
                + other_out1[dim] * other_correction1;
        }
        running_sum0 =
            running_sum0 * this_correction0 + other_sum0 * other_correction0;
        running_sum1 =
            running_sum1 * this_correction1 + other_sum1 * other_correction1;
        running_max0 = new_max0;
        running_max1 = new_max1;
    }
    const float inv_sum0 = 1.0f / (running_sum0 + 1e-6f);
    const float inv_sum1 = 1.0f / (running_sum1 + 1e-6f);
    for (int i = 0; i < kVElemsPerLane; ++i) {
        const int dim = int(lane) + i * kNumSimdLanes;
        out[q_base0 + dim] = static_cast<half>(v_acc0[i] * inv_sum0);
        out[q_base1 + dim] = static_cast<half>(v_acc1[i] * inv_sum1);
    }
}
"""


def _dense_gqa2_decode_source(*, threads: int) -> str:
    if threads not in (128, 256):
        raise ValueError("dense GQA2 decode threads must be 128 or 256")
    return _DECODE_DENSE_GQA2_H128_SOURCE.replace(
        "SGL_DENSE_GQA2_DECODE_THREADS", str(threads)
    )


_DECODE_PAGED_H128_B16_SOURCE = """
constexpr int kHeadDim = 128;
constexpr int kBlockSize = 16;
constexpr int kNumThreads = 256;
constexpr int kNumSimdLanes = 32;
constexpr int kNumWarps = kNumThreads / kNumSimdLanes;
constexpr int kVElemsPerLane = kHeadDim / kNumSimdLanes;

const uint local = thread_position_in_threadgroup.x;
const uint group = thread_position_in_grid.x / kNumThreads;
const uint lane = thread_index_in_simdgroup;
const uint warp_idx = local / kNumSimdLanes;
const int head = int(group);
const int batch_idx = int(thread_position_in_grid.y);
if (local >= kNumThreads || batch_idx >= batch || head >= num_heads) {
    return;
}

const int seq_len = min(context_lens[batch_idx], max_blocks * kBlockSize);
if (seq_len <= 0) {
    return;
}
const int num_context_blocks = (seq_len + kBlockSize - 1) / kBlockSize;
const int kv_head = head / (num_heads / num_kv_heads);
const int q_base = (batch_idx * num_heads + head) * kHeadDim;
const int out_base = (batch_idx * num_heads + head) * kHeadDim;
const int cache_block_stride = kBlockSize * num_kv_heads * kHeadDim;
const int cache_offset_stride = num_kv_heads * kHeadDim;
const int cache_head_stride = kHeadDim;
const int token_lane = int(lane >> 1);
const int dim_lane = int(lane & 1);

threadgroup float scratch[kNumWarps * kHeadDim + 2 * kNumWarps];
float running_max = -INFINITY;
float running_sum = 0.0f;
float v_acc[kVElemsPerLane];
for (int i = 0; i < kVElemsPerLane; ++i) {
    v_acc[i] = 0.0f;
}

half4 q_vecs[kHeadDim / 8];
for (int i = 0; i < kHeadDim / 8; ++i) {
    const int dim = dim_lane * 4 + i * 8;
    q_vecs[i] = *reinterpret_cast<const device half4*>(q + q_base + dim);
}

for (
    int block_idx = int(warp_idx);
    block_idx < num_context_blocks;
    block_idx += kNumWarps
) {
    const int block = block_tables[batch_idx * max_blocks + block_idx];
    const int token_idx = block_idx * kBlockSize + token_lane;
    float partial_score = 0.0f;
    if (token_idx < seq_len) {
        const int k_base =
            block * cache_block_stride
            + token_lane * cache_offset_stride
            + kv_head * cache_head_stride;
        for (int i = 0; i < kHeadDim / 8; ++i) {
            const int dim = dim_lane * 4 + i * 8;
            const half4 k_vec =
                *reinterpret_cast<const device half4*>(k_cache + k_base + dim);
            partial_score += dot(
                static_cast<float4>(q_vecs[i]), static_cast<float4>(k_vec)
            );
        }
    }
    const float paired_score = partial_score + simd_shuffle_xor(partial_score, 1);
    const float lane_score = token_idx < seq_len ? paired_score * scale : -INFINITY;

    const int block_valid_tokens =
        min(kBlockSize, seq_len - block_idx * kBlockSize);
    float block_max = lane_score;
    for (int mask = kNumSimdLanes / 2; mask >= 1; mask >>= 1) {
        block_max = max(block_max, simd_shuffle_xor(block_max, mask));
    }

    float new_max = max(running_max, block_max);
    if (new_max == -INFINITY) {
        new_max = 0.0f;
    }
    float old_correction =
        running_max == -INFINITY ? 0.0f : sgl_fast_exp(running_max - new_max);
    for (int i = 0; i < kVElemsPerLane; ++i) {
        v_acc[i] *= old_correction;
    }
    running_sum *= old_correction;
    running_max = new_max;

    for (int token = 0; token < block_valid_tokens; ++token) {
        const float token_score =
            simd_shuffle(lane_score, static_cast<ushort>(token * 2));
        const float weight = sgl_fast_exp(token_score - running_max);
        running_sum += weight;
        const int v_base =
            block * cache_block_stride
            + token * cache_offset_stride
            + kv_head * cache_head_stride;
        for (int i = 0; i < kVElemsPerLane; ++i) {
            const int dim = int(lane) + i * kNumSimdLanes;
            v_acc[i] += weight * static_cast<float>(v_cache[v_base + dim]);
        }
    }
}

threadgroup_barrier(mem_flags::mem_threadgroup);
threadgroup float* merge_max = scratch;
threadgroup float* merge_sum = merge_max + kNumWarps;
threadgroup float* merge_out = merge_sum + kNumWarps;
if (lane == 0) {
    merge_max[warp_idx] = running_max;
    merge_sum[warp_idx] = running_sum;
}
threadgroup float* this_out = merge_out + warp_idx * kHeadDim;
for (int i = 0; i < kVElemsPerLane; ++i) {
    this_out[int(lane) + i * kNumSimdLanes] = v_acc[i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (warp_idx == 0) {
    for (int warp = 1; warp < kNumWarps; ++warp) {
        const float other_max = merge_max[warp];
        const float other_sum = merge_sum[warp];
        if (other_max == -INFINITY && other_sum == 0.0f) {
            continue;
        }
        float new_max = max(running_max, other_max);
        if (new_max == -INFINITY) {
            new_max = 0.0f;
        }
        const float this_correction =
            running_max == -INFINITY ? 0.0f : sgl_fast_exp(running_max - new_max);
        const float other_correction =
            other_max == -INFINITY ? 0.0f : sgl_fast_exp(other_max - new_max);
        const threadgroup float* other_out = merge_out + warp * kHeadDim;
        for (int i = 0; i < kVElemsPerLane; ++i) {
            const int dim = int(lane) + i * kNumSimdLanes;
            v_acc[i] =
                v_acc[i] * this_correction + other_out[dim] * other_correction;
        }
        running_sum = running_sum * this_correction + other_sum * other_correction;
        running_max = new_max;
    }
    const float inv_sum = 1.0f / (running_sum + 1e-6f);
    for (int i = 0; i < kVElemsPerLane; ++i) {
        const int dim = int(lane) + i * kNumSimdLanes;
        out[out_base + dim] = static_cast<half>(v_acc[i] * inv_sum);
    }
}
"""

# Harness-only comparison variant. The default runtime source is redefined
# below to the accepted shared-score kernel.
_DECODE_PAGED_H128_B16_REGSCORE_SOURCE = _DECODE_PAGED_H128_B16_SOURCE

_DECODE_PAGED_H128_B16_SOURCE = """
constexpr int kHeadDim = 128;
constexpr int kBlockSize = 16;
constexpr int kNumThreads = 256;
constexpr int kNumSimdLanes = 32;
constexpr int kNumWarps = kNumThreads / kNumSimdLanes;
constexpr int kVElemsPerLane = kHeadDim / kNumSimdLanes;

const uint local = thread_position_in_threadgroup.x;
const uint group = thread_position_in_grid.x / kNumThreads;
const uint lane = thread_index_in_simdgroup;
const uint warp_idx = local / kNumSimdLanes;
const int head = int(group);
const int batch_idx = int(thread_position_in_grid.y);
if (local >= kNumThreads || batch_idx >= batch || head >= num_heads) {
    return;
}

const int seq_len = min(context_lens[batch_idx], max_blocks * kBlockSize);
if (seq_len <= 0) {
    return;
}
const int num_context_blocks = (seq_len + kBlockSize - 1) / kBlockSize;
const int kv_head = head / (num_heads / num_kv_heads);
const int q_base = (batch_idx * num_heads + head) * kHeadDim;
const int out_base = (batch_idx * num_heads + head) * kHeadDim;
const int cache_block_stride = kBlockSize * num_kv_heads * kHeadDim;
const int cache_offset_stride = num_kv_heads * kHeadDim;
const int cache_head_stride = kHeadDim;
const int token_lane = int(lane >> 1);
const int dim_lane = int(lane & 1);

threadgroup float scratch[kNumWarps * kHeadDim + 2 * kNumWarps];
threadgroup float* warp_scores = scratch + warp_idx * kBlockSize;
float running_max = -INFINITY;
float running_sum = 0.0f;
float v_acc[kVElemsPerLane];
for (int i = 0; i < kVElemsPerLane; ++i) {
    v_acc[i] = 0.0f;
}

half4 q_vecs[kHeadDim / 8];
for (int i = 0; i < kHeadDim / 8; ++i) {
    const int dim = dim_lane * 4 + i * 8;
    q_vecs[i] = *reinterpret_cast<const device half4*>(q + q_base + dim);
}

for (
    int block_idx = int(warp_idx);
    block_idx < num_context_blocks;
    block_idx += kNumWarps
) {
    const int block = block_tables[batch_idx * max_blocks + block_idx];
    const int token_idx = block_idx * kBlockSize + token_lane;
    float partial_score = 0.0f;
    if (token_idx < seq_len) {
        const int k_base =
            block * cache_block_stride
            + token_lane * cache_offset_stride
            + kv_head * cache_head_stride;
        for (int i = 0; i < kHeadDim / 8; ++i) {
            const int dim = dim_lane * 4 + i * 8;
            const half4 k_vec =
                *reinterpret_cast<const device half4*>(k_cache + k_base + dim);
            partial_score += dot(
                static_cast<float4>(q_vecs[i]), static_cast<float4>(k_vec)
            );
        }
    }
    const float paired_score = partial_score + simd_shuffle_xor(partial_score, 1);
    if (dim_lane == 0) {
        warp_scores[token_lane] =
            token_idx < seq_len ? paired_score * scale : -INFINITY;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const int block_valid_tokens =
        min(kBlockSize, seq_len - block_idx * kBlockSize);
    float block_max = -INFINITY;
    for (
        int token = int(lane);
        token < block_valid_tokens;
        token += kNumSimdLanes
    ) {
        block_max = max(block_max, warp_scores[token]);
    }
    for (int mask = kNumSimdLanes / 2; mask >= 1; mask >>= 1) {
        block_max = max(block_max, simd_shuffle_xor(block_max, mask));
    }

    float new_max = max(running_max, block_max);
    if (new_max == -INFINITY) {
        new_max = 0.0f;
    }
    float old_correction =
        running_max == -INFINITY ? 0.0f : sgl_fast_exp(running_max - new_max);
    for (int i = 0; i < kVElemsPerLane; ++i) {
        v_acc[i] *= old_correction;
    }
    running_sum *= old_correction;
    running_max = new_max;

    for (int token = 0; token < block_valid_tokens; ++token) {
        const float weight = sgl_fast_exp(warp_scores[token] - running_max);
        running_sum += weight;
        const int v_base =
            block * cache_block_stride
            + token * cache_offset_stride
            + kv_head * cache_head_stride;
        for (int i = 0; i < kVElemsPerLane; ++i) {
            const int dim = int(lane) + i * kNumSimdLanes;
            v_acc[i] += weight * static_cast<float>(v_cache[v_base + dim]);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

threadgroup_barrier(mem_flags::mem_threadgroup);
threadgroup float* merge_max = scratch;
threadgroup float* merge_sum = merge_max + kNumWarps;
threadgroup float* merge_out = merge_sum + kNumWarps;
if (lane == 0) {
    merge_max[warp_idx] = running_max;
    merge_sum[warp_idx] = running_sum;
}
threadgroup float* this_out = merge_out + warp_idx * kHeadDim;
for (int i = 0; i < kVElemsPerLane; ++i) {
    this_out[int(lane) + i * kNumSimdLanes] = v_acc[i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (warp_idx == 0) {
    for (int warp = 1; warp < kNumWarps; ++warp) {
        const float other_max = merge_max[warp];
        const float other_sum = merge_sum[warp];
        if (other_max == -INFINITY && other_sum == 0.0f) {
            continue;
        }
        float new_max = max(running_max, other_max);
        if (new_max == -INFINITY) {
            new_max = 0.0f;
        }
        const float this_correction =
            running_max == -INFINITY ? 0.0f : sgl_fast_exp(running_max - new_max);
        const float other_correction =
            other_max == -INFINITY ? 0.0f : sgl_fast_exp(other_max - new_max);
        const threadgroup float* other_out = merge_out + warp * kHeadDim;
        for (int i = 0; i < kVElemsPerLane; ++i) {
            const int dim = int(lane) + i * kNumSimdLanes;
            v_acc[i] =
                v_acc[i] * this_correction + other_out[dim] * other_correction;
        }
        running_sum = running_sum * this_correction + other_sum * other_correction;
        running_max = new_max;
    }
    const float inv_sum = 1.0f / (running_sum + 1e-6f);
    for (int i = 0; i < kVElemsPerLane; ++i) {
        const int dim = int(lane) + i * kNumSimdLanes;
        out[out_base + dim] = static_cast<half>(v_acc[i] * inv_sum);
    }
}
"""

_decode_dense_h128_kernels = {}
_decode_dense_gqa2_h128_kernels = {}
_decode_paged_h128_b16_kernel = None
_decode_paged_h128_b16_regscore_kernel = None
_decode_paged_p1_h128_kernel = None
_decode_paged_p1_h128_scalar_kernel = None


def _get_decode_dense_h128_kernel(mx, *, mode: str, threads: int):
    if mode not in ("regscore", "sharedscore"):
        raise ValueError("dense h128 decode mode must be 'regscore' or 'sharedscore'")
    if threads not in (128, 256):
        raise ValueError("dense h128 decode threads must be 128 or 256")
    key = (mode, threads)
    if key not in _decode_dense_h128_kernels:
        _decode_dense_h128_kernels[key] = mx.fast.metal_kernel(
            name=f"sgl_decode_dense_h128_{mode}_{threads}",
            input_names=[
                "q",
                "k",
                "v",
                "scale",
                "batch",
                "num_heads",
                "num_kv_heads",
                "seq_len",
            ],
            output_names=["out"],
            header=_DECODE_PAGED_H128_B16_HEADER,
            source=_dense_decode_source(mode=mode, threads=threads),
            ensure_row_contiguous=True,
        )
    return _decode_dense_h128_kernels[key]


def decode_attention_dense_h128_unchecked(
    q: "mx.array",
    k: "mx.array",
    v: "mx.array",
    scale: float,
    *,
    mode: str = "sharedscore",
    threads: int = 256,
) -> "mx.array":
    import mlx.core as mx

    if q.dtype != mx.float16 or k.dtype != mx.float16 or v.dtype != mx.float16:
        raise ValueError("dense h128 decode supports only float16 tensors")
    if q.ndim != 4 or q.shape[2] != 1 or q.shape[3] != 128:
        raise ValueError("dense h128 decode query must have shape (B, H, 1, 128)")
    if k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        raise ValueError("dense h128 decode K/V caches must have matching 4-D shapes")
    if q.shape[0] != k.shape[0]:
        raise ValueError("dense h128 decode query and K/V batch dimensions must match")
    if k.shape[3] != 128:
        raise ValueError("dense h128 decode K/V head dimension must be 128")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("dense h128 decode query heads must be divisible by KV heads")
    if k.shape[2] <= 0:
        raise ValueError("dense h128 decode sequence length must be positive")

    q = mx.contiguous(q)
    k = mx.contiguous(k)
    v = mx.contiguous(v)
    batch = int(q.shape[0])
    num_heads = int(q.shape[1])
    num_kv_heads = int(k.shape[1])
    seq_len = int(k.shape[2])
    scale = _scale_value(scale, "decode_attention_dense_h128_unchecked")
    kernel = _get_decode_dense_h128_kernel(mx, mode=mode, threads=threads)
    return kernel(
        inputs=[
            q,
            k,
            v,
            float(scale),
            batch,
            num_heads,
            num_kv_heads,
            seq_len,
        ],
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
        grid=(num_heads * threads, batch, 1),
        threadgroup=(threads, 1, 1),
    )[0]


def _get_decode_dense_gqa2_h128_kernel(mx, *, threads: int):
    if threads not in (128, 256):
        raise ValueError("dense GQA2 h128 decode threads must be 128 or 256")
    if threads not in _decode_dense_gqa2_h128_kernels:
        _decode_dense_gqa2_h128_kernels[threads] = mx.fast.metal_kernel(
            name=f"sgl_decode_dense_gqa2_h128_{threads}",
            input_names=[
                "q",
                "k",
                "v",
                "scale",
                "batch",
                "num_heads",
                "num_kv_heads",
                "seq_len",
            ],
            output_names=["out"],
            header=_DECODE_PAGED_H128_B16_HEADER,
            source=_dense_gqa2_decode_source(threads=threads),
            ensure_row_contiguous=True,
        )
    return _decode_dense_gqa2_h128_kernels[threads]


def decode_attention_dense_gqa2_h128_unchecked(
    q: "mx.array",
    k: "mx.array",
    v: "mx.array",
    scale: float,
    *,
    threads: int = 256,
) -> "mx.array":
    import mlx.core as mx

    if q.dtype != mx.float16 or k.dtype != mx.float16 or v.dtype != mx.float16:
        raise ValueError("dense GQA2 h128 decode supports only float16 tensors")
    if q.ndim != 4 or q.shape[2] != 1 or q.shape[3] != 128:
        raise ValueError("dense GQA2 h128 decode query must have shape (B, H, 1, 128)")
    if k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        raise ValueError(
            "dense GQA2 h128 decode K/V caches must have matching 4-D shapes"
        )
    if q.shape[0] != k.shape[0]:
        raise ValueError(
            "dense GQA2 h128 decode query and K/V batch dimensions must match"
        )
    if q.shape[1] != 2 * k.shape[1]:
        raise ValueError(
            "dense GQA2 h128 decode requires exactly two query heads per KV head"
        )
    if k.shape[3] != 128:
        raise ValueError("dense GQA2 h128 decode K/V head dimension must be 128")
    if k.shape[2] <= 0:
        raise ValueError("dense GQA2 h128 decode sequence length must be positive")

    q = mx.contiguous(q)
    k = mx.contiguous(k)
    v = mx.contiguous(v)
    batch = int(q.shape[0])
    num_heads = int(q.shape[1])
    num_kv_heads = int(k.shape[1])
    seq_len = int(k.shape[2])
    scale = _scale_value(scale, "decode_attention_dense_gqa2_h128_unchecked")
    kernel = _get_decode_dense_gqa2_h128_kernel(mx, threads=threads)
    return kernel(
        inputs=[
            q,
            k,
            v,
            float(scale),
            batch,
            num_heads,
            num_kv_heads,
            seq_len,
        ],
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
        grid=(num_kv_heads * threads, batch, 1),
        threadgroup=(threads, 1, 1),
    )[0]


def _get_decode_paged_h128_b16_kernel(mx):
    global _decode_paged_h128_b16_kernel
    if _decode_paged_h128_b16_kernel is None:
        _decode_paged_h128_b16_kernel = mx.fast.metal_kernel(
            name="sgl_decode_paged_h128_b16_lazy",
            input_names=[
                "q",
                "k_cache",
                "v_cache",
                "block_tables",
                "context_lens",
                "scale",
                "batch",
                "num_heads",
                "num_kv_heads",
                "max_blocks",
            ],
            output_names=["out"],
            header=_DECODE_PAGED_H128_B16_HEADER,
            source=_DECODE_PAGED_H128_B16_SOURCE,
            ensure_row_contiguous=True,
        )
    return _decode_paged_h128_b16_kernel


def _get_decode_paged_h128_b16_regscore_kernel(mx):
    global _decode_paged_h128_b16_regscore_kernel
    if _decode_paged_h128_b16_regscore_kernel is None:
        _decode_paged_h128_b16_regscore_kernel = mx.fast.metal_kernel(
            name="sgl_decode_paged_h128_b16_regscore_lazy",
            input_names=[
                "q",
                "k_cache",
                "v_cache",
                "block_tables",
                "context_lens",
                "scale",
                "batch",
                "num_heads",
                "num_kv_heads",
                "max_blocks",
            ],
            output_names=["out"],
            header=_DECODE_PAGED_H128_B16_HEADER,
            source=_DECODE_PAGED_H128_B16_REGSCORE_SOURCE,
            ensure_row_contiguous=True,
        )
    return _decode_paged_h128_b16_regscore_kernel


def _decode_attention_paged_lazy_with_kernel(
    q: "mx.array",
    k_cache: "mx.array",
    v_cache: "mx.array",
    block_tables: "mx.array",
    context_lens: "mx.array",
    scale: float,
    kernel_getter,
) -> "mx.array":
    import mlx.core as mx

    if (
        q.dtype != mx.float16
        or k_cache.dtype != mx.float16
        or v_cache.dtype != mx.float16
    ):
        raise ValueError("lazy paged decode supports only float16 tensors")
    if q.ndim != 4 or q.shape[2] != 1 or q.shape[3] != 128:
        raise ValueError("lazy paged decode query must have shape (B, H, 1, 128)")
    if k_cache.ndim != 4 or v_cache.ndim != 4 or k_cache.shape != v_cache.shape:
        raise ValueError("lazy paged decode K/V caches must have matching 4-D shapes")
    if k_cache.shape[1] != 16 or k_cache.shape[3] != 128:
        raise ValueError("lazy paged decode requires block_size=16 and head_dim=128")
    if q.shape[1] % k_cache.shape[2] != 0:
        raise ValueError("lazy paged decode query heads must be divisible by KV heads")
    if block_tables.ndim != 2 or block_tables.shape[0] != q.shape[0]:
        raise ValueError(
            "lazy paged decode block_tables must have shape (B, max_blocks)"
        )
    if context_lens.ndim != 1 or context_lens.shape[0] != q.shape[0]:
        raise ValueError("lazy paged decode context_lens must have shape (B,)")

    q = mx.contiguous(q)
    block_tables = mx.contiguous(block_tables)
    context_lens = mx.contiguous(context_lens)
    batch = int(q.shape[0])
    num_heads = int(q.shape[1])
    num_kv_heads = int(k_cache.shape[2])
    max_blocks = int(block_tables.shape[1])
    kernel = kernel_getter(mx)
    return kernel(
        inputs=[
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            float(scale),
            batch,
            num_heads,
            num_kv_heads,
            max_blocks,
        ],
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
        grid=(num_heads * 256, batch, 1),
        threadgroup=(256, 1, 1),
    )[0]


def decode_attention_paged_lazy_unchecked(
    q: "mx.array",
    k_cache: "mx.array",
    v_cache: "mx.array",
    block_tables: "mx.array",
    context_lens: "mx.array",
    scale: float,
) -> "mx.array":
    return _decode_attention_paged_lazy_with_kernel(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale,
        _get_decode_paged_h128_b16_kernel,
    )


def decode_attention_paged_lazy_regscore_unchecked(
    q: "mx.array",
    k_cache: "mx.array",
    v_cache: "mx.array",
    block_tables: "mx.array",
    context_lens: "mx.array",
    scale: float,
) -> "mx.array":
    return _decode_attention_paged_lazy_with_kernel(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale,
        _get_decode_paged_h128_b16_regscore_kernel,
    )


_DECODE_PAGED_P1_H128_SOURCE = """
constexpr int kHeadDim = 128;
constexpr int kNumThreads = 256;
constexpr int kNumSimdLanes = 32;
constexpr int kNumWarps = kNumThreads / kNumSimdLanes;
constexpr int kElemsPerLane = kHeadDim / kNumSimdLanes;

const uint local = thread_position_in_threadgroup.x;
const uint group = thread_position_in_grid.x / kNumThreads;
const uint lane = thread_index_in_simdgroup;
const uint warp_idx = local / kNumSimdLanes;
const int head = int(group);
const int batch_idx = int(thread_position_in_grid.y);
if (local >= kNumThreads || batch_idx >= batch || head >= num_heads) {
    return;
}

const int seq_len = min(context_lens[batch_idx], max_blocks);
if (seq_len <= 0) {
    return;
}
const int kv_head = head / (num_heads / num_kv_heads);
const int q_base = (batch_idx * num_heads + head) * kHeadDim;
const int out_base = (batch_idx * num_heads + head) * kHeadDim;
const int cache_block_stride = num_kv_heads * kHeadDim;
const int cache_head_stride = kHeadDim;
const int lane_dim = int(lane) * kElemsPerLane;

threadgroup float scratch[kNumWarps * kHeadDim + 2 * kNumWarps];
float running_max = -INFINITY;
float running_sum = 0.0f;
float4 v_acc = float4(0.0f);

const half4 q_vec =
    *reinterpret_cast<const device half4*>(q + q_base + lane_dim);

for (int token_idx = int(warp_idx); token_idx < seq_len; token_idx += kNumWarps) {
    const int block = block_tables[batch_idx * max_blocks + token_idx];
    const int kv_base = block * cache_block_stride + kv_head * cache_head_stride;
    const half4 k_vec =
        *reinterpret_cast<const device half4*>(k_cache + kv_base + lane_dim);
    float score = dot(static_cast<float4>(q_vec), static_cast<float4>(k_vec));
    for (int mask = kNumSimdLanes / 2; mask >= 1; mask >>= 1) {
        score += simd_shuffle_xor(score, mask);
    }
    score *= scale;

    float new_max = max(running_max, score);
    if (new_max == -INFINITY) {
        new_max = 0.0f;
    }
    const float old_correction =
        running_max == -INFINITY ? 0.0f : sgl_fast_exp(running_max - new_max);
    const float weight = sgl_fast_exp(score - new_max);
    v_acc *= old_correction;
    running_sum = running_sum * old_correction + weight;
    running_max = new_max;

    const half4 v_vec =
        *reinterpret_cast<const device half4*>(v_cache + kv_base + lane_dim);
    v_acc += weight * static_cast<float4>(v_vec);
}

threadgroup_barrier(mem_flags::mem_threadgroup);
threadgroup float* merge_max = scratch;
threadgroup float* merge_sum = merge_max + kNumWarps;
threadgroup float* merge_out = merge_sum + kNumWarps;
if (lane == 0) {
    merge_max[warp_idx] = running_max;
    merge_sum[warp_idx] = running_sum;
}
threadgroup float* this_out = merge_out + warp_idx * kHeadDim;
this_out[lane_dim + 0] = v_acc.x;
this_out[lane_dim + 1] = v_acc.y;
this_out[lane_dim + 2] = v_acc.z;
this_out[lane_dim + 3] = v_acc.w;
threadgroup_barrier(mem_flags::mem_threadgroup);

if (warp_idx == 0) {
    for (int warp = 1; warp < kNumWarps; ++warp) {
        const float other_max = merge_max[warp];
        const float other_sum = merge_sum[warp];
        if (other_max == -INFINITY && other_sum == 0.0f) {
            continue;
        }
        float new_max = max(running_max, other_max);
        if (new_max == -INFINITY) {
            new_max = 0.0f;
        }
        const float this_correction =
            running_max == -INFINITY ? 0.0f : sgl_fast_exp(running_max - new_max);
        const float other_correction =
            other_max == -INFINITY ? 0.0f : sgl_fast_exp(other_max - new_max);
        const threadgroup float* other_out = merge_out + warp * kHeadDim;
        v_acc.x = v_acc.x * this_correction + other_out[lane_dim + 0] * other_correction;
        v_acc.y = v_acc.y * this_correction + other_out[lane_dim + 1] * other_correction;
        v_acc.z = v_acc.z * this_correction + other_out[lane_dim + 2] * other_correction;
        v_acc.w = v_acc.w * this_correction + other_out[lane_dim + 3] * other_correction;
        running_sum = running_sum * this_correction + other_sum * other_correction;
        running_max = new_max;
    }
    const float inv_sum = 1.0f / (running_sum + 1e-6f);
    out[out_base + lane_dim + 0] = static_cast<half>(v_acc.x * inv_sum);
    out[out_base + lane_dim + 1] = static_cast<half>(v_acc.y * inv_sum);
    out[out_base + lane_dim + 2] = static_cast<half>(v_acc.z * inv_sum);
    out[out_base + lane_dim + 3] = static_cast<half>(v_acc.w * inv_sum);
}
"""


_DECODE_PAGED_P1_H128_SCALAR_SOURCE = """
constexpr int kHeadDim = 128;
constexpr int kNumThreads = 256;
constexpr int kNumSimdLanes = 32;
constexpr int kNumWarps = kNumThreads / kNumSimdLanes;
constexpr int kElemsPerLane = kHeadDim / kNumSimdLanes;

const uint local = thread_position_in_threadgroup.x;
const uint group = thread_position_in_grid.x / kNumThreads;
const uint lane = thread_index_in_simdgroup;
const uint warp_idx = local / kNumSimdLanes;
const int head = int(group);
const int batch_idx = int(thread_position_in_grid.y);
if (local >= kNumThreads || batch_idx >= batch || head >= num_heads) {
    return;
}

const int seq_len = min(context_lens[batch_idx], max_blocks);
if (seq_len <= 0) {
    return;
}
const int kv_head = head / (num_heads / num_kv_heads);
const int q_base = (batch_idx * num_heads + head) * kHeadDim;
const int out_base = (batch_idx * num_heads + head) * kHeadDim;
const int cache_block_stride = num_kv_heads * kHeadDim;
const int cache_head_stride = kHeadDim;
const int lane_dim = int(lane) * kElemsPerLane;

threadgroup float scratch[kNumWarps * kHeadDim + 2 * kNumWarps];
float running_max = -INFINITY;
float running_sum = 0.0f;
float4 v_acc = float4(0.0f);

for (int token_idx = int(warp_idx); token_idx < seq_len; token_idx += kNumWarps) {
    const int block = block_tables[batch_idx * max_blocks + token_idx];
    const int kv_base = block * cache_block_stride + kv_head * cache_head_stride;

    float score = 0.0f;
    score += static_cast<float>(q[q_base + lane_dim + 0])
        * static_cast<float>(k_cache[kv_base + lane_dim + 0]);
    score += static_cast<float>(q[q_base + lane_dim + 1])
        * static_cast<float>(k_cache[kv_base + lane_dim + 1]);
    score += static_cast<float>(q[q_base + lane_dim + 2])
        * static_cast<float>(k_cache[kv_base + lane_dim + 2]);
    score += static_cast<float>(q[q_base + lane_dim + 3])
        * static_cast<float>(k_cache[kv_base + lane_dim + 3]);
    for (int mask = kNumSimdLanes / 2; mask >= 1; mask >>= 1) {
        score += simd_shuffle_xor(score, mask);
    }
    score *= scale;

    float new_max = max(running_max, score);
    if (new_max == -INFINITY) {
        new_max = 0.0f;
    }
    const float old_correction =
        running_max == -INFINITY ? 0.0f : sgl_fast_exp(running_max - new_max);
    const float weight = sgl_fast_exp(score - new_max);
    v_acc *= old_correction;
    running_sum = running_sum * old_correction + weight;
    running_max = new_max;

    v_acc.x += weight * static_cast<float>(v_cache[kv_base + lane_dim + 0]);
    v_acc.y += weight * static_cast<float>(v_cache[kv_base + lane_dim + 1]);
    v_acc.z += weight * static_cast<float>(v_cache[kv_base + lane_dim + 2]);
    v_acc.w += weight * static_cast<float>(v_cache[kv_base + lane_dim + 3]);
}

threadgroup_barrier(mem_flags::mem_threadgroup);
threadgroup float* merge_max = scratch;
threadgroup float* merge_sum = merge_max + kNumWarps;
threadgroup float* merge_out = merge_sum + kNumWarps;
if (lane == 0) {
    merge_max[warp_idx] = running_max;
    merge_sum[warp_idx] = running_sum;
}
threadgroup float* this_out = merge_out + warp_idx * kHeadDim;
this_out[lane_dim + 0] = v_acc.x;
this_out[lane_dim + 1] = v_acc.y;
this_out[lane_dim + 2] = v_acc.z;
this_out[lane_dim + 3] = v_acc.w;
threadgroup_barrier(mem_flags::mem_threadgroup);

if (warp_idx == 0) {
    for (int warp = 1; warp < kNumWarps; ++warp) {
        const float other_max = merge_max[warp];
        const float other_sum = merge_sum[warp];
        if (other_max == -INFINITY && other_sum == 0.0f) {
            continue;
        }
        float new_max = max(running_max, other_max);
        if (new_max == -INFINITY) {
            new_max = 0.0f;
        }
        const float this_correction =
            running_max == -INFINITY ? 0.0f : sgl_fast_exp(running_max - new_max);
        const float other_correction =
            other_max == -INFINITY ? 0.0f : sgl_fast_exp(other_max - new_max);
        const threadgroup float* other_out = merge_out + warp * kHeadDim;
        v_acc.x = v_acc.x * this_correction + other_out[lane_dim + 0] * other_correction;
        v_acc.y = v_acc.y * this_correction + other_out[lane_dim + 1] * other_correction;
        v_acc.z = v_acc.z * this_correction + other_out[lane_dim + 2] * other_correction;
        v_acc.w = v_acc.w * this_correction + other_out[lane_dim + 3] * other_correction;
        running_sum = running_sum * this_correction + other_sum * other_correction;
        running_max = new_max;
    }
    const float inv_sum = 1.0f / (running_sum + 1e-6f);
    out[out_base + lane_dim + 0] = static_cast<T>(v_acc.x * inv_sum);
    out[out_base + lane_dim + 1] = static_cast<T>(v_acc.y * inv_sum);
    out[out_base + lane_dim + 2] = static_cast<T>(v_acc.z * inv_sum);
    out[out_base + lane_dim + 3] = static_cast<T>(v_acc.w * inv_sum);
}
"""


def _get_decode_paged_p1_h128_kernel(mx):
    global _decode_paged_p1_h128_kernel
    if _decode_paged_p1_h128_kernel is None:
        _decode_paged_p1_h128_kernel = mx.fast.metal_kernel(
            name="sgl_decode_paged_p1_h128_lazy",
            input_names=[
                "q",
                "k_cache",
                "v_cache",
                "block_tables",
                "context_lens",
                "scale",
                "batch",
                "num_heads",
                "num_kv_heads",
                "max_blocks",
            ],
            output_names=["out"],
            header=_DECODE_PAGED_H128_B16_HEADER,
            source=_DECODE_PAGED_P1_H128_SOURCE,
            ensure_row_contiguous=True,
        )
    return _decode_paged_p1_h128_kernel


def _get_decode_paged_p1_h128_scalar_kernel(mx):
    global _decode_paged_p1_h128_scalar_kernel
    if _decode_paged_p1_h128_scalar_kernel is None:
        _decode_paged_p1_h128_scalar_kernel = mx.fast.metal_kernel(
            name="sgl_decode_paged_p1_h128_scalar_lazy",
            input_names=[
                "q",
                "k_cache",
                "v_cache",
                "block_tables",
                "context_lens",
                "scale",
                "batch",
                "num_heads",
                "num_kv_heads",
                "max_blocks",
            ],
            output_names=["out"],
            header=_DECODE_PAGED_H128_B16_HEADER,
            source=_DECODE_PAGED_P1_H128_SCALAR_SOURCE,
            ensure_row_contiguous=True,
        )
    return _decode_paged_p1_h128_scalar_kernel


def decode_attention_paged_p1_lazy_unchecked(
    q: "mx.array",
    k_cache: "mx.array",
    v_cache: "mx.array",
    block_tables: "mx.array",
    context_lens: "mx.array",
    scale: float,
) -> "mx.array":
    import mlx.core as mx

    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("lazy p1 paged decode requires matching Q/K/V dtypes")
    if q.dtype not in (mx.float16, mx.bfloat16):
        raise ValueError("lazy p1 paged decode supports only float16/bfloat16 tensors")
    if q.ndim != 4 or q.shape[2] != 1 or q.shape[3] != 128:
        raise ValueError("lazy p1 paged decode query must have shape (B, H, 1, 128)")
    if k_cache.shape != v_cache.shape:
        raise ValueError("lazy p1 paged decode K/V caches must have matching shapes")
    if k_cache.ndim == 4:
        if k_cache.shape[1] != 1 or k_cache.shape[3] != 128:
            raise ValueError(
                "lazy p1 paged decode requires block_size=1 and head_dim=128"
            )
        num_kv_heads = int(k_cache.shape[2])
    elif k_cache.ndim == 3:
        if k_cache.shape[2] != 128:
            raise ValueError("lazy p1 paged decode requires head_dim=128")
        num_kv_heads = int(k_cache.shape[1])
    else:
        raise ValueError("lazy p1 paged decode requires 3-D or 4-D p1 K/V caches")
    if q.shape[1] % num_kv_heads != 0:
        raise ValueError(
            "lazy p1 paged decode query heads must be divisible by KV heads"
        )
    if block_tables.ndim != 2 or block_tables.shape[0] != q.shape[0]:
        raise ValueError(
            "lazy p1 paged decode block_tables must have shape (B, max_blocks)"
        )
    if context_lens.ndim != 1 or context_lens.shape[0] != q.shape[0]:
        raise ValueError("lazy p1 paged decode context_lens must have shape (B,)")

    q = mx.contiguous(q)
    k_cache = mx.contiguous(k_cache)
    v_cache = mx.contiguous(v_cache)
    block_tables = mx.contiguous(block_tables)
    context_lens = mx.contiguous(context_lens)
    batch = int(q.shape[0])
    num_heads = int(q.shape[1])
    max_blocks = int(block_tables.shape[1])
    kernel = (
        _get_decode_paged_p1_h128_kernel(mx)
        if q.dtype == mx.float16
        else _get_decode_paged_p1_h128_scalar_kernel(mx)
    )
    kwargs = {}
    if q.dtype == mx.bfloat16:
        kwargs["template"] = [("T", q.dtype)]
    return kernel(
        inputs=[
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            float(scale),
            batch,
            num_heads,
            num_kv_heads,
            max_blocks,
        ],
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
        grid=(num_heads * 256, batch, 1),
        threadgroup=(256, 1, 1),
        **kwargs,
    )[0]


def _try_prefill_attention_paged_dense_prefix(
    q: "mx.array",
    k: "mx.array",
    v: "mx.array",
    k_cache: "mx.array",
    v_cache: "mx.array",
    block_tables: "mx.array",
    prefix_lens: "mx.array",
    cu_seqlens_q: "mx.array",
    scale: float,
    causal: bool,
    mx,
):
    _, block_size, _, _ = _paged_cache_shape(k_cache, "prefill_attention_paged")
    if q.dtype != mx.float16 or q.shape[2] != 128:
        return None

    mx.eval(prefix_lens, cu_seqlens_q)
    prefix_values = [int(value) for value in prefix_lens.tolist()]
    if not prefix_values or len(set(prefix_values)) != 1:
        return None
    prefix_len = prefix_values[0]
    if prefix_len <= 0:
        return None

    cu_q = [int(value) for value in cu_seqlens_q.tolist()]
    q_lens = [end - start for start, end in zip(cu_q, cu_q[1:])]
    if not q_lens or len(set(q_lens)) != 1:
        return None
    q_len = q_lens[0]
    batch = len(q_lens)
    if q_len <= 0 or q.shape[0] != batch * q_len:
        return None
    if q_len < 128:
        return None

    prefix_blocks = (prefix_len + block_size - 1) // block_size
    if prefix_blocks <= 0 or prefix_blocks > block_tables.shape[1]:
        return None

    _, _, num_kv_heads, head_dim = _paged_cache_shape(
        k_cache, "prefill_attention_paged"
    )
    prefix_tables = block_tables[:, :prefix_blocks]
    k_blocks = k_cache[prefix_tables]
    v_blocks = v_cache[prefix_tables]
    max_prefix_tokens = prefix_blocks * block_size
    k_prefix = k_blocks.reshape(batch, max_prefix_tokens, num_kv_heads, head_dim)[
        :, :prefix_len, :, :
    ]
    v_prefix = v_blocks.reshape(batch, max_prefix_tokens, num_kv_heads, head_dim)[
        :, :prefix_len, :, :
    ]
    k_suffix = k.reshape(batch, q_len, num_kv_heads, head_dim)
    v_suffix = v.reshape(batch, q_len, num_kv_heads, head_dim)
    full_len = prefix_len + q_len
    k_full = mx.contiguous(
        mx.concatenate([k_prefix, k_suffix], axis=1).reshape(
            batch * full_len, num_kv_heads, head_dim
        )
    )
    v_full = mx.contiguous(
        mx.concatenate([v_prefix, v_suffix], axis=1).reshape(
            batch * full_len, num_kv_heads, head_dim
        )
    )
    cu_k = mx.array([i * full_len for i in range(batch + 1)], dtype=mx.int32)
    return flash_attn_varlen_func(
        q,
        k_full,
        v_full,
        cu_seqlens_q,
        cu_k,
        max_seqlen_q=q_len,
        max_seqlen_k=full_len,
        softmax_scale=scale,
        causal=causal,
    )


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
    num_blocks, block_size, num_kv_heads, head_dim = _paged_cache_shape(
        k_cache, "prefill_attention_paged"
    )
    _paged_cache_shape(v_cache, "prefill_attention_paged")
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
    if q.shape[2] != k.shape[2] or q.shape[2] != head_dim:
        raise ValueError("prefill_attention_paged head dimensions must match")
    if k.shape[1] != num_kv_heads:
        raise ValueError("prefill_attention_paged KV head counts must match")
    if k.shape[1] == 0 or q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            "prefill_attention_paged query heads must be divisible by KV heads"
        )
    if q.shape[2] == 0 or q.shape[2] > _MAX_DECODE_HEAD_DIM:
        raise ValueError(
            "prefill_attention_paged head dimension must be in the range [1, 256]"
        )
    if block_size == 0:
        raise ValueError("prefill_attention_paged block size must be positive")
    if q.shape[0] > 0 and num_blocks == 0:
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
        num_blocks,
        block_size,
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

    dense_prefix_out = _try_prefill_attention_paged_dense_prefix(
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
        mx,
    )
    if dense_prefix_out is not None:
        return dense_prefix_out

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
    mx.synchronize()
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
    _require_float_or_bfloat_dtype(k, "paged_kv_scatter", mx)
    _require_float_or_bfloat_dtype(v, "paged_kv_scatter", mx)
    _require_float_or_bfloat_dtype(k_cache, "paged_kv_scatter", mx)
    _require_float_or_bfloat_dtype(v_cache, "paged_kv_scatter", mx)
    if k.ndim != 3 or v.ndim != 3:
        raise ValueError(
            "paged_kv_scatter K/V tensors must have shape (num_tokens, KVH, D)"
        )
    num_blocks, block_size, num_kv_heads, head_dim = _paged_cache_shape(
        k_cache, "paged_kv_scatter"
    )
    _paged_cache_shape(v_cache, "paged_kv_scatter")
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
    if k.shape[1] != num_kv_heads:
        raise ValueError("paged_kv_scatter KV head counts must match")
    if k.shape[2] != head_dim:
        raise ValueError("paged_kv_scatter head dimensions must match")
    if k.shape[1] == 0:
        raise ValueError("paged_kv_scatter KV head count must be positive")
    if k.shape[2] == 0 or k.shape[2] > _MAX_DECODE_HEAD_DIM:
        raise ValueError(
            "paged_kv_scatter head dimension must be in the range [1, 256]"
        )
    if block_size == 0:
        raise ValueError("paged_kv_scatter block size must be positive")
    if k.shape[0] > 0 and num_blocks == 0:
        raise ValueError("paged_kv_scatter KV cache must contain at least one block")

    cache_slot_count = num_blocks * block_size
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


def paged_kv_scatter_all_layers_unchecked(
    k_layers: Sequence["mx.array"],
    v_layers: Sequence["mx.array"],
    k_caches: Sequence["mx.array"],
    v_caches: Sequence["mx.array"],
    slot_mapping: "mx.array",
    *,
    eager: bool = False,
) -> None:
    metal = _require_metal_extension()

    import mlx.core as mx

    num_layers = len(k_layers)
    if (
        len(v_layers) != num_layers
        or len(k_caches) != num_layers
        or len(v_caches) != num_layers
    ):
        raise ValueError(
            "paged_kv_scatter_all_layers_unchecked layer counts must match"
        )
    if num_layers == 0:
        return
    _require_array(
        slot_mapping, "paged_kv_scatter_all_layers_unchecked slot_mapping", mx
    )
    if slot_mapping.dtype != mx.int32:
        raise TypeError(
            "paged_kv_scatter_all_layers_unchecked slot_mapping must be int32"
        )

    slot_mapping = mx.contiguous(slot_mapping)
    k_contiguous = []
    v_contiguous = []
    for layer_id in range(num_layers):
        if k_caches[layer_id].dtype != v_caches[layer_id].dtype:
            raise ValueError(
                "paged_kv_scatter_all_layers_unchecked K/V cache dtypes must match"
            )
        cache_dtype = k_caches[layer_id].dtype
        k_layer = k_layers[layer_id]
        v_layer = v_layers[layer_id]
        if k_layer.dtype != cache_dtype:
            k_layer = k_layer.astype(cache_dtype)
        if v_layer.dtype != cache_dtype:
            v_layer = v_layer.astype(cache_dtype)
        k_contiguous.append(mx.contiguous(k_layer))
        v_contiguous.append(mx.contiguous(v_layer))
    mx.eval(slot_mapping, *k_contiguous, *v_contiguous, *k_caches, *v_caches)

    for layer_id in range(num_layers):
        metal.paged_kv_scatter(
            k_contiguous[layer_id],
            v_contiguous[layer_id],
            k_caches[layer_id],
            v_caches[layer_id],
            slot_mapping,
        )

    if eager and slot_mapping.size:
        _, block_size, _, _ = _paged_cache_shape(
            k_caches[0], "paged_kv_scatter_all_layers_unchecked"
        )
        block_ids = slot_mapping // block_size
        block_offsets = slot_mapping % block_size
        refs = []
        for layer_id in range(num_layers):
            if block_size == 1 and k_caches[layer_id].ndim == 3:
                refs.extend(
                    [
                        k_caches[layer_id][block_ids],
                        v_caches[layer_id][block_ids],
                    ]
                )
            else:
                refs.extend(
                    [
                        k_caches[layer_id][block_ids, block_offsets],
                        v_caches[layer_id][block_ids, block_offsets],
                    ]
                )
        mx.eval(*refs)


__all__ = [
    "decode_attention",
    "decode_attention_paged",
    "decode_attention_paged_lazy_unchecked",
    "decode_attention_paged_lazy_regscore_unchecked",
    "decode_attention_paged_p1_lazy_unchecked",
    "decode_attention_paged_unchecked",
    "decode_attention_paged_with_kv",
    "decode_attention_paged_with_kv_unchecked",
    "decode_attention_ragged",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "paged_kv_scatter",
    "paged_kv_scatter_all_layers_unchecked",
    "prefill_attention_paged",
]
