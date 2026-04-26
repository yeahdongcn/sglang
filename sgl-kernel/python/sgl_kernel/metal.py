"""Python entry points for the sgl_kernel Metal extension."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx

_METALLIB_NAME = "sgl_metal_kernels.metallib"

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


def _validate_rope_inputs(
    q: "mx.array",
    k: "mx.array",
    positions: "mx.array",
    head_dim: int,
    rope_dim: int,
    num_qo_heads: int,
    num_kv_heads: int,
) -> None:
    if _metal is None:
        raise ImportError(
            "sgl_kernel._metal is not available; reinstall sgl-kernel with "
            "`python setup_metal.py install`"
        ) from _IMPORT_ERROR

    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("q and k must have shape [tokens, heads, head_dim]")
    if positions.ndim != 1:
        raise ValueError("positions must have shape [tokens]")
    if q.shape[0] != positions.shape[0] or k.shape[0] != positions.shape[0]:
        raise ValueError("positions length must match q/k token dimension")
    if q.shape[1] != num_qo_heads or k.shape[1] != num_kv_heads:
        raise ValueError("num_qo_heads/num_kv_heads must match q/k head dimensions")
    if q.shape[2] != head_dim or k.shape[2] != head_dim:
        raise ValueError("head_dim must match q/k last dimension")
    if rope_dim <= 0 or (rope_dim % 2) != 0:
        raise ValueError("rope_dim must be a positive even integer")
    if rope_dim != head_dim:
        raise ValueError(
            "rope_neox Metal kernel currently requires rope_dim == head_dim"
        )


def rope_neox(
    q: "mx.array",
    k: "mx.array",
    cos_sin_cache: "mx.array",
    positions: "mx.array",
    *,
    head_dim: int,
    rope_dim: int,
    num_qo_heads: int,
    num_kv_heads: int,
) -> tuple["mx.array", "mx.array"]:
    """Apply NeoX-style RoPE in-place to ``q`` and ``k``.

    ``q`` and ``k`` are mutated and also returned. The kernel reads each
    element pair into registers before writing, so passing the same buffers
    as inputs and outputs is race-free. Skipping the separate output buffers
    avoids a ~200 us ``mx.zeros_like`` + zero-fill round-trip per call.

    The inputs are evaluated up-front so the underlying Metal buffers are
    materialised before they are bound to the kernel; this is the only safe
    way to use the AOT C++ entry point with potentially-lazy upstream ops.
    """
    import mlx.core as mx

    _validate_rope_inputs(
        q, k, positions, head_dim, rope_dim, num_qo_heads, num_kv_heads
    )

    mx.eval(q, k, cos_sin_cache, positions)

    _metal.rope_neox(
        q,
        k,
        cos_sin_cache,
        positions,
        q,
        k,
        head_dim,
        rope_dim,
        num_qo_heads,
        num_kv_heads,
    )
    return q, k
