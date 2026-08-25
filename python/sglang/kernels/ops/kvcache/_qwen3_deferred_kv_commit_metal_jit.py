"""Metal JIT implementation of the deferred Qwen3 KV-cache commit.

MLX custom kernels cannot mutate Torch-owned input buffers.  The whole-model
MLX island therefore returns one stacked K tensor and one stacked V tensor.
This kernel commits all 28 layer rows to SRT's ordinary NHD pool in two Metal
launches, instead of enqueueing 56 ``index_copy_`` operations.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from sglang.kernels.ops.attention.qwen3_mps import QWEN3_06B_METAL_SPEC

_NUM_LAYERS = 28
_LAYERS_PER_LAUNCH = 14
_BF16_VALUES_PER_VECTOR = 8
_THREADGROUP_WIDTH = 256


def _kernel_source(*, pool_slots: int) -> str:
    spec = QWEN3_06B_METAL_SPEC
    row_width = spec.num_kv_heads * spec.head_dim

    def entry(name: str, layer_offset: int, *, scalar: bool = False) -> str:
        element_type = "ushort" if scalar else "uint4"
        k_args = ",\n".join(
            f"    device {element_type}* k{i} [[buffer({3 + i})]]"
            for i in range(_LAYERS_PER_LAUNCH)
        )
        v_args = ",\n".join(
            f"    device {element_type}* v{i} [[buffer({3 + _LAYERS_PER_LAUNCH + i})]]"
            for i in range(_LAYERS_PER_LAUNCH)
        )
        if scalar:
            k_cases = "\n".join(
                f"    case {i}: for (uint element = 0; element < "
                "BF16_VALUES_PER_VECTOR; ++element) "
                f"k{i}[destination + element] = new_k[source + element]; break;"
                for i in range(_LAYERS_PER_LAUNCH)
            )
            v_cases = "\n".join(
                f"    case {i}: for (uint element = 0; element < "
                "BF16_VALUES_PER_VECTOR; ++element) "
                f"v{i}[destination + element] = new_v[source + element]; break;"
                for i in range(_LAYERS_PER_LAUNCH)
            )
            source_index = "source_vector * BF16_VALUES_PER_VECTOR"
            destination_index = "destination_vector * BF16_VALUES_PER_VECTOR"
        else:
            k_cases = "\n".join(
                f"    case {i}: k{i}[destination] = new_k[source]; break;"
                for i in range(_LAYERS_PER_LAUNCH)
            )
            v_cases = "\n".join(
                f"    case {i}: v{i}[destination] = new_v[source]; break;"
                for i in range(_LAYERS_PER_LAUNCH)
            )
            source_index = "source_vector"
            destination_index = "destination_vector"
        return f"""
kernel void {name}(
    const device {element_type}* new_k [[buffer(0)]],
    const device {element_type}* new_v [[buffer(1)]],
    const device long* slots [[buffer(2)]],
{k_args},
{v_args},
    uint3 position [[thread_position_in_grid]],
    uint3 grid_size [[threads_per_grid]]) {{
  const uint row = position.x;
  const uint vector = position.y;
  const uint local_layer = position.z;
  const long raw_slot = slots[row];
  if (vector >= KV_VECTORS_PER_ROW || local_layer >= LAYERS_PER_LAUNCH ||
      raw_slot < 0 || ulong(raw_slot) >= ulong(POOL_SLOTS)) {{
    return;
  }}

  const ulong source_layer = ulong(local_layer + {layer_offset});
  const ulong source_vector =
      (source_layer * ulong(grid_size.x) + ulong(row)) * KV_VECTORS_PER_ROW + vector;
  const ulong destination_vector = ulong(raw_slot) * KV_VECTORS_PER_ROW + vector;
  const ulong source = {source_index};
  const ulong destination = {destination_index};
  switch (local_layer) {{
{k_cases}
  }}
  switch (local_layer) {{
{v_cases}
  }}
}}
"""

    return f"""
#include <metal_stdlib>
using namespace metal;

#define KV_ROW_WIDTH {row_width}
#define BF16_VALUES_PER_VECTOR {_BF16_VALUES_PER_VECTOR}
#define KV_VECTORS_PER_ROW (KV_ROW_WIDTH / {_BF16_VALUES_PER_VECTOR})
#define LAYERS_PER_LAUNCH {_LAYERS_PER_LAUNCH}
#define POOL_SLOTS {pool_slots}

{entry("qwen3_commit_kv_layers_0_13_bf16", 0)}
{entry("qwen3_commit_kv_layers_14_27_bf16", _LAYERS_PER_LAUNCH)}
{entry("qwen3_commit_kv_layers_0_13_bf16_scalar", 0, scalar=True)}
{entry("qwen3_commit_kv_layers_14_27_bf16_scalar", _LAYERS_PER_LAUNCH, scalar=True)}
"""


@lru_cache(maxsize=4)
def _compile_library(pool_slots: int):
    compile_shader = getattr(torch.mps, "compile_shader", None)
    if not callable(compile_shader):
        raise RuntimeError(
            "Qwen3 deferred KV commit requires torch.mps.compile_shader from Torch 2.13"
        )
    return compile_shader(_kernel_source(pool_slots=pool_slots))


def warmup_qwen3_kv_commit(*, pool_slots: int) -> None:
    library = _compile_library(pool_slots)
    library.qwen3_commit_kv_layers_0_13_bf16
    library.qwen3_commit_kv_layers_14_27_bf16
    library.qwen3_commit_kv_layers_0_13_bf16_scalar
    library.qwen3_commit_kv_layers_14_27_bf16_scalar


def _require_bf16_contiguous(name: str, tensor: torch.Tensor) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "mps"
        or tensor.dtype != torch.bfloat16
        or not tensor.is_contiguous()
    ):
        raise RuntimeError(f"{name} must be a contiguous MPS bfloat16 tensor")


def _is_uint4_aligned(tensor: torch.Tensor) -> bool:
    return tensor.data_ptr() % 16 == 0


def qwen3_commit_deferred_kv(
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    slots: torch.Tensor,
    k_pools: list[torch.Tensor] | tuple[torch.Tensor, ...],
    v_pools: list[torch.Tensor] | tuple[torch.Tensor, ...],
) -> None:
    """Commit stacked ``[28, num_rows, 8, 128]`` KV without host sync.

    ``num_rows`` may be a decode batch or flattened prefill token rows; each
    row is committed to the corresponding entry in ``slots``.
    """
    _require_bf16_contiguous("new_k", new_k)
    _require_bf16_contiguous("new_v", new_v)
    expected_tail = (
        QWEN3_06B_METAL_SPEC.num_kv_heads,
        QWEN3_06B_METAL_SPEC.head_dim,
    )
    if new_k.ndim != 4 or tuple(new_k.shape[2:]) != expected_tail:
        raise RuntimeError(
            f"new_k must have shape [28, num_rows, 8, 128], found {tuple(new_k.shape)}"
        )
    if tuple(new_k.shape) != tuple(new_v.shape) or new_k.shape[0] != _NUM_LAYERS:
        raise RuntimeError(
            "new_k/new_v must have matching 28-layer shapes, found "
            f"{tuple(new_k.shape)} and {tuple(new_v.shape)}"
        )
    num_rows = int(new_k.shape[1])
    if (
        not isinstance(slots, torch.Tensor)
        or slots.device.type != "mps"
        or slots.dtype != torch.int64
        or not slots.is_contiguous()
        or tuple(slots.shape) != (num_rows,)
    ):
        raise RuntimeError(f"slots must be contiguous MPS int64[{num_rows}]")
    if len(k_pools) != _NUM_LAYERS or len(v_pools) != _NUM_LAYERS:
        raise RuntimeError("Qwen3 deferred KV commit requires exactly 28 K/V pools")

    pool_slots = None
    for layer, (k_pool, v_pool) in enumerate(zip(k_pools, v_pools)):
        _require_bf16_contiguous(f"k_pools[{layer}]", k_pool)
        _require_bf16_contiguous(f"v_pools[{layer}]", v_pool)
        if k_pool.ndim != 3 or tuple(k_pool.shape[1:]) != expected_tail:
            raise RuntimeError(f"k_pools[{layer}] must use NHD [slots, 8, 128] layout")
        if tuple(v_pool.shape) != tuple(k_pool.shape):
            raise RuntimeError(f"K/V pool shape differs at layer {layer}")
        if pool_slots is None:
            pool_slots = int(k_pool.shape[0])
        elif int(k_pool.shape[0]) != pool_slots:
            raise RuntimeError("all Qwen3 layer KV pools must have the same slot count")

    if num_rows == 0:
        return
    assert pool_slots is not None
    library = _compile_library(pool_slots)
    row_width = expected_tail[0] * expected_tail[1]
    if row_width % _BF16_VALUES_PER_VECTOR != 0:
        raise RuntimeError(
            "Qwen3 deferred KV rows must contain a whole number of uint4 vectors"
        )
    threads = (
        num_rows,
        row_width // _BF16_VALUES_PER_VECTOR,
        _LAYERS_PER_LAUNCH,
    )
    group_size = (1, _THREADGROUP_WIDTH, 1)
    vectorized = all(
        _is_uint4_aligned(tensor) for tensor in (new_k, new_v, *k_pools, *v_pools)
    )
    first_kernel = (
        library.qwen3_commit_kv_layers_0_13_bf16
        if vectorized
        else library.qwen3_commit_kv_layers_0_13_bf16_scalar
    )
    second_kernel = (
        library.qwen3_commit_kv_layers_14_27_bf16
        if vectorized
        else library.qwen3_commit_kv_layers_14_27_bf16_scalar
    )
    first_kernel(
        new_k,
        new_v,
        slots,
        *k_pools[:_LAYERS_PER_LAUNCH],
        *v_pools[:_LAYERS_PER_LAUNCH],
        threads=threads,
        group_size=group_size,
    )
    second_kernel(
        new_k,
        new_v,
        slots,
        *k_pools[_LAYERS_PER_LAUNCH:],
        *v_pools[_LAYERS_PER_LAUNCH:],
        threads=threads,
        group_size=group_size,
    )


__all__ = ["qwen3_commit_deferred_kv", "warmup_qwen3_kv_commit"]
