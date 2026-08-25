"""Qwen3-specific semantic KV-cache operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.kernels.fused_op import BaseFusedOp, register_fused_op
from sglang.kernels.ops.attention.qwen3_mps import QWEN3_06B_METAL_SPEC
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
)

if TYPE_CHECKING:
    import torch

_MPS = frozenset({CapabilityRequirement.MPS})
_NUM_LAYERS = 28


def _is_mps_bf16_contiguous(value) -> bool:
    return (
        getattr(getattr(value, "device", None), "type", None) == "mps"
        and str(getattr(value, "dtype", "")) == "torch.bfloat16"
        and bool(getattr(value, "is_contiguous", lambda: False)())
    )


class Qwen3DeferredKvCommitOp(BaseFusedOp):
    """Commit 28 stacked layer outputs into Torch-owned NHD KV pools."""

    op = "kvcache.qwen3_deferred_kv_commit"
    priority = (KernelBackend.METAL_JIT, KernelBackend.TORCH)
    capabilities = {KernelBackend.METAL_JIT: _MPS}
    format_signature = FormatSignature(
        supported_dtypes=("bfloat16",),
        in_place=True,
        description="Qwen3 stacked deferred K/V rows into 28 NHD layer pools",
    )
    descriptions = {
        KernelBackend.TORCH: "per-layer Torch index_copy_ correctness reference",
        KernelBackend.METAL_JIT: "two-launch Torch-stream Metal KV commit",
    }

    def backend_eligible(self, backend, *args, **kwargs) -> bool:
        if not super().backend_eligible(backend, *args, **kwargs):
            return False
        if backend is not KernelBackend.METAL_JIT:
            return True
        if len(args) < 5:
            return False
        new_k, new_v, slots, k_pools, v_pools = args[:5]
        expected_tail = (
            QWEN3_06B_METAL_SPEC.num_kv_heads,
            QWEN3_06B_METAL_SPEC.head_dim,
        )
        if (
            not _is_mps_bf16_contiguous(new_k)
            or not _is_mps_bf16_contiguous(new_v)
            or getattr(new_k, "ndim", 0) != 4
            or tuple(new_k.shape[:1]) != (_NUM_LAYERS,)
            or tuple(new_k.shape[2:]) != expected_tail
            or tuple(new_v.shape) != tuple(new_k.shape)
        ):
            return False
        num_rows = int(new_k.shape[1])
        if (
            getattr(getattr(slots, "device", None), "type", None) != "mps"
            or str(getattr(slots, "dtype", "")) != "torch.int64"
            or not bool(getattr(slots, "is_contiguous", lambda: False)())
            or tuple(getattr(slots, "shape", ())) != (num_rows,)
            or not isinstance(k_pools, (list, tuple))
            or not isinstance(v_pools, (list, tuple))
            or len(k_pools) != _NUM_LAYERS
            or len(v_pools) != _NUM_LAYERS
        ):
            return False
        pool_slots = None
        for k_pool, v_pool in zip(k_pools, v_pools):
            if (
                not _is_mps_bf16_contiguous(k_pool)
                or not _is_mps_bf16_contiguous(v_pool)
                or getattr(k_pool, "ndim", 0) != 3
                or tuple(k_pool.shape[1:]) != expected_tail
                or tuple(v_pool.shape) != tuple(k_pool.shape)
            ):
                return False
            if pool_slots is None:
                pool_slots = int(k_pool.shape[0])
            elif int(k_pool.shape[0]) != pool_slots:
                return False
        return True

    def forward_native(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        slots: torch.Tensor,
        k_pools: list[torch.Tensor] | tuple[torch.Tensor, ...],
        v_pools: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> None:
        if new_k.ndim != 4 or new_k.shape[0] != _NUM_LAYERS:
            raise RuntimeError(
                "Qwen3 deferred KV commit expects new_k with 28 layer rows"
            )
        if tuple(new_v.shape) != tuple(new_k.shape):
            raise RuntimeError("Qwen3 deferred K/V tensors must have matching shapes")
        if len(k_pools) != _NUM_LAYERS or len(v_pools) != _NUM_LAYERS:
            raise RuntimeError("Qwen3 deferred KV commit requires 28 K/V pools")
        for layer, (k_pool, v_pool) in enumerate(zip(k_pools, v_pools)):
            k_pool.index_copy_(0, slots, new_k[layer])
            v_pool.index_copy_(0, slots, new_v[layer])

    def forward_metal_jit(self, *args, **kwargs) -> None:
        from sglang.kernels.ops.kvcache._qwen3_deferred_kv_commit_metal_jit import (
            qwen3_commit_deferred_kv as metal_jit,
        )

        metal_jit(*args, **kwargs)


_DEFERRED_KV_COMMIT = register_fused_op(
    Qwen3DeferredKvCommitOp(), __name__, "_DEFERRED_KV_COMMIT"
)


def qwen3_commit_deferred_kv(*args, **kwargs) -> None:
    """Commit stacked Qwen3 K/V rows through the selected implementation."""
    _DEFERRED_KV_COMMIT(*args, **kwargs)


def set_qwen3_kv_commit_priority(priority: tuple[KernelBackend, ...]) -> None:
    """Install the process-local deferred-commit provider order."""
    _DEFERRED_KV_COMMIT.set_priority(priority)


def warmup_qwen3_kv_commit(*, pool_slots: int) -> None:
    """Compile and resolve the fixed two-launch Metal implementation."""
    from sglang.kernels.ops.kvcache._qwen3_deferred_kv_commit_metal_jit import (
        warmup_qwen3_kv_commit as warmup,
    )

    warmup(pool_slots=pool_slots)


__all__ = [
    "Qwen3DeferredKvCommitOp",
    "qwen3_commit_deferred_kv",
    "set_qwen3_kv_commit_priority",
    "warmup_qwen3_kv_commit",
]
