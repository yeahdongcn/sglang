"""Qwen3 attention primitives for the Torch MPS execution path.

The public callables in this module are semantic operators.  Their pure-Torch
implementations are correctness references; the Metal implementations share a
fixed Qwen3-0.6B shader and execute on Torch's MPS stream, either from the
packaged AOT ``metallib`` or through ``torch.mps.compile_shader`` as fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sglang.kernels.fused_op import BaseFusedOp, register_fused_op
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
)

if TYPE_CHECKING:
    import torch

_MPS = frozenset({CapabilityRequirement.MPS})
_PRIORITY = (KernelBackend.METAL_AOT, KernelBackend.METAL_JIT, KernelBackend.TORCH)
_METAL_BACKENDS = frozenset({KernelBackend.METAL_AOT, KernelBackend.METAL_JIT})


@dataclass(frozen=True)
class Qwen3MetalKernelSpec:
    """Static shape contract shared by the Torch reference and Metal paths."""

    head_dim: int
    num_q_heads: int
    num_kv_heads: int

    @property
    def qkv_width(self) -> int:
        return (self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim

    @property
    def attention_scale(self) -> float:
        return self.head_dim**-0.5


QWEN3_06B_METAL_SPEC = Qwen3MetalKernelSpec(
    head_dim=128,
    num_q_heads=16,
    num_kv_heads=8,
)


def is_qwen3_attention_scale_supported(
    scale: float,
    spec: Qwen3MetalKernelSpec = QWEN3_06B_METAL_SPEC,
) -> bool:
    """Whether ``scale`` exactly matches the constant compiled into Metal.

    The specialized shader does not consume a runtime scale argument. A
    tolerance here would select a kernel with different semantics and fail
    only later in its implementation adapter.
    """
    try:
        return float(scale) == spec.attention_scale
    except (TypeError, ValueError):
        return False


def is_qwen3_rms_epsilon_supported(epsilon: float) -> bool:
    """Whether one runtime RMS epsilon is valid for the fused shader."""
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError):
        return False
    return math.isfinite(epsilon) and epsilon > 0


def is_qwen3_metal_aot_available() -> bool:
    """Whether the optional Torch-loadable Qwen3 metallib is installed."""
    from sglang.kernels.ops.attention._qwen3_metal_aot import (
        is_qwen3_metal_aot_available as is_available,
    )

    return is_available()


def preferred_qwen3_mps_backend() -> KernelBackend:
    """Resolve the static AOT-first backend used by the Qwen3 MPS plan."""
    if is_qwen3_metal_aot_available():
        return KernelBackend.METAL_AOT
    return KernelBackend.METAL_JIT


def _is_mps_bf16_contiguous(value) -> bool:
    return (
        getattr(getattr(value, "device", None), "type", None) == "mps"
        and str(getattr(value, "dtype", "")) == "torch.bfloat16"
        and bool(getattr(value, "is_contiguous", lambda: False)())
    )


def _rms_norm(value: torch.Tensor, weight: torch.Tensor, epsilon: float):
    import torch

    normalized = value.float() * torch.rsqrt(
        value.float().square().mean(dim=-1, keepdim=True) + epsilon
    )
    return (normalized * weight.float()).to(value.dtype)


def _neox_rope(value: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor):
    import torch

    cosine, sine = cos_sin[positions.long()].chunk(2, dim=-1)
    cosine = cosine[:, None, :]
    sine = sine[:, None, :]
    first, second = value.chunk(2, dim=-1)
    return torch.cat(
        (first * cosine - second * sine, second * cosine + first * sine),
        dim=-1,
    )


class Qwen3QKNormRopeStoreOp(BaseFusedOp):
    """Q/K RMSNorm + NeoX RoPE + in-place NHD KV-pool store."""

    op = "attention.qwen3_qknorm_rope_store"
    priority = _PRIORITY
    capabilities = {
        KernelBackend.METAL_AOT: _MPS,
        KernelBackend.METAL_JIT: _MPS,
    }
    format_signature = FormatSignature(
        supported_dtypes=("bfloat16",),
        in_place=True,
        description="Qwen3 QK norm/rope and standard NHD KV-pool write",
    )
    descriptions = {
        KernelBackend.TORCH: "staged Torch correctness reference",
        KernelBackend.METAL_AOT: "Torch-stream precompiled Metal kernel",
        KernelBackend.METAL_JIT: "Torch-stream Metal JIT kernel",
    }

    def backend_eligible(self, backend, *args, **kwargs) -> bool:
        if not super().backend_eligible(backend, *args, **kwargs):
            return False
        if backend not in _METAL_BACKENDS:
            return True
        spec = kwargs.get("spec") or QWEN3_06B_METAL_SPEC
        if spec != QWEN3_06B_METAL_SPEC:
            return False
        if backend is KernelBackend.METAL_AOT and not is_qwen3_metal_aot_available():
            return False
        if len(args) < 9:
            return False
        epsilon_is_valid = is_qwen3_rms_epsilon_supported(kwargs.get("epsilon"))
        qkv, q_weight, k_weight, cos_sin, positions, slots, q_out, k_pool, v_pool = (
            args[:9]
        )
        num_tokens = qkv.shape[0] if getattr(qkv, "ndim", 0) == 2 else -1
        pool_tail = (spec.num_kv_heads, spec.head_dim)
        return (
            epsilon_is_valid
            and all(
                _is_mps_bf16_contiguous(value)
                for value in (
                    qkv,
                    q_weight,
                    k_weight,
                    cos_sin,
                    q_out,
                    k_pool,
                    v_pool,
                )
            )
            and tuple(qkv.shape) == (num_tokens, spec.qkv_width)
            and tuple(q_weight.shape) == (spec.head_dim,)
            and tuple(k_weight.shape) == (spec.head_dim,)
            and getattr(cos_sin, "ndim", 0) == 2
            and cos_sin.shape[1] == spec.head_dim
            and tuple(q_out.shape) == (num_tokens, spec.num_q_heads, spec.head_dim)
            and getattr(k_pool, "ndim", 0) == 3
            and tuple(k_pool.shape[1:]) == pool_tail
            and tuple(v_pool.shape) == tuple(k_pool.shape)
            and getattr(getattr(positions, "device", None), "type", None) == "mps"
            and getattr(getattr(slots, "device", None), "type", None) == "mps"
            and str(getattr(positions, "dtype", "")) == "torch.int64"
            and str(getattr(slots, "dtype", "")) == "torch.int64"
            and bool(positions.is_contiguous())
            and bool(slots.is_contiguous())
            and tuple(positions.shape) == (num_tokens,)
            and tuple(slots.shape) == (num_tokens,)
        )

    def forward_native(
        self,
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        q_out: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        *,
        epsilon: float,
        spec=None,
    ) -> None:
        if spec is None:
            spec = QWEN3_06B_METAL_SPEC
        num_tokens = qkv.shape[0]
        q, k, v = qkv.split(
            (
                spec.num_q_heads * spec.head_dim,
                spec.num_kv_heads * spec.head_dim,
                spec.num_kv_heads * spec.head_dim,
            ),
            dim=-1,
        )
        q = _rms_norm(
            q.reshape(num_tokens, spec.num_q_heads, spec.head_dim),
            q_weight,
            epsilon,
        )
        k = _rms_norm(
            k.reshape(num_tokens, spec.num_kv_heads, spec.head_dim),
            k_weight,
            epsilon,
        )
        q_out.copy_(_neox_rope(q, cos_sin_cache, positions))
        k_pool[slots] = _neox_rope(k, cos_sin_cache, positions)
        v_pool[slots] = v.reshape(num_tokens, spec.num_kv_heads, spec.head_dim)

    def forward_metal_jit(self, *args, **kwargs) -> None:
        from sglang.kernels.ops.attention._qwen3_metal_jit import (
            qwen3_fused_qknorm_rope_store as metal_jit,
        )

        metal_jit(*args, **kwargs)

    def forward_metal_aot(self, *args, **kwargs) -> None:
        from sglang.kernels.ops.attention._qwen3_metal_aot import (
            qwen3_fused_qknorm_rope_store as metal_aot,
        )

        metal_aot(*args, **kwargs)


class Qwen3RadixDecodeOp(BaseFusedOp):
    """Decode attention over the standard Radix request/slot tables."""

    op = "attention.qwen3_radix_decode"
    priority = _PRIORITY
    capabilities = {
        KernelBackend.METAL_AOT: _MPS,
        KernelBackend.METAL_JIT: _MPS,
    }
    format_signature = FormatSignature(
        supported_dtypes=("bfloat16",),
        in_place=True,
        description="Qwen3 GQA decode over NHD KV pool and Radix slot table",
    )
    descriptions = {
        KernelBackend.TORCH: "gather plus Torch SDPA correctness reference",
        KernelBackend.METAL_AOT: "direct Radix-table precompiled Metal kernel",
        KernelBackend.METAL_JIT: "direct Radix-table Torch-stream Metal JIT kernel",
    }

    def backend_eligible(self, backend, *args, **kwargs) -> bool:
        if not super().backend_eligible(backend, *args, **kwargs):
            return False
        if backend not in _METAL_BACKENDS:
            return True
        spec = kwargs.get("spec") or QWEN3_06B_METAL_SPEC
        if spec != QWEN3_06B_METAL_SPEC:
            return False
        if backend is KernelBackend.METAL_AOT and not is_qwen3_metal_aot_available():
            return False
        if len(args) < 7:
            return False
        q, k_pool, v_pool, req_to_token, req_pool_indices, seq_lens, out = args[:7]
        batch_size = q.shape[0] if getattr(q, "ndim", 0) == 3 else -1
        pool_tail = (spec.num_kv_heads, spec.head_dim)
        return (
            all(_is_mps_bf16_contiguous(value) for value in (q, k_pool, v_pool, out))
            and tuple(q.shape) == (batch_size, spec.num_q_heads, spec.head_dim)
            and tuple(out.shape) == tuple(q.shape)
            and getattr(k_pool, "ndim", 0) == 3
            and tuple(k_pool.shape[1:]) == pool_tail
            and tuple(v_pool.shape) == tuple(k_pool.shape)
            and getattr(req_to_token, "ndim", 0) == 2
            and getattr(getattr(req_to_token, "device", None), "type", None) == "mps"
            and getattr(getattr(req_pool_indices, "device", None), "type", None)
            == "mps"
            and getattr(getattr(seq_lens, "device", None), "type", None) == "mps"
            and str(getattr(req_to_token, "dtype", "")) == "torch.int32"
            and str(getattr(req_pool_indices, "dtype", "")) == "torch.int64"
            and str(getattr(seq_lens, "dtype", "")) == "torch.int64"
            and bool(req_to_token.is_contiguous())
            and bool(req_pool_indices.is_contiguous())
            and bool(seq_lens.is_contiguous())
            and tuple(req_pool_indices.shape) == (batch_size,)
            and tuple(seq_lens.shape) == (batch_size,)
            and is_qwen3_attention_scale_supported(
                kwargs.get("scale", spec.attention_scale), spec
            )
        )

    def forward_native(
        self,
        q: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out: torch.Tensor,
        *,
        scale: float,
        spec=None,
    ) -> None:
        import torch
        import torch.nn.functional as F

        if q.shape[0] == 0:
            return
        outputs = []
        enable_gqa = q.shape[1] != k_pool.shape[1]
        for batch_index in range(q.shape[0]):
            sequence_length = int(seq_lens[batch_index].item())
            request_index = int(req_pool_indices[batch_index].item())
            slots = req_to_token[request_index, :sequence_length].long()
            key = k_pool[slots].movedim(0, 1)
            value = v_pool[slots].movedim(0, 1)
            outputs.append(
                F.scaled_dot_product_attention(
                    q[batch_index][:, None, :].unsqueeze(0),
                    key.unsqueeze(0),
                    value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scale,
                    is_causal=False,
                )
                .squeeze(0)
                .squeeze(1)
            )
        out.copy_(torch.stack(outputs))

    def forward_metal_jit(self, *args, **kwargs) -> None:
        from sglang.kernels.ops.attention._qwen3_metal_jit import (
            qwen3_radix_decode as metal_jit,
        )

        metal_jit(*args, **kwargs)

    def forward_metal_aot(self, *args, **kwargs) -> None:
        from sglang.kernels.ops.attention._qwen3_metal_aot import (
            qwen3_radix_decode as metal_aot,
        )

        metal_aot(*args, **kwargs)


_QKNORM_ROPE_STORE = register_fused_op(
    Qwen3QKNormRopeStoreOp(), __name__, "_QKNORM_ROPE_STORE"
)
_RADIX_DECODE = register_fused_op(Qwen3RadixDecodeOp(), __name__, "_RADIX_DECODE")


def qwen3_qknorm_rope_store(*args, **kwargs) -> None:
    _QKNORM_ROPE_STORE(*args, **kwargs)


def qwen3_radix_decode(*args, **kwargs) -> None:
    _RADIX_DECODE(*args, **kwargs)


def is_qwen3_qknorm_rope_store_backend_eligible(
    backend: KernelBackend, *args, **kwargs
) -> bool:
    """Check one pinned QKV provider against this call's dynamic contract."""
    return _QKNORM_ROPE_STORE.backend_eligible(backend, *args, **kwargs)


def is_qwen3_radix_decode_backend_eligible(
    backend: KernelBackend, *args, **kwargs
) -> bool:
    """Check one pinned decode provider against this call's dynamic contract."""
    return _RADIX_DECODE.backend_eligible(backend, *args, **kwargs)


def set_qwen3_mps_priority(
    qknorm_rope_store: tuple[KernelBackend, ...],
    radix_decode: tuple[KernelBackend, ...],
) -> None:
    """Install per-semantic-op priorities for the process-local MPS plan.

    The semantic operator objects are singletons because they are also used by
    direct model/unit-test call sites.  Their priority is configured once at
    the platform boundary; model providers then pin the selected backend and
    do not pay selector work in the serving hot path.
    """
    _QKNORM_ROPE_STORE.set_priority(qknorm_rope_store)
    _RADIX_DECODE.set_priority(radix_decode)


def warmup_qwen3_mps_kernels(
    qknorm_backend: KernelBackend | None = None,
    radix_backend: KernelBackend | None = None,
    *,
    backend: KernelBackend | None = None,
) -> tuple[KernelBackend, KernelBackend]:
    """Warm independently selected Qwen3 Metal pipelines.

    ``backend=`` preserves the old all-or-one diagnostic API.  New model plans
    pass ``qknorm_backend`` and ``radix_backend`` separately, so selecting one
    custom attention operation never compiles an unrelated pipeline.
    """
    if backend is not None:
        if qknorm_backend is not None or radix_backend is not None:
            raise TypeError("backend cannot be combined with per-op backends")
        qknorm_backend = radix_backend = backend
    if qknorm_backend is None:
        qknorm_backend = preferred_qwen3_mps_backend()
    if radix_backend is None:
        radix_backend = preferred_qwen3_mps_backend()

    def warm_one(selected: KernelBackend, *, qknorm: bool) -> None:
        if selected is KernelBackend.TORCH:
            return
        if selected is KernelBackend.METAL_AOT:
            from sglang.kernels.ops.attention._qwen3_metal_aot import (
                warmup_qwen3_metal_aot_kernels,
            )

            warmup_qwen3_metal_aot_kernels(
                qknorm_rope_store=qknorm,
                radix_decode=not qknorm,
            )
            return
        if selected is KernelBackend.METAL_JIT:
            from sglang.kernels.ops.attention._qwen3_metal_jit import (
                warmup_qwen3_metal_kernels,
            )

            warmup_qwen3_metal_kernels(
                qknorm_rope_store=qknorm,
                radix_decode=not qknorm,
            )
            return
        raise RuntimeError(f"unsupported Qwen3 MPS kernel backend {selected.value!r}")

    # If both operations use the same provider, invoke its combined adapter so
    # the shader/library is loaded or compiled exactly once.
    if qknorm_backend is radix_backend:
        if qknorm_backend is KernelBackend.METAL_AOT:
            from sglang.kernels.ops.attention._qwen3_metal_aot import (
                warmup_qwen3_metal_aot_kernels,
            )

            warmup_qwen3_metal_aot_kernels()
        elif qknorm_backend is KernelBackend.METAL_JIT:
            from sglang.kernels.ops.attention._qwen3_metal_jit import (
                warmup_qwen3_metal_kernels,
            )

            warmup_qwen3_metal_kernels()
        elif qknorm_backend is not KernelBackend.TORCH:
            raise RuntimeError(
                f"unsupported Qwen3 MPS kernel backend {qknorm_backend.value!r}"
            )
    else:
        warm_one(qknorm_backend, qknorm=True)
        warm_one(radix_backend, qknorm=False)
    return qknorm_backend, radix_backend


__all__ = [
    "QWEN3_06B_METAL_SPEC",
    "Qwen3MetalKernelSpec",
    "Qwen3QKNormRopeStoreOp",
    "Qwen3RadixDecodeOp",
    "is_qwen3_attention_scale_supported",
    "is_qwen3_qknorm_rope_store_backend_eligible",
    "is_qwen3_radix_decode_backend_eligible",
    "is_qwen3_metal_aot_available",
    "is_qwen3_rms_epsilon_supported",
    "preferred_qwen3_mps_backend",
    "qwen3_qknorm_rope_store",
    "qwen3_radix_decode",
    "set_qwen3_mps_priority",
    "warmup_qwen3_mps_kernels",
]
