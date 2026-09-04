# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os

import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.magi2_mhc_kernel import (
    mhc_mix_output,
    mhc_sinkhorn,
)


_MHC_BF16_PROJECT_ENV = "SGLANG_MAGI2_MHC_BF16_PROJECT"
_MHC_BF16_NORM_ENV = "SGLANG_MAGI2_MHC_BF16_NORM"


def _is_cuda_alike_tensor(tensor: torch.Tensor | torch.device) -> bool:
    """MUSA tensors use CUDA-compatible kernels but ``Tensor.is_cuda`` is false."""
    device = tensor.device if isinstance(tensor, torch.Tensor) else tensor
    return device.type in {"cuda", "musa", "privateuseone"}


def _is_musa_tensor(tensor: torch.Tensor | torch.device) -> bool:
    """Identify MUSA without relying on ``torch.cuda.is_available()``."""
    device = tensor.device if isinstance(tensor, torch.Tensor) else tensor
    return device.type in {"musa", "privateuseone"}


def _bf16_project_enabled(tensor: torch.Tensor) -> bool:
    """Return whether the experimental BF16 mHC projection is requested.

    The gate is intentionally explicit and shape-independent.  The model-level
    caller still checks the four-stream MAGI contract before dispatching; all
    other callers retain the original fp32 projection.
    """
    return (
        os.environ.get(_MHC_BF16_PROJECT_ENV) == "1"
        and _is_musa_tensor(tensor)
        and tensor.dtype in (torch.float32, torch.bfloat16)
    )


def mhc_bf16_norm_enabled(tensor: torch.Tensor) -> bool:
    """Return whether MHC RMSNorm may emit BF16 for the opt-in fast path."""
    return (
        os.environ.get(_MHC_BF16_NORM_ENV) == "1"
        and _bf16_project_enabled(tensor)
    )


def sinkhorn_knopp(h: torch.Tensor, *, num_iters: int, eps: float) -> torch.Tensor:
    if h.device.type in {"cuda", "musa", "privateuseone"}:
        return mhc_sinkhorn(h, num_iters=num_iters, eps=eps)
    m = torch.exp(h - h.amax(dim=(-2, -1), keepdim=True))
    for _ in range(num_iters):
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
    return m


class Magi2MHC(nn.Module):
    """``phi_fused`` projects to ``2 * n + n * n``: pre-mix, post-mix, and a flattened stream-to-stream matrix."""

    def __init__(
        self,
        *,
        num_stream: int,
        hidden_size: int,
        alpha_init: float = 0.01,
        sinkhorn_iters: int = 20,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.num_stream = num_stream
        self.hidden_size = hidden_size
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        # Scaled against the full concatenated stream, not hidden_size.
        self.matmul_scale = 1.0 / math.sqrt(num_stream * hidden_size)

        n = num_stream
        self.phi_fused = nn.Parameter(
            torch.zeros(n * hidden_size, 2 * n + n * n, dtype=torch.float32)
        )
        self.alpha_pre = nn.Parameter(torch.full((1,), alpha_init))
        self.alpha_post = nn.Parameter(torch.full((1,), alpha_init))
        self.alpha_res = nn.Parameter(torch.full((1,), alpha_init))
        self.bias_pre = nn.Parameter(torch.zeros(n))
        self.bias_post = nn.Parameter(torch.zeros(n))
        self.bias_res = nn.Parameter(torch.zeros(n, n))
        # The BF16 copy is deliberately a non-state attribute.  It is derived
        # from the checkpoint parameter after device placement and must not
        # alter the loader/FSDP parameter namespace.  MAGI inference freezes
        # ``phi_fused`` after loading, so one conversion per module is enough.
        self._phi_fused_bf16: torch.Tensor | None = None
        self._phi_fused_bf16_source: tuple[int, int] | None = None

    def _bf16_phi_fused(self) -> torch.Tensor:
        cached = self._phi_fused_bf16
        try:
            version = self.phi_fused._version
        except RuntimeError:
            # Inference tensors do not expose a version counter.  Their
            # storage is immutable for this use, so the data pointer is still
            # a sufficient cache identity.
            version = -1
        source = (self.phi_fused.data_ptr(), version)
        if (
            cached is None
            or cached.device != self.phi_fused.device
            or cached.shape != self.phi_fused.shape
            or self._phi_fused_bf16_source != source
        ):
            cached = self.phi_fused.detach().to(dtype=torch.bfloat16).contiguous()
            self._phi_fused_bf16 = cached
            self._phi_fused_bf16_source = source
        return cached

    def project(self, streams_flat: torch.Tensor) -> tuple[torch.Tensor, ...]:
        n = self.num_stream
        if (
            _bf16_project_enabled(streams_flat)
            and n == 4
            and streams_flat.ndim == 2
            and streams_flat.shape[-1] == n * self.hidden_size
            and self.phi_fused.dtype == torch.float32
        ):
            # S5000's BF16 GEMM path is substantially faster for the tiny
            # N=24 projection.  Return fp32 logits so the downstream sigmoid
            # and Sinkhorn contracts remain unchanged.  Conversion of the
            # 24-column result is tiny compared with the fp32 GEMM itself.
            fused = torch.matmul(
                streams_flat.to(dtype=torch.bfloat16), self._bf16_phi_fused()
            ).float()
        else:
            fused = torch.matmul(streams_flat.float(), self.phi_fused)
        h_pre, h_post, h_res = torch.split(fused, [n, n, n * n], dim=-1)
        return h_pre, h_post, h_res.view(-1, n, n)

    def mix_input(self, streams: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.alpha_pre * self.matmul_scale * h_pre + self.bias_pre)
        # Left in torch: inductor fuses this into its consumer, where a custom op
        # is opaque and cannot be fused (measured 5s/clip slower as a kernel).
        return torch.einsum("tn,tnc->tc", gate.to(streams.dtype), streams)

    def mix_output(
        self,
        streams: torch.Tensor,
        block_out: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
    ) -> torch.Tensor:
        # Scaled by 2 so the gate spans (0, 2) and can amplify, not just damp.
        post = 2.0 * torch.sigmoid(
            self.alpha_post * self.matmul_scale * h_post + self.bias_post
        )
        # Left in torch: a fused kernel measured 0.2% end to end, inside the noise.
        res = sinkhorn_knopp(
            self.alpha_res * self.matmul_scale * h_res.float() + self.bias_res,
            num_iters=self.sinkhorn_iters,
            eps=self.eps,
        )
        if _is_cuda_alike_tensor(streams):
            return mhc_mix_output(streams, block_out, post, res)

        mixed = torch.einsum("tij,tjc->tic", res.to(streams.dtype), streams)
        written = torch.einsum("tn,tc->tnc", post.to(streams.dtype), block_out)
        return mixed + written
