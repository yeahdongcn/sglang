# SPDX-License-Identifier: Apache-2.0

import torch

import sglang.multimodal_gen.runtime.models.dits.magi2_mhc as mhc_module
from sglang.multimodal_gen.runtime.models.dits.magi2_mhc import Magi2MHC


def _flatten_projected(parts: tuple[torch.Tensor, ...]) -> torch.Tensor:
    h_pre, h_post, h_res = parts
    return torch.cat((h_pre, h_post, h_res.reshape(h_res.shape[0], -1)), dim=-1)


def test_bf16_projection_is_opt_in_and_cpu_falls_back(monkeypatch):
    monkeypatch.setenv("SGLANG_MAGI2_MHC_BF16_PROJECT", "1")
    mhc = Magi2MHC(num_stream=4, hidden_size=8)
    torch.manual_seed(11)
    x = torch.randn(5, 32)

    expected = torch.matmul(x, mhc.phi_fused)
    actual = _flatten_projected(mhc.project(x))

    # CPU keeps the reference fp32 path even when the environment variable is
    # present; this is the fallback contract for non-MUSA callers.
    torch.testing.assert_close(actual, expected)
    assert mhc._phi_fused_bf16 is None


def test_bf16_projection_cache_and_error_budget(monkeypatch):
    monkeypatch.setenv("SGLANG_MAGI2_MHC_BF16_PROJECT", "1")
    # Exercise the exact gated implementation without requiring a MUSA worker.
    monkeypatch.setattr(mhc_module, "_is_musa_tensor", lambda _: True)

    mhc = Magi2MHC(num_stream=4, hidden_size=16)
    torch.manual_seed(17)
    with torch.no_grad():
        mhc.phi_fused.normal_(mean=0.0, std=0.08)
    x = torch.randn(13, 64)

    expected = torch.matmul(x, mhc.phi_fused)
    first = _flatten_projected(mhc.project(x))
    cached = mhc._phi_fused_bf16
    second = _flatten_projected(mhc.project(x))

    assert cached is not None
    assert mhc._phi_fused_bf16 is cached
    torch.testing.assert_close(first, expected, rtol=0.02, atol=0.12)
    torch.testing.assert_close(second, first, rtol=0, atol=0)

    # A post-load/training update must invalidate the derived cache rather than
    # silently using stale BF16 weights.
    with torch.no_grad():
        mhc.phi_fused.add_(0.01)
    mhc.project(x)
    assert mhc._phi_fused_bf16 is not cached


def test_bf16_norm_gate_requires_bf16_project(monkeypatch):
    x = torch.randn(2, 32)
    monkeypatch.setattr(mhc_module, "_is_musa_tensor", lambda _: True)
    monkeypatch.setenv("SGLANG_MAGI2_MHC_BF16_NORM", "1")

    monkeypatch.delenv("SGLANG_MAGI2_MHC_BF16_PROJECT", raising=False)
    assert not mhc_module.mhc_bf16_norm_enabled(x)

    monkeypatch.setenv("SGLANG_MAGI2_MHC_BF16_PROJECT", "1")
    assert mhc_module.mhc_bf16_norm_enabled(x)


def test_fast_mix_gate_is_opt_in_and_preserves_cpu_fallback(monkeypatch):
    mhc = Magi2MHC(num_stream=4, hidden_size=8)
    torch.manual_seed(29)
    streams = torch.randn(7, 4, 8, dtype=torch.bfloat16)
    h_pre = torch.randn(7, 4)
    monkeypatch.setenv("SGLANG_MAGI2_MHC_FAST_MIX", "1")
    expected = mhc.mix_input(streams, h_pre)
    assert not mhc_module.mhc_fast_mix_enabled(streams)

    monkeypatch.setattr(mhc_module, "_is_musa_tensor", lambda _: True)
    fast = mhc.mix_input(streams, h_pre)
    torch.testing.assert_close(fast, expected, rtol=0, atol=1e-2)
