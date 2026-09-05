# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.magi2_common import (
    swiglu7_interleaved,
)


def test_fast_swiglu7_keeps_cpu_fallback(monkeypatch):
    torch.manual_seed(17)
    x = torch.randn(9, 32, dtype=torch.bfloat16)
    monkeypatch.delenv("SGLANG_MAGI2_FAST_SWIGLU7", raising=False)
    expected = swiglu7_interleaved(x)
    monkeypatch.setenv("SGLANG_MAGI2_FAST_SWIGLU7", "1")
    torch.testing.assert_close(swiglu7_interleaved(x), expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not (hasattr(torch.version, "musa") and torch.version.musa is not None),
    reason="requires MUSA",
)
def test_fast_swiglu7_matches_reference_on_musa(monkeypatch):
    visible = os.environ.get("MUSA_VISIBLE_DEVICES")
    if visible:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == visible
    torch.musa.set_device(0)
    torch.manual_seed(23)
    x = torch.randn(33, 2560, device="musa", dtype=torch.bfloat16)
    monkeypatch.delenv("SGLANG_MAGI2_FAST_SWIGLU7", raising=False)
    expected = swiglu7_interleaved(x)
    monkeypatch.setenv("SGLANG_MAGI2_FAST_SWIGLU7", "1")
    actual = swiglu7_interleaved(x)
    torch.musa.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
