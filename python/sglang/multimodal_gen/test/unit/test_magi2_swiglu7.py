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


def test_fast_swiglu7_empty_and_odd_inputs_fall_back_safely(monkeypatch):
    monkeypatch.setenv("SGLANG_MAGI2_FAST_SWIGLU7", "1")
    empty = torch.empty(0, 2560, dtype=torch.bfloat16)
    torch.testing.assert_close(
        swiglu7_interleaved(empty), torch.empty(0, 1280, dtype=torch.bfloat16)
    )
    with pytest.raises(RuntimeError):
        swiglu7_interleaved(torch.empty(3, 7, dtype=torch.bfloat16))


def test_fast_swiglu7_boundary_values_use_reference_contract(monkeypatch):
    # CPU exercises the exact fallback contract and documents the values used
    # by the MUSA comparison below.
    values = torch.tensor(
        [-float("inf"), -8.0, -7.0, -0.0, 0.0, 7.0, 8.0, float("inf"), float("nan")],
        dtype=torch.float32,
    )
    x = torch.stack((values, values.flip(0)), dim=-1).reshape(1, -1)
    monkeypatch.delenv("SGLANG_MAGI2_FAST_SWIGLU7", raising=False)
    output = swiglu7_interleaved(x)
    assert torch.isnan(output).any()
    assert torch.isinf(output).logical_not().all()


@pytest.mark.parametrize("shape", [(33, 2560), (3, 16384), (3, 21840), (2, 514)])
@pytest.mark.skipif(
    not (hasattr(torch.version, "musa") and torch.version.musa is not None),
    reason="requires MUSA",
)
def test_fast_swiglu7_matches_reference_on_musa(monkeypatch, shape):
    visible = os.environ.get("MUSA_VISIBLE_DEVICES")
    if visible:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == visible
    torch.musa.set_device(0)
    torch.manual_seed(23)
    x = torch.randn(*shape, device="musa", dtype=torch.bfloat16)
    monkeypatch.delenv("SGLANG_MAGI2_FAST_SWIGLU7", raising=False)
    expected = swiglu7_interleaved(x)
    monkeypatch.setenv("SGLANG_MAGI2_FAST_SWIGLU7", "1")
    actual = swiglu7_interleaved(x)
    torch.musa.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not (hasattr(torch.version, "musa") and torch.version.musa is not None),
    reason="requires MUSA",
)
def test_fast_swiglu7_matches_boundary_nan_mask_on_musa(monkeypatch):
    torch.musa.set_device(0)
    values = torch.tensor(
        [-float("inf"), -8.0, -7.0, -0.0, 0.0, 7.0, 8.0, float("inf"), float("nan")],
        device="musa",
        dtype=torch.bfloat16,
    )
    x = torch.stack((values, values.flip(0)), dim=-1).reshape(1, -1)
    monkeypatch.delenv("SGLANG_MAGI2_FAST_SWIGLU7", raising=False)
    expected = swiglu7_interleaved(x)
    monkeypatch.setenv("SGLANG_MAGI2_FAST_SWIGLU7", "1")
    actual = swiglu7_interleaved(x)
    torch.musa.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
