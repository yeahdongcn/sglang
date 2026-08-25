"""Packed xgrammar bitmask coverage for CPU and Apple MPS."""

import unittest
from unittest import mock

import pytest
import torch

from sglang.srt.constrained.torch_ops.token_filter_torch_ops import (
    apply_token_bitmask_inplace_torch,
)
from sglang.srt.constrained.xgrammar_backend import (
    XGrammarGrammarBackend,
    _allocate_token_bitmask,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")


def _packed_rows(device: str) -> torch.Tensor:
    # Row 0 allows 0, 2, 31, 32, and 39. Row 1 allows 1 and 33.
    return torch.tensor(
        [
            [-(1 << 31) + 5, 129],
            [2, 2],
        ],
        dtype=torch.int32,
        device=device,
    )


def _expected() -> torch.Tensor:
    expected = torch.full((2, 40), float("-inf"))
    expected[0, [0, 2, 31, 32, 39]] = torch.tensor([0.0, 2.0, 31.0, 32.0, 39.0])
    expected[1, [1, 33]] = torch.tensor([41.0, 73.0])
    return expected


def test_torch_packed_mask_matches_bit_contract_on_cpu():
    logits = torch.arange(80, dtype=torch.float32).reshape(2, 40)

    apply_token_bitmask_inplace_torch(logits, _packed_rows("cpu"))

    torch.testing.assert_close(logits, _expected())


def test_one_dimensional_mask_and_logits_are_supported():
    logits = torch.arange(5, dtype=torch.float32)
    bitmask = torch.tensor([0b10101], dtype=torch.int32)

    apply_token_bitmask_inplace_torch(logits, bitmask)

    torch.testing.assert_close(
        logits,
        torch.tensor([0.0, float("-inf"), 2.0, float("-inf"), 4.0]),
    )


def test_mps_target_uses_pageable_host_mask():
    with mock.patch(
        "sglang.srt.constrained.xgrammar_backend.current_platform."
        "is_pin_memory_available",
        return_value=False,
    ) as pin_available:
        mask = _allocate_token_bitmask(40, 2, "mps")

    pin_available.assert_called_once_with()
    assert mask.device.type == "cpu"
    assert not mask.is_pinned()


def test_host_mask_uses_platform_pinning_even_for_cpu_staging_device():
    sentinel = object()
    with (
        mock.patch(
            "sglang.srt.constrained.xgrammar_backend.current_platform."
            "is_pin_memory_available",
            return_value=True,
        ),
        mock.patch(
            "sglang.srt.constrained.xgrammar_backend.torch.full",
            return_value=sentinel,
        ) as full,
    ):
        result = _allocate_token_bitmask(40, 2, "cpu")

    assert result is sentinel
    assert full.call_args.kwargs["pin_memory"] is True


@unittest.skipUnless(torch.backends.mps.is_available(), "requires Apple MPS")
def test_xgrammar_backend_applies_packed_mask_on_mps():
    logits = torch.arange(80, dtype=torch.float32, device="mps").reshape(2, 40)
    mask = _packed_rows("mps")

    XGrammarGrammarBackend.apply_vocab_mask(logits, mask)
    torch.mps.synchronize()

    torch.testing.assert_close(logits.cpu(), _expected())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
