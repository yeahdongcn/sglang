# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.models.dits.magi2_mhc import sinkhorn_knopp


def test_sinkhorn_cpu_reference_is_unchanged():
    h = torch.randn(7, 4, 4)
    expected = torch.exp(h - h.amax(dim=(-2, -1), keepdim=True))
    for _ in range(3):
        expected = expected / (expected.sum(dim=-2, keepdim=True) + 1e-6)
        expected = expected / (expected.sum(dim=-1, keepdim=True) + 1e-6)
    actual = sinkhorn_knopp(h, num_iters=3, eps=1e-6)
    torch.testing.assert_close(actual, expected)
