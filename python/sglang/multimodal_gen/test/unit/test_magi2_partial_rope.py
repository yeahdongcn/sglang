# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.models.dits.magi2_common import (
    apply_partial_rope,
)


def test_partial_rope_cpu_matches_tiled_reference():
    torch.manual_seed(11)
    x = torch.randn(17, 3, 128, dtype=torch.float32)
    cos = torch.randn(17, 32, dtype=torch.float32)
    sin = torch.randn(17, 32, dtype=torch.float32)

    rotary_dim = cos.shape[-1] * 2
    rotated, tail = x[..., :rotary_dim], x[..., rotary_dim:]
    expected = rotated * cos.repeat(1, 2).unsqueeze(1)
    first, second = rotated.chunk(2, dim=-1)
    expected = expected + torch.cat((-second, first), dim=-1) * sin.repeat(
        1, 2
    ).unsqueeze(1)
    expected = torch.cat((expected, tail), dim=-1)

    torch.testing.assert_close(apply_partial_rope(x, cos, sin), expected)
