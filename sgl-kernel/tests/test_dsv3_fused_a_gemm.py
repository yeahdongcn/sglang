import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import dsv3_fused_a_gemm
from utils import get_device


@pytest.mark.parametrize("num_tokens", [i + 1 for i in range(16)])
def test_dsv3_fused_a_gemm(num_tokens):
    kHdIn = 7168
    kHdOut = 2112

    mat_a = torch.randn(
        (num_tokens, kHdIn), dtype=torch.bfloat16, device=get_device()
    ).contiguous()
    mat_b = torch.randn(
        (kHdOut, kHdIn), dtype=torch.bfloat16, device=get_device()
    ).transpose(0, 1)
    output = torch.empty(
        (num_tokens, kHdOut), dtype=torch.bfloat16, device=get_device()
    ).contiguous()

    ref = F.linear(mat_a, mat_b.T)

    output = dsv3_fused_a_gemm(mat_a, mat_b)

    assert torch.allclose(
        output, ref, rtol=1e-2, atol=1e-3
    ), "Fused GEMM output mismatch with torch.nn.functional.linear reference"


if __name__ == "__main__":
    pytest.main([__file__])
