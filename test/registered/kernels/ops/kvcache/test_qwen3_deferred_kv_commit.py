"""Small-memory validation for the two-launch deferred KV commit."""

from __future__ import annotations

from unittest import mock

import pytest
import torch

import sglang.kernels.fused_op as fused_op
from sglang.kernels.ops.kvcache import (
    qwen3_commit_deferred_kv,
)
from sglang.kernels.ops.kvcache.qwen3 import _DEFERRED_KV_COMMIT
from sglang.kernels.spec import KernelBackend, PlatformInfo
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mps_ci(est_time=8, suite="stage-a-unit-test-mps")

_HAS_MPS_JIT = torch.backends.mps.is_available() and callable(
    getattr(torch.mps, "compile_shader", None)
)


def test_deferred_kv_commit_auto_falls_back_to_torch_for_cpu_storage(monkeypatch):
    """Tensor contract drift must be a selector miss, not a Metal error."""
    monkeypatch.setattr(fused_op, "_platform", lambda: PlatformInfo(device_type="mps"))
    monkeypatch.setattr(fused_op, "get_fused_op_backend", lambda: None)
    layers, num_rows, pool_slots, kv_heads, head_dim = 28, 2, 4, 8, 128
    new_k = torch.randn(layers, num_rows, kv_heads, head_dim, dtype=torch.bfloat16)
    new_v = torch.randn_like(new_k)
    slots = torch.tensor([3, 1], dtype=torch.int64)
    k_pools = [
        torch.zeros(pool_slots, kv_heads, head_dim, dtype=torch.bfloat16)
        for _ in range(layers)
    ]
    v_pools = [torch.zeros_like(pool) for pool in k_pools]

    assert not _DEFERRED_KV_COMMIT.backend_eligible(
        KernelBackend.METAL_JIT, new_k, new_v, slots, k_pools, v_pools
    )
    qwen3_commit_deferred_kv(new_k, new_v, slots, k_pools, v_pools)
    for layer in range(layers):
        torch.testing.assert_close(k_pools[layer][slots], new_k[layer])
        torch.testing.assert_close(v_pools[layer][slots], new_v[layer])


@pytest.mark.skipif(not _HAS_MPS_JIT, reason="requires Torch 2.13 MPS Metal JIT")
def test_deferred_kv_commit_non_bf16_mps_storage_selects_torch(monkeypatch):
    monkeypatch.setattr(fused_op, "get_fused_op_backend", lambda: None)
    layers, num_rows, pool_slots, kv_heads, head_dim = 28, 1, 2, 8, 128
    new_k = torch.randn(
        layers, num_rows, kv_heads, head_dim, device="mps", dtype=torch.float32
    )
    new_v = torch.randn_like(new_k)
    slots = torch.zeros(num_rows, device="mps", dtype=torch.int64)
    k_pools = [
        torch.zeros(pool_slots, kv_heads, head_dim, device="mps", dtype=torch.float32)
        for _ in range(layers)
    ]
    v_pools = [torch.zeros_like(pool) for pool in k_pools]

    assert not _DEFERRED_KV_COMMIT.backend_eligible(
        KernelBackend.METAL_JIT, new_k, new_v, slots, k_pools, v_pools
    )
    qwen3_commit_deferred_kv(new_k, new_v, slots, k_pools, v_pools)
    torch.mps.synchronize()
    for layer in range(layers):
        torch.testing.assert_close(k_pools[layer][slots].cpu(), new_k[layer].cpu())
        torch.testing.assert_close(v_pools[layer][slots].cpu(), new_v[layer].cpu())


@pytest.mark.skipif(not _HAS_MPS_JIT, reason="requires Torch 2.13 MPS Metal JIT")
def test_deferred_kv_commit_unaligned_contiguous_storage_uses_scalar_metal(
    monkeypatch,
):
    """Pinned Metal remains valid by selecting its exact scalar-copy kernel."""
    monkeypatch.setattr(fused_op, "get_fused_op_backend", lambda: None)
    layers, num_rows, pool_slots, kv_heads, head_dim = 28, 1, 2, 8, 128
    elements = layers * num_rows * kv_heads * head_dim
    storage = torch.randn(elements + 1, device="mps", dtype=torch.bfloat16)
    new_k = storage[1:].view(layers, num_rows, kv_heads, head_dim)
    assert new_k.is_contiguous()
    assert new_k.data_ptr() % 16 != 0
    new_v_storage = torch.randn(elements + 1, device="mps", dtype=torch.bfloat16)
    new_v = new_v_storage[1:].view(layers, num_rows, kv_heads, head_dim)
    slots = torch.zeros(num_rows, device="mps", dtype=torch.int64)
    pool_elements = pool_slots * kv_heads * head_dim

    def unaligned_pool():
        storage = torch.zeros(pool_elements + 1, device="mps", dtype=torch.bfloat16)
        pool = storage[1:].view(pool_slots, kv_heads, head_dim)
        assert pool.is_contiguous() and pool.data_ptr() % 16 != 0
        return storage, pool

    k_storage_and_pools = [unaligned_pool() for _ in range(layers)]
    v_storage_and_pools = [unaligned_pool() for _ in range(layers)]
    k_pools = [pool for _, pool in k_storage_and_pools]
    v_pools = [pool for _, pool in v_storage_and_pools]

    assert _DEFERRED_KV_COMMIT.backend_eligible(
        KernelBackend.METAL_JIT, new_k, new_v, slots, k_pools, v_pools
    )
    qwen3_commit_deferred_kv(
        new_k,
        new_v,
        slots,
        k_pools,
        v_pools,
        backend=KernelBackend.METAL_JIT,
    )
    torch.mps.synchronize()
    for layer in range(layers):
        torch.testing.assert_close(k_pools[layer][slots].cpu(), new_k[layer].cpu())
        torch.testing.assert_close(v_pools[layer][slots].cpu(), new_v[layer].cpu())


@pytest.mark.skipif(not _HAS_MPS_JIT, reason="requires Torch 2.13 MPS Metal JIT")
def test_deferred_kv_commit_updates_prefill_rows_in_two_async_launches():
    torch.manual_seed(101)
    layers, num_rows, slots_count, kv_heads, head_dim = 28, 4, 16, 8, 128
    new_k = torch.randn(
        layers, num_rows, kv_heads, head_dim, device="mps", dtype=torch.bfloat16
    )
    new_v = torch.randn_like(new_k)
    slots = torch.tensor([3, 11, 5, 14], device="mps", dtype=torch.int64)
    k_pools = [
        torch.zeros(slots_count, kv_heads, head_dim, device="mps", dtype=torch.bfloat16)
        for _ in range(layers)
    ]
    v_pools = [torch.zeros_like(pool) for pool in k_pools]

    with mock.patch.object(
        torch.mps, "synchronize", wraps=torch.mps.synchronize
    ) as synchronize:
        qwen3_commit_deferred_kv(new_k, new_v, slots, k_pools, v_pools)
    assert synchronize.call_count == 0

    torch.mps.synchronize()
    for layer in range(layers):
        torch.testing.assert_close(k_pools[layer][slots].cpu(), new_k[layer].cpu())
        torch.testing.assert_close(v_pools[layer][slots].cpu(), new_v[layer].cpu())
        untouched = torch.tensor(
            [0, 1, 2, 4, 6, 7, 8, 9, 10, 12, 13, 15],
            device="mps",
            dtype=torch.int64,
        )
        assert torch.count_nonzero(k_pools[layer][untouched]).item() == 0
        assert torch.count_nonzero(v_pools[layer][untouched]).item() == 0


@pytest.mark.skipif(not _HAS_MPS_JIT, reason="requires Torch 2.13 MPS Metal JIT")
def test_deferred_kv_commit_matches_torch_reference():
    torch.manual_seed(102)
    layers, num_rows, slots_count, kv_heads, head_dim = 28, 2, 8, 8, 128
    new_k = torch.randn(
        layers, num_rows, kv_heads, head_dim, device="mps", dtype=torch.bfloat16
    )
    new_v = torch.randn_like(new_k)
    slots = torch.tensor([6, 1], device="mps", dtype=torch.int64)

    def pools():
        k_pools = [
            torch.zeros(
                slots_count,
                kv_heads,
                head_dim,
                device="mps",
                dtype=torch.bfloat16,
            )
            for _ in range(layers)
        ]
        return k_pools, [torch.zeros_like(pool) for pool in k_pools]

    metal_k, metal_v = pools()
    torch_k, torch_v = pools()
    qwen3_commit_deferred_kv(
        new_k,
        new_v,
        slots,
        metal_k,
        metal_v,
        backend=KernelBackend.METAL_JIT,
    )
    qwen3_commit_deferred_kv(
        new_k,
        new_v,
        slots,
        torch_k,
        torch_v,
        backend=KernelBackend.TORCH,
    )
    torch.mps.synchronize()

    for metal_pool, torch_pool in zip((*metal_k, *metal_v), (*torch_k, *torch_v)):
        torch.testing.assert_close(metal_pool.cpu(), torch_pool.cpu(), atol=0, rtol=0)


@pytest.mark.skipif(not _HAS_MPS_JIT, reason="requires Torch 2.13 MPS Metal JIT")
def test_deferred_kv_commit_matches_torch_for_512_shuffled_rows():
    """Exercise long-prefill row indexing across both 14-layer launches."""
    torch.manual_seed(103)
    layers, num_rows, pool_slots, kv_heads, head_dim = 28, 512, 521, 8, 128
    slot_generator = torch.Generator(device="cpu").manual_seed(103)
    slots_cpu = torch.randperm(pool_slots, generator=slot_generator, dtype=torch.int64)[
        :num_rows
    ].contiguous()
    assert slots_cpu.unique().numel() == num_rows
    untouched_cpu = torch.ones(pool_slots, dtype=torch.bool)
    untouched_cpu[slots_cpu] = False

    new_k = torch.randn(
        layers, num_rows, kv_heads, head_dim, device="mps", dtype=torch.bfloat16
    )
    new_v = torch.randn_like(new_k)
    slots = slots_cpu.to(device="mps")

    def pools():
        k_pools = []
        v_pools = []
        for _ in range(layers):
            k_pool = torch.zeros(
                pool_slots,
                kv_heads,
                head_dim,
                device="mps",
                dtype=torch.bfloat16,
            )
            k_pools.append(k_pool)
            v_pools.append(torch.zeros_like(k_pool))
        return k_pools, v_pools

    metal_k, metal_v = pools()
    torch_k, torch_v = pools()
    try:
        qwen3_commit_deferred_kv(
            new_k,
            new_v,
            slots,
            metal_k,
            metal_v,
            backend=KernelBackend.METAL_JIT,
        )
        qwen3_commit_deferred_kv(
            new_k,
            new_v,
            slots,
            torch_k,
            torch_v,
            backend=KernelBackend.TORCH,
        )
        torch.mps.synchronize()

        # Compare and release one layer at a time. This keeps CPU staging small
        # and progressively drops the four 521-row MPS pools after both launches
        # have completed. Layers 13 and 14 straddle the two-launch boundary.
        for layer in reversed(range(layers)):
            metal_k_cpu = metal_k.pop().cpu()
            metal_v_cpu = metal_v.pop().cpu()
            torch_k_cpu = torch_k.pop().cpu()
            torch_v_cpu = torch_v.pop().cpu()
            torch.testing.assert_close(metal_k_cpu, torch_k_cpu, atol=0, rtol=0)
            torch.testing.assert_close(metal_v_cpu, torch_v_cpu, atol=0, rtol=0)

            if layer in (13, 14):
                source_k_cpu = new_k[layer].cpu()
                source_v_cpu = new_v[layer].cpu()
                torch.testing.assert_close(
                    metal_k_cpu[slots_cpu], source_k_cpu, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    metal_v_cpu[slots_cpu], source_v_cpu, atol=0, rtol=0
                )
                assert torch.count_nonzero(metal_k_cpu[untouched_cpu]).item() == 0
                assert torch.count_nonzero(metal_v_cpu[untouched_cpu]).item() == 0
                del source_k_cpu, source_v_cpu

            del metal_k_cpu, metal_v_cpu, torch_k_cpu, torch_v_cpu
    finally:
        # If either launch or an assertion fails, fence before dropping storage
        # that an already-submitted Metal command buffer may still reference.
        try:
            torch.mps.synchronize()
        finally:
            metal_k.clear()
            metal_v.clear()
            torch_k.clear()
            torch_v.clear()
            del new_k, new_v, slots
            torch.mps.empty_cache()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
