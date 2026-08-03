"""Small MLX-only contracts for the Qwen3 model-island primitives."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")

mx = pytest.importorskip("mlx.core")

from sglang.kernels.ops.attention.qwen3_mlx import (  # noqa: E402
    qwen3_qkv_prepare_deferred,
    warmup_qwen3_qkv_prepare_deferred,
)
from sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx import (  # noqa: E402
    _prepare_qkv,
    _rms_norm,
    _rope_neox,
    _rope_neox_qk,
    _swiglu,
)


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="Qwen3 MLX primitives require Metal"
)
def test_qwen3_qk_rope_pair_matches_two_independent_rotations():
    mx.random.seed(17)
    tokens = 3
    head_dim = 128
    q = mx.random.normal((tokens, 16, head_dim)).astype(mx.bfloat16)
    k = mx.random.normal((tokens, 8, head_dim)).astype(mx.bfloat16)
    cos_sin = mx.random.normal((64, head_dim)).astype(mx.bfloat16)
    positions = mx.array([0, 7, 31], dtype=mx.int64)

    expected_q = _rope_neox(q, cos_sin, positions)
    expected_k = _rope_neox(k, cos_sin, positions)
    actual_q, actual_k = _rope_neox_qk(q, k, cos_sin, positions)
    mx.eval(expected_q, expected_k, actual_q, actual_k)

    assert bool(mx.all(expected_q == actual_q).item())
    assert bool(mx.all(expected_k == actual_k).item())


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="Qwen3 MLX primitives require Metal"
)
def test_qwen3_rms_norm_preserves_multiply_before_bf16_cast_contract():
    mx.random.seed(29)
    value = mx.random.normal((3, 16, 128)).astype(mx.bfloat16)
    weight = mx.random.normal((128,)).astype(mx.bfloat16)
    epsilon = 1e-6

    actual = _rms_norm(value, weight, epsilon)
    expected = mx.fast.rms_norm(
        value.astype(mx.float32),
        weight.astype(mx.float32),
        epsilon,
    ).astype(mx.bfloat16)
    mx.eval(actual, expected)

    assert bool(mx.all(actual == expected).item())


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="Qwen3 MLX primitives require Metal"
)
@pytest.mark.parametrize("rows", [1, 8, 512])
def test_qwen3_swiglu_matches_reference_for_normal_scale(rows):
    mx.random.seed(131 + rows)
    gate = mx.random.normal((rows, 3072)).astype(mx.bfloat16)
    up = mx.random.normal((rows, 3072)).astype(mx.bfloat16)
    actual = _swiglu(gate, up)
    expected = (mx.sigmoid(gate) * gate) * up
    mx.eval(actual, expected)
    assert bool(mx.all(actual == expected).item())


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="Qwen3 MLX primitives require Metal"
)
@pytest.mark.parametrize("tokens", [1, 8, 512])
def test_qwen3_fused_qkv_matches_staged_correctness_path(tokens):
    mx.random.seed(41 + tokens)
    epsilon = 1e-6
    qkv = mx.random.normal((tokens, 4096)).astype(mx.bfloat16)
    q_weight = mx.random.normal((128,)).astype(mx.bfloat16)
    k_weight = mx.random.normal((128,)).astype(mx.bfloat16)
    cos_sin = mx.random.normal((64, 128)).astype(mx.bfloat16)
    positions = mx.arange(tokens, dtype=mx.int64) % 64
    layer = SimpleNamespace(
        q_norm=SimpleNamespace(array=q_weight),
        k_norm=SimpleNamespace(array=k_weight),
        rope_cache=SimpleNamespace(array=cos_sin),
        qk_epsilon=epsilon,
    )

    warmup_qwen3_qkv_prepare_deferred(epsilon)
    expected = _prepare_qkv(qkv, layer, positions)
    actual = qwen3_qkv_prepare_deferred(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        positions,
        epsilon=epsilon,
    )
    mx.eval(*expected, *actual)

    for expected_array, actual_array in zip(expected, actual):
        assert bool(mx.all(expected_array == actual_array).item())


def test_qwen3_fused_qkv_rejects_wrong_packed_width():
    qkv = mx.zeros((1, 16), dtype=mx.bfloat16)
    weight = mx.ones((128,), dtype=mx.bfloat16)
    cos_sin = mx.ones((1, 128), dtype=mx.bfloat16)
    positions = mx.zeros((1,), dtype=mx.int64)

    with pytest.raises(RuntimeError, match="qkv shape mismatch"):
        qwen3_qkv_prepare_deferred(
            qkv,
            weight,
            weight,
            cos_sin,
            positions,
            epsilon=1e-6,
        )


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="Qwen3 MLX primitives require Metal"
)
def test_qwen3_fused_qkv_reuses_one_pipeline_across_rope_cache_lengths():
    mx.random.seed(97)
    epsilon = 1e-5
    qkv = mx.random.normal((1, 4096)).astype(mx.bfloat16)
    q_weight = mx.random.normal((128,)).astype(mx.bfloat16)
    k_weight = mx.random.normal((128,)).astype(mx.bfloat16)
    warmup_qwen3_qkv_prepare_deferred(epsilon)

    for cache_length in (1, 37, 4096):
        cos_sin = mx.random.normal((cache_length, 128)).astype(mx.bfloat16)
        positions = mx.array(
            [0 if cache_length == 1 else cache_length - 1], dtype=mx.int64
        )
        expected_layer = SimpleNamespace(
            q_norm=SimpleNamespace(array=q_weight),
            k_norm=SimpleNamespace(array=k_weight),
            rope_cache=SimpleNamespace(array=cos_sin),
            qk_epsilon=epsilon,
        )
        expected = _prepare_qkv(qkv, expected_layer, positions)
        actual = qwen3_qkv_prepare_deferred(
            qkv,
            q_weight,
            k_weight,
            cos_sin,
            positions,
            epsilon=epsilon,
        )
        mx.eval(*expected, *actual)
        for expected_array, actual_array in zip(expected, actual):
            assert bool(mx.all(expected_array == actual_array).item())


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="Qwen3 MLX primitives require Metal"
)
def test_qwen3_fused_qkv_materializes_strided_projection_rows():
    mx.random.seed(109)
    epsilon = 1e-6
    base = mx.random.normal((3, 8192)).astype(mx.bfloat16)
    qkv = base[:, ::2]
    q_weight = mx.random.normal((128,)).astype(mx.bfloat16)
    k_weight = mx.random.normal((128,)).astype(mx.bfloat16)
    cos_sin = mx.random.normal((32, 128)).astype(mx.bfloat16)
    positions = mx.array([0, 7, 31], dtype=mx.int64)
    actual = qwen3_qkv_prepare_deferred(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        positions,
        epsilon=epsilon,
    )
    expected_layer = SimpleNamespace(
        q_norm=SimpleNamespace(array=q_weight),
        k_norm=SimpleNamespace(array=k_weight),
        rope_cache=SimpleNamespace(array=cos_sin),
        qk_epsilon=epsilon,
    )
    expected = _prepare_qkv(qkv, expected_layer, positions)
    mx.eval(*actual, *expected)
    for expected_array, actual_array in zip(expected, actual):
        assert bool(mx.all(expected_array == actual_array).item())


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="Qwen3 MLX primitives require Metal"
)
def test_qwen3_fused_qkv_defensively_zeroes_invalid_rope_positions():
    mx.random.seed(113)
    qkv = mx.random.normal((2, 4096)).astype(mx.bfloat16)
    q_weight = mx.random.normal((128,)).astype(mx.bfloat16)
    k_weight = mx.random.normal((128,)).astype(mx.bfloat16)
    cos_sin = mx.random.normal((4, 128)).astype(mx.bfloat16)
    positions = mx.array([-1, 4], dtype=mx.int64)
    q_out, k_out, v_out = qwen3_qkv_prepare_deferred(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        positions,
        epsilon=1e-6,
    )
    expected_v = qkv[:, 3072:].reshape((2, 8, 128))
    mx.eval(q_out, k_out, v_out, expected_v)
    assert bool(mx.all(q_out == 0).item())
    assert bool(mx.all(k_out == 0).item())
    assert bool(mx.all(v_out == expected_v).item())


def test_qwen3_fused_qkv_rejects_non_contract_dtypes_and_shapes():
    qkv = mx.zeros((1, 4096), dtype=mx.bfloat16)
    weight = mx.ones((128,), dtype=mx.bfloat16)
    cache = mx.ones((4, 128), dtype=mx.bfloat16)
    positions = mx.zeros((1,), dtype=mx.int64)

    with pytest.raises(RuntimeError, match="qkv must be bfloat16"):
        qwen3_qkv_prepare_deferred(
            qkv.astype(mx.float32), weight, weight, cache, positions, epsilon=1e-6
        )
    with pytest.raises(RuntimeError, match="positions must be int64"):
        qwen3_qkv_prepare_deferred(
            qkv,
            weight,
            weight,
            cache,
            positions.astype(mx.int32),
            epsilon=1e-6,
        )
    with pytest.raises(RuntimeError, match="cos_sin must have shape"):
        qwen3_qkv_prepare_deferred(
            qkv,
            weight,
            weight,
            mx.ones((4, 64), dtype=mx.bfloat16),
            positions,
            epsilon=1e-6,
        )


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="Qwen3 MLX primitives require Metal"
)
def test_qwen3_fused_qkv_accepts_positive_exponent_kernel_name():
    qkv = mx.zeros((1, 4096), dtype=mx.bfloat16)
    weight = mx.ones((128,), dtype=mx.bfloat16)
    cache = mx.ones((1, 128), dtype=mx.bfloat16)
    positions = mx.zeros((1,), dtype=mx.int64)
    outputs = qwen3_qkv_prepare_deferred(
        qkv,
        weight,
        weight,
        cache,
        positions,
        epsilon=1.0,
    )
    mx.eval(*outputs)
    assert [array.shape for array in outputs] == [
        (1, 16, 128),
        (1, 8, 128),
        (1, 8, 128),
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
