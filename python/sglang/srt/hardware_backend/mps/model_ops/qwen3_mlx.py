"""Decode-only whole-transformer MLX island for Torch-owned Qwen3-0.6B.

The standard SRT ``ModelRunner`` remains authoritative for loading, request
bookkeeping, Radix tables, and KV allocation.  This module only borrows those
Torch MPS allocations for one decode forward.  It deliberately does not use
``mlx-lm`` or construct a second model.

One accelerated forward has three synchronization boundaries:

* one Torch-to-MLX producer fence in :func:`mlx_call_multi`;
* one shared ``mx.eval`` for the final hidden state and all deferred K/V rows;
* one deferred semantic K/V commit for all 28 layers, using the independently
  selected Torch or Torch-stream Metal implementation.

Only eligible decode batches may use the whole-model island. Non-decode and
unsupported dynamic request shapes stay on the standard Torch/Metal path.
Storage replacement after installation is a lifecycle violation and fails
loudly. There is no per-layer framework crossing and no post-commit host fence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Optional

import torch

from sglang.kernels.ops.attention.qwen3_mlx import (
    qwen3_radix_decode_deferred,
    warmup_qwen3_radix_decode_deferred,
)
from sglang.kernels.ops.attention.qwen3_mps import QWEN3_06B_METAL_SPEC
from sglang.kernels.ops.kvcache.qwen3 import (
    qwen3_commit_deferred_kv,
    warmup_qwen3_kv_commit,
)
from sglang.kernels.spec import KernelBackend
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import (
    get_req_to_token_pool,
    get_token_to_kv_pool,
)
from sglang.srt.utils.tensor_bridge import (
    MlxTensorView,
    borrow_torch_tensors,
    mlx_call_multi,
)

QWEN3_06B_NUM_LAYERS = 28
QWEN3_06B_HIDDEN_SIZE = 1024
QWEN3_06B_INTERMEDIATE_SIZE = 3072
QWEN3_06B_ROPE_BASE = 1_000_000.0
MLX_WHOLE_MODEL_CACHE_LIMIT_BYTES = 256 * 1024 * 1024


@lru_cache(maxsize=1)
def _configure_mlx_memory_cache() -> None:
    import mlx.core as mx

    mx.set_cache_limit(MLX_WHOLE_MODEL_CACHE_LIMIT_BYTES)


def _is_mps_vector(
    value: Any,
    *,
    dtype: torch.dtype,
    length: int,
) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.device.type == "mps"
        and value.dtype == dtype
        and tuple(value.shape) == (length,)
        and value.is_contiguous()
    )


@dataclass(frozen=True)
class _LayerViews:
    input_norm: MlxTensorView
    qkv: MlxTensorView
    q_norm: MlxTensorView
    k_norm: MlxTensorView
    rope_cache: MlxTensorView
    o_proj: MlxTensorView
    post_attention_norm: MlxTensorView
    gate_up: MlxTensorView
    down: MlxTensorView
    k_pool: MlxTensorView
    v_pool: MlxTensorView
    input_epsilon: float
    qk_epsilon: float
    post_attention_epsilon: float


@dataclass(frozen=True)
class _DecodeViews:
    embedding: MlxTensorView
    layers: tuple[_LayerViews, ...]
    final_norm: MlxTensorView
    final_epsilon: float
    pool_identity: int
    pool_slots: int


@dataclass(frozen=True)
class _MlxArrayRef:
    """Minimal array holder used while tracing a compiled MLX graph."""

    array: Any


@dataclass(frozen=True)
class _DecodeStatic:
    layer_epsilons: tuple[tuple[float, float, float], ...]
    final_epsilon: float
    pool_identity: int
    pool_slots: int


def _flatten_decode_arrays(views: _DecodeViews) -> list[Any]:
    """Flatten every dynamic MLX capture in a stable, testable order."""
    arrays = [views.embedding.array]
    for layer in views.layers:
        arrays.extend(
            (
                layer.input_norm.array,
                layer.qkv.array,
                layer.q_norm.array,
                layer.k_norm.array,
                layer.rope_cache.array,
                layer.o_proj.array,
                layer.post_attention_norm.array,
                layer.gate_up.array,
                layer.down.array,
                layer.k_pool.array,
                layer.v_pool.array,
            )
        )
    arrays.append(views.final_norm.array)
    expected = 2 + QWEN3_06B_NUM_LAYERS * 11
    if len(arrays) != expected:
        raise RuntimeError(
            f"Qwen3 MLX graph expected {expected} captured arrays; found {len(arrays)}"
        )
    return arrays


def _decode_static(views: _DecodeViews) -> _DecodeStatic:
    return _DecodeStatic(
        layer_epsilons=tuple(
            (
                layer.input_epsilon,
                layer.qk_epsilon,
                layer.post_attention_epsilon,
            )
            for layer in views.layers
        ),
        final_epsilon=views.final_epsilon,
        pool_identity=views.pool_identity,
        pool_slots=views.pool_slots,
    )


def _unflatten_decode_arrays(arrays: list[Any], static: _DecodeStatic) -> _DecodeViews:
    """Rebuild the lightweight graph view consumed by ``_mlx_decode_graph``."""
    iterator = iter(arrays)
    embedding = _MlxArrayRef(next(iterator))
    layers = []
    for input_epsilon, qk_epsilon, post_attention_epsilon in static.layer_epsilons:
        layers.append(
            _LayerViews(
                *(_MlxArrayRef(next(iterator)) for _ in range(11)),
                input_epsilon=input_epsilon,
                qk_epsilon=qk_epsilon,
                post_attention_epsilon=post_attention_epsilon,
            )
        )
    final_norm = _MlxArrayRef(next(iterator))
    try:
        next(iterator)
    except StopIteration:
        pass
    else:  # pragma: no cover - paired with the flatten invariant
        raise AssertionError("unexpected Qwen3 MLX graph capture count")
    return _DecodeViews(
        embedding=embedding,
        layers=tuple(layers),
        final_norm=final_norm,
        final_epsilon=static.final_epsilon,
        pool_identity=static.pool_identity,
        pool_slots=static.pool_slots,
    )


def _array_signature(arrays: tuple[Any, ...] | list[Any]) -> tuple[Any, ...]:
    return tuple(
        (tuple(array.shape), str(array.dtype).rsplit(".", 1)[-1], int(array.ndim))
        for array in arrays
    )


@dataclass
class _CompiledBs1Decode:
    """Exactly one shape-specialized decode executable and its borrowed owner."""

    owner: _DecodeViews
    captures: list[Any]
    static: _DecodeStatic
    capture_signature: tuple[Any, ...]
    dynamic_signature: Optional[tuple[Any, ...]] = None
    compiled: Optional[Callable[..., Any]] = None
    warmup_count: int = 0
    call_count: int = 0
    fallback_count: int = 0

    @classmethod
    def create(cls, views: _DecodeViews) -> _CompiledBs1Decode:
        import mlx.core as mx

        captures = _flatten_decode_arrays(views)
        static = _decode_static(views)
        bundle = cls(
            owner=views,
            captures=captures,
            static=static,
            capture_signature=_array_signature(captures),
        )

        def graph(input_ids, positions, req_to_token, req_pool_indices, seq_lens):
            graph_views = _unflatten_decode_arrays(bundle.captures, bundle.static)
            return _mlx_decode_graph(
                graph_views,
                input_ids,
                positions,
                req_to_token,
                req_pool_indices,
                seq_lens,
            )

        # Captures must be an MLX-recognized mutable list.  Keeping the list
        # object stable lets an online weight refresh replace its leaves after
        # the Torch producer fence without accumulating compiled generations.
        # This decode-only provider retains exactly one full-model executable:
        # hidden states return to the standard Torch logits/sampling tail.
        bundle.compiled = mx.compile(graph, inputs=captures, shapeless=False)
        return bundle

    @staticmethod
    def _is_bs1(inputs: tuple[Any, ...]) -> bool:
        if len(inputs) != 5:
            return False
        input_ids, positions, req_to_token, req_pool_indices, seq_lens = inputs
        return (
            tuple(input_ids.shape) == (1,)
            and tuple(positions.shape) == (1,)
            and int(req_to_token.ndim) == 2
            and tuple(req_pool_indices.shape) == (1,)
            and tuple(seq_lens.shape) == (1,)
        )

    def can_run(self, inputs: tuple[Any, ...]) -> bool:
        if not self._is_bs1(inputs):
            return False
        signature = _array_signature(inputs)
        return self.dynamic_signature in (None, signature)

    def can_run_hidden(self, inputs: tuple[Any, ...]) -> bool:
        return self.compiled is not None and self.can_run(inputs)

    def __call__(self, *inputs: Any):
        signature = _array_signature(inputs)
        if self.dynamic_signature is None:
            self.dynamic_signature = signature
        elif signature != self.dynamic_signature:
            raise RuntimeError("Qwen3 compiled MLX decode input signature changed")
        assert self.compiled is not None
        self.call_count += 1
        return self.compiled(*inputs)

    def warmup(self, *inputs: Any):
        """Compile/evaluate one signature without counting a served decode."""
        signature = _array_signature(inputs)
        if self.dynamic_signature is None:
            self.dynamic_signature = signature
        elif signature != self.dynamic_signature:
            raise RuntimeError("Qwen3 compiled MLX warmup signature changed")
        assert self.compiled is not None
        self.warmup_count += 1
        return self.compiled(*inputs)

    def refresh(self, views: _DecodeViews) -> None:
        new_captures = _flatten_decode_arrays(views)
        if _decode_static(views) != self.static:
            raise RuntimeError(
                "Qwen3 compiled MLX decode static metadata changed; rebuild the "
                "platform operator plan"
            )
        if _array_signature(new_captures) != self.capture_signature:
            raise RuntimeError(
                "Qwen3 compiled MLX decode capture signature changed; rebuild the "
                "platform operator plan"
            )
        self.captures[:] = new_captures
        self.owner = views

    def close(self) -> None:
        self.compiled = None
        self.captures.clear()


def _require_mps_parameter(
    name: str,
    value: Any,
    *,
    shape: Optional[tuple[int, ...]] = None,
) -> torch.Tensor:
    if not isinstance(value, torch.nn.Parameter):
        raise RuntimeError(f"{name} must be a Torch Parameter")
    tensor = value.detach()
    if (
        tensor.device.type != "mps"
        or tensor.dtype != torch.bfloat16
        or not tensor.is_contiguous()
    ):
        raise RuntimeError(
            f"{name} must be a contiguous MPS bfloat16 Parameter; found "
            f"device={tensor.device}, dtype={tensor.dtype}, "
            f"contiguous={tensor.is_contiguous()}"
        )
    if shape is not None and tuple(tensor.shape) != shape:
        raise RuntimeError(
            f"{name} shape mismatch: expected {shape}, found {tuple(tensor.shape)}"
        )
    return tensor


def _require_linear(name: str, module: Any, shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(getattr(module, "quant_method", None), UnquantizedLinearMethod):
        raise RuntimeError(f"{name} must use the unquantized Torch linear method")
    if getattr(module, "bias", None) is not None:
        raise RuntimeError(f"{name} must not have a bias")
    return _require_mps_parameter(name + ".weight", module.weight, shape=shape)


def _require_norm(name: str, module: Any, width: int) -> tuple[torch.Tensor, float]:
    weight = _require_mps_parameter(name + ".weight", module.weight, shape=(width,))
    epsilon = float(getattr(module, "variance_epsilon", 0.0))
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise RuntimeError(f"{name} requires a finite positive epsilon")
    for field_name in (
        "variance_size_override",
        "cast_x_before_out_mul",
        "fp32_residual",
        "override_orig_dtype",
    ):
        value = getattr(module, field_name, None)
        if value not in (None, False):
            raise RuntimeError(
                f"{name} has unsupported RMSNorm option {field_name}={value!r}"
            )
    return weight, epsilon


def _require_equal_rope_cache_length(
    expected_length: Optional[int], found_length: int, layer_id: int
) -> int:
    """Keep one immutable RoPE-cache bound for every captured decoder layer.

    The MLX graph specializes the cache bound once and indexes that bound from
    layer zero.  Accepting a shorter cache in a later layer would therefore
    make the graph's safety check describe a different allocation than the
    layer it is indexing.  Validate this at provider installation, where it is
    free, instead of adding a per-token host synchronization.
    """
    found_length = int(found_length)
    if found_length <= 0:
        raise RuntimeError(
            f"Qwen3 MLX decode requires a non-empty RoPE cache at layer {layer_id}"
        )
    if expected_length is not None and found_length != expected_length:
        raise RuntimeError(
            "Qwen3 MLX decode requires equal RoPE cache lengths; "
            f"layer {layer_id} has {found_length}, expected {expected_length}"
        )
    return found_length if expected_length is None else expected_length


def _collect_torch_sources(model: Any, kv_pool: Any):
    """Validate the exact model/pool contract and return source tensors.

    The returned nested tuples are used only to build lifetime-bound MLX views;
    the provider never takes ownership away from the Torch modules or pool.
    """
    from sglang.srt.models.qwen3 import Qwen3Model

    if not isinstance(model, Qwen3Model) or type(model) is not Qwen3Model:
        raise RuntimeError(
            "the MLX decode island supports exactly the dense Qwen3Model, "
            "not a subclass"
        )
    # ``Module.eval()`` is the inference contract. Parameters commonly keep
    # ``requires_grad=True`` after eval, so that flag is not a valid
    # read-only/storage-lifetime test. Every borrowed view below is detached
    # and the island never mutates model weights.
    if model.training:
        raise RuntimeError("Qwen3 MLX decode requires the model to be in eval mode")
    config = model.config
    expected_config = {
        "hidden_size": QWEN3_06B_HIDDEN_SIZE,
        "intermediate_size": QWEN3_06B_INTERMEDIATE_SIZE,
        "num_hidden_layers": QWEN3_06B_NUM_LAYERS,
        "num_attention_heads": QWEN3_06B_METAL_SPEC.num_q_heads,
        "num_key_value_heads": QWEN3_06B_METAL_SPEC.num_kv_heads,
        "head_dim": QWEN3_06B_METAL_SPEC.head_dim,
    }
    for name, expected in expected_config.items():
        found = getattr(config, name, None)
        if found is None and name == "head_dim":
            found = int(config.hidden_size) // int(config.num_attention_heads)
        if found is None or int(found) != expected:
            raise RuntimeError(
                f"Qwen3 MLX decode requires {name}={expected}; found {found}"
            )
    if getattr(config, "hidden_act", "silu") not in ("silu", "swiglu"):
        raise RuntimeError("Qwen3 MLX decode supports only the SiLU gated MLP")
    if bool(getattr(config, "attention_bias", False)):
        raise RuntimeError("Qwen3 MLX decode does not support attention bias")
    rope_parameters = getattr(config, "rope_parameters", None) or getattr(
        config, "rope_scaling", None
    )
    if rope_parameters not in (None, {}):
        rope_type = rope_parameters.get(
            "rope_type", rope_parameters.get("type", "default")
        )
        if rope_type != "default":
            raise RuntimeError(
                "Qwen3 MLX decode supports only default, unscaled RoPE; "
                f"found rope_type={rope_type!r}"
            )
    rope_theta = (
        rope_parameters.get("rope_theta", 0.0)
        if rope_parameters
        else getattr(config, "rope_theta", 0.0)
    )
    if not math.isclose(
        float(rope_theta),
        QWEN3_06B_ROPE_BASE,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError(
            f"Qwen3 MLX decode requires rope_theta=1000000; found {rope_theta!r}"
        )
    if bool(getattr(config, "use_sliding_window", False)):
        raise RuntimeError(
            "Qwen3 MLX decode requires full attention; sliding-window attention "
            "is unsupported"
        )
    for field_name in ("sliding_window", "attention_chunk_size", "window_size"):
        value = getattr(config, field_name, None)
        if value not in (None, 0, -1):
            raise RuntimeError(
                "Qwen3 MLX decode requires full attention; unsupported "
                f"{field_name}={value!r}"
            )
    if (
        int(getattr(model, "start_layer", -1)) != 0
        or int(getattr(model, "end_layer", -1)) != QWEN3_06B_NUM_LAYERS
    ):
        raise RuntimeError("Qwen3 MLX decode requires the complete non-PP layer stack")
    if len(model.layers) != QWEN3_06B_NUM_LAYERS:
        raise RuntimeError("Qwen3 MLX decode requires exactly 28 decoder layers")

    if getattr(kv_pool, "kv_cache_layout", None) != "nhd" or bool(
        getattr(kv_pool, "is_quantized_kv_cache", False)
    ):
        raise RuntimeError("Qwen3 MLX decode requires an unquantized NHD KV pool")

    embedding = _require_mps_parameter("embed_tokens.weight", model.embed_tokens.weight)
    if embedding.ndim != 2 or embedding.shape[1] != QWEN3_06B_HIDDEN_SIZE:
        raise RuntimeError(
            "Qwen3 MLX decode embedding must have shape [vocab, 1024], found "
            f"{tuple(embedding.shape)}"
        )

    layer_sources = []
    pool_slots = None
    rope_cache_length = None
    for expected_layer_id, layer in enumerate(model.layers):
        attention = layer.self_attn
        layer_id = int(attention.attn.layer_id)
        if layer_id != expected_layer_id:
            raise RuntimeError(
                "Qwen3 MLX decode requires dense ordered layer ids; "
                f"expected {expected_layer_id}, found {layer_id}"
            )
        input_norm, input_epsilon = _require_norm(
            f"layers.{layer_id}.input_layernorm",
            layer.input_layernorm,
            QWEN3_06B_HIDDEN_SIZE,
        )
        post_norm, post_epsilon = _require_norm(
            f"layers.{layer_id}.post_attention_layernorm",
            layer.post_attention_layernorm,
            QWEN3_06B_HIDDEN_SIZE,
        )
        q_norm, q_epsilon = _require_norm(
            f"layers.{layer_id}.self_attn.q_norm",
            attention.q_norm,
            QWEN3_06B_METAL_SPEC.head_dim,
        )
        k_norm, k_epsilon = _require_norm(
            f"layers.{layer_id}.self_attn.k_norm",
            attention.k_norm,
            QWEN3_06B_METAL_SPEC.head_dim,
        )
        if not math.isclose(q_epsilon, k_epsilon, rel_tol=0.0, abs_tol=0.0):
            raise RuntimeError("Qwen3 Q/K RMSNorm epsilons must match")

        qkv = _require_linear(
            f"layers.{layer_id}.self_attn.qkv_proj",
            attention.qkv_proj,
            (QWEN3_06B_METAL_SPEC.qkv_width, QWEN3_06B_HIDDEN_SIZE),
        )
        o_proj = _require_linear(
            f"layers.{layer_id}.self_attn.o_proj",
            attention.o_proj,
            (
                QWEN3_06B_HIDDEN_SIZE,
                QWEN3_06B_METAL_SPEC.num_q_heads * QWEN3_06B_METAL_SPEC.head_dim,
            ),
        )
        gate_up = _require_linear(
            f"layers.{layer_id}.mlp.gate_up_proj",
            layer.mlp.gate_up_proj,
            (2 * QWEN3_06B_INTERMEDIATE_SIZE, QWEN3_06B_HIDDEN_SIZE),
        )
        down = _require_linear(
            f"layers.{layer_id}.mlp.down_proj",
            layer.mlp.down_proj,
            (QWEN3_06B_HIDDEN_SIZE, QWEN3_06B_INTERMEDIATE_SIZE),
        )

        rope = attention.rotary_emb
        rope_cache = getattr(rope, "cos_sin_cache", None)
        if (
            not isinstance(rope_cache, torch.Tensor)
            or rope_cache.device.type != "mps"
            or rope_cache.dtype != torch.bfloat16
            or rope_cache.ndim != 2
            or rope_cache.shape[1] != QWEN3_06B_METAL_SPEC.head_dim
            or not rope_cache.is_contiguous()
            or not bool(getattr(rope, "is_neox_style", False))
        ):
            raise RuntimeError(
                "Qwen3 MLX decode requires a contiguous bf16 NeoX cos/sin cache"
            )
        if not math.isclose(
            float(getattr(rope, "base", 0.0)),
            QWEN3_06B_ROPE_BASE,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError("Qwen3 MLX decode found an unsupported RoPE base")
        rope_cache_length = _require_equal_rope_cache_length(
            rope_cache_length,
            int(rope_cache.shape[0]),
            layer_id,
        )
        if not math.isclose(
            float(attention.scaling),
            QWEN3_06B_METAL_SPEC.attention_scale,
            rel_tol=1e-6,
            abs_tol=0.0,
        ):
            raise RuntimeError("Qwen3 MLX decode attention scale is unsupported")

        k_pool, v_pool = kv_pool.get_kv_buffer(layer_id)
        expected_tail = (
            QWEN3_06B_METAL_SPEC.num_kv_heads,
            QWEN3_06B_METAL_SPEC.head_dim,
        )
        for name, tensor in (("K", k_pool), ("V", v_pool)):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "mps"
                or tensor.dtype != torch.bfloat16
                or tensor.ndim != 3
                or tuple(tensor.shape[1:]) != expected_tail
                or not tensor.is_contiguous()
            ):
                raise RuntimeError(
                    f"layer {layer_id} {name} pool must be contiguous MPS bf16 NHD"
                )
        if tuple(k_pool.shape) != tuple(v_pool.shape):
            raise RuntimeError(f"layer {layer_id} K/V pool shapes differ")
        if pool_slots is None:
            pool_slots = int(k_pool.shape[0])
        elif int(k_pool.shape[0]) != pool_slots:
            raise RuntimeError("all Qwen3 KV layers must have the same slot count")

        layer_sources.append(
            (
                input_norm,
                qkv,
                q_norm,
                k_norm,
                rope_cache,
                o_proj,
                post_norm,
                gate_up,
                down,
                k_pool,
                v_pool,
                input_epsilon,
                q_epsilon,
                post_epsilon,
            )
        )

    final_norm, final_epsilon = _require_norm("norm", model.norm, QWEN3_06B_HIDDEN_SIZE)
    assert pool_slots is not None
    return embedding, tuple(layer_sources), final_norm, final_epsilon, pool_slots


def _build_views(model: Any, kv_pool: Any, *, synchronize: bool) -> _DecodeViews:
    embedding, layer_sources, final_norm, final_epsilon, pool_slots = (
        _collect_torch_sources(model, kv_pool)
    )
    tensors = [embedding]
    for sources in layer_sources:
        tensors.extend(sources[:11])
    tensors.append(final_norm)
    views = iter(borrow_torch_tensors(*tensors, synchronize=synchronize))
    embedding_view = next(views)
    layer_views = []
    for sources in layer_sources:
        layer_views.append(
            _LayerViews(
                *(next(views) for _ in range(11)),
                input_epsilon=sources[11],
                qk_epsilon=sources[12],
                post_attention_epsilon=sources[13],
            )
        )
    final_norm_view = next(views)
    try:
        next(views)
    except StopIteration:
        pass
    else:  # pragma: no cover - internal construction invariant
        raise AssertionError("unexpected Qwen3 decode view count")
    return _DecodeViews(
        embedding=embedding_view,
        layers=tuple(layer_views),
        final_norm=final_norm_view,
        final_epsilon=final_epsilon,
        pool_identity=id(kv_pool),
        pool_slots=pool_slots,
    )


def _require_req_to_token(req_to_token_pool: Any) -> torch.Tensor:
    req_to_token = getattr(req_to_token_pool, "req_to_token", None)
    if (
        not isinstance(req_to_token, torch.Tensor)
        or req_to_token.device.type != "mps"
        or req_to_token.dtype != torch.int32
        or req_to_token.ndim != 2
        or not req_to_token.is_contiguous()
    ):
        raise RuntimeError(
            "Qwen3 MLX requires a contiguous MPS int32 request-to-token table"
        )
    return req_to_token


def validate_qwen3_mlx_static_contract(
    model: torch.nn.Module,
    kv_pool: Any,
    req_to_token_pool: Any,
) -> None:
    """Validate deterministic eligibility without importing or invoking MLX."""
    from sglang.srt.models.qwen3 import Qwen3ForCausalLM

    if type(model) is not Qwen3ForCausalLM:
        raise RuntimeError("whole-model MLX supports only Qwen3ForCausalLM")
    _collect_torch_sources(model.model, kv_pool)
    _require_req_to_token(req_to_token_pool)


def _rms_norm(value, weight, epsilon: float):
    import mlx.core as mx

    value_fp32 = value.astype(mx.float32)
    # MLX's fused RMSNorm accepts fp32 activations/weights. Feeding it the
    # widened values preserves SRT's reduction/rounding contract while
    # avoiding a chain of elementwise reduction nodes in every decoder layer.
    return mx.fast.rms_norm(
        value_fp32,
        weight.astype(mx.float32),
        epsilon,
    ).astype(mx.bfloat16)


def _add_rms_norm(value, residual, weight, epsilon: float):
    """Match SRT's fused residual-add RMSNorm rounding contract."""
    import mlx.core as mx

    summed = value.astype(mx.float32) + residual.astype(mx.float32)
    normed = mx.fast.rms_norm(
        summed,
        weight.astype(mx.float32),
        epsilon,
    ).astype(mx.bfloat16)
    return normed, summed.astype(mx.bfloat16)


def _swiglu(gate, up):
    """Evaluate Qwen3's gated MLP activation on the MLX graph."""
    import mlx.core as mx

    return (mx.sigmoid(gate) * gate) * up


def _rope_neox(value, cos_sin, positions):
    import mlx.core as mx

    selected = mx.take(cos_sin, positions, axis=0)
    cosine, sine = mx.split(selected, 2, axis=-1)
    first, second = mx.split(value, 2, axis=-1)
    cosine = cosine[:, None, :]
    sine = sine[:, None, :]
    return mx.concatenate(
        (first * cosine - second * sine, second * cosine + first * sine), axis=-1
    ).astype(mx.bfloat16)


def _prepare_qkv(qkv, layer, positions):
    """Run the staged correctness reference for Q/K norm, RoPE, and dense V."""
    import mlx.core as mx

    spec = QWEN3_06B_METAL_SPEC
    q, k, v = mx.split(
        qkv,
        (
            spec.num_q_heads * spec.head_dim,
            (spec.num_q_heads + spec.num_kv_heads) * spec.head_dim,
        ),
        axis=-1,
    )
    batch = q.shape[0]
    q = _rms_norm(
        q.reshape(batch, spec.num_q_heads, spec.head_dim),
        layer.q_norm.array,
        layer.qk_epsilon,
    )
    k = _rms_norm(
        k.reshape(batch, spec.num_kv_heads, spec.head_dim),
        layer.k_norm.array,
        layer.qk_epsilon,
    )
    # ``v`` is the final subview of a wider fused QKV row.  Materialize the
    # strict dense layout required by deferred decode and deferred KV commit.
    v = mx.contiguous(v.reshape(batch, spec.num_kv_heads, spec.head_dim))
    return (
        _rope_neox(q, layer.rope_cache.array, positions),
        _rope_neox(k, layer.rope_cache.array, positions),
        v,
    )


def _mlx_model_graph(
    views: _DecodeViews,
    input_ids,
    positions,
    attention_forward,
):
    """Build the shared lazy transformer and return hidden plus deferred K/V."""
    import mlx.core as mx

    hidden = mx.take(views.embedding.array, input_ids, axis=0)
    residual = None
    new_keys = []
    new_values = []
    spec = QWEN3_06B_METAL_SPEC

    for layer in views.layers:
        if residual is None:
            residual = hidden
            normed = _rms_norm(hidden, layer.input_norm.array, layer.input_epsilon)
        else:
            normed, residual = _add_rms_norm(
                hidden,
                residual,
                layer.input_norm.array,
                layer.input_epsilon,
            )
        qkv = normed @ mx.transpose(layer.qkv.array)
        q, k, v = _prepare_qkv(qkv, layer, positions)
        batch = q.shape[0]

        attention = attention_forward(layer, q, k, v)
        attention = attention.reshape(batch, spec.num_q_heads * spec.head_dim)
        attention = attention @ mx.transpose(layer.o_proj.array)
        mlp_input, residual = _add_rms_norm(
            attention,
            residual,
            layer.post_attention_norm.array,
            layer.post_attention_epsilon,
        )
        gate_up = mlp_input @ mx.transpose(layer.gate_up.array)
        gate, up = mx.split(gate_up, 2, axis=-1)
        hidden = _swiglu(gate, up) @ mx.transpose(layer.down.array)
        new_keys.append(k)
        new_values.append(v)

    hidden, _ = _add_rms_norm(
        hidden, residual, views.final_norm.array, views.final_epsilon
    )
    # Layer-major layout lets the Torch commit kernel consume both 14-layer
    # halves as contiguous ranges without materializing a transpose.
    return hidden, mx.stack(new_keys, axis=0), mx.stack(new_values, axis=0)


def _mlx_decode_graph(
    views: _DecodeViews,
    input_ids,
    positions,
    req_to_token,
    req_pool_indices,
    seq_lens,
):
    """Build one decode graph over the Torch-owned Radix prefix."""

    def attention_forward(layer, q, k, v):
        return qwen3_radix_decode_deferred(
            q,
            k,
            v,
            layer.k_pool.array,
            layer.v_pool.array,
            req_to_token,
            req_pool_indices,
            seq_lens,
            scale=QWEN3_06B_METAL_SPEC.attention_scale,
        )

    return _mlx_model_graph(views, input_ids, positions, attention_forward)


@dataclass
class Qwen3MlxModelProvider:
    """Decode-only MLX forward provider bound to one typed Qwen3 model."""

    views: Optional[_DecodeViews]
    model_identity: Optional[int] = None
    # Deferred K/V commit is independently selected and pinned at startup.
    kv_commit_backend: KernelBackend = KernelBackend.TORCH
    req_pool_identity: Optional[int] = None
    req_to_token_data_ptr: Optional[int] = None
    req_to_token_shape: Optional[tuple[int, ...]] = None
    call_count: int = 0
    decode_call_count: int = 0
    max_decode_batch_size: int = 0
    selector_call_count: int = 0
    selector_fallback_count: int = 0
    last_selector_fallback_reason: Optional[str] = None
    started: bool = False
    closed: bool = False
    _pending_commit_sources: Optional[tuple[torch.Tensor, torch.Tensor]] = field(
        default=None, init=False, repr=False
    )
    _compiled_decode: Optional[_CompiledBs1Decode] = field(
        default=None, init=False, repr=False
    )

    @staticmethod
    def _resolve_req_to_token(req_to_token_pool: Any) -> torch.Tensor:
        return _require_req_to_token(req_to_token_pool)

    def start(self, req_to_token_pool: Any) -> None:
        """Compile and warm every owned pipeline before publication."""
        if self.closed:
            raise RuntimeError("cannot start a closed Qwen3 MLX provider")
        if self.started:
            req_to_token = self._resolve_req_to_token(req_to_token_pool)
            if (
                id(req_to_token_pool) != self.req_pool_identity
                or req_to_token.data_ptr() != self.req_to_token_data_ptr
                or tuple(req_to_token.shape) != self.req_to_token_shape
            ):
                raise RuntimeError(
                    "Qwen3 MLX provider cannot be rebound to different request storage"
                )
            return
        if self.views is None:
            raise RuntimeError("Qwen3 MLX views must exist before provider startup")

        _configure_mlx_memory_cache()
        req_to_token = self._resolve_req_to_token(req_to_token_pool)
        self.req_pool_identity = id(req_to_token_pool)
        self.req_to_token_data_ptr = int(req_to_token.data_ptr())
        self.req_to_token_shape = tuple(req_to_token.shape)
        if self.kv_commit_backend is KernelBackend.METAL_JIT:
            warmup_qwen3_kv_commit(pool_slots=self.views.pool_slots)
        elif self.kv_commit_backend is not KernelBackend.TORCH:
            raise RuntimeError(
                "Qwen3 MLX deferred KV commit supports only metal_jit or torch; "
                f"found {self.kv_commit_backend.value!r}"
            )
        warmup_qwen3_radix_decode_deferred(
            request_rows=int(req_to_token.shape[0]),
            table_stride=int(req_to_token.shape[1]),
            pool_slots=self.views.pool_slots,
        )
        self.enable_compiled_decode()
        self.warmup_compiled_decode(req_to_token)
        self.started = True

    def enable_compiled_decode(self) -> None:
        """Create the hidden-output bs=1 ``mx.compile`` executable."""
        if self.closed:
            raise RuntimeError("cannot compile a closed Qwen3 MLX provider")
        if self._compiled_decode is not None:
            return
        if self.views is None:
            raise RuntimeError(
                "Qwen3 MLX decode views must exist before graph initialization"
            )
        self._compiled_decode = _CompiledBs1Decode.create(self.views)

    def warmup_compiled_decode(self, req_to_token: torch.Tensor) -> None:
        """Pay the bs=1 compile cost during startup without mutating Torch KV."""
        compiled = self._compiled_decode
        if compiled is None:
            raise RuntimeError(
                "Qwen3 compiled MLX decode must be enabled before warmup"
            )
        if (
            not isinstance(req_to_token, torch.Tensor)
            or req_to_token.device.type != "mps"
            or req_to_token.dtype != torch.int32
            or req_to_token.ndim != 2
            or not req_to_token.is_contiguous()
        ):
            raise RuntimeError(
                "Qwen3 compiled MLX warmup requires the contiguous MPS int32 "
                "request-to-token table"
            )

        import mlx.core as mx

        (table_view,) = borrow_torch_tensors(req_to_token, synchronize=True)
        inputs = (
            mx.zeros((1,), dtype=mx.int64),
            mx.zeros((1,), dtype=mx.int64),
            table_view.array,
            mx.zeros((1,), dtype=mx.int64),
            mx.ones((1,), dtype=mx.int64),
        )
        outputs = compiled.warmup(*inputs)
        mx.eval(*outputs)

    def disable_compiled_decode(self) -> None:
        """Drop the compiled callable after its device work has been fenced."""
        compiled = self._compiled_decode
        if compiled is not None:
            compiled.close()
            self._compiled_decode = None

    def close(self) -> None:
        """Release borrowed state at an explicit worker lifecycle boundary."""
        if self.closed:
            return
        synchronize = getattr(torch.mps, "synchronize", None)
        mps_backend = getattr(torch.backends, "mps", None)
        is_available = getattr(mps_backend, "is_available", None)
        if callable(synchronize) and callable(is_available) and is_available():
            synchronize()
        self._pending_commit_sources = None
        self.disable_compiled_decode()
        self.views = None
        self.req_pool_identity = None
        self.req_to_token_data_ptr = None
        self.req_to_token_shape = None
        self.started = False
        self.closed = True

    def get_compiled_decode_state(self) -> dict[str, Any]:
        """Return the small set of serving diagnostics owned by the graph."""
        compiled = self._compiled_decode
        return {
            "enabled": bool(compiled is not None and compiled.compiled is not None),
            "warmup_count": int(getattr(compiled, "warmup_count", 0)),
            "call_count": int(getattr(compiled, "call_count", 0)),
            "fallback_count": int(getattr(compiled, "fallback_count", 0)),
        }

    def _common_fallback_reason(
        self,
        forward_batch: Any,
        *,
        model: Any = None,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Any = None,
    ) -> Optional[str]:
        if input_embeds is not None:
            return "input_embeds"
        if pp_proxy_tensors is not None:
            return "pipeline_parallel"
        if model is not None:
            if self.model_identity is not None and id(model) != self.model_identity:
                raise RuntimeError(
                    "Qwen3 MLX provider was invoked by a different model instance"
                )
            if bool(getattr(model, "training", False)):
                return "training"
            if getattr(model, "layers_to_capture", None):
                return "aux_hidden_capture"
        if getattr(
            forward_batch, "capture_hidden_mode", CaptureHiddenMode.NULL
        ) not in (None, CaptureHiddenMode.NULL):
            return "hidden_capture"
        if bool(getattr(forward_batch, "return_hidden_states_before_norm", False)):
            return "pre_norm_hidden_capture"
        for name in (
            "input_embeds",
            "replace_embeds",
            "replace_positions",
            "mm_input_embeds",
            "cross_attention_custom_mask",
            "mrope_positions",
            "token_type_ids",
            "spec_info",
            "tbo_split_seq_index",
            "encoder_lens",
            "encoder_out_cache_loc",
        ):
            if getattr(forward_batch, name, None) is not None:
                return name
        spec_algorithm = getattr(forward_batch, "spec_algorithm", None)
        if spec_algorithm is not None:
            is_none = getattr(spec_algorithm, "is_none", None)
            if not callable(is_none) or not is_none():
                return "spec_algorithm"
        mm_inputs = getattr(forward_batch, "mm_inputs", None)
        if mm_inputs and any(value is not None for value in mm_inputs):
            return "multimodal"
        lora_ids = getattr(forward_batch, "lora_ids", None)
        if lora_ids and any(value is not None for value in lora_ids):
            return "lora"
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
            return "input_ids_layout"
        token_count = int(input_ids.numel())
        if token_count <= 0:
            return "empty_batch"
        if not _is_mps_vector(input_ids, dtype=torch.int64, length=token_count):
            return "input_ids_layout"
        if not _is_mps_vector(positions, dtype=torch.int64, length=token_count):
            return "positions_layout"
        return None

    def _decode_fallback_reason(
        self,
        forward_batch: Any,
        *,
        model: Any,
        input_ids: Optional[torch.Tensor],
        positions: Optional[torch.Tensor],
        input_embeds: Optional[torch.Tensor],
        pp_proxy_tensors: Any,
    ) -> Optional[str]:
        if getattr(forward_batch, "forward_mode", None) is not ForwardMode.DECODE:
            return "forward_mode"
        reason = self._common_fallback_reason(
            forward_batch,
            model=model,
            input_ids=input_ids,
            positions=positions,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if reason is not None:
            return reason
        assert isinstance(input_ids, torch.Tensor)
        token_count = int(input_ids.numel())
        if int(getattr(forward_batch, "batch_size", -1)) != token_count:
            return "batch_size"
        for name in ("req_pool_indices", "seq_lens", "out_cache_loc"):
            if not _is_mps_vector(
                getattr(forward_batch, name, None),
                dtype=torch.int64,
                length=token_count,
            ):
                return name + "_layout"
        non_padded = getattr(forward_batch, "num_token_non_padded_cpu", None)
        if non_padded is not None and int(non_padded) != token_count:
            return "padded_batch"
        return None

    def should_run(
        self,
        forward_batch: Any,
        *,
        model: Any = None,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Any = None,
    ) -> bool:
        if self.closed:
            raise RuntimeError("Qwen3 MLX provider remained bound after close()")
        if not self.started:
            raise RuntimeError("Qwen3 MLX provider was bound before start() completed")
        reason = self._decode_fallback_reason(
            forward_batch,
            model=model,
            input_ids=input_ids,
            positions=positions,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        self.selector_call_count += 1
        self.last_selector_fallback_reason = reason
        if reason is not None:
            self.selector_fallback_count += 1
            return False
        return True

    def invalidate_views(self) -> None:
        if self.closed:
            return
        self.views = None
        # The next bridge producer fence retires any asynchronous Torch commit.

    def _retire_completed_commit_sources(self) -> None:
        """Release sources whose asynchronous Torch commit is now fenced."""
        self._pending_commit_sources = None

    def _after_mps_fence(self, views: _DecodeViews) -> None:
        """Publish refreshed captures only after older Torch reads finish."""
        self._retire_completed_commit_sources()
        compiled = self._compiled_decode
        if compiled is not None and compiled.owner is not views:
            compiled.refresh(views)

    def _validate_decode_inputs(
        self,
        model: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: Optional[torch.Tensor],
        pp_proxy_tensors: Any,
    ) -> tuple[Any, torch.Tensor]:
        reason = self._decode_fallback_reason(
            forward_batch,
            model=model,
            input_ids=input_ids,
            positions=positions,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if reason is not None:
            raise RuntimeError(
                "Qwen3 MLX decode eligibility changed after selection: " + reason
            )

        batch = int(input_ids.shape[0]) if input_ids.ndim == 1 else -1
        dynamic = (
            ("input_ids", input_ids, torch.int64),
            ("positions", positions, torch.int64),
            ("req_pool_indices", forward_batch.req_pool_indices, torch.int64),
            ("seq_lens", forward_batch.seq_lens, torch.int64),
            ("out_cache_loc", forward_batch.out_cache_loc, torch.int64),
        )
        for name, tensor, dtype in dynamic:
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "mps"
                or tensor.dtype != dtype
                or tuple(tensor.shape) != (batch,)
                or not tensor.is_contiguous()
            ):
                raise RuntimeError(
                    f"Qwen3 MLX decode requires contiguous MPS {dtype} {name}[batch]"
                )
        if batch <= 0 or int(getattr(forward_batch, "batch_size", batch)) != batch:
            raise RuntimeError("Qwen3 MLX decode requires a non-empty unpadded batch")

        kv_pool = get_token_to_kv_pool()
        req_pool = get_req_to_token_pool()
        req_to_token = getattr(req_pool, "req_to_token", None)
        if (
            not isinstance(req_to_token, torch.Tensor)
            or req_to_token.device.type != "mps"
            or req_to_token.dtype != torch.int32
            or req_to_token.ndim != 2
            or not req_to_token.is_contiguous()
        ):
            raise RuntimeError(
                "Qwen3 MLX decode requires a contiguous MPS int32 Radix table"
            )
        if (
            id(req_pool) != self.req_pool_identity
            or int(req_to_token.data_ptr()) != self.req_to_token_data_ptr
            or tuple(req_to_token.shape) != self.req_to_token_shape
        ):
            raise RuntimeError(
                "Qwen3 MLX decode detected replaced request-table storage; "
                "reinstall the platform operator plan"
            )
        return kv_pool, req_to_token

    @staticmethod
    def _validate_pool_storage(kv_pool: Any, views: _DecodeViews) -> None:
        if id(kv_pool) != views.pool_identity:
            raise RuntimeError(
                "Qwen3 MLX decode detected a replaced KV pool; reinstall the "
                "platform operator plan before serving"
            )
        for layer_id, layer in enumerate(views.layers):
            k_pool, v_pool = kv_pool.get_kv_buffer(layer_id)
            if not layer.k_pool.matches(k_pool) or not layer.v_pool.matches(v_pool):
                raise RuntimeError(
                    "Qwen3 MLX decode detected replaced KV storage at layer "
                    f"{layer_id}; reinstall the platform operator plan before serving"
                )

    def forward(
        self,
        model: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Any = None,
    ) -> torch.Tensor:
        if self.closed or not self.started:
            raise RuntimeError("Qwen3 MLX provider is not in a serving state")
        if self.model_identity is not None and id(model) != self.model_identity:
            raise RuntimeError(
                "Qwen3 MLX provider was invoked by a replaced model instance"
            )
        kv_pool, req_to_token = self._validate_decode_inputs(
            model,
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors,
        )
        self.max_decode_batch_size = max(
            self.max_decode_batch_size,
            int(input_ids.shape[0]),
        )

        views = self.views
        if views is None:
            # The bridge below fences these refreshed views with the dynamic
            # scheduler tensors before publishing them to the compiled graph.
            views = _build_views(model, kv_pool, synchronize=False)
            self.views = views
        else:
            self._validate_pool_storage(kv_pool, views)

        from sglang.srt.utils.async_probe import (
            maybe_detect_in_closed_range,
            maybe_detect_oob,
        )

        maybe_detect_oob(
            positions,
            0,
            int(views.layers[0].rope_cache.torch_tensor.shape[0]),
            "Qwen3 MLX decode positions",
        )
        maybe_detect_oob(
            forward_batch.req_pool_indices,
            0,
            int(req_to_token.shape[0]),
            "Qwen3 MLX decode request rows",
        )
        maybe_detect_in_closed_range(
            forward_batch.seq_lens,
            1,
            int(req_to_token.shape[1]),
            "Qwen3 MLX decode sequence lengths",
        )
        maybe_detect_oob(
            forward_batch.out_cache_loc,
            0,
            views.pool_slots,
            "Qwen3 MLX deferred KV slots",
        )

        compiled = self._compiled_decode
        needs_capture_refresh = compiled is not None and compiled.owner is not views

        def refresh_after_mps_fence() -> None:
            self._after_mps_fence(views)

        after_mps_fence = (
            refresh_after_mps_fence
            if self._pending_commit_sources is not None or needs_capture_refresh
            else None
        )
        dynamic_inputs = (
            input_ids,
            positions,
            req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
        )
        if compiled is not None and compiled.can_run_hidden(dynamic_inputs):
            graph = compiled
        else:
            if compiled is not None:
                compiled.fallback_count += 1

            def graph(ids, pos, table, req, lens):
                return _mlx_decode_graph(views, ids, pos, table, req, lens)

        hidden_states, new_k, new_v = mlx_call_multi(
            graph,
            *dynamic_inputs,
            device=input_ids.device,
            after_mps_fence=after_mps_fence,
        )
        k_pools = tuple(layer.k_pool.torch_tensor for layer in views.layers)
        v_pools = tuple(layer.v_pool.torch_tensor for layer in views.layers)
        # Publish ownership before launch. If validation or launch raises, the
        # MLX-exported buffers remain alive until close or the next bridge fence.
        self._pending_commit_sources = (new_k, new_v)
        qwen3_commit_deferred_kv(
            new_k,
            new_v,
            forward_batch.out_cache_loc,
            k_pools,
            v_pools,
            backend=self.kv_commit_backend,
        )
        self.call_count += 1
        self.decode_call_count += 1
        return hidden_states


def create_qwen3_mlx_model_provider(
    model: torch.nn.Module,
    kv_pool: Any,
    req_to_token_pool: Any,
    *,
    kv_commit_backend: KernelBackend = KernelBackend.TORCH,
) -> Qwen3MlxModelProvider:
    """Validate, compile, warm, and return an unpublished decode provider."""
    from sglang.srt.models.qwen3 import Qwen3ForCausalLM

    if type(model) is not Qwen3ForCausalLM:
        raise RuntimeError("whole-model MLX supports only Qwen3ForCausalLM")
    qwen3_model = model.model
    views = _build_views(qwen3_model, kv_pool, synchronize=True)
    provider = Qwen3MlxModelProvider(
        views=views,
        model_identity=id(qwen3_model),
        kv_commit_backend=kv_commit_backend,
    )
    try:
        provider.start(req_to_token_pool)
    except Exception:
        provider.close()
        raise
    return provider


__all__ = [
    "MLX_WHOLE_MODEL_CACHE_LIMIT_BYTES",
    "QWEN3_06B_NUM_LAYERS",
    "Qwen3MlxModelProvider",
    "create_qwen3_mlx_model_provider",
    "validate_qwen3_mlx_static_contract",
]
