"""CPU-safe lifecycle tests for the Qwen3 Metal attention plan."""

from contextlib import ExitStack, contextmanager
from importlib.util import find_spec
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.kernels.spec import KernelBackend
from sglang.srt.hardware_backend.mps.model_ops.plan import (
    Qwen3MetalAttentionPlan,
    install_qwen3_metal_attention,
)
from sglang.srt.hardware_backend.mps.model_ops.registry import (
    MPS_MODEL_OPERATOR_REGISTRY,
)
from sglang.srt.hardware_backend.mps.model_ops.selection import (
    Qwen3MetalAttentionSelection,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_mps_ci(est_time=2, suite="stage-a-unit-test-mps")


def _model_config(*, supported: bool = True, layers: int = 1):
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["Qwen3ForCausalLM" if supported else "LlamaForCausalLM"],
            hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=layers,
        ),
        hf_text_config=None,
    )


def _server_args(**overrides):
    values = dict(
        device="mps",
        attention_backend="mps",
        prefill_attention_backend=None,
        decode_attention_backend=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _selection(
    *,
    qkv=(KernelBackend.TORCH,),
    decode=(KernelBackend.TORCH,),
):
    return Qwen3MetalAttentionSelection(
        qknorm_rope_store=qkv,
        radix_decode=decode,
    )


def _model():
    return torch.nn.Module()


def _attention():
    return SimpleNamespace(
        op_provider=None,
        attn=SimpleNamespace(decode_provider=None),
    )


def test_qwen3_installer_is_selected_by_the_model_neutral_registry():
    spec = MPS_MODEL_OPERATOR_REGISTRY.resolve(_model_config())

    assert spec is not None
    assert spec.name == "qwen3_dense_attention"
    assert spec.installer_path.endswith(":install_qwen3_metal_attention")


@contextmanager
def _install_patches(selection, attention=None, *, aot=True, jit=True):
    patches = [
        mock.patch.object(
            Qwen3MetalAttentionSelection, "from_env", return_value=selection
        ),
        mock.patch(
            "sglang.kernels.ops.attention.qwen3_mps.is_qwen3_metal_aot_available",
            return_value=aot,
        ),
        mock.patch(
            "sglang.kernels.metal.is_metal_jit_available",
            return_value=jit,
        ),
    ]
    if attention is not None:
        patches.extend(
            [
                mock.patch(
                    "sglang.srt.hardware_backend.mps.model_ops.plan._qwen3_modules",
                    return_value=[attention],
                ),
                mock.patch(
                    "sglang.srt.hardware_backend.mps.model_ops.plan."
                    "_validate_kv_pool_contract",
                    return_value=[mock.Mock()],
                ),
                mock.patch(
                    "sglang.kernels.fused_op.get_fused_op_backend",
                    return_value=None,
                ),
            ]
        )
    with ExitStack() as stack:
        for patcher in patches:
            stack.enter_context(patcher)
        yield


def test_default_torch_gates_do_not_discover_or_warm_providers():
    selection = _selection()
    with (
        _install_patches(selection),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._qwen3_modules"
        ) as discover,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.warmup_qwen3_mps_provider"
        ) as warmup,
    ):
        plan = install_qwen3_metal_attention(_model(), _model_config(), _server_args())

    discover.assert_not_called()
    warmup.assert_not_called()
    assert not plan.enabled
    assert plan.qkv_kernel_backend == "torch"
    assert plan.decode_kernel_backend == "torch"


@pytest.mark.skipif(
    find_spec("mlx") is None, reason="requires the optional MLX runtime"
)
def test_whole_model_mlx_provider_is_published_after_static_validation():
    selection = Qwen3MetalAttentionSelection(
        qknorm_rope_store=(KernelBackend.TORCH,),
        radix_decode=(KernelBackend.TORCH,),
        model_forward=("mlx", "torch"),
        deferred_kv_commit=(KernelBackend.TORCH,),
    )
    model = SimpleNamespace(
        model=SimpleNamespace(model_forward_provider=None),
    )
    provider = mock.Mock(
        call_count=0,
        decode_call_count=0,
        max_decode_batch_size=0,
        selector_call_count=0,
        selector_fallback_count=0,
        last_selector_fallback_reason=None,
        get_compiled_decode_state=lambda: {
            "enabled": True,
            "warmup_count": 1,
            "call_count": 0,
            "fallback_count": 0,
        },
    )

    with (
        _install_patches(selection),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.get_fused_op_backend",
            return_value=None,
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._whole_model_mlx_fallback_reason",
            return_value=None,
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx.validate_qwen3_mlx_static_contract"
        ) as validate,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx.create_qwen3_mlx_model_provider",
            return_value=provider,
        ) as create,
    ):
        plan = install_qwen3_metal_attention(
            model,
            _model_config(layers=28),
            _server_args(disable_overlap_schedule=True),
            req_to_token_pool=object(),
            token_to_kv_pool=object(),
        )

    validate.assert_called_once()
    create.assert_called_once()
    assert model.model.model_forward_provider is provider
    assert plan.whole_model_backend == "mlx"
    assert plan.deferred_kv_commit_backend == "torch"
    assert plan.get_state()["whole_model_compile_enabled"]

    plan.close()
    provider.close.assert_called_once_with()
    assert model.model.model_forward_provider is None


def test_qkv_and_decode_select_independent_providers_atomically():
    selection = _selection(
        qkv=(KernelBackend.METAL_AOT, KernelBackend.TORCH),
        decode=(KernelBackend.METAL_JIT, KernelBackend.TORCH),
    )
    attention = _attention()
    with (
        _install_patches(selection, attention),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.validate_qwen3_qkv_module"
        ) as validate_qkv,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan."
            "validate_qwen3_decode_module"
        ) as validate_decode,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.warmup_qwen3_mps_provider"
        ) as warmup,
    ):
        plan = install_qwen3_metal_attention(
            _model(),
            _model_config(),
            _server_args(),
            token_to_kv_pool=object(),
        )

    validate_qkv.assert_called_once_with(attention)
    validate_decode.assert_called_once_with(attention)
    warmup.assert_called_once_with(KernelBackend.METAL_AOT, KernelBackend.METAL_JIT)
    assert attention.op_provider is attention.attn.decode_provider
    assert plan.qkv_kernel_backend == "metal_aot"
    assert plan.decode_kernel_backend == "metal_jit"
    assert plan.enabled


def test_qkv_contract_miss_preserves_decode_provider():
    selection = _selection(
        qkv=(KernelBackend.METAL_JIT, KernelBackend.TORCH),
        decode=(KernelBackend.METAL_JIT, KernelBackend.TORCH),
    )
    attention = _attention()
    with (
        _install_patches(selection, attention),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.validate_qwen3_qkv_module",
            side_effect=RuntimeError("unsupported rope contract"),
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan."
            "validate_qwen3_decode_module"
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.warmup_qwen3_mps_provider"
        ) as warmup,
    ):
        plan = install_qwen3_metal_attention(
            _model(),
            _model_config(),
            _server_args(),
            token_to_kv_pool=object(),
        )

    warmup.assert_called_once_with(KernelBackend.TORCH, KernelBackend.METAL_JIT)
    assert attention.op_provider is None
    assert attention.attn.decode_provider is not None
    assert plan.qkv_kernel_backend == "torch"
    assert plan.decode_kernel_backend == "metal_jit"
    assert plan.qkv_fallback_reason == "unsupported rope contract"


def test_selected_compile_failure_is_fatal_before_publication():
    selection = _selection(
        qkv=(KernelBackend.METAL_JIT, KernelBackend.TORCH),
    )
    attention = _attention()
    with (
        _install_patches(selection, attention),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.validate_qwen3_qkv_module"
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.warmup_qwen3_mps_provider",
            side_effect=RuntimeError("shader compile failed"),
        ),
        pytest.raises(RuntimeError, match="shader compile failed"),
    ):
        install_qwen3_metal_attention(
            _model(),
            _model_config(),
            _server_args(),
            token_to_kv_pool=object(),
        )

    assert attention.op_provider is None
    assert attention.attn.decode_provider is None


def test_close_is_idempotent_and_does_not_clear_replacement_bindings():
    attention = _attention()
    provider = mock.Mock(
        qkv_kernel_backend=KernelBackend.METAL_JIT,
        decode_kernel_backend=KernelBackend.METAL_JIT,
    )
    plan = Qwen3MetalAttentionPlan(
        qkv_kernel_backend="metal_jit",
        decode_kernel_backend="metal_jit",
        _bindings=((attention, provider),),
    )
    replacement = object()
    attention.op_provider = replacement
    attention.attn.decode_provider = replacement

    plan.close()
    plan.close()

    assert attention.op_provider is replacement
    assert attention.attn.decode_provider is replacement
    assert not plan.enabled


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
