"""CPU-safe contracts for the per-operation MPS model-op plan."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.kernels.ops.activation import _SILU_AND_MUL
from sglang.kernels.ops.attention.qwen3_mps import _QKNORM_ROPE_STORE, _RADIX_DECODE
from sglang.kernels.ops.kvcache.qwen3 import _DEFERRED_KV_COMMIT
from sglang.kernels.ops.layernorm import _FUSED_ADD_RMSNORM, _RMSNORM
from sglang.kernels.spec import KernelBackend
from sglang.srt.hardware_backend.mps.model_ops.plan import (
    MpsOperatorPlan,
    _configure_generic_mps_ops,
    _whole_model_mlx_fallback_reason,
)
from sglang.srt.hardware_backend.mps.model_ops.router import install_mps_operators
from sglang.srt.hardware_backend.mps.model_ops.selection import (
    MpsGenericOperatorSelection,
    MpsOperatorSelection,
)
from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=2, suite="stage-a-unit-test-mps")


@pytest.fixture(autouse=True)
def _restore_process_local_priorities():
    operators = (
        _QKNORM_ROPE_STORE,
        _RADIX_DECODE,
        _DEFERRED_KV_COMMIT,
        _RMSNORM,
        _FUSED_ADD_RMSNORM,
        _SILU_AND_MUL,
    )
    original = tuple(operator.get_priority() for operator in operators)
    try:
        yield
    finally:
        for operator, priority in zip(operators, original):
            operator.set_priority(priority)


def _model_config(*, supported=True, layers=1, dtype=torch.bfloat16):
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["Qwen3ForCausalLM" if supported else "LlamaForCausalLM"],
            hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=layers,
        ),
        hf_text_config=None,
        dtype=dtype,
        quantization=None,
    )


def _server_args(**overrides):
    values = dict(
        device="mps",
        attention_backend="mps",
        tp_size=1,
        pp_size=1,
        dp_size=1,
        sampling_backend="pytorch",
        enable_fp32_lm_head=False,
        enable_dp_lm_head=False,
        disable_overlap_schedule=True,
        weight_cache_mode="off",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _model():
    model = torch.nn.Module()
    model.model = SimpleNamespace(model_forward_provider=None)
    return model


def _attention():
    return SimpleNamespace(
        op_provider=None,
        attn=SimpleNamespace(decode_provider=None),
    )


class _RejectingDecodeBinding:
    def __init__(self):
        self._decode_provider = None

    @property
    def decode_provider(self):
        return self._decode_provider

    @decode_provider.setter
    def decode_provider(self, value):
        # Simulate a contributor-owned setter that mutates and then fails.
        self._decode_provider = value
        if value is not None:
            raise RuntimeError("decode publication failed")


class _RejectingDecodeBindingAndRestore:
    def __init__(self):
        self._decode_provider = None

    @property
    def decode_provider(self):
        return self._decode_provider

    @decode_provider.setter
    def decode_provider(self, value):
        self._decode_provider = value
        if value is None:
            raise RuntimeError("decode rollback failed")
        raise RuntimeError("decode publication failed")


def _provider():
    provider = mock.MagicMock()
    provider.get_compiled_decode_state.return_value = {
        "enabled": False,
        "total_enabled": True,
        "primary_variant": "greedy",
        "warmup_count": 0,
        "call_count": 0,
        "total_warmup_count": 1,
        "total_call_count": 3,
        "fallback_count": 0,
        "greedy_enabled": True,
        "greedy_warmup_count": 1,
        "greedy_call_count": 3,
    }
    provider.greedy_tail_static_fallback_reason = None
    provider.greedy_tail_backend = "mlx"
    provider.greedy_tail_call_count = 3
    provider.greedy_tail_torch_call_count = 0
    provider.greedy_tail_fallback_count = 0
    provider.last_greedy_tail_fallback_reason = None
    return provider


def _selection(
    *,
    model_forward=("torch",),
    greedy_tail=("torch",),
    qkv=(KernelBackend.TORCH,),
    decode=(KernelBackend.TORCH,),
    commit=(KernelBackend.TORCH,),
    rmsnorm=(KernelBackend.TORCH,),
    fused_add_rmsnorm=(KernelBackend.TORCH,),
    silu_and_mul=(KernelBackend.TORCH,),
):
    return MpsOperatorSelection(
        model_forward=model_forward,
        greedy_tail=greedy_tail,
        qknorm_rope_store=qkv,
        radix_decode=decode,
        deferred_kv_commit=commit,
        rmsnorm=rmsnorm,
        fused_add_rmsnorm=fused_add_rmsnorm,
        silu_and_mul=silu_and_mul,
    )


def _generic_selection(
    *,
    rmsnorm=(KernelBackend.TORCH,),
    fused_add_rmsnorm=(KernelBackend.TORCH,),
    silu_and_mul=(KernelBackend.TORCH,),
):
    return MpsGenericOperatorSelection(
        rmsnorm=rmsnorm,
        fused_add_rmsnorm=fused_add_rmsnorm,
        silu_and_mul=silu_and_mul,
    )


@contextmanager
def _patch_generic_backends(backends):
    with (
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._configure_generic_mps_ops",
            return_value=backends,
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.router._configure_generic_mps_ops",
            return_value=backends,
        ),
    ):
        yield


def _plan_patches(selection):
    generic_backends = {
        "rmsnorm": KernelBackend.TORCH,
        "fused_add_rmsnorm": KernelBackend.TORCH,
        "silu_and_mul": KernelBackend.TORCH,
    }
    return (
        mock.patch.object(MpsOperatorSelection, "from_env", return_value=selection),
        _patch_generic_backends(generic_backends),
    )


def test_unknown_model_keeps_torch_fallback_and_reports_priorities():
    generic_backends = {
        "rmsnorm": KernelBackend.TORCH,
        "fused_add_rmsnorm": KernelBackend.TORCH,
        "silu_and_mul": KernelBackend.TORCH,
    }
    with (
        mock.patch.object(
            MpsGenericOperatorSelection,
            "from_env",
            return_value=_generic_selection(),
        ),
        _patch_generic_backends(generic_backends),
    ):
        plan = install_mps_operators(
            _model(), _model_config(supported=False), _server_args()
        )

    state = plan.get_state()
    assert not state["enabled"]
    assert state["model"] == "LlamaForCausalLM"
    assert state["provider_spec"] is None
    assert set(state["provider_priorities"]) == {
        "rmsnorm",
        "fused_add_rmsnorm",
        "silu_and_mul",
    }
    assert "no registered" in state["qkv_fallback_reason"]
    assert "no registered" in state["decode_fallback_reason"]
    assert state["whole_model_call_count"] == 0
    assert state["whole_model_compile_primary_variant"] == "off"


def test_unknown_model_reports_independently_enabled_generic_ops():
    selection = _generic_selection(
        rmsnorm=(KernelBackend.METAL_JIT, KernelBackend.TORCH),
    )
    generic_backends = {
        "rmsnorm": KernelBackend.METAL_JIT,
        "fused_add_rmsnorm": KernelBackend.TORCH,
        "silu_and_mul": KernelBackend.TORCH,
    }
    with (
        mock.patch.object(
            MpsGenericOperatorSelection, "from_env", return_value=selection
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.router._configure_generic_mps_ops",
            return_value=generic_backends,
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.router.get_fused_op_backend",
            return_value=None,
        ),
    ):
        plan = install_mps_operators(
            _model(), _model_config(supported=False), _server_args()
        )

    state = plan.get_state()
    assert state["provider_spec"] is None
    assert state["enabled"]
    assert state["generic_kernel_backends"] == {
        "rmsnorm": "metal_jit",
        "fused_add_rmsnorm": "torch",
        "silu_and_mul": "torch",
    }


def test_global_torch_force_overrides_generic_priority_without_metal_warmup():
    selection = _selection(
        rmsnorm=(KernelBackend.METAL_JIT, KernelBackend.TORCH),
        fused_add_rmsnorm=(KernelBackend.METAL_JIT, KernelBackend.TORCH),
        silu_and_mul=(KernelBackend.METAL_JIT, KernelBackend.TORCH),
    )
    with (
        mock.patch(
            "sglang.kernels.ops.layernorm._rmsnorm_metal_jit.warmup_mps_rmsnorm_kernels"
        ) as warmup_norm,
        mock.patch(
            "sglang.kernels.ops.activation._silu_and_mul_metal_jit."
            "warmup_silu_and_mul_metal_kernel"
        ) as warmup_silu,
    ):
        selected = _configure_generic_mps_ops(selection, KernelBackend.TORCH)

    assert selected == {
        "rmsnorm": KernelBackend.TORCH,
        "fused_add_rmsnorm": KernelBackend.TORCH,
        "silu_and_mul": KernelBackend.TORCH,
    }
    warmup_norm.assert_not_called()
    warmup_silu.assert_not_called()


def test_global_metal_jit_force_warms_all_generic_ops_and_reports_it():
    selection = _selection()
    with (
        mock.patch(
            "sglang.kernels.ops.layernorm._rmsnorm_metal_jit.warmup_mps_rmsnorm_kernels"
        ) as warmup_norm,
        mock.patch(
            "sglang.kernels.ops.activation._silu_and_mul_metal_jit."
            "warmup_silu_and_mul_metal_kernel"
        ) as warmup_silu,
    ):
        selected = _configure_generic_mps_ops(selection, KernelBackend.METAL_JIT)

    assert selected == {
        "rmsnorm": KernelBackend.METAL_JIT,
        "fused_add_rmsnorm": KernelBackend.METAL_JIT,
        "silu_and_mul": KernelBackend.METAL_JIT,
    }
    warmup_norm.assert_called_once_with(rmsnorm=True, fused_add_rmsnorm=True)
    warmup_silu.assert_called_once_with()


def test_global_force_rejects_backend_not_implemented_by_generic_ops():
    with pytest.raises(RuntimeError, match="do not support.*metal_aot"):
        _configure_generic_mps_ops(_selection(), KernelBackend.METAL_AOT)


def test_qkv_and_decode_can_select_different_providers_atomically():
    selection = _selection(
        qkv=(KernelBackend.METAL_AOT, KernelBackend.TORCH),
        decode=(KernelBackend.METAL_JIT, KernelBackend.TORCH),
    )
    model = _model()
    attention = _attention()
    selection_patch, generic_patch = _plan_patches(selection)
    with (
        selection_patch,
        generic_patch,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._qwen3_modules",
            return_value=[attention],
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.validate_qwen3_qkv_module"
        ) as validate_qkv,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.validate_qwen3_decode_module"
        ) as validate_decode,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._validate_kv_pool_contract",
            return_value=[mock.Mock()],
        ) as validate_pool,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.warmup_qwen3_mps_provider"
        ) as warmup,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.get_fused_op_backend",
            return_value=None,
        ),
        mock.patch(
            "sglang.kernels.ops.attention.qwen3_mps.is_qwen3_metal_aot_available",
            return_value=True,
        ),
    ):
        plan = install_mps_operators(
            model,
            _model_config(),
            _server_args(),
            req_to_token_pool=object(),
            token_to_kv_pool=object(),
        )

    validate_qkv.assert_called_once_with(attention)
    validate_decode.assert_called_once_with(attention)
    validate_pool.assert_called_once()
    warmup.assert_called_once_with(KernelBackend.METAL_AOT, KernelBackend.METAL_JIT)
    provider = attention.op_provider
    assert provider is attention.attn.decode_provider
    assert provider.qkv_kernel_backend is KernelBackend.METAL_AOT
    assert provider.decode_kernel_backend is KernelBackend.METAL_JIT
    state = plan.get_state()
    assert state["qkv_kernel_backend"] == "metal_aot"
    assert state["decode_kernel_backend"] == "metal_jit"
    assert state["patched_qkv_modules"] == 1
    assert state["patched_decode_modules"] == 1


def test_torch_native_decode_demotes_only_decode_provider_and_reports_reason():
    selection = _selection(
        qkv=(KernelBackend.METAL_AOT, KernelBackend.TORCH),
        decode=(KernelBackend.METAL_JIT, KernelBackend.TORCH),
    )
    model = _model()
    attention = _attention()
    selection_patch, generic_patch = _plan_patches(selection)
    with (
        selection_patch,
        generic_patch,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._qwen3_modules",
            return_value=[attention],
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.validate_qwen3_qkv_module"
        ) as validate_qkv,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.validate_qwen3_decode_module"
        ) as validate_decode,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._validate_kv_pool_contract",
            return_value=[mock.Mock()],
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.warmup_qwen3_mps_provider"
        ) as warmup,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.get_fused_op_backend",
            return_value=None,
        ),
        mock.patch(
            "sglang.kernels.ops.attention.qwen3_mps.is_qwen3_metal_aot_available",
            return_value=True,
        ),
    ):
        plan = install_mps_operators(
            model,
            _model_config(),
            _server_args(decode_attention_backend="torch_native"),
            token_to_kv_pool=object(),
        )

    validate_qkv.assert_called_once_with(attention)
    validate_decode.assert_not_called()
    warmup.assert_called_once_with(KernelBackend.METAL_AOT, KernelBackend.TORCH)
    assert attention.op_provider.qkv_kernel_backend is KernelBackend.METAL_AOT
    assert attention.attn.decode_provider is None
    state = plan.get_state()
    assert state["attention_backend"] == "prefill=mps,decode=torch_native"
    assert state["qkv_kernel_backend"] == "metal_aot"
    assert state["decode_kernel_backend"] == "torch"
    assert state["patched_qkv_modules"] == 1
    assert state["patched_decode_modules"] == 0
    assert "does not consume" in state["decode_fallback_reason"]


def test_torch_native_decode_rejects_strict_global_metal_force():
    selection = _selection()
    selection_patch, generic_patch = _plan_patches(selection)
    with (
        selection_patch,
        generic_patch,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._qwen3_modules"
        ) as discover,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.get_fused_op_backend",
            return_value=KernelBackend.METAL_JIT,
        ),
        pytest.raises(
            RuntimeError,
            match="SGLANG_FORCE_FUSED_OP_BACKEND cannot be honored.*torch_native",
        ),
    ):
        install_mps_operators(
            _model(),
            _model_config(),
            _server_args(decode_attention_backend="torch_native"),
            token_to_kv_pool=object(),
        )

    discover.assert_not_called()


def test_decode_only_does_not_bind_qkv_or_validate_qkv_contract():
    selection = _selection(decode=(KernelBackend.METAL_JIT, KernelBackend.TORCH))
    model = _model()
    attention = _attention()
    selection_patch, generic_patch = _plan_patches(selection)
    with (
        selection_patch,
        generic_patch,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._qwen3_modules",
            return_value=[attention],
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.validate_qwen3_qkv_module"
        ) as validate_qkv,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.validate_qwen3_decode_module"
        ) as validate_decode,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._validate_kv_pool_contract",
            return_value=[mock.Mock()],
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.warmup_qwen3_mps_provider"
        ) as warmup,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.get_fused_op_backend",
            return_value=None,
        ),
    ):
        plan = install_mps_operators(
            model,
            _model_config(),
            _server_args(),
            token_to_kv_pool=object(),
        )

    validate_qkv.assert_not_called()
    validate_decode.assert_called_once_with(attention)
    warmup.assert_called_once_with(KernelBackend.TORCH, KernelBackend.METAL_JIT)
    assert attention.op_provider is None
    assert (
        attention.attn.decode_provider.decode_kernel_backend is KernelBackend.METAL_JIT
    )
    assert plan.get_state()["patched_qkv_modules"] == 0


def test_qkv_contract_miss_preserves_independent_decode_provider():
    selection = _selection(
        qkv=(KernelBackend.METAL_JIT, KernelBackend.TORCH),
        decode=(KernelBackend.METAL_JIT, KernelBackend.TORCH),
    )
    model = _model()
    attention = _attention()
    selection_patch, generic_patch = _plan_patches(selection)
    with (
        selection_patch,
        generic_patch,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._qwen3_modules",
            return_value=[attention],
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.validate_qwen3_qkv_module",
            side_effect=RuntimeError("unsupported rope contract"),
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.validate_qwen3_decode_module"
        ) as validate_decode,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._validate_kv_pool_contract",
            return_value=[mock.Mock()],
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.warmup_qwen3_mps_provider"
        ) as warmup,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.get_fused_op_backend",
            return_value=None,
        ),
    ):
        plan = install_mps_operators(
            model,
            _model_config(),
            _server_args(),
            token_to_kv_pool=object(),
        )

    validate_decode.assert_called_once_with(attention)
    warmup.assert_called_once_with(KernelBackend.TORCH, KernelBackend.METAL_JIT)
    assert attention.op_provider is None
    assert attention.attn.decode_provider is not None
    state = plan.get_state()
    assert state["qkv_kernel_backend"] == "torch"
    assert state["decode_kernel_backend"] == "metal_jit"
    assert state["qkv_fallback_reason"] == "unsupported rope contract"
    assert state["decode_fallback_reason"] is None


def test_aot_absence_falls_through_to_explicit_jit_priority():
    selection = _selection(
        qkv=(
            KernelBackend.METAL_AOT,
            KernelBackend.METAL_JIT,
            KernelBackend.TORCH,
        )
    )
    model = _model()
    attention = _attention()
    selection_patch, generic_patch = _plan_patches(selection)
    with (
        selection_patch,
        generic_patch,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._qwen3_modules",
            return_value=[attention],
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.validate_qwen3_qkv_module"
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._validate_kv_pool_contract",
            return_value=[mock.Mock()],
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.warmup_qwen3_mps_provider"
        ) as warmup,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.get_fused_op_backend",
            return_value=None,
        ),
        mock.patch(
            "sglang.kernels.ops.attention.qwen3_mps.is_qwen3_metal_aot_available",
            return_value=False,
        ),
    ):
        plan = install_mps_operators(
            model,
            _model_config(),
            _server_args(),
            token_to_kv_pool=object(),
        )

    warmup.assert_called_once_with(KernelBackend.METAL_JIT, KernelBackend.TORCH)
    assert plan.qkv_kernel_backend == "metal_jit"


def test_whole_model_mlx_gate_passes_independent_commit_backend():
    selection = _selection(
        model_forward=("mlx", "torch"),
        greedy_tail=("mlx", "torch"),
        commit=(KernelBackend.METAL_JIT, KernelBackend.TORCH),
    )
    model = _model()
    whole_model = _provider()
    req_pool = object()
    kv_pool = object()
    selection_patch, generic_patch = _plan_patches(selection)
    with (
        selection_patch,
        generic_patch,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.get_fused_op_backend",
            return_value=None,
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx."
            "validate_qwen3_mlx_static_contract"
        ) as validate_mlx,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.qwen3_mlx."
            "create_qwen3_mlx_model_provider",
            return_value=whole_model,
        ) as create_mlx,
    ):
        plan = install_mps_operators(
            model,
            _model_config(),
            _server_args(),
            req_to_token_pool=req_pool,
            token_to_kv_pool=kv_pool,
        )

    validate_mlx.assert_called_once_with(model, kv_pool, req_pool)
    create_mlx.assert_called_once_with(
        model,
        kv_pool,
        req_pool,
        kv_commit_backend=KernelBackend.METAL_JIT,
        server_args=_server_args(),
        greedy_tail_backend="mlx",
    )
    assert model.model.model_forward_provider is whole_model
    state = plan.get_state()
    assert state["whole_model_backend"] == "mlx"
    assert state["whole_model_greedy_tail_enabled"]
    assert state["whole_model_greedy_tail_backend"] == "mlx"
    assert state["whole_model_greedy_tail_call_count"] == 3
    assert state["whole_model_greedy_tail_torch_call_count"] == 0
    assert state["whole_model_compile_call_count"] == 0
    assert state["whole_model_compile_total_enabled"]
    assert state["whole_model_compile_primary_variant"] == "greedy"
    assert state["whole_model_compile_total_call_count"] == 3
    assert state["whole_model_greedy_compile_enabled"]
    assert state["whole_model_greedy_compile_call_count"] == 3


def test_default_torch_priority_does_not_discover_or_warm_attention():
    selection = _selection()
    selection_patch, generic_patch = _plan_patches(selection)
    with (
        selection_patch,
        generic_patch,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._qwen3_modules"
        ) as discover,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.warmup_qwen3_mps_provider"
        ) as warmup,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.get_fused_op_backend",
            return_value=None,
        ),
    ):
        plan = install_mps_operators(_model(), _model_config(), _server_args())

    discover.assert_not_called()
    warmup.assert_not_called()
    state = plan.get_state()
    assert state["provider_spec"] == "qwen3_dense"
    assert not state["enabled"]
    assert state["qkv_kernel_backend"] == "torch"
    assert state["decode_kernel_backend"] == "torch"
    assert state["whole_model_backend"] == "torch"


def test_selected_metal_compile_failure_is_fatal_and_atomic():
    selection = _selection(qkv=(KernelBackend.METAL_JIT, KernelBackend.TORCH))
    model = _model()
    attention = _attention()
    selection_patch, generic_patch = _plan_patches(selection)
    with (
        selection_patch,
        generic_patch,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._qwen3_modules",
            return_value=[attention],
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.validate_qwen3_qkv_module"
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._validate_kv_pool_contract",
            return_value=[mock.Mock()],
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.warmup_qwen3_mps_provider",
            side_effect=RuntimeError("shader compile failed"),
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.get_fused_op_backend",
            return_value=None,
        ),
        pytest.raises(RuntimeError, match="shader compile failed"),
    ):
        install_mps_operators(
            model,
            _model_config(),
            _server_args(),
            token_to_kv_pool=object(),
        )

    assert attention.op_provider is None
    assert attention.attn.decode_provider is None
    assert model.model.model_forward_provider is None


def test_provider_publication_failure_rolls_back_earlier_bindings():
    selection = _selection(
        qkv=(KernelBackend.METAL_JIT, KernelBackend.TORCH),
        decode=(KernelBackend.METAL_JIT, KernelBackend.TORCH),
    )
    model = _model()
    attention = SimpleNamespace(
        op_provider=None,
        attn=_RejectingDecodeBinding(),
    )
    selection_patch, generic_patch = _plan_patches(selection)
    with (
        selection_patch,
        generic_patch,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._qwen3_modules",
            return_value=[attention],
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.validate_qwen3_qkv_module"
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.validate_qwen3_decode_module"
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._validate_kv_pool_contract",
            return_value=[mock.Mock()],
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.warmup_qwen3_mps_provider"
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.get_fused_op_backend",
            return_value=None,
        ),
        pytest.raises(RuntimeError, match="decode publication failed"),
    ):
        install_mps_operators(
            model,
            _model_config(),
            _server_args(),
            token_to_kv_pool=object(),
        )

    assert attention.op_provider is None
    assert attention.attn.decode_provider is None
    assert model.model.model_forward_provider is None


def test_provider_rollback_is_best_effort_and_preserves_publication_error():
    selection = _selection(
        qkv=(KernelBackend.METAL_JIT, KernelBackend.TORCH),
        decode=(KernelBackend.METAL_JIT, KernelBackend.TORCH),
    )
    model = _model()
    attention = SimpleNamespace(
        op_provider=None,
        attn=_RejectingDecodeBindingAndRestore(),
    )
    selection_patch, generic_patch = _plan_patches(selection)
    with (
        selection_patch,
        generic_patch,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._qwen3_modules",
            return_value=[attention],
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.validate_qwen3_qkv_module"
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.validate_qwen3_decode_module"
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._validate_kv_pool_contract",
            return_value=[mock.Mock()],
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.warmup_qwen3_mps_provider"
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.get_fused_op_backend",
            return_value=None,
        ),
        pytest.raises(RuntimeError, match="decode publication failed"),
    ):
        install_mps_operators(
            model,
            _model_config(),
            _server_args(),
            token_to_kv_pool=object(),
        )

    # Cleanup continues after a restoration setter fails, and the original
    # publication error remains the user-facing startup failure.
    assert attention.op_provider is None
    assert attention.attn.decode_provider is None
    assert model.model.model_forward_provider is None


def test_global_metal_force_makes_static_contract_miss_fatal():
    selection = _selection()
    model = _model()
    attention = _attention()
    selection_patch, generic_patch = _plan_patches(selection)
    with (
        selection_patch,
        generic_patch,
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan._qwen3_modules",
            return_value=[attention],
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.validate_qwen3_qkv_module",
            side_effect=RuntimeError("unsupported QKV layout"),
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mps.model_ops.plan.get_fused_op_backend",
            return_value=KernelBackend.METAL_JIT,
        ),
        pytest.raises(
            RuntimeError,
            match="SGLANG_FORCE_FUSED_OP_BACKEND.*QKV Metal contract",
        ),
    ):
        install_mps_operators(
            model,
            _model_config(),
            _server_args(),
            token_to_kv_pool=object(),
        )

    assert attention.op_provider is None
    assert attention.attn.decode_provider is None
    assert model.model.model_forward_provider is None


def test_dumper_keeps_whole_model_execution_on_torch_modules():
    with mock.patch(
        "sglang.srt.debug_utils.dumper.dumper",
        SimpleNamespace(may_enable=True),
    ):
        reason = _whole_model_mlx_fallback_reason(
            _model_config(),
            _server_args(),
            forced_backend=None,
        )

    assert "Dumper observability" in reason


def test_plan_close_is_idempotent_and_preserves_replacement_bindings():
    model = _model()
    old_attention = mock.MagicMock()
    old_model_provider = _provider()
    plan = MpsOperatorPlan(
        enabled=True,
        _model=model,
        _attention_bindings=((old_attention, old_attention),),
        _model_forward_provider=old_model_provider,
    )
    replacement = object()
    model.model.model_forward_provider = replacement
    old_attention.op_provider = replacement
    old_attention.attn.decode_provider = replacement

    plan.close()
    plan.close()

    old_model_provider.close.assert_called_once_with()
    assert model.model.model_forward_provider is replacement
    assert old_attention.op_provider is replacement
    assert old_attention.attn.decode_provider is replacement
    assert not plan.get_state()["enabled"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
