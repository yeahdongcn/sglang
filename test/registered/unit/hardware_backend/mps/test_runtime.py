"""Tests for the single Torch-owned MPS runtime gate."""

import importlib.util
import subprocess
import sys
import types
import unittest
from importlib.metadata import PackageNotFoundError, version
from unittest import mock

import torch
from packaging.version import Version

from sglang.srt.hardware_backend.mps import runtime
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")


def _fake_mlx(version: str | None, *, metal_available: bool = True):
    fake_mlx = types.ModuleType("mlx")
    fake_core = types.ModuleType("mlx.core")
    if version is not None:
        fake_core.__version__ = version
    fake_core.metal = types.SimpleNamespace(
        is_available=lambda: metal_available,
    )
    fake_mlx.core = fake_core
    return fake_mlx, fake_core


def _has_stable_distribution_at_least(distribution: str, minimum: Version) -> bool:
    try:
        installed = Version(version(distribution))
    except (PackageNotFoundError, ValueError):
        return False
    return not installed.is_prerelease and installed >= minimum


class TestMpsRuntime(unittest.TestCase):
    def tearDown(self):
        runtime.validate_mps_runtime.cache_clear()

    def test_weight_update_device_fallback_without_current_device(self):
        from sglang.srt.model_executor.model_runner_components.weight_updater import (
            _get_weight_update_device,
        )

        fake_device_module = types.SimpleNamespace()
        with mock.patch.object(
            torch, "get_device_module", return_value=fake_device_module
        ):
            self.assertEqual(_get_weight_update_device("mps"), torch.device("mps"))

    def test_weight_update_device_keeps_cuda_current_device(self):
        from sglang.srt.model_executor.model_runner_components.weight_updater import (
            _get_weight_update_device,
        )

        fake_device_module = types.SimpleNamespace(current_device=lambda: 3)
        with mock.patch.object(
            torch, "get_device_module", return_value=fake_device_module
        ):
            self.assertEqual(_get_weight_update_device("cuda"), 3)

    def test_non_mps_server_does_not_import_mlx(self):
        script = """
import sys
from sglang.srt.server_args import ServerArgs
ServerArgs(model_path="dummy", device="cpu")
assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )

    def test_version_helpers(self):
        self.assertTrue(runtime._is_stable_series("0.32.7", (0, 32)))
        self.assertFalse(runtime._is_stable_series("0.31.9", (0, 32)))
        self.assertFalse(runtime._is_stable_series("0.33.0", (0, 32)))
        self.assertFalse(runtime._is_stable_series("0.32.1rc1", (0, 32)))
        self.assertFalse(runtime._is_stable_series("0.32.1.dev1", (0, 32)))
        self.assertFalse(runtime._is_stable_series("unknown", (0, 32)))
        self.assertTrue(runtime._is_stable_at_least("0.32.0", runtime._MIN_MLX_VERSION))
        self.assertTrue(runtime._is_stable_at_least("0.33.0", runtime._MIN_MLX_VERSION))
        self.assertTrue(runtime._is_stable_at_least("1.0.0", runtime._MIN_MLX_VERSION))
        self.assertFalse(
            runtime._is_stable_at_least("0.31.9", runtime._MIN_MLX_VERSION)
        )
        self.assertFalse(
            runtime._is_stable_at_least("0.33.0rc1", runtime._MIN_MLX_VERSION)
        )
        self.assertFalse(
            runtime._is_stable_at_least("unknown", runtime._MIN_MLX_VERSION)
        )

    def test_effective_model_quantization_is_validated(self):
        for quantization in (None, "unquant"):
            with self.subTest(quantization=quantization):
                self.assertIsNone(
                    runtime.validate_mps_model_config(
                        types.SimpleNamespace(
                            quantization=quantization,
                            is_multimodal=False,
                        )
                    )
                )

        with self.assertRaisesRegex(ValueError, "detected quantization='awq'"):
            runtime.validate_mps_model_config(types.SimpleNamespace(quantization="awq"))

        with self.assertRaisesRegex(ValueError, "multimodal serving"):
            runtime.validate_mps_model_config(
                types.SimpleNamespace(quantization=None, is_multimodal=True)
            )

        with self.assertRaisesRegex(ValueError, "supports dense models only"):
            runtime.validate_mps_model_config(
                types.SimpleNamespace(
                    quantization=None,
                    is_multimodal=False,
                    hf_text_config=types.SimpleNamespace(num_experts=8),
                    hf_config=None,
                ),
                lora_enabled=True,
            )

    def test_server_validates_checkpoint_derived_quantization(self):
        from sglang.srt.server_args import ServerArgs

        model_config = types.SimpleNamespace(quantization="awq")
        args = types.SimpleNamespace(get_model_config=lambda: model_config)
        with self.assertRaisesRegex(ValueError, "detected quantization='awq'"):
            ServerArgs._validate_mps_resolved_model_config(args)

    def test_platform_rejects_effective_quantization_before_installer(self):
        from sglang.srt.platforms.mps import MpsSRTPlatform

        model_config = types.SimpleNamespace(
            quantization="awq",
            is_multimodal=False,
        )
        with (
            mock.patch(
                "sglang.srt.hardware_backend.mps.model_ops.router.install_mps_operators"
            ) as install,
            self.assertRaisesRegex(ValueError, "detected quantization='awq'"),
        ):
            MpsSRTPlatform().bind_model_runtime_operators(
                model=object(),
                model_config=model_config,
                server_args=types.SimpleNamespace(enable_lora=False),
                req_to_token_pool=object(),
                token_to_kv_pool=object(),
            )
        install.assert_not_called()

    def test_unvalidated_runtime_pairs_are_rejected(self):
        cases = (
            ("2.12.1", "0.32.0", "stable Torch 2.13.x"),
            ("2.13.0", "0.31.0", "MLX >= 0.32.0"),
            ("2.14.0", "0.32.0", "stable Torch 2.13.x"),
            ("2.13.1rc1", "0.32.0", "stable Torch 2.13.x"),
            ("2.13.0", "0.32.1.dev1", "MLX >= 0.32.0"),
        )
        for torch_version, mlx_version, message in cases:
            with self.subTest(torch=torch_version, mlx=mlx_version):
                fake_mlx, fake_core = _fake_mlx(mlx_version)
                runtime.validate_mps_runtime.cache_clear()
                with (
                    mock.patch.dict(
                        sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}
                    ),
                    mock.patch.object(torch, "__version__", torch_version),
                    mock.patch.object(
                        torch.backends.mps, "is_available", return_value=True
                    ),
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    runtime.validate_mps_runtime()

    def test_missing_mlx_version_has_an_actionable_error(self):
        fake_mlx, fake_core = _fake_mlx(None)
        with (
            mock.patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}),
            mock.patch.object(torch, "__version__", "2.13.0"),
            mock.patch.object(torch.backends.mps, "is_available", return_value=True),
            self.assertRaisesRegex(RuntimeError, "MLX unknown"),
        ):
            runtime.validate_mps_runtime()

    def test_missing_mlx_has_an_actionable_error(self):
        with (
            mock.patch.dict(sys.modules, {"mlx": None, "mlx.core": None}),
            self.assertRaisesRegex(RuntimeError, "MLX is not installed"),
        ):
            runtime.validate_mps_runtime()

    def test_unavailable_metal_devices_have_actionable_errors(self):
        cases = (
            (False, True, "PyTorch MPS device"),
            (True, False, "MLX Metal device"),
        )
        for torch_mps_available, mlx_metal_available, message in cases:
            with self.subTest(message=message):
                fake_mlx, fake_core = _fake_mlx(
                    "0.32.0", metal_available=mlx_metal_available
                )
                runtime.validate_mps_runtime.cache_clear()
                with (
                    mock.patch.dict(
                        sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}
                    ),
                    mock.patch.object(torch, "__version__", "2.13.0"),
                    mock.patch.object(
                        torch.backends.mps,
                        "is_available",
                        return_value=torch_mps_available,
                    ),
                    mock.patch.object(
                        torch.mps, "compile_shader", mock.Mock(), create=True
                    ),
                    mock.patch.object(
                        torch.mps, "load_metallib", mock.Mock(), create=True
                    ),
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    runtime.validate_mps_runtime()

    def test_missing_torch_metal_shader_compiler_has_an_actionable_error(self):
        fake_mlx, fake_core = _fake_mlx("0.32.0")
        with (
            mock.patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}),
            mock.patch.object(torch, "__version__", "2.13.0"),
            mock.patch.object(torch.backends.mps, "is_available", return_value=True),
            mock.patch.object(torch.mps, "compile_shader", None, create=True),
            self.assertRaisesRegex(RuntimeError, "torch.mps.compile_shader"),
        ):
            runtime.validate_mps_runtime()

    def test_missing_torch_metallib_loader_has_an_actionable_error(self):
        fake_mlx, fake_core = _fake_mlx("0.32.0")
        with (
            mock.patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}),
            mock.patch.object(torch, "__version__", "2.13.0"),
            mock.patch.object(torch.backends.mps, "is_available", return_value=True),
            mock.patch.object(torch.mps, "compile_shader", mock.Mock(), create=True),
            mock.patch.object(torch.mps, "load_metallib", None, create=True),
            self.assertRaisesRegex(RuntimeError, "torch.mps.load_metallib"),
        ):
            runtime.validate_mps_runtime()

    def test_missing_torch_mps_memory_api_has_an_actionable_error(self):
        for memory_api in ("recommended_max_memory", "driver_allocated_memory"):
            with self.subTest(memory_api=memory_api):
                fake_mlx, fake_core = _fake_mlx("0.32.0")
                runtime.validate_mps_runtime.cache_clear()
                with (
                    mock.patch.dict(
                        sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}
                    ),
                    mock.patch.object(torch, "__version__", "2.13.0"),
                    mock.patch.object(
                        torch.backends.mps, "is_available", return_value=True
                    ),
                    mock.patch.object(
                        torch.mps, "compile_shader", mock.Mock(), create=True
                    ),
                    mock.patch.object(
                        torch.mps, "load_metallib", mock.Mock(), create=True
                    ),
                    mock.patch.object(torch.mps, memory_api, None, create=True),
                    self.assertRaisesRegex(RuntimeError, memory_api),
                ):
                    runtime.validate_mps_runtime()

    def test_validated_runtime_accepts_torch_patches_and_newer_stable_mlx(self):
        fake_mlx, fake_core = _fake_mlx("0.33.0")
        with (
            mock.patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}),
            mock.patch.object(torch, "__version__", "2.13.7"),
            mock.patch.object(torch.backends.mps, "is_available", return_value=True),
            mock.patch.object(torch.mps, "compile_shader", mock.Mock(), create=True),
            mock.patch.object(torch.mps, "load_metallib", mock.Mock(), create=True),
        ):
            self.assertIsNone(runtime.validate_mps_runtime())

    def test_incompatible_runtime_aborts_explicit_mps_before_dummy_shortcut(self):
        from sglang.srt.server_args import ServerArgs

        # Importing the platform may validate the real local runtime once.
        # Exercise resolution against the patched version pair below instead.
        runtime.validate_mps_runtime.cache_clear()
        fake_mlx, fake_core = _fake_mlx("0.32.0")
        with (
            mock.patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}),
            mock.patch.object(torch, "__version__", "2.12.1"),
            self.assertRaisesRegex(RuntimeError, "stable Torch 2.13.x"),
        ):
            ServerArgs(model_path="dummy", device="mps").resolve_once()

    def test_mps_server_validates_runtime_before_model_loading(self):
        from sglang.srt.server_args import ServerArgs

        args = types.SimpleNamespace(device="mps")
        with mock.patch(
            "sglang.srt.server_args.validate_mps_runtime"
        ) as validate_runtime:
            ServerArgs._handle_hardware_runtime_validation(args)
        validate_runtime.assert_called_once_with()

    def test_auto_detected_mps_validates_before_model_path_resolution(self):
        from sglang.srt.server_args import ServerArgs

        args = types.SimpleNamespace(device=None)
        with (
            mock.patch(
                "sglang.srt.server_args.current_platform.is_mps",
                return_value=True,
            ),
            mock.patch.object(ServerArgs, "_validate_mps_server_args") as validate,
        ):
            ServerArgs._handle_hardware_runtime_validation(args)
        validate.assert_called_once_with(args)

    def test_mps_debug_backend_override_is_allowed(self):
        from sglang.srt.environ import envs
        from sglang.srt.server_args import ServerArgs

        args = types.SimpleNamespace(device="mps")
        with (
            envs.SGLANG_FORCE_FUSED_OP_BACKEND.override("torch"),
            mock.patch("sglang.srt.server_args.validate_mps_runtime"),
        ):
            ServerArgs._validate_mps_server_args(args)

    def test_model_features_are_not_rejected_by_operator_runtime_gate(self):
        from sglang.srt.server_args import ServerArgs

        cases = (
            ("forward_hooks", [{"module": "model.layers.0"}]),
            ("model_impl", "transformers"),
        )
        for field, value in cases:
            with self.subTest(field=field):
                args = types.SimpleNamespace(device="mps", **{field: value})
                with mock.patch("sglang.srt.server_args.validate_mps_runtime"):
                    ServerArgs._validate_mps_server_args(args)

    def test_mps_and_torch_native_attention_contracts_are_accepted(self):
        from sglang.srt.server_args import ServerArgs

        for backend in (None, "mps", "torch_native"):
            with self.subTest(backend=backend):
                args = types.SimpleNamespace(device="mps", attention_backend=backend)
                with mock.patch("sglang.srt.server_args.validate_mps_runtime"):
                    ServerArgs._validate_mps_server_args(args)

    def test_mps_rejects_unsupported_execution_contracts(self):
        from sglang.srt.server_args import ServerArgs

        cases = (
            ("attention_backend", "triton", "attention backend"),
            ("prefill_attention_backend", "flashinfer", "attention backend"),
            ("sampling_backend", "flashinfer", "sampling backend"),
            ("kv_cache_dtype", "fp8_e4m3", "KV cache dtypes"),
            ("quantization", "awq", "unquantized model weights"),
            (
                "enable_torch_compile",
                True,
                "does not yet provide an SGLang torch.compile graph runner",
            ),
            ("tp_size", 2, "tp_size=1"),
            (
                "disaggregation_mode",
                "prefill",
                "does not yet support disaggregated serving",
            ),
            (
                "enable_multimodal",
                True,
                "multimodal serving is not yet validated",
            ),
            (
                "speculative_algorithm",
                "EAGLE",
                "does not yet support speculative decoding",
            ),
            ("lora_paths", ["adapter"], "lora-backend torch_native"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                args = types.SimpleNamespace(device="mps", **{field: value})
                with (
                    mock.patch("sglang.srt.server_args.validate_mps_runtime"),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    ServerArgs._validate_mps_server_args(args)

    def test_mps_accepts_torch_native_lora_contract(self):
        from sglang.srt.server_args import ServerArgs

        args = types.SimpleNamespace(
            device="mps",
            enable_lora=True,
            lora_backend="torch_native",
            enable_lora_overlap_loading=False,
        )
        with mock.patch("sglang.srt.server_args.validate_mps_runtime"):
            ServerArgs._validate_mps_server_args(args)

    def test_explicitly_disabled_lora_paths_do_not_enable_lora(self):
        from sglang.srt.server_args import ServerArgs

        args = types.SimpleNamespace(
            device="mps",
            enable_lora=False,
            lora_paths=["ignored-adapter"],
            lora_backend="csgmv",
        )
        with mock.patch("sglang.srt.server_args.validate_mps_runtime"):
            ServerArgs._validate_mps_server_args(args)

    def test_mps_defaults_remain_serial_and_supported(self):
        from sglang.srt.server_args import ServerArgs

        declared = {}
        args = types.SimpleNamespace(
            device="mps",
            _declare=lambda _source, **fields: declared.update(fields),
        )
        with mock.patch("sglang.srt.server_args.validate_mps_runtime"):
            ServerArgs._handle_mps_backends(args)
        self.assertTrue(declared["disable_overlap_schedule"])

    def test_hnd_kv_cache_aborts_single_mps_path(self):
        from sglang.srt.environ import envs
        from sglang.srt.server_args import ServerArgs

        args = types.SimpleNamespace(device="mps")
        with (
            envs.SGLANG_USE_HND_KVCACHE.override(True),
            mock.patch("sglang.srt.server_args.validate_mps_runtime"),
            self.assertRaisesRegex(ValueError, "standard NHD"),
        ):
            ServerArgs._validate_mps_server_args(args)

    def test_dllm_aborts_single_mps_path(self):
        from sglang.srt.server_args import ServerArgs

        args = types.SimpleNamespace(device="mps", dllm_algorithm="LowConfidence")
        with (
            mock.patch("sglang.srt.server_args.validate_mps_runtime"),
            self.assertRaisesRegex(ValueError, "DLLM execution"),
        ):
            ServerArgs._validate_mps_server_args(args)

    @unittest.skipUnless(
        importlib.util.find_spec("mlx") is not None
        and torch.backends.mps.is_available()
        and runtime._is_stable_series(torch.__version__, (2, 13))
        and _has_stable_distribution_at_least("mlx", Version("0.32.0")),
        "requires the supported MPS runtime",
    )
    def test_current_runtime_is_supported(self):
        runtime.validate_mps_runtime()


if __name__ == "__main__":
    unittest.main()
