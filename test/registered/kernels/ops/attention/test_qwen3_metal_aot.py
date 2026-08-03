"""Packaging and loader contracts for the Torch-owned Qwen3 Metal AOT path."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.attention.qwen3_mps import (
    is_qwen3_metal_aot_available,
    warmup_qwen3_mps_kernels,
)
from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")

_HAS_METAL_AOT = torch.backends.mps.is_available() and is_qwen3_metal_aot_available()


class _TrackingLibrary:
    def __init__(self):
        self.lookups = []

    def __getattr__(self, name):
        self.lookups.append(name)
        return object()


class TestQwen3MetalWarmupAdapters(unittest.TestCase):
    """The two semantic pipelines can be warmed without duplicate loading."""

    def test_aot_warmups_are_independent_and_share_library(self):
        from sglang.kernels.ops.attention import _qwen3_metal_aot as aot

        library = _TrackingLibrary()
        with mock.patch.object(aot, "_load_library", return_value=library) as load:
            aot.warmup_qwen3_metal_aot_qknorm_rope_store()
            self.assertEqual(library.lookups, ["qwen3_qknorm_rope_store_bf16"])
            aot.warmup_qwen3_metal_aot_radix_decode()
            self.assertEqual(
                library.lookups,
                [
                    "qwen3_qknorm_rope_store_bf16",
                    "qwen3_radix_decode_bf16",
                ],
            )
            load.assert_has_calls([mock.call(), mock.call()])

        library.lookups.clear()
        with mock.patch.object(aot, "_load_library", return_value=library) as load:
            aot.warmup_qwen3_metal_aot_kernels()
            load.assert_called_once_with()
            self.assertEqual(
                library.lookups,
                [
                    "qwen3_qknorm_rope_store_bf16",
                    "qwen3_radix_decode_bf16",
                ],
            )

    def test_jit_warmups_are_independent_and_share_compiled_library(self):
        from sglang.kernels.ops.attention import _qwen3_metal_jit as jit

        library = _TrackingLibrary()
        jit._compile_qwen3_library.cache_clear()
        try:
            with mock.patch.object(
                jit.torch.mps, "compile_shader", return_value=library, create=True
            ) as compile_library:
                jit.warmup_qwen3_metal_qknorm_rope_store()
                self.assertEqual(library.lookups, ["qwen3_qknorm_rope_store_bf16"])
                jit.warmup_qwen3_metal_radix_decode()
                self.assertEqual(
                    library.lookups,
                    [
                        "qwen3_qknorm_rope_store_bf16",
                        "qwen3_radix_decode_bf16",
                    ],
                )
                compile_library.assert_called_once()

            library.lookups.clear()
            with mock.patch.object(
                jit.torch.mps, "compile_shader", return_value=library, create=True
            ) as compile_library:
                jit.warmup_qwen3_metal_kernels()
                compile_library.assert_not_called()
                self.assertEqual(
                    library.lookups,
                    [
                        "qwen3_qknorm_rope_store_bf16",
                        "qwen3_radix_decode_bf16",
                    ],
                )
        finally:
            jit._compile_qwen3_library.cache_clear()


class TestQwen3MpsWarmupSelection(unittest.TestCase):
    """Provider combinations warm only their selected semantic pipelines."""

    def test_matching_provider_uses_one_combined_warmup(self):
        for backend in (KernelBackend.METAL_AOT, KernelBackend.METAL_JIT):
            with self.subTest(backend=backend.value):
                with (
                    mock.patch(
                        "sglang.kernels.ops.attention._qwen3_metal_aot."
                        "warmup_qwen3_metal_aot_kernels"
                    ) as aot,
                    mock.patch(
                        "sglang.kernels.ops.attention._qwen3_metal_jit."
                        "warmup_qwen3_metal_kernels"
                    ) as jit,
                ):
                    selected = warmup_qwen3_mps_kernels(backend, backend)

                expected, unselected = (
                    (aot, jit) if backend is KernelBackend.METAL_AOT else (jit, aot)
                )
                expected.assert_called_once_with()
                unselected.assert_not_called()
                self.assertEqual(selected, (backend, backend))

    def test_qkv_aot_and_decode_jit_warm_independently(self):
        with (
            mock.patch(
                "sglang.kernels.ops.attention._qwen3_metal_aot."
                "warmup_qwen3_metal_aot_kernels"
            ) as aot,
            mock.patch(
                "sglang.kernels.ops.attention._qwen3_metal_jit."
                "warmup_qwen3_metal_kernels"
            ) as jit,
        ):
            selected = warmup_qwen3_mps_kernels(
                KernelBackend.METAL_AOT,
                KernelBackend.METAL_JIT,
            )

        aot.assert_called_once_with(
            qknorm_rope_store=True,
            radix_decode=False,
        )
        jit.assert_called_once_with(
            qknorm_rope_store=False,
            radix_decode=True,
        )
        self.assertEqual(
            selected,
            (KernelBackend.METAL_AOT, KernelBackend.METAL_JIT),
        )

    def test_torch_side_does_not_warm_a_metal_pipeline(self):
        cases = (
            (
                KernelBackend.TORCH,
                KernelBackend.METAL_JIT,
                None,
                mock.call(qknorm_rope_store=False, radix_decode=True),
            ),
            (
                KernelBackend.METAL_AOT,
                KernelBackend.TORCH,
                mock.call(qknorm_rope_store=True, radix_decode=False),
                None,
            ),
            (KernelBackend.TORCH, KernelBackend.TORCH, None, None),
        )
        for qkv, decode, expected_aot, expected_jit in cases:
            with self.subTest(qkv=qkv.value, decode=decode.value):
                with (
                    mock.patch(
                        "sglang.kernels.ops.attention._qwen3_metal_aot."
                        "warmup_qwen3_metal_aot_kernels"
                    ) as aot,
                    mock.patch(
                        "sglang.kernels.ops.attention._qwen3_metal_jit."
                        "warmup_qwen3_metal_kernels"
                    ) as jit,
                ):
                    selected = warmup_qwen3_mps_kernels(qkv, decode)

                self.assertEqual(selected, (qkv, decode))
                self.assertEqual(aot.call_args, expected_aot)
                self.assertEqual(jit.call_args, expected_jit)


@unittest.skipUnless(_HAS_METAL_AOT, "requires packaged Qwen3 Metal AOT")
class TestQwen3MetalAotPackage(unittest.TestCase):
    def test_packaged_library_resolves_both_pipelines_in_subprocess(self):
        # A malformed metallib can abort inside Metal pipeline creation. Keep
        # that failure isolated so CI reports stderr instead of killing pytest.
        program = """
from sgl_kernel.metal import load_metal_library
library = load_metal_library()
library.qwen3_qknorm_rope_store_bf16
library.qwen3_radix_decode_bf16
print(type(library).__name__)
"""
        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("PrecompiledShaderLibrary", completed.stdout)

    def test_loader_caches_one_torch_library(self):
        import sgl_kernel.metal as metal

        fake_library = SimpleNamespace()
        metal.load_metal_library.cache_clear()
        try:
            with mock.patch.object(
                torch.mps,
                "load_metallib",
                return_value=fake_library,
            ) as load:
                self.assertIs(metal.load_metal_library(), fake_library)
                self.assertIs(metal.load_metal_library(), fake_library)
            load.assert_called_once_with(metal.metal_library_path())
        finally:
            metal.load_metal_library.cache_clear()

    def test_wheel_contains_metallib_without_old_mlx_extension(self):
        files = importlib.metadata.distribution("sglang-kernel").files or []
        names = [str(path) for path in files]
        self.assertTrue(
            any(
                name.endswith("sgl_kernel/sgl_metal_kernels.metallib") for name in names
            )
        )
        self.assertFalse(
            any(
                name.rsplit("/", 1)[-1].startswith("_metal")
                and name.endswith((".so", ".dylib"))
                for name in names
            )
        )


if __name__ == "__main__":
    unittest.main()
