# Copyright 2026 SGLang Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Build the Apple Silicon ``sglang-kernel`` wheel.

The Metal wheel is a Python package plus a precompiled ``.metallib``.  Torch
2.13 loads that library directly with :func:`torch.mps.load_metallib`, so this
build deliberately has no MLX, nanobind, or C++ host-extension dependency.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Distribution, find_packages, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py

root = Path(__file__).parent.resolve()

operator_namespace = "sgl_kernel"
metallib_name = "sgl_metal_kernels.metallib"
# Match the oldest platform supported by the required Torch 2.13 macOS wheel.
# This also keeps the wheel tag honest when the build runs under a newer SDK.
os.environ["MACOSX_DEPLOYMENT_TARGET"] = "14.0"

# This is also the source compiled by ``torch.mps.compile_shader``. Keeping one
# source avoids JIT/AOT numerical drift while the first supported contract is
# intentionally fixed to dense Qwen3-0.6B, TP=1, bf16.
qwen3_source = root.parent / "ops" / "attention" / "_qwen3_06b_attention.metal"


def _ensure_toolchain() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise SystemExit("setup_metal.py only supports macOS (Apple Silicon).")
    if shutil.which("xcrun") is None:
        raise SystemExit(
            "Apple toolchain not found. Install Xcode and select it with "
            "xcode-select before building the Metal wheel."
        )
    for tool in ("metal", "metallib"):
        try:
            subprocess.check_output(
                ["xcrun", "-sdk", "macosx", tool, "-help"],
                stderr=subprocess.STDOUT,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise SystemExit(
                f"Apple Metal tool {tool!r} is unavailable. Install a full "
                "Xcode toolchain (Command Line Tools alone may be insufficient)."
            ) from exc
    if not qwen3_source.is_file():
        raise SystemExit(f"canonical Qwen3 Metal source not found: {qwen3_source}")


def _get_version() -> str:
    with open(root / "pyproject.toml") as file:
        for line in file:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')
    raise RuntimeError("version is missing from pyproject.toml")


class MetalDistribution(Distribution):
    """Mark the data-only metallib wheel as platform-specific."""

    def has_ext_modules(self) -> bool:
        return True


class MetalBdistWheel(bdist_wheel):
    """Emit one Python-agnostic, Apple-Silicon-specific data wheel."""

    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        _, _, platform_tag = super().get_tag()
        return "py3", "none", platform_tag


class BuildMetalPackage(build_py):
    """Compile the canonical MSL source and stage it into ``sgl_kernel``."""

    def run(self) -> None:
        super().run()

        package_dir = Path(self.build_lib) / operator_namespace
        package_dir.mkdir(parents=True, exist_ok=True)
        # An incremental build may reuse a directory produced by the retired
        # MLX/nanobind implementation.  Never let that stale host extension
        # leak into the data-only Torch metallib wheel.
        for stale_extension in package_dir.glob("_metal*.so"):
            stale_extension.unlink()
        for stale_extension in package_dir.glob("_metal*.dylib"):
            stale_extension.unlink()
        generated_dir = Path(self.build_lib).parent / "metal"
        generated_dir.mkdir(parents=True, exist_ok=True)

        air_path = generated_dir / "qwen3_06b_attention.air"
        output_path = package_dir / metallib_name
        metal_std = os.environ.get("SGL_METAL_STD", "metal3.1")

        self.spawn(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                f"-std={metal_std}",
                "-O3",
                "-c",
                str(qwen3_source),
                "-o",
                str(air_path),
            ]
        )
        self.spawn(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metallib",
                str(air_path),
                "-o",
                str(output_path),
            ]
        )


_ensure_toolchain()
os.chdir(root)

setup(
    name="sglang-kernel",
    version=_get_version(),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    package_data={operator_namespace: ["*.metallib"]},
    include_package_data=True,
    cmdclass={
        "bdist_wheel": MetalBdistWheel,
        "build_py": BuildMetalPackage,
    },
    distclass=MetalDistribution,
)
