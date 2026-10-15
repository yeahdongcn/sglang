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
# ============================================================================

"""Build the optional MLX Metal AOT extension.

The public ``mlx.extension`` API owns the CMake, nanobind, AIR, and metallib
build plumbing. Keeping that logic in MLX avoids a second copy of the MLX
ABI/linker assumptions in SGLang and makes the generated extension installable
through the normal setuptools ``build_ext`` flow.

This script intentionally remains a standalone setuptools entry point because
the CUDA ``sglang-kernel`` package has a separate scikit-build project. Run it
from the repository root with ``install`` or ``build_ext --inplace``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# MLX PR #4004 adds this public API. Keep the check here instead of allowing
# an opaque ``ImportError`` from a later setup.py import: released MLX 0.32.x
# wheels do not contain ``mlx.extension`` yet.
_BUILD_REQUIRES = (
    ("setuptools", "setuptools>=80"),
    ("nanobind", "nanobind==3.1.0"),
)


def _ensure_toolchain() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise SystemExit("setup_metal.py only supports macOS (Apple Silicon).")
    if shutil.which("c++") is None or shutil.which("xcrun") is None:
        raise SystemExit(
            "Apple toolchain not found. Install the Xcode Command Line Tools "
            "with `xcode-select --install` (or a full Xcode install) and retry."
        )
    try:
        subprocess.check_output(
            ["xcrun", "-sdk", "macosx", "metal", "--version"],
            stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(
            "Apple Metal shader compiler not found. Install a full Xcode "
            "(not just Command Line Tools) so that `xcrun -sdk macosx metal` "
            "is available, then retry."
        ) from exc
    cmake = shutil.which("cmake")
    if cmake is None:
        raise SystemExit(
            "CMake 3.25 or newer is required by MLX's MetalExtension. "
            "Install it with `uv pip install cmake` and retry."
        )
    cmake_output = subprocess.check_output([cmake, "--version"], text=True)
    match = re.search(r"cmake version (\d+)\.(\d+)", cmake_output)
    if match is None or (int(match.group(1)), int(match.group(2))) < (3, 25):
        raise SystemExit(
            "CMake 3.25 or newer is required by MLX's MetalExtension; "
            f"found {cmake_output.splitlines()[0] if cmake_output else 'unknown'}."
        )


def _ensure_build_requires() -> None:
    missing = []
    for import_name, pip_name in _BUILD_REQUIRES:
        try:
            importlib.import_module(import_name)
            minimum = {"setuptools": (80, 0), "nanobind": (3, 1)}.get(import_name)
            if minimum is not None:
                installed = importlib.metadata.version(import_name)
                match = re.match(r"(\d+)\.(\d+)", installed)
                if (
                    match is None
                    or (int(match.group(1)), int(match.group(2))) < minimum
                ):
                    missing.append(pip_name)
        except (ImportError, importlib.metadata.PackageNotFoundError):
            missing.append(pip_name)
    if not missing:
        return
    print(
        f"[sgl-kernel:metal] installing build requirements: {missing}",
        flush=True,
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", *missing]
    )


def _load_mlx_extension():
    try:
        from mlx import extension
    except ImportError as exc:
        raise SystemExit(
            "This Metal AOT build requires `mlx.extension`, introduced by "
            "MLX PR #4004. Install an MLX build from main (or a release "
            "containing that API) before running setup_metal.py."
        ) from exc
    if not hasattr(extension, "MetalExtension") or not hasattr(
        extension, "BuildExtension"
    ):
        raise SystemExit(
            "This Metal AOT build requires MLX's `MetalExtension` and "
            "`BuildExtension`, introduced by MLX PR #4004. Install an MLX "
            "build from main (or a release containing that API) before "
            "running setup_metal.py."
        )
    return extension


def _get_version() -> str:
    pyproject = ROOT / "pyproject.toml"
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        if line.startswith("version"):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError(f"Could not find a version in {pyproject}")


def main() -> None:
    _ensure_toolchain()
    _ensure_build_requires()
    extension = _load_mlx_extension()

    # MLX derives both the native module name and the default metallib name
    # from the final component of the extension name. Keep all three names
    # identical so setuptools installs the .so, .pyi, and .metallib together.
    extension_name = "sgl_kernel.sgl_metal_kernels"
    sources = [
        str(ROOT / "csrc" / "metal" / "rope_pool_fused.cpp"),
        str(ROOT / "csrc" / "metal" / "rope_pool_fused.metal"),
    ]

    os.chdir(ROOT)
    from setuptools import find_packages, setup  # noqa: PLC0415

    mlx_version = importlib.metadata.version("mlx")
    setup(
        name="sglang-kernel",
        version=_get_version(),
        packages=find_packages(where="python"),
        package_dir={"": "python"},
        package_data={"sgl_kernel": ["*.metallib", "*.pyi"]},
        include_package_data=True,
        ext_modules=[
            extension.MetalExtension(
                extension_name,
                sources=sources,
                include_dirs=[
                    str(ROOT / "csrc"),
                    str(ROOT / "csrc" / "metal"),
                ],
                extra_compile_args={
                    "cxx": ["-O3", "-fvisibility=hidden"],
                    "metal": ["-O3"],
                },
            )
        ],
        cmdclass={"build_ext": extension.BuildExtension},
        # Match MLX's extension ABI. This is deliberately the installed
        # version, so a source checkout of MLX main remains testable before a
        # release is published.
        install_requires=[f"mlx=={mlx_version}"],
    )


if __name__ == "__main__":
    main()
