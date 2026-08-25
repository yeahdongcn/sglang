"""Framework-neutral compilation of Metal sources into one data artifact.

The output of this module is a plain ``.metallib``.  It deliberately does not
link a C++/nanobind host extension or import Torch/MLX; a wheel build can stage
the artifact as package data and each host runtime can own a small loader
adapter.  Kernel sources and entry-point contracts remain with their semantic
operator modules.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

CommandRunner = Callable[[Sequence[str]], None]


def _run(command: Sequence[str]) -> None:
    subprocess.check_call(command)


def ensure_metal_toolchain() -> None:
    """Raise an actionable error when the Apple Metal compiler is unavailable."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("Metal AOT compilation requires macOS on Apple Silicon")
    if shutil.which("xcrun") is None:
        raise RuntimeError(
            "Apple toolchain not found; install Xcode and select it with "
            "xcode-select"
        )
    for tool in ("metal", "metallib"):
        try:
            subprocess.check_output(
                ["xcrun", "-sdk", "macosx", tool, "-help"],
                stderr=subprocess.STDOUT,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"Apple Metal tool {tool!r} is unavailable; install a full "
                "Xcode toolchain"
            ) from exc


def compile_metallib(
    sources: Iterable[str | Path],
    output: str | Path,
    *,
    build_dir: str | Path,
    include_dirs: Iterable[str | Path] = (),
    metal_std: str = "metal3.1",
    compiler_flags: Iterable[str] = ("-O3",),
    runner: CommandRunner = _run,
    check_toolchain: bool = True,
) -> Path:
    """Compile ``sources`` and link them into one ``.metallib`` artifact.

    The function contains no model- or operator-specific source discovery.
    Callers pass an ordered source manifest, making the build input explicit
    and keeping packaging independent from any host tensor framework.
    """
    source_paths = tuple(Path(source).expanduser().resolve() for source in sources)
    if not source_paths:
        raise ValueError("at least one Metal source is required")
    missing = tuple(path for path in source_paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Metal source is missing: {missing[0]}")
    if not metal_std:
        raise ValueError("metal_std must be non-empty")
    if check_toolchain:
        ensure_metal_toolchain()

    build_path = Path(build_dir).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    build_path.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    include_args = [
        argument
        for directory in include_dirs
        for argument in ("-I", str(Path(directory).expanduser().resolve()))
    ]
    air_paths = []
    for index, source_path in enumerate(source_paths):
        # Prefix the stem so two operator groups may use the same source name.
        air_path = build_path / f"{index:03d}_{source_path.stem}.air"
        runner(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                f"-std={metal_std}",
                *compiler_flags,
                *include_args,
                "-c",
                str(source_path),
                "-o",
                str(air_path),
            ]
        )
        air_paths.append(air_path)

    runner(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metallib",
            *(str(path) for path in air_paths),
            "-o",
            str(output_path),
        ]
    )
    return output_path


__all__ = ["compile_metallib", "ensure_metal_toolchain"]
