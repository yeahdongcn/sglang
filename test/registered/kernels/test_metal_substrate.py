"""CPU-safe contracts for the operator-agnostic Metal substrate."""

import sys
from types import SimpleNamespace

import pytest

from sglang.kernels import metal
from sglang.kernels.metal_build import compile_metallib
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def _clear_library_caches():
    metal.clear_metal_library_caches()
    yield
    metal.clear_metal_library_caches()


def test_jit_compilation_is_explicit_and_cached(monkeypatch):
    calls = []
    library = object()
    monkeypatch.setattr(
        metal,
        "_torch_mps_function",
        lambda name: lambda source: calls.append((name, source)) or library,
    )

    assert metal.compile_metal_library("kernel void probe() {}") is library
    assert metal.compile_metal_library("kernel void probe() {}") is library
    assert calls == [("compile_shader", "kernel void probe() {}")]


def test_jit_rejects_empty_source_before_runtime_lookup(monkeypatch):
    lookup = []
    monkeypatch.setattr(
        metal,
        "_torch_mps_function",
        lambda name: lookup.append(name),
    )
    with pytest.raises(ValueError, match="non-empty"):
        metal.compile_metal_library("  ")
    assert lookup == []


def test_aot_loading_normalizes_paths_and_is_cached(monkeypatch, tmp_path):
    library_path = tmp_path / "probe.metallib"
    library_path.write_bytes(b"metallib")
    calls = []
    library = object()
    monkeypatch.setattr(
        metal,
        "_torch_mps_function",
        lambda name: lambda path: calls.append((name, path)) or library,
    )

    assert metal.load_metal_library(library_path) is library
    assert (
        metal.load_metal_library(library_path.parent / "." / library_path.name)
        is library
    )
    assert calls == [("load_metallib", library_path.resolve())]


def test_aot_missing_file_fails_before_runtime_lookup(monkeypatch, tmp_path):
    lookup = []
    monkeypatch.setattr(
        metal,
        "_torch_mps_function",
        lambda name: lookup.append(name),
    )
    with pytest.raises(RuntimeError, match="Metal library is missing"):
        metal.load_metal_library(tmp_path / "missing.metallib")
    assert lookup == []


def test_entry_point_warmup_is_operator_agnostic():
    first = object()
    second = object()
    library = SimpleNamespace(first=first, second=second)
    assert metal.resolve_metal_entry_points(library, ("first", "second")) == (
        first,
        second,
    )
    with pytest.raises(ValueError, match="non-empty"):
        metal.resolve_metal_entry_points(library, ("",))


def test_data_only_builder_compiles_manifest_and_links_once(tmp_path):
    source_a = tmp_path / "group_a" / "kernel.metal"
    source_b = tmp_path / "group_b" / "kernel.metal"
    source_a.parent.mkdir()
    source_b.parent.mkdir()
    source_a.write_text("kernel void first() {}")
    source_b.write_text("kernel void second() {}")
    output = tmp_path / "package" / "sgl_metal_kernels.metallib"
    commands = []

    result = compile_metallib(
        (source_a, source_b),
        output,
        build_dir=tmp_path / "build",
        include_dirs=(tmp_path / "include",),
        runner=lambda command: commands.append(list(command)),
        check_toolchain=False,
    )

    assert result == output.resolve()
    assert len(commands) == 3
    assert commands[0][3:5] == ["metal", "-std=metal3.1"]
    assert commands[1][3:5] == ["metal", "-std=metal3.1"]
    assert commands[0][-1].endswith("000_kernel.air")
    assert commands[1][-1].endswith("001_kernel.air")
    assert commands[2][3] == "metallib"
    assert commands[2][-1] == str(output.resolve())


def test_data_only_builder_rejects_empty_or_missing_manifest(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        compile_metallib(
            (),
            tmp_path / "out.metallib",
            build_dir=tmp_path / "build",
            check_toolchain=False,
        )
    with pytest.raises(FileNotFoundError, match="Metal source is missing"):
        compile_metallib(
            (tmp_path / "missing.metal",),
            tmp_path / "out.metallib",
            build_dir=tmp_path / "build",
            check_toolchain=False,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
