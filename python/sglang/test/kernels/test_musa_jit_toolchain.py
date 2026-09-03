from sglang.kernels.jit.utils.compile import toolchain


def test_musa_toolchain_selects_mcc_and_mp_target(monkeypatch):
    monkeypatch.setattr(toolchain, "is_musa_runtime", lambda: True)
    monkeypatch.setenv("MUSA_HOME", "/opt/musa")
    monkeypatch.setenv("MTGPU_TARGET", "mp_31")

    assert toolchain.cuda_home() == "/opt/musa"
    assert toolchain.device_compiler_path() == "/opt/musa/bin/mcc"
    assert toolchain.gpu_arch_name() == "mp_31"
    assert toolchain.target_flags() == [
        "--cuda-gpu-arch=mp_31",
        "--offload-arch=mp_31",
        "-x",
        "musa",
        "-mtgpu",
    ]
    assert toolchain.base_cuda_flags() == ["-fPIC", "-D__MUSA__", "-DUSE_MUSA"]
