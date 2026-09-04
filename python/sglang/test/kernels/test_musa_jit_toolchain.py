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
    flags = toolchain.base_cuda_flags()
    assert flags[:3] == ["-fPIC", "-D__MUSA__", "-DUSE_MUSA"]
    assert "-Od3" in flags
    assert "-ffast-math" in flags
    assert "-fmusa-flush-denormals-to-zero" in flags
    assert "-DENABLE_BF16" in flags
    assert "-DENABLE_FP8" in flags
    assert "-DFLASHINFER_ENABLE_FP8_E4M3" in flags
