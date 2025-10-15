import importlib
import logging

import pytest


def setup_module(module):
    # Ensure module is reloaded fresh for each test run
    import sglang.srt.custom_op_backend as cob

    importlib.reload(cob)


def test_register_decorator_creates_entry(monkeypatch):
    import sglang.srt.custom_op_backend as cob

    # Define a dummy backend and register it
    @cob.register_custom_op_backend("test_backend")
    def _create():
        return {"name": "test_backend"}

    assert "test_backend" in cob.CUSTOM_OP_BACKENDS
    assert cob.CUSTOM_OP_BACKENDS["test_backend"] is _create


@pytest.mark.parametrize(
    "env_checks, expected_backend, expected_return, expect_warning",
    [
        ({"is_cuda": True}, "cuda", {"name": "cuda"}, False),
        ({"is_hip": True}, "hip", {"name": "hip"}, False),
        ({"is_npu": True}, "npu", {"name": "npu"}, False),
        ({"is_xpu": True}, "xpu", {"name": "xpu"}, False),
        ({"is_cpu": True}, "cpu", {"name": "cpu"}, False),
        ({}, "native", {}, True),
    ],
)
def test_create_custom_op_backend_selection(
    monkeypatch, caplog, env_checks, expected_backend, expected_return, expect_warning
):
    """Test that create_custom_op_backend selects the right backend and logs appropriately."""
    caplog.set_level(logging.DEBUG)

    # Reload module to start from a clean state
    cob = importlib.import_module("sglang.srt.custom_op_backend")
    importlib.reload(cob)

    # Register simple factory functions for each backend we expect
    for name in ["cuda", "hip", "npu", "xpu", "cpu"]:

        @cob.register_custom_op_backend(name)
        def _make(name=name):
            return {"name": name}

    # Prepare dummy utils functions
    class DummyUtils:
        pass

    utils = DummyUtils()
    # Default all checks to False
    for fn in ("is_cuda", "is_hip", "is_npu", "is_xpu", "is_cpu"):
        setattr(utils, fn, lambda: False)

    # Override any specified checks to True
    for k, v in env_checks.items():
        setattr(utils, k, lambda v=v: v)

    # Monkeypatch the import path used inside the function
    import sys
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils if False else utils)

    # Alternative approach: monkeypatch the attributes on the real utils module if present
    try:
        import sglang.srt.utils as real_utils

        for fn in ("is_cuda", "is_hip", "is_npu", "is_xpu", "is_cpu"):
            val = getattr(utils, fn)
            monkeypatch.setattr(real_utils, fn, val)
    except Exception:
        # If real module not available, ensure import in target module uses our dummy by injecting into sys.modules
        import sys

        monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils)

    # Call the function under test
    result = cob.create_custom_op_backend()

    if expect_warning:
        assert any(
            "native fallback" in rec.message.lower()
            or "No registered backend" in rec.message
            for rec in caplog.records
        )
        assert result == {}
    else:
        assert result == expected_return
