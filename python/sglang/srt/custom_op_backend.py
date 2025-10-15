import importlib
import logging

from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    is_xpu,
)

logger = logging.getLogger(__name__)

# ============================================================
# Backend Registry
# ============================================================

_CUSTOM_OP_BACKENDS = {}


def register_custom_op_backend(name):
    """Decorator to register backend factory functions."""

    def decorator(fn):
        _CUSTOM_OP_BACKENDS[name] = fn
        return fn

    return decorator


def get_custom_op_backend(name):
    """Return an initialized backend instance (LazyBackend)."""
    if name not in _CUSTOM_OP_BACKENDS:
        raise ValueError(f"Custom op backend '{name}' not registered.")
    return _CUSTOM_OP_BACKENDS[name]()


# ============================================================
# Lazy Import Utilities
# ============================================================


def import_from(module_name, attr_name):
    """Helper to dynamically import and retrieve an attribute."""
    mod = importlib.import_module(module_name)
    return getattr(mod, attr_name)


class LazyOp:
    """Lazily imports and caches a single op when accessed."""

    def __init__(self, import_fn):
        self._import_fn = import_fn
        self._cached = None

    def __call__(self, *args, **kwargs):
        if self._cached is None:
            self._cached = self._import_fn()
        return self._cached(*args, **kwargs)


class LazyBackend:
    """Per-backend namespace of lazy ops."""

    def __init__(self, backend_name, op_specs):
        self.backend_name = backend_name
        self._ops = {name: LazyOp(fn) for name, fn in op_specs.items()}

    def get(self, name):
        if name not in self._ops:
            raise KeyError(f"No op '{name}' for backend '{self.backend_name}'")
        return self._ops[name]

    def __getattr__(self, name):
        return self.get(name)


# ============================================================
# Backend Definitions
# ============================================================


@register_custom_op_backend("cuda")
def create_cuda_backend():
    logger.info("Using CUDA backend for custom ops (lazy loaded)")
    return LazyBackend(
        "cuda",
        {
            "silu_and_mul": lambda: import_from("sgl_kernel", "silu_and_mul"),
            "gelu_and_mul": lambda: import_from("sgl_kernel", "gelu_and_mul"),
        },
    )


@register_custom_op_backend("hip")
def create_hip_backend():
    logger.info("Using HIP backend for custom ops (lazy loaded)")
    return LazyBackend(
        "hip",
        {
            "silu_and_mul": lambda: import_from("sgl_kernel", "silu_and_mul"),
            "gelu_and_mul": lambda: import_from("sgl_kernel", "gelu_and_mul"),
            "gelu_quick": lambda: import_from("sgl_kernel", "gelu_quick"),
            "gelu_tanh_and_mul": lambda: import_from("sgl_kernel", "gelu_tanh_and_mul"),
        },
    )


@register_custom_op_backend("xpu")
def create_xpu_backend():
    logger.info("Using XPU backend for custom ops (lazy loaded)")
    return LazyBackend(
        "xpu",
        {
            "silu_and_mul": lambda: import_from("sgl_kernel", "silu_and_mul"),
            "gelu_and_mul": lambda: import_from("sgl_kernel", "gelu_and_mul"),
        },
    )


@register_custom_op_backend("npu")
def create_npu_backend():
    logger.info("Using NPU backend for custom ops (lazy loaded)")
    import torch_npu

    return LazyBackend("npu", {})


@register_custom_op_backend("cpu")
def create_cpu_backend():
    logger.info("Using CPU backend for custom ops (lazy loaded)")
    return LazyBackend("cpu", {})


# ============================================================
# Unified Access Registry
# ============================================================


class CustomOpRegistry:
    """Dot-access and auto-detection registry for all backends."""

    def __init__(self):
        self._backend_cache = {}
        self._active_backend_name = None

    # --- Backend detection ---
    def _detect_backend(self) -> str:
        if is_cuda():
            return "cuda"
        if is_hip():
            return "hip"
        if is_npu():
            return "npu"
        if is_xpu():
            return "xpu"
        if is_cpu():
            return "cpu_amx" if cpu_has_amx_support() else "cpu"
        return "native"

    # --- Lazy backend init ---
    def _get_backend(self, backend_name=None):
        backend_name = (
            backend_name or self._active_backend_name or self._detect_backend()
        )
        if backend_name not in self._backend_cache:
            backend = get_custom_op_backend(backend_name)
            self._backend_cache[backend_name] = backend
            logger.info(f"Initialized backend: {backend_name}")
        self._active_backend_name = backend_name
        return self._backend_cache[backend_name]

    # --- Public API ---
    def get_op(self, op_name: str):
        """Get an op (auto backend detection + validation)."""
        backend = self._get_backend()
        try:
            return backend.get(op_name)
        except KeyError:
            raise KeyError(
                f"Custom op '{op_name}' not found in backend '{backend.backend_name}'"
            )

    def __getattr__(self, backend_name):
        """Dot access for explicit backend ops, e.g. custom_ops.cuda.silu_and_mul"""
        return self._get_backend(backend_name)


# Singleton access
custom_ops = CustomOpRegistry()
