"""Lazy model-specific MPS operator routing.

Keep this package initializer dependency-free. Importing ``registry`` or
``router`` for an unknown model must not import any family provider.
"""

__all__: list[str] = []
