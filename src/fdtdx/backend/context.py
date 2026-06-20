"""Forced backend override via a context manager / environment variable.

The override is mandatory for validation: on macOS the only way to obtain a JAX
reference is to force JAX (which runs on CPU, since JAX-Metal is unusable). The
context manager takes precedence over the ``FDTDMEX_BACKEND`` environment variable.

Example::

    with fdtdx.use_backend("jax"):
        ref = fdtdx.run_fdtd(arrays, objects, config)   # forced JAX (CPU oracle)
    with fdtdx.use_backend("mlx"):
        out = fdtdx.run_fdtd(arrays, objects, config)   # forced MLX (Metal)
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Literal

_BACKEND_OVERRIDE: ContextVar[str | None] = ContextVar("fdtdmex_backend_override", default=None)


@contextlib.contextmanager
def use_backend(backend: Literal["mlx", "jax"]):
    """Force the forward-FDTD backend within the ``with`` block.

    Args:
        backend: ``"mlx"`` to force the Metal forward loop, ``"jax"`` to force the
            default JAX engine.
    """
    if backend not in ("mlx", "jax"):
        raise ValueError(f"backend must be 'mlx' or 'jax', got {backend!r}")
    token = _BACKEND_OVERRIDE.set(backend)
    try:
        yield
    finally:
        _BACKEND_OVERRIDE.reset(token)


def get_backend_override() -> str | None:
    """Return the active override (``"mlx"``/``"jax"``) or ``None``."""
    return _BACKEND_OVERRIDE.get()
