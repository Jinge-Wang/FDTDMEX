"""Platform / availability probes for the MLX backend.

Cheap, cached checks used by the dispatcher. ``mlx`` is an Apple-Silicon-only
dependency (see pyproject platform marker), so ``mlx_available()`` is the real gate:
on Linux/CUDA it returns False and the dispatcher falls back to JAX.
"""

from __future__ import annotations

import functools
import platform


@functools.lru_cache(maxsize=1)
def is_apple_silicon() -> bool:
    """True on macOS running on Apple-Silicon (arm64)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


@functools.lru_cache(maxsize=1)
def mlx_available() -> bool:
    """True if ``mlx.core`` can be imported in this environment."""
    try:
        import mlx.core  # noqa: F401

        return True
    except Exception:
        return False
