"""FDTDMEX — brand alias for the fdtdx fork with the MLX (Metal) forward backend.

This package is a thin alias kept for brand/back-compat continuity. The real engine,
including the forward MLX backend under :mod:`fdtdx.backend` and :mod:`fdtdx.mlx`, lives
in :mod:`fdtdx`. Prefer ``import fdtdx``; ``import fdtdmex`` simply re-exports the full
fdtdx public API.
"""

from importlib.metadata import PackageNotFoundError, version

from fdtdx import *  # noqa: F401,F403

try:
    __version__ = version("fdtdx")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
