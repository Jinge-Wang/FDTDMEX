"""Backend selection for the forward FDTD time loop.

On Apple Silicon, forward-only ``run_fdtd`` calls are routed to a native MLX (Metal)
implementation of the time loop (see :mod:`fdtdx.mlx`); everything else runs the
default JAX engine. The routing is decided in :mod:`fdtdx.backend.dispatch`; a forced
override is available via :func:`fdtdx.backend.context.use_backend` or the
``FDTDMEX_BACKEND`` environment variable.

This package imports no heavy backends at module load (no ``mlx``/``jax``); the MLX
engine is imported lazily only when the MLX path is actually taken, so importing
``fdtdx`` on non-Apple platforms is unaffected.
"""
