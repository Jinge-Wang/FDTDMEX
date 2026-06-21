"""Decide whether a forward ``run_fdtd`` call routes to the MLX (Metal) backend.

``maybe_run_mlx_forward`` is called from the top of ``fdtdx.fdtd.wrapper.run_fdtd``. It
returns a completed ``SimulationState`` when the MLX path handled the run, or ``None`` to
let the default JAX engine run (the guarded fallthrough).

Routing:
- A forced override (``fdtdx.use_backend(...)`` context manager, else ``FDTDMEX_BACKEND``
  env var) wins. Forced "mlx" raises if the case is infeasible; forced "jax" always falls
  back. The override is what lets validation run the same case through both backends on one
  Mac (the JAX oracle on CPU).
- AUTO: MLX iff Apple-Silicon + mlx importable + forward-only + the case uses only
  features the current milestone supports; otherwise JAX (warn-once on the first decline).

Milestone gating lives in ``_unsupported_reason``; widen it as kernels land (M1: lossless
iso/diag materials, uniform grid, point-dipole sources, no detectors).
"""

from __future__ import annotations

import os
from enum import Enum

from loguru import logger

from fdtdx.backend.context import get_backend_override
from fdtdx.backend.platform import is_apple_silicon, mlx_available


class Backend(str, Enum):
    MLX = "mlx"
    JAX = "jax"


# Source/detector types the MLX engine currently handles. Widened per milestone.
def _supported_source_types() -> tuple:
    from fdtdx.objects.sources.dipole import PointDipoleSource

    return (PointDipoleSource,)


def _supported_detector_types() -> tuple:
    from fdtdx.objects.detectors.energy import EnergyDetector
    from fdtdx.objects.detectors.field import FieldDetector
    from fdtdx.objects.detectors.poynting_flux import PoyntingFluxDetector

    return (EnergyDetector, FieldDetector, PoyntingFluxDetector)


_warned_reasons: set[str] = set()


def _unsupported_reason(config, objects, stopping_condition) -> str | None:
    """Return a human-readable reason the case can't run on MLX yet, or ``None``."""
    if config.gradient_config is not None:
        return "gradient computation requested (MLX backend is forward-only)"
    if stopping_condition is not None:
        return "custom stopping_condition not supported by the MLX backend yet"
    if config.has_nonuniform_grid:
        return "non-uniform grids not supported by the MLX backend yet (M4)"
    if getattr(config, "use_complex_fields", None) is True or objects.bloch_objects:
        return "complex/Bloch fields not supported by the MLX backend yet"

    supported_sources = _supported_source_types()
    for s in objects.sources:
        if not isinstance(s, supported_sources):
            return f"source type {type(s).__name__} not supported by the MLX backend yet"

    supported_detectors = _supported_detector_types()
    for d in objects.detectors:
        if not isinstance(d, supported_detectors):
            return f"detector type {type(d).__name__} not supported by the MLX backend yet"
        if getattr(d, "as_slices", False):
            return f"{type(d).__name__}(as_slices=True) not supported by the MLX backend yet"

    return None


def _unsupported_reason_arrays(arrays) -> str | None:
    """Material/array-level support checks (need the ArrayContainer)."""
    if arrays.inv_permittivities.shape[0] == 9:
        return "full-anisotropic permittivity (9-tensor) not supported yet (M3)"
    inv_mu = arrays.inv_permeabilities
    if getattr(inv_mu, "ndim", 0) > 0 and inv_mu.shape[0] == 9:
        return "full-anisotropic permeability (9-tensor) not supported yet (M3)"
    for label, sigma in (("electric", arrays.electric_conductivity), ("magnetic", arrays.magnetic_conductivity)):
        if sigma is not None and getattr(sigma, "ndim", 0) > 0 and sigma.shape[0] == 9:
            return f"full-anisotropic {label} conductivity (9-tensor) not supported yet (M3)"
    if arrays.dispersive_c1 is not None:
        return "dispersive (ADE) materials not supported yet"
    return None


def select_backend(arrays, objects, config, stopping_condition) -> Backend:
    """Return the backend to use, honoring forced overrides and milestone gating."""
    override = get_backend_override() or (os.environ.get("FDTDMEX_BACKEND", "").lower() or None)

    if override == "jax":
        return Backend.JAX
    if override == "mlx":
        reason = _unsupported_reason(config, objects, stopping_condition) or _unsupported_reason_arrays(arrays)
        if reason is not None:
            raise NotImplementedError(f"FDTDMEX_BACKEND=mlx but this case is unsupported: {reason}")
        return Backend.MLX

    # AUTO
    if not (is_apple_silicon() and mlx_available() and config.gradient_config is None):
        return Backend.JAX
    reason = _unsupported_reason(config, objects, stopping_condition) or _unsupported_reason_arrays(arrays)
    if reason is not None:
        if reason not in _warned_reasons:
            _warned_reasons.add(reason)
            logger.warning(f"MLX backend declined, falling back to JAX: {reason}")
        return Backend.JAX
    return Backend.MLX


def maybe_run_mlx_forward(arrays, objects, config, key, stopping_condition):
    """Run the forward loop on MLX and return a SimulationState, or ``None`` for JAX."""
    backend = select_backend(arrays, objects, config, stopping_condition)
    if backend is not Backend.MLX:
        return None
    return _run_mlx_forward(arrays, objects, config)


def _run_mlx_forward(arrays, objects, config):
    import jax.numpy as jnp

    from fdtdx.mlx.bridge import buffers_to_detector_states, to_array_container, to_mlx_state
    from fdtdx.mlx.detector_freeze import allocate_buffers, freeze_detectors
    from fdtdx.mlx.loop import run_forward_mlx
    from fdtdx.mlx.source_freeze import freeze_sources

    # Match checkpointed_fdtd: zero dynamic fields + detector states before stepping.
    arrays = arrays.reset()

    state = to_mlx_state(arrays, config)
    source_plans = freeze_sources(objects, config)
    detector_plans = freeze_detectors(objects, config)
    detector_buffers = allocate_buffers(detector_plans)
    num_steps = int(config.time_steps_total)
    c = float(config.courant_number)

    state, detector_buffers = run_forward_mlx(
        state, source_plans, detector_plans, detector_buffers, num_steps, c, simulate_boundaries=True
    )

    detector_states = buffers_to_detector_states(detector_buffers) if detector_plans else None
    out_arrays = to_array_container(arrays, state, detector_states)
    return jnp.asarray(num_steps, dtype=jnp.int32), out_arrays
