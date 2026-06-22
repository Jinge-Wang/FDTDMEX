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

Milestone gating lives in ``_unsupported_reason`` / ``_unsupported_reason_arrays``; widen it as
kernels land. The MLX path covers iso/diag/full-tensor anisotropy (incl. lossy + 9-tensor
conductivity), CPML + periodic + PEC/PMC boundaries, dipole + (tilted) TFSF plane sources, the four
detector types, non-uniform (rectilinear) grids, and Drude-Lorentz (ADE) dispersion (Phase 3).
Still gated to JAX: gradients, dispersive/randomized plane sources, Bloch/complex propagation, and
mode sources/detectors.
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
    from fdtdx.objects.sources.linear_polarization import LinearlyPolarizedPlaneSource

    return (PointDipoleSource, LinearlyPolarizedPlaneSource)


def _supported_detector_types() -> tuple:
    from fdtdx.objects.detectors.energy import EnergyDetector
    from fdtdx.objects.detectors.field import FieldDetector
    from fdtdx.objects.detectors.phasor import PhasorDetector
    from fdtdx.objects.detectors.poynting_flux import PoyntingFluxDetector

    return (EnergyDetector, FieldDetector, PoyntingFluxDetector, PhasorDetector)


_warned_reasons: set[str] = set()


def _metal_kernel_enabled() -> bool:
    """Whether the custom-Metal-kernel forward path is enabled (env ``FDTDMEX_METAL_KERNEL``).

    Default **on** (Phase 2 M3: CPML folded into the kernel, non-uniform metric + heterogeneous
    full-tensor inclusions covered, parity-clean across the eligible surface). The loop still falls
    back to the compiled MLX-op cores for any case the kernel can't handle (``kernel_eligible``).
    Set ``FDTDMEX_METAL_KERNEL=0`` (or ``false``/``no``/``off``) to force the MLX-op path.
    """
    return os.environ.get("FDTDMEX_METAL_KERNEL", "").lower() not in ("0", "false", "no", "off")


def _unsupported_reason(config, objects, stopping_condition) -> str | None:
    """Return a human-readable reason the case can't run on MLX yet, or ``None``."""
    if config.gradient_config is not None:
        return "gradient computation requested (MLX backend is forward-only)"
    if stopping_condition is not None:
        return "custom stopping_condition not supported by the MLX backend yet"
    if getattr(config, "use_complex_fields", None) is True:
        return "forced complex fields not supported by the MLX backend yet"
    for b in objects.bloch_objects:
        if b.needs_complex_fields:
            return "Bloch (nonzero-k, complex) boundaries not supported by the MLX backend yet"
    # PEC/PMC are supported (Phase 3): frozen keep-masks applied post-injection in the loop
    # (fdtdx.mlx.boundary_mask + loop.py), composing with both the Metal kernel and MLX-op cores.

    from fdtdx.objects.sources.linear_polarization import LinearlyPolarizedPlaneSource

    supported_sources = _supported_source_types()
    for s in objects.sources:
        if not isinstance(s, supported_sources):
            return f"source type {type(s).__name__} not supported by the MLX backend yet"
        if isinstance(s, LinearlyPolarizedPlaneSource):
            # Tilt (azimuth/elevation) is fine: it bakes into the frozen _E/_H profiles and the
            # per-cell Yee time offsets, which the source freeze handles. Randomized and dispersive
            # plane sources are not yet supported.
            if (
                getattr(s, "max_angle_random_offset", 0.0)
                or getattr(s, "max_vertical_offset", 0.0)
                or getattr(s, "max_horizontal_offset", 0.0)
            ):
                return f"randomized plane source ({type(s).__name__}) not supported by the MLX backend yet"
            if getattr(s, "_temporal_H_filter", None) is not None:
                return f"dispersive plane source ({type(s).__name__}) not supported by the MLX backend yet"

    supported_detectors = _supported_detector_types()
    for d in objects.detectors:
        if not isinstance(d, supported_detectors):
            return f"detector type {type(d).__name__} not supported by the MLX backend yet"
        if getattr(d, "as_slices", False):
            return f"{type(d).__name__}(as_slices=True) not supported by the MLX backend yet"

    return None


def _unsupported_reason_arrays(arrays) -> str | None:
    """Material/array-level support checks (need the ArrayContainer)."""
    # Phase 3: lossy full-tensor (9-component) anisotropy and 9-tensor (full-rank) electric/magnetic
    # conductivity are supported -- the aniso A/B update (``_update_aniso``) consumes ``sigma``
    # directly (``compute_anisotropic_update_matrices_mlx``), so these run on the MLX-op cores (the
    # lossless block-hybrid Metal kernel stays as-is; ``kernel_eligible`` falls these back).
    #
    # Phase 3 item 3: Drude-Lorentz (ADE) dispersion is supported -- polarization P is threaded
    # through the E-side of the loop (``mlx.update._update_E`` / the Metal E-kernel ADE fold), with
    # coefficients carried in ``MLXState``. fdtdx forbids dispersion + off-diagonal tensors, so it is
    # always iso/diagonal: lossless rides the Metal kernel, lossy+dispersive uses the MLX-op cores.
    # (Dispersive *plane sources* remain gated separately in ``_unsupported_reason``.)
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

    from fdtdx.fdtd.update import get_wrap_padding_axes
    from fdtdx.mlx.bridge import buffers_to_detector_states, to_array_container, to_mlx_state
    from fdtdx.mlx.detector_freeze import allocate_buffers, freeze_detectors
    from fdtdx.mlx.loop import run_forward_mlx
    from fdtdx.mlx.source_freeze import freeze_sources

    # Match checkpointed_fdtd: zero dynamic fields + detector states before stepping.
    arrays = arrays.reset()

    # periodic_axes is needed during bridging so the non-uniform aniso width padding wraps to
    # match the field padding, so resolve it before building the state.
    periodic_axes = get_wrap_padding_axes(objects)
    state = to_mlx_state(arrays, config, periodic_axes, objects=objects)
    source_plans = freeze_sources(objects, config, arrays)
    detector_plans = freeze_detectors(objects, config)
    detector_buffers = allocate_buffers(detector_plans)
    num_steps = int(config.time_steps_total)
    c = float(config.courant_number)

    state, detector_buffers = run_forward_mlx(
        state,
        source_plans,
        detector_plans,
        detector_buffers,
        num_steps,
        c,
        simulate_boundaries=True,
        use_metal_kernel=_metal_kernel_enabled(),
    )

    detector_states = buffers_to_detector_states(detector_buffers) if detector_plans else None
    out_arrays = to_array_container(arrays, state, detector_states)
    return jnp.asarray(num_steps, dtype=jnp.int32), out_arrays
