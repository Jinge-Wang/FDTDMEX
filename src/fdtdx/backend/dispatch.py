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
  features the MLX engine supports; otherwise JAX (warn-once on the first decline).

Feature gating lives in ``_unsupported_reason`` / ``_unsupported_reason_arrays``. The MLX path covers iso/diag/full-tensor anisotropy (incl. lossy + 9-tensor
conductivity), CPML + periodic + PEC/PMC boundaries, dipole + (tilted) TFSF plane sources, the four
detector types, non-uniform (rectilinear) grids, Drude-Lorentz (ADE) dispersion, and the three
built-in stopping conditions (``fdtdx.mlx.stop``).
Still gated to JAX: gradients, dispersive/randomized plane sources, Bloch/complex propagation,
mode sources/detectors, and any other ``StoppingCondition`` subclass.
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


# Source/detector types the MLX engine currently handles.
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

    Default **on**: CPML is folded into the kernel, with the non-uniform metric and heterogeneous
    full-tensor inclusions covered across the eligible surface. The loop still falls
    back to the compiled MLX-op cores for any case the kernel can't handle (``kernel_eligible``).
    Set ``FDTDMEX_METAL_KERNEL=0`` (or ``false``/``no``/``off``) to force the MLX-op path.
    """
    return os.environ.get("FDTDMEX_METAL_KERNEL", "").lower() not in ("0", "false", "no", "off")


def _unsupported_reason(config, objects, stopping_condition) -> str | None:
    """Return a human-readable reason the case can't run on MLX yet, or ``None``."""
    if config.gradient_config is not None:
        return "gradient computation requested (MLX backend is forward-only)"
    if stopping_condition is not None:
        # TimeStep / EnergyThreshold / DetectorConvergence are reduced to a StopPlan and evaluated
        # at the loop's eval cadence; any other subclass keeps the JAX fallback.
        from fdtdx.mlx.stop import stop_condition_unsupported_reason

        reason = stop_condition_unsupported_reason(stopping_condition, config, objects)
        if reason is not None:
            return reason
    if getattr(config, "use_complex_fields", None) is True:
        return "forced complex fields not supported by the MLX backend yet"
    # Mirror-symmetry reduction (config.symmetry) lives in the JAX curl/halo code
    # (fdtdx.fdtd.update.pad_fields_with_symmetry_mirror) and the mode-source unfolding; the MLX
    # loop has no equivalent, so a reduced domain would run without its mirror planes.
    if any(s != 0 for s in getattr(config, "symmetry", (0, 0, 0))):
        return "config.symmetry (mirror-reduced domain) not supported by the MLX backend yet"
    for b in objects.bloch_objects:
        if b.needs_complex_fields:
            return "Bloch (nonzero-k, complex) boundaries not supported by the MLX backend yet"
    # PEC/PMC are supported: frozen keep-masks applied post-injection in the loop
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

    from fdtdx.objects.detectors.phasor import PhasorDetector

    supported_detectors = _supported_detector_types()
    for d in objects.detectors:
        if not isinstance(d, supported_detectors):
            return f"detector type {type(d).__name__} not supported by the MLX backend yet"
        if getattr(d, "as_slices", False):
            return f"{type(d).__name__}(as_slices=True) not supported by the MLX backend yet"
        if isinstance(d, PhasorDetector):
            # Upstream subclasses (PhasorPoyntingFluxDetector, ClosedSurfacePhasorPoyntingFluxDetector)
            # post-process the phasors; the MLX phasor plan only knows the plain running DFT.
            if type(d) is not PhasorDetector:
                return f"detector type {type(d).__name__} not supported by the MLX backend yet"
            # Temporal apodization windows (#428) and explicit DFT subsampling (#406) are not
            # threaded through the MLX phasor plan.
            if getattr(d, "apodization", None) is not None:
                return "PhasorDetector(apodization=...) not supported by the MLX backend yet"
            if getattr(d, "dft_subsample", 1) != 1:
                return "PhasorDetector(dft_subsample!=1) not supported by the MLX backend yet"

    return None


def _unsupported_reason_arrays(arrays) -> str | None:
    """Material/array-level support checks (need the ArrayContainer)."""
    # lossy full-tensor (9-component) anisotropy and 9-tensor (full-rank) electric/magnetic
    # conductivity are supported -- the aniso A/B update (``_update_aniso``) consumes ``sigma``
    # directly (``compute_anisotropic_update_matrices_mlx``), so these run on the MLX-op cores (the
    # lossless block-hybrid Metal kernel stays as-is; ``kernel_eligible`` falls these back).
    #
    # Drude-Lorentz (ADE) dispersion is supported -- polarization P is threaded
    # through the E-side of the loop (``mlx.update._update_E`` / the Metal E-kernel ADE fold), with
    # coefficients carried in ``MLXState``. fdtdx forbids dispersion + off-diagonal tensors, so it is
    # always iso/diagonal: lossless rides the Metal kernel, lossy+dispersive uses the MLX-op cores.
    # (Dispersive *plane sources* remain gated separately in ``_unsupported_reason``.)
    #
    # CCPR dispersion (upstream #383) adds a ``b * dE/dt`` term carried in a 4th ADE coefficient
    # ``dispersive_c4`` (non-``None`` only when a pole has non-zero ``coupling_edot``). The MLX ADE
    # fold only threads ``c1/c2/c3``, so a CCPR pole would silently drop that term -> gate to JAX.
    # (Non-dispersive *complex* permittivity/conductivity, #382, needs no gate: ``from_complex_*``
    # splits it into a real ε plus an equivalent conductivity, i.e. the already-supported lossy path.)
    if getattr(arrays, "dispersive_c4", None) is not None:
        return "CCPR dispersion (dispersive_c4 / dE-dt term) not supported by the MLX backend yet"
    return None


def select_backend(arrays, objects, config, stopping_condition) -> Backend:
    """Return the backend to use, honoring forced overrides and feature gating."""
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
    return _run_mlx_forward(arrays, objects, config, stopping_condition)


def run_forward_from_plans(
    state,
    source_plans,
    detector_plans,
    num_steps,
    courant,
    *,
    simulate_boundaries=True,
    stop_plan=None,
    progress=None,
):
    """Run the MLX forward time loop from an already-resolved ``MLXState`` + frozen plans.

    This is the post-freeze tail of :func:`_run_mlx_forward`, factored out so the HDF5 IO layer
    (``fdtdmex.io.sim_run``) can drive a run from a *deserialized* state + plans — no
    ``ObjectContainer`` and no re-resolution needed. Returns
    ``(final_state, detector_states, steps_run)`` where ``detector_states`` is the host (jnp)
    ``{name: {key: array}}`` mapping (or ``None`` when there are no detectors) and ``steps_run`` is
    the number of steps actually executed — ``num_steps`` unless ``stop_plan`` ended the run early.

    ``stop_plan`` is an optional :class:`fdtdx.mlx.stop.StopPlan`; ``progress``, when given, is
    forwarded to the time loop and called ``progress(step, num_steps)`` for streamed run telemetry
    (default ``None`` = no telemetry, no overhead).
    """
    from fdtdx.mlx.bridge import buffers_to_detector_states
    from fdtdx.mlx.detector_freeze import allocate_buffers
    from fdtdx.mlx.loop import run_forward_mlx

    detector_buffers = allocate_buffers(detector_plans)
    state, detector_buffers, steps_run = run_forward_mlx(
        state,
        source_plans,
        detector_plans,
        detector_buffers,
        int(num_steps),
        float(courant),
        simulate_boundaries=simulate_boundaries,
        use_metal_kernel=_metal_kernel_enabled(),
        stop_plan=stop_plan,
        progress=progress,
    )
    detector_states = buffers_to_detector_states(detector_buffers) if detector_plans else None
    return state, detector_states, steps_run


#: Default stop-check cadence in steps; the loop snaps it to a multiple of its own ``eval_every``
#: so a check never adds a synchronisation point of its own.
STOP_CHECK_EVERY = 8


def stop_check_every() -> int:
    """Steps between stopping-condition checks (env ``FDTDMEX_STOP_CHECK_EVERY``, default 8).

    A check is one fused reduction over E/H plus a scalar sync, on a step where the loop already
    synchronises. Measured on an M4 Pro, 96^3 isotropic box, Metal kernel on, 200 steps (median of
    five): 2619 steps/s with no condition, 2242 at the default cadence of 8 (-14 %), 2406 at 16
    (-8 %), 2504 at 32 (-4 %). Raising the cadence buys that throughput back and costs stop
    precision: a run can overshoot the JAX stop step by up to one interval.
    """
    raw = os.environ.get("FDTDMEX_STOP_CHECK_EVERY", "")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning(f"ignoring invalid FDTDMEX_STOP_CHECK_EVERY={raw!r}, using {STOP_CHECK_EVERY}")
    return STOP_CHECK_EVERY


def _run_mlx_forward(arrays, objects, config, stopping_condition=None):
    import jax.numpy as jnp

    from fdtdx.fdtd.update import get_wrap_padding_axes
    from fdtdx.mlx.bridge import to_array_container, to_mlx_state
    from fdtdx.mlx.detector_freeze import freeze_detectors
    from fdtdx.mlx.source_freeze import freeze_sources
    from fdtdx.mlx.stop import build_stop_plan

    # Match checkpointed_fdtd: zero dynamic fields + detector states before stepping.
    arrays = arrays.reset()

    # Same (0, arrays) pair checkpointed_fdtd feeds to the condition's setup(), so the resolved
    # defaults and the pre-run validation are upstream's.
    stop_plan = build_stop_plan(
        stopping_condition,
        (jnp.asarray(0, dtype=jnp.int32), arrays),
        config,
        objects,
        check_every=stop_check_every(),
    )

    # periodic_axes is needed during bridging so the non-uniform aniso width padding wraps to
    # match the field padding, so resolve it before building the state.
    periodic_axes = get_wrap_padding_axes(objects)
    state = to_mlx_state(arrays, config, periodic_axes, objects=objects)
    source_plans = freeze_sources(objects, config, arrays)
    detector_plans = freeze_detectors(objects, config)
    num_steps = int(config.time_steps_total)
    if stop_plan is not None:
        num_steps = min(num_steps, stop_plan.max_steps)
    c = float(config.courant_number)

    state, detector_states, steps_run = run_forward_from_plans(
        state, source_plans, detector_plans, num_steps, c, stop_plan=stop_plan
    )
    out_arrays = to_array_container(arrays, state, detector_states, objects=objects)
    # Same contract as the JAX early-stop path: the returned time step is the number of steps that
    # actually ran, and detector rows for steps that never ran keep their reset() zeros.
    return jnp.asarray(steps_run, dtype=jnp.int32), out_arrays
