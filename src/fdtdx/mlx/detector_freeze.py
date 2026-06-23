"""Freeze placed detector objects into plain-array recording plans + buffers.

Mirrors ``fdtdx.fdtd.update.update_detector_states`` + each detector's ``update``. Buffers
are allocated from the detector's own ``init_state`` contract (``_shape_dtype_single_time_step``
x ``_num_latent_time_steps``) so the round-tripped ``detector_states`` are indistinguishable
from a JAX run and downstream ``draw_plot`` / S-params need no changes.

Supports EnergyDetector, FieldDetector, PoyntingFluxDetector and PhasorDetector (uniform
grid; the EnergyDetector slice mode still falls back to JAX). The on/off + time->row mapping
are host-precomputed; the phasor accumulates into a single latent row.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from fdtdx.objects.detectors.energy import EnergyDetector
from fdtdx.objects.detectors.field import FieldDetector
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.detectors.poynting_flux import PoyntingFluxDetector

# Oversampling margin for phasor DFT auto-subsampling: keep ~k samples per period of the highest
# recorded frequency. The FDTD dt is already ~10-20x below that Nyquist, so a stride > 1 still
# leaves a comfortable margin while cutting per-step recording/interpolation by that factor.
_DFT_OVERSAMPLE = 12

_DTYPE_MAP = {
    "float32": mx.float32,
    "float64": mx.float32,  # MLX GPU is float32-first
    "complex64": mx.complex64,
    "complex128": mx.complex64,
}

_COMPONENT_PICKS = {
    "Ex": ("E", 0),
    "Ey": ("E", 1),
    "Ez": ("E", 2),
    "Hx": ("H", 0),
    "Hy": ("H", 1),
    "Hz": ("H", 2),
}


def _mx_dtype(np_like) -> mx.Dtype:
    name = np.dtype(np_like).name
    if name not in _DTYPE_MAP:
        raise NotImplementedError(f"MLX detector buffer dtype {name} not supported yet")
    return _DTYPE_MAP[name]


def _to_mx(x) -> mx.array:
    return mx.array(np.ascontiguousarray(np.asarray(x)))


@dataclass
class DetectorPlan:
    """Frozen per-detector recording metadata. Kind-specific fields are optional."""

    name: str
    kind: str  # "energy" | "field" | "poynting" | "phasor"
    buffer_key: str  # "energy" | "fields" | "poynting_flux" | "phasor"
    grid_slice: tuple
    on_steps: np.ndarray
    time_to_idx: np.ndarray
    exact_interp: bool
    reduce_volume: bool
    buffer_shapes: dict[str, tuple]
    buffer_dtypes: dict[str, mx.Dtype]
    # energy / field / phasor (reduce): physical cell-volume weights, shape (Nx, Ny, Nz)
    cell_volume_weights: mx.array | None = None
    # field / phasor: ordered list of (which_field, component_index)
    component_picks: list = field(default_factory=list)
    # poynting
    keep_all_components: bool = False
    propagation_axis: int = 0
    direction_sign: float = 1.0
    face_area_weights: mx.array | None = None
    # phasor: per-step exp(i*omega*n*dt) factors (T, num_freqs) complex, and the static scale
    phasors: mx.array | None = None
    static_scale: float = 1.0


def _buffers_meta(d):
    sd = d._shape_dtype_single_time_step()
    latent = d._num_latent_time_steps()
    shapes = {name: (latent, *spec.shape) for name, spec in sd.items()}
    dtypes = {name: _mx_dtype(spec.dtype) for name, spec in sd.items()}
    return shapes, dtypes


def _common(d, kind, buffer_key):
    shapes, dtypes = _buffers_meta(d)
    return dict(
        name=d.name,
        kind=kind,
        buffer_key=buffer_key,
        grid_slice=tuple(d.grid_slice),
        on_steps=np.asarray(d._is_on_at_time_step_arr),
        time_to_idx=np.asarray(d._time_step_to_arr_idx),
        exact_interp=bool(d.exact_interpolation),
        reduce_volume=bool(getattr(d, "reduce_volume", False)),
        buffer_shapes=shapes,
        buffer_dtypes=dtypes,
    )


def _energy_plan(d: EnergyDetector, config) -> DetectorPlan:
    if d.as_slices:
        raise NotImplementedError("EnergyDetector(as_slices=True) not supported on MLX yet")
    return DetectorPlan(**_common(d, "energy", "energy"), cell_volume_weights=_to_mx(d._cached_cell_volume_weights))


def _field_plan(d: FieldDetector, config) -> DetectorPlan:
    return DetectorPlan(
        **_common(d, "field", "fields"),
        cell_volume_weights=_to_mx(d._cached_cell_volume_weights),
        component_picks=[_COMPONENT_PICKS[c] for c in d.components],
    )


def _poynting_plan(d: PoyntingFluxDetector, config) -> DetectorPlan:
    plan = DetectorPlan(
        **_common(d, "poynting", "poynting_flux"),
        keep_all_components=bool(d.keep_all_components),
        propagation_axis=int(d.propagation_axis),
        direction_sign=-1.0 if d.direction == "-" else 1.0,
    )
    if d.reduce_volume:
        plan.face_area_weights = _to_mx(d._cached_face_area_weights)
    return plan


def _dft_stride(omega: np.ndarray, dt: float) -> int:
    """Steps between phasor samples. ``FDTDMEX_DFT_STRIDE`` overrides (``=1`` forces every step,
    the exact-parity mode the element-wise oracle tests use); otherwise auto from the highest
    frequency: keep ~``_DFT_OVERSAMPLE`` samples per its period, ``floor(1/(k*f_max*dt))``."""
    env = os.environ.get("FDTDMEX_DFT_STRIDE")
    if env is not None:
        return max(1, int(env))
    f_max = float(np.max(np.abs(omega))) / (2.0 * np.pi)
    if f_max <= 0.0 or dt <= 0.0:
        return 1
    return max(1, math.floor(1.0 / (_DFT_OVERSAMPLE * f_max * dt)))


def _phasor_plan(d: PhasorDetector, config) -> DetectorPlan:
    dt = float(config.time_step_duration)
    num_steps = int(config.time_steps_total)
    omega = np.asarray(d._angular_frequencies)  # (num_freqs,)
    times = np.arange(num_steps, dtype=np.float64) * dt
    phasors = np.exp(1j * omega[None, :] * times[:, None]).astype(np.complex64)  # (T, num_freqs)

    common = _common(d, "phasor", "phasor")
    # Auto-subsample the running DFT: keep every ``stride``-th otherwise-active step. The phasor
    # table is indexed by the global step, so it stays correct on the kept steps; only ``on_steps``
    # is thinned. Each kept sample then stands in for ``stride`` steps, so the normalization gains a
    # factor ``stride`` (the Riemann-sum weight) to match the every-step magnitude.
    stride = _dft_stride(omega, dt)
    if stride > 1:
        on = common["on_steps"].astype(bool)
        kept = np.zeros_like(on)
        kept[np.nonzero(on)[0][::stride]] = True
        common["on_steps"] = kept

    if d.scaling_mode == "continuous":
        static_scale = (2.0 / int(d._num_time_steps_on)) * stride
    elif d.scaling_mode == "pulse":
        static_scale = 1.0 * stride
    else:
        raise NotImplementedError(f"PhasorDetector scaling_mode={d.scaling_mode!r} not supported on MLX yet")
    return DetectorPlan(
        **common,
        cell_volume_weights=_to_mx(d._cached_cell_volume_weights),
        component_picks=[_COMPONENT_PICKS[c] for c in d.components],
        phasors=_to_mx(phasors),
        static_scale=static_scale,
    )


_BUILDERS = (
    (EnergyDetector, _energy_plan),
    (FieldDetector, _field_plan),
    (PoyntingFluxDetector, _poynting_plan),
    (PhasorDetector, _phasor_plan),
)


def freeze_detectors(objects, config) -> list[DetectorPlan]:
    """Build :class:`DetectorPlan` list for all (forward, supported) detectors."""
    plans: list[DetectorPlan] = []
    for d in objects.forward_detectors:
        for cls, builder in _BUILDERS:
            if isinstance(d, cls):
                plans.append(builder(d, config))
                break
        else:  # pragma: no cover - guarded by the dispatcher
            raise NotImplementedError(f"MLX detector freeze not implemented for {type(d).__name__}")
    return plans


def allocate_buffers(plans: list[DetectorPlan]) -> dict[str, dict[str, mx.array]]:
    """Allocate zeroed recording buffers for every detector plan."""
    return {
        p.name: {name: mx.zeros(shape, dtype=p.buffer_dtypes[name]) for name, shape in p.buffer_shapes.items()}
        for p in plans
    }
