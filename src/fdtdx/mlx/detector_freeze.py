"""Freeze placed detector objects into plain-array recording plans + buffers.

Mirrors ``fdtdx.fdtd.update.update_detector_states`` + each detector's ``update``. Buffers
are allocated from the detector's own ``init_state`` contract (``_shape_dtype_single_time_step``
x ``_num_latent_time_steps``) so the round-tripped ``detector_states`` are indistinguishable
from a JAX run and downstream ``draw_plot`` / S-params need no changes.

Supports EnergyDetector, FieldDetector and PoyntingFluxDetector (uniform grid; the
EnergyDetector slice mode still falls back to JAX). The on/off + time->row mapping are
host-precomputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from fdtdx.objects.detectors.energy import EnergyDetector
from fdtdx.objects.detectors.field import FieldDetector
from fdtdx.objects.detectors.poynting_flux import PoyntingFluxDetector

_DTYPE_MAP = {
    "float32": mx.float32,
    "float64": mx.float32,  # MLX GPU is float32-first
    "complex64": mx.complex64,
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
    kind: str  # "energy" | "field" | "poynting"
    buffer_key: str  # "energy" | "fields" | "poynting_flux"
    grid_slice: tuple
    on_steps: np.ndarray
    time_to_idx: np.ndarray
    exact_interp: bool
    reduce_volume: bool
    buffer_shapes: dict[str, tuple]
    buffer_dtypes: dict[str, mx.Dtype]
    # energy / field (reduce): physical cell-volume weights, shape (Nx, Ny, Nz)
    cell_volume_weights: mx.array | None = None
    # field: ordered list of (which_field, component_index)
    component_picks: list = field(default_factory=list)
    # poynting
    keep_all_components: bool = False
    propagation_axis: int = 0
    direction_sign: float = 1.0
    face_area_weights: mx.array | None = None


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


def _energy_plan(d: EnergyDetector) -> DetectorPlan:
    if d.as_slices:
        raise NotImplementedError("EnergyDetector(as_slices=True) not supported on MLX yet")
    return DetectorPlan(
        **_common(d, "energy", "energy"),
        cell_volume_weights=_to_mx(d._cached_cell_volume_weights),
    )


def _field_plan(d: FieldDetector) -> DetectorPlan:
    return DetectorPlan(
        **_common(d, "field", "fields"),
        cell_volume_weights=_to_mx(d._cached_cell_volume_weights),
        component_picks=[_COMPONENT_PICKS[c] for c in d.components],
    )


def _poynting_plan(d: PoyntingFluxDetector) -> DetectorPlan:
    plan = DetectorPlan(
        **_common(d, "poynting", "poynting_flux"),
        keep_all_components=bool(d.keep_all_components),
        propagation_axis=int(d.propagation_axis),
        direction_sign=-1.0 if d.direction == "-" else 1.0,
    )
    if d.reduce_volume:
        plan.face_area_weights = _to_mx(d._cached_face_area_weights)
    return plan


_BUILDERS = (
    (EnergyDetector, _energy_plan),
    (FieldDetector, _field_plan),
    (PoyntingFluxDetector, _poynting_plan),
)


def freeze_detectors(objects, config) -> list[DetectorPlan]:
    """Build :class:`DetectorPlan` list for all (forward, supported) detectors."""
    plans: list[DetectorPlan] = []
    for d in objects.forward_detectors:
        for cls, builder in _BUILDERS:
            if isinstance(d, cls):
                plans.append(builder(d))
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
