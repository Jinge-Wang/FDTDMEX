"""Freeze placed detector objects into plain-array recording plans + buffers.

Mirrors ``fdtdx.fdtd.update.update_detector_states`` + each detector's ``update``. Buffers
are allocated from the detector's own ``init_state`` contract (``_shape_dtype_single_time_step``
x ``_num_latent_time_steps``) so the round-tripped ``detector_states`` are indistinguishable
from a JAX run and downstream ``draw_plot`` / S-params need no changes.

M1b supports :class:`fdtdx.objects.detectors.energy.EnergyDetector` in full-volume and
``reduce_volume`` modes. Slice / Field / Poynting / Phasor detectors land later.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from fdtdx.objects.detectors.energy import EnergyDetector

_DTYPE_MAP = {
    "float32": mx.float32,
    "float64": mx.float32,  # MLX GPU is float32-first; energy is real
    "complex64": mx.complex64,
}


def _mx_dtype(np_like) -> mx.Dtype:
    name = np.dtype(np_like).name
    if name not in _DTYPE_MAP:
        raise NotImplementedError(f"MLX detector buffer dtype {name} not supported yet")
    return _DTYPE_MAP[name]


@dataclass
class DetectorPlan:
    """Frozen per-detector recording metadata."""

    name: str
    kind: str  # "energy"
    grid_slice: tuple
    on_steps: np.ndarray  # bool[T]
    time_to_idx: np.ndarray  # int[T] -> buffer row (or -1 when off)
    exact_interp: bool
    reduce_volume: bool
    cell_volume_weights: mx.array  # (Nx, Ny, Nz)
    buffer_shapes: dict[str, tuple]
    buffer_dtypes: dict[str, mx.Dtype]


def _energy_plan(d: EnergyDetector, num_steps: int) -> DetectorPlan:
    if d.as_slices:
        raise NotImplementedError("EnergyDetector(as_slices=True) not supported on MLX yet")

    sd = d._shape_dtype_single_time_step()
    latent = d._num_latent_time_steps()
    buffer_shapes = {name: (latent, *spec.shape) for name, spec in sd.items()}
    buffer_dtypes = {name: _mx_dtype(spec.dtype) for name, spec in sd.items()}

    return DetectorPlan(
        name=d.name,
        kind="energy",
        grid_slice=tuple(d.grid_slice),
        on_steps=np.asarray(d._is_on_at_time_step_arr),
        time_to_idx=np.asarray(d._time_step_to_arr_idx),
        exact_interp=bool(d.exact_interpolation),
        reduce_volume=bool(d.reduce_volume),
        cell_volume_weights=mx.array(np.ascontiguousarray(np.asarray(d._cached_cell_volume_weights))),
        buffer_shapes=buffer_shapes,
        buffer_dtypes=buffer_dtypes,
    )


def freeze_detectors(objects, config) -> list[DetectorPlan]:
    """Build :class:`DetectorPlan` list for all (forward, supported) detectors."""
    num_steps = int(config.time_steps_total)
    plans: list[DetectorPlan] = []
    for d in objects.forward_detectors:
        if isinstance(d, EnergyDetector):
            plans.append(_energy_plan(d, num_steps))
        else:  # pragma: no cover - guarded by the dispatcher
            raise NotImplementedError(f"MLX detector freeze not implemented for {type(d).__name__}")
    return plans


def allocate_buffers(plans: list[DetectorPlan]) -> dict[str, dict[str, mx.array]]:
    """Allocate zeroed recording buffers for every detector plan."""
    buffers: dict[str, dict[str, mx.array]] = {}
    for p in plans:
        buffers[p.name] = {
            name: mx.zeros(shape, dtype=p.buffer_dtypes[name]) for name, shape in p.buffer_shapes.items()
        }
    return buffers
