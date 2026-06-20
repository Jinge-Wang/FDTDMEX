"""Record detector measurements into their buffers at a given step.

Mirrors each detector's ``update``. The detector sees interpolated E and the interpolated
time-averaged H when ``exact_interpolation`` is set (the default), else the raw E / raw new
H — matching ``fdtdx.fdtd.update.update_detector_states``. The on/off + time->row mapping are
host-precomputed, so an inactive step is skipped and the active row is a plain int.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from fdtdx.mlx.detector_freeze import DetectorPlan
from fdtdx.mlx.metrics import compute_energy_mlx


def _slice_material(mat: Any, grid_slice: tuple) -> Any:
    if hasattr(mat, "ndim") and getattr(mat, "ndim", 0) > 0:
        return mat[(slice(None), *grid_slice)]
    return mat


def update_detectors(
    plans: list[DetectorPlan],
    buffers: dict[str, dict[str, mx.array]],
    E_interp: mx.array,
    H_interp: mx.array,
    E_raw: mx.array,
    H_raw: mx.array,
    inv_eps: Any,
    inv_mu: Any,
    n: int,
) -> None:
    """Write step-``n`` measurements into ``buffers`` in place."""
    for p in plans:
        if not bool(p.on_steps[n]):
            continue
        row = int(p.time_to_idx[n])

        E = E_interp if p.exact_interp else E_raw
        H = H_interp if p.exact_interp else H_raw

        if p.kind == "energy":
            sl = (slice(None), *p.grid_slice)
            energy = compute_energy_mlx(
                E[sl], H[sl], _slice_material(inv_eps, p.grid_slice), _slice_material(inv_mu, p.grid_slice)
            )
            if p.reduce_volume:
                value = mx.sum(energy * p.cell_volume_weights).reshape(1)
            else:
                value = energy
            buf = buffers[p.name]["energy"]
            buf[row] = value
        else:  # pragma: no cover - guarded by the dispatcher
            raise NotImplementedError(f"MLX detector accumulate not implemented for kind={p.kind}")
