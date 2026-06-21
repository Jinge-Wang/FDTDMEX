"""Record detector measurements into their buffers at a given step.

Mirrors each detector's ``update``. The detector sees interpolated E and the interpolated
time-averaged H when ``exact_interpolation`` is set (the default), else the raw E / raw new
H — matching ``fdtdx.fdtd.update.update_detector_states``. The on/off + time->row mapping are
host-precomputed, so an inactive step is skipped and the active row is a plain int.

Supports EnergyDetector / FieldDetector / PoyntingFluxDetector (uniform grid).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from fdtdx.mlx.detector_freeze import DetectorPlan
from fdtdx.mlx.metrics import compute_energy_mlx, compute_poynting_flux_mlx


def _slice_material(mat: Any, grid_slice: tuple) -> Any:
    if hasattr(mat, "ndim") and getattr(mat, "ndim", 0) > 0:
        return mat[(slice(None), *grid_slice)]
    return mat


def _volume_weighted_spatial_mean(values: mx.array, weights: mx.array, leading_dims: int) -> mx.array:
    """Average over the trailing spatial axes weighted by physical cell volumes.

    Weights are normalized to sum to 1 *before* the contraction. Physical cell volumes are
    tiny (spacing**3, ~1e-21), and MLX complex/real division computes ``|denom|**2`` which
    underflows in float32 for such small denominators — normalizing first avoids dividing by
    a tiny number while staying mathematically identical.
    """
    norm_weights = weights / mx.sum(weights)
    weight_shape = (1,) * leading_dims + norm_weights.shape
    spatial_axes = tuple(range(leading_dims, values.ndim))
    return mx.sum(values * norm_weights.reshape(weight_shape), axis=spatial_axes)


def _record_energy(p: DetectorPlan, E: mx.array, H: mx.array, inv_eps: Any, inv_mu: Any) -> mx.array:
    sl = (slice(None), *p.grid_slice)
    energy = compute_energy_mlx(
        E[sl], H[sl], _slice_material(inv_eps, p.grid_slice), _slice_material(inv_mu, p.grid_slice)
    )
    if p.reduce_volume:
        return mx.sum(energy * p.cell_volume_weights).reshape(1)
    return energy


def _record_field(p: DetectorPlan, E: mx.array, H: mx.array) -> mx.array:
    Esl = E[(slice(None), *p.grid_slice)]
    Hsl = H[(slice(None), *p.grid_slice)]
    parts = [(Esl[idx] if which == "E" else Hsl[idx]) for which, idx in p.component_picks]
    EH = mx.stack(parts, axis=0)
    if p.reduce_volume:
        EH = _volume_weighted_spatial_mean(EH, p.cell_volume_weights, leading_dims=1)
    return EH


def _record_poynting(p: DetectorPlan, E: mx.array, H: mx.array) -> mx.array:
    Esl = E[(slice(None), *p.grid_slice)]
    Hsl = H[(slice(None), *p.grid_slice)]
    pf = compute_poynting_flux_mlx(Esl, Hsl)
    if not p.keep_all_components:
        pf = pf[p.propagation_axis]
    pf = p.direction_sign * pf
    if p.reduce_volume:
        pf = pf * p.face_area_weights
        if p.keep_all_components:
            pf = mx.sum(pf, axis=(1, 2, 3))
        else:
            pf = mx.sum(pf).reshape(1)
    return pf


def _record_phasor(p: DetectorPlan, E: mx.array, H: mx.array, n: int) -> mx.array:
    Esl = E[(slice(None), *p.grid_slice)]
    Hsl = H[(slice(None), *p.grid_slice)]
    parts = [(Esl[idx] if which == "E" else Hsl[idx]) for which, idx in p.component_picks]
    EH = mx.stack(parts, axis=0)  # (C, *grid) real

    ph = p.phasors[n]  # (num_freqs,) complex
    ph = ph.reshape((ph.shape[0], *(1,) * EH.ndim))  # (F, 1, 1, 1, 1)
    new = (EH.astype(ph.dtype) * ph) * p.static_scale  # (F, C, *grid) complex
    if p.reduce_volume:
        new = _volume_weighted_spatial_mean(new, p.cell_volume_weights, leading_dims=2)
    return new


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

        if p.kind == "phasor":
            buf = buffers[p.name]["phasor"]
            buf[0] = buf[0] + _record_phasor(p, E, H, n)
            continue

        if p.kind == "energy":
            value = _record_energy(p, E, H, inv_eps, inv_mu)
        elif p.kind == "field":
            value = _record_field(p, E, H)
        elif p.kind == "poynting":
            value = _record_poynting(p, E, H)
        else:  # pragma: no cover - guarded by the dispatcher
            raise NotImplementedError(f"MLX detector accumulate not implemented for kind={p.kind}")

        buffers[p.name][p.buffer_key][row] = value
