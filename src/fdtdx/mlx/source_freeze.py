"""Freeze placed source objects into plain-array injection plans.

Each fdtdx source reduces, at integer step ``n``, to a precomputed spatial array combined
with a host-precomputed per-step coefficient. The ``jax.lax.cond`` on/off gate and the
temporal profile are evaluated on the host, so the MLX loop needs no JAX objects.

Supported (uniform grid):
- :class:`PointDipoleSource` (soft add): ``field[:, slice] += coeff[n] * inv_oriented``.
- :class:`LinearlyPolarizedPlaneSource` (UniformPlaneSource / GaussianPlaneSource) TFSF
  injection in the isotropic/diagonal case, non-tilted (scalar Yee time offsets),
  non-dispersive. Tilted / dispersive / mode sources fall back to JAX (gated by dispatch).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import mlx.core as mx
import numpy as np

from fdtdx.objects.sources.dipole import PointDipoleSource
from fdtdx.objects.sources.linear_polarization import LinearlyPolarizedPlaneSource


def _to_mx(x) -> mx.array:
    return mx.array(np.ascontiguousarray(np.asarray(x)))


@dataclass
class SourcePlan:
    """Frozen source injection. ``kind`` selects the fields used."""

    kind: str  # "dipole" | "tfsf"
    grid_slice: tuple
    on_steps: np.ndarray
    # dipole (soft add into all 3 components of one field):
    dipole_field: str = "E"
    inv_oriented: mx.array | None = None
    coeff: np.ndarray | None = None
    # tfsf (per-component add into E[h]/E[v] and H[v]/H[h]):
    sign: float = 1.0
    h_axis: int = 0
    v_axis: int = 0
    spatialE_h: mx.array | None = None
    spatialE_v: mx.array | None = None
    spatialH_v: mx.array | None = None
    spatialH_h: mx.array | None = None
    # per-step per-cell temporal amplitudes, shape (T, *slice_shape):
    amp_H_v: mx.array | None = None
    amp_H_h: mx.array | None = None
    amp_E_h: mx.array | None = None
    amp_E_v: mx.array | None = None


def _dipole_plan(source: PointDipoleSource, config) -> SourcePlan:
    dt = float(config.time_step_duration)
    c = float(config.courant_number)
    period = float(source.wave_character.get_period())
    phase = float(source.wave_character.phase_shift)
    static = float(source.amplitude * source.static_amplitude_factor)
    sign = -1.0  # forward injection

    on_arr = np.asarray(source._is_on_at_time_step_arr)
    adj_idx = np.asarray(source._time_step_to_on_idx)
    n_steps = int(on_arr.shape[0])

    electric = source.source_type == "electric"
    half = 0.0 if electric else 0.5

    coeff = np.zeros(n_steps, dtype=np.float32)
    for n in range(n_steps):
        if not bool(on_arr[n]):
            continue
        adj = float(adj_idx[n]) + half
        amp = float(source.temporal_profile.get_amplitude(time=adj * dt, period=period, phase_shift=phase))
        coeff[n] = sign * c * static * amp

    oriented = source._inv_eps_oriented if electric else source._inv_mu_oriented
    return SourcePlan(
        kind="dipole",
        grid_slice=tuple(source.grid_slice),
        on_steps=on_arr,
        dipole_field="E" if electric else "H",
        inv_oriented=_to_mx(oriented),
        coeff=coeff,
    )


def _tfsf_plan(source: LinearlyPolarizedPlaneSource, config, arrays) -> SourcePlan:
    if getattr(source, "_temporal_H_filter", None) is not None:
        raise NotImplementedError("dispersive TFSF source not supported on MLX yet")

    dt = float(config.time_step_duration)
    c = float(config.courant_number)
    period = float(source.wave_character.get_period())
    phase = float(source.wave_character.phase_shift)
    static = float(source.static_amplitude_factor)
    sign = 1.0 if source.direction == "+" else -1.0

    h, v = source.horizontal_axis, source.vertical_axis
    on_arr = np.asarray(source._is_on_at_time_step_arr)
    adj = np.asarray(source._time_step_to_on_idx).astype(np.float64)

    off_E = np.asarray(source._time_offset_E)
    off_H = np.asarray(source._time_offset_H)

    E_prof = np.asarray(source._E)  # (3, *grid_shape)
    H_prof = np.asarray(source._H)

    gs = tuple(source.grid_slice)
    inv_eps = np.asarray(arrays.inv_permittivities)[(slice(None), *gs)]
    ne = inv_eps.shape[0]

    def eps_comp(a):  # JAX clamps OOB integer indexing -> isotropic uses component 0
        return inv_eps[min(a, ne - 1)]

    inv_mu_full = arrays.inv_permeabilities
    if hasattr(inv_mu_full, "ndim") and getattr(inv_mu_full, "ndim", 0) > 0:
        inv_mu = np.asarray(inv_mu_full)[(slice(None), *gs)]
        nm = inv_mu.shape[0]

        def mu_comp(a):
            return inv_mu[min(a, nm - 1)]
    else:
        mu_scalar = float(inv_mu_full)

        def mu_comp(a):
            return mu_scalar

    # Spatial parts of the scattered-field correction (mirrors tfsf.update_E/update_H).
    spatialE_h = H_prof[v] * c * eps_comp(h)
    spatialE_v = H_prof[h] * c * eps_comp(v)
    spatialH_v = E_prof[h] * c * mu_comp(v)
    spatialH_h = E_prof[v] * c * mu_comp(h)

    spatial_shape = spatialE_h.shape  # source grid_shape

    def amp(off_axis, half):
        """Per-step per-cell amplitude, broadcast to (T, *spatial_shape) -> mx."""
        off = np.asarray(off_axis)
        if off.ndim == 0:
            t = (adj + half + float(off))[:, None, None, None] * dt  # (T, 1, 1, 1)
        else:
            t = (adj.reshape((-1, 1, 1, 1)) + half + off[None, ...]) * dt  # (T, *off_shape)
        a = np.asarray(source.temporal_profile.get_amplitude(time=jnp.asarray(t), period=period, phase_shift=phase))
        a = (a * static).astype(np.float32)
        a = np.broadcast_to(a, (a.shape[0], *spatial_shape))
        return _to_mx(a)

    return SourcePlan(
        kind="tfsf",
        grid_slice=gs,
        on_steps=on_arr,
        sign=sign,
        h_axis=h,
        v_axis=v,
        spatialE_h=_to_mx(spatialE_h),
        spatialE_v=_to_mx(spatialE_v),
        spatialH_v=_to_mx(spatialH_v),
        spatialH_h=_to_mx(spatialH_h),
        amp_H_v=amp(off_H[v], 0.0),
        amp_H_h=amp(off_H[h], 0.0),
        amp_E_h=amp(off_E[h], 0.5),
        amp_E_v=amp(off_E[v], 0.5),
    )


def freeze_sources(objects, config, arrays) -> list[SourcePlan]:
    """Build the list of :class:`SourcePlan` for all (supported) sources."""
    plans: list[SourcePlan] = []
    for s in objects.sources:
        if isinstance(s, PointDipoleSource):
            plans.append(_dipole_plan(s, config))
        elif isinstance(s, LinearlyPolarizedPlaneSource):
            plans.append(_tfsf_plan(s, config, arrays))
        else:  # pragma: no cover - guarded by the dispatcher's support check
            raise NotImplementedError(f"MLX source freeze not implemented for {type(s).__name__}")
    return plans
