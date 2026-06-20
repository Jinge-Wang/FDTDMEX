"""Freeze placed source objects into plain-array injection plans.

Each fdtdx source reduces, at integer step ``n``, to "add a precomputed spatial array
times a scalar into a fixed Yee slice". The ``jax.lax.cond`` on/off gate and the temporal
profile collapse to a host-precomputed per-step coefficient ``coeff[n]`` (folding sign,
courant number, static amplitude, temporal amplitude and on/off), so the MLX loop needs no
JAX objects and an inactive step is simply skipped.

M1 supports :class:`fdtdx.objects.sources.dipole.PointDipoleSource` (electric -> E,
magnetic -> H). Plane / TFSF sources land in M2.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from fdtdx.objects.sources.dipole import PointDipoleSource


@dataclass
class SourcePlan:
    """Frozen per-source injection: ``field[:, grid_slice] += coeff[n] * inv_oriented``."""

    field: str  # "E" or "H"
    grid_slice: tuple  # (slice, slice, slice)
    inv_oriented: mx.array  # (3, *slice_shape)
    coeff: np.ndarray  # (T,) host scalars


def _dipole_plan(source: PointDipoleSource, config) -> SourcePlan:
    dt = float(config.time_step_duration)
    c = float(config.courant_number)
    period = float(source.wave_character.get_period())
    phase = float(source.wave_character.phase_shift)
    static = float(source.amplitude * source.static_amplitude_factor)
    sign = -1.0  # forward injection

    on_arr = np.asarray(source._is_on_at_time_step_arr)  # bool[T]
    adj_idx = np.asarray(source._time_step_to_on_idx)  # adjusted time-step index per step
    n_steps = int(on_arr.shape[0])

    electric = source.source_type == "electric"
    # H-dipole samples at (adj + 0.5) per update_H call site; E-dipole at adj.
    half = 0.0 if electric else 0.5

    coeff = np.zeros(n_steps, dtype=np.float32)
    for n in range(n_steps):
        if not bool(on_arr[n]):
            continue
        adj = float(adj_idx[n]) + half
        amp = float(source.temporal_profile.get_amplitude(time=adj * dt, period=period, phase_shift=phase))
        coeff[n] = sign * c * static * amp

    oriented = source._inv_eps_oriented if electric else source._inv_mu_oriented
    inv_oriented = mx.array(np.ascontiguousarray(np.asarray(oriented)))

    return SourcePlan(
        field="E" if electric else "H",
        grid_slice=tuple(source.grid_slice),
        inv_oriented=inv_oriented,
        coeff=coeff,
    )


def freeze_sources(objects, config) -> list[SourcePlan]:
    """Build the list of :class:`SourcePlan` for all (supported) sources."""
    plans: list[SourcePlan] = []
    for s in objects.sources:
        if isinstance(s, PointDipoleSource):
            plans.append(_dipole_plan(s, config))
        else:  # pragma: no cover - guarded by the dispatcher's support check
            raise NotImplementedError(f"MLX source freeze not implemented for {type(s).__name__}")
    return plans
