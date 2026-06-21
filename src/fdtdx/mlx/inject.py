"""Apply frozen source plans into the E/H fields at a given step.

Dipole sources fold sign/courant/amplitude/on-off into ``coeff[n]`` (a zero coeff is
skipped). TFSF plane sources add the precomputed scattered-field spatial parts times a
host-precomputed per-step temporal amplitude into the two transverse components, gated by
``on_steps[n]``. All injections are functional (out-of-place), keeping the update race-free.
"""

from __future__ import annotations

import mlx.core as mx

from fdtdx.mlx.source_freeze import SourcePlan


def _add_all(field: mx.array, grid_slice: tuple, delta: mx.array) -> mx.array:
    """Add ``delta`` into ``field[:, *grid_slice]`` (all 3 components)."""
    sl = (slice(None), *tuple(grid_slice))
    field[sl] = field[sl] + delta
    return field


def _add_component(field: mx.array, axis: int, grid_slice: tuple, delta: mx.array) -> mx.array:
    """Add ``delta`` into ``field[axis, *grid_slice]`` (one component)."""
    sl = (axis, *tuple(grid_slice))
    field[sl] = field[sl] + delta
    return field


def inject_sources_E(E: mx.array, plans: list[SourcePlan], n: int) -> mx.array:
    """Add all active electric-source contributions into E at step ``n``."""
    for p in plans:
        if p.kind == "dipole":
            if p.dipole_field != "E":
                continue
            cval = float(p.coeff[n])
            if cval != 0.0:
                E = _add_all(E, p.grid_slice, cval * p.inv_oriented)
        elif p.kind == "tfsf" and bool(p.on_steps[n]):
            E = _add_component(E, p.h_axis, p.grid_slice, (p.sign * p.amp_H_v[n]) * p.spatialE_h)
            E = _add_component(E, p.v_axis, p.grid_slice, (-p.sign * p.amp_H_h[n]) * p.spatialE_v)
    return E


def inject_sources_H(H: mx.array, plans: list[SourcePlan], n: int) -> mx.array:
    """Add all active magnetic-source contributions into H at step ``n``."""
    for p in plans:
        if p.kind == "dipole":
            if p.dipole_field != "H":
                continue
            cval = float(p.coeff[n])
            if cval != 0.0:
                H = _add_all(H, p.grid_slice, cval * p.inv_oriented)
        elif p.kind == "tfsf" and bool(p.on_steps[n]):
            H = _add_component(H, p.v_axis, p.grid_slice, (p.sign * p.amp_E_h[n]) * p.spatialH_v)
            H = _add_component(H, p.h_axis, p.grid_slice, (-p.sign * p.amp_E_v[n]) * p.spatialH_h)
    return H
