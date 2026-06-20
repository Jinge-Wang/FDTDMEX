"""Apply frozen source plans into the E/H fields at a given step.

The per-step coefficient already folds in sign / courant / amplitude / on-off, so an
inactive step (``coeff[n] == 0``) is skipped entirely. Injection is a functional slice
add (out-of-place), keeping the update race-free.
"""

from __future__ import annotations

import mlx.core as mx

from fdtdx.mlx.source_freeze import SourcePlan


def _add_into_slice(field: mx.array, grid_slice: tuple, delta: mx.array) -> mx.array:
    """Return ``field`` with ``delta`` added into ``field[:, *grid_slice]`` (out-of-place)."""
    sl = (slice(None), *tuple(grid_slice))
    field[sl] = field[sl] + delta
    return field


def inject_sources_E(E: mx.array, plans: list[SourcePlan], n: int) -> mx.array:
    """Add all active electric-source contributions into E at step ``n``."""
    for p in plans:
        if p.field != "E":
            continue
        cval = float(p.coeff[n])
        if cval == 0.0:
            continue
        E = _add_into_slice(E, p.grid_slice, cval * p.inv_oriented)
    return E


def inject_sources_H(H: mx.array, plans: list[SourcePlan], n: int) -> mx.array:
    """Add all active magnetic-source contributions into H at step ``n``."""
    for p in plans:
        if p.field != "H":
            continue
        cval = float(p.coeff[n])
        if cval == 0.0:
            continue
        H = _add_into_slice(H, p.grid_slice, cval * p.inv_oriented)
    return H
