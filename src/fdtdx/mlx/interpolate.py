"""MLX field co-location for detectors.

Translation of ``fdtdx.core.physics.curl.interpolate_fields``: all six Yee components are
averaged onto the E_z grid point (i, j, k+1/2) via half-step averages. The backward
center-to-edge steps (along x and y) use ``_backward_edge_average``, which on a non-uniform
grid weights the two cell-center samples by the *opposite* half cell widths; the forward
edge-to-center z step lands on the exact geometric midpoint and stays a plain mean. On a
uniform grid (``interp_widths is None``) every step is a plain mean, matching the uniform path exactly.
Inputs are pre-padded (3, Nx+2, Ny+2, Nz+2).
"""

from __future__ import annotations

import mlx.core as mx

from fdtdx.mlx.curl import _sl


def _bea(current: mx.array, previous: mx.array, axis: int, interp_widths) -> mx.array:
    """Backward center-to-edge average along ``axis`` (port of ``_backward_edge_average``)."""
    if interp_widths is None:
        return 0.5 * (current + previous)
    cur_half, prev_half = interp_widths[axis]
    return (current * prev_half + previous * cur_half) / (cur_half + prev_half)


def interpolate_fields_mlx(E_pad: mx.array, H_pad: mx.array, interp_widths=None) -> tuple[mx.array, mx.array]:
    """Return (E_interp, H_interp), each (3, Nx, Ny, Nz), co-located at the E_z point."""
    E_x, E_y, E_z = E_pad[0], E_pad[1], E_pad[2]
    H_x, H_y, H_z = H_pad[0], H_pad[1], H_pad[2]

    # E_x: (i+1/2, j, k) -> (i, j, k+1/2): x backward (weighted), z forward (plain mean)
    E_x_lower_z = _bea(E_x[1:-1, 1:-1, 1:-1], E_x[:-2, 1:-1, 1:-1], 0, interp_widths)
    E_x_upper_z = _bea(E_x[1:-1, 1:-1, 2:], E_x[:-2, 1:-1, 2:], 0, interp_widths)
    E_x = (E_x_lower_z + E_x_upper_z) / 2.0

    # E_y: (i, j+1/2, k) -> (i, j, k+1/2): y backward (weighted), z forward (plain mean)
    E_y_lower_z = _bea(E_y[1:-1, 1:-1, 1:-1], E_y[1:-1, :-2, 1:-1], 1, interp_widths)
    E_y_upper_z = _bea(E_y[1:-1, 1:-1, 2:], E_y[1:-1, :-2, 2:], 1, interp_widths)
    E_y = (E_y_lower_z + E_y_upper_z) / 2.0

    # E_z: already at target
    E_z = E_z[1:-1, 1:-1, 1:-1]

    # H_x: (i, j+1/2, k+1/2) -> y backward (weighted)
    H_x = _bea(H_x[1:-1, 1:-1, 1:-1], H_x[1:-1, :-2, 1:-1], 1, interp_widths)
    # H_y: (i+1/2, j, k+1/2) -> x backward (weighted)
    H_y = _bea(H_y[1:-1, 1:-1, 1:-1], H_y[:-2, 1:-1, 1:-1], 0, interp_widths)

    # H_z: (i+1/2, j+1/2, k) -> x backward, y backward (weighted), z forward (plain mean)
    H_z_lower_z_x = _bea(H_z[1:-1, 1:-1, 1:-1], H_z[:-2, 1:-1, 1:-1], 0, interp_widths)
    H_z_lower_z_xy = _bea(
        H_z_lower_z_x, _bea(H_z[1:-1, :-2, 1:-1], H_z[:-2, :-2, 1:-1], 0, interp_widths), 1, interp_widths
    )
    H_z_upper_z_x = _bea(H_z[1:-1, 1:-1, 2:], H_z[:-2, 1:-1, 2:], 0, interp_widths)
    H_z_upper_z_xy = _bea(
        H_z_upper_z_x, _bea(H_z[1:-1, :-2, 2:], H_z[:-2, :-2, 2:], 0, interp_widths), 1, interp_widths
    )
    H_z = (H_z_lower_z_xy + H_z_upper_z_xy) / 2.0

    E_interp = mx.stack([E_x, E_y, E_z], axis=0)
    H_interp = mx.stack([H_x, H_y, H_z], axis=0)
    return E_interp, H_interp


def _bounds(sl: slice, n: int) -> tuple[int, int]:
    """``(start, stop)`` of a detector ``grid_slice`` entry, with ``None`` resolved against ``n``."""
    s = 0 if sl.start is None else int(sl.start)
    e = n if sl.stop is None else int(sl.stop)
    return s, e


def _region_padded_block(field: mx.array, grid_slice: tuple, periodic_axes: tuple) -> mx.array:
    """Extract ``pad_fields_mlx(field)[:, x0:x1+2, y0:y1+2, z0:z1+2]`` *without* padding the whole
    field. For each axis the window is the real crop ``field[s-1:e+1]`` (interior) plus, only when
    the region touches a true domain edge, the single ghost cell with the **same** semantics as
    ``pad_fields_mlx`` (zero on PML/PEC axes, the wrapped neighbour on periodic axes). The result
    feeds ``interpolate_fields_mlx`` unchanged, which strips the halo back to exactly the region.
    """
    full = field.shape  # (3, Nx, Ny, Nz)
    out = field
    for a in range(3):
        axis = a + 1
        n = full[axis]
        s, e = _bounds(grid_slice[a], n)
        core = _sl(out, max(0, s - 1), min(n, e + 1), axis)
        parts = [core]
        if s == 0:  # low domain edge -> synthesize the pad ghost
            ghost = _sl(out, n - 1, n, axis) if periodic_axes[a] else mx.zeros_like(_sl(out, 0, 1, axis))
            parts = [ghost, *parts]
        if e >= n:  # high domain edge
            ghost = _sl(out, 0, 1, axis) if periodic_axes[a] else mx.zeros_like(_sl(out, 0, 1, axis))
            parts = [*parts, ghost]
        out = mx.concatenate(parts, axis=axis) if len(parts) > 1 else core
    return out


def _slice_interp_widths(interp_widths, grid_slice: tuple):
    """Crop the per-axis ``(cur_half, prev_half)`` weight tables to the detector region so the
    weighted backward-average lines up with the region output. ``None`` (uniform) passes through."""
    if interp_widths is None:
        return None
    out = []
    for a in range(3):
        cur, prev = interp_widths[a]
        s, e = _bounds(grid_slice[a], cur.shape[a])
        out.append((_sl(cur, s, e, a), _sl(prev, s, e, a)))
    return tuple(out)


def interpolate_region_mlx(
    E_raw: mx.array,
    H_prev: mx.array,
    H_cur: mx.array,
    grid_slice: tuple,
    periodic_axes: tuple,
    interp_widths,
) -> tuple[mx.array, mx.array]:
    """Co-locate E and the time-averaged H onto the E_z point, but only over a detector's
    ``grid_slice`` (+ the 1-cell halo the stencil reads). Element-wise identical to slicing the
    full-domain ``interpolate_fields_mlx`` to the region: the H time-average commutes with the
    windowed pad (``0.5*(pad(H_prev)+pad(H_cur)) == pad(0.5*(H_prev+H_cur))``)."""
    E_sub = _region_padded_block(E_raw, grid_slice, periodic_axes)
    H_sub = 0.5 * (
        _region_padded_block(H_prev, grid_slice, periodic_axes) + _region_padded_block(H_cur, grid_slice, periodic_axes)
    )
    return interpolate_fields_mlx(E_sub, H_sub, _slice_interp_widths(interp_widths, grid_slice))
