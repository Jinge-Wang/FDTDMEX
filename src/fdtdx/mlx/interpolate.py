"""MLX field co-location for detectors.

Translation of ``fdtdx.core.physics.curl.interpolate_fields``: all six Yee components are
averaged onto the E_z grid point (i, j, k+1/2) via half-step averages. The backward
center-to-edge steps (along x and y) use ``_backward_edge_average``, which on a non-uniform
grid weights the two cell-center samples by the *opposite* half cell widths; the forward
edge-to-center z step lands on the exact geometric midpoint and stays a plain mean. On a
uniform grid (``interp_widths is None``) every step is a plain mean, matching M1-M3 exactly.
Inputs are pre-padded (3, Nx+2, Ny+2, Nz+2).
"""

from __future__ import annotations

import mlx.core as mx


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
