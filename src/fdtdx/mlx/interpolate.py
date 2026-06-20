"""MLX field co-location for detectors (uniform grid).

Translation of ``fdtdx.core.physics.curl.interpolate_fields``: all six Yee components are
averaged onto the E_z grid point (i, j, k+1/2) via half-step averages. Uniform-grid only
(``_backward_edge_average`` reduces to the arithmetic mean); spacing-weighted non-uniform
interpolation lands in M4. Inputs are pre-padded (3, Nx+2, Ny+2, Nz+2).
"""

from __future__ import annotations

import mlx.core as mx


def _avg(current: mx.array, previous: mx.array) -> mx.array:
    return 0.5 * (current + previous)


def interpolate_fields_mlx(E_pad: mx.array, H_pad: mx.array) -> tuple[mx.array, mx.array]:
    """Return (E_interp, H_interp), each (3, Nx, Ny, Nz), co-located at the E_z point."""
    E_x, E_y, E_z = E_pad[0], E_pad[1], E_pad[2]
    H_x, H_y, H_z = H_pad[0], H_pad[1], H_pad[2]

    # E_x: (i+1/2, j, k) -> (i, j, k+1/2): x backward, z forward
    E_x_lower_z = _avg(E_x[1:-1, 1:-1, 1:-1], E_x[:-2, 1:-1, 1:-1])
    E_x_upper_z = _avg(E_x[1:-1, 1:-1, 2:], E_x[:-2, 1:-1, 2:])
    E_x = (E_x_lower_z + E_x_upper_z) / 2.0

    # E_y: (i, j+1/2, k) -> (i, j, k+1/2): y backward, z forward
    E_y_lower_z = _avg(E_y[1:-1, 1:-1, 1:-1], E_y[1:-1, :-2, 1:-1])
    E_y_upper_z = _avg(E_y[1:-1, 1:-1, 2:], E_y[1:-1, :-2, 2:])
    E_y = (E_y_lower_z + E_y_upper_z) / 2.0

    # E_z: already at target
    E_z = E_z[1:-1, 1:-1, 1:-1]

    # H_x: (i, j+1/2, k+1/2) -> y backward
    H_x = _avg(H_x[1:-1, 1:-1, 1:-1], H_x[1:-1, :-2, 1:-1])
    # H_y: (i+1/2, j, k+1/2) -> x backward
    H_y = _avg(H_y[1:-1, 1:-1, 1:-1], H_y[:-2, 1:-1, 1:-1])

    # H_z: (i+1/2, j+1/2, k) -> x backward, y backward, z forward
    H_z_lower_z_x = _avg(H_z[1:-1, 1:-1, 1:-1], H_z[:-2, 1:-1, 1:-1])
    H_z_lower_z_xy = _avg(H_z_lower_z_x, _avg(H_z[1:-1, :-2, 1:-1], H_z[:-2, :-2, 1:-1]))
    H_z_upper_z_x = _avg(H_z[1:-1, 1:-1, 2:], H_z[:-2, 1:-1, 2:])
    H_z_upper_z_xy = _avg(H_z_upper_z_x, _avg(H_z[1:-1, :-2, 2:], H_z[:-2, :-2, 2:]))
    H_z = (H_z_lower_z_xy + H_z_upper_z_xy) / 2.0

    E_interp = mx.stack([E_x, E_y, E_z], axis=0)
    H_interp = mx.stack([H_x, H_y, H_z], axis=0)
    return E_interp, H_interp
