"""MLX Yee curl operators with CPML auxiliary-field recurrences.

Line-for-line translation of ``fdtdx.core.physics.curl.curl_H`` / ``curl_E`` using
``mx.roll`` for the staggered finite differences and the precomputed CPML ``a``/``b`` /
``1/kappa`` (see :mod:`fdtdx.mlx.pml`). Uniform-grid metric only (the JAX ``_metric_scale``
factor is 1.0 on uniform grids); spacing-weighted metrics for non-uniform grids land in M4.

Fields are passed pre-padded with one zero ghost cell per side (shape (3, Nx+2, Ny+2,
Nz+2)); the ``[1:-1, 1:-1, 1:-1]`` interior slice recovers (Nx, Ny, Nz), exactly as the
JAX reference.
"""

from __future__ import annotations

import mlx.core as mx


def pad_zero(field: mx.array) -> mx.array:
    """Zero-pad a (3, Nx, Ny, Nz) field to (3, Nx+2, Ny+2, Nz+2).

    Matches ``fdtdx.core.misc.pad_fields`` for non-periodic (PML/PEC) axes.
    """
    return mx.pad(field, [(0, 0), (1, 1), (1, 1), (1, 1)])


def curl_H_mlx(
    H_pad: mx.array,
    psi_E: mx.array,
    cpml_a: mx.array,
    cpml_b: mx.array,
    inv_kappa: mx.array,
    simulate_boundaries: bool,
) -> tuple[mx.array, mx.array]:
    """Curl of H (-> E-type field) plus updated psi_E. See ``curl_H`` in fdtdx."""
    a, b, ik = cpml_a, cpml_b, inv_kappa

    dyHz = (H_pad[2] - mx.roll(H_pad[2], 1, axis=1))[1:-1, 1:-1, 1:-1]
    dzHy = (H_pad[1] - mx.roll(H_pad[1], 1, axis=2))[1:-1, 1:-1, 1:-1]
    dzHx = (H_pad[0] - mx.roll(H_pad[0], 1, axis=2))[1:-1, 1:-1, 1:-1]
    dxHz = (H_pad[2] - mx.roll(H_pad[2], 1, axis=0))[1:-1, 1:-1, 1:-1]
    dxHy = (H_pad[1] - mx.roll(H_pad[1], 1, axis=0))[1:-1, 1:-1, 1:-1]
    dyHx = (H_pad[0] - mx.roll(H_pad[0], 1, axis=1))[1:-1, 1:-1, 1:-1]

    psi_Exy, psi_Exz, psi_Eyz, psi_Eyx, psi_Ezx, psi_Ezy = (
        psi_E[0],
        psi_E[1],
        psi_E[2],
        psi_E[3],
        psi_E[4],
        psi_E[5],
    )

    if simulate_boundaries:
        psi_Exy = b[1] * psi_Exy + a[1] * dyHz
        psi_Exz = b[2] * psi_Exz + a[2] * dzHy
        psi_Eyz = b[2] * psi_Eyz + a[2] * dzHx
        psi_Eyx = b[0] * psi_Eyx + a[0] * dxHz
        psi_Ezx = b[0] * psi_Ezx + a[0] * dxHy
        psi_Ezy = b[1] * psi_Ezy + a[1] * dyHx

    psi_E_updated = mx.stack([psi_Exy, psi_Exz, psi_Eyz, psi_Eyx, psi_Ezx, psi_Ezy], axis=0)

    curl_x = (ik[1] * dyHz + psi_Exy) - (ik[2] * dzHy + psi_Exz)
    curl_y = (ik[2] * dzHx + psi_Eyz) - (ik[0] * dxHz + psi_Eyx)
    curl_z = (ik[0] * dxHy + psi_Ezx) - (ik[1] * dyHx + psi_Ezy)
    curl = mx.stack([curl_x, curl_y, curl_z], axis=0)

    return curl, psi_E_updated


def curl_E_mlx(
    E_pad: mx.array,
    psi_H: mx.array,
    cpml_a: mx.array,
    cpml_b: mx.array,
    inv_kappa: mx.array,
    simulate_boundaries: bool,
) -> tuple[mx.array, mx.array]:
    """Curl of E (-> H-type field) plus updated psi_H. See ``curl_E`` in fdtdx."""
    a, b, ik = cpml_a, cpml_b, inv_kappa

    dyEz = (mx.roll(E_pad[2], -1, axis=1) - E_pad[2])[1:-1, 1:-1, 1:-1]
    dzEy = (mx.roll(E_pad[1], -1, axis=2) - E_pad[1])[1:-1, 1:-1, 1:-1]
    dzEx = (mx.roll(E_pad[0], -1, axis=2) - E_pad[0])[1:-1, 1:-1, 1:-1]
    dxEz = (mx.roll(E_pad[2], -1, axis=0) - E_pad[2])[1:-1, 1:-1, 1:-1]
    dxEy = (mx.roll(E_pad[1], -1, axis=0) - E_pad[1])[1:-1, 1:-1, 1:-1]
    dyEx = (mx.roll(E_pad[0], -1, axis=1) - E_pad[0])[1:-1, 1:-1, 1:-1]

    psi_Hxy, psi_Hxz, psi_Hyz, psi_Hyx, psi_Hzx, psi_Hzy = (
        psi_H[0],
        psi_H[1],
        psi_H[2],
        psi_H[3],
        psi_H[4],
        psi_H[5],
    )

    if simulate_boundaries:
        psi_Hxy = b[4] * psi_Hxy + a[4] * dyEz
        psi_Hxz = b[5] * psi_Hxz + a[5] * dzEy
        psi_Hyz = b[5] * psi_Hyz + a[5] * dzEx
        psi_Hyx = b[3] * psi_Hyx + a[3] * dxEz
        psi_Hzx = b[3] * psi_Hzx + a[3] * dxEy
        psi_Hzy = b[4] * psi_Hzy + a[4] * dyEx

    psi_H_updated = mx.stack([psi_Hxy, psi_Hxz, psi_Hyz, psi_Hyx, psi_Hzx, psi_Hzy], axis=0)

    curl_x = (ik[1] * dyEz + psi_Hxy) - (ik[2] * dzEy + psi_Hxz)
    curl_y = (ik[2] * dzEx + psi_Hyz) - (ik[0] * dxEz + psi_Hyx)
    curl_z = (ik[0] * dxEy + psi_Hzx) - (ik[1] * dyEx + psi_Hzy)
    curl = mx.stack([curl_x, curl_y, curl_z], axis=0)

    return curl, psi_H_updated
