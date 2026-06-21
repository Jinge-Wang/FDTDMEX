"""MLX Yee curl operators with CPML auxiliary-field recurrences.

Line-for-line translation of ``fdtdx.core.physics.curl.curl_H`` / ``curl_E`` using
``mx.roll`` for the staggered finite differences and the precomputed CPML ``a``/``b`` /
``1/kappa`` (see :mod:`fdtdx.mlx.pml`). Each finite difference along an axis is multiplied by
that axis's metric scale (port of ``_metric_scale``): the scalar ``1.0`` on uniform grids (so
the uniform path is unchanged), or ``reference_spacing / cell_width`` broadcasting along the
axis on non-uniform grids. ``curl_E`` uses the forward stencil (primal widths); ``curl_H`` the
backward stencil (dual widths).

Fields are passed pre-padded with one zero ghost cell per side (shape (3, Nx+2, Ny+2,
Nz+2)); the ``[1:-1, 1:-1, 1:-1]`` interior slice recovers (Nx, Ny, Nz), exactly as the
JAX reference.
"""

from __future__ import annotations

import mlx.core as mx


def _wrap_pad_axis(arr: mx.array, axis: int) -> mx.array:
    """Pad one ghost cell per side on ``axis`` with periodic (wrap) values."""
    lo = [slice(None)] * arr.ndim
    lo[axis] = slice(-1, None)
    hi = [slice(None)] * arr.ndim
    hi[axis] = slice(0, 1)
    return mx.concatenate([arr[tuple(lo)], arr, arr[tuple(hi)]], axis=axis)


def pad_fields_mlx(field: mx.array, periodic_axes: tuple = (False, False, False)) -> mx.array:
    """Pad a (3, Nx, Ny, Nz) field to (3, Nx+2, Ny+2, Nz+2).

    Mirrors ``fdtdx.core.misc.pad_fields``: wrap (periodic) padding on periodic axes,
    zero (constant) padding on the others (PML/PEC). Periodic boundaries in fdtdx are
    Bloch boundaries with zero phase, so wrap padding alone reproduces them.
    """
    out = field
    for i, periodic in enumerate(periodic_axes):
        axis = i + 1
        if periodic:
            out = _wrap_pad_axis(out, axis)
        else:
            pw = [(0, 0)] * out.ndim
            pw[axis] = (1, 1)
            out = mx.pad(out, pw)
    return out


def pad_zero(field: mx.array) -> mx.array:
    """Zero-pad a (3, Nx, Ny, Nz) field to (3, Nx+2, Ny+2, Nz+2) (no periodic axes)."""
    return pad_fields_mlx(field, (False, False, False))


def curl_H_mlx(
    H_pad: mx.array,
    psi_E: mx.array,
    cpml_a: mx.array,
    cpml_b: mx.array,
    inv_kappa: mx.array,
    simulate_boundaries: bool,
    metric: tuple = (1.0, 1.0, 1.0),
) -> tuple[mx.array, mx.array]:
    """Curl of H (-> E-type field) plus updated psi_E. See ``curl_H`` in fdtdx.

    ``metric`` is the per-axis backward-stencil (dual-width) derivative scale; each finite
    difference along an axis is multiplied by ``metric[axis]`` (``1.0`` on uniform grids).
    """
    a, b, ik = cpml_a, cpml_b, inv_kappa
    mx_, my_, mz_ = metric

    dyHz = (H_pad[2] - mx.roll(H_pad[2], 1, axis=1))[1:-1, 1:-1, 1:-1] * my_
    dzHy = (H_pad[1] - mx.roll(H_pad[1], 1, axis=2))[1:-1, 1:-1, 1:-1] * mz_
    dzHx = (H_pad[0] - mx.roll(H_pad[0], 1, axis=2))[1:-1, 1:-1, 1:-1] * mz_
    dxHz = (H_pad[2] - mx.roll(H_pad[2], 1, axis=0))[1:-1, 1:-1, 1:-1] * mx_
    dxHy = (H_pad[1] - mx.roll(H_pad[1], 1, axis=0))[1:-1, 1:-1, 1:-1] * mx_
    dyHx = (H_pad[0] - mx.roll(H_pad[0], 1, axis=1))[1:-1, 1:-1, 1:-1] * my_

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
    metric: tuple = (1.0, 1.0, 1.0),
) -> tuple[mx.array, mx.array]:
    """Curl of E (-> H-type field) plus updated psi_H. See ``curl_E`` in fdtdx.

    ``metric`` is the per-axis forward-stencil (primal-width) derivative scale; each finite
    difference along an axis is multiplied by ``metric[axis]`` (``1.0`` on uniform grids).
    """
    a, b, ik = cpml_a, cpml_b, inv_kappa
    mx_, my_, mz_ = metric

    dyEz = (mx.roll(E_pad[2], -1, axis=1) - E_pad[2])[1:-1, 1:-1, 1:-1] * my_
    dzEy = (mx.roll(E_pad[1], -1, axis=2) - E_pad[1])[1:-1, 1:-1, 1:-1] * mz_
    dzEx = (mx.roll(E_pad[0], -1, axis=2) - E_pad[0])[1:-1, 1:-1, 1:-1] * mz_
    dxEz = (mx.roll(E_pad[2], -1, axis=0) - E_pad[2])[1:-1, 1:-1, 1:-1] * mx_
    dxEy = (mx.roll(E_pad[1], -1, axis=0) - E_pad[1])[1:-1, 1:-1, 1:-1] * mx_
    dyEx = (mx.roll(E_pad[0], -1, axis=1) - E_pad[0])[1:-1, 1:-1, 1:-1] * my_

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
