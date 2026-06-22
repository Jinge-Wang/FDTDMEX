"""MLX Yee curl operators with CPML auxiliary-field recurrences.

Line-for-line translation of ``fdtdx.core.physics.curl.curl_H`` / ``curl_E``. Finite differences
are computed **pad-free**, by slicing: a backward/forward difference along an axis is assembled from
``f[1:] - f[:-1]`` plus a single edge cell whose ghost value is **zero on PML/PEC axes** or the
**wrapped neighbour on periodic axes** — reproducing byte-for-byte what one-cell zero/wrap padding
produced, without the full-array ``mx.pad`` copy per field per step. Each difference is then scaled
by that axis's metric (port of ``_metric_scale``): the scalar ``1.0`` on uniform grids (skipped as a
no-op), or ``reference_spacing / cell_width`` broadcasting along the axis on non-uniform grids.
``curl_E`` uses the forward stencil (primal widths); ``curl_H`` the backward stencil (dual widths).

``pad_fields_mlx`` is retained for the anisotropic averaging and detector interpolation paths.
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


def _sl(arr: mx.array, start, stop, axis: int) -> mx.array:
    """``arr`` sliced ``[start:stop]`` along ``axis`` (other axes untouched)."""
    idx = [slice(None)] * arr.ndim
    idx[axis] = slice(start, stop)
    return arr[tuple(idx)]


def _bwd_diff(f: mx.array, axis: int, periodic: bool) -> mx.array:
    """Backward difference ``d[i] = f[i] - f[i-1]`` along ``axis`` (same shape as ``f``).

    Low-edge ghost: ``f[-1]`` (wrap) on periodic axes, else ``0`` — matching one-cell padding.
    """
    body = _sl(f, 1, None, axis) - _sl(f, 0, -1, axis)  # f[i] - f[i-1] for i = 1..N-1
    lo = _sl(f, 0, 1, axis)  # zero ghost: f[0] - 0
    if periodic:
        lo = lo - _sl(f, -1, None, axis)  # wrap: f[0] - f[N-1]
    return mx.concatenate([lo, body], axis=axis)


def _fwd_diff(f: mx.array, axis: int, periodic: bool) -> mx.array:
    """Forward difference ``d[i] = f[i+1] - f[i]`` along ``axis`` (same shape as ``f``).

    High-edge ghost: ``f[0]`` (wrap) on periodic axes, else ``0`` — matching one-cell padding.
    """
    body = _sl(f, 1, None, axis) - _sl(f, 0, -1, axis)  # f[i+1] - f[i] for i = 0..N-2
    hi = -_sl(f, -1, None, axis)  # zero ghost: 0 - f[N-1]
    if periodic:
        hi = _sl(f, 0, 1, axis) - _sl(f, -1, None, axis)  # wrap: f[0] - f[N-1]
    return mx.concatenate([body, hi], axis=axis)


def _mul_metric(d: mx.array, m) -> mx.array:
    """Scale a difference by its per-axis metric; skip the no-op multiply on uniform grids."""
    if isinstance(m, float) and m == 1.0:
        return d
    return d * m


def curl_H_mlx(
    H: mx.array,
    psi_E: mx.array,
    cpml_a: mx.array,
    cpml_b: mx.array,
    inv_kappa: mx.array,
    simulate_boundaries: bool,
    metric: tuple = (1.0, 1.0, 1.0),
    periodic_axes: tuple = (False, False, False),
) -> tuple[mx.array, mx.array]:
    """Curl of H (-> E-type field) plus updated psi_E. See ``curl_H`` in fdtdx.

    ``H`` is the un-padded (3, Nx, Ny, Nz) field; backward differences are taken pad-free
    (``_bwd_diff``). ``metric`` is the per-axis dual-width derivative scale (``1.0`` on uniform).
    """
    a, b, ik = cpml_a, cpml_b, inv_kappa
    mx_, my_, mz_ = metric
    px, py, pz = periodic_axes

    dyHz = _mul_metric(_bwd_diff(H[2], 1, py), my_)
    dzHy = _mul_metric(_bwd_diff(H[1], 2, pz), mz_)
    dzHx = _mul_metric(_bwd_diff(H[0], 2, pz), mz_)
    dxHz = _mul_metric(_bwd_diff(H[2], 0, px), mx_)
    dxHy = _mul_metric(_bwd_diff(H[1], 0, px), mx_)
    dyHx = _mul_metric(_bwd_diff(H[0], 1, py), my_)

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
    else:
        psi_E_updated = psi_E  # untouched -> don't rebuild the (6,N^3) stack

    curl_x = (ik[1] * dyHz + psi_Exy) - (ik[2] * dzHy + psi_Exz)
    curl_y = (ik[2] * dzHx + psi_Eyz) - (ik[0] * dxHz + psi_Eyx)
    curl_z = (ik[0] * dxHy + psi_Ezx) - (ik[1] * dyHx + psi_Ezy)
    curl = mx.stack([curl_x, curl_y, curl_z], axis=0)

    return curl, psi_E_updated


def curl_E_mlx(
    E: mx.array,
    psi_H: mx.array,
    cpml_a: mx.array,
    cpml_b: mx.array,
    inv_kappa: mx.array,
    simulate_boundaries: bool,
    metric: tuple = (1.0, 1.0, 1.0),
    periodic_axes: tuple = (False, False, False),
) -> tuple[mx.array, mx.array]:
    """Curl of E (-> H-type field) plus updated psi_H. See ``curl_E`` in fdtdx.

    ``E`` is the un-padded (3, Nx, Ny, Nz) field; forward differences are taken pad-free
    (``_fwd_diff``). ``metric`` is the per-axis primal-width derivative scale (``1.0`` on uniform).
    """
    a, b, ik = cpml_a, cpml_b, inv_kappa
    mx_, my_, mz_ = metric
    px, py, pz = periodic_axes

    dyEz = _mul_metric(_fwd_diff(E[2], 1, py), my_)
    dzEy = _mul_metric(_fwd_diff(E[1], 2, pz), mz_)
    dzEx = _mul_metric(_fwd_diff(E[0], 2, pz), mz_)
    dxEz = _mul_metric(_fwd_diff(E[2], 0, px), mx_)
    dxEy = _mul_metric(_fwd_diff(E[1], 0, px), mx_)
    dyEx = _mul_metric(_fwd_diff(E[0], 1, py), my_)

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
    else:
        psi_H_updated = psi_H  # untouched -> don't rebuild the (6,N^3) stack

    curl_x = (ik[1] * dyEz + psi_Hxy) - (ik[2] * dzEy + psi_Hxz)
    curl_y = (ik[2] * dzEx + psi_Hyz) - (ik[0] * dxEz + psi_Hyx)
    curl_z = (ik[0] * dxEy + psi_Hzx) - (ik[1] * dyEx + psi_Hzy)
    curl = mx.stack([curl_x, curl_y, curl_z], axis=0)

    return curl, psi_H_updated
