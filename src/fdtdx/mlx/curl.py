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


# Per-ψ-component derivative axis. The 6 ψ components (and the 6 differences below) are ordered
# so that component i is the difference along axis _AX[i]; its CPML coefficients live only in the
# two PML slabs perpendicular to that axis. curl_H (E-side) uses a/b index _AX[i]; curl_E (H-side)
# uses _AX[i]+3. The inv_kappa combine uses index _AX[i] for BOTH (mirrors fdtdx exactly).
_AX = (1, 2, 2, 0, 0, 1)


def _slab_take(f: mx.array, axis: int, lo: int, hi: int) -> mx.array:
    """Gather the ``[0:lo]`` and ``[N-hi:N]`` boundary slab of ``f`` along ``axis`` (size ``lo+hi``)."""
    if lo and hi:
        return mx.concatenate([_sl(f, 0, lo, axis), _sl(f, f.shape[axis] - hi, None, axis)], axis=axis)
    if lo:
        return _sl(f, 0, lo, axis)
    if hi:
        return _sl(f, f.shape[axis] - hi, None, axis)
    return _sl(f, 0, 0, axis)  # empty (no PML on this axis)


def _slab_add(comp: mx.array, axis: int, lo: int, hi: int, corr: mx.array) -> mx.array:
    """Add the slab-shaped ``corr`` (size ``lo+hi`` along ``axis``) back into ``comp``'s slab cells."""
    n = comp.shape[axis]
    segs = []
    if lo:
        segs.append(_sl(comp, 0, lo, axis) + _sl(corr, 0, lo, axis))
    segs.append(_sl(comp, lo, n - hi, axis))  # untouched interior
    if hi:
        segs.append(_sl(comp, n - hi, None, axis) + _sl(corr, lo, lo + hi, axis))
    return mx.concatenate(segs, axis=axis) if len(segs) > 1 else segs[0]


def slab_to_full(slab: mx.array, axis: int, lo: int, hi: int, n: int) -> mx.array:
    """Inverse of ``_slab_take``: scatter a slab array back into a full ``n``-long axis (zeros
    in the interior). Used once per run to hand ψ back to the host container."""
    full_shape = list(slab.shape)
    full_shape[axis] = n
    if lo + hi == 0:
        return mx.zeros(full_shape, dtype=slab.dtype)
    segs = []
    if lo:
        segs.append(_sl(slab, 0, lo, axis))
    mid_shape = list(slab.shape)
    mid_shape[axis] = n - lo - hi
    segs.append(mx.zeros(mid_shape, dtype=slab.dtype))
    if hi:
        segs.append(_sl(slab, lo, lo + hi, axis))
    return mx.concatenate(segs, axis=axis)


def _cpml_curl(d, psi, a, b, ik, ab_off, extents, simulate_boundaries):
    """Assemble the curl from the 6 metric-scaled differences ``d`` as a cheap full-domain plain
    part plus a slab-localised CPML correction; advance the slab ψ. Exact algebraic split of the
    full-domain combine ``(ik*d_a + psi_a) - (ik*d_b + psi_b)`` (ik=1, psi=0 outside the slabs).

    ``psi`` is a 6-tuple of per-component slab arrays; ``a``/``b``/``ik`` are full ``(6,N³)``;
    ``ab_off`` is 0 (E-side) or 3 (H-side); ``extents`` is ``((lo,hi),…)`` per axis.
    """
    curl = [d[0] - d[1], d[2] - d[3], d[4] - d[5]]
    if not simulate_boundaries:
        return mx.stack(curl, axis=0), psi

    psi_new = list(psi)
    for i in range(6):
        k = _AX[i]
        lo, hi = extents[k]
        if lo + hi == 0:
            continue  # no PML on this axis -> ik=1, ψ=0 there: correction is exactly zero
        d_slab = _slab_take(d[i], k, lo, hi)
        a_s = _slab_take(a[ab_off + k], k, lo, hi)
        b_s = _slab_take(b[ab_off + k], k, lo, hi)
        ikm1_s = _slab_take(ik[k], k, lo, hi) - 1.0
        pnew = b_s * psi[i] + a_s * d_slab
        psi_new[i] = pnew
        corr = ikm1_s * d_slab + pnew
        if i % 2 == 1:
            corr = -corr
        curl[i // 2] = _slab_add(curl[i // 2], k, lo, hi, corr)
    return mx.stack(curl, axis=0), tuple(psi_new)


def curl_H_mlx(
    H: mx.array,
    psi_E,
    cpml_a: mx.array,
    cpml_b: mx.array,
    inv_kappa: mx.array,
    simulate_boundaries: bool,
    metric: tuple = (1.0, 1.0, 1.0),
    periodic_axes: tuple = (False, False, False),
    extents: tuple = ((0, 0), (0, 0), (0, 0)),
):
    """Curl of H (-> E-type field) + advanced slab ψ_E. See ``curl_H`` in fdtdx.

    Pad-free backward differences (``_bwd_diff``); CPML applied as a slab-localised correction
    (``_cpml_curl``). ``psi_E`` is a 6-tuple of per-component boundary-slab arrays.
    """
    mx_, my_, mz_ = metric
    px, py, pz = periodic_axes
    d = [
        _mul_metric(_bwd_diff(H[2], 1, py), my_),  # dyHz
        _mul_metric(_bwd_diff(H[1], 2, pz), mz_),  # dzHy
        _mul_metric(_bwd_diff(H[0], 2, pz), mz_),  # dzHx
        _mul_metric(_bwd_diff(H[2], 0, px), mx_),  # dxHz
        _mul_metric(_bwd_diff(H[1], 0, px), mx_),  # dxHy
        _mul_metric(_bwd_diff(H[0], 1, py), my_),  # dyHx
    ]
    return _cpml_curl(d, psi_E, cpml_a, cpml_b, inv_kappa, 0, extents, simulate_boundaries)


def curl_E_mlx(
    E: mx.array,
    psi_H,
    cpml_a: mx.array,
    cpml_b: mx.array,
    inv_kappa: mx.array,
    simulate_boundaries: bool,
    metric: tuple = (1.0, 1.0, 1.0),
    periodic_axes: tuple = (False, False, False),
    extents: tuple = ((0, 0), (0, 0), (0, 0)),
):
    """Curl of E (-> H-type field) + advanced slab ψ_H. See ``curl_E`` in fdtdx.

    Pad-free forward differences (``_fwd_diff``); CPML applied as a slab-localised correction
    (``_cpml_curl``, H-side coefficient offset 3). ``psi_H`` is a 6-tuple of slab arrays.
    """
    mx_, my_, mz_ = metric
    px, py, pz = periodic_axes
    d = [
        _mul_metric(_fwd_diff(E[2], 1, py), my_),  # dyEz
        _mul_metric(_fwd_diff(E[1], 2, pz), mz_),  # dzEy
        _mul_metric(_fwd_diff(E[0], 2, pz), mz_),  # dzEx
        _mul_metric(_fwd_diff(E[2], 0, px), mx_),  # dxEz
        _mul_metric(_fwd_diff(E[1], 0, px), mx_),  # dxEy
        _mul_metric(_fwd_diff(E[0], 1, py), my_),  # dyEx
    ]
    return _cpml_curl(d, psi_H, cpml_a, cpml_b, inv_kappa, 3, extents, simulate_boundaries)
