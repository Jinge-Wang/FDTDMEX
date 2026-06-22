"""Custom Metal E/H update kernels for the isotropic/diagonal forward path (Phase 2 M2).

The bulk update (``E + c·inv_eps·curl_H(H)`` / ``H - c·inv_mu·curl_E(E)``) runs as one
``mx.fast.metal_kernel`` per field — thread-per-cell, the curl read from global with neighbour
reuse via cache (the M1 design, which already reaches the ~3 RT bandwidth floor). This generalises
the scalar-``Cb`` M1 microbench ([`benchmarks/m1_kernel.py`](../../../benchmarks/m1_kernel.py)) to a
per-cell material read: ``cb = c·inv_eps`` (isotropic 1-component, or diagonal 3-component) passed
as a buffer, with the low/high-edge ghost wrapping on periodic axes (matching ``curl._bwd_diff`` /
``_fwd_diff``).

**CPML is a spatial hybrid (not folded into the kernel).** The kernel computes the full-domain
*plain* curl update; on PML boundary slabs that value differs from the truth by
``c·inv_eps·corr`` where ``corr = (1/κ-1)·d_slab + ψ_new`` — exactly the per-component term
``curl._cpml_curl`` accumulates. So we recompute the slab differences with ``curl._slab_diff``,
advance ψ on the slabs, and add ``cb·corr`` back into the kernel output via ``curl._slab_add``
(disjoint, additive → race-free, out-of-place). Interior cells (the bulk) get the kernel value
untouched; only the thin slabs touch MLX ops.

Eligibility (else the loop uses the compiled MLX-op cores): isotropic/diagonal ``inv_eps``/
``inv_mu`` (not 9-tensor), no conductivity, and a uniform metric. Lossy / full-tensor /
non-uniform-grid cases keep the MLX-op path (M3+).
"""

from __future__ import annotations

import mlx.core as mx

from fdtdx.mlx.curl import _AX, _slab_add, _slab_diff, _slab_take

#: Bumped each time kernel cores are built; tests assert the kernel path actually engaged.
KERNEL_CORES_BUILT = 0

#: H/E component differentiated for each of the 6 ψ-components (mirrors ``curl.curl_H_mlx`` order).
_COMP = (2, 1, 0, 2, 1, 0)

#: Per spatial axis (x, y, z): thread coordinate var, the per-axis extent name, and the contiguous
#: stride (in cells). Domains are not cubic, so each axis carries its own extent. The grid is
#: (NZ, NY, NX) with thread_position_in_grid .x→k(z), .y→j(y), .z→i(x); z is contiguous (stride 1).
_AXVAR = ("i", "j", "k")
_NNAME = ("NX", "NY", "NZ")
_STRIDE = ("SX", "SY", "1u")  # x plane (NY*NZ), y row (NZ), z element (1)


def _is_uniform_metric(metric) -> bool:
    return all(isinstance(m, float) and m == 1.0 for m in metric)


def _is_full_tensor(arr) -> bool:
    return isinstance(arr, mx.array) and arr.ndim > 0 and arr.shape[0] == 9


def kernel_eligible(state) -> bool:
    """Whether the custom Metal kernels can run this case (else fall back to the MLX-op cores)."""
    if _is_full_tensor(state.inv_eps) or _is_full_tensor(state.inv_mu):
        return False
    if state.sigma_E is not None or state.sigma_H is not None:
        return False
    if not (_is_uniform_metric(state.metric_fwd) and _is_uniform_metric(state.metric_bwd)):
        return False
    return True


def _ghost(base: str, axis: int, wrap: bool, forward: bool) -> str:
    """The neighbour read for the backward (``forward=False``) or forward edge along ``axis``.

    Backward: ``coord>0 ? F[base+idx-stride] : (wrap ? F[base+idx+(N-1)*stride] : 0)``.
    Forward:  ``coord+1<N ? F[base+idx+stride] : (wrap ? F[base+idx-(N-1)*stride] : 0)``.
    ``base`` is "" / "N3+" / "2u*N3+" (the field component offset); ``F`` is filled by the caller.
    """
    coord, nn, stride = _AXVAR[axis], _NNAME[axis], _STRIDE[axis]
    if forward:
        guard, near, far = f"{coord}+1u<{nn}", f"{base}idx+{stride}", f"{base}idx-({nn}-1u)*{stride}"
    else:
        guard, near, far = f"{coord}>0u", f"{base}idx-{stride}", f"{base}idx+({nn}-1u)*{stride}"
    if wrap:
        return f"({guard}) ? F[{near}] : F[{far}]"
    return f"({guard}) ? F[{near}] : 0.0f"


def _common(nx: int, ny: int, nz: int) -> str:
    return f"""
        uint k = thread_position_in_grid.x;
        uint j = thread_position_in_grid.y;
        uint i = thread_position_in_grid.z;
        const uint NX = {nx}u; const uint NY = {ny}u; const uint NZ = {nz}u;
        if (i >= NX || j >= NY || k >= NZ) return;
        const uint SX = NY*NZ; const uint SY = NZ;
        const uint N3 = NX*NY*NZ;
        uint idx = i*SX + j*SY + k;
    """


def _cb_lines(diagonal: bool, scalar) -> str:
    """The three per-component coefficients ``cbx/cby/cbz`` (= c·inv_eps or c·inv_mu)."""
    if scalar is not None:
        lit = f"{float(scalar)}f"
        return f"        const float cbx = {lit}; const float cby = {lit}; const float cbz = {lit};\n"
    if diagonal:
        return "        float cbx = cb[idx]; float cby = cb[N3+idx]; float cbz = cb[2u*N3+idx];\n"
    return "        float cbx = cb[idx]; float cby = cbx; float cbz = cbx;\n"


def _e_source(shape: tuple, periodic: tuple, diagonal: bool) -> str:
    # backward differences of H; (var, component base, axis)
    nbrs = [
        ("Hx_jm", "", 1),
        ("Hx_km", "", 2),
        ("Hy_im", "N3+", 0),
        ("Hy_km", "N3+", 2),
        ("Hz_im", "2u*N3+", 0),
        ("Hz_jm", "2u*N3+", 1),
    ]
    lines = [
        "        float Hx = H[idx];        float Hy = H[N3+idx];        float Hz = H[2u*N3+idx];",
    ]
    for var, base, axis in nbrs:
        expr = _ghost(base, axis, periodic[axis], forward=False).replace("F[", "H[")
        lines.append(f"        float {var} = {expr};")
    body = "\n".join(lines)
    return (
        _common(*shape)
        + _cb_lines(diagonal, None)
        + body
        + """
        float cx = (Hz - Hz_jm) - (Hy - Hy_km);
        float cy = (Hx - Hx_km) - (Hz - Hz_im);
        float cz = (Hy - Hy_im) - (Hx - Hx_jm);
        out[idx]       = E[idx]       + cbx*cx;
        out[N3+idx]    = E[N3+idx]    + cby*cy;
        out[2u*N3+idx] = E[2u*N3+idx] + cbz*cz;
    """
    )


def _h_source(shape: tuple, periodic: tuple, diagonal: bool, scalar) -> str:
    # forward differences of E; (var, component base, axis)
    nbrs = [
        ("Ez_jp", "2u*N3+", 1),
        ("Ey_kp", "N3+", 2),
        ("Ex_kp", "", 2),
        ("Ez_ip", "2u*N3+", 0),
        ("Ey_ip", "N3+", 0),
        ("Ex_jp", "", 1),
    ]
    lines = [
        "        float Ex = E[idx];        float Ey = E[N3+idx];        float Ez = E[2u*N3+idx];",
    ]
    for var, base, axis in nbrs:
        expr = _ghost(base, axis, periodic[axis], forward=True).replace("F[", "E[")
        lines.append(f"        float {var} = {expr};")
    body = "\n".join(lines)
    return (
        _common(*shape)
        + _cb_lines(diagonal, scalar)
        + body
        + """
        float cx = (Ez_jp - Ez) - (Ey_kp - Ey);
        float cy = (Ex_kp - Ex) - (Ez_ip - Ez);
        float cz = (Ey_ip - Ey) - (Ex_jp - Ex);
        out[idx]       = H[idx]       - cbx*cx;
        out[N3+idx]    = H[N3+idx]    - cby*cy;
        out[2u*N3+idx] = H[2u*N3+idx] - cbz*cz;
    """
    )


def _slab_correction(field_list, src, psi, a, b, ik, ab_off, extents, periodic, metric, cb, sign):
    """Patch ``field_list`` (the 3 component arrays of the kernel's plain update) on the PML slabs
    and advance ψ. ``src`` is the field being differentiated (H for E-update, E for H-update);
    ``ab_off`` 0 (E) or 3 (H); ``sign`` +1 (E adds the curl) or -1 (H subtracts). Mirrors
    ``curl._cpml_curl`` then folds the per-cell material so the slab value matches the MLX-op path.
    """
    forward = ab_off == 3
    psi_new = list(psi)
    for i in range(6):
        k = _AX[i]
        lo, hi = extents[k]
        if lo + hi == 0:
            continue
        d_slab = _slab_diff(src[_COMP[i]], k, lo, hi, periodic[k], metric[k], forward)
        a_s = _slab_take(a[ab_off + k], k, lo, hi)
        b_s = _slab_take(b[ab_off + k], k, lo, hi)
        ikm1_s = _slab_take(ik[k], k, lo, hi) - 1.0
        pnew = b_s * psi[i] + a_s * d_slab
        psi_new[i] = pnew
        corr = ikm1_s * d_slab + pnew
        if i % 2 == 1:
            corr = -corr
        comp = i // 2
        cb_comp = cb if isinstance(cb, float) else cb[0 if cb.shape[0] == 1 else comp]
        cb_s = cb_comp if isinstance(cb_comp, float) else _slab_take(cb_comp, k, lo, hi)
        field_list[comp] = _slab_add(field_list[comp], k, lo, hi, sign * cb_s * corr)
    return field_list, tuple(psi_new)


def build_kernel_cores(state, c: float, sb: bool, compile_step: bool = True):
    """Build ``(e_core, h_core)`` Metal-kernel closures with the standard core signature
    ``core(F, G, psi) -> (F_new, psi_new)`` (drop-in for the compiled MLX-op cores in ``loop``).

    The bulk kernel is one fused node, but the slab-CPML correction is a chain of small ops; run
    eager they dispatch one-by-one and dominate the CPML-on step. ``mx.compile`` (the metal kernel
    composes as a normal graph node) fuses the whole core, recovering the kernel speed with CPML on.
    """
    global KERNEL_CORES_BUILT
    KERNEL_CORES_BUILT += 1

    shape = tuple(int(s) for s in state.E.shape[1:])  # (Nx, Ny, Nz)
    per = tuple(state.periodic_axes)
    a, b, ik = state.cpml_a, state.cpml_b, state.inv_kappa
    ext, mbwd, mfwd = state.cpml_extents, state.metric_bwd, state.metric_fwd

    cb_E = c * state.inv_eps  # (1|3, N, N, N)
    e_diag = cb_E.shape[0] == 3
    inv_mu = state.inv_mu
    mu_scalar = float(c * inv_mu) if not isinstance(inv_mu, mx.array) else None
    cb_H = None if mu_scalar is not None else c * inv_mu
    h_diag = cb_H is not None and cb_H.shape[0] == 3
    mx.eval(cb_E) if mu_scalar is not None else mx.eval(cb_E, cb_H)

    nx, ny, nz = shape
    grid, tg = (nz, ny, nx), (min(32, nz), min(4, ny), min(4, nx))
    kE = mx.fast.metal_kernel(
        name="fdtdmex_E",
        input_names=["E", "H", "cb"],
        output_names=["out"],
        source=_e_source(shape, per, e_diag),
        ensure_row_contiguous=True,
    )
    h_inputs = ["E", "H"] if mu_scalar is not None else ["E", "H", "cb"]
    kH = mx.fast.metal_kernel(
        name="fdtdmex_H",
        input_names=h_inputs,
        output_names=["out"],
        source=_h_source(shape, per, h_diag, mu_scalar),
        ensure_row_contiguous=True,
    )

    def e_core(E, H, psi_E):
        (E_full,) = kE(inputs=[E, H, cb_E], output_shapes=[E.shape], output_dtypes=[E.dtype], grid=grid, threadgroup=tg)
        if not sb:
            return E_full, psi_E
        comps, psi_new = _slab_correction(
            [E_full[0], E_full[1], E_full[2]], H, psi_E, a, b, ik, 0, ext, per, mbwd, cb_E, 1.0
        )
        return mx.stack(comps, axis=0), psi_new

    def h_core(E, H, psi_H):
        inputs = [E, H] if mu_scalar is not None else [E, H, cb_H]
        (H_full,) = kH(inputs=inputs, output_shapes=[H.shape], output_dtypes=[H.dtype], grid=grid, threadgroup=tg)
        if not sb:
            return H_full, psi_H
        cb = mu_scalar if mu_scalar is not None else cb_H
        comps, psi_new = _slab_correction(
            [H_full[0], H_full[1], H_full[2]], E, psi_H, a, b, ik, 3, ext, per, mfwd, cb, -1.0
        )
        return mx.stack(comps, axis=0), psi_new

    if compile_step and sb:
        return mx.compile(e_core), mx.compile(h_core)
    return e_core, h_core
