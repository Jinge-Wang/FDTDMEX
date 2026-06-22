"""Custom Metal E/H update kernels for the isotropic/diagonal forward path (Phase 2 M2 + M3).

The bulk update (``E + c·inv_eps·curl_H(H)`` / ``H - c·inv_mu·curl_E(E)``) runs as one
``mx.fast.metal_kernel`` per field — thread-per-cell, the curl read from global with neighbour
reuse via cache (the M1 design, which already reaches the ~3 RT bandwidth floor). This generalises
the scalar-``Cb`` M1 microbench ([`benchmarks/m1_kernel.py`](../../../benchmarks/m1_kernel.py)) to a
per-cell material read: ``cb = c·inv_eps`` (isotropic 1-component, or diagonal 3-component) passed
as a buffer, with the low/high-edge ghost wrapping on periodic axes (matching ``curl._bwd_diff`` /
``_fwd_diff``).

**CPML is folded into the kernel (M3).** Each thread computes the six metric-scaled differences
``d[i]`` of the plain curl; a thread that lies in a PML boundary slab additionally advances that
slab cell's ψ recurrence and adds the κ-stretch + ψ correction directly into the curl before the
``cb`` multiply, so the kernel writes the *final* E/H — no post-kernel full-array rebuild. ψ is
carried (and returned) as the compact per-component boundary-slab buffers already in ``MLXState``;
the CPML coefficients ``a``/``b``/``1/κ`` are sliced to those same slabs once at build time and
captured. The two ψ-components that share a PML axis share its slab geometry and its ``a``/``b``/
``1/κ`` (per-cell, depth-only along the boundary normal), so each axis contributes one coefficient
triple. (M2 computed the slab correction with MLX ops via ``concatenate`` — ~22 RT on top of the
5 RT bulk; folding it in drops CPML-on toward the bulk floor.)

Eligibility (else the loop uses the compiled MLX-op cores): isotropic/diagonal ``inv_eps``/
``inv_mu`` (not 9-tensor), no conductivity, and a uniform metric. Lossy / full-tensor /
non-uniform-grid cases keep the MLX-op path (M3+).
"""

from __future__ import annotations

import mlx.core as mx

from fdtdx.mlx.curl import _slab_take

#: Bumped each time kernel cores are built; tests assert the kernel path actually engaged.
KERNEL_CORES_BUILT = 0

#: Per spatial axis (x, y, z): thread coordinate var (matching ``_common`` below). The grid is
#: (NZ, NY, NX) with thread_position_in_grid .x→k(z), .y→j(y), .z→i(x); z is contiguous (stride 1).
_AXVAR = ("i", "j", "k")
_NNAME = ("NX", "NY", "NZ")
_STRIDE = ("SX", "SY", "1u")  # x plane (NY*NZ), y row (NZ), z element (1)

#: For each PML axis, the (ψ-component index, curl-component target, sign) pairs whose differences
#: are perpendicular to that axis. Mirrors ``curl._cpml_curl``: component ``i`` differentiates along
#: ``_AX[i]``, contributes to curl component ``i//2`` with sign ``+`` (even ``i``) / ``-`` (odd ``i``).
_AXIS_COMPS = {
    0: ((3, "cy", "-"), (4, "cz", "+")),  # x slab: d3 (dxHz/dxEz), d4 (dxHy/dxEy)
    1: ((0, "cx", "+"), (5, "cz", "-")),  # y slab: d0 (dyHz/dyEz), d5 (dyHx/dyEx)
    2: ((1, "cx", "-"), (2, "cy", "+")),  # z slab: d1 (dzHy/dzEy), d2 (dzHx/dzEx)
}


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


def _active_axes(extents) -> list[int]:
    """PML axes (in 0,1,2 order) that carry a boundary slab."""
    return [k for k in range(3) if extents[k][0] + extents[k][1] > 0]


def _slab_comps(extents) -> list[int]:
    """ψ-component indices (0..5) whose axis has a slab, in (axis, component) order — the order the
    ψ buffers are passed in / returned out of the kernel."""
    comps: list[int] = []
    for k in _active_axes(extents):
        comps.extend(ci for ci, _tgt, _sgn in _AXIS_COMPS[k])
    return comps


def _corr_blocks(extents, nx: int, ny: int, nz: int) -> str:
    """MSL that, for each PML-slab thread, advances ψ and folds the κ-stretch + ψ correction into
    ``cx``/``cy``/``cz`` (identical for E and H; the side enters only via the captured a/b arrays
    and the final ``out`` sign). Reads ``d0..d5`` and the slab buffers ``as{k}``/``bs{k}``/``is{k}``/
    ``psi{i}``; writes the advanced slab ψ to ``pso{i}``. Slab strides are baked from the extents."""
    blocks = []
    for k in _active_axes(extents):
        lo, hi = extents[k]
        coord = _AXVAR[k]
        n = (nx, ny, nz)[k]
        nmhi = n - hi
        d = lo + hi
        # slab membership + slab-local coordinate along axis k (compact buffer ordering = [0:lo] ++ [n-hi:n])
        if lo and hi:
            cond = f"({coord} < {lo}u) || ({coord} >= {nmhi}u)"
            sloc = f"({coord} < {lo}u) ? {coord} : ({lo}u + ({coord} - {nmhi}u))"
        elif lo:
            cond = f"{coord} < {lo}u"
            sloc = coord
        else:
            cond = f"{coord} >= {nmhi}u"
            sloc = f"{coord} - {nmhi}u"
        # linear index into the (..., d on axis k, ...) compact slab buffer
        if k == 0:
            sidx = "sl*SX + j*SY + k"
        elif k == 1:
            sidx = f"i*{d * nz}u + sl*SY + k"
        else:
            sidx = f"i*{ny * d}u + j*{d}u + sl"
        terms = []
        for ci, tgt, sgn in _AXIS_COMPS[k]:
            terms.append(
                f"            {{ float p = bb*psi{ci}[si] + aa*d{ci}; pso{ci}[si] = p; {tgt} {sgn}= (ik1*d{ci} + p); }}"
            )
        blocks.append(
            f"""
        if ({cond}) {{
            uint sl = {sloc};
            uint si = {sidx};
            float aa = as{k}[si]; float bb = bs{k}[si]; float ik1 = is{k}[si] - 1.0f;
{chr(10).join(terms)}
        }}"""
        )
    return "".join(blocks)


def _field_source(shape, periodic, diagonal, scalar, extents, sb: bool, forward: bool) -> str:
    """Generate the MSL body for the E-kernel (``forward=False``) or H-kernel (``forward=True``).

    ``out = F (± cb·curl)`` with ``+`` for E and ``-`` for H; ``F`` is the field being updated and
    ``G`` (``H`` for E, ``E`` for H) the one being differentiated. With ``sb`` the CPML correction
    blocks (``_corr_blocks``) are interleaved before the ``cb`` multiply.
    """
    nx, ny, nz = shape
    if forward:  # H-kernel: forward differences of E
        Gname, Fname, out_sign = "E", "H", "-"
        nbrs = [
            ("Ez_jp", "2u*N3+", 1),
            ("Ey_kp", "N3+", 2),
            ("Ex_kp", "", 2),
            ("Ez_ip", "2u*N3+", 0),
            ("Ey_ip", "N3+", 0),
            ("Ex_jp", "", 1),
        ]
        comp0 = "        float Ex = E[idx];        float Ey = E[N3+idx];        float Ez = E[2u*N3+idx];"
        d_lines = [
            "        float d0 = Ez_jp - Ez; float d1 = Ey_kp - Ey;",
            "        float d2 = Ex_kp - Ex; float d3 = Ez_ip - Ez;",
            "        float d4 = Ey_ip - Ey; float d5 = Ex_jp - Ex;",
        ]
    else:  # E-kernel: backward differences of H
        Gname, Fname, out_sign = "H", "E", "+"
        nbrs = [
            ("Hx_jm", "", 1),
            ("Hx_km", "", 2),
            ("Hy_im", "N3+", 0),
            ("Hy_km", "N3+", 2),
            ("Hz_im", "2u*N3+", 0),
            ("Hz_jm", "2u*N3+", 1),
        ]
        comp0 = "        float Hx = H[idx];        float Hy = H[N3+idx];        float Hz = H[2u*N3+idx];"
        d_lines = [
            "        float d0 = Hz - Hz_jm; float d1 = Hy - Hy_km;",
            "        float d2 = Hx - Hx_km; float d3 = Hz - Hz_im;",
            "        float d4 = Hy - Hy_im; float d5 = Hx - Hx_jm;",
        ]

    lines = [comp0]
    for var, base, axis in nbrs:
        expr = _ghost(base, axis, periodic[axis], forward=forward).replace("F[", f"{Gname}[")
        lines.append(f"        float {var} = {expr};")
    body = "\n".join(lines + d_lines)

    corr = _corr_blocks(extents, nx, ny, nz) if sb else ""
    return (
        _common(nx, ny, nz)
        + _cb_lines(diagonal, scalar)
        + body
        + """
        float cx = d0 - d1;
        float cy = d2 - d3;
        float cz = d4 - d5;"""
        + corr
        + f"""
        out[idx]       = {Fname}[idx]       {out_sign} cbx*cx;
        out[N3+idx]    = {Fname}[N3+idx]    {out_sign} cby*cy;
        out[2u*N3+idx] = {Fname}[2u*N3+idx] {out_sign} cbz*cz;
    """
    )


def build_kernel_cores(state, c: float, sb: bool, compile_step: bool = True):
    """Build ``(e_core, h_core)`` Metal-kernel closures with the standard core signature
    ``core(F, G, psi) -> (F_new, psi_new)`` (drop-in for the compiled MLX-op cores in ``loop``).

    CPML is folded into the kernel (M3): with ``sb`` the kernel takes the per-axis slab coefficient
    buffers + the compact ψ slabs as extra inputs and returns the advanced ψ slabs as extra outputs,
    so the whole CPML-on step is one Metal node per field (no slab MLX-op rebuild).
    """
    global KERNEL_CORES_BUILT
    KERNEL_CORES_BUILT += 1

    shape = tuple(int(s) for s in state.E.shape[1:])  # (Nx, Ny, Nz)
    nx, ny, nz = shape
    per = tuple(state.periodic_axes)
    a, b, ik = state.cpml_a, state.cpml_b, state.inv_kappa
    ext = tuple(state.cpml_extents)

    cb_E = c * state.inv_eps  # (1|3, N, N, N)
    e_diag = cb_E.shape[0] == 3
    inv_mu = state.inv_mu
    mu_scalar = float(c * inv_mu) if not isinstance(inv_mu, mx.array) else None
    cb_H = None if mu_scalar is not None else c * inv_mu
    h_diag = cb_H is not None and cb_H.shape[0] == 3

    do_cpml = sb and len(_active_axes(ext)) > 0
    axes = _active_axes(ext)
    comps = _slab_comps(ext)

    # Captured slab coefficient buffers, one triple per active axis. E uses a/b[k]; H uses a/b[3+k];
    # both use 1/κ[k] (mirrors curl._cpml_curl). ψ flows through as core args (the MLXState slabs).
    coeff_E, coeff_H, coeff_names = [], [], []
    for k in axes:
        lo, hi = ext[k]
        coeff_E += [_slab_take(a[k], k, lo, hi), _slab_take(b[k], k, lo, hi), _slab_take(ik[k], k, lo, hi)]
        coeff_H += [_slab_take(a[3 + k], k, lo, hi), _slab_take(b[3 + k], k, lo, hi), _slab_take(ik[k], k, lo, hi)]
        coeff_names += [f"as{k}", f"bs{k}", f"is{k}"]
    psi_names = [f"psi{i}" for i in comps]
    pso_names = [f"pso{i}" for i in comps]

    to_eval = [cb_E] + ([] if cb_H is None else [cb_H]) + (coeff_E + coeff_H if do_cpml else [])
    mx.eval(*to_eval)

    grid, tg = (nz, ny, nx), (min(32, nz), min(4, ny), min(4, nx))

    e_inputs = ["E", "H", "cb"] + (coeff_names + psi_names if do_cpml else [])
    h_base = ["E", "H"] if mu_scalar is not None else ["E", "H", "cb"]
    h_inputs = h_base + (coeff_names + psi_names if do_cpml else [])
    e_outputs = ["out"] + (pso_names if do_cpml else [])
    h_outputs = ["out"] + (pso_names if do_cpml else [])

    kE = mx.fast.metal_kernel(
        name="fdtdmex_E",
        input_names=e_inputs,
        output_names=e_outputs,
        source=_field_source(shape, per, e_diag, None, ext, do_cpml, forward=False),
        ensure_row_contiguous=True,
    )
    kH = mx.fast.metal_kernel(
        name="fdtdmex_H",
        input_names=h_inputs,
        output_names=h_outputs,
        source=_field_source(shape, per, h_diag, mu_scalar, ext, do_cpml, forward=True),
        ensure_row_contiguous=True,
    )

    def _run(kern, base_inputs, psi, coeff, F_template):
        inputs = list(base_inputs)
        out_shapes = [F_template.shape]
        out_dtypes = [F_template.dtype]
        psi_new = list(psi)
        if do_cpml:
            inputs += coeff + [psi[i] for i in comps]
            for i in comps:
                out_shapes.append(psi[i].shape)
                out_dtypes.append(psi[i].dtype)
        outs = kern(inputs=inputs, output_shapes=out_shapes, output_dtypes=out_dtypes, grid=grid, threadgroup=tg)
        if do_cpml:
            for n, i in enumerate(comps):
                psi_new[i] = outs[1 + n]
        return outs[0], tuple(psi_new)

    def e_core(E, H, psi_E):
        return _run(kE, [E, H, cb_E], psi_E, coeff_E, E)

    def h_core(E, H, psi_H):
        base = [E, H] if mu_scalar is not None else [E, H, cb_H]
        return _run(kH, base, psi_H, coeff_H, H)

    if compile_step and sb:
        return mx.compile(e_core), mx.compile(h_core)
    return e_core, h_core
