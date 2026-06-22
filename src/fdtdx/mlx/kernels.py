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

**Non-uniform grids are handled in-kernel (M3).** Each difference is scaled by its per-axis
``reference_spacing/cell_width`` buffer (``m{k}``, 1-D) before the curl combine and the CPML
recurrence — mirroring ``curl._mul_metric``. Uniform axes carry a scalar ``1.0`` and emit no
multiply, so the uniform path is byte-for-byte unchanged.

**Heterogeneous full-tensor materials use a block hybrid (M3).** The kernel runs the diagonal bulk
(``cb`` = the tensor's diagonal); the off-diagonal inclusion's bounding box (``_offdiag_box``) gets
the validated MLX-op aniso update (``update._update_E``/``_update_H``) over a haloed interior slice,
spliced back with ``_set_box``. Box cells are bit-identical to the whole-domain ops path (same local
stencil, real neighbours via the halo); diagonal cells inside the box reduce to the same diagonal
update as the bulk. Gated to lossless, uniform-grid, compact, PML-disjoint interior inclusions.

Eligibility (else the loop uses the compiled MLX-op cores), via ``kernel_eligible``: no conductivity;
isotropic/diagonal ``inv_eps``/``inv_mu``, or a full-tensor whose off-diagonal inclusion is a compact
(< half-domain), PML-disjoint interior box on a uniform grid. Lossy media, non-uniform full-tensor,
and scattered/oversized inclusions keep the MLX-op path.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from fdtdx.mlx.curl import _slab_take
from fdtdx.mlx.update import _update_E, _update_H

#: Halo (cells) around an anisotropic inclusion's bbox handed to the MLX-op aniso correction — the
#: curl needs a 1-cell halo and the off-diagonal averaging another, so 2 strips the slice-edge ghost.
_BOX_MARGIN = 2
#: Max anisotropic-inclusion bbox fraction of the domain for the block hybrid to stay a win (else
#: the whole-domain MLX-op aniso path is no slower than kernel-bulk + a near-full-domain correction).
_BOX_MAX_FRAC = 0.5

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


def _is_full_tensor(arr) -> bool:
    return isinstance(arr, mx.array) and arr.ndim > 0 and arr.shape[0] == 9


def _is_uniform_metric(metric) -> bool:
    return all(isinstance(m, float) and m == 1.0 for m in metric)


def _diag_cb(inv_material, c: float):
    """``c·inv_material`` reduced to the per-component diagonal the bulk kernel reads. A 9-tensor
    keeps only its diagonal (entries 0/4/8 → a (3,N³) buffer); (1)/(3) materials pass through."""
    if _is_full_tensor(inv_material):
        return c * mx.stack([inv_material[0], inv_material[4], inv_material[8]], axis=0)
    return c * inv_material


def _offdiag_box(inv_material):
    """Bounding box ``((x0,x1),(y0,y1),(z0,z1))`` (half-open) of the cells whose 9-tensor has any
    non-zero *off-diagonal* entry — the only cells the bulk (diagonal) kernel gets wrong. ``None``
    if the tensor is diagonal everywhere (then the diagonal ``cb`` already covers it). Host-side
    (numpy), computed once at build."""
    arr = np.array(inv_material)  # (9, nx, ny, nz)
    off = np.zeros(arr.shape[1:], dtype=bool)
    for idx in (1, 2, 3, 5, 6, 7):  # off-diagonal entries of the row-major 3x3
        off |= arr[idx] != 0.0
    if not off.any():
        return None
    box = []
    for ax in range(3):
        other = tuple(a for a in range(3) if a != ax)
        nz = np.nonzero(off.any(axis=other))[0]
        box.append((int(nz[0]), int(nz[-1]) + 1))
    return tuple(box)


def _box_ok(box, shape, extents) -> bool:
    """Whether ``box`` is a compact interior inclusion the block hybrid can correct: it must clear
    every PML slab *and* leave a ``_BOX_MARGIN`` halo to the domain edge (so the haloed slice is all
    real interior cells, no CPML), and stay under ``_BOX_MAX_FRAC`` of the domain (else no win)."""
    if box is None:
        return True  # diagonal-only 9-tensor: no correction needed, kernel handles it
    vol, tot = 1, 1
    for ax in range(3):
        lo, hi = box[ax]
        n = shape[ax]
        plo, phi = extents[ax]
        if lo < max(plo, _BOX_MARGIN) or hi > n - max(phi, _BOX_MARGIN):
            return False
        vol *= hi - lo
        tot *= n
    return vol <= _BOX_MAX_FRAC * tot


def _set_box(full, box, core):
    """Functionally splice ``core`` (the corrected inclusion block) into ``full`` at ``box`` (out-of-
    place, race-free): replace the z-range inside the x/y sub-block, then the y-range, then the x-range."""
    (x0, x1), (y0, y1), (z0, z1) = box
    xy = full[:, x0:x1, y0:y1, :]
    xy = mx.concatenate([xy[:, :, :, :z0], core, xy[:, :, :, z1:]], axis=3)
    xblk = full[:, x0:x1, :, :]
    xblk = mx.concatenate([xblk[:, :, :y0, :], xy, xblk[:, :, y1:, :]], axis=2)
    return mx.concatenate([full[:, :x0, :, :], xblk, full[:, x1:, :, :]], axis=1)


def kernel_eligible(state) -> bool:
    """Whether the custom Metal kernels can run this case (else fall back to the MLX-op cores).

    Non-uniform metric is handled in-kernel (M3: each difference is scaled by its per-axis
    ``reference_spacing/cell_width`` buffer). Heterogeneous full-tensor materials are handled by the
    block hybrid (M3: kernel for the iso/diag bulk, MLX-op aniso over a compact interior inclusion
    bbox) — eligible only lossless, uniform-grid, with that bbox compact + PML-disjoint.
    """
    if state.sigma_E is not None or state.sigma_H is not None:
        return False
    # Drude-Lorentz dispersion needs no gate here: it is always iso/diagonal (fdtdx forbids it with
    # off-diagonal tensors), so a lossless dispersive run is already eligible and rides the E-kernel's
    # ADE fold; a lossy+dispersive run is excluded above by the ``sigma`` check (→ MLX-op cores).
    fe, fm = _is_full_tensor(state.inv_eps), _is_full_tensor(state.inv_mu)
    if fe or fm:
        if not (_is_uniform_metric(state.metric_fwd) and _is_uniform_metric(state.metric_bwd)):
            return False
        shape = tuple(int(s) for s in state.E.shape[1:])
        if fe and not _box_ok(_offdiag_box(state.inv_eps), shape, state.cpml_extents):
            return False
        if fm and not _box_ok(_offdiag_box(state.inv_mu), shape, state.cpml_extents):
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


def _metric_side(metric) -> tuple[list[int], list, list[str]]:
    """Split a per-axis metric tuple (``metric_fwd``/``metric_bwd``) into the non-uniform axes, their
    1-D scale buffers (flattened to length ``N`` along that axis), and the kernel input names ``m{k}``.
    Uniform axes (scalar ``1.0``) are dropped — they need no buffer and emit no multiply."""
    axes, bufs, names = [], [], []
    for k in range(3):
        m = metric[k]
        if isinstance(m, mx.array):
            axes.append(k)
            bufs.append(m.reshape(-1))
            names.append(f"m{k}")
    return axes, bufs, names


def _metric_lines(metric_axes) -> str:
    """Scale each difference ``d{i}`` by its per-axis metric buffer ``m{k}`` (``=reference_spacing/
    cell_width``, 1-D, indexed by the cell's coordinate along axis ``k``). Mirrors ``curl._mul_metric``
    applied to each ``_bwd_diff``/``_fwd_diff`` before the curl combine — so both the plain curl and
    the CPML recurrence see metric-scaled differences. ``metric_axes`` is the set of non-uniform axes;
    uniform axes carry a scalar ``1.0`` and emit nothing (the byte-for-byte uniform path)."""
    lines = []
    for k in sorted(metric_axes):
        coord = _AXVAR[k]
        for ci, _tgt, _sgn in _AXIS_COMPS[k]:
            lines.append(f"        d{ci} = d{ci} * m{k}[{coord}];")
    return ("\n" + "\n".join(lines)) if lines else ""


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


def _ade_lines(num_poles: int, inv_c: float) -> str:
    """MSL tail for the **E-kernel only** when Drude-Lorentz dispersion is active (build-time
    ``num_poles > 0``). Replaces the plain ``out = E + cb·curl`` writes: form ``E_upd`` (the
    post-curl/CPML field), run the per-pole ADE recurrence (unrolled), and write the back-reaction
    plus the swapped P buffers.

    The recurrence uses the *pre-update* field ``E[..]`` (=E^n, the kernel never writes ``E``),
    matching ``fdtdx.fdtd.update`` (and ``mlx.update._update_E``). Per pole/component:
    ``pn = c1*pc + c2*pp + c3*E^n``; ``E += inv_eps*sum(pc - pn)``; ``Pcurr<-pn``, ``Pprev<-pc``.
    ``inv_eps_comp = cb_comp / c = cb_comp · inv_c`` (``c`` is a build-time scalar). Buffer layout
    (row-contiguous): ``P[p,comp,idx] = p·3·N3 + comp·N3 + idx``; ``c{1,2,3}[p,0,idx] = p·N3 + idx``.
    """
    comps = (("x", "idx"), ("y", "N3+idx"), ("z", "2u*N3+idx"))
    lit = f"{float(inv_c)}f"
    out = [
        "\n        float eux = E[idx]       + cbx*cx;",
        "        float euy = E[N3+idx]    + cby*cy;",
        "        float euz = E[2u*N3+idx] + cbz*cz;",
        "        float dltx = 0.0f; float dlty = 0.0f; float dltz = 0.0f;",
    ]
    # c1/c2/c3 are packed into one (3, poles, 1, N, N, N) buffer ``dc``: slab ``which`` starts at
    # ``which*poles*N3``; within it, pole ``p`` is at ``p*N3 + idx`` (the singleton component axis = 1).
    npn3 = f"{num_poles}u*N3"
    for p in range(num_poles):
        coff = f"{p}u*N3+idx"
        out.append(f"        {{ float c1=dc[{coff}]; float c2=dc[{npn3}+{coff}]; float c3=dc[2u*{npn3}+{coff}];")
        for comp, eoff in comps:
            poff = f"{p}u*3u*N3+{eoff}"
            out.append(
                f"          {{ float pc=Pc[{poff}]; float pp=Pp[{poff}]; float pn=c1*pc+c2*pp+c3*E[{eoff}];"
                f" dlt{comp}+=pc-pn; Pco[{poff}]=pn; Ppo[{poff}]=pc; }}"
            )
        out.append("        }")
    out += [
        f"        out[idx]       = eux + (cbx*{lit})*dltx;",
        f"        out[N3+idx]    = euy + (cby*{lit})*dlty;",
        f"        out[2u*N3+idx] = euz + (cbz*{lit})*dltz;",
    ]
    return "\n".join(out)


def _field_source(
    shape,
    periodic,
    diagonal,
    scalar,
    extents,
    sb: bool,
    metric_axes,
    forward: bool,
    dispersive: bool = False,
    num_poles: int = 0,
    inv_c: float = 1.0,
) -> str:
    """Generate the MSL body for the E-kernel (``forward=False``) or H-kernel (``forward=True``).

    ``out = F (± cb·curl)`` with ``+`` for E and ``-`` for H; ``F`` is the field being updated and
    ``G`` (``H`` for E, ``E`` for H) the one being differentiated. Differences on the non-uniform
    ``metric_axes`` are scaled by their per-axis ``m{k}`` buffer; with ``sb`` the CPML correction
    blocks (``_corr_blocks``) are interleaved before the ``cb`` multiply. With ``dispersive`` (E-kernel
    only) the plain output writes are replaced by the ADE tail (``_ade_lines``). When ``dispersive``
    is false the emitted source is byte-for-byte the pre-dispersion kernel.
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
    if dispersive:
        tail = _ade_lines(num_poles, inv_c)
    else:
        tail = f"""
        out[idx]       = {Fname}[idx]       {out_sign} cbx*cx;
        out[N3+idx]    = {Fname}[N3+idx]    {out_sign} cby*cy;
        out[2u*N3+idx] = {Fname}[2u*N3+idx] {out_sign} cbz*cz;
    """
    return (
        _common(nx, ny, nz)
        + _cb_lines(diagonal, scalar)
        + body
        + _metric_lines(metric_axes)
        + """
        float cx = d0 - d1;
        float cy = d2 - d3;
        float cz = d4 - d5;"""
        + corr
        + tail
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

    inv_eps, inv_mu = state.inv_eps, state.inv_mu
    cb_E = _diag_cb(inv_eps, c)  # (1|3, N, N, N) — diagonal of a 9-tensor
    e_diag = cb_E.shape[0] == 3
    mu_scalar = float(c * inv_mu) if not isinstance(inv_mu, mx.array) else None
    cb_H = None if mu_scalar is not None else _diag_cb(inv_mu, c)
    h_diag = cb_H is not None and cb_H.shape[0] == 3

    # Heterogeneous full-tensor inclusions (block hybrid): the kernel does the diagonal bulk; the
    # off-diagonal inclusion bbox gets the validated MLX-op aniso update over a haloed slice, spliced
    # back. box=None → diagonal everywhere (no correction). Lossless + uniform + interior (gated by
    # kernel_eligible), so the slice carries no CPML and no metric.
    box_E = _offdiag_box(inv_eps) if _is_full_tensor(inv_eps) else None
    box_H = _offdiag_box(inv_mu) if _is_full_tensor(inv_mu) else None

    # Drude-Lorentz (ADE): folded into the E-kernel (E is iso/diagonal here — fdtdx forbids dispersion
    # with off-diagonal tensors, so box_E is always None for a dispersive run). Coefficients captured
    # as constants; the per-pole recurrence is unrolled in the MSL. ``inv_c`` recovers inv_eps = cb/c.
    dispersive = state.dispersive_c1 is not None
    num_poles = int(state.dispersive_c1.shape[0]) if dispersive else 0
    # Pack c1/c2/c3 into a single (3, poles, 1, N, N, N) buffer so the dispersive E-kernel stays under
    # Metal's 31-buffer limit (full 3-axis CPML + ψ already uses many bindings). Pole count is unrolled
    # in the MSL, so it never adds buffers — the worst case (3-axis CPML) is a fixed 30 bindings.
    disp_c = mx.stack([state.dispersive_c1, state.dispersive_c2, state.dispersive_c3], axis=0) if dispersive else None
    inv_c = 1.0 / c

    do_cpml = sb and len(_active_axes(ext)) > 0
    axes = _active_axes(ext)
    comps = _slab_comps(ext)

    # Non-uniform metric: per-axis 1-D scale buffers (=reference_spacing/cell_width), captured as
    # constants. E-update differentiates with the dual-width (backward) metric, H with the forward.
    # Uniform axes carry the scalar 1.0 and contribute nothing (the byte-for-byte uniform path).
    metric_axes_E, metric_E, metric_names_E = _metric_side(state.metric_bwd)
    metric_axes_H, metric_H, metric_names_H = _metric_side(state.metric_fwd)

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

    to_eval = [cb_E] + ([] if cb_H is None else [cb_H]) + metric_E + metric_H + (coeff_E + coeff_H if do_cpml else [])
    if dispersive:
        to_eval += [disp_c]
    mx.eval(*to_eval)

    grid, tg = (nz, ny, nx), (min(32, nz), min(4, ny), min(4, nx))

    cpml_in = (coeff_names + psi_names) if do_cpml else []
    # ADE buffers come last on the E-kernel (after cb/metric/cpml). dc1/2/3 are the per-cell pole
    # coefficients; Pc/Pp the in polarization, Pco/Ppo the swapped out polarization.
    disp_in = ["dc", "Pc", "Pp"] if dispersive else []
    disp_out = ["Pco", "Ppo"] if dispersive else []
    e_inputs = ["E", "H", "cb", *metric_names_E, *cpml_in, *disp_in]
    h_base = ["E", "H"] if mu_scalar is not None else ["E", "H", "cb"]
    h_inputs = [*h_base, *metric_names_H, *cpml_in]
    e_outputs = ["out"] + (pso_names if do_cpml else []) + disp_out
    h_outputs = ["out"] + (pso_names if do_cpml else [])

    kE = mx.fast.metal_kernel(
        name="fdtdmex_E",
        input_names=e_inputs,
        output_names=e_outputs,
        source=_field_source(
            shape,
            per,
            e_diag,
            None,
            ext,
            do_cpml,
            metric_axes_E,
            forward=False,
            dispersive=dispersive,
            num_poles=num_poles,
            inv_c=inv_c,
        ),
        ensure_row_contiguous=True,
    )
    kH = mx.fast.metal_kernel(
        name="fdtdmex_H",
        input_names=h_inputs,
        output_names=h_outputs,
        source=_field_source(shape, per, h_diag, mu_scalar, ext, do_cpml, metric_axes_H, forward=True),
        ensure_row_contiguous=True,
    )

    def _run(kern, base_inputs, metric_bufs, psi, coeff, F_template):
        inputs = list(base_inputs) + list(metric_bufs)
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

    m = _BOX_MARGIN
    uni_metric, no_pml, dummy_psi = (1.0, 1.0, 1.0), ((0, 0), (0, 0), (0, 0)), (None,) * 6

    def _box_correct(F_kernel, E, H, box, inv_material, update_fn):
        """Replace the inclusion bbox of the diagonal-kernel result with the full aniso update, run
        on a haloed interior slice (no CPML/metric — gated interior + uniform). Diagonal cells in the
        box get the equivalent diagonal result, so splicing the whole box is consistent with the bulk."""
        (x0, x1), (y0, y1), (z0, z1) = box
        sx, sy, sz = slice(x0 - m, x1 + m), slice(y0 - m, y1 + m), slice(z0 - m, z1 + m)
        E_s, H_s, im_s = E[:, sx, sy, sz], H[:, sx, sy, sz], inv_material[:, sx, sy, sz]
        F_box, _ = update_fn(
            E_s, H_s, dummy_psi, im_s, None, a, b, ik, uni_metric, (False, False, False), no_pml, None, c, False
        )
        return _set_box(F_kernel, box, F_box[:, m:-m, m:-m, m:-m])

    def e_core(E, H, psi_E):
        E_new, psi_new = _run(kE, [E, H, cb_E], metric_E, psi_E, coeff_E, E)
        if box_E is not None:
            E_new = _box_correct(E_new, E, H, box_E, inv_eps, _update_E)
        return E_new, psi_new

    def e_core_dispersive(E, H, psi_E, P_curr, P_prev):
        """E-core with the ADE polarization threaded as extra in/out buffers (box_E is None for a
        dispersive run, so no block-hybrid correction). Output order matches ``e_outputs``:
        ``out``, ψ slabs (if CPML), then ``Pco``/``Ppo``."""
        inputs = [E, H, cb_E, *metric_E]
        out_shapes, out_dtypes = [E.shape], [E.dtype]
        psi_new = list(psi_E)
        if do_cpml:
            inputs += coeff_E + [psi_E[i] for i in comps]
            for i in comps:
                out_shapes.append(psi_E[i].shape)
                out_dtypes.append(psi_E[i].dtype)
        inputs += [disp_c, P_curr, P_prev]
        out_shapes += [P_curr.shape, P_prev.shape]
        out_dtypes += [P_curr.dtype, P_prev.dtype]
        outs = kE(inputs=inputs, output_shapes=out_shapes, output_dtypes=out_dtypes, grid=grid, threadgroup=tg)
        if do_cpml:
            for n, i in enumerate(comps):
                psi_new[i] = outs[1 + n]
        return outs[0], tuple(psi_new), outs[-2], outs[-1]

    def h_core(E, H, psi_H):
        base = [E, H] if mu_scalar is not None else [E, H, cb_H]
        H_new, psi_new = _run(kH, base, metric_H, psi_H, coeff_H, H)
        if box_H is not None:
            H_new = _box_correct(H_new, E, H, box_H, inv_mu, _update_H)
        return H_new, psi_new

    e_core_final = e_core_dispersive if dispersive else e_core
    if compile_step and sb:
        return mx.compile(e_core_final), mx.compile(h_core)
    return e_core_final, h_core
