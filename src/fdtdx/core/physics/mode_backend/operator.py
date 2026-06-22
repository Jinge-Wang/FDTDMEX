"""Sparse finite-difference difference operators for the 2-D Yee mode cross-section.

These are the standard forward/backward Yee difference matrices on a (possibly non-uniform)
rectilinear transverse grid, with Dirichlet (PEC) or Neumann (PMC) walls at the *min* edge of
each transverse axis and PEC at the max edge. They are the building blocks of the full-vectorial
transverse-E mode operator assembled in :mod:`fdtdx.core.physics.mode_backend.solve`.

Formulation provenance (independent implementation, no third-party code copied):
- Zhu & Brown, "Full-vectorial finite-difference analysis of microstructured optical fibers,"
  Opt. Express 10(17):853-864 (2002).
- A. B. Fallahkhair, K. S. Li, T. E. Murphy, "Vector Finite Difference Modesolver for Anisotropic
  Dielectric Waveguides," J. Lightwave Technol. 26(11):1423-1431 (2008).

Index/flatten convention: a field sampled on the ``(Nx, Ny)`` transverse grid is flattened in
C-order (``index = ix * Ny + iy``), i.e. x is the outer index and y the inner one. The x-derivative
matrices are ``kron(Dx, I_Ny)`` and the y-derivative matrices ``kron(I_Nx, Dy)``. Reshape an
eigenvector back to a 2-D field with ``vec.reshape(Nx, Ny)`` (C-order).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def primal_dual_steps(coords_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the primal (forward) and dual (backward) grid steps for one axis.

    Args:
        coords_m: 1-D array of ``N + 1`` cell-edge coordinates in metres.

    Returns:
        A pair ``(dl_f, dl_b)`` of length-``N`` step arrays. ``dl_f`` are the primal cell widths
        (used by the forward derivative, which lives at the dual/H locations); ``dl_b`` are the dual
        steps (centre-to-centre distances, used by the backward derivative at the primal/E locations).
        Matches the staggering used by the transverse-E mode operator.
    """
    coords_m = np.asarray(coords_m, dtype=float)
    dl_f = coords_m[1:] - coords_m[:-1]
    # dual steps: midpoint distances, with the first entry duplicated so lengths match
    dl_mid = 0.5 * (dl_f[:-1] + dl_f[1:])
    dl_b = np.concatenate(([dl_f[0]], dl_mid))
    return dl_f, dl_b


def _diff_forward(dls: np.ndarray, n: int, pmc: bool) -> sp.csr_matrix:
    """1-D forward difference (-1 on the diagonal, +1 on the super-diagonal), scaled by ``1/dls``.

    PEC (``pmc=False``) zeroes the first diagonal entry so the field just outside the min edge is the
    negative image (Dirichlet); PMC keeps it (Neumann).
    """
    if n == 1:
        return sp.csr_matrix((1, 1))
    d = sp.diags([-1.0, 1.0], [0, 1], shape=(n, n), format="lil")
    if not pmc:
        d[0, 0] = 0.0
    d = sp.diags(1.0 / dls).dot(d.tocsr())
    return d.tocsr()


def _diff_backward(dls: np.ndarray, n: int, pmc: bool) -> sp.csr_matrix:
    """1-D backward difference (+1 on the diagonal, -1 on the sub-diagonal), scaled by ``1/dls``.

    PEC zeroes the first diagonal entry; PMC sets it to ``2`` (image charge of the same sign).
    """
    if n == 1:
        return sp.csr_matrix((1, 1))
    d = sp.diags([1.0, -1.0], [0, -1], shape=(n, n), format="lil")
    d[0, 0] = 2.0 if pmc else 0.0
    d = sp.diags(1.0 / dls).dot(d.tocsr())
    return d.tocsr()


def build_derivative_matrices(
    coords_x_m: np.ndarray,
    coords_y_m: np.ndarray,
    dmin_pmc: tuple[bool, bool] = (False, False),
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Assemble the four 2-D Yee difference matrices ``(dxf, dxb, dyf, dyb)`` in SI units.

    Args:
        coords_x_m: ``Nx + 1`` cell-edge coordinates (metres) along the first transverse axis.
        coords_y_m: ``Ny + 1`` cell-edge coordinates (metres) along the second transverse axis.
        dmin_pmc: per-axis PMC flag at the min edge; ``True`` imposes a magnetic (Neumann) wall,
            ``False`` an electric (PEC/Dirichlet) wall. Max edges are always PEC.

    Returns:
        ``(dxf, dxb, dyf, dyb)`` sparse ``(N, N)`` matrices with ``N = Nx * Ny`` in C-order flatten.
        Not normalised by ``k0`` (the solver divides by ``k0``).
    """
    nx = len(coords_x_m) - 1
    ny = len(coords_y_m) - 1
    dlf_x, dlb_x = primal_dual_steps(coords_x_m)
    dlf_y, dlb_y = primal_dual_steps(coords_y_m)

    eye_x = sp.eye(nx, format="csr")
    eye_y = sp.eye(ny, format="csr")

    dxf = sp.kron(_diff_forward(dlf_x, nx, dmin_pmc[0]), eye_y, format="csr")
    dxb = sp.kron(_diff_backward(dlb_x, nx, dmin_pmc[0]), eye_y, format="csr")
    dyf = sp.kron(eye_x, _diff_forward(dlf_y, ny, dmin_pmc[1]), format="csr")
    dyb = sp.kron(eye_x, _diff_backward(dlb_y, ny, dmin_pmc[1]), format="csr")
    return dxf, dxb, dyf, dyb
