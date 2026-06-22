"""Full-vectorial transverse-E eigenmode solver for the 2-D Yee cross-section (diagonal media).

Assembles the standard transverse-E mode operator from the diagonal permittivity / permeability
components and the difference matrices, solves the sparse generalized eigenproblem with a
shift-invert Arnoldi iteration (``scipy.sparse.linalg.eigs``) near the target ``n_eff``, and
recovers all six field components.

The eigenvector is the transverse electric field ``[Ex; Ey]`` and the eigenvalue is
``-(n_eff + i k_eff)**2`` (derivatives are normalised by ``k0``). ``Ez`` and the full ``H`` are
reconstructed from Maxwell's equations. ``H`` is returned scaled by ``-1j / eta0`` so that the
caller's ``* eta0`` step (see :mod:`fdtdx.core.physics.modes`) yields the field convention the
``ModePlaneSource`` / ``ModeOverlapDetector`` front-end expects.

Formulation provenance: Zhu & Brown 2002; Fallahkhair, Li & Murphy 2008 (see ``operator.py``).
Independent implementation — no third-party solver code is copied.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from fdtdx.constants import eta0


def _spdiag(vec: np.ndarray, n: int) -> sp.csr_matrix:
    return sp.spdiags(vec, [0], n, n).tocsr()


def solve_modes_diagonal(
    eps_xx: np.ndarray,
    eps_yy: np.ndarray,
    eps_zz: np.ndarray,
    mu_xx: np.ndarray,
    mu_yy: np.ndarray,
    mu_zz: np.ndarray,
    der_mats: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    k0: float,
    num_modes: int,
    neff_guess: float,
    direction: str = "+",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the diagonal-media full-vectorial mode problem.

    Args:
        eps_xx, eps_yy, eps_zz: flattened (length ``N``, C-order) diagonal relative permittivity
            components sampled at the Yee Ex/Ey/Ez locations.
        mu_xx, mu_yy, mu_zz: flattened diagonal relative permeability components.
        der_mats: ``(dxf, dxb, dyf, dyb)`` SI difference matrices from
            :func:`fdtdx.core.physics.mode_backend.operator.build_derivative_matrices`.
        k0: free-space wavenumber ``2 pi f / c`` (1/m).
        num_modes: number of modes to return (sorted by descending ``Re(n_eff)``).
        neff_guess: shift-invert target effective index.
        direction: ``"+"`` or ``"-"`` propagation direction.

    Returns:
        ``(E, H, neff, keff)`` where ``E`` and ``H`` have shape ``(3, N, num_modes)`` and ``neff``,
        ``keff`` have shape ``(num_modes,)``.
    """
    n = eps_xx.size
    # Normalise derivatives by k0 (dimensionless operator); eigenvalue is then -(neff)^2.
    dxf, dxb, dyf, dyb = (m / k0 for m in der_mats)

    inv_eps_zz = _spdiag(1.0 / eps_zz, n)
    inv_mu_zz = _spdiag(1.0 / mu_zz, n)

    # Transverse-E operator blocks (standard Yee mode formulation).
    p_mu = sp.bmat([[None, _spdiag(mu_yy, n)], [_spdiag(-mu_xx, n), None]], format="csr")
    p_partial = sp.bmat(
        [
            [-dxf.dot(inv_eps_zz).dot(dyb), dxf.dot(inv_eps_zz).dot(dxb)],
            [-dyf.dot(inv_eps_zz).dot(dyb), dyf.dot(inv_eps_zz).dot(dxb)],
        ],
        format="csr",
    )
    q_ep = sp.bmat([[None, _spdiag(eps_yy, n)], [_spdiag(-eps_xx, n), None]], format="csr")
    q_partial = sp.bmat(
        [
            [-dxb.dot(inv_mu_zz).dot(dyf), dxb.dot(inv_mu_zz).dot(dxf)],
            [-dyb.dot(inv_mu_zz).dot(dyf), dyb.dot(inv_mu_zz).dot(dxf)],
        ],
        format="csr",
    )
    qmat = (q_ep + q_partial).tocsr()
    # PQ factorisation: p_partial @ q_partial = 0, so mat = p_mu @ qmat + p_partial @ q_ep.
    mat = (p_mu.dot(qmat) + p_partial.dot(q_ep)).tocsr()

    # Deterministic starting vector with the min-edge rows zeroed (consistent with PEC).
    rng = np.random.default_rng(0)
    nx_ny = n
    vec_init = rng.random(2 * nx_ny) + 1j * rng.random(2 * nx_ny)

    num_modes = min(num_modes, mat.shape[0] - 2)
    eig_guess = -(neff_guess**2)
    vals, vecs = spl.eigs(
        mat.astype(np.complex128),
        k=num_modes,
        sigma=eig_guess,
        v0=vec_init,
    )

    # eigenvalue = -(neff + i keff)^2  ->  neff + i keff = sqrt(-eigenvalue)
    n_complex = np.emath.sqrt(-vals + 0j)
    neff = np.real(n_complex)
    keff = np.imag(n_complex)

    order = np.argsort(neff)[::-1]
    neff = neff[order]
    keff = keff[order]
    vecs = vecs[:, order]

    ex = vecs[:n, :]
    ey = vecs[n:, :]

    denom = (1j * neff - keff)[None, :]
    h_field = qmat.dot(vecs)
    hx = h_field[:n, :] / denom
    hy = h_field[n:, :] / denom
    hz = inv_mu_zz.dot(dxf.dot(ey) - dyf.dot(ex))

    # Ez = -inv_eps_zz * div^H (q_ep Exy) / (i neff); q_partial drops out of the divergence.
    h_partial = q_ep.dot(vecs) / denom
    ez = inv_eps_zz.dot(dxb.dot(h_partial[n:, :]) - dyb.dot(h_partial[:n, :]))

    E = np.stack((ex, ey, ez), axis=0)
    H = np.stack((hx, hy, hz), axis=0)

    # Return to the standard H-field normalisation expected downstream.
    H = H * (-1j / eta0)

    if direction == "-":
        H[0] *= -1
        H[1] *= -1
        E[2] *= -1

    return E, H, neff, keff
