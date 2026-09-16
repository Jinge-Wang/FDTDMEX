"""Full-vectorial transverse-E eigenmode solver for the 2-D Yee cross-section.

Assembles the standard transverse-E mode operator from the permittivity / permeability components
and the difference matrices, solves the sparse generalized eigenproblem with a shift-invert Arnoldi
iteration (``scipy.sparse.linalg.eigs``) near the target ``n_eff``, and recovers all six field
components.

Anisotropy tier. The permittivity may carry the transverse off-diagonal entries ``eps_xy`` and
``eps_yx`` as well as the three diagonal ones; see :func:`assemble_mode_operator` for where they
enter (one block, ``q_ep``) and :mod:`fdtdx.core.physics.mode_backend` for why the two entries that
couple a transverse axis to the propagation axis (``eps_xz``, ``eps_zx``, ``eps_yz``, ``eps_zy``)
cannot enter an eigenproblem that is linear in ``n_eff**2``.

The eigenvector is the transverse electric field ``[Ex; Ey]`` and the eigenvalue is
``-(n_eff + i k_eff)**2`` (derivatives are normalised by ``k0``). ``Ez`` and the full ``H`` are
reconstructed from Maxwell's equations. ``H`` is returned scaled by ``-1j / eta0`` so that the
caller's ``* eta0`` step (see :mod:`fdtdx.core.physics.modes`) yields the field convention the
``ModePlaneSource`` / ``ModeOverlapDetector`` front-end expects.

Precision: the operator is assembled and solved in ``complex128`` unconditionally, whatever float
dtype the surrounding simulation runs at. The eigenvalue is ``-(n_eff)^2``, so a single-precision
material perturbs ``n_eff`` at the 1e-7 level before the eigensolver starts; a gradient check
against a finite difference cannot survive that. Only the *returned* field dtype follows the
caller's ``dtype`` argument (see :func:`fdtdx.core.physics.modes.compute_mode`).

Formulation provenance: Zhu & Brown 2002; Fallahkhair, Li & Murphy 2008 (see ``operator.py``).
Independent implementation — no third-party solver code is copied.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from fdtdx.constants import eta0


def _spdiag(vec: np.ndarray, n: int) -> sp.csr_matrix:
    return sp.spdiags(np.asarray(vec, dtype=np.complex128), [0], n, n).tocsr()


class ModeOperator(NamedTuple):
    """The assembled transverse-E mode operator and the blocks a caller may need with it.

    Exposed so that code outside the eigen-solve can reuse the same discrete operator: a shifted
    driven solve ``(mat + n_target^2 I) v = f`` at a prescribed effective index (the "frequency
    domain" solve of the topology-optimization literature), an exact discrete eigenvalue
    sensitivity, or a residual check on a mode obtained elsewhere.

    Attributes:
        mat: The ``(2N, 2N)`` operator with ``mat v = -(n_eff + i k_eff)^2 v`` for the stacked
            transverse electric field ``v = [Ex; Ey]``.
        qmat: ``q_ep + q_partial``; maps ``v`` to the (unscaled) transverse magnetic field.
        q_ep: The permittivity block of ``qmat``, used to recover ``Ez``.
        inv_eps_zz: Diagonal of ``1 / eps_zz``.
        inv_mu_zz: Diagonal of ``1 / mu_zz``.
        der_mats: The four difference matrices ``(dxf, dxb, dyf, dyb)`` after division by ``k0``.
    """

    mat: sp.csr_matrix
    qmat: sp.csr_matrix
    q_ep: sp.csr_matrix
    inv_eps_zz: sp.csr_matrix
    inv_mu_zz: sp.csr_matrix
    der_mats: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]


def assemble_mode_operator(
    eps_xx: np.ndarray,
    eps_yy: np.ndarray,
    eps_zz: np.ndarray,
    mu_xx: np.ndarray,
    mu_yy: np.ndarray,
    mu_zz: np.ndarray,
    der_mats: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    k0: float,
    eps_xy: np.ndarray | None = None,
    eps_yx: np.ndarray | None = None,
) -> ModeOperator:
    """Assemble the ``k0``-normalized transverse-E mode operator without solving anything.

    The transverse off-diagonal permittivity enters in exactly one place. Writing the transverse
    electric displacement as ``D_x = eps_xx Ex + eps_xy Ey`` and ``D_y = eps_yx Ex + eps_yy Ey``,
    the operator is built from

    .. code-block:: text

        q_ep = [[ eps_yx,  eps_yy],        (diagonal media: [[0, eps_yy], [-eps_xx, 0]])
                [-eps_xx, -eps_xy]]

    and *every* other block is unchanged: ``mat = p_mu qmat + p_partial q_ep`` still holds, because
    the two places the permittivity appears in the transverse-E derivation are the constitutive
    relation ``D = eps E`` (this block) and the longitudinal divergence ``div(D_t) = -i n eps_zz Ez``,
    which is the same block again. So the whole tensor extension is one substitution of ``D`` for
    ``eps_cc E_c``, and passing ``eps_xy = eps_yx = None`` restores the diagonal assembly entry for
    entry rather than by cancellation.

    Args:
        eps_xx, eps_yy, eps_zz: flattened (length ``N``, C-order) diagonal relative permittivity
            components sampled at the Yee Ex/Ey/Ez locations.
        mu_xx, mu_yy, mu_zz: flattened diagonal relative permeability components.
        der_mats: ``(dxf, dxb, dyf, dyb)`` SI difference matrices from
            :func:`fdtdx.core.physics.mode_backend.operator.build_derivative_matrices`.
        k0: free-space wavenumber ``2 pi f / c`` (1/m).
        eps_xy: flattened ``eps_xy``, or ``None`` for a medium with no transverse off-diagonal.
        eps_yx: flattened ``eps_yx``, or ``None``. ``eps_yx != eps_xy`` is a non-reciprocal medium;
            it assembles, but the closed-form left eigenvector of the differentiable path does not
            apply to it (see :mod:`fdtdx.core.physics.mode_backend.jax_solve`).

    Returns:
        ModeOperator: The operator and its blocks.

    Raises:
        ValueError: If only one of ``eps_xy`` / ``eps_yx`` is given.
    """
    if (eps_xy is None) != (eps_yx is None):
        raise ValueError("pass both eps_xy and eps_yx, or neither")
    n = eps_xx.size
    # The mode solve is unconditionally double precision: a simulation running at float32 still
    # gets its operator assembled at complex128, because the eigenvalue is quadratic in n_eff and a
    # single-precision material rounds n_eff at the 1e-7 level before the solver ever sees it.
    eps_xx, eps_yy, eps_zz, mu_xx, mu_yy, mu_zz = (
        np.asarray(component, dtype=np.complex128) for component in (eps_xx, eps_yy, eps_zz, mu_xx, mu_yy, mu_zz)
    )
    # Normalise derivatives by k0 (dimensionless operator); eigenvalue is then -(neff)^2. The
    # difference matrices are real float64 by construction (they come from the edge coordinates).
    dxf, dxb, dyf, dyb = (m.astype(np.float64) / float(k0) for m in der_mats)

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
    if eps_xy is None or eps_yx is None:
        q_ep = sp.bmat([[None, _spdiag(eps_yy, n)], [_spdiag(-eps_xx, n), None]], format="csr")
    else:
        eps_xy_c = np.asarray(eps_xy, dtype=np.complex128)
        eps_yx_c = np.asarray(eps_yx, dtype=np.complex128)
        q_ep = sp.bmat(
            [
                [_spdiag(eps_yx_c, n), _spdiag(eps_yy, n)],
                [_spdiag(-eps_xx, n), _spdiag(-eps_xy_c, n)],
            ],
            format="csr",
        )
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
    return ModeOperator(
        mat=mat,
        qmat=qmat,
        q_ep=q_ep,
        inv_eps_zz=inv_eps_zz,
        inv_mu_zz=inv_mu_zz,
        der_mats=(dxf, dxb, dyf, dyb),
    )


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
    eps_xy: np.ndarray | None = None,
    eps_yx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the full-vectorial mode problem.

    The name is historical: the solve carries the transverse off-diagonal permittivity too, through
    the optional ``eps_xy`` / ``eps_yx`` arguments. Left out, the assembly is the diagonal one entry
    for entry.

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
        eps_xy: flattened ``eps_xy``, or ``None`` for no transverse off-diagonal.
        eps_yx: flattened ``eps_yx``, or ``None``.

    Returns:
        ``(E, H, neff, keff)`` where ``E`` and ``H`` have shape ``(3, N, num_modes)`` and ``neff``,
        ``keff`` have shape ``(num_modes,)``.
    """
    n = eps_xx.size
    operator = assemble_mode_operator(
        eps_xx, eps_yy, eps_zz, mu_xx, mu_yy, mu_zz, der_mats, k0, eps_xy=eps_xy, eps_yx=eps_yx
    )
    mat, qmat, q_ep = operator.mat, operator.qmat, operator.q_ep
    inv_eps_zz, inv_mu_zz = operator.inv_eps_zz, operator.inv_mu_zz
    dxf, dxb, dyf, dyb = operator.der_mats

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
    order = np.argsort(np.real(n_complex))[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    return reconstruct_fields(
        vecs=vecs,
        eigenvalues=vals,
        qmat=qmat,
        q_ep=q_ep,
        inv_eps_zz=inv_eps_zz,
        inv_mu_zz=inv_mu_zz,
        der_mats=(dxf, dxb, dyf, dyb),
        direction=direction,
    )


def reconstruct_fields(
    vecs: np.ndarray,
    eigenvalues: np.ndarray,
    qmat: sp.csr_matrix,
    q_ep: sp.csr_matrix,
    inv_eps_zz: sp.csr_matrix,
    inv_mu_zz: sp.csr_matrix,
    der_mats: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    direction: str = "+",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover the six field components from the transverse-E eigenvectors (stage 3).

    Split out of :func:`solve_modes_diagonal` so that the eigen-solve and the reconstruction can be
    exercised separately - in particular so the JAX rewrite in
    :func:`fdtdx.core.physics.mode_backend.jax_operator.reconstruct_fields_jax` can be compared
    against this one on the *same* eigenpairs, without ARPACK's Krylov space in the way.

    Args:
        vecs (np.ndarray): Transverse-E eigenvectors ``[Ex; Ey]``, shape ``(2N, M)``.
        eigenvalues (np.ndarray): The matching eigenvalues ``-(n_eff + i k_eff)^2``, shape ``(M,)``.
        qmat (sp.csr_matrix): ``q_ep + q_partial`` from the assembly.
        q_ep (sp.csr_matrix): The permittivity block of ``qmat``.
        inv_eps_zz (sp.csr_matrix): Diagonal of ``1 / eps_zz``.
        inv_mu_zz (sp.csr_matrix): Diagonal of ``1 / mu_zz``.
        der_mats: The four ``k0``-normalised difference matrices ``(dxf, dxb, dyf, dyb)``.
        direction (str): ``"+"`` or ``"-"`` propagation direction.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ``(E, H, neff, keff)`` with ``E``
        and ``H`` of shape ``(3, N, M)``.
    """
    dxf, dxb, dyf, dyb = der_mats
    n = vecs.shape[0] // 2

    n_complex = np.emath.sqrt(-np.asarray(eigenvalues) + 0j)
    neff = np.real(n_complex)
    keff = np.imag(n_complex)

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

    # Fix the global phase of every mode: the largest |E_t| entry is made real and positive, so a
    # lossless mode comes out with a real transverse E (the tidy3d convention, and what the
    # "+"/"-" reciprocity relation downstream assumes) instead of the arbitrary eigenvector phase.
    e_t = np.concatenate((E[0], E[1]), axis=0)
    pivot = e_t[np.argmax(np.abs(e_t), axis=0), np.arange(e_t.shape[1])]
    phase = np.where(np.abs(pivot) > 0, pivot / np.where(np.abs(pivot) > 0, np.abs(pivot), 1.0), 1.0)
    E = E / phase[None, None, :]
    H = H / phase[None, None, :]

    if direction == "-":
        H[0] *= -1
        H[1] *= -1
        E[2] *= -1

    return E, H, neff, keff
