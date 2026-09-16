"""Four-component eigenmode solver carrying the *full* permittivity tensor.

The transverse-E solver of :mod:`fdtdx.core.physics.mode_backend.solve` has eigenvalue
``-(n_eff)**2``. An eigenvalue that is a function of ``n_eff**2`` alone can only describe a medium
with a mirror symmetry about the cross-section plane, so the four entries that couple a transverse
axis to the propagation axis — ``eps_xz``, ``eps_zx``, ``eps_yz``, ``eps_zy`` — cannot enter it. The
bulk dispersion relation of a uniaxial crystal whose optic axis is tilted in the propagation plane
makes the obstruction explicit:

.. code-block:: text

    eps_zz n^2 + 2 eps_xz n s + eps_xx s^2 = n_o^2 n_e^2        (s = k_x / k0)

— the term linear in ``n`` is what the transverse formulation has no room for.

This module is the formulation that carries it: the eigenvector is the four transverse field
components ``[Ex, Ey, hx, hy]`` and the eigenvalue is ``n_eff`` itself, linear, in a plain (not
generalised) sparse eigenproblem. With ``exp(i (beta z - omega t))``, ``h = eta0 H``,
``n = beta / k0`` and every transverse derivative normalised by ``k0``, eliminating the two
longitudinal components from Maxwell's curl equations gives

.. code-block:: text

    Z = (1/eps_zz) [ i (Dx hy - Dy hx) - eps_zx Ex - eps_zy Ey ]      (= E_z,   on the node)
    K = (1/mu_zz)  [ Dx Ey - Dy Ex ]                                  (= i h_z, at the cell centre)

    n Ex = -i Dx Z + mu_yy hy
    n Ey = -i Dy Z - mu_xx hx
    n hx = -Dx K - ( eps_yx Ex + eps_yy Ey + eps_yz Z )
    n hy = -Dy K + ( eps_xx Ex + eps_xy Ey + eps_xz Z )

The longitudinal entries appear in exactly two places, and they are the same statement ``D = eps E``
read down a column (``eps_zx``, ``eps_zy`` inside ``Z``) and along a row (``eps_xz``, ``eps_yz``
multiplying ``Z``). A diagonal permeability is assumed, which is what makes ``E_z`` and ``h_z``
separable rather than a coupled 2x2 solve.

Relation to the transverse tier. Set the four entries to zero: ``Z`` then depends on ``h`` alone and
``K`` on ``E`` alone, the operator becomes block anti-diagonal ``M = [[0, A], [B, 0]]`` with
``A = p_partial + p_mu`` and ``B = -qmat`` — the existing blocks entry for entry — and

.. code-block:: text

    mat_transverse = -A B

exactly. So the transverse operator is the Schur square of this one and its eigenvalues ``-(n)^2``
are those of the ``+-n`` pairs here. That identity is what makes the tier switch a statement about
code paths rather than about tolerances.

Discretisation. The staggering is the one the existing difference matrices already imply: ``Ex`` and
``hy`` at the x-edge centres, ``Ey`` and ``hx`` at the y-edge centres, ``Z`` on the nodes, ``K`` at
the cell centres. The *new* terms need a half-cell move — ``eps_zx Ex`` is formed on the node and
``eps_xz Z`` on an edge centre — and taking them pointwise would place a first-difference symbol
half a cell from where it is used, a phase error of order ``dx``. They therefore go through the Yee
averages of :func:`fdtdx.core.physics.mode_backend.operator.build_average_matrices`, whose wall
images are the ones the difference matrices already encode.

Direction. The forward and backward modes of a medium without the cross-section mirror plane are
*not* mirror images of each other — that is the same statement as the odd power of ``beta`` above —
so ``direction="-"`` here solves the backward spectrum directly (shift at ``-neff_guess``) instead
of reflecting the forward one. With the four entries zero the two agree, which a test asserts.

Precision: ``complex128`` unconditionally, as in the transverse path.

Formulation provenance: the four-component transverse-field reduction is standard
(J. Vector formulations in e.g. Vassallo, *Optical Waveguide Concepts*, ch. 3; Berreman,
J. Opt. Soc. Am. 62, 502 (1972), for the 4x4 form in a stratified medium). Independent
implementation — no third-party solver code is copied.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from fdtdx.constants import eta0

__all__ = [
    "FullTensorOperator",
    "assemble_full_tensor_operator",
    "reconstruct_fields_full",
    "solve_modes_full_tensor",
]


def _spdiag(vec: np.ndarray, n: int) -> sp.csr_matrix:
    return sp.spdiags(np.asarray(vec, dtype=np.complex128), [0], n, n).tocsr()


class FullTensorOperator(NamedTuple):
    """The assembled four-component operator and the blocks the reconstruction needs.

    Attributes:
        mat: The ``(4N, 4N)`` operator with ``mat v = n_eff v`` for the stacked transverse field
            ``v = [Ex; Ey; hx; hy]``.
        z_blocks: ``(Z_Ex, Z_Ey, Z_hx, Z_hy)``, the four ``(N, N)`` blocks with
            ``E_z = Z_Ex Ex + Z_Ey Ey + Z_hx hx + Z_hy hy``.
        k_blocks: ``(K_Ex, K_Ey)`` with ``K = K_Ex Ex + K_Ey Ey`` and ``h_z = -i K``.
        num_cells: ``N``, the number of transverse cells.
    """

    mat: sp.csr_matrix
    z_blocks: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]
    k_blocks: tuple[sp.csr_matrix, sp.csr_matrix]
    num_cells: int


def assemble_full_tensor_operator(
    eps: dict[str, np.ndarray],
    mu_xx: np.ndarray,
    mu_yy: np.ndarray,
    mu_zz: np.ndarray,
    der_mats: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    avg_mats: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    k0: float,
) -> FullTensorOperator:
    """Assemble the ``k0``-normalised four-component mode operator without solving anything.

    Args:
        eps: The nine flattened (length ``N``, C-order) relative permittivity components, keyed
            ``"xx"``, ``"xy"``, ``"xz"``, ``"yx"``, ``"yy"``, ``"yz"``, ``"zx"``, ``"zy"``,
            ``"zz"``. Entries the caller does not have may be omitted and are taken as zero,
            except the three diagonal ones.
        mu_xx: Flattened ``mu_xx`` (at the ``hx`` locations).
        mu_yy: Flattened ``mu_yy``.
        mu_zz: Flattened ``mu_zz``.
        der_mats: ``(dxf, dxb, dyf, dyb)`` SI difference matrices from
            :func:`fdtdx.core.physics.mode_backend.operator.build_derivative_matrices`.
        avg_mats: ``(axb, axf, ayb, ayf)`` from
            :func:`fdtdx.core.physics.mode_backend.operator.build_average_matrices`.
        k0: Free-space wavenumber ``2 pi f / c`` (1/m).

    Returns:
        FullTensorOperator: The operator and the blocks the field reconstruction needs.

    Raises:
        KeyError: If one of ``"xx"``, ``"yy"``, ``"zz"`` is missing.
    """
    n = int(np.asarray(eps["xx"]).size)
    zero = np.zeros(n, dtype=np.complex128)

    def entry(name: str) -> np.ndarray:
        value = eps.get(name)
        return zero if value is None else np.asarray(value, dtype=np.complex128)

    e_xx, e_xy, e_xz = entry("xx"), entry("xy"), entry("xz")
    e_yx, e_yy, e_yz = entry("yx"), entry("yy"), entry("yz")
    e_zx, e_zy, e_zz = entry("zx"), entry("zy"), entry("zz")
    m_xx = np.asarray(mu_xx, dtype=np.complex128)
    m_yy = np.asarray(mu_yy, dtype=np.complex128)
    m_zz = np.asarray(mu_zz, dtype=np.complex128)

    dxf, dxb, dyf, dyb = (m.astype(np.float64) / float(k0) for m in der_mats)
    axb, axf, ayb, ayf = (m.astype(np.float64) for m in avg_mats)

    inv_eps_zz = _spdiag(1.0 / e_zz, n)
    inv_mu_zz = _spdiag(1.0 / m_zz, n)

    # E_z, on the node. The two longitudinal entries of the bottom row of eps enter here.
    z_ex = -(inv_eps_zz.dot(_spdiag(e_zx, n))).dot(axb)
    z_ey = -(inv_eps_zz.dot(_spdiag(e_zy, n))).dot(ayb)
    z_hx = -1j * inv_eps_zz.dot(dyb)
    z_hy = 1j * inv_eps_zz.dot(dxb)

    # i h_z, at the cell centre.
    k_ex = -inv_mu_zz.dot(dyf)
    k_ey = inv_mu_zz.dot(dxf)

    # The two longitudinal entries of the right column of eps. They multiply E_z, so they are
    # sampled where E_z is — on the node, with eps_zz — and the product is averaged onto the
    # transverse location afterwards. Sampling them at the transverse location instead (averaging
    # E_z first, then multiplying) is equally plausible per-term and breaks discrete reciprocity:
    # measured on a lossless reciprocal cross-section with a 0.3 longitudinal entry, it puts an
    # imaginary part of 7.6e-4 on n_eff at 100 nm falling only as the first power of the cell size,
    # where this order keeps it at 2e-16 on every grid.
    xz = axf.dot(_spdiag(e_xz, n))
    yz = ayf.dot(_spdiag(e_yz, n))

    row_ex = [-1j * dxf.dot(z_ex), -1j * dxf.dot(z_ey), -1j * dxf.dot(z_hx), -1j * dxf.dot(z_hy) + _spdiag(m_yy, n)]
    row_ey = [-1j * dyf.dot(z_ex), -1j * dyf.dot(z_ey), -1j * dyf.dot(z_hx) - _spdiag(m_xx, n), -1j * dyf.dot(z_hy)]
    row_hx = [
        -dxb.dot(k_ex) - _spdiag(e_yx, n) - yz.dot(z_ex),
        -dxb.dot(k_ey) - _spdiag(e_yy, n) - yz.dot(z_ey),
        -yz.dot(z_hx),
        -yz.dot(z_hy),
    ]
    row_hy = [
        -dyb.dot(k_ex) + _spdiag(e_xx, n) + xz.dot(z_ex),
        -dyb.dot(k_ey) + _spdiag(e_xy, n) + xz.dot(z_ey),
        xz.dot(z_hx),
        xz.dot(z_hy),
    ]
    mat = sp.bmat([row_ex, row_ey, row_hx, row_hy], format="csr")
    return FullTensorOperator(
        mat=mat.astype(np.complex128),
        z_blocks=(z_ex, z_ey, z_hx, z_hy),
        k_blocks=(k_ex, k_ey),
        num_cells=n,
    )


def reconstruct_fields_full(
    operator: FullTensorOperator,
    vecs: np.ndarray,
    eigenvalues: np.ndarray,
    direction: str = "+",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover the six field components from the four-component eigenvectors.

    Four of the six are the eigenvector itself, which is the practical difference from the
    transverse path: only ``E_z`` and ``h_z`` are rebuilt, and both are one sparse product.

    Args:
        operator (FullTensorOperator): The assembled operator.
        vecs (np.ndarray): Eigenvectors ``[Ex; Ey; hx; hy]``, shape ``(4N, M)``.
        eigenvalues (np.ndarray): The matching eigenvalues ``n_eff + i k_eff`` (or their negatives
            for ``direction="-"``), shape ``(M,)``.
        direction (str): ``"+"`` or ``"-"``. The eigenvalues are expected to carry the sign of the
            direction already; this argument only decides how they are reported.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ``(E, H, neff, keff)`` with ``E``
        and ``H`` of shape ``(3, N, M)``, in the same convention as
        :func:`fdtdx.core.physics.mode_backend.solve.reconstruct_fields` (``H`` pre-scaled by
        ``1 / eta0``, global phase fixed on the largest transverse-E entry).
    """
    n = operator.num_cells
    z_ex, z_ey, z_hx, z_hy = operator.z_blocks
    k_ex, k_ey = operator.k_blocks

    n_complex = np.asarray(eigenvalues, dtype=np.complex128)
    if direction == "-":
        n_complex = -n_complex
    neff = np.real(n_complex)
    keff = np.imag(n_complex)

    # Gauge: an eigenvector is defined up to a complex factor and the eigensolver picks one, so
    # divide it out here exactly as the transverse path does.
    scale = np.linalg.norm(vecs, axis=0)
    vecs = vecs / np.where(scale > 0, scale, 1.0)[None, :]

    ex = vecs[:n, :]
    ey = vecs[n : 2 * n, :]
    hx = vecs[2 * n : 3 * n, :]
    hy = vecs[3 * n :, :]

    ez = z_ex.dot(ex) + z_ey.dot(ey) + z_hx.dot(hx) + z_hy.dot(hy)
    hz = -1j * (k_ex.dot(ex) + k_ey.dot(ey))

    field_e = np.stack((ex, ey, ez), axis=0)
    field_h = np.stack((hx, hy, hz), axis=0) / eta0

    e_t = np.concatenate((field_e[0], field_e[1]), axis=0)
    pivot = e_t[np.argmax(np.abs(e_t), axis=0), np.arange(e_t.shape[1])]
    phase = np.where(np.abs(pivot) > 0, pivot / np.where(np.abs(pivot) > 0, np.abs(pivot), 1.0), 1.0)
    field_e = field_e / phase[None, None, :]
    field_h = field_h / phase[None, None, :]
    return field_e, field_h, neff, keff


def _sort_order(vals: np.ndarray, direction: str) -> np.ndarray:
    """Descending ``Re(n_eff)`` of the *physical* index, whichever direction was solved for."""
    key = np.real(vals) if direction == "+" else -np.real(vals)
    return np.argsort(key)[::-1]


def solve_modes_full_tensor(
    eps: dict[str, np.ndarray],
    mu_xx: np.ndarray,
    mu_yy: np.ndarray,
    mu_zz: np.ndarray,
    der_mats: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    avg_mats: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    k0: float,
    num_modes: int,
    neff_guess: float,
    direction: str = "+",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the mode problem with every entry of the permittivity tensor carried.

    Args:
        eps: The nine flattened relative permittivity components (see
            :func:`assemble_full_tensor_operator`).
        mu_xx: Flattened ``mu_xx``.
        mu_yy: Flattened ``mu_yy``.
        mu_zz: Flattened ``mu_zz``.
        der_mats: ``(dxf, dxb, dyf, dyb)`` SI difference matrices.
        avg_mats: ``(axb, axf, ayb, ayf)`` averaging matrices.
        k0: Free-space wavenumber ``2 pi f / c`` (1/m).
        num_modes: Number of modes to return, sorted by descending ``Re(n_eff)``.
        neff_guess: Shift-invert target effective index (positive; the sign of ``direction`` is
            applied here).
        direction: ``"+"`` or ``"-"``. The backward spectrum is solved directly rather than
            reflected, because a medium carrying the longitudinal entries has no mirror plane at
            the cross-section.

    Returns:
        ``(E, H, neff, keff)`` where ``E`` and ``H`` have shape ``(3, N, num_modes)`` and ``neff``,
        ``keff`` have shape ``(num_modes,)``.
    """
    operator = assemble_full_tensor_operator(eps, mu_xx, mu_yy, mu_zz, der_mats, avg_mats, k0)
    size = 4 * operator.num_cells
    rng = np.random.default_rng(0)
    vec_init = rng.random(size) + 1j * rng.random(size)
    num_modes = min(num_modes, size - 2)
    shift = float(neff_guess) if direction == "+" else -float(neff_guess)
    vals, vecs = spl.eigs(operator.mat, k=num_modes, sigma=shift, v0=vec_init)
    order = _sort_order(vals, direction)
    return reconstruct_fields_full(operator, vecs[:, order], vals[order], direction=direction)
