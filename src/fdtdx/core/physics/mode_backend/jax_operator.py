"""JAX-native assembly and field reconstruction for the transverse-E mode operator.

The numpy/scipy pipeline in :mod:`fdtdx.core.physics.mode_backend.solve` is three separable
stages: assemble the operator from the permittivity, solve the sparse eigenproblem, reconstruct the
six field components. Only the middle stage is opaque to automatic differentiation. This module is
the JAX rewrite of stages 1 and 3, so that a gradient can flow from the permittivity into the
operator and out through the reconstructed fields, with the eigen-solve isolated behind the single
``custom_vjp`` of :mod:`fdtdx.core.physics.mode_backend.jax_solve`.

How the sparsity is handled. Every entry of the operator is a rational function of the permittivity
and the grid steps, and the *pattern* depends only on the grid. Each operator block is a product
``A diag(d) B`` of two difference matrices around one diagonal material factor, so its entries are
the sums

.. code-block:: text

    (A diag(d) B)[i, j] = sum_k A[i, k] * d[k] * B[k, j]

whose ``(i, j, k)`` index triples are fixed by the grid. This module precomputes those triples once
in scipy (``_sandwich_terms``) and then evaluates the values in ``jnp``. Contributions are *not*
merged: the operator is carried as a coordinate list with duplicate ``(row, col)`` entries, which
:class:`jax.experimental.sparse.BCOO` and ``scipy.sparse.coo_matrix`` both sum on use. That removes
all index bookkeeping from the differentiable path — the only traced object is a flat vector of
values.

Precision. The operator is assembled at ``complex128`` unconditionally, as the numpy path is, which
under JAX means the module refuses to run without ``jax_enable_x64``. The eigenvalue is
``-(n_eff)^2``, so single precision costs about 1e-7 of ``n_eff`` before the eigensolver starts and
no gradient check survives that.

Scope. Diagonal media, straight waveguides, PEC walls at both edges of both transverse axes — the
same tier the numpy backend supports. PMC (``symmetry=1``) walls assemble correctly but have no
exact discrete left eigenvector in closed form (see :mod:`jax_solve`), so only their forward is
supported here.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.experimental import sparse as jsparse

from fdtdx.constants import eta0

__all__ = [
    "JaxModeOperator",
    "SparseCOO",
    "assemble_mode_operator_jax",
    "reconstruct_fields_jax",
    "require_x64",
]


def require_x64(what: str) -> None:
    """Refuse to build a single-precision mode operator.

    Args:
        what (str): Name of the caller, used in the message.

    Raises:
        ValueError: If ``jax_enable_x64`` is off, in which case ``jnp`` cannot represent
            ``complex128`` at all and the assembly would silently round the material.
    """
    if not jax.config.jax_enable_x64:
        raise ValueError(
            f"{what} needs double precision: the mode eigenvalue is -(n_eff)^2, so a complex64 "
            "operator perturbs n_eff at the 1e-7 level before the eigensolver starts. JAX cannot "
            "hold complex128 with x64 disabled - call jax.config.update('jax_enable_x64', True) "
            "at process start, or use the numpy backend (fdtdx.core.physics.mode_backend.solve), "
            "which is double precision whatever the caller's dtype."
        )


class SparseCOO(NamedTuple):
    """A sparse matrix as a coordinate list whose pattern is static and whose values are traced.

    Duplicate ``(row, col)`` pairs are allowed and are summed on use, which is what lets the
    assembly emit one entry per contribution instead of merging patterns.

    Attributes:
        rows: Row indices, a concrete numpy array (never traced).
        cols: Column indices, a concrete numpy array (never traced).
        data: Values, a JAX array of length ``len(rows)``; this is the differentiable object.
        shape: Matrix shape.
    """

    rows: np.ndarray
    cols: np.ndarray
    data: jax.Array
    shape: tuple[int, int]

    def to_bcoo(self) -> jsparse.BCOO:
        """Return the same matrix as a :class:`jax.experimental.sparse.BCOO`.

        Returns:
            jsparse.BCOO: The sparse matrix; duplicate indices are summed by every BCOO operation.
        """
        indices = jnp.stack((jnp.asarray(self.rows), jnp.asarray(self.cols)), axis=1)
        return jsparse.BCOO((self.data, indices), shape=self.shape)

    def matmul(self, x: jax.Array) -> jax.Array:
        """Multiply by a dense matrix or vector.

        Args:
            x (jax.Array): Dense operand of shape ``(n,)`` or ``(n, m)``.

        Returns:
            jax.Array: The product, of shape ``(m_rows,)`` or ``(m_rows, m)``.
        """
        vec = x.ndim == 1
        dense = x[:, None] if vec else x
        contrib = self.data[:, None] * dense[self.cols, :]
        out = jax.ops.segment_sum(contrib, jnp.asarray(self.rows), num_segments=self.shape[0])
        return out[:, 0] if vec else out

    def to_scipy(self) -> sp.csr_matrix:
        """Materialise the matrix in scipy, summing duplicate entries.

        Returns:
            sp.csr_matrix: The same matrix in CSR form at ``complex128``.
        """
        data = np.asarray(jax.lax.stop_gradient(self.data), dtype=np.complex128)
        return sp.coo_matrix((data, (self.rows, self.cols)), shape=self.shape).tocsr()


class JaxModeOperator(NamedTuple):
    """The JAX-assembled mode operator and everything stage 3 needs from it.

    Attributes:
        mat: The ``(2N, 2N)`` operator with ``mat v = -(n_eff + i k_eff)^2 v``.
        qmat: ``q_ep + q_partial``; maps the transverse E field to the unscaled transverse H field.
        eps_xx: Flattened ``eps_xx``, kept so stage 3 can apply ``q_ep`` without a matmul.
        eps_yy: Flattened ``eps_yy``.
        eps_xy: Flattened ``eps_xy``, or ``None`` when the medium has no transverse off-diagonal.
        eps_yx: Flattened ``eps_yx``, or ``None``.
        inv_eps_zz: Flattened ``1 / eps_zz``.
        inv_mu_zz: Flattened ``1 / mu_zz``.
        der_bcoo: The four ``k0``-normalised difference matrices ``(dxf, dxb, dyf, dyb)`` as BCOO.
        left_weights: ``(w_x, w_y)``, the two staggered cell areas that turn the rotated transverse
            H field into the discrete left eigenvector (see :mod:`jax_solve`).
        num_cells: ``N``, the number of transverse cells.
    """

    mat: SparseCOO
    qmat: SparseCOO
    eps_xx: jax.Array
    eps_yy: jax.Array
    eps_xy: jax.Array | None
    eps_yx: jax.Array | None
    inv_eps_zz: jax.Array
    inv_mu_zz: jax.Array
    der_bcoo: tuple[jsparse.BCOO, jsparse.BCOO, jsparse.BCOO, jsparse.BCOO]
    left_weights: tuple[np.ndarray, np.ndarray]
    num_cells: int


def _sandwich_terms(a: sp.spmatrix, b: sp.spmatrix) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Index triples of the product ``A diag(d) B``, with the diagonal left symbolic.

    Args:
        a (sp.spmatrix): Left factor.
        b (sp.spmatrix): Right factor.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ``(rows, cols, coef, kidx)`` such
        that ``(A diag(d) B)[rows[t], cols[t]] += coef[t] * d[kidx[t]]``, with duplicates.
    """
    a_coo = sp.coo_matrix(a)
    b_csr = sp.csr_matrix(b)
    if a_coo.nnz == 0 or b_csr.nnz == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.zeros(0, dtype=np.float64), empty_i.copy()
    counts = np.diff(b_csr.indptr)[a_coo.col]
    total = int(counts.sum())
    if total == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.zeros(0, dtype=np.float64), empty_i.copy()
    rep = np.repeat(np.arange(a_coo.nnz), counts)
    group_start = np.repeat(np.cumsum(counts) - counts, counts)
    pos = b_csr.indptr[a_coo.col][rep] + (np.arange(total) - group_start)
    rows = a_coo.row[rep].astype(np.int64)
    cols = b_csr.indices[pos].astype(np.int64)
    coef = a_coo.data[rep] * b_csr.data[pos]
    kidx = a_coo.col[rep].astype(np.int64)
    return rows, cols, coef, kidx


class _Block(NamedTuple):
    """One assembled block term: values ``coef * d[kidx]`` placed at ``(rows, cols)``."""

    rows: np.ndarray
    cols: np.ndarray
    coef: np.ndarray
    kidx: np.ndarray


def _sandwich_block(
    a: sp.spmatrix,
    b: sp.spmatrix,
    sign: complex,
    row_offset: int,
    col_offset: int,
) -> _Block:
    """Build one ``sign * A diag(d) B`` block, shifted into the block layout.

    ``sign`` may be complex: the four-component operator of
    :mod:`fdtdx.core.physics.mode_backend.jax_full_tensor` carries factors of ``i`` on the terms
    that pass through the longitudinal constitutive relation.
    """
    rows, cols, coef, kidx = _sandwich_terms(a, b)
    return _Block(rows + row_offset, cols + col_offset, sign * coef, kidx)


def _stack_blocks(
    blocks: list[tuple[_Block, jax.Array, jax.Array | None, jax.Array | None]],
    shape: tuple[int, int],
    num_cells: int | None = None,
) -> SparseCOO:
    """Concatenate blocks into one coordinate list, applying row and column scalings.

    Args:
        blocks: ``(block, diag, row_scale, col_scale)`` tuples. ``diag`` is the material vector the
            block's ``kidx`` indexes; ``row_scale`` / ``col_scale`` are optional per-cell factors
            applied by the block's row / column index (each of length ``N``, i.e. within a block,
            so the offsets are removed before indexing).
        shape: Shape of the assembled matrix.
        num_cells: ``N``, the block size the row and column offsets are removed modulo. Defaults to
            ``shape[0] // 2``, which is the transverse operator's two-block layout; the
            four-component operator passes it explicitly.

    Returns:
        SparseCOO: The concatenated coordinate list.
    """
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    data_all: list[jax.Array] = []
    n = shape[0] // 2 if num_cells is None else num_cells
    for block, diag, row_scale, col_scale in blocks:
        values = jnp.asarray(block.coef, dtype=jnp.complex128) * diag[block.kidx]
        if row_scale is not None:
            values = values * row_scale[block.rows % n]
        if col_scale is not None:
            values = values * col_scale[block.cols % n]
        rows_all.append(block.rows)
        cols_all.append(block.cols)
        data_all.append(values)
    return SparseCOO(
        rows=np.concatenate(rows_all),
        cols=np.concatenate(cols_all),
        data=jnp.concatenate(data_all),
        shape=shape,
    )


def assemble_mode_operator_jax(
    eps_xx: jax.Array,
    eps_yy: jax.Array,
    eps_zz: jax.Array,
    mu_xx: jax.Array,
    mu_yy: jax.Array,
    mu_zz: jax.Array,
    der_mats: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    k0: float,
    cell_steps: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    eps_xy: jax.Array | None = None,
    eps_yx: jax.Array | None = None,
) -> JaxModeOperator:
    """Assemble the ``k0``-normalised transverse-E mode operator in JAX.

    Algebraically identical to :func:`fdtdx.core.physics.mode_backend.solve.assemble_mode_operator`;
    the block products are expanded so that the only traced quantities are the material vectors.
    ``mat = p_mu qmat + p_partial q_ep`` becomes, block by block,

    .. code-block:: text

        mat11 =  mu_yy Q21 - diag(mu_yy eps_xx) - P12 eps_xx
        mat12 =  mu_yy Q22 + P11 eps_yy
        mat21 = -mu_xx Q11 - P22 eps_xx
        mat22 = -mu_xx Q12 - diag(mu_xx eps_yy) + P21 eps_yy

    with ``P.. = +-d.f (1/eps_zz) d.b`` and ``Q.. = +-d.b (1/mu_zz) d.f``.

    With a transverse off-diagonal permittivity the substitution is ``eps_xx Ex -> D_x`` and
    ``eps_yy Ey -> D_y`` wherever the permittivity multiplies a field, i.e. four more terms:

    .. code-block:: text

        mat11 += P11 eps_yx
        mat12 += -diag(mu_yy eps_xy) - P12 eps_xy
        mat21 += -diag(mu_xx eps_yx) + P21 eps_yx
        mat22 += -P22 eps_xy

    and two more in ``qmat`` (the ``q_ep`` block becomes ``[[eps_yx, eps_yy], [-eps_xx, -eps_xy]]``).
    Left at ``None``, no term is emitted at all, so the diagonal operator is assembled entry for
    entry rather than by cancellation.

    Args:
        eps_xx (jax.Array): Flattened (length ``N``, C-order) ``eps_xx`` at the Yee Ex locations.
        eps_yy (jax.Array): Flattened ``eps_yy``.
        eps_zz (jax.Array): Flattened ``eps_zz``.
        mu_xx (jax.Array): Flattened ``mu_xx``.
        mu_yy (jax.Array): Flattened ``mu_yy``.
        mu_zz (jax.Array): Flattened ``mu_zz``.
        der_mats: ``(dxf, dxb, dyf, dyb)`` SI difference matrices from
            :func:`fdtdx.core.physics.mode_backend.operator.build_derivative_matrices`.
        k0 (float): Free-space wavenumber ``2 pi f / c`` (1/m).
        cell_steps: ``((dlf_x, dlb_x), (dlf_y, dlb_y))`` from
            :func:`fdtdx.core.physics.mode_backend.operator.primal_dual_steps`, used only to build
            the left-eigenvector weights.
        eps_xy (jax.Array | None): Flattened ``eps_xy``, or ``None`` for no transverse off-diagonal.
        eps_yx (jax.Array | None): Flattened ``eps_yx``, or ``None``.

    Returns:
        JaxModeOperator: The operator and the pieces stage 3 needs.

    Raises:
        ValueError: If ``jax_enable_x64`` is off, or if only one of ``eps_xy`` / ``eps_yx`` is given.
    """
    require_x64("assemble_mode_operator_jax")
    if (eps_xy is None) != (eps_yx is None):
        raise ValueError("pass both eps_xy and eps_yx, or neither")
    eps_xx, eps_yy, eps_zz, mu_xx, mu_yy, mu_zz = (
        jnp.asarray(component, dtype=jnp.complex128) for component in (eps_xx, eps_yy, eps_zz, mu_xx, mu_yy, mu_zz)
    )
    if eps_xy is not None and eps_yx is not None:
        eps_xy = jnp.asarray(eps_xy, dtype=jnp.complex128)
        eps_yx = jnp.asarray(eps_yx, dtype=jnp.complex128)
    n = int(eps_xx.size)
    dxf, dxb, dyf, dyb = (sp.csr_matrix(m, dtype=np.float64) / float(k0) for m in der_mats)

    inv_eps_zz = 1.0 / eps_zz
    inv_mu_zz = 1.0 / mu_zz

    # P blocks (material factor 1 / eps_zz) and Q blocks (material factor 1 / mu_zz).
    p11 = _sandwich_block(dxf, dyb, -1.0, 0, 0)
    p12 = _sandwich_block(dxf, dxb, +1.0, 0, 0)
    p21 = _sandwich_block(dyf, dyb, -1.0, 0, 0)
    p22 = _sandwich_block(dyf, dxb, +1.0, 0, 0)
    q11 = _sandwich_block(dxb, dyf, -1.0, 0, 0)
    q12 = _sandwich_block(dxb, dxf, +1.0, 0, 0)
    q21 = _sandwich_block(dyb, dyf, -1.0, 0, 0)
    q22 = _sandwich_block(dyb, dxf, +1.0, 0, 0)

    def shifted(block: _Block, row_offset: int, col_offset: int) -> _Block:
        return _Block(block.rows + row_offset, block.cols + col_offset, block.coef, block.kidx)

    diag_idx = np.arange(n, dtype=np.int64)
    ones = np.ones(n, dtype=np.float64)

    def diag_block(row_offset: int, col_offset: int, sign: float) -> _Block:
        return _Block(diag_idx + row_offset, diag_idx + col_offset, sign * ones, diag_idx)

    mat_blocks: list[tuple[_Block, jax.Array, jax.Array | None, jax.Array | None]] = [
        # mat11
        (shifted(q21, 0, 0), inv_mu_zz, mu_yy, None),
        (diag_block(0, 0, -1.0), mu_yy * eps_xx, None, None),
        (shifted(p12, 0, 0), -inv_eps_zz, None, eps_xx),
        # mat12
        (shifted(q22, 0, n), inv_mu_zz, mu_yy, None),
        (shifted(p11, 0, n), inv_eps_zz, None, eps_yy),
        # mat21
        (shifted(q11, n, 0), -inv_mu_zz, mu_xx, None),
        (shifted(p22, n, 0), -inv_eps_zz, None, eps_xx),
        # mat22
        (shifted(q12, n, n), -inv_mu_zz, mu_xx, None),
        (diag_block(n, n, -1.0), mu_xx * eps_yy, None, None),
        (shifted(p21, n, n), inv_eps_zz, None, eps_yy),
    ]
    qmat_blocks: list[tuple[_Block, jax.Array, jax.Array | None, jax.Array | None]] = [
        (shifted(q11, 0, 0), inv_mu_zz, None, None),
        (shifted(q12, 0, n), inv_mu_zz, None, None),
        (diag_block(0, n, +1.0), eps_yy, None, None),
        (shifted(q21, n, 0), inv_mu_zz, None, None),
        (diag_block(n, 0, -1.0), eps_xx, None, None),
        (shifted(q22, n, n), inv_mu_zz, None, None),
    ]
    if eps_xy is not None and eps_yx is not None:
        mat_blocks += [
            (shifted(p11, 0, 0), inv_eps_zz, None, eps_yx),
            (diag_block(0, n, -1.0), mu_yy * eps_xy, None, None),
            (shifted(p12, 0, n), -inv_eps_zz, None, eps_xy),
            (diag_block(n, 0, -1.0), mu_xx * eps_yx, None, None),
            (shifted(p21, n, 0), inv_eps_zz, None, eps_yx),
            (shifted(p22, n, n), -inv_eps_zz, None, eps_xy),
        ]
        qmat_blocks += [
            (diag_block(0, 0, +1.0), eps_yx, None, None),
            (diag_block(n, n, -1.0), eps_xy, None, None),
        ]

    mat = _stack_blocks(mat_blocks, shape=(2 * n, 2 * n))
    qmat = _stack_blocks(qmat_blocks, shape=(2 * n, 2 * n))

    (dlf_x, dlb_x), (dlf_y, dlb_y) = cell_steps
    left_weights = (
        np.outer(dlf_x, dlb_y).ravel(),
        np.outer(dlb_x, dlf_y).ravel(),
    )

    def to_bcoo(m: sp.csr_matrix) -> jsparse.BCOO:
        coo = sp.coo_matrix(m)
        indices = jnp.stack((jnp.asarray(coo.row), jnp.asarray(coo.col)), axis=1)
        return jsparse.BCOO((jnp.asarray(coo.data, dtype=jnp.complex128), indices), shape=coo.shape)

    return JaxModeOperator(
        mat=mat,
        qmat=qmat,
        eps_xx=eps_xx,
        eps_yy=eps_yy,
        eps_xy=eps_xy,
        eps_yx=eps_yx,
        inv_eps_zz=inv_eps_zz,
        inv_mu_zz=inv_mu_zz,
        der_bcoo=(to_bcoo(dxf), to_bcoo(dxb), to_bcoo(dyf), to_bcoo(dyb)),
        left_weights=left_weights,
        num_cells=n,
    )


def left_eigenvectors_jax(operator: JaxModeOperator, vecs: jax.Array) -> jax.Array:
    """Discrete left eigenvectors of the mode operator, for free from the right ones.

    The transverse-E operator is not symmetric, but it is *reciprocal*: with
    ``h = qmat v = [hx; hy]`` the unscaled transverse magnetic field of the mode, the vector

    .. code-block:: text

        u = [ w_x * hy ; -w_y * hx ],   w_x = dlf_x (x) dlb_y,  w_y = dlb_x (x) dlf_y

    satisfies ``u^T mat = lambda u^T`` to machine precision, on uniform and non-uniform grids, for
    complex permittivity and anisotropic permeability alike. The two weights are the staggered cell
    areas at the Hy and Hx sample points; they are what makes the relation hold on a graded mesh
    (without them the residual is a few percent). This is the discrete form of the reciprocity
    statement that the backward mode carries ``-H_t``, and it is why the eigenvalue adjoint costs one
    contraction instead of a second eigen-solve.

    The relation is exact for PEC walls at the min edge of both axes. A PMC (``symmetry=1``) min edge
    breaks it at the wall row (measured residual 7e-2), because the backward difference there carries
    a ``2`` on its diagonal and has no matching entry to transpose onto.

    Args:
        operator (JaxModeOperator): The assembled operator.
        vecs (jax.Array): Right eigenvectors, shape ``(2N, M)``.

    Returns:
        jax.Array: Left eigenvectors, shape ``(2N, M)``, in the same order.
    """
    n = operator.num_cells
    h = operator.qmat.matmul(vecs)
    w_x = jnp.asarray(operator.left_weights[0], dtype=jnp.complex128)[:, None]
    w_y = jnp.asarray(operator.left_weights[1], dtype=jnp.complex128)[:, None]
    return jnp.concatenate((w_x * h[n:, :], -w_y * h[:n, :]), axis=0)


def reconstruct_fields_jax(
    operator: JaxModeOperator,
    vecs: jax.Array,
    eigenvalues: jax.Array,
    direction: str = "+",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Recover the six field components from the transverse-E eigenvectors, in JAX.

    The JAX rewrite of stage 3 of :func:`fdtdx.core.physics.mode_backend.solve.solve_modes_diagonal`:
    ``Hx``/``Hy`` from ``qmat``, ``Hz`` and ``Ez`` from the curl relations, the ``-1j / eta0`` scaling
    of ``H``, and the pivot phase fix that makes the largest transverse-E entry real and positive.
    Everything here is traceable, so a permittivity gradient reaches the fields through the material
    factors even while the eigenvector itself is held fixed.

    Args:
        operator (JaxModeOperator): The assembled operator.
        vecs (jax.Array): Transverse-E eigenvectors ``[Ex; Ey]``, shape ``(2N, M)``.
        eigenvalues (jax.Array): The matching eigenvalues ``-(n_eff + i k_eff)^2``, shape ``(M,)``.
        direction (str): ``"+"`` or ``"-"`` propagation direction.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array, jax.Array]: ``(E, H, neff, keff)`` with ``E`` and
        ``H`` of shape ``(3, N, M)`` and the indices of shape ``(M,)``.

    Raises:
        ValueError: If ``jax_enable_x64`` is off. Double precision is a property of the whole
            differentiable path, not only of the assembly: ``Ez`` is a difference of two nearly
            equal terms divided by ``i n_eff``, so a complex64 reconstruction loses about seven
            digits of the longitudinal component even when the eigenvector is exact.
    """
    require_x64("reconstruct_fields_jax")
    n = operator.num_cells
    dxf, dxb, dyf, dyb = operator.der_bcoo
    n_complex = jnp.sqrt(-jnp.asarray(eigenvalues, dtype=jnp.complex128))
    neff = jnp.real(n_complex)
    keff = jnp.imag(n_complex)

    # Gauge. An eigenvector is defined only up to a complex factor, and the eigensolver picks one:
    # ARPACK returns ||v||_2 = 1 with an arbitrary phase, which is not a differentiable function of
    # the permittivity. Dividing by ||v||_2 here (and by the pivot phase below) makes everything
    # downstream invariant under v -> c v for any complex c, so the ambiguity cannot reach a
    # gradient. Numerically this is a division by 1 for an eigensolver that already normalises;
    # what it buys is that the *field* adjoint of jax_solve is then free to fix its own gauge.
    scale = jnp.linalg.norm(vecs, axis=0)
    vecs = vecs / jnp.where(scale > 0, scale, 1.0)[None, :]

    ex = vecs[:n, :]
    ey = vecs[n:, :]

    denom = (1j * neff - keff)[None, :]
    h_field = operator.qmat.matmul(vecs)
    hx = h_field[:n, :] / denom
    hy = h_field[n:, :] / denom
    hz = operator.inv_mu_zz[:, None] * ((dxf @ ey) - (dyf @ ex))

    # Ez = -inv_eps_zz * div^H (q_ep Exy) / (i neff); q_partial drops out of the divergence.
    # q_ep [ex; ey] = [D_y; -D_x], so no matmul is needed here.
    q_ep_top = operator.eps_yy[:, None] * ey
    q_ep_bottom = -operator.eps_xx[:, None] * ex
    if operator.eps_xy is not None and operator.eps_yx is not None:
        q_ep_top = q_ep_top + operator.eps_yx[:, None] * ex
        q_ep_bottom = q_ep_bottom - operator.eps_xy[:, None] * ey
    q_ep_top = q_ep_top / denom
    q_ep_bottom = q_ep_bottom / denom
    ez = operator.inv_eps_zz[:, None] * ((dxb @ q_ep_bottom) - (dyb @ q_ep_top))

    field_e = jnp.stack((ex, ey, ez), axis=0)
    field_h = jnp.stack((hx, hy, hz), axis=0) * (-1j / eta0)

    e_t = jnp.concatenate((field_e[0], field_e[1]), axis=0)
    pivot = e_t[jnp.argmax(jnp.abs(e_t), axis=0), jnp.arange(e_t.shape[1])]
    magnitude = jnp.abs(pivot)
    phase = jnp.where(magnitude > 0, pivot / jnp.where(magnitude > 0, magnitude, 1.0), 1.0 + 0.0j)
    field_e = field_e / phase[None, None, :]
    field_h = field_h / phase[None, None, :]

    if direction == "-":
        field_h = field_h.at[0].multiply(-1.0)
        field_h = field_h.at[1].multiply(-1.0)
        field_e = field_e.at[2].multiply(-1.0)

    return field_e, field_h, neff, keff
