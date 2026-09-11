"""The differentiable seam around the sparse mode eigen-solve.

Stage 2 of the mode pipeline is the one step that cannot be written in JAX: a shift-invert Arnoldi
iteration with a sparse LU (``scipy.sparse.linalg.eigs``). It stays in numpy, behind a single
``jax.pure_callback`` wrapped in one ``jax.custom_vjp``, and this module supplies the backward rule
so that a permittivity gradient crosses it.

The backward is the reciprocity contraction. For ``A v = lambda v`` with left eigenvector ``u``
(``u^T A = lambda u^T``) the exact eigenvalue sensitivity is

.. code-block:: text

    d lambda / d A_ij = u_i v_j / (u^T v)

and for a *diagonal* medium this operator has its left eigenvector in closed form: it is the
rotated, area-weighted transverse magnetic field the forward already computed (see
:func:`fdtdx.core.physics.mode_backend.jax_operator.left_eigenvectors_jax`). So the backward costs
one contraction over the operator's nonzeros, not a second eigen-solve, and ``u^T v`` is the modal
power - the same denominator :mod:`fdtdx.core.physics.mode_adjoint` uses in its field-level form of the
same identity.

Where the closed form stops holding, measured on a 14 x 12 strip as the relative residual
``||u^T A - lambda u^T|| / (||u|| |lambda|)``:

.. code-block:: text

    diagonal, uniform grid                              3.5e-12
    diagonal, graded grid                               4.0e-12
    symmetric transverse tensor, uniform grid           4.7e-12
    symmetric transverse tensor, graded grid            4.7e-02     <- fails
    non-reciprocal tensor (eps_xy != eps_yx)            2.1e-01     <- fails

The reason is the staggering: ``eps_xy`` couples ``Ey`` into the row that carries ``hx`` and
``Ex`` into the row that carries ``hy``, so transposing the operator swaps the two staggered cell
areas ``w_x`` and ``w_y``. They are equal on a uniform grid and not on a graded one. The general
answer is therefore to *solve* for the left eigenvectors: a second shift-invert Arnoldi run on
``A^T`` at the same shift, reusing the sparse LU factorisation of ``A - sigma I`` through its
transposed triangular solves, so the extra cost is the Arnoldi iteration and not a second
factorisation. :class:`EigenSolveSpec` selects between the two (``solve_left``), and
:func:`fdtdx.core.physics.mode_backend.jax_solve.left_eigenvector_residual` is the check that says
which one a given cross-section needs.

Degeneracy. At a crossing the individual eigenvectors of a degenerate block are defined only up to a
rotation within the invariant subspace, and the per-mode sensitivity above is not a function of the
permittivity at all - it depends on which basis the eigensolver happened to return. What *is* well
defined is the sum of the eigenvalues of the block, i.e. the trace of the operator restricted to the
subspace. This module therefore

1. detects near-degenerate blocks by relative eigenvalue gap (:data:`DEFAULT_DEGENERACY_RTOL`),
2. orthonormalises each block (modified Gram-Schmidt, Hermitian inner product) so the returned
   fields are an orthogonal basis rather than an arbitrary one, and
3. reports the **subspace-averaged** derivative for every member of the block:
   ``d lambda_m / dA = (1/|G|) d(sum_{k in G} lambda_k) / dA``, which is basis-independent.

Step 1 runs in the *backward*, on eigenvalues that are tracers whenever the caller applied
``jax.jit``, so it is written in array algebra rather than as a Python loop over values (see
:func:`degeneracy_mask`): masking the full Gram matrix to the block pattern makes it block diagonal,
and one linear solve then produces every block's ``(U_G^T V_G)^-1 U_G^T`` at once, with a
non-degenerate mode's 1x1 block reducing to the division by ``u^T v``. The whole seam therefore
traces under ``jit``, which is where an optimisation loop runs it.

The alternative - a symmetry-selected derivative, i.e. picking the basis vector that carries a
chosen symmetry and differentiating that one - needs a symmetry operator the solver does not have,
and is not attempted here.

What the derivative of a degenerate pair *means*, operationally: a perturbation that splits the
block makes each individual ``n_eff`` the root of a small problem whose branches separate linearly
in the step, so a per-mode finite difference does not converge to any per-mode derivative. The
average over the block does. Measured on a 12 x 12 square guide with a random per-cell direction:
the mean of the two branches' central differences matches this backward to 7e-10, while either
branch alone is off by 7e-5 at ``h = 1e-3`` and 7e-6 at ``h = 1e-4``.

Field gradients. The eigen*vector* cotangent is a bordered shifted solve, one per mode that carries
a non-zero cotangent. With ``u`` normalised so that ``u^T v = 1``,

.. code-block:: text

    [ A^T - lambda I   u ] [ y ]   [ ct_v - u (v^T ct_v) ]
    [ v^T              0 ] [ xi ] = [ 0                  ]      then   A_bar += -y v^T

The bordering is what makes the system non-singular at an exact eigenvalue, and it selects the one
solution with ``v^T y = 0`` - the group-inverse branch, which is the branch that matches the gauge
below. The eigenvalue and eigenvector contributions add.

The gauge. An eigenvector is defined up to a complex factor and ARPACK picks ``||v||_2 = 1`` with an
arbitrary phase, neither of which is a differentiable function of the permittivity. The fix is not in
this module: :func:`fdtdx.core.physics.mode_backend.jax_operator.reconstruct_fields_jax` divides by
``||v||_2`` and by the pivot phase, so everything downstream is invariant under ``v -> c v`` and the
gauge cannot reach a gradient. That is why the bordered solve may fix its own gauge freely.

A degenerate block has no per-mode eigenvector derivative at all - the individual vectors are an
arbitrary basis of the invariant subspace - so a field cotangent on a degenerate mode raises rather
than returning a basis-dependent number. Eigen*value* cotangents keep the subspace average.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from fdtdx.core.physics.mode_backend.jax_operator import (
    JaxModeOperator,
    SparseCOO,
    left_eigenvectors_jax,
    reconstruct_fields_jax,
    require_x64,
)

__all__ = [
    "DEFAULT_DEGENERACY_RTOL",
    "EigenSolveSpec",
    "degeneracy_mask",
    "degenerate_groups",
    "dense_mode_eigs",
    "orthonormalize_degenerate_blocks",
    "solve_modes_diagonal_jax",
    "sparse_mode_eigs",
]

#: Relative eigenvalue gap below which two modes are treated as degenerate. A square silicon
#: waveguide's two fundamental modes come out ~1e-15 apart on a symmetric grid; a physically
#: distinct neighbour is many orders of magnitude further away, so the threshold is not delicate.
DEFAULT_DEGENERACY_RTOL = 1e-8

#: Largest cross-section the dense oracle will accept, in cells. ``jnp.linalg.eig`` is O(n^3) on a
#: ``2N x 2N`` operator: 28 x 28 cells is a 1568 x 1568 matrix and about 6 s on one CPU.
MAX_DENSE_ORACLE_CELLS = 28 * 28


@dataclass(frozen=True, eq=False)
class EigenSolveSpec:
    """Everything about the eigen-solve that is not differentiated.

    Compared by identity (``eq=False``), so it can ride in ``jax.custom_vjp``'s ``nondiff_argnums``
    without the index arrays needing to be hashable.

    Attributes:
        mat_rows: Row indices of the operator's coordinate list.
        mat_cols: Column indices of the operator's coordinate list.
        qmat_rows: Row indices of ``qmat``'s coordinate list.
        qmat_cols: Column indices of ``qmat``'s coordinate list.
        left_weights: The two staggered cell-area vectors of the left eigenvector.
        num_cells: ``N``, the transverse cell count.
        num_modes: How many eigenpairs to return.
        sigma: Default shift-invert target, ``-(neff_guess)^2``, used when
            :func:`sparse_mode_eigs` is called without one. The shift is passed to the eigen-solve
            as a *runtime* scalar so that a caller may derive it from a traced permittivity, which
            is why it is not the only place it can come from.
        degeneracy_rtol: Relative gap below which eigenvalues form one degenerate block.
        solve_left: Solve for the left eigenvectors on ``A^T`` instead of using the closed form.
            Required for a tensorial cross-section on a non-uniform grid and for a non-reciprocal
            one; see the module docstring for the measured residuals.
        eigenvalue_kind: What the eigenvalue means, which is the only thing about the operator this
            module has to know. ``"neg_n2"`` is the transverse-E convention ``-(n_eff)^2``;
            ``"neff"`` is the four-component convention, the effective index itself, whose
            backward modes sit at negative real part. It decides the sort order and nothing else —
            the left-eigenvector solve, the degeneracy handling and the bordered field adjoint are
            all convention-free.
        backward: Sort for the backward (negative ``Re n_eff``) half of the spectrum. Only
            meaningful with ``eigenvalue_kind="neff"``, where the two halves are genuinely
            different modes.
    """

    mat_rows: np.ndarray
    mat_cols: np.ndarray
    qmat_rows: np.ndarray
    qmat_cols: np.ndarray
    left_weights: tuple[np.ndarray, np.ndarray]
    num_cells: int
    num_modes: int
    sigma: complex
    degeneracy_rtol: float = DEFAULT_DEGENERACY_RTOL
    solve_left: bool = False
    eigenvalue_kind: str = "neg_n2"
    backward: bool = False

    def sort_order(self, vals: np.ndarray) -> np.ndarray:
        """Indices that sort the eigenvalues by descending physical ``Re(n_eff)``.

        Args:
            vals (np.ndarray): The eigenvalues as the solver returned them.

        Returns:
            np.ndarray: The permutation.
        """
        if self.eigenvalue_kind == "neff":
            key = -np.real(vals) if self.backward else np.real(vals)
        else:
            key = np.real(np.emath.sqrt(-np.asarray(vals) + 0j))
        return np.argsort(key)[::-1]


def degenerate_groups(eigenvalues: np.ndarray, rtol: float = DEFAULT_DEGENERACY_RTOL) -> list[list[int]]:
    """Group eigenvalues that are degenerate to within a relative gap.

    Args:
        eigenvalues (np.ndarray): The eigenvalues, in the order they are returned.
        rtol (float): Two eigenvalues join the same block when ``|a - b| <= rtol * max(|a|, |b|)``.

    Returns:
        list[list[int]]: One list of indices per block, blocks in order of first appearance.
    """
    vals = np.asarray(eigenvalues)
    n = vals.size
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            scale = max(abs(vals[i]), abs(vals[j]), np.finfo(float).tiny)
            if abs(vals[i] - vals[j]) <= rtol * scale:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in sorted(groups.values(), key=min)]


def degeneracy_mask(eigenvalues: jax.Array, rtol: float = DEFAULT_DEGENERACY_RTOL) -> jax.Array:
    """The same grouping as :func:`degenerate_groups`, as a traceable ``(M, M)`` indicator.

    The backward runs inside ``jax.custom_vjp`` and therefore inside whatever transformation the
    caller applied: under ``jax.jit`` the eigenvalues are tracers and cannot be grouped by a Python
    loop. This is the grouping written in array operations instead - the pairwise relative-gap test,
    then its transitive closure by boolean squaring (``ceil(log2 M)`` rounds, which is exact because
    the number of modes is static). ``mask[i, j]`` is 1.0 when modes ``i`` and ``j`` share a block.

    Args:
        eigenvalues (jax.Array): The eigenvalues, shape ``(M,)``. May be traced.
        rtol (float): Two eigenvalues share a block when ``|a - b| <= rtol * max(|a|, |b|)``.

    Returns:
        jax.Array: Real ``(M, M)`` block-membership indicator, symmetric with a unit diagonal.
    """
    magnitude = jnp.abs(eigenvalues)
    scale = jnp.maximum(jnp.maximum(magnitude[:, None], magnitude[None, :]), jnp.finfo(jnp.float64).tiny)
    mask = jnp.abs(eigenvalues[:, None] - eigenvalues[None, :]) <= rtol * scale
    rounds = int(np.ceil(np.log2(max(int(eigenvalues.shape[0]), 2))))
    for _ in range(rounds):
        mask = (mask.astype(jnp.float64) @ mask.astype(jnp.float64)) > 0
    return mask.astype(jnp.float64)


def orthonormalize_degenerate_blocks(
    vecs: np.ndarray,
    eigenvalues: np.ndarray,
    rtol: float = DEFAULT_DEGENERACY_RTOL,
) -> np.ndarray:
    """Orthonormalise the eigenvectors inside every degenerate block.

    An eigensolver returns an arbitrary basis of a degenerate invariant subspace, and the two
    vectors it returns need not even be orthogonal. Modified Gram-Schmidt in the Hermitian inner
    product turns them into an orthonormal basis of the same subspace, which is what makes the
    returned mode fields usable (an overlap integral against a non-orthogonal pair double-counts).
    It does not make the *individual* mode physical - only the subspace is - which is why the
    gradient is averaged over the block.

    Args:
        vecs (np.ndarray): Eigenvectors as columns, shape ``(2N, M)``.
        eigenvalues (np.ndarray): The matching eigenvalues, shape ``(M,)``.
        rtol (float): Relative gap defining a block.

    Returns:
        np.ndarray: The eigenvectors with each degenerate block orthonormalised; non-degenerate
        columns are returned untouched.
    """
    out = np.array(vecs, dtype=np.complex128, copy=True)
    for group in degenerate_groups(eigenvalues, rtol):
        if len(group) < 2:
            continue
        basis: list[np.ndarray] = []
        for idx in group:
            v = out[:, idx].copy()
            for b in basis:
                v = v - b * np.vdot(b, v)
            norm = np.linalg.norm(v)
            if norm > 1e-12:
                v = v / norm
            basis.append(v)
            out[:, idx] = v
    return out


def _operator_size(spec: EigenSolveSpec) -> int:
    """Side length of the operator: ``2N`` for the transverse tier, ``4N`` for the full tensor."""
    return (4 if spec.eigenvalue_kind == "neff" else 2) * spec.num_cells


def _sparse_operator(spec: EigenSolveSpec, data: np.ndarray) -> sp.csr_matrix:
    """Materialise the coordinate-list operator, summing duplicate entries."""
    size = _operator_size(spec)
    return sp.coo_matrix(
        (np.asarray(data, dtype=np.complex128), (spec.mat_rows, spec.mat_cols)),
        shape=(size, size),
    ).tocsr()


class _LuSolveOperator(spl.LinearOperator):
    """``x -> lu.solve(x)`` (or the transposed solve) as the operator ARPACK's shift-invert wants."""

    def __init__(self, lu: Any, size: int, trans: str = "N") -> None:
        super().__init__(dtype=np.complex128, shape=(size, size))
        self._lu = lu
        self._trans = trans

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self._lu.solve(x, trans=self._trans)


def _arpack_eigs(spec: EigenSolveSpec, want_left: bool, data: np.ndarray, sigma: np.ndarray):
    """Run the shift-invert Arnoldi solve on the coordinate-list operator (numpy, no JAX).

    One sparse LU of ``A - sigma I`` serves both runs: the right eigenpairs come from its forward
    triangular solves and the left ones from the transposed solves of the same factors, so asking
    for the left eigenvectors costs a second Arnoldi iteration and no second factorisation.

    Args:
        spec (EigenSolveSpec): The static part of the problem.
        want_left (bool): Also solve ``A^T`` and return the matched left eigenvectors.
        data (np.ndarray): The operator's nonzero values, duplicates allowed.
        sigma (np.ndarray): The shift-invert target, ``-(neff_guess)^2``, as a runtime scalar so a
            caller may derive it from a traced permittivity.

    Returns:
        ``(eigenvalues, eigenvectors)``, or ``(eigenvalues, eigenvectors, left)`` when
        ``want_left``; sorted by descending ``Re(n_eff)`` with degenerate blocks orthonormalised.
    """
    size = _operator_size(spec)
    mat = _sparse_operator(spec, data)
    rng = np.random.default_rng(0)
    vec_init = rng.random(size) + 1j * rng.random(size)
    num_modes = min(spec.num_modes, size - 2)
    shift = complex(np.asarray(sigma).reshape(-1)[0])
    lu = spl.splu((mat - sp.diags(np.full(size, shift, dtype=np.complex128), format="csr")).tocsc())
    op_inv = _LuSolveOperator(lu, size)
    vals, vecs = spl.eigs(mat, k=num_modes, sigma=shift, v0=vec_init, OPinv=op_inv)
    order = spec.sort_order(vals)
    vals = np.ascontiguousarray(vals[order])
    vecs = orthonormalize_degenerate_blocks(vecs[:, order], vals, spec.degeneracy_rtol)
    if not want_left:
        return vals.astype(np.complex128), vecs.astype(np.complex128)

    op_inv_t = _LuSolveOperator(lu, size, trans="T")
    vals_left, vecs_left = spl.eigs(mat.T.tocsr(), k=num_modes, sigma=shift, v0=vec_init, OPinv=op_inv_t)
    # The two spectra are the same set; pair them off by nearest eigenvalue. Inside a degenerate
    # block the pairing is arbitrary and harmless: the backward only needs the left vectors to span
    # the block's left invariant subspace, which the masked Gram solve then re-bases.
    taken = np.zeros(vals_left.size, dtype=bool)
    matched = np.empty((size, vals.size), dtype=np.complex128)
    for i, value in enumerate(vals):
        distance = np.where(taken, np.inf, np.abs(vals_left - value))
        j = int(np.argmin(distance))
        taken[j] = True
        matched[:, i] = vecs_left[:, j]
    return vals.astype(np.complex128), vecs.astype(np.complex128), matched


def _callback_eigs(spec: EigenSolveSpec, mat_data: jax.Array, sigma: jax.Array, want_left: bool):
    """``_arpack_eigs`` behind ``jax.pure_callback``, with the right declared shapes."""
    size = _operator_size(spec)
    num_modes = min(spec.num_modes, size - 2)
    shapes: tuple[jax.ShapeDtypeStruct, ...] = (
        jax.ShapeDtypeStruct((num_modes,), jnp.complex128),
        jax.ShapeDtypeStruct((size, num_modes), jnp.complex128),
    )
    if want_left:
        shapes = (*shapes, jax.ShapeDtypeStruct((size, num_modes), jnp.complex128))
    return jax.pure_callback(partial(_arpack_eigs, spec, want_left), shapes, mat_data, sigma)


def _field_adjoint(
    spec: EigenSolveSpec,
    mat_data: np.ndarray,
    vals: np.ndarray,
    vecs: np.ndarray,
    left: np.ndarray,
    ct_vecs: np.ndarray,
) -> np.ndarray:
    """The eigenvector cotangent's contribution to the operator's nonzeros (numpy, no JAX).

    One bordered shifted solve per mode whose cotangent is not identically zero:

    .. code-block:: text

        [ A^T - lambda I   u ] [ y ]   [ ct_v - u (v^T ct_v) ]
        [ v^T              0 ] [ xi ] = [ 0                  ]    ,  A_bar += -y v^T

    with ``u`` scaled so ``u^T v = 1``. When every cotangent is zero the whole routine is skipped,
    which is what keeps an ``n_eff``-only gradient at the cost the module docstring quotes.

    Args:
        spec (EigenSolveSpec): The static part of the problem.
        mat_data (np.ndarray): The operator's nonzero values.
        vals (np.ndarray): Eigenvalues, shape ``(M,)``.
        vecs (np.ndarray): Right eigenvectors, shape ``(2N, M)``.
        left (np.ndarray): Left eigenvectors, shape ``(2N, M)``.
        ct_vecs (np.ndarray): Eigenvector cotangents, shape ``(2N, M)``.

    Returns:
        np.ndarray: The contribution to the cotangent of ``mat_data``, shape ``(nnz,)``.

    Raises:
        NotImplementedError: If a mode carrying a non-zero cotangent sits in a degenerate block.
    """
    nnz = spec.mat_rows.size
    out = np.zeros(nnz, dtype=np.complex128)
    ct_vecs = np.asarray(ct_vecs, dtype=np.complex128)
    active = [m for m in range(ct_vecs.shape[1]) if np.any(ct_vecs[:, m] != 0.0)]
    if not active:
        return out
    groups = degenerate_groups(np.asarray(vals), spec.degeneracy_rtol)
    block_of = {index: group for group in groups for index in group}
    size = _operator_size(spec)
    mat_t = _sparse_operator(spec, mat_data).T.tocsr()
    for m in active:
        if len(block_of[m]) > 1:
            raise NotImplementedError(
                f"mode {m} shares a degenerate block with modes {block_of[m]}, so its individual "
                "eigenvector is an arbitrary basis of the invariant subspace and has no derivative. "
                "Differentiate the effective index (which carries the subspace average), break the "
                "degeneracy, or hold the fields with jax.lax.stop_gradient."
            )
        v = vecs[:, m]
        u = left[:, m]
        u = u / (u @ v)
        rhs = ct_vecs[:, m] - u * (v @ ct_vecs[:, m])
        shifted = mat_t - sp.diags(np.full(size, complex(vals[m]), dtype=np.complex128), format="csr")
        bordered = sp.bmat([[shifted, u.reshape(-1, 1)], [v.reshape(1, -1), None]], format="csc")
        solution = spl.spsolve(bordered, np.concatenate((rhs, [0.0 + 0.0j])))
        y = solution[:size]
        out -= y[spec.mat_rows] * v[spec.mat_cols]
    return out


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _sparse_mode_eigs(
    spec: EigenSolveSpec,
    mat_data: jax.Array,
    qmat_data: jax.Array,
    sigma: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Eigenpairs of the sparse mode operator, differentiable in the operator's values.

    The forward is ARPACK behind ``jax.pure_callback``. The backward is the reciprocity contraction
    described in the module docstring, with subspace averaging over degenerate blocks.

    ``qmat_data`` is passed in only so the backward can build the left eigenvector from the same
    solve; its own cotangent is zero, because the eigenvalues do not depend on ``qmat`` except
    through ``mat``.

    The cotangent of the returned **eigenvectors** is a bordered shifted solve per mode, skipped
    entirely when every cotangent is zero (so an ``n_eff``-only gradient pays nothing for it). It is
    refused for a mode inside a degenerate block, where the individual eigenvector is an arbitrary
    basis of the invariant subspace. The refusal is raised inside the numpy callback, so it fires
    under ``jax.jit`` as well as eagerly.

    Args:
        spec (EigenSolveSpec): The static part of the problem.
        mat_data (jax.Array): The operator's nonzero values.
        qmat_data (jax.Array): ``qmat``'s nonzero values.
        sigma (jax.Array): The shift-invert target as a runtime scalar, so a caller may derive it
            from a traced permittivity. It selects which modes come back and is not differentiated.

    Returns:
        tuple[jax.Array, jax.Array]: ``(eigenvalues, eigenvectors)``, shapes ``(M,)`` and ``(2N, M)``.
    """
    del qmat_data
    return _callback_eigs(spec, mat_data, sigma, want_left=False)


def _closed_form_left(spec: EigenSolveSpec, vecs: jax.Array, qmat_data: jax.Array) -> jax.Array:
    """``u = [w_x hy; -w_y hx]`` from the transverse magnetic field the forward already produced."""
    n = spec.num_cells
    size = 2 * n
    qmat = SparseCOO(rows=spec.qmat_rows, cols=spec.qmat_cols, data=qmat_data, shape=(size, size))
    h = qmat.matmul(vecs)
    w_x = jnp.asarray(spec.left_weights[0], dtype=jnp.complex128)[:, None]
    w_y = jnp.asarray(spec.left_weights[1], dtype=jnp.complex128)[:, None]
    return jnp.concatenate((w_x * h[n:, :], -w_y * h[:n, :]), axis=0)


def _sparse_mode_eigs_fwd(spec, mat_data, qmat_data, sigma):
    if spec.solve_left:
        vals, vecs, left = _callback_eigs(spec, mat_data, sigma, want_left=True)
    else:
        vals, vecs = _callback_eigs(spec, mat_data, sigma, want_left=False)
        left = _closed_form_left(spec, vecs, qmat_data)
    return (vals, vecs), (vals, vecs, left, qmat_data, mat_data, sigma)


def _sparse_mode_eigs_bwd(spec, residuals, cotangents):
    vals, vecs, left, qmat_data, mat_data, sigma = residuals
    ct_vals, ct_vecs = cotangents
    rows = jnp.asarray(spec.mat_rows)
    cols = jnp.asarray(spec.mat_cols)
    mask = degeneracy_mask(vals, spec.degeneracy_rtol)
    counts = jnp.sum(mask, axis=1)
    # Masking the full Gram matrix to the block pattern makes it block diagonal, so one solve is
    # every block's (U_G^T V_G)^-1 U_G^T at once - and a non-degenerate mode's 1x1 block is just
    # the division by u^T v.
    gram = (left.T @ vecs) * mask
    y = jnp.linalg.solve(gram, left.T)
    # Each member of a block carries the block's *averaged* eigenvalue derivative, so a cotangent on
    # any member spreads evenly over the block. Off a block this is the identity.
    weights = (mask / counts[:, None]).astype(ct_vals.dtype) @ ct_vals
    data_bar = jnp.sum(vecs[cols, :] * (weights[None, :] * y[:, rows].T), axis=1)
    field_bar = jax.pure_callback(
        partial(_field_adjoint, spec),
        jax.ShapeDtypeStruct(mat_data.shape, jnp.complex128),
        mat_data,
        vals,
        vecs,
        left,
        ct_vecs,
    )
    return data_bar + field_bar, jnp.zeros_like(qmat_data), jnp.zeros_like(sigma)


_sparse_mode_eigs.defvjp(_sparse_mode_eigs_fwd, _sparse_mode_eigs_bwd)


def sparse_mode_eigs(
    spec: EigenSolveSpec,
    mat_data: jax.Array,
    qmat_data: jax.Array,
    sigma: jax.Array | complex | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Eigenpairs of the sparse mode operator, differentiable in the operator's values.

    Thin wrapper over the ``custom_vjp``: it materialises the shift, which may be omitted (then
    ``spec.sigma`` is used) or handed in as a traced scalar derived from the permittivity.

    Args:
        spec (EigenSolveSpec): The static part of the problem.
        mat_data (jax.Array): The operator's nonzero values.
        qmat_data (jax.Array): ``qmat``'s nonzero values.
        sigma (jax.Array | complex | None): Shift-invert target; ``None`` uses ``spec.sigma``.

    Returns:
        tuple[jax.Array, jax.Array]: ``(eigenvalues, eigenvectors)``.
    """
    if sigma is None:
        sigma = spec.sigma
    shift = jax.lax.stop_gradient(jnp.asarray(sigma, dtype=jnp.complex128).reshape(()))
    return _sparse_mode_eigs(spec, mat_data, qmat_data, shift)


def dense_mode_eigs(
    operator: JaxModeOperator,
    max_cells: int = MAX_DENSE_ORACLE_CELLS,
) -> tuple[jax.Array, jax.Array]:
    """Dense ``jnp.linalg.eig`` reference for the sparse path. **Test oracle only.**

    JAX gained non-Hermitian eigenvector derivatives in April 2026, behind the explicit
    ``enable_eigvec_derivs=True`` flag and on CPU/GPU only. That makes a *fully* differentiable
    reference available for small cross-sections, against which the sparse path's eigenvalues,
    eigenvectors and gradients can be checked without any hand-written adjoint in the way. It is
    ``O(n^3)`` on a ``2N x 2N`` matrix, so it never leaves the tests.

    Args:
        operator (JaxModeOperator): The assembled operator.
        max_cells (int): Refuse cross-sections larger than this many cells.

    Returns:
        tuple[jax.Array, jax.Array]: ``(eigenvalues, eigenvectors)`` sorted by descending
        ``Re(n_eff)``, eigenvectors as columns.

    Raises:
        ValueError: If the cross-section is larger than ``max_cells``.
    """
    require_x64("dense_mode_eigs")
    if operator.num_cells > max_cells:
        raise ValueError(
            f"dense_mode_eigs is an O(n^3) test oracle and refuses a {operator.num_cells}-cell "
            f"cross-section (limit {max_cells}, about 28 x 28). Use sparse_mode_eigs."
        )
    dense = operator.mat.to_bcoo().todense()
    vals, vecs = jax.lax.linalg.eig(
        dense,
        compute_left_eigenvectors=False,
        compute_right_eigenvectors=True,
        enable_eigvec_derivs=True,
    )
    order = jnp.argsort(-jnp.real(jnp.sqrt(-vals)))
    return vals[order], vecs[:, order]


def solve_modes_diagonal_jax(
    eps_xx: jax.Array,
    eps_yy: jax.Array,
    eps_zz: jax.Array,
    mu_xx: jax.Array,
    mu_yy: jax.Array,
    mu_zz: jax.Array,
    der_mats: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    cell_steps: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    k0: float,
    num_modes: int,
    neff_guess: float | jax.Array,
    direction: str = "+",
    degeneracy_rtol: float = DEFAULT_DEGENERACY_RTOL,
    dmin_pmc: tuple[bool, bool] = (False, False),
    eps_xy: jax.Array | None = None,
    eps_yx: jax.Array | None = None,
    solve_left: bool | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The whole diagonal-media mode solve, with the permittivity gradient connected end to end.

    Assembly and reconstruction run in JAX; the eigen-solve is ARPACK behind the ``custom_vjp`` of
    :func:`sparse_mode_eigs`. The signature mirrors
    :func:`fdtdx.core.physics.mode_backend.solve.solve_modes_diagonal` and returns the same four
    arrays, so the two can be compared directly. This is the differentiable path; the forward path
    the mode front end uses is still the numpy one, which runs inside ``jax.pure_callback`` and is
    unchanged.

    Args:
        eps_xx (jax.Array): Flattened ``eps_xx`` at the Yee Ex locations, length ``N``.
        eps_yy (jax.Array): Flattened ``eps_yy``.
        eps_zz (jax.Array): Flattened ``eps_zz``.
        mu_xx (jax.Array): Flattened ``mu_xx``.
        mu_yy (jax.Array): Flattened ``mu_yy``.
        mu_zz (jax.Array): Flattened ``mu_zz``.
        der_mats: ``(dxf, dxb, dyf, dyb)`` SI difference matrices.
        cell_steps: ``((dlf_x, dlb_x), (dlf_y, dlb_y))`` primal/dual steps of the two axes.
        k0 (float): Free-space wavenumber ``2 pi f / c`` (1/m).
        num_modes (int): Number of modes to return.
        neff_guess (float | jax.Array): Shift-invert target effective index. May be traced: the
            shift reaches the eigen-solve as a runtime scalar, not as part of the static spec.
        direction (str): ``"+"`` or ``"-"``.
        degeneracy_rtol (float): Relative gap defining a degenerate block.
        dmin_pmc (tuple[bool, bool]): The per-axis min-edge wall types the difference matrices were
            built with, needed here only to refuse the case the adjoint cannot serve.
        eps_xy (jax.Array | None): Flattened ``eps_xy``, or ``None`` for no transverse off-diagonal.
        eps_yx (jax.Array | None): Flattened ``eps_yx``, or ``None``.
        solve_left (bool | None): Solve for the left eigenvectors rather than using the closed form.
            ``None`` decides on the tier: closed form for a diagonal medium, solved as soon as a
            transverse off-diagonal is present, which is where the closed form stops holding on a
            graded grid (module docstring). Pass ``False`` to force the free adjoint on a tensorial
            uniform grid, where it is exact.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array, jax.Array]: ``(E, H, neff, keff)`` with ``E`` and
        ``H`` of shape ``(3, N, M)``.

    Raises:
        NotImplementedError: If a min edge is PMC. The closed-form left eigenvector is exact for
            electric walls only - a magnetic wall carries a ``2`` on the diagonal of the backward
            difference with no entry to transpose onto, and the relative residual of ``u^T A -
            lambda u^T`` goes from 3e-12 to order 1. The forward is unaffected; only the gradient is.
    """
    from fdtdx.core.physics.mode_backend.jax_operator import assemble_mode_operator_jax

    if any(dmin_pmc):
        raise NotImplementedError(
            "the differentiable mode path supports electric (PEC) min-edge walls only. Its backward "
            "uses the closed-form left eigenvector u = [w_x hy; -w_y hx], which is exact for PEC and "
            "not for PMC (measured relative residual 3e-12 vs order 1). Use the numpy forward path "
            "for a PMC-symmetric solve, or drop the symmetry and solve the full cross-section."
        )

    operator = assemble_mode_operator_jax(
        eps_xx, eps_yy, eps_zz, mu_xx, mu_yy, mu_zz, der_mats, k0, cell_steps, eps_xy=eps_xy, eps_yx=eps_yx
    )
    if solve_left is None:
        solve_left = eps_xy is not None
    spec = EigenSolveSpec(
        mat_rows=operator.mat.rows,
        mat_cols=operator.mat.cols,
        qmat_rows=operator.qmat.rows,
        qmat_cols=operator.qmat.cols,
        left_weights=operator.left_weights,
        num_cells=operator.num_cells,
        num_modes=num_modes,
        sigma=0j,
        degeneracy_rtol=degeneracy_rtol,
        solve_left=bool(solve_left),
    )
    sigma = -(jnp.asarray(neff_guess, dtype=jnp.complex128) ** 2)
    vals, vecs = sparse_mode_eigs(spec, operator.mat.data, operator.qmat.data, sigma)
    return reconstruct_fields_jax(operator, vecs, vals, direction=direction)


def left_eigenvector_residual(
    operator: JaxModeOperator,
    vecs: jax.Array,
    eigenvalues: jax.Array,
    left: jax.Array | None = None,
) -> jax.Array:
    """Relative residual of the left eigenvectors, per mode.

    A cheap self-check on the reciprocity relation the free backward depends on: ~1e-12 for a
    diagonal medium on either grid and for a symmetric tensor on a uniform grid, ~5e-2 for a
    symmetric tensor on a graded grid, ~2e-1 for a non-reciprocal tensor and for a PMC min edge. So
    a caller can tell whether ``solve_left=False`` is safe on its own cross-section.

    Args:
        operator (JaxModeOperator): The assembled operator.
        vecs (jax.Array): Right eigenvectors, shape ``(2N, M)``.
        eigenvalues (jax.Array): The matching eigenvalues, shape ``(M,)``.
        left (jax.Array | None): Left eigenvectors to test; ``None`` uses the closed form.

    Returns:
        jax.Array: ``||u^T A - lambda u^T|| / (||u|| |lambda|)`` per mode, shape ``(M,)``.
    """
    left = left_eigenvectors_jax(operator, vecs) if left is None else left
    # ``matmul`` computes ``A @ x``; ``u^T A`` is ``(A^T u)^T``, so swap the index arrays.
    transposed = SparseCOO(
        rows=operator.mat.cols, cols=operator.mat.rows, data=operator.mat.data, shape=operator.mat.shape
    )
    lhs = transposed.matmul(left)
    residual = lhs - jnp.asarray(eigenvalues)[None, :] * left
    return jnp.linalg.norm(residual, axis=0) / (jnp.linalg.norm(left, axis=0) * jnp.abs(jnp.asarray(eigenvalues)))
