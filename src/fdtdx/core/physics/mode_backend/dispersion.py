"""Group index and chromatic dispersion from one solve, by differentiating the operator in omega.

The group index is ``n_g = n_eff + omega d n_eff / d omega``. The usual way to get it is three
solves and a finite difference in frequency (Tidy3D's ``group_index_step``), which costs three
eigen-solves, loses digits to the step, and needs the three solves to be the same mode. It is not necessary: the mode operator's frequency
dependence is **explicit and exact**, and the derivative is one contraction on the backward of the
eigen-solve that has already run.

Why it is exact. Every entry of the transverse-E operator is either a material product with no
derivative in it, or a product of exactly two ``k0``-normalised difference matrices. So

.. code-block:: text

    A(k0) = D + S / k0**2,      D = diag(-mu_yy eps_xx, -mu_xx eps_yy)

with ``D`` frequency-independent and ``S`` built from the *unnormalised* difference matrices. One
assembly at a reference ``k0`` therefore gives the operator at every frequency,

.. code-block:: text

    A(k0) = s A(k0_ref) + (1 - s) D,      s = (k0_ref / k0)**2

which is exact, not a Taylor step - and it is a traceable function of ``k0``, so
:func:`jax.grad` through the ``custom_vjp`` of
:mod:`fdtdx.core.physics.mode_backend.jax_solve` produces ``d lambda / d k0`` from one contraction
against ``dA/dk0``. The eigen-solve runs **once**.

The same algebra also gives the answer in closed form, which the tests use as a second, independent
statement of the same thing: with the left eigenvector ``u`` and the right one ``v``,

.. code-block:: text

    d lambda / d k0 = -(2 / k0) (lambda - u^T D v / u^T v)   and    n_g = - (u^T D v / u^T v) / n_eff

so the group index is a single quadratic form in the mode's own fields - the discrete version of the
energy-over-power ratio.

Scope. The material is taken as it is handed in, i.e. non-dispersive at the solve frequency. A
dispersive medium adds ``sum_c (d n_eff / d eps_c) (d eps_c / d omega)``; the same backward already
produces the first factor per cell, so a caller with a material model can chain it in JAX and add
the term. Nothing here assumes a straight guide: a bent cross-section transformed by
:mod:`fdtdx.core.physics.mode_backend.bend` has a frequency-independent scale factor, so its group
index comes out of exactly the same contraction.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from fdtdx.constants import c
from fdtdx.core.physics.mode_backend.jax_operator import (
    JaxModeOperator,
    SparseCOO,
    assemble_mode_operator_jax,
    require_x64,
)
from fdtdx.core.physics.mode_backend.jax_solve import (
    DEFAULT_DEGENERACY_RTOL,
    EigenSolveSpec,
    sparse_mode_eigs,
)

__all__ = [
    "ModeDispersion",
    "frequency_scaled_operator_data",
    "group_index_closed_form",
    "mode_dispersion",
    "operator_diagonal_part",
]


class ModeDispersion(NamedTuple):
    """One mode's index and its frequency derivative.

    Attributes:
        neff: Complex effective index ``n_eff + i k_eff`` of the selected mode. For a bent guide it
            is the azimuthal index at the reference radius, ``m / (k0 R)``.
        group_index: ``n_eff + omega d n_eff / d omega``, complex. Its real part is the group index
            proper; the imaginary part is the frequency derivative of the loss.
        dneff_domega: ``d n_eff / d omega`` in s/rad, the quantity the two above are built from.
        mode_index: Position of the selected mode in the sorted list, as an integer array.
    """

    neff: jax.Array
    group_index: jax.Array
    dneff_domega: jax.Array
    mode_index: jax.Array


def operator_diagonal_part(operator: JaxModeOperator, mu_xx: jax.Array, mu_yy: jax.Array) -> jax.Array:
    """The frequency-independent diagonal ``D`` of the operator, as a flat ``2N`` vector.

    Args:
        operator (JaxModeOperator): The assembled operator.
        mu_xx (jax.Array): Flattened ``mu_xx``, length ``N``.
        mu_yy (jax.Array): Flattened ``mu_yy``.

    Returns:
        jax.Array: ``[-mu_yy eps_xx ; -mu_xx eps_yy]``, length ``2N``.
    """
    mu_xx = jnp.asarray(mu_xx, dtype=jnp.complex128)
    mu_yy = jnp.asarray(mu_yy, dtype=jnp.complex128)
    return jnp.concatenate((-mu_yy * operator.eps_xx, -mu_xx * operator.eps_yy))


def frequency_scaled_operator_data(
    operator: JaxModeOperator,
    diagonal: jax.Array,
    k0_ref: float,
    k0: jax.Array | float,
) -> tuple[np.ndarray, np.ndarray, jax.Array]:
    """The operator at ``k0``, from one assembly at ``k0_ref``, as a coordinate list.

    ``A(k0) = s A(k0_ref) + (1 - s) D`` with ``s = (k0_ref / k0)**2``. The correction is diagonal, so
    it is appended as ``2N`` extra coordinate entries rather than merged - duplicates are summed by
    both consumers of the list.

    Args:
        operator (JaxModeOperator): The operator assembled at ``k0_ref``.
        diagonal (jax.Array): ``D`` from :func:`operator_diagonal_part`.
        k0_ref (float): The wavenumber the operator was assembled at.
        k0 (jax.Array | float): The wavenumber wanted; may be traced, and may be complex (that is
            what makes a holomorphic derivative in frequency possible).

    Returns:
        tuple[np.ndarray, np.ndarray, jax.Array]: ``(rows, cols, data)`` of the augmented list.
    """
    size = 2 * operator.num_cells
    index = np.arange(size, dtype=np.int64)
    scale = (float(k0_ref) / jnp.asarray(k0, dtype=jnp.complex128)) ** 2
    data = jnp.concatenate((operator.mat.data * scale, diagonal * (1.0 - scale)))
    rows = np.concatenate((operator.mat.rows, index))
    cols = np.concatenate((operator.mat.cols, index))
    return rows, cols, data


def _polarization_penalty(vecs: jax.Array, num_cells: int, filter_pol: Literal["te", "tm"] | None) -> jax.Array:
    """A large sort key for modes of the wrong polarization, mirroring ``sort_modes``' rule.

    ``"te"`` keeps the modes whose transverse electric energy is mostly on the *first* transverse
    axis, ``"tm"`` those on the second - the same convention as
    :func:`fdtdx.core.physics.modes.compute_mode_polarization_fraction`.
    """
    if filter_pol is None:
        return jnp.zeros(vecs.shape[1])
    energy_x = jnp.sum(jnp.abs(vecs[:num_cells, :]) ** 2, axis=0)
    energy_y = jnp.sum(jnp.abs(vecs[num_cells:, :]) ** 2, axis=0)
    fraction = (energy_x if filter_pol == "te" else energy_y) / (energy_x + energy_y + 1e-18)
    return jnp.where(fraction >= 0.5, 0.0, 1e6)


def _select(
    vals: jax.Array,
    vecs: jax.Array,
    num_cells: int,
    mode_index: int,
    target_neff: float | None,
    filter_pol: Literal["te", "tm"] | None,
) -> jax.Array:
    """Position in the sorted list of the mode the caller asked for."""
    neff = jnp.real(jnp.sqrt(-vals))
    key = jnp.abs(neff - float(target_neff)) if target_neff is not None else -neff
    order = jnp.argsort(key + _polarization_penalty(vecs, num_cells, filter_pol))
    return order[mode_index]


def group_index_closed_form(
    operator: JaxModeOperator,
    diagonal: jax.Array,
    vals: jax.Array,
    vecs: jax.Array,
    left: jax.Array,
) -> jax.Array:
    """``n_g`` of every mode without any differentiation, from the same quadratic form.

    Written out, ``n_g = -(u^T D v) / ((u^T v) n_eff)``. It is the closed form of what
    :func:`mode_dispersion` obtains by ``jax.grad``, and the two agreeing is the check that the
    frequency dependence assumed by the scaling is the operator's actual one.

    Args:
        operator (JaxModeOperator): The assembled operator (used only for its shapes).
        diagonal (jax.Array): ``D`` from :func:`operator_diagonal_part`.
        vals (jax.Array): Eigenvalues, shape ``(M,)``.
        vecs (jax.Array): Right eigenvectors, shape ``(2N, M)``.
        left (jax.Array): Left eigenvectors, shape ``(2N, M)``.

    Returns:
        jax.Array: Complex group index per mode, shape ``(M,)``.
    """
    del operator
    numerator = jnp.sum(left * diagonal[:, None] * vecs, axis=0)
    denominator = jnp.sum(left * vecs, axis=0)
    return -(numerator / denominator) / jnp.sqrt(-vals)


def mode_dispersion(
    eps_xx: jax.Array,
    eps_yy: jax.Array,
    eps_zz: jax.Array,
    mu_xx: jax.Array,
    mu_yy: jax.Array,
    mu_zz: jax.Array,
    der_mats: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    cell_steps: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    frequency: float,
    num_modes: int,
    neff_guess: float,
    mode_index: int = 0,
    target_neff: float | None = None,
    filter_pol: Literal["te", "tm"] | None = None,
    degeneracy_rtol: float = DEFAULT_DEGENERACY_RTOL,
    dmin_pmc: tuple[bool, bool] = (False, False),
) -> ModeDispersion:
    """Effective index and group index of one mode, from a single eigen-solve.

    Args:
        eps_xx (jax.Array): Flattened ``eps_xx`` at the Yee Ex locations, length ``N``.
        eps_yy (jax.Array): Flattened ``eps_yy``.
        eps_zz (jax.Array): Flattened ``eps_zz``.
        mu_xx (jax.Array): Flattened ``mu_xx``.
        mu_yy (jax.Array): Flattened ``mu_yy``.
        mu_zz (jax.Array): Flattened ``mu_zz``.
        der_mats: ``(dxf, dxb, dyf, dyb)`` SI difference matrices.
        cell_steps: ``((dlf_x, dlb_x), (dlf_y, dlb_y))`` primal/dual steps of the two axes.
        frequency (float): Operating frequency in Hz.
        num_modes (int): How many eigenpairs to solve for before selecting.
        neff_guess (float): Shift-invert target effective index.
        mode_index (int): Position in the sorted list to take.
        target_neff (float | None): When given, sort by distance from it instead of by descending
            ``Re(n_eff)``.
        filter_pol (Literal["te", "tm"] | None): Optional polarization filter, same rule as
            :func:`fdtdx.core.physics.modes.sort_modes`.
        degeneracy_rtol (float): Relative gap defining a degenerate block.
        dmin_pmc (tuple[bool, bool]): Min-edge wall types; a magnetic wall is refused, because the
            backward's closed-form left eigenvector does not hold there.

    Returns:
        ModeDispersion: The selected mode's index, group index and ``d n_eff / d omega``.

    Raises:
        NotImplementedError: If a min edge is PMC.
        ValueError: If ``jax_enable_x64`` is off.
    """
    require_x64("mode_dispersion")
    if any(dmin_pmc):
        raise NotImplementedError(
            "the differentiable mode path supports electric (PEC) min-edge walls only; the group "
            "index rests on the same closed-form left eigenvector as the permittivity gradient."
        )
    k0_ref = 2.0 * np.pi * float(frequency) / c
    operator = assemble_mode_operator_jax(eps_xx, eps_yy, eps_zz, mu_xx, mu_yy, mu_zz, der_mats, k0_ref, cell_steps)
    diagonal = operator_diagonal_part(operator, jnp.asarray(mu_xx), jnp.asarray(mu_yy))
    rows, cols, _ = frequency_scaled_operator_data(operator, diagonal, k0_ref, k0_ref)
    spec = EigenSolveSpec(
        mat_rows=rows,
        mat_cols=cols,
        qmat_rows=operator.qmat.rows,
        qmat_cols=operator.qmat.cols,
        left_weights=operator.left_weights,
        num_cells=operator.num_cells,
        num_modes=num_modes,
        sigma=complex(-(neff_guess**2)),
        degeneracy_rtol=degeneracy_rtol,
    )

    def eigenvalues(k0: jax.Array) -> tuple[jax.Array, jax.Array]:
        _, _, data = frequency_scaled_operator_data(operator, diagonal, k0_ref, k0)
        return sparse_mode_eigs(spec, data, operator.qmat.data)

    def selected_index(k0: jax.Array) -> tuple[jax.Array, jax.Array]:
        values, vecs = eigenvalues(k0)
        selected = _select(values, vecs, operator.num_cells, mode_index, target_neff, filter_pol)
        return jnp.sqrt(-values[selected]), selected

    # One forward and one backward. The selection is made inside, on the solve that is already
    # running, so the eigen-solve happens exactly once - it is an integer gather, which contributes
    # no cotangent, and the eigenvector cotangent stays zero (the eigenvector adjoint is phase 2).
    # The backward is then one contraction against dA/dk0, which is the exact -2 (A - D) / k0 of the
    # module docstring.
    (neff, selected), dneff_dk0 = jax.value_and_grad(selected_index, holomorphic=True, has_aux=True)(
        jnp.asarray(k0_ref, dtype=jnp.complex128)
    )
    dneff_domega = dneff_dk0 / c
    omega = 2.0 * np.pi * float(frequency)
    return ModeDispersion(
        neff=neff,
        group_index=neff + omega * dneff_domega,
        dneff_domega=dneff_domega,
        mode_index=selected,
    )


def solve_with_left_eigenvectors(
    operator: JaxModeOperator,
    spec: EigenSolveSpec,
    data: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Eigenpairs plus the closed-form left eigenvectors, for the closed-form group index.

    Args:
        operator (JaxModeOperator): The assembled operator.
        spec (EigenSolveSpec): The static part of the eigen-solve.
        data (jax.Array): The operator's nonzero values.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: ``(eigenvalues, right vectors, left vectors)``.
    """
    vals, vecs = sparse_mode_eigs(spec, data, operator.qmat.data)
    qmat = SparseCOO(
        rows=spec.qmat_rows,
        cols=spec.qmat_cols,
        data=operator.qmat.data,
        shape=(2 * spec.num_cells, 2 * spec.num_cells),
    )
    field_h = qmat.matmul(vecs)
    n = spec.num_cells
    w_x = jnp.asarray(spec.left_weights[0], dtype=jnp.complex128)[:, None]
    w_y = jnp.asarray(spec.left_weights[1], dtype=jnp.complex128)[:, None]
    left = jnp.concatenate((w_x * field_h[n:, :], -w_y * field_h[:n, :]), axis=0)
    return vals, vecs, left
