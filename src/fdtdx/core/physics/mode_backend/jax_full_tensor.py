"""JAX assembly, eigen-solve and field reconstruction for the four-component mode operator.

The JAX counterpart of :mod:`fdtdx.core.physics.mode_backend.full_tensor`, standing in the same
relation to it as :mod:`fdtdx.core.physics.mode_backend.jax_operator` does to
:mod:`fdtdx.core.physics.mode_backend.solve`: the operator is carried as a coordinate list whose
*pattern* is fixed by the grid and whose *values* are the only traced object, so a permittivity
gradient reaches the eigen-solve and comes back through the same ``custom_vjp``
(:func:`fdtdx.core.physics.mode_backend.jax_solve.sparse_mode_eigs`).

Every block of the operator is one of

.. code-block:: text

    r * A diag(d) B          (two difference or averaging matrices around one material vector)
    r * A diag(d) B, scaled by a second material vector on the row index
    diag(d)

which is exactly the shape :func:`fdtdx.core.physics.mode_backend.jax_operator._sandwich_terms`
precomputes, so no new index bookkeeping is introduced. The longitudinal terms carry the *product*
of two tensor entries in one material vector — ``eps_xz eps_zx / eps_zz`` and ``eps_xz / eps_zz`` —
because both are sampled on the node, where ``E_z`` lives.

The backward. ``eps_xz`` and ``eps_zx`` enter the operator *multiplicatively together* in the
``hy``-``Ex`` term, so the chain rule through the assembly is what makes ``d n_eff / d eps_xz``
correct; there is no separate first-order formula to keep in step, and nothing in the backward has
to know that the tier changed. The left eigenvector is always solved for on this path: the
closed-form one of the transverse tier is a statement about the transverse operator and has no
asserted counterpart here.

Precision: ``complex128`` unconditionally, so the module refuses to run without ``jax_enable_x64``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.experimental import sparse as jsparse

from fdtdx.constants import eta0
from fdtdx.core.physics.mode_backend.jax_operator import (
    SparseCOO,
    _Block,
    _sandwich_block,
    _stack_blocks,
    require_x64,
)
from fdtdx.core.physics.mode_backend.jax_solve import (
    DEFAULT_DEGENERACY_RTOL,
    EigenSolveSpec,
    sparse_mode_eigs,
)

__all__ = [
    "JaxFullTensorOperator",
    "assemble_full_tensor_operator_jax",
    "reconstruct_fields_full_jax",
    "solve_modes_full_tensor_jax",
]

#: The nine permittivity entries in the row-major order the cross-section uses.
TENSOR_ENTRIES = ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")


class JaxFullTensorOperator(NamedTuple):
    """The JAX-assembled four-component operator and what the reconstruction needs from it.

    Attributes:
        mat: The ``(4N, 4N)`` operator with ``mat v = n_eff v`` for ``v = [Ex; Ey; hx; hy]``.
        inv_eps_zz: Flattened ``1 / eps_zz``.
        inv_mu_zz: Flattened ``1 / mu_zz``.
        a_zx: Flattened ``eps_zx / eps_zz``.
        a_zy: Flattened ``eps_zy / eps_zz``.
        der_bcoo: The four ``k0``-normalised difference matrices ``(dxf, dxb, dyf, dyb)`` as BCOO.
        avg_bcoo: The four averaging matrices ``(axb, axf, ayb, ayf)`` as BCOO.
        num_cells: ``N``, the number of transverse cells.
    """

    mat: SparseCOO
    inv_eps_zz: jax.Array
    inv_mu_zz: jax.Array
    a_zx: jax.Array
    a_zy: jax.Array
    der_bcoo: tuple[jsparse.BCOO, jsparse.BCOO, jsparse.BCOO, jsparse.BCOO]
    avg_bcoo: tuple[jsparse.BCOO, jsparse.BCOO, jsparse.BCOO, jsparse.BCOO]
    num_cells: int


def _to_bcoo(m: sp.spmatrix) -> jsparse.BCOO:
    coo = sp.coo_matrix(m)
    indices = jnp.stack((jnp.asarray(coo.row), jnp.asarray(coo.col)), axis=1)
    return jsparse.BCOO((jnp.asarray(coo.data, dtype=jnp.complex128), indices), shape=coo.shape)


def assemble_full_tensor_operator_jax(
    eps: dict[str, jax.Array],
    mu_xx: jax.Array,
    mu_yy: jax.Array,
    mu_zz: jax.Array,
    der_mats: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    avg_mats: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    k0: float,
) -> JaxFullTensorOperator:
    """Assemble the ``k0``-normalised four-component mode operator in JAX.

    Algebraically identical to
    :func:`fdtdx.core.physics.mode_backend.full_tensor.assemble_full_tensor_operator`; the block
    products are expanded so that the only traced quantities are the material vectors.

    Args:
        eps (dict[str, jax.Array]): The nine flattened permittivity components (keys
            ``"xx"`` ... ``"zz"``); off-diagonal entries may be omitted and are taken as zero.
        mu_xx (jax.Array): Flattened ``mu_xx``.
        mu_yy (jax.Array): Flattened ``mu_yy``.
        mu_zz (jax.Array): Flattened ``mu_zz``.
        der_mats: ``(dxf, dxb, dyf, dyb)`` SI difference matrices.
        avg_mats: ``(axb, axf, ayb, ayf)`` averaging matrices.
        k0 (float): Free-space wavenumber ``2 pi f / c`` (1/m).

    Returns:
        JaxFullTensorOperator: The operator and the pieces the reconstruction needs.

    Raises:
        ValueError: If ``jax_enable_x64`` is off.
    """
    require_x64("assemble_full_tensor_operator_jax")
    n = int(jnp.asarray(eps["xx"]).size)
    zero = jnp.zeros(n, dtype=jnp.complex128)

    def entry(name: str) -> jax.Array:
        value = eps.get(name)
        return zero if value is None else jnp.asarray(value, dtype=jnp.complex128)

    e_xx, e_xy, e_xz = entry("xx"), entry("xy"), entry("xz")
    e_yx, e_yy, e_yz = entry("yx"), entry("yy"), entry("yz")
    e_zx, e_zy, e_zz = entry("zx"), entry("zy"), entry("zz")
    m_xx = jnp.asarray(mu_xx, dtype=jnp.complex128)
    m_yy = jnp.asarray(mu_yy, dtype=jnp.complex128)
    m_zz = jnp.asarray(mu_zz, dtype=jnp.complex128)

    inv_eps_zz = 1.0 / e_zz
    inv_mu_zz = 1.0 / m_zz
    a_zx = e_zx * inv_eps_zz
    a_zy = e_zy * inv_eps_zz

    dxf, dxb, dyf, dyb = (sp.csr_matrix(m, dtype=np.float64) / float(k0) for m in der_mats)
    axb, axf, ayb, ayf = (sp.csr_matrix(m, dtype=np.float64) for m in avg_mats)

    diag_idx = np.arange(n, dtype=np.int64)
    ones = np.ones(n, dtype=np.float64)

    def sandwich(a, b, sign: complex, row: int, col: int) -> _Block:
        return _sandwich_block(a, b, sign, row * n, col * n)

    def diagonal(sign: complex, row: int, col: int) -> _Block:
        return _Block(diag_idx + row * n, diag_idx + col * n, sign * ones, diag_idx)

    # (block, material vector the block's kidx indexes, row scaling, column scaling). The two
    # right-column longitudinal entries multiply E_z and are therefore sampled on the node with
    # eps_zz, i.e. *inside* the average that carries the product out to the transverse location,
    # which is what keeps the discrete spectrum reciprocal (see the numpy module).
    blocks: list[tuple[_Block, jax.Array, jax.Array | None, jax.Array | None]] = [
        # row Ex:  n Ex = -i Dx Z + mu_yy hy
        (sandwich(dxf, axb, 1j, 0, 0), a_zx, None, None),
        (sandwich(dxf, ayb, 1j, 0, 1), a_zy, None, None),
        (sandwich(dxf, dyb, -1.0, 0, 2), inv_eps_zz, None, None),
        (sandwich(dxf, dxb, 1.0, 0, 3), inv_eps_zz, None, None),
        (diagonal(1.0, 0, 3), m_yy, None, None),
        # row Ey:  n Ey = -i Dy Z - mu_xx hx
        (sandwich(dyf, axb, 1j, 1, 0), a_zx, None, None),
        (sandwich(dyf, ayb, 1j, 1, 1), a_zy, None, None),
        (sandwich(dyf, dyb, -1.0, 1, 2), inv_eps_zz, None, None),
        (sandwich(dyf, dxb, 1.0, 1, 3), inv_eps_zz, None, None),
        (diagonal(-1.0, 1, 2), m_xx, None, None),
        # row hx:  n hx = -Dx K - ( eps_yx Ex + eps_yy Ey + eps_yz Z )
        (sandwich(dxb, dyf, 1.0, 2, 0), inv_mu_zz, None, None),
        (sandwich(dxb, dxf, -1.0, 2, 1), inv_mu_zz, None, None),
        (diagonal(-1.0, 2, 0), e_yx, None, None),
        (diagonal(-1.0, 2, 1), e_yy, None, None),
        (sandwich(ayf, axb, 1.0, 2, 0), e_yz * a_zx, None, None),
        (sandwich(ayf, ayb, 1.0, 2, 1), e_yz * a_zy, None, None),
        (sandwich(ayf, dyb, 1j, 2, 2), e_yz * inv_eps_zz, None, None),
        (sandwich(ayf, dxb, -1j, 2, 3), e_yz * inv_eps_zz, None, None),
        # row hy:  n hy = -Dy K + ( eps_xx Ex + eps_xy Ey + eps_xz Z )
        (sandwich(dyb, dyf, 1.0, 3, 0), inv_mu_zz, None, None),
        (sandwich(dyb, dxf, -1.0, 3, 1), inv_mu_zz, None, None),
        (diagonal(1.0, 3, 0), e_xx, None, None),
        (diagonal(1.0, 3, 1), e_xy, None, None),
        (sandwich(axf, axb, -1.0, 3, 0), e_xz * a_zx, None, None),
        (sandwich(axf, ayb, -1.0, 3, 1), e_xz * a_zy, None, None),
        (sandwich(axf, dyb, -1j, 3, 2), e_xz * inv_eps_zz, None, None),
        (sandwich(axf, dxb, 1j, 3, 3), e_xz * inv_eps_zz, None, None),
    ]
    mat = _stack_blocks(blocks, shape=(4 * n, 4 * n), num_cells=n)
    return JaxFullTensorOperator(
        mat=mat,
        inv_eps_zz=inv_eps_zz,
        inv_mu_zz=inv_mu_zz,
        a_zx=a_zx,
        a_zy=a_zy,
        der_bcoo=(_to_bcoo(dxf), _to_bcoo(dxb), _to_bcoo(dyf), _to_bcoo(dyb)),
        avg_bcoo=(_to_bcoo(axb), _to_bcoo(axf), _to_bcoo(ayb), _to_bcoo(ayf)),
        num_cells=n,
    )


def reconstruct_fields_full_jax(
    operator: JaxFullTensorOperator,
    vecs: jax.Array,
    eigenvalues: jax.Array,
    direction: str = "+",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Recover the six field components from the four-component eigenvectors, in JAX.

    Args:
        operator (JaxFullTensorOperator): The assembled operator.
        vecs (jax.Array): Eigenvectors ``[Ex; Ey; hx; hy]``, shape ``(4N, M)``.
        eigenvalues (jax.Array): The matching eigenvalues, shape ``(M,)``.
        direction (str): ``"+"`` or ``"-"``; the eigenvalues already carry the sign of the
            direction, and this only decides how they are reported.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array, jax.Array]: ``(E, H, neff, keff)`` with ``E`` and
        ``H`` of shape ``(3, N, M)``.

    Raises:
        ValueError: If ``jax_enable_x64`` is off.
    """
    require_x64("reconstruct_fields_full_jax")
    n = operator.num_cells
    dxf, dxb, dyf, dyb = operator.der_bcoo
    axb, _axf, ayb, _ayf = operator.avg_bcoo

    n_complex = jnp.asarray(eigenvalues, dtype=jnp.complex128)
    if direction == "-":
        n_complex = -n_complex
    neff = jnp.real(n_complex)
    keff = jnp.imag(n_complex)

    scale = jnp.linalg.norm(vecs, axis=0)
    vecs = vecs / jnp.where(scale > 0, scale, 1.0)[None, :]

    ex = vecs[:n, :]
    ey = vecs[n : 2 * n, :]
    hx = vecs[2 * n : 3 * n, :]
    hy = vecs[3 * n :, :]

    ez = (
        1j * operator.inv_eps_zz[:, None] * ((dxb @ hy) - (dyb @ hx))
        - operator.a_zx[:, None] * (axb @ ex)
        - operator.a_zy[:, None] * (ayb @ ey)
    )
    hz = -1j * operator.inv_mu_zz[:, None] * ((dxf @ ey) - (dyf @ ex))

    field_e = jnp.stack((ex, ey, ez), axis=0)
    field_h = jnp.stack((hx, hy, hz), axis=0) / eta0

    e_t = jnp.concatenate((field_e[0], field_e[1]), axis=0)
    pivot = e_t[jnp.argmax(jnp.abs(e_t), axis=0), jnp.arange(e_t.shape[1])]
    magnitude = jnp.abs(pivot)
    phase = jnp.where(magnitude > 0, pivot / jnp.where(magnitude > 0, magnitude, 1.0), 1.0 + 0.0j)
    field_e = field_e / phase[None, None, :]
    field_h = field_h / phase[None, None, :]
    return field_e, field_h, neff, keff


def solve_modes_full_tensor_jax(
    eps: dict[str, jax.Array],
    mu_xx: jax.Array,
    mu_yy: jax.Array,
    mu_zz: jax.Array,
    der_mats: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    avg_mats: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    k0: float,
    num_modes: int,
    neff_guess: float | jax.Array,
    direction: str = "+",
    degeneracy_rtol: float = DEFAULT_DEGENERACY_RTOL,
    dmin_pmc: tuple[bool, bool] = (False, False),
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The whole full-tensor mode solve, with the permittivity gradient connected end to end.

    Args:
        eps (dict[str, jax.Array]): The nine flattened permittivity components.
        mu_xx (jax.Array): Flattened ``mu_xx``.
        mu_yy (jax.Array): Flattened ``mu_yy``.
        mu_zz (jax.Array): Flattened ``mu_zz``.
        der_mats: ``(dxf, dxb, dyf, dyb)`` SI difference matrices.
        avg_mats: ``(axb, axf, ayb, ayf)`` averaging matrices.
        k0 (float): Free-space wavenumber ``2 pi f / c`` (1/m).
        num_modes (int): Number of modes to return.
        neff_guess (float | jax.Array): Shift-invert target effective index, positive. May be
            traced: it reaches the eigen-solve as a runtime scalar.
        direction (str): ``"+"`` or ``"-"``. The backward spectrum is solved directly.
        degeneracy_rtol (float): Relative gap defining a degenerate block.
        dmin_pmc (tuple[bool, bool]): The per-axis min-edge wall types, needed here only to refuse
            the case the adjoint has not been validated for.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array, jax.Array]: ``(E, H, neff, keff)``.

    Raises:
        NotImplementedError: If a min edge is magnetic.
    """
    if any(dmin_pmc):
        raise NotImplementedError(
            "the differentiable full-tensor mode path supports electric (PEC) min-edge walls only. "
            "Solve the full cross-section instead of using a symmetry plane."
        )
    operator = assemble_full_tensor_operator_jax(eps, mu_xx, mu_yy, mu_zz, der_mats, avg_mats, k0)
    spec = EigenSolveSpec(
        mat_rows=operator.mat.rows,
        mat_cols=operator.mat.cols,
        qmat_rows=np.zeros(0, dtype=np.int64),
        qmat_cols=np.zeros(0, dtype=np.int64),
        left_weights=(np.zeros(0), np.zeros(0)),
        num_cells=operator.num_cells,
        num_modes=num_modes,
        sigma=0j,
        degeneracy_rtol=degeneracy_rtol,
        solve_left=True,
        eigenvalue_kind="neff",
        backward=direction == "-",
    )
    sign = -1.0 if direction == "-" else 1.0
    sigma = sign * jnp.asarray(neff_guess, dtype=jnp.complex128)
    vals, vecs = sparse_mode_eigs(spec, operator.mat.data, jnp.zeros(0, dtype=jnp.complex128), sigma)
    return reconstruct_fields_full_jax(operator, vecs, vals, direction=direction)
