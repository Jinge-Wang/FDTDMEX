"""Full-tensor (9-component) anisotropic update helpers for MLX.

Translation of ``fdtdx.fdtd.misc.compute_anisotropic_update_matrices`` and
``avg_anisotropic_E/H_component``, plus the per-cell 3x3 algebra. The JAX path uses
``jnp.linalg.solve``; here the per-cell inverse is the analytic 3x3 cofactor formula
(exact, and MLX-GPU friendly).

Off-diagonal averaging co-locates a field component with another component's Yee point via two
separable half-steps. One step moves a cell-centered sample to an edge (center->edge), the
other moves an edge sample to a cell center (edge->center). On a non-uniform grid only the
center->edge step needs spacing weights (the target edge is *not* halfway between the two cell
centers); the edge->center step lands on the cell center, which is the exact geometric midpoint
of its two edges, so it stays an unweighted mean. By the Yee staggering this means the backward
roll (+1) is always the width-weighted one and the forward roll (-1) is the plain mean -- the
same rule fdtdx's ``interpolate_fields`` follows (only its ``_backward_edge_average`` is
weighted). ``aniso_widths is None`` (uniform grid) collapses every step to a plain mean, so the
result is byte-identical to fdtdx's unweighted 4-point average. This weighted form is 2nd-order
on graded meshes -- intentionally *more correct* than fdtdx, which leaves this average
unweighted even on non-uniform grids.

Convention: tensors are shape ``(3, 3, Nx, Ny, Nz)``; ``M[i, j]`` is a spatial array.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def expand_to_3x3_mlx(arr: Any) -> mx.array:
    """Expand (1|3|9, Nx,Ny,Nz) or a scalar to a (3, 3, ...) tensor (mirrors expand_to_3x3)."""
    if not isinstance(arr, mx.array):
        a = mx.array(float(arr))
        z = mx.zeros(())
        m = mx.stack([mx.stack([a, z, z]), mx.stack([z, a, z]), mx.stack([z, z, a])])  # (3, 3)
        return m.reshape(3, 3, 1, 1, 1)

    n = arr.shape[0]
    spatial = arr.shape[1:]
    if n == 9:
        return arr.reshape((3, 3, *spatial))
    z = mx.zeros(spatial, dtype=arr.dtype)
    if n == 1:
        a = arr[0]
        return mx.stack([mx.stack([a, z, z]), mx.stack([z, a, z]), mx.stack([z, z, a])])
    if n == 3:
        a, b, c = arr[0], arr[1], arr[2]
        return mx.stack([mx.stack([a, z, z]), mx.stack([z, b, z]), mx.stack([z, z, c])])
    raise ValueError(f"cannot expand leading dim {n} to a 3x3 tensor")


def _eye3(ref: mx.array) -> mx.array:
    return mx.eye(3).reshape(3, 3, 1, 1, 1).astype(ref.dtype)


def matmul3x3(A: mx.array, B: mx.array) -> mx.array:
    """Per-cell 3x3 matrix product: out[i,k] = sum_j A[i,j] B[j,k]."""
    rows = []
    for i in range(3):
        rows.append(mx.stack([A[i, 0] * B[0, k] + A[i, 1] * B[1, k] + A[i, 2] * B[2, k] for k in range(3)]))
    return mx.stack(rows)


def inv3x3(M: mx.array) -> mx.array:
    """Per-cell analytic inverse of a 3x3 tensor (3, 3, ...)."""
    m = [[M[i, j] for j in range(3)] for i in range(3)]
    det = (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )
    inv = [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det,
            -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) / det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det,
        ],
        [
            -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) / det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det,
            -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) / det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det,
            -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) / det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det,
        ],
    ]
    return mx.stack([mx.stack(inv[i]) for i in range(3)])


def compute_anisotropic_update_matrices_mlx(
    inv_material: mx.array, sigma: mx.array | None, c: float, eta_factor: float
) -> tuple[mx.array, mx.array]:
    """Return (A, B), each (3, 3, ...). Lossless (sigma None) -> A = I, B = c * inv_material."""
    if sigma is None:
        return _eye3(inv_material), c * inv_material
    factor = (c * eta_factor / 2.0) * matmul3x3(inv_material, sigma)
    eye = _eye3(factor)
    m1_inv = inv3x3(eye + factor)
    A = matmul3x3(m1_inv, eye - factor)
    B = c * matmul3x3(m1_inv, inv_material)
    return A, B


def _forward_avg(f: mx.array, axis: int) -> mx.array:
    """Unweighted edge->center half-step along ``axis`` (forward roll, exact midpoint)."""
    return 0.5 * (f + mx.roll(f, -1, axis=axis))


def _backward_avg(f: mx.array, axis: int, w_pad) -> mx.array:
    """Center->edge half-step along ``axis`` (backward roll).

    ``w_pad is None`` -> unweighted mean. Otherwise weight each cell-centered sample by the
    *opposite* cell's width, so the edge value is the correct linear interpolant on a graded mesh
    (the 1/2 in the half-widths cancels, so full widths are used directly).
    """
    fn = mx.roll(f, 1, axis=axis)
    if w_pad is None:
        return 0.5 * (f + fn)
    wn = mx.roll(w_pad, 1, axis=axis)
    return (f * wn + fn * w_pad) / (w_pad + wn)


def avg_anisotropic_E_component_mlx(field_pad: mx.array, component: int, location: int, aniso_widths=None) -> mx.array:
    """Average an E component to another component's Yee point (spacing-weighted on graded meshes).

    The field is at a cell center along its own (``component``) axis and at an edge along the
    target (``location``) axis, so the center->edge step (``component``) is weighted and the
    edge->center step (``location``) is a plain mean.
    """
    f = field_pad[component]
    w = None if aniso_widths is None else aniso_widths[component]
    g = _forward_avg(f, location)
    h = _backward_avg(g, component, w)
    return h[1:-1, 1:-1, 1:-1]


def avg_anisotropic_H_component_mlx(field_pad: mx.array, component: int, location: int, aniso_widths=None) -> mx.array:
    """Average an H component to another component's Yee point (spacing-weighted on graded meshes).

    H sits at an edge along its own (``component``) axis and at a cell center along the target
    (``location``) axis, so here the center->edge step (``location``) is weighted and the
    edge->center step (``component``) is a plain mean.
    """
    f = field_pad[component]
    w = None if aniso_widths is None else aniso_widths[location]
    g = _backward_avg(f, location, w)
    h = _forward_avg(g, component)
    return h[1:-1, 1:-1, 1:-1]


def quad_form3(v: mx.array, M: mx.array) -> mx.array:
    """Per-cell quadratic form sum_ij v[i] M[i,j] v[j] (real fields). Used for 9-tensor energy."""
    total = None
    for i in range(3):
        for j in range(3):
            term = v[i] * M[i, j] * v[j]
            total = term if total is None else total + term
    return total
