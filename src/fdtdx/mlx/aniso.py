"""Full-tensor (9-component) anisotropic update helpers for MLX.

Translation of ``fdtdx.fdtd.misc.compute_anisotropic_update_matrices`` and
``avg_anisotropic_E/H_component``, plus the per-cell 3x3 algebra. The JAX path uses
``jnp.linalg.solve``; here the per-cell inverse is the analytic 3x3 cofactor formula
(exact, and MLX-GPU friendly). Off-diagonal averaging is the unweighted 4-point average,
exactly matching fdtdx on a uniform grid (the spacing-weighted version is M4).

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


def _avg(field_comp: mx.array, roll1, roll2) -> mx.array:
    f = field_comp
    s = f + mx.roll(f, roll1[0], axis=roll1[1]) + mx.roll(f, roll2[0], axis=roll2[1])
    s = s + mx.roll(mx.roll(f, roll1[0], axis=roll1[1]), roll2[0], axis=roll2[1])
    return (s / 4.0)[1:-1, 1:-1, 1:-1]


def avg_anisotropic_E_component_mlx(field_pad: mx.array, component: int, location: int) -> mx.array:
    """Unweighted 4-point average of an E component to a Yee location (mirrors fdtdx)."""
    return _avg(field_pad[component], (-1, location), (1, component))


def avg_anisotropic_H_component_mlx(field_pad: mx.array, component: int, location: int) -> mx.array:
    """Unweighted 4-point average of an H component to a Yee location (mirrors fdtdx)."""
    return _avg(field_pad[component], (1, location), (-1, component))


def quad_form3(v: mx.array, M: mx.array) -> mx.array:
    """Per-cell quadratic form sum_ij v[i] M[i,j] v[j] (real fields). Used for 9-tensor energy."""
    total = None
    for i in range(3):
        for j in range(3):
            term = v[i] * M[i, j] * v[j]
            total = term if total is None else total + term
    return total
