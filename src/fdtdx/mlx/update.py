"""MLX E/H field updates (isotropic/diagonal fast path + full-anisotropic 9-tensor path).

Translation of ``fdtdx.fdtd.update.update_E`` / ``update_H``. The non-9-tensor branch is the
component-wise fast path; the 9-tensor branch builds per-cell A/B matrices and uses the
unweighted off-diagonal averaging (see :mod:`fdtdx.mlx.aniso`). Source injection and detector
recording are handled by the loop driver. Conductivity follows Schneider ch. 3.12.
"""

from __future__ import annotations

import mlx.core as mx

from fdtdx.constants import eta0
from fdtdx.mlx.aniso import (
    avg_anisotropic_E_component_mlx,
    avg_anisotropic_H_component_mlx,
    compute_anisotropic_update_matrices_mlx,
    expand_to_3x3_mlx,
)
from fdtdx.mlx.curl import curl_E_mlx, curl_H_mlx, pad_fields_mlx
from fdtdx.mlx.state import MLXState


def _is_full_tensor(arr) -> bool:
    return isinstance(arr, mx.array) and arr.ndim > 0 and arr.shape[0] == 9


def update_E_mlx(state: MLXState, c: float, simulate_boundaries: bool = True) -> tuple[mx.array, mx.array]:
    """Return ``(E_new, psi_E_new)`` from ``dE/dt = (1/eps) curl(H)``."""
    H_pad = pad_fields_mlx(state.H, state.periodic_axes)
    curl, psi_E = curl_H_mlx(H_pad, state.psi_E, state.cpml_a, state.cpml_b, state.inv_kappa, simulate_boundaries)

    inv_eps = state.inv_eps
    sigma_E = state.sigma_E

    if not _is_full_tensor(inv_eps) and not _is_full_tensor(sigma_E):
        factor = 1.0
        if sigma_E is not None:
            factor = 1.0 - c * sigma_E * eta0 * inv_eps / 2.0
        E = factor * state.E + c * curl * inv_eps
        if sigma_E is not None:
            E = E / (1.0 + c * sigma_E * eta0 * inv_eps / 2.0)
        return E, psi_E

    return _update_aniso(state.E, curl, inv_eps, sigma_E, c, eta0, add=True, periodic_axes=state.periodic_axes), psi_E


def update_H_mlx(state: MLXState, c: float, simulate_boundaries: bool = True) -> tuple[mx.array, mx.array]:
    """Return ``(H_new, psi_H_new)`` from ``dH/dt = -(1/mu) curl(E)``."""
    E_pad = pad_fields_mlx(state.E, state.periodic_axes)
    curl, psi_H = curl_E_mlx(E_pad, state.psi_H, state.cpml_a, state.cpml_b, state.inv_kappa, simulate_boundaries)

    inv_mu = state.inv_mu
    sigma_H = state.sigma_H

    if not _is_full_tensor(inv_mu) and not _is_full_tensor(sigma_H):
        factor = 1.0
        if sigma_H is not None:
            factor = 1.0 - c * sigma_H / eta0 * inv_mu / 2.0
        H = factor * state.H - c * curl * inv_mu
        if sigma_H is not None:
            H = H / (1.0 + c * sigma_H / eta0 * inv_mu / 2.0)
        return H, psi_H

    return _update_aniso(
        state.H, curl, inv_mu, sigma_H, c, 1.0 / eta0, add=False, periodic_axes=state.periodic_axes
    ), psi_H


def _update_aniso(F, curl, inv_material, sigma, c: float, eta_factor: float, add: bool, periodic_axes):
    """Full-anisotropic E (add=True) or H (add=False) update via per-cell A/B matrices.

    ``F`` and ``curl`` are the un-padded (3, Nx, Ny, Nz) fields. Off-diagonal terms use the
    other components averaged to this component's Yee location.
    """
    inv_t = expand_to_3x3_mlx(inv_material)
    sigma_t = expand_to_3x3_mlx(sigma) if sigma is not None else None
    A, B = compute_anisotropic_update_matrices_mlx(inv_t, sigma_t, c, eta_factor)

    avg = avg_anisotropic_E_component_mlx if add else avg_anisotropic_H_component_mlx
    Fp = pad_fields_mlx(F, periodic_axes)
    Cp = pad_fields_mlx(curl, periodic_axes)

    # F[other] averaged to each diagonal component's location.
    Fx_y, Fx_z = avg(Fp, 0, 1), avg(Fp, 0, 2)
    Fy_x, Fy_z = avg(Fp, 1, 0), avg(Fp, 1, 2)
    Fz_x, Fz_y = avg(Fp, 2, 0), avg(Fp, 2, 1)
    Cx_y, Cx_z = avg(Cp, 0, 1), avg(Cp, 0, 2)
    Cy_x, Cy_z = avg(Cp, 1, 0), avg(Cp, 1, 2)
    Cz_x, Cz_y = avg(Cp, 2, 0), avg(Cp, 2, 1)

    a_term_x = A[0, 0] * F[0] + A[0, 1] * Fy_x + A[0, 2] * Fz_x
    a_term_y = A[1, 0] * Fx_y + A[1, 1] * F[1] + A[1, 2] * Fz_y
    a_term_z = A[2, 0] * Fx_z + A[2, 1] * Fy_z + A[2, 2] * F[2]

    b_term_x = B[0, 0] * curl[0] + B[0, 1] * Cy_x + B[0, 2] * Cz_x
    b_term_y = B[1, 0] * Cx_y + B[1, 1] * curl[1] + B[1, 2] * Cz_y
    b_term_z = B[2, 0] * Cx_z + B[2, 1] * Cy_z + B[2, 2] * curl[2]

    if add:
        Fx, Fy, Fz = a_term_x + b_term_x, a_term_y + b_term_y, a_term_z + b_term_z
    else:
        Fx, Fy, Fz = a_term_x - b_term_x, a_term_y - b_term_y, a_term_z - b_term_z

    return mx.stack([Fx, Fy, Fz], axis=0)
