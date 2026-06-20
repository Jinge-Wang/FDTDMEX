"""MLX E/H field updates (isotropic / diagonal-anisotropic path).

Translation of the non-9-tensor branch of ``fdtdx.fdtd.update.update_E`` / ``update_H``.
The full-anisotropic (9-tensor) path lands in M3. Source injection and detector recording
are handled by the loop driver (:mod:`fdtdx.mlx.loop`), matching fdtdx where update_E/H
apply curl + material, then sources, then boundary post-updates.

Conductivity (``sigma_E``/``sigma_H``) follows Schneider ch. 3.12 and is a no-op when the
arrays are ``None`` (M1 is lossless); it is wired here so M2 only needs to populate them.
"""

from __future__ import annotations

import mlx.core as mx

from fdtdx.constants import eta0
from fdtdx.mlx.curl import curl_E_mlx, curl_H_mlx, pad_zero
from fdtdx.mlx.state import MLXState


def update_E_mlx(state: MLXState, c: float, simulate_boundaries: bool = True) -> tuple[mx.array, mx.array]:
    """Return ``(E_new, psi_E_new)`` from ``dE/dt = (1/eps) curl(H)``."""
    H_pad = pad_zero(state.H)
    curl, psi_E = curl_H_mlx(H_pad, state.psi_E, state.cpml_a, state.cpml_b, state.inv_kappa, simulate_boundaries)

    inv_eps = state.inv_eps
    sigma_E = state.sigma_E

    factor = 1.0
    if sigma_E is not None:
        factor = 1.0 - c * sigma_E * eta0 * inv_eps / 2.0

    E = factor * state.E + c * curl * inv_eps

    if sigma_E is not None:
        E = E / (1.0 + c * sigma_E * eta0 * inv_eps / 2.0)

    return E, psi_E


def update_H_mlx(state: MLXState, c: float, simulate_boundaries: bool = True) -> tuple[mx.array, mx.array]:
    """Return ``(H_new, psi_H_new)`` from ``dH/dt = -(1/mu) curl(E)``."""
    E_pad = pad_zero(state.E)
    curl, psi_H = curl_E_mlx(E_pad, state.psi_H, state.cpml_a, state.cpml_b, state.inv_kappa, simulate_boundaries)

    inv_mu = state.inv_mu
    sigma_H = state.sigma_H

    factor = 1.0
    if sigma_H is not None:
        factor = 1.0 - c * sigma_H / eta0 * inv_mu / 2.0

    H = factor * state.H - c * curl * inv_mu

    if sigma_H is not None:
        H = H / (1.0 + c * sigma_H / eta0 * inv_mu / 2.0)

    return H, psi_H
