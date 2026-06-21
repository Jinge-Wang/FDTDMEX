"""MLX electromagnetic field metrics (isotropic / diagonal path).

Translation of ``fdtdx.core.physics.metrics.compute_energy`` for the non-9-tensor case:
``0.5 * sum_i (eps_i |E_i|^2 + mu_i |H_i|^2)`` where eps_i = 1/inv_eps_i. The full-tensor
energy lands in M3.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def compute_energy_mlx(E: mx.array, H: mx.array, inv_eps: Any, inv_mu: Any) -> mx.array:
    """Energy density, shape (Nx, Ny, Nz). inv_eps/inv_mu may be (1|3, ...) arrays or scalars."""
    E_sq = mx.square(mx.abs(E))
    energy_E = mx.sum(0.5 * (1.0 / inv_eps) * E_sq, axis=0)

    H_sq = mx.square(mx.abs(H))
    energy_H = mx.sum(0.5 * (1.0 / inv_mu) * H_sq, axis=0)

    return energy_E + energy_H


def compute_poynting_flux_mlx(E: mx.array, H: mx.array) -> mx.array:
    """Poynting vector S = E x conj(H), shape (3, Nx, Ny, Nz).

    Returns the real part (real fields -> conj is a no-op), matching
    ``compute_poynting_flux(...).real`` as consumed by PoyntingFluxDetector.
    """
    Hc = mx.conjugate(H) if H.dtype in (mx.complex64,) else H
    Sx = E[1] * Hc[2] - E[2] * Hc[1]
    Sy = E[2] * Hc[0] - E[0] * Hc[2]
    Sz = E[0] * Hc[1] - E[1] * Hc[0]
    S = mx.stack([Sx, Sy, Sz], axis=0)
    return S.real if S.dtype in (mx.complex64,) else S
