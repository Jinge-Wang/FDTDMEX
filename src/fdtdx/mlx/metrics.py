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
