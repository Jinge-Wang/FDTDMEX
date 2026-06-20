"""Host-side precompute of the (time-invariant) CPML recurrence coefficients.

Mirrors the inline ``a``/``b`` computation in ``fdtdx.core.physics.curl`` (lines
245-246 / 317-318). Because ``alpha``, ``kappa``, ``sigma`` and ``dt`` are all constant
over the simulation, ``a`` and ``b`` are computed once on the host (numpy) and shipped to
MLX, so the MLX curl never recomputes them per step. ``1/kappa`` is precomputed for the
same reason. Computed in the field dtype (float32 by default) to match the JAX path.
"""

from __future__ import annotations

import numpy as np


def precompute_cpml_coeffs(
    alpha: np.ndarray,
    kappa: np.ndarray,
    sigma: np.ndarray,
    dt: float,
    eps0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(a, b, inv_kappa)``, each shape ``(6, Nx, Ny, Nz)``.

    Matches ``b = expm1(-dt/eps0 * (sigma/kappa + alpha)) + 1`` and
    ``a = nan_to_num((b - 1) * sigma / (sigma + alpha*kappa) / kappa)``.
    """
    alpha = np.asarray(alpha)
    kappa = np.asarray(kappa)
    sigma = np.asarray(sigma)

    b = np.expm1(-dt / eps0 * (sigma / kappa + alpha)) + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (b - 1.0) * sigma / (sigma + alpha * kappa) / kappa
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    inv_kappa = 1.0 / kappa

    dtype = alpha.dtype
    return a.astype(dtype), b.astype(dtype), inv_kappa.astype(dtype)
