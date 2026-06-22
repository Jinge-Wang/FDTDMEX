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


def detect_pml_slabs(
    a: np.ndarray, b: np.ndarray, inv_kappa: np.ndarray, pad: int = 1
) -> list[tuple[int, int]]:
    """Per-axis ``(lo, hi)`` PML slab thickness: the CPML correction is confined to indices
    ``[0:lo]`` and ``[N-hi:N]`` along each axis ``k``.

    ``a``/``b``/``inv_kappa`` have shape ``(6, Nx, Ny, Nz)``; index ``k`` is the E-side axis-``k``
    profile, ``k+3`` the H-side. Each is non-trivial (``a≠0`` / ``b≠1`` / ``inv_kappa≠1``) only in
    the two slabs perpendicular to axis ``k``. The slab correction ``(inv_kappa-1)·d + ψ`` is
    exactly zero wherever all three are trivial, so the detected support is exact; ``pad`` widens
    each slab by a safety margin of provably-zero cells (cheap, keeps it exact under float ramps).
    """
    slabs: list[tuple[int, int]] = []
    for k in range(3):
        active = np.zeros(a.shape[1:], dtype=bool)
        for idx in (k, k + 3):
            active |= a[idx] != 0.0
            active |= b[idx] != 1.0
            active |= inv_kappa[idx] != 1.0
        other = tuple(ax for ax in range(3) if ax != k)
        mask = active.any(axis=other)  # 1-D along axis k
        n = int(mask.shape[0])
        lo = 0
        while lo < n and mask[lo]:
            lo += 1
        hi = 0
        while hi < n and mask[n - 1 - hi]:
            hi += 1
        # widen by `pad` provably-zero cells, but never overlap (lo+hi <= n)
        if lo:
            lo = min(n, lo + pad)
        if hi:
            hi = min(n, hi + pad)
        if lo + hi > n:
            lo, hi = n, 0  # whole axis is PML (degenerate tiny domain) -> one slab covers it
        slabs.append((lo, hi))
    return slabs
