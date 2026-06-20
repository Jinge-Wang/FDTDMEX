"""Plain-MLX simulation state carried through the forward loop.

Mirrors the dynamic + material fields of :class:`fdtdx.fdtd.container.ArrayContainer`
that the forward kernels touch. CPML ``a``/``b`` coefficients and ``1/kappa`` are
time-invariant (they depend only on alpha/kappa/sigma/dt), so they are precomputed once
on the host in the bridge and carried here as ready-to-use arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass
class MLXState:
    """Mutable container of ``mx.array`` field/material data for the time loop."""

    #: Electric field, shape (3, Nx, Ny, Nz).
    E: mx.array
    #: Magnetic field (eta0-normalized), shape (3, Nx, Ny, Nz).
    H: mx.array
    #: CPML auxiliary field for the E update, shape (6, Nx, Ny, Nz).
    psi_E: mx.array
    #: CPML auxiliary field for the H update, shape (6, Nx, Ny, Nz).
    psi_H: mx.array

    #: Inverse permittivity, shape (1|3|9, Nx, Ny, Nz).
    inv_eps: mx.array
    #: Inverse permeability, scalar float or shape (1|3|9, Nx, Ny, Nz).
    inv_mu: Any

    #: Precomputed CPML coefficients, each shape (6, Nx, Ny, Nz).
    cpml_a: mx.array
    cpml_b: mx.array
    #: Precomputed 1 / kappa, shape (6, Nx, Ny, Nz).
    inv_kappa: mx.array

    #: Optional electric conductivity, shape (1|3|9, Nx, Ny, Nz) (M2+).
    sigma_E: Any = None
    #: Optional magnetic conductivity, shape (1|3|9, Nx, Ny, Nz) (M2+).
    sigma_H: Any = None
