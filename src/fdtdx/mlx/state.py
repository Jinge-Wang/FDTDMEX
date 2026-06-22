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
    #: CPML auxiliary field for the E update — slab-CPML 6-tuple of per-component boundary-slab
    #: arrays (component ``i`` is stored only on the PML slabs perpendicular to axis ``curl._AX[i]``).
    psi_E: Any
    #: CPML auxiliary field for the H update — slab-CPML 6-tuple of per-component boundary-slab arrays.
    psi_H: Any

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

    #: Per-axis periodic (wrap-padding) flags; True where a periodic/Bloch-k0 boundary sits.
    periodic_axes: tuple = (False, False, False)

    #: Per-axis PML slab extents ``((lo, hi), …)`` for slab-CPML; ``(0, 0)`` = no PML on that axis.
    cpml_extents: tuple = ((0, 0), (0, 0), (0, 0))

    #: Per-axis derivative metric scale for the curl (M4, non-uniform grids). Each entry is the
    #: scalar ``1.0`` on uniform grids, or an ``mx.array`` broadcasting along that axis equal to
    #: ``reference_spacing / cell_width``. ``metric_fwd`` uses the forward stencil (``curl_E``);
    #: ``metric_bwd`` the backward stencil / dual widths (``curl_H``).
    metric_fwd: tuple = (1.0, 1.0, 1.0)
    metric_bwd: tuple = (1.0, 1.0, 1.0)

    #: Per-axis half-width weights ``(cur_half, prev_half)`` for the spacing-weighted detector
    #: interpolation (``_backward_edge_average``). ``None`` on uniform grids (plain mean).
    interp_widths: Any = None

    #: Per-axis padded cell-width arrays (length N+2, broadcasting along that axis) for the
    #: spacing-weighted anisotropic off-diagonal averaging. ``None`` on uniform grids.
    aniso_widths: Any = None

    #: Phase 3 PEC keep-mask, shape (3, Nx, Ny, Nz): ``0.0`` where a PEC face zeros tangential E,
    #: ``1.0`` elsewhere. ``None`` when no PEC boundaries. Applied after E source injection.
    pec_keep: Any = None
    #: Phase 3 PMC keep-mask, shape (3, Nx, Ny, Nz): ``0.0`` where a PMC face zeros tangential H,
    #: ``1.0`` elsewhere. ``None`` when no PMC boundaries. Applied after H source injection.
    pmc_keep: Any = None
