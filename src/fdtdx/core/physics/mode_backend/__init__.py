"""Tidy3D-free full-vectorial waveguide mode solver.

A drop-in replacement for the Tidy3D mode-solver call used in
:mod:`fdtdx.core.physics.modes`. :func:`fdtdmex_mode_computation_wrapper` mirrors the signature and
``list[ModeTupleType]`` return of ``tidy3d_mode_computation_wrapper`` (Tidy3D's z-propagation
convention, pre-eta0 scaling) so the surrounding ``compute_mode`` post-processing — axis rotation,
eta0-scaling, Poynting normalisation — is unchanged.

Scope (stages 1-2): straight waveguide; uniform and rectilinear transverse grids; isotropic and
diagonally-anisotropic permittivity/permeability. Fully tensorial (off-diagonal) media and bends are
deferred — they raise :class:`NotImplementedError`, and ``compute_mode`` routes those to Tidy3D when
it is installed.
"""

from __future__ import annotations

from typing import List, Literal, Sequence

import numpy as np

from fdtdx.constants import c
from fdtdx.core.misc import expand_to_3x3
from fdtdx.core.physics.mode_backend.operator import build_derivative_matrices
from fdtdx.core.physics.mode_backend.solve import solve_modes_diagonal

# Off-diagonal magnitude above which the cross-section is treated as fully tensorial (deferred).
TOL_TENSORIAL = 1e-6


def _diag_components(cross_section: np.ndarray, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened (C-order) xx/yy/zz diagonal components, asserting the tensor is diagonal."""
    tensor = np.asarray(expand_to_3x3(cross_section))  # (3, 3, Nx, Ny)
    off = np.ones((3, 3)) - np.eye(3)
    off_mag = np.max(np.abs(tensor[off.astype(bool)])) if tensor.size else 0.0
    if off_mag > TOL_TENSORIAL:
        raise NotImplementedError(
            "The fdtdmex mode backend supports isotropic / diagonally-anisotropic media only; "
            "the cross-section has significant off-diagonal tensor components. Set "
            "mode_backend='tidy3d' (requires the optional tidy3d dependency) for fully tensorial media."
        )
    xx = np.asarray(tensor[0, 0]).reshape(nx, ny).ravel()
    yy = np.asarray(tensor[1, 1]).reshape(nx, ny).ravel()
    zz = np.asarray(tensor[2, 2]).reshape(nx, ny).ravel()
    return xx, yy, zz


def fdtdmex_mode_computation_wrapper(
    frequency: float,
    permittivity_cross_section,
    coords: Sequence[np.ndarray],
    direction: Literal["+", "-"],
    permeability_cross_section=None,
    target_neff: float | None = None,
    num_modes: int = 10,
    bend_radius: float | None = None,
    bend_axis: int | None = None,
    plane_center: tuple[float, float] | None = None,
    symmetry: tuple[int, int] = (0, 0),
) -> List["ModeTupleType"]:  # noqa: F821 - ModeTupleType imported lazily to avoid an import cycle
    """Compute waveguide modes with the native full-vectorial FD solver.

    Args mirror ``tidy3d_mode_computation_wrapper``: ``coords`` are the two transverse cell-edge
    arrays in micrometres; ``permittivity_cross_section`` / ``permeability_cross_section`` are the
    rotated cross-sections (1/3/9 components x Nx x Ny) in Tidy3D's transverse convention.

    Returns:
        ``list[ModeTupleType]`` sorted by descending ``Re(n_eff)``, in the same convention as the
        Tidy3D wrapper (so ``compute_mode``'s downstream handling is unchanged).
    """
    # Lazy import breaks the modes.py <-> mode_backend import cycle.
    from fdtdx.core.physics.modes import ModeTupleType

    if bend_radius is not None:
        raise NotImplementedError(
            "Bend modes (conformal transform) are not implemented in the fdtdmex mode backend; "
            "use mode_backend='tidy3d' for bends."
        )

    perm = np.asarray(permittivity_cross_section)
    nx, ny = perm.shape[1], perm.shape[2]
    eps_xx, eps_yy, eps_zz = _diag_components(perm, nx, ny)

    mu = np.asarray(permeability_cross_section) if permeability_cross_section is not None else None
    if mu is None or mu.ndim < 3:
        # Uniform (scalar) permeability: broadcast to a flat per-cell array.
        mu_val = 1.0 if mu is None else complex(np.asarray(mu).reshape(-1)[0])
        mu_xx = mu_yy = mu_zz = np.full(nx * ny, mu_val)
    else:
        mu_xx, mu_yy, mu_zz = _diag_components(mu, nx, ny)

    coords_x_m = np.asarray(coords[0], dtype=float) * 1e-6
    coords_y_m = np.asarray(coords[1], dtype=float) * 1e-6
    if len(coords_x_m) != nx + 1 or len(coords_y_m) != ny + 1:
        raise ValueError("coords length must be one more than the cross-section size on each axis")

    k0 = 2.0 * np.pi * frequency / c

    # Shift-invert target: just above the largest real index unless the user specifies one.
    if target_neff is None:
        n_max = float(np.sqrt(np.max(np.real([eps_xx, eps_yy, eps_zz]))))
    else:
        n_max = float(target_neff)
    neff_guess = n_max * (1.0 + 1e-6) + 1e-6

    dmin_pmc = (symmetry[0] == 1, symmetry[1] == 1)
    der_mats = build_derivative_matrices(coords_x_m, coords_y_m, dmin_pmc=dmin_pmc)

    E, H, neff, keff = solve_modes_diagonal(
        eps_xx,
        eps_yy,
        eps_zz,
        mu_xx,
        mu_yy,
        mu_zz,
        der_mats,
        k0=k0,
        num_modes=num_modes,
        neff_guess=neff_guess,
        direction=direction,
    )

    n_solved = E.shape[2]
    modes: list[ModeTupleType] = []
    for i in range(n_solved):
        modes.append(
            ModeTupleType(
                neff=complex(neff[i] + 1j * keff[i]),
                Ex=E[0, :, i].reshape(nx, ny),
                Ey=E[1, :, i].reshape(nx, ny),
                Ez=E[2, :, i].reshape(nx, ny),
                Hx=H[0, :, i].reshape(nx, ny),
                Hy=H[1, :, i].reshape(nx, ny),
                Hz=H[2, :, i].reshape(nx, ny),
            )
        )
    return modes
