"""Kottke / Farjadpour subpixel smoothing of a permittivity grid (Phase 4 Track A, WS-C).

Smoothing an isotropic material across a tilted interface produces an *anisotropic* effective tensor:
the inverse permittivity continuous across the interface is the harmonic mean (normal direction), the
permittivity continuous along it is the arithmetic mean (tangential directions). Kottke's result blends
the two through the interface-normal projector ``P = n n^T``:

    chi1inv = P (minveps - 1/meps) + I (1/meps)

with ``meps = <eps>`` and ``minveps = <1/eps>`` the cell averages and ``n`` the unit interface normal.

This module computes that tensor from a **supersampled** scalar permittivity raster (a fine grid at an
integer ``factor`` of the simulation resolution): each coarse cell aggregates its ``factor**ndim`` fine
subcells, so the subpixel fill fraction and normal are recovered without needing the original CSG.
Output is the ``(9, Nx, Ny, Nz)`` inverse-permittivity layout the FDTD engine and the mode solver
consume. Uniform cells collapse exactly to the isotropic ``I/eps``.

Algorithm references (independent implementation, no GPL code copied; see ``CLAUDE.md`` on the MEEP
``anisotropic_averaging.cpp`` reference):
- Farjadpour et al., "Improving accuracy by subpixel smoothing in FDTD," Opt. Lett. 31, 2972 (2006).
- Kottke, Farjadpour, Joannopoulos, Johnson, Phys. Rev. E 77, 036611 (2008).
"""

from __future__ import annotations

import numpy as np

# Below this relative spread of eps within a cell, treat it as uniform (no interface -> isotropic).
_UNIFORM_REL_TOL = 1e-9


def _block_reduce_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Mean over non-overlapping ``factor``-sized blocks along every axis."""
    shape = []
    for n in arr.shape:
        if n % factor != 0:
            raise ValueError(f"axis length {n} is not divisible by factor {factor}")
        shape.extend([n // factor, factor])
    reshaped = arr.reshape(shape)
    # Average over the per-axis 'factor' sub-axes (the odd positions 1, 3, 5, ...).
    reduce_axes = tuple(range(1, 2 * arr.ndim, 2))
    return reshaped.mean(axis=reduce_axes)


def _fine_gradient_normals(eps_fine: np.ndarray, factor: int) -> np.ndarray:
    """Unit interface-normal field (per coarse cell) from the *fine*-grid permittivity gradient.

    The gradient is taken on the fine raster (so the interface is resolved within each coarse cell),
    its components are block-averaged to the coarse grid, and the result is normalised. Axes of fine
    length 1 contribute a zero component; zero-gradient (uniform) cells return a zero vector.

    Returns an array of shape ``(3, *coarse_shape)``.
    """
    comps = []
    for axis in range(eps_fine.ndim):
        if eps_fine.shape[axis] > 1:
            g = np.gradient(eps_fine, axis=axis)
        else:
            g = np.zeros_like(eps_fine)
        comps.append(_block_reduce_mean(g, factor))
    while len(comps) < 3:
        comps.append(np.zeros_like(comps[0]))
    g = np.stack(comps, axis=0)  # (3, *coarse)
    norm = np.sqrt(np.sum(g**2, axis=0))
    norm_safe = np.where(norm > 0, norm, 1.0)
    return g / norm_safe


def smooth_inverse_permittivity(eps_fine: np.ndarray, factor: int) -> np.ndarray:
    """Subpixel-smooth a fine scalar permittivity raster into a coarse 9-tensor inverse permittivity.

    Args:
        eps_fine: real scalar permittivity sampled at ``factor`` x the target (coarse) resolution.
            Shape ``(Nx*f, Ny*f, Nz*f)``; any axis may be ``factor`` (a single coarse cell) or, if the
            coarse extent is 1, the fine axis length equals ``factor``.
        factor: integer supersampling factor (>= 1). ``factor == 1`` returns the plain isotropic
            ``1/eps`` with no smoothing.

    Returns:
        ``(9, Nx, Ny, Nz)`` inverse-permittivity tensor (row-major xx,xy,xz,yx,...). Interface cells
        carry the anisotropic Kottke tensor; uniform cells are exactly isotropic.
    """
    eps_fine = np.asarray(eps_fine, dtype=float)
    if eps_fine.ndim != 3:
        raise ValueError(f"eps_fine must be 3-D (Nx*f, Ny*f, Nz*f), got shape {eps_fine.shape}")
    if factor < 1:
        raise ValueError("factor must be >= 1")

    coarse_shape = tuple(n // factor for n in eps_fine.shape)

    meps = _block_reduce_mean(eps_fine, factor)  # <eps>
    minveps = _block_reduce_mean(1.0 / eps_fine, factor)  # <1/eps>
    # Per-cell spread to detect interfaces (uniform cells get isotropic treatment).
    mean_sq = _block_reduce_mean(eps_fine**2, factor)
    var = np.clip(mean_sq - meps**2, 0.0, None)
    interface = var > (_UNIFORM_REL_TOL * meps) ** 2

    normals = _fine_gradient_normals(eps_fine, factor)  # (3, *coarse)

    inv_meps = 1.0 / meps
    tensor = np.zeros((3, 3, *coarse_shape), dtype=float)
    delta = minveps - inv_meps  # harmonic-minus-arithmetic, the anisotropic part

    for a in range(3):
        for b in range(3):
            P_ab = normals[a] * normals[b]
            iso = inv_meps if a == b else np.zeros_like(inv_meps)
            smoothed = P_ab * delta + iso
            # uniform cells -> exact isotropic 1/eps on the diagonal, 0 off-diagonal
            iso_only = inv_meps if a == b else np.zeros_like(inv_meps)
            tensor[a, b] = np.where(interface, smoothed, iso_only)

    return tensor.reshape(9, *coarse_shape)


def smooth_cross_section_2d(eps_fine_2d: np.ndarray, factor: int) -> np.ndarray:
    """Subpixel-smooth a 2-D cross-section; returns ``(9, Nx, Ny)`` inverse-permittivity.

    Convenience wrapper for the mode solver (WS-B): the transverse plane is treated as a 3-D raster
    with a singleton third axis, so in-plane interfaces give the in-plane anisotropic tensor.
    """
    eps_fine_2d = np.asarray(eps_fine_2d, dtype=float)
    if eps_fine_2d.ndim != 2:
        raise ValueError(f"eps_fine_2d must be 2-D, got shape {eps_fine_2d.shape}")
    tensor = smooth_inverse_permittivity(eps_fine_2d[:, :, None], factor)
    return tensor.reshape(9, tensor.shape[1], tensor.shape[2])
