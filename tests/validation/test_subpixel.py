"""Validation for Kottke/Farjadpour subpixel smoothing (WS-C) and its mode-solver coupling.

Checks the smoothed inverse-permittivity tensor against the analytic effective medium (harmonic mean
along the interface normal, arithmetic mean tangentially), that uniform cells stay isotropic, and that
feeding the smoothed tensor to the native mode solver (WS-B) removes the staircase error for an
interface that falls between grid cells.
"""

import numpy as np
import pytest

from fdtdx.core.physics.mode_backend.operator import build_derivative_matrices
from fdtdx.core.physics.mode_backend.solve import solve_modes_diagonal
from fdtdx.core.physics.subpixel import smooth_inverse_permittivity

pytestmark = pytest.mark.validation

C0 = 299792458.0


def test_half_filled_cell_matches_effective_medium():
    """A cell split 50/50 by an x-normal interface -> harmonic eps_xx, arithmetic eps_yy/eps_zz."""
    e1, e2, f = 2.0, 12.0, 20
    fine = np.empty((f, f, f))
    fine[: f // 2] = e1
    fine[f // 2 :] = e2
    T = smooth_inverse_permittivity(fine, f).reshape(3, 3)

    meps = 0.5 * (e1 + e2)
    minveps = 0.5 * (1 / e1 + 1 / e2)
    assert np.isclose(T[0, 0], minveps)  # normal (x): harmonic
    assert np.isclose(T[1, 1], 1 / meps)  # tangential (y): arithmetic
    assert np.isclose(T[2, 2], 1 / meps)  # tangential (z): arithmetic
    off = T - np.diag(np.diag(T))
    assert np.max(np.abs(off)) < 1e-12


def test_uniform_cell_is_isotropic():
    f = 16
    fine = np.full((f, f, f), 7.0)
    T = smooth_inverse_permittivity(fine, f).reshape(3, 3)
    assert np.allclose(np.diag(T), 1 / 7.0)
    assert np.max(np.abs(T - np.diag(np.diag(T)))) < 1e-12


def test_tilted_interface_is_anisotropic():
    """A 45-degree interface in the xy-plane gives equal eps_xx/eps_yy and a non-zero xy term."""
    e1, e2, f = 2.0, 12.0, 24
    xs = np.arange(f)
    Xi, Yi, _ = np.meshgrid(xs, xs, np.arange(f), indexing="ij")
    fine = np.where(Xi + Yi < f, e1, e2).astype(float)
    T = smooth_inverse_permittivity(fine, f).reshape(3, 3)
    assert np.isclose(T[0, 0], T[1, 1], rtol=1e-6)  # symmetric in x/y
    assert abs(T[0, 1]) > 1e-3  # genuine off-diagonal coupling
    assert np.isclose(T[2, 2], T[2, 2])  # z untouched (finite)


def _analytic_slab_te0(width, k0, n_core=1.5, n_clad=1.0):
    def disp(neff):
        ka = k0 * np.sqrt(n_core**2 - neff**2)
        ga = k0 * np.sqrt(neff**2 - n_clad**2)
        return ka * np.tan(ka * width / 2) - ga

    ns = np.linspace(n_clad + 1e-6, n_core - 1e-6, 400000)
    v = disp(ns)
    roots = [
        0.5 * (ns[i] + ns[i + 1])
        for i in range(len(ns) - 1)
        if v[i] * v[i + 1] < 0 and abs(v[i]) < 1e6 and abs(v[i + 1]) < 1e6
    ]
    return max(roots)


def test_smoothing_reduces_mode_staircase_error():
    """Smoothing the slab interface lowers the mode-solver n_eff error vs a staircased grid."""
    lam = 1.55e-6
    k0 = 2 * np.pi / lam
    n_core, n_clad = 1.5, 1.0
    Lx, dx, fac = 6e-6, 40e-9, 20
    nx = int(round(Lx / dx))
    edges = (np.arange(nx + 1) - nx / 2) * dx
    centers = 0.5 * (edges[:-1] + edges[1:])
    fcent = (np.arange(nx * fac) - nx * fac / 2 + 0.5) * (dx / fac)
    o = np.ones(nx)
    der = build_derivative_matrices(edges, np.array([0.0, dx]), (False, False))

    stair, smooth = [], []
    for w in (1.000e-6, 1.013e-6, 1.027e-6):  # interface falls between coarse cells
        na = _analytic_slab_te0(w, k0)
        eps_s = np.where(np.abs(centers) <= w / 2, n_core**2, n_clad**2)
        _, _, nst, _ = solve_modes_diagonal(eps_s, eps_s, eps_s, o, o, o, der, k0, 4, n_core * 0.999, "+")
        stair.append(float(np.min(np.abs(nst - na))))

        fine = np.broadcast_to(
            np.where(np.abs(fcent) <= w / 2, n_core**2, n_clad**2)[:, None, None], (nx * fac, fac, fac)
        )
        T = smooth_inverse_permittivity(np.ascontiguousarray(fine), fac)
        ex, ey, ez = 1 / T[0].ravel(), 1 / T[4].ravel(), 1 / T[8].ravel()
        _, _, nsm, _ = solve_modes_diagonal(ex, ey, ez, o, o, o, der, k0, 4, n_core * 0.999, "+")
        smooth.append(float(np.min(np.abs(nsm - na))))

    assert np.mean(smooth) < np.mean(stair) / 3, (
        f"smoothing should cut staircase error >3x: staircase {np.mean(stair):.2e}, smoothed {np.mean(smooth):.2e}"
    )
