"""Physics validation for the native (Tidy3D-free) full-vectorial FD mode solver (WS-B).

The contract is physics, not byte-parity: effective indices are checked against the analytic
symmetric-slab dispersion relation (the lineage Tidy3D / EMpy / MPB all descend from), across
uniform and rectilinear transverse grids and diagonal anisotropy. A dev-time cross-check against
Tidy3D on an identical strip-waveguide cross-section runs through the full ``compute_mode`` pipeline
(rotation + eta0-scaling + Poynting normalisation) and is skipped when Tidy3D is not installed.
"""

import numpy as np
import pytest

from fdtdx.core.physics.mode_backend.operator import build_derivative_matrices
from fdtdx.core.physics.mode_backend.solve import solve_modes_diagonal

pytestmark = pytest.mark.validation

C0 = 299792458.0


def _analytic_slab_te0(n_core: float, n_clad: float, width: float, k0: float) -> float:
    """Fundamental even TE mode n_eff of a symmetric dielectric slab.

    Solves the transcendental dispersion ``kappa*tan(kappa*w/2) = gamma`` for the largest root.
    """

    def disp(neff: np.ndarray) -> np.ndarray:
        kappa = k0 * np.sqrt(n_core**2 - neff**2)
        gamma = k0 * np.sqrt(neff**2 - n_clad**2)
        return kappa * np.tan(kappa * width / 2) - gamma

    ns = np.linspace(n_clad + 1e-6, n_core - 1e-6, 400000)
    vals = disp(ns)
    roots = [
        0.5 * (ns[i] + ns[i + 1])
        for i in range(len(ns) - 1)
        if vals[i] * vals[i + 1] < 0 and abs(vals[i]) < 1e6 and abs(vals[i + 1]) < 1e6
    ]
    return max(roots)


def _solve_slab(edges_x: np.ndarray, eps_x, eps_y, eps_z, k0: float, n_guess: float):
    """Solve a slab invariant in y (Ny = 1) on the given x cell-edges."""
    n = len(edges_x) - 1
    der = build_derivative_matrices(edges_x, np.array([0.0, edges_x[1] - edges_x[0]]), (False, False))
    o = np.ones(n)
    eps_x = eps_x if np.ndim(eps_x) else np.full(n, eps_x)
    E, H, neff, keff = solve_modes_diagonal(eps_x, eps_y, eps_z, o, o, o, der, k0, 6, n_guess, "+")
    return E, neff


def test_slab_te0_uniform_grid():
    lam = 1.55e-6
    k0 = 2 * np.pi / lam
    n_core, n_clad, width = 1.5, 1.0, 1.0e-6
    neff_a = _analytic_slab_te0(n_core, n_clad, width, k0)

    Lx, dx = 6.0e-6, 10e-9
    nx = int(round(Lx / dx))
    edges_x = (np.arange(nx + 1) - nx / 2) * dx
    centers = 0.5 * (edges_x[:-1] + edges_x[1:])
    eps = np.where(np.abs(centers) <= width / 2, n_core**2, n_clad**2)

    E, neff = _solve_slab(edges_x, eps, eps, eps, k0, n_core * 0.999)
    err = float(np.min(np.abs(neff - neff_a)))
    assert err < 1e-3, f"TE0 n_eff error {err:.2e} (analytic {neff_a:.5f})"
    # Fundamental TE slab mode (varies in x, invariant in y) is polarised along y.
    i = int(np.argmin(np.abs(neff - neff_a)))
    comp = [np.linalg.norm(E[k, :, i]) for k in range(3)]
    assert np.argmax(comp) == 1, "TE0 slab mode should be E_y dominant"


def test_slab_te0_rectilinear_grid():
    """A graded (non-uniform) transverse mesh must reach the same analytic n_eff."""
    lam = 1.55e-6
    k0 = 2 * np.pi / lam
    n_core, n_clad, width = 1.5, 1.0, 1.0e-6
    neff_a = _analytic_slab_te0(n_core, n_clad, width, k0)

    Lx = 6.0e-6
    pts, x = [], -Lx / 2
    while x < Lx / 2 - 1e-12:
        pts.append(x)
        x += 5e-9 if abs(x) < 0.8e-6 else 25e-9  # fine near the core interface, coarse outside
    edges_x = np.array(pts + [Lx / 2])
    centers = 0.5 * (edges_x[:-1] + edges_x[1:])
    eps = np.where(np.abs(centers) <= width / 2, n_core**2, n_clad**2)

    _, neff = _solve_slab(edges_x, eps, eps, eps, k0, n_core * 0.999)
    err = float(np.min(np.abs(neff - neff_a)))
    assert err < 1e-3, f"rectilinear TE0 n_eff error {err:.2e}"


def test_slab_te0_diagonal_anisotropy():
    """Only the in-plane-perpendicular component (eps_yy) drives the TE slab mode."""
    lam = 1.55e-6
    k0 = 2 * np.pi / lam
    n_core, n_clad, width = 1.5, 1.0, 1.0e-6
    neff_a = _analytic_slab_te0(n_core, n_clad, width, k0)

    Lx, dx = 6.0e-6, 10e-9
    nx = int(round(Lx / dx))
    edges_x = (np.arange(nx + 1) - nx / 2) * dx
    centers = 0.5 * (edges_x[:-1] + edges_x[1:])
    incore = np.abs(centers) <= width / 2
    eps_x = np.where(incore, 2.10, 1.0)  # distinct from eps_y -> must not change TE0
    eps_y = np.where(incore, n_core**2, 1.0)
    eps_z = np.where(incore, 2.50, 1.0)

    E, neff = _solve_slab(edges_x, eps_x, eps_y, eps_z, k0, n_core * 0.999)
    i = int(np.argmin(np.abs(neff - neff_a)))
    assert abs(neff[i] - neff_a) < 1e-3, f"diagonal-aniso TE0 error {abs(neff[i] - neff_a):.2e}"
    comp = [np.linalg.norm(E[k, :, i]) for k in range(3)]
    assert np.argmax(comp) == 1, "TE0 must be E_y dominant (driven by eps_yy)"


def test_strip_waveguide_matches_tidy3d():
    """Dev-time cross-check: native vs Tidy3D n_eff through the full compute_mode pipeline."""
    pytest.importorskip("tidy3d")
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_platform_name", "cpu")
    from fdtdx.core.physics.modes import compute_mode

    lam = 1.55e-6
    freq = C0 / lam
    res = 20e-9
    nx, ny = 100, 80
    n_core, n_clad = 3.48, 1.44
    cw, ch = 0.5e-6, 0.22e-6
    xs = (np.arange(nx) - nx / 2 + 0.5) * res
    ys = (np.arange(ny) - ny / 2 + 0.5) * res
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    eps = np.where((np.abs(X) <= cw / 2) & (np.abs(Y) <= ch / 2), n_core**2, n_clad**2)
    inv_eps = jnp.asarray((1.0 / eps)[None, :, :, None])

    kw = dict(
        frequency=freq,
        inv_permittivities=inv_eps,
        inv_permeabilities=1.0,
        resolution=res,
        direction="+",
        mode_index=0,
        filter_pol="te",
    )
    _, _, neff_fd = compute_mode(**kw, mode_backend="fdtdmex")
    _, _, neff_t3 = compute_mode(**kw, mode_backend="tidy3d")
    assert abs(complex(neff_fd) - complex(neff_t3)) < 1e-3
