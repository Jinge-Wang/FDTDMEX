"""Bend modes: the conformal map, its sign convention, and a cylindrical-Bessel reference.

The reference at the bottom of this file is the *exact continuum* answer to the same problem the
finite-difference solver discretises - a 1-D (2-D geometry) guide bent with radius R, in the same
window and with the same PEC walls - written with real-order Bessel functions. It is what pins the
transform itself rather than only its limits.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import jv, yv

from fdtdx.core.physics.mode_backend.bend import (
    BEND_FORMS,
    bend_radius_ratio,
    transform_cross_section,
)
from fdtdx.core.physics.modes import compute_mode

C_LIGHT = 299792458.0


@pytest.fixture(autouse=True)
def _float64():
    """The mode solve is complex128 internally; without x64 the *returned* index is complex64."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


# The 2-D effective-index ring of cases/fdtdx/ring2d_thermal_tuning_tidy3d.py.
N_CORE = 2.44834
N_BG = 1.444
WIDTH = 0.5e-6
LAMBDA0 = 1.55e-6


#: The two grids every extrapolated statement below is made on. Both represent the 0.5 um core
#: exactly (20 and 40 cells); a grid that does not - 20 nm puts 26 cells in it - solves a different
#: guide, and extrapolating across the pair is then meaningless.
WINDOW = 3.0e-6
COARSE, FINE = 25e-9, 12.5e-9


def _ring_cross_section(resolution: float, window: float = WINDOW):
    """Inverse permittivity ``(1, 1, ny, 2)``: propagation along x, radial y, z invariant."""
    ny = round(window / resolution)
    ys = (np.arange(ny) - ny / 2 + 0.5) * resolution
    eps = np.where(np.abs(ys) <= WIDTH / 2, N_CORE**2, N_BG**2)
    return jnp.asarray((1.0 / eps)[None, None, :, None] * np.ones((1, 1, ny, 2)))


def _neff(resolution, bend_radius=None, window=WINDOW, pol="te"):
    kwargs = {} if bend_radius is None else dict(bend_radius=bend_radius, bend_axis=2)
    _, _, neff = compute_mode(
        frequency=C_LIGHT / LAMBDA0,
        inv_permittivities=_ring_cross_section(resolution, window),
        inv_permeabilities=1.0,
        resolution=resolution,
        filter_pol=pol,
        dtype=jnp.float64,
        **kwargs,
    )
    return complex(neff)


def _extrapolated_shift(bend_radius, pol="te"):
    """The bend shift with the first-order cell-size term extrapolated away (two grids)."""
    coarse = _neff(COARSE, bend_radius, pol=pol).real - _neff(COARSE, pol=pol).real
    fine = _neff(FINE, bend_radius, pol=pol).real - _neff(FINE, pol=pol).real
    return 2 * fine - coarse


class TestBendTransform:
    """The map itself, without an eigen-solve."""

    def test_ratio_is_one_at_the_plane_centre(self):
        edges = np.linspace(-1.0, 1.0, 5)
        ratio = bend_radius_ratio(edges, 10.0, 0.0)
        assert ratio[1] == pytest.approx(1.0 - 0.25 / 10.0)
        assert np.mean(ratio) == pytest.approx(1.0)

    def test_negative_radius_mirrors_the_transform(self):
        edges = np.linspace(-1.0, 1.0, 9)
        forward = np.asarray(bend_radius_ratio(edges, 7.0, 0.0))
        backward = np.asarray(bend_radius_ratio(edges, -7.0, 0.0))
        np.testing.assert_allclose(forward, backward[::-1], rtol=0, atol=1e-15)

    def test_tensor_form_scales_transverse_up_and_propagation_down(self):
        edges = np.linspace(-1.0, 1.0, 5)
        eps = np.full((1, 4, 3), 4.0 + 0j)
        out = transform_cross_section(eps, 1.0, (edges, np.linspace(0, 1, 4)), 5.0, 1, (0.0, 0.5))
        ratio = np.asarray(bend_radius_ratio(edges, 5.0, 0.0))
        np.testing.assert_allclose(out.permittivity[0, :, 0], 4.0 * ratio, rtol=1e-14)
        np.testing.assert_allclose(out.permittivity[1, :, 0], 4.0 * ratio, rtol=1e-14)
        np.testing.assert_allclose(out.permittivity[2, :, 0], 4.0 / ratio, rtol=1e-14)
        # A scalar permeability comes back as the same three components, scaled the same way.
        np.testing.assert_allclose(out.permeability[2, :, 0], 1.0 / ratio, rtol=1e-14)

    def test_scalar_forms_leave_the_permeability_alone(self):
        edges = np.linspace(-1.0, 1.0, 5)
        eps = np.full((1, 4, 3), 4.0 + 0j)
        for form in ("conformal", "exponential"):
            out = transform_cross_section(eps, 1.0, (edges, np.linspace(0, 1, 4)), 5.0, 1, (0.0, 0.5), form=form)
            np.testing.assert_allclose(np.asarray(out.permeability), 1.0, rtol=0, atol=1e-15)

    def test_conformal_form_relabels_the_radial_edges(self):
        edges = np.linspace(-1.0, 1.0, 5)
        out = transform_cross_section(
            np.full((1, 4, 3), 4.0 + 0j), 1.0, (edges, np.linspace(0, 1, 4)), 5.0, 1, (0.0, 0.5), form="conformal"
        )
        expected = 5.0 * np.log(1.0 + edges / 5.0)
        np.testing.assert_allclose(out.coords[0], expected, rtol=1e-14)
        # cells stay cells: the relabelling is monotone and keeps the count
        assert np.all(np.diff(out.coords[0]) > 0)
        assert len(out.coords[0]) == len(edges)

    def test_the_bend_axis_is_the_one_normal_to_the_bend_plane(self):
        # bend_axis=0 leaves axis 0 uniform and grades axis 1, and the other way round.
        eps = np.full((1, 4, 4), 4.0 + 0j)
        coords = (np.linspace(-1.0, 1.0, 5), np.linspace(-1.0, 1.0, 5))
        graded_y = np.asarray(transform_cross_section(eps, 1.0, coords, 5.0, 0, (0.0, 0.0)).permittivity)
        graded_x = np.asarray(transform_cross_section(eps, 1.0, coords, 5.0, 1, (0.0, 0.0)).permittivity)
        assert np.allclose(graded_y[0, 0, :], graded_y[0, 3, :])  # uniform along axis 0
        assert not np.allclose(graded_y[0, :, 0], graded_y[0, :, 3])
        assert np.allclose(graded_x[0, :, 0], graded_x[0, :, 3])  # uniform along axis 1
        assert not np.allclose(graded_x[0, 0, :], graded_x[0, 3, :])

    def test_refusals(self):
        edges = np.linspace(-1.0, 1.0, 5)
        coords = (edges, np.linspace(0, 1, 4))
        eps = np.full((1, 4, 3), 4.0 + 0j)
        with pytest.raises(ValueError, match="non-zero"):
            transform_cross_section(eps, 1.0, coords, 0.0, 1, (0.0, 0.5))
        with pytest.raises(ValueError, match="centre of curvature"):
            transform_cross_section(eps, 1.0, coords, 0.5, 1, (0.0, 0.5))
        with pytest.raises(ValueError, match="bend form"):
            transform_cross_section(eps, 1.0, coords, 5.0, 1, (0.0, 0.5), form="spiral")
        with pytest.raises(ValueError, match="bend_axis"):
            transform_cross_section(eps, 1.0, coords, 5.0, 2, (0.0, 0.5))
        with pytest.raises(NotImplementedError, match="1 or 3 components"):
            transform_cross_section(np.full((9, 4, 3), 4.0 + 0j), 1.0, coords, 5.0, 1, (0.0, 0.5))

    def test_every_form_is_a_no_op_in_the_straight_limit(self):
        edges = np.linspace(-1.0, 1.0, 9)
        eps = np.full((1, 8, 2), 6.0 + 0j)
        for form in BEND_FORMS:
            out = transform_cross_section(eps, 1.0, (edges, np.linspace(0, 1, 3)), 1e9, 1, (0.0, 0.5), form=form)
            np.testing.assert_allclose(np.asarray(out.permittivity), 6.0, rtol=1e-8)


class TestBendThroughComputeMode:
    """The front end: ``bend_radius`` now reaches the native backend."""

    def test_bend_runs_on_the_native_backend_and_leaves_neff_real(self):
        # Without tidy3d installed this used to raise; a PEC-walled window keeps the spectrum real,
        # so the radiation loss of the bend does *not* show up as Im(n_eff).
        neff = _neff(20e-9, bend_radius=5e-6)
        assert neff.real > N_BG
        assert abs(neff.imag) < 1e-12

    def test_a_large_radius_reproduces_the_straight_solve(self):
        straight = _neff(25e-9)
        bent = _neff(25e-9, bend_radius=1.0)  # 1 m
        # Not zero: the bend's first-order discretisation term is n_eff * 0.3 dx / R, here 1.6e-8.
        assert abs(bent.real - straight.real) < 1e-7

    def test_the_bend_raises_the_effective_index(self):
        assert _neff(10e-9, bend_radius=5e-6).real > _neff(10e-9).real

    def test_the_extrapolated_shift_grows_as_one_over_radius_squared(self):
        # On a single grid the bend's first-order discretisation term can outweigh the physics at
        # large radius (at 10 nm cells and R = 20 um the raw shift is still negative), so the
        # scaling law is asserted on the two-grid extrapolation, which is how a bent solve is meant
        # to be read.
        assert _extrapolated_shift(1e-5) / _extrapolated_shift(2e-5) == pytest.approx(4.0, rel=0.05)

    def test_the_discretisation_error_of_a_bend_is_odd_in_the_radius(self):
        """+R and -R differ by twice the artefact, and it halves with the cell size."""
        straight_coarse, straight_fine = _neff(25e-9).real, _neff(12.5e-9).real
        coarse = [_neff(25e-9, sign * 1e-5).real - straight_coarse for sign in (1, -1)]
        fine = [_neff(12.5e-9, sign * 1e-5).real - straight_fine for sign in (1, -1)]
        artefact_coarse = 0.5 * (coarse[1] - coarse[0])
        artefact_fine = 0.5 * (fine[1] - fine[0])
        assert artefact_coarse / artefact_fine == pytest.approx(2.0, rel=0.05)
        # ... so on a mirror-symmetric cross-section the mean over the two signs also removes it
        assert 0.5 * (coarse[0] + coarse[1]) == pytest.approx(0.5 * (fine[0] + fine[1]), rel=5e-3)


class TestBendAgainstBesselReference:
    """The transform against an exact cylindrical solution of the same guide."""

    @pytest.mark.parametrize("pol,slab", [("te", "tm"), ("tm", "te")])
    def test_extrapolated_bend_shift_matches_the_bessel_reference(self, pol, slab):
        radius = 1e-5
        exact = bessel_bent_neff(2 * np.pi / LAMBDA0, radius, WIDTH, N_CORE, N_BG, WINDOW / 2, slab)
        exact_shift = exact - bessel_straight_neff(2 * np.pi / LAMBDA0, WIDTH, N_CORE, N_BG, slab)
        assert _extrapolated_shift(radius, pol) == pytest.approx(exact_shift, rel=5e-3)

    def test_the_literal_exponential_form_is_wrong_by_tens_of_percent(self):
        """The quoted ``n exp(x/R)`` is only first-order faithful; the shift is second order."""
        import functools

        import fdtdx.core.physics.modes as modes_module

        radius = 1e-5
        exact_shift = bessel_bent_neff(
            2 * np.pi / LAMBDA0, radius, WIDTH, N_CORE, N_BG, WINDOW / 2, "tm"
        ) - bessel_straight_neff(2 * np.pi / LAMBDA0, WIDTH, N_CORE, N_BG, "tm")
        original = modes_module.transform_cross_section
        modes_module.transform_cross_section = functools.partial(original, form="exponential")
        try:
            extrapolated = _extrapolated_shift(radius)
        finally:
            modes_module.transform_cross_section = original
        assert extrapolated / exact_shift > 1.5


class TestBendIsDifferentiable:
    """The transform is a JAX function of the permittivity and of the radius."""

    def test_gradient_flows_to_the_radius_and_to_the_permittivity(self):
        if not jax.config.jax_enable_x64:
            pytest.skip("the differentiable mode path needs jax_enable_x64")
        edges = np.linspace(-1.0, 1.0, 5)
        coords = (edges, np.linspace(0.0, 1.0, 3))

        def total(eps, radius):
            out = transform_cross_section(eps, 1.0, coords, radius, 1, (0.0, 0.5))
            return jnp.sum(jnp.abs(out.permittivity) ** 2)

        eps = jnp.full((1, 4, 2), 4.0 + 0j)
        grad_radius = jax.grad(total, argnums=1)(eps, 5.0)
        step = 1e-4
        finite = (total(eps, 5.0 + step) - total(eps, 5.0 - step)) / (2 * step)
        assert float(grad_radius) == pytest.approx(float(finite), rel=1e-6)

        # The permittivity gradient is taken on a holomorphic (complex-valued) objective, because
        # the transformed permittivity is complex and the scale factors are real.
        def summed(e):
            return jnp.sum(transform_cross_section(e, 1.0, coords, 5.0, 1, (0.0, 0.5)).permittivity)

        grad_eps = jax.grad(summed, holomorphic=True)(eps)
        ratio = np.asarray(bend_radius_ratio(edges, 5.0, 0.0))
        # d(sum of the three scaled components)/d eps_cell = 2 r/R + R/r for the tensor form
        np.testing.assert_allclose(np.asarray(grad_eps)[0, :, 0].real, 2 * ratio + 1 / ratio, rtol=1e-12)


# ---------------------------------------------------------------------------------------------
# The reference: a bent 1-D guide solved exactly with real-order Bessel functions.
# ---------------------------------------------------------------------------------------------
def _bessel_pair(nu, k, r):
    x = k * r
    return (
        jv(nu, x),
        yv(nu, x),
        k * 0.5 * (jv(nu - 1, x) - jv(nu + 1, x)),
        k * 0.5 * (yv(nu - 1, x) - yv(nu + 1, x)),
    )


@np.errstate(all="ignore")
def _wall_residual(nu, k0, radii, indices, polarization):
    """Shoot the out-of-plane field outward from the inner PEC wall; return the outer wall residual.

    ``polarization="te"`` is E out of plane (Dirichlet at a PEC wall, E and dE/dr continuous);
    ``"tm"`` is H out of plane (Neumann at a PEC wall, H and ``(1/n^2) dH/dr`` continuous).
    """
    transverse_magnetic = polarization == "tm"
    weight = (lambda n: 1.0 / n**2) if transverse_magnetic else (lambda n: 1.0)
    j, y, dj, dy = _bessel_pair(nu, k0 * indices[0], radii[0])
    det = j * dy - y * dj
    if not np.isfinite(det) or det == 0.0:
        return np.nan
    a, b = (dy / det, -dj / det) if transverse_magnetic else (-y / det, j / det)
    for m in range(len(indices) - 1):
        r = radii[m + 1]
        j, y, dj, dy = _bessel_pair(nu, k0 * indices[m], r)
        value = a * j + b * y
        slope = weight(indices[m]) * (a * dj + b * dy) / weight(indices[m + 1])
        j2, y2, dj2, dy2 = _bessel_pair(nu, k0 * indices[m + 1], r)
        det = j2 * dy2 - y2 * dj2
        a, b = (value * dy2 - slope * y2) / det, (slope * j2 - value * dj2) / det
    j, y, dj, dy = _bessel_pair(nu, k0 * indices[-1], radii[-1])
    scale = max(abs(a), abs(b), 1e-300)
    return ((a * dj + b * dy) if transverse_magnetic else (a * j + b * y)) / scale


def bessel_bent_neff(k0, radius, width, n_core, n_bg, half_window, polarization="tm", n_scan=1500):
    """Effective index at the reference radius of the bent guide's fundamental mode."""
    radii = np.array([radius - half_window, radius - width / 2, radius + width / 2, radius + half_window])
    indices = np.array([n_bg, n_core, n_bg])
    grid = np.linspace(k0 * radius * n_bg * 1.001, k0 * radius * n_core * 0.999, n_scan)
    values = np.array([_wall_residual(nu, k0, radii, indices, polarization) for nu in grid])
    roots = [
        brentq(_wall_residual, grid[i], grid[i + 1], args=(k0, radii, indices, polarization), xtol=1e-13)
        for i in range(len(grid) - 1)
        if np.isfinite(values[i]) and np.isfinite(values[i + 1]) and values[i] * values[i + 1] < 0
    ]
    if not roots:
        raise RuntimeError("no bent-guide root found")
    return max(roots) / (k0 * radius)


def bessel_straight_neff(k0, width, n_core, n_bg, polarization="tm"):
    """The symmetric-slab root of the same guide, i.e. the ``R -> infinity`` limit."""
    ratio = (n_core**2 / n_bg**2) if polarization == "tm" else 1.0

    def residual(neff):
        kappa = k0 * np.sqrt(n_core**2 - neff**2)
        gamma = k0 * np.sqrt(neff**2 - n_bg**2)
        return kappa * np.sin(kappa * width / 2) - ratio * gamma * np.cos(kappa * width / 2)

    grid = np.linspace(n_bg + 1e-12, n_core - 1e-12, 4000)
    values = np.array([residual(v) for v in grid])
    roots = [
        brentq(residual, grid[i], grid[i + 1], xtol=1e-15)
        for i in range(len(grid) - 1)
        if values[i] * values[i + 1] < 0
    ]
    return max(roots)


class TestBesselReference:
    """The reference itself, checked where it has an independent answer."""

    def test_the_large_radius_limit_is_the_slab_root(self):
        # Not taken further than 100 um: the Bessel order is k0 n R, and the evanescent-region
        # values overflow float64 long before the limit is reached numerically.
        k0 = 2 * np.pi / LAMBDA0
        for pol in ("te", "tm"):
            straight = bessel_straight_neff(k0, WIDTH, N_CORE, N_BG, pol)
            bent = bessel_bent_neff(k0, 1e-4, WIDTH, N_CORE, N_BG, 1.5e-6, pol)
            assert 0 < bent - straight < 3e-5

    def test_the_bend_shift_is_second_order_in_one_over_radius(self):
        k0 = 2 * np.pi / LAMBDA0
        straight = bessel_straight_neff(k0, WIDTH, N_CORE, N_BG, "tm")
        shifts = [bessel_bent_neff(k0, r, WIDTH, N_CORE, N_BG, 1.5e-6, "tm") - straight for r in (4e-5, 2e-5)]
        assert shifts[1] / shifts[0] == pytest.approx(4.0, rel=0.05)
