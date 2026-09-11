"""The mode-solver gradient: zero before, finite differences after.

Every check here is a finite difference of the same solver the gradient claims to differentiate, plus
one comparison against the analytic dispersion relation of a uniform slab, which is the only
configuration in this file whose answer is known in closed form.

The whole file needs float64: a central finite difference of an effective index around 3.4 has no
signal at all in complex64.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.core.physics.mode_adjoint import (
    ModeSolveSettings,
    mode_neff,
    mode_neff_parts,
    mode_sensitivity,
    mode_solve,
    track_mode,
    tracked_mode_index,
)
from fdtdx.core.physics.modes import compute_mode

C0 = 299792458.0
LAM = 1.55e-6
FREQ = C0 / LAM
N_SI, N_SIO2 = 3.48, 1.55
N_TIN, K_TIN = 3.1477, 5.8429
#: TiN at 1.55 um in the e^{+i k0 n z} convention the solver returns, so Im(n_eff) > 0 is loss.
EPS_TIN = (N_TIN**2 - K_TIN**2) + 2j * N_TIN * K_TIN


@pytest.fixture
def float64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _strip_cross_section(scale: float = 1.0) -> jnp.ndarray:
    """The 40 x 30 cell strip at 40 nm the zero-gradient measurement of D1 was made on.

    Core 12.0 in a background of 2.25, propagation along the first axis. Keep these two numbers as
    they are: the recorded finite difference of 2.0400 belongs to this cross-section.
    """
    eps = np.full((40, 30), 2.25)
    eps[14:26, 10:16] = 12.0
    return jnp.asarray(eps * scale)[None, None, :, :]


def _tensor_cross_section(theta_deg: float = 25.0, scale: float = 1.0) -> jnp.ndarray:
    """The same strip on the 9-component tier, its core a uniaxial tensor rotated in the plane.

    Propagation along the first axis, so the solver-frame transverse pair is the physical (y, z)
    one and the rotation writes a genuine off-diagonal the diagonal tier cannot represent.
    """
    theta = np.deg2rad(theta_deg)
    n_o2, n_e2 = 2.25, 12.0
    core = np.zeros((40, 30), dtype=bool)
    core[14:26, 10:16] = True
    tensor = np.zeros((3, 3, 40, 30))
    for axis in range(3):
        tensor[axis, axis] = n_o2
    tensor[1, 1] = np.where(core, n_e2 * np.cos(theta) ** 2 + n_o2 * np.sin(theta) ** 2, n_o2)
    tensor[2, 2] = np.where(core, n_e2 * np.sin(theta) ** 2 + n_o2 * np.cos(theta) ** 2, n_o2)
    off = np.where(core, (n_e2 - n_o2) * np.sin(theta) * np.cos(theta), 0.0)
    tensor[1, 2] = off
    tensor[2, 1] = off
    tensor[0, 0] = np.where(core, n_e2, n_o2)
    return jnp.asarray(tensor.reshape(9, 40, 30) * scale)[:, None, :, :]


def _settings(**kwargs) -> ModeSolveSettings:
    base = {"frequency": FREQ, "resolution": 40e-9, "mode_index": 0}
    base.update(kwargs)
    return ModeSolveSettings.create(**base)


# ------------------------------------------------------------------------------------------------
# the hole this module closes
# ------------------------------------------------------------------------------------------------


def test_compute_mode_returns_a_silently_zero_gradient(float64):
    """The state of affairs this module exists to fix: jax.grad through compute_mode is zero."""

    def neff_of(scale):
        _, _, beta = compute_mode(
            frequency=FREQ,
            inv_permittivities=1.0 / _strip_cross_section(scale),
            inv_permeabilities=1.0,
            resolution=40e-9,
            direction="+",
            mode_index=0,
            dtype=jnp.float64,
        )
        return jnp.real(beta)

    grad = float(jax.grad(neff_of)(1.0))
    step = 1e-3
    finite_difference = float((neff_of(1.0 + step) - neff_of(1.0 - step)) / (2 * step))
    assert grad == 0.0
    assert finite_difference == pytest.approx(2.0400, rel=1e-3)


def test_mode_neff_reproduces_that_finite_difference(float64):
    """The same derivative, now delivered by jax.grad through the custom_vjp."""
    settings = _settings()

    def neff_of(scale):
        return mode_neff_parts(_strip_cross_section() * scale, settings)[0]

    grad = float(jax.grad(neff_of)(1.0))
    step = 1e-3
    finite_difference = float((neff_of(1.0 + step) - neff_of(1.0 - step)) / (2 * step))
    assert finite_difference == pytest.approx(2.0400, rel=1e-3)
    assert grad == pytest.approx(finite_difference, rel=1e-5)


def test_mode_neff_is_the_complex_index_compute_mode_returns(float64):
    settings = _settings()
    eps = _strip_cross_section()
    _, _, beta = compute_mode(
        frequency=FREQ,
        inv_permittivities=1.0 / eps,
        inv_permeabilities=1.0,
        resolution=40e-9,
        direction="+",
        mode_index=0,
        dtype=jnp.float64,
    )
    assert complex(mode_neff(eps, settings)) == pytest.approx(complex(beta), rel=1e-12)


# ------------------------------------------------------------------------------------------------
# (a) the uniform slab, against its analytic dispersion relation
# ------------------------------------------------------------------------------------------------


def _slab_te0(n_core: float, thickness: float = 1e-6, n_clad: float = N_SIO2, lam: float = LAM) -> float:
    """Effective index of the fundamental even TE mode of a symmetric slab, by bisection.

    Solves ``kappa tan(kappa d / 2) = gamma`` with ``kappa = k0 sqrt(n_core^2 - neff^2)`` and
    ``gamma = k0 sqrt(neff^2 - n_clad^2)`` on the first branch, ``kappa d / 2 < pi / 2``.
    """
    k0 = 2 * np.pi / lam

    def residual(neff: float) -> float:
        kappa = k0 * np.sqrt(n_core**2 - neff**2)
        gamma = k0 * np.sqrt(neff**2 - n_clad**2)
        return kappa * np.tan(kappa * thickness / 2) - gamma

    lo = max(n_clad + 1e-12, np.sqrt(n_core**2 - (np.pi / (k0 * thickness)) ** 2) + 1e-12)
    hi = n_core - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(lo) * residual(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _slab_cross_section(n_core: float, cell: float, nx: int = 4, height: float = 4e-6, thickness: float = 1e-6):
    """A slab layered along the second transverse axis, uniform along the first; propagation along z."""
    ny = round(height / cell)
    n_core_cells = round(thickness / cell)
    low = (ny - n_core_cells) // 2
    eps = np.full((nx, ny), N_SIO2**2)
    eps[:, low : low + n_core_cells] = n_core**2
    core_mask = np.zeros_like(eps)
    core_mask[:, low : low + n_core_cells] = 1.0
    return jnp.asarray(eps)[None, :, :, None], core_mask[None, :, :, None]


def test_uniform_slab_gradient_matches_the_analytic_dispersion_relation(float64):
    """d n_eff / d n_core of a 1 um Si slab, adjoint versus the analytic slab dispersion relation."""
    cell = 25e-9
    eps, core_mask = _slab_cross_section(N_SI, cell)
    settings = _settings(resolution=cell)

    neff, sensitivity = mode_sensitivity(eps, settings)
    analytic_neff = _slab_te0(N_SI)
    assert complex(neff).real == pytest.approx(analytic_neff, abs=5e-4)

    # A uniform core index change n -> n + t is a permittivity change 2 n over the core cells.
    direction = jnp.asarray(2 * N_SI * core_mask)
    adjoint = float(jnp.real(jnp.sum(direction * sensitivity)))

    step = 1e-4
    plus = float(mode_neff_parts(eps + step * direction, settings)[0])
    minus = float(mode_neff_parts(eps - step * direction, settings)[0])
    finite_difference = (plus - minus) / (2 * step)

    analytic_slope = (_slab_te0(N_SI + 1e-6) - _slab_te0(N_SI - 1e-6)) / 2e-6
    assert adjoint == pytest.approx(finite_difference, rel=1e-6)
    assert adjoint == pytest.approx(analytic_slope, rel=1e-3)


# ------------------------------------------------------------------------------------------------
# (b) the phase-shifter cross-section, perturbation confined to the core
# ------------------------------------------------------------------------------------------------


def _phase_shifter_cross_section(cell_um: float, window=(14.0, 4.0), w_metal: float = 4.0):
    """The reference device of Jokisch et al. 2024, section 2: Si guide, SiO2 cladding, TiN heater.

    Origin at the centre of the waveguide, ``x`` across and ``y`` up, both in micrometres.
    """
    nx = round(window[0] / cell_um)
    ny = round(window[1] / cell_um)
    x = -window[0] / 2 + (np.arange(nx) + 0.5) * cell_um
    y = -window[1] / 2 + (np.arange(ny) + 0.5) * cell_um
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    eps = np.full((nx, ny), N_SIO2**2, dtype=np.complex128)
    core = (np.abs(grid_x) < 5.0) & (np.abs(grid_y) < 0.5)
    eps[core] = N_SI**2
    metal = (grid_x >= -5.0) & (grid_x <= -5.0 + w_metal) & (grid_y >= 0.5) & (grid_y <= 0.75)
    eps[metal] = EPS_TIN
    return jnp.asarray(eps)[None, :, :, None], core.astype(np.float64)[None, :, :, None]


def _core_perturbation_check(cell_um: float, window, fd_step: float = 1e-4):
    """Adjoint versus a central finite difference for a uniform core index change."""
    eps, core = _phase_shifter_cross_section(cell_um, window=window)
    settings = _settings(resolution=cell_um * 1e-6)
    neff, sensitivity = mode_sensitivity(eps, settings)
    direction = jnp.asarray(2 * N_SI * core)
    adjoint = complex(jnp.sum(direction * sensitivity))

    def parts(t):
        real, imaginary = mode_neff_parts(eps + t * direction, settings)
        return complex(float(real), float(imaginary))

    finite_difference = (parts(fd_step) - parts(-fd_step)) / (2 * fd_step)
    return complex(neff), adjoint, finite_difference


def test_phase_shifter_core_perturbation_matches_finite_differences(float64):
    """20 nm cells, perturbation confined to the silicon core, on a reduced window for speed.

    The window is 6 x 2.4 um instead of the paper's 14 x 4 um so the eigen-solve stays inside a unit
    test's budget; the paper-sized run at the same cell size is the opt-in test below and its
    numbers are in the D2 report.
    """
    neff, adjoint, finite_difference = _core_perturbation_check(0.02, window=(6.0, 2.4))
    assert neff.imag != 0.0
    assert adjoint.real == pytest.approx(finite_difference.real, rel=1e-3)
    assert adjoint.imag == pytest.approx(finite_difference.imag, rel=1e-3)


@pytest.mark.skipif(
    not os.environ.get("FDTDX_SLOW_MODE_ADJOINT"),
    reason="the paper-sized 20 nm cross-section takes minutes; set FDTDX_SLOW_MODE_ADJOINT=1 to run it",
)
def test_phase_shifter_core_perturbation_at_the_paper_window(float64):
    _, adjoint, finite_difference = _core_perturbation_check(0.02, window=(14.0, 4.0))
    assert adjoint.real == pytest.approx(finite_difference.real, rel=1e-3)
    assert adjoint.imag == pytest.approx(finite_difference.imag, rel=1e-3)


def test_the_imaginary_index_is_differentiated_by_the_same_backward(float64):
    """One backward gives both parts: d Im(n_eff) / d Im(eps) matches its own finite difference."""
    eps, _ = _phase_shifter_cross_section(0.05, window=(7.0, 2.4), w_metal=2.0)
    settings = _settings(resolution=50e-9)
    _, sensitivity = mode_sensitivity(eps, settings)

    rng = np.random.default_rng(11)
    direction = jnp.asarray(1j * rng.normal(size=eps.shape) * (np.abs(np.asarray(eps).imag) > 0))
    adjoint = complex(jnp.sum(direction * sensitivity))

    step = 1e-3

    def parts(t):
        real, imaginary = mode_neff_parts(eps + t * direction, settings)
        return complex(float(real), float(imaginary))

    finite_difference = (parts(step) - parts(-step)) / (2 * step)
    assert adjoint.real == pytest.approx(finite_difference.real, rel=2e-3)
    assert adjoint.imag == pytest.approx(finite_difference.imag, rel=2e-3)


# ------------------------------------------------------------------------------------------------
# (c) mode tracking
# ------------------------------------------------------------------------------------------------


def _two_guides(eps_left: float) -> jnp.ndarray:
    """Two well-separated strips; the left one's permittivity decides which mode sorts first."""
    eps = np.full((60, 30), N_SIO2**2)
    eps[8:20, 12:18] = eps_left
    eps[40:52, 12:18] = 12.0
    return jnp.asarray(eps)[None, :, :, None]


def test_track_mode_picks_the_swapped_partner():
    """Pure selection: two fields presented in the opposite order are matched by overlap."""
    rng = np.random.default_rng(5)
    mode_a = rng.normal(size=(3, 4, 5)) + 1j * rng.normal(size=(3, 4, 5))
    mode_b = rng.normal(size=(3, 4, 5)) + 1j * rng.normal(size=(3, 4, 5))
    # remove the component of b along a so the two are orthogonal, as eigenvectors are
    mode_b = mode_b - mode_a * (np.vdot(mode_a, mode_b) / np.vdot(mode_a, mode_a))

    match = track_mode(mode_a, [mode_b, mode_a * (0.3 - 1.7j)])
    assert match.index == 1
    assert match.overlap == pytest.approx(1.0, abs=1e-12)
    assert match.overlaps[0] == pytest.approx(0.0, abs=1e-12)

    match = track_mode(mode_b, [mode_b * 4.0, mode_a])
    assert match.index == 0


def test_track_mode_gates_on_a_minimum_overlap():
    rng = np.random.default_rng(7)
    field = rng.normal(size=(2, 3))
    other = np.array([[1.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
    other = other - field * (np.vdot(field, other) / np.vdot(field, field))
    with pytest.raises(ValueError, match="best mode overlap"):
        track_mode(field, [other], min_overlap=0.5)
    with pytest.raises(ValueError, match="at least one candidate"):
        track_mode(field, [])


def test_the_mode_is_tracked_not_re_sorted_across_a_finite_difference_step(float64):
    """A finite-difference step that reorders the mode list: fixed index lies, tracking does not."""
    settings = _settings(resolution=40e-9)
    base, step = 12.05, 0.10

    reference = mode_solve(_two_guides(base), settings)  # the left guide's mode, index 0 here
    _, sensitivity = mode_sensitivity(_two_guides(base), settings)
    left = np.zeros((1, 60, 30, 1))
    left[0, 8:20, 12:18, 0] = 1.0
    adjoint = float(jnp.real(jnp.sum(jnp.asarray(left) * sensitivity)))

    tracked = []
    fixed_index = []
    for sign in (+1, -1):
        permittivity = _two_guides(base + sign * step)
        match = tracked_mode_index(permittivity, settings, np.asarray(reference.E), num_candidates=2)
        tracked.append(complex(mode_neff(permittivity, settings.with_mode_index(match.index))).real)
        fixed_index.append(complex(mode_neff(permittivity, settings)).real)
        if sign < 0:
            # the step down puts the left guide's mode behind the right guide's
            assert match.index == 1
            assert match.overlap > 0.99
            assert match.overlaps[0] < 0.2
        else:
            assert match.index == 0

    tracked_slope = (tracked[0] - tracked[1]) / (2 * step)
    fixed_slope = (fixed_index[0] - fixed_index[1]) / (2 * step)
    assert adjoint == pytest.approx(tracked_slope, rel=2e-2)
    assert abs(fixed_slope - adjoint) > 0.1 * abs(adjoint)


# ------------------------------------------------------------------------------------------------
# contract and layout
# ------------------------------------------------------------------------------------------------


def test_the_isotropic_and_diagonal_tiers_give_the_same_gradient(float64):
    settings = _settings()
    isotropic = _strip_cross_section()
    diagonal = jnp.broadcast_to(isotropic, (3, *isotropic.shape[1:]))

    def scaled(eps):
        return lambda s: mode_neff_parts(eps * s, settings)[0]

    grad_isotropic = float(jax.grad(scaled(isotropic))(1.0))
    grad_diagonal = float(jax.grad(scaled(diagonal))(1.0))
    assert grad_isotropic == pytest.approx(grad_diagonal, rel=1e-10)


def test_the_nine_component_tier_now_has_a_mode_gradient(float64):
    """Track J phase 2: the tensor tier solves, and every entry of it carries a sensitivity."""
    eps = _tensor_cross_section(theta_deg=25.0)
    neff, sensitivity = mode_sensitivity(eps, _settings())
    assert sensitivity.shape == eps.shape
    assert float(jnp.real(neff)) > 2.0
    # propagation is along the first axis, so the transverse pair is (y, z): the carried
    # off-diagonal entries are yz and zy, flat indices 5 and 7.
    assert float(jnp.max(jnp.abs(sensitivity[5]))) > 0.0
    assert float(jnp.max(jnp.abs(sensitivity[7]))) > 0.0
    # the four entries that couple a transverse axis to propagation are dropped by the solver,
    # so the solved n_eff does not depend on them and their sensitivity is exactly zero.
    for dropped in (1, 2, 3, 6):
        assert float(jnp.max(jnp.abs(sensitivity[dropped]))) == 0.0


def test_a_cross_section_without_a_propagation_axis_is_refused(float64):
    with pytest.raises(ValueError, match="exactly one"):
        mode_neff_parts(jnp.ones((1, 4, 5, 6)), _settings())


def test_the_settings_need_exactly_one_grid_description():
    with pytest.raises(ValueError, match="exactly one"):
        ModeSolveSettings.create(frequency=FREQ)
    with pytest.raises(ValueError, match="exactly one"):
        ModeSolveSettings.create(frequency=FREQ, resolution=1e-8, transverse_coords=(np.zeros(3), np.zeros(3)))


def test_the_rectilinear_grid_path_gives_the_same_gradient_as_the_uniform_one(float64):
    """A uniform grid handed over as explicit edge coordinates must not change the sensitivity."""
    cell = 40e-9
    eps = _strip_cross_section()
    uniform = _settings(resolution=cell)
    coords = (np.arange(41) * cell, np.arange(31) * cell)
    rectilinear = ModeSolveSettings.create(frequency=FREQ, transverse_coords=coords, mode_index=0)

    def scaled(settings):
        return lambda s: mode_neff_parts(eps * s, settings)[0]

    assert float(jax.grad(scaled(uniform))(1.0)) == pytest.approx(float(jax.grad(scaled(rectilinear))(1.0)), rel=1e-8)


def test_the_returned_mode_fields_are_differentiable(float64):
    """Track J phase 2: mode_solve routes through the JAX pipeline, so the fields carry a gradient."""
    settings = _settings()

    def field_sum(scale):
        return jnp.log10(jnp.sum(jnp.abs(mode_solve(_strip_cross_section() * scale, settings).E) ** 2))

    gradient = float(jax.grad(field_sum)(1.0))
    step = 1e-5
    finite_difference = (float(field_sum(1.0 + step)) - float(field_sum(1.0 - step))) / (2 * step)
    assert gradient != 0.0
    assert gradient == pytest.approx(finite_difference, rel=1e-6)


def test_the_callback_path_still_stops_the_field_gradient(float64):
    """differentiable_fields=False is the old behaviour, kept and still silent about it."""
    settings = _settings()

    def field_sum(scale):
        solution = mode_solve(_strip_cross_section() * scale, settings, differentiable_fields=False)
        return jnp.sum(jnp.abs(solution.E) ** 2)

    assert float(jax.grad(field_sum)(1.0)) == 0.0


def test_a_setting_outside_the_differentiable_path_is_named_not_silently_dropped(float64):
    settings = _settings(filter_pol="te")
    with pytest.raises(ValueError, match="filter_pol"):
        mode_solve(_strip_cross_section(), settings, differentiable_fields=True)
    with pytest.warns(UserWarning, match="no gradient"):
        mode_solve(_strip_cross_section(), settings)


def test_the_exposed_operator_reproduces_the_solved_eigenpair(float64):
    """The assembly the backward's provenance rests on: mat v = -(n_eff)^2 v for the returned mode."""
    import scipy.sparse.linalg as spl

    from fdtdx.core.physics.mode_backend.operator import build_derivative_matrices
    from fdtdx.core.physics.mode_backend.solve import assemble_mode_operator

    cell, nx, ny = 40e-9, 40, 30
    eps = np.asarray(_strip_cross_section()[0, 0])
    settings = _settings(resolution=cell)
    solution = mode_solve(_strip_cross_section(), settings)

    flat = eps.reshape(-1).astype(np.complex128)
    ones = np.ones_like(flat)
    der = build_derivative_matrices(np.arange(nx + 1) * cell, np.arange(ny + 1) * cell)
    operator = assemble_mode_operator(flat, flat, flat, ones, ones, ones, der, k0=2 * np.pi / LAM)

    # the mode solver's own transverse field, stacked the way the operator expects
    field = np.asarray(solution.E)[:, 0]  # (3, nx, ny) with propagation along axis 0
    vector = np.concatenate((field[1].reshape(-1), field[2].reshape(-1)))
    eigenvalue = -(complex(solution.neff) ** 2)
    residual = operator.mat.dot(vector) - eigenvalue * vector
    assert np.linalg.norm(residual) / np.linalg.norm(vector) < 1e-8

    values = spl.eigs(operator.mat.astype(np.complex128), k=4, sigma=eigenvalue, return_eigenvectors=False)
    assert np.min(np.abs(values - eigenvalue)) / abs(eigenvalue) < 1e-10
