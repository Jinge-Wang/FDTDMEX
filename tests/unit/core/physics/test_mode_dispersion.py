"""Group index from one solve: the operator's frequency dependence, differentiated.

Three independent statements of the same number are pinned here - the JAX contraction, the closed
form of the same quadratic form, and a three-point finite difference in frequency - plus the
analytic derivative of the slab dispersion relation, which is the only one that knows nothing about
the discretisation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse.linalg as spl
from scipy.optimize import brentq

from fdtdx.core.physics.mode_backend.dispersion import (
    frequency_scaled_operator_data,
    group_index_closed_form,
    mode_dispersion,
    operator_diagonal_part,
    solve_with_left_eigenvectors,
)
from fdtdx.core.physics.mode_backend.jax_operator import SparseCOO, assemble_mode_operator_jax
from fdtdx.core.physics.mode_backend.jax_solve import EigenSolveSpec
from fdtdx.core.physics.mode_backend.operator import build_derivative_matrices, primal_dual_steps
from fdtdx.core.physics.mode_backend.solve import solve_modes_diagonal
from fdtdx.core.physics.modes import group_index

C_LIGHT = 299792458.0
LAMBDA0 = 1.55e-6

# The 2-D effective-index ring guide of cases/fdtdx/ring2d_thermal_tuning_tidy3d.py.
N_CORE, N_BG, WIDTH = 2.44834, 1.444, 0.5e-6
# The phase-shifter cross-section of Jokisch et al. 2024, as used by the other mode tests.
N_SI, N_SIO2 = 3.48, 1.55
N_TIN, K_TIN = 3.1477, 5.8429
EPS_TIN = (N_TIN**2 - K_TIN**2) + 2j * N_TIN * K_TIN


@pytest.fixture(autouse=True)
def _float64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _slab_cross_section(resolution, window=4.0e-6):
    ny = round(window / resolution)
    ys = (np.arange(ny) - ny / 2 + 0.5) * resolution
    eps = np.where(np.abs(ys) <= WIDTH / 2, N_CORE**2, N_BG**2)
    return jnp.asarray((1.0 / eps)[None, None, :, None] * np.ones((1, 1, ny, 2)))


def _phase_shifter_problem(cell_um=0.05, window=(6.0, 2.4)):
    """Flattened material and grid of the heater cross-section, ready for the backend."""
    nx, ny = round(window[0] / cell_um), round(window[1] / cell_um)
    x = -window[0] / 2 + (np.arange(nx) + 0.5) * cell_um
    y = -window[1] / 2 + (np.arange(ny) + 0.5) * cell_um
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    eps = np.full((nx, ny), N_SIO2**2, dtype=np.complex128)
    eps[(np.abs(grid_x) < 2.0) & (np.abs(grid_y) < 0.25)] = N_SI**2
    eps[(grid_x >= -2.0) & (grid_x <= 0.0) & (grid_y >= 0.25) & (grid_y <= 0.5)] = EPS_TIN
    flat = eps.ravel()
    ones = np.ones(nx * ny, dtype=np.complex128)
    coords_x = np.arange(nx + 1) * cell_um * 1e-6
    coords_y = np.arange(ny + 1) * cell_um * 1e-6
    return {
        "eps": (flat, flat.copy(), flat.copy()),
        "mu": (ones, ones.copy(), ones.copy()),
        "der": build_derivative_matrices(coords_x, coords_y),
        "steps": (primal_dual_steps(coords_x), primal_dual_steps(coords_y)),
        "guess": float(np.sqrt(np.max(np.real(flat)))) * (1.0 + 1e-6) + 1e-6,
    }


def _slab_root(k0, polarization="tm"):
    """The symmetric-slab dispersion relation's fundamental root, without any tan pole."""
    ratio = (N_CORE**2 / N_BG**2) if polarization == "tm" else 1.0

    def residual(neff):
        kappa = k0 * np.sqrt(N_CORE**2 - neff**2)
        gamma = k0 * np.sqrt(neff**2 - N_BG**2)
        return kappa * np.sin(kappa * WIDTH / 2) - ratio * gamma * np.cos(kappa * WIDTH / 2)

    grid = np.linspace(N_BG + 1e-12, N_CORE - 1e-12, 4000)
    values = np.array([residual(v) for v in grid])
    roots = [
        brentq(residual, grid[i], grid[i + 1], xtol=1e-15)
        for i in range(len(grid) - 1)
        if values[i] * values[i + 1] < 0
    ]
    return max(roots)


def _slab_group_index(k0, polarization="tm"):
    """``n + k0 dn/dk0`` with ``dn/dk0`` from the implicit function theorem, partials by autodiff."""
    ratio = (N_CORE**2 / N_BG**2) if polarization == "tm" else 1.0

    def residual(neff, wavenumber):
        kappa = wavenumber * jnp.sqrt(N_CORE**2 - neff**2)
        gamma = wavenumber * jnp.sqrt(neff**2 - N_BG**2)
        return kappa * jnp.sin(kappa * WIDTH / 2) - ratio * gamma * jnp.cos(kappa * WIDTH / 2)

    neff = _slab_root(k0, polarization)
    d_neff = jax.grad(residual, 0)(neff, k0)
    d_k0 = jax.grad(residual, 1)(neff, k0)
    return neff + k0 * float(-d_k0 / d_neff)


class TestFrequencyDependenceOfTheOperator:
    """``A(k0) = D + S / k0**2`` exactly - the identity the whole module rests on."""

    def test_the_scaled_operator_is_the_freshly_assembled_one(self):
        problem = _phase_shifter_problem(cell_um=0.2)
        k0_ref = 2 * np.pi / LAMBDA0
        k0_other = 2 * np.pi / (1.31e-6)
        operator = assemble_mode_operator_jax(*problem["eps"], *problem["mu"], problem["der"], k0_ref, problem["steps"])
        diagonal = operator_diagonal_part(operator, jnp.asarray(problem["mu"][0]), jnp.asarray(problem["mu"][1]))
        rows, cols, data = frequency_scaled_operator_data(operator, diagonal, k0_ref, k0_other)
        scaled = SparseCOO(rows, cols, data, operator.mat.shape).to_scipy().todense()
        fresh = (
            assemble_mode_operator_jax(*problem["eps"], *problem["mu"], problem["der"], k0_other, problem["steps"])
            .mat.to_scipy()
            .todense()
        )
        difference = np.max(np.abs(scaled - fresh)) / np.max(np.abs(fresh))
        assert difference < 1e-14

    def test_at_the_reference_wavenumber_it_is_the_identity(self):
        problem = _phase_shifter_problem(cell_um=0.3)
        k0_ref = 2 * np.pi / LAMBDA0
        operator = assemble_mode_operator_jax(*problem["eps"], *problem["mu"], problem["der"], k0_ref, problem["steps"])
        diagonal = operator_diagonal_part(operator, jnp.asarray(problem["mu"][0]), jnp.asarray(problem["mu"][1]))
        _, _, data = frequency_scaled_operator_data(operator, diagonal, k0_ref, k0_ref)
        # the appended diagonal correction is exactly zero there
        np.testing.assert_allclose(np.asarray(data[: operator.mat.data.size]), np.asarray(operator.mat.data), rtol=0)
        np.testing.assert_allclose(np.asarray(data[operator.mat.data.size :]), 0.0, atol=0)


class TestGroupIndexAgainstTheAnalyticSlab:
    """The only reference that knows nothing about the discretisation."""

    def test_extrapolated_group_index_matches_the_slab_dispersion_relation(self):
        k0 = 2 * np.pi / LAMBDA0
        exact = _slab_group_index(k0)
        values = []
        for resolution in (25e-9, 12.5e-9):
            dispersion = group_index(
                frequency=C_LIGHT / LAMBDA0,
                inv_permittivities=_slab_cross_section(resolution),
                inv_permeabilities=1.0,
                resolution=resolution,
                filter_pol="te",
            )
            values.append(complex(dispersion.group_index).real)
        # second order in the cell size, so one Richardson step over the pair
        extrapolated = (4 * values[1] - values[0]) / 3
        assert values[0] == pytest.approx(exact, rel=3e-4)
        assert extrapolated == pytest.approx(exact, rel=2e-5)

    def test_the_group_index_exceeds_the_effective_index(self):
        dispersion = group_index(
            frequency=C_LIGHT / LAMBDA0,
            inv_permittivities=_slab_cross_section(25e-9),
            inv_permeabilities=1.0,
            resolution=25e-9,
            filter_pol="te",
        )
        assert complex(dispersion.group_index).real > complex(dispersion.neff).real


class TestGroupIndexAgainstAFrequencyFiniteDifference:
    """The same discretisation, differenced in frequency - what a three-solve solver would do."""

    def test_the_contraction_is_the_limit_of_the_three_point_difference(self):
        problem = _phase_shifter_problem(cell_um=0.05)
        frequency = C_LIGHT / LAMBDA0
        dispersion = mode_dispersion(
            *problem["eps"],
            *problem["mu"],
            problem["der"],
            problem["steps"],
            frequency=frequency,
            num_modes=4,
            neff_guess=problem["guess"],
        )

        def neff_at(f):
            k0 = 2 * np.pi * f / C_LIGHT
            _, _, neff, keff = solve_modes_diagonal(
                *problem["eps"], *problem["mu"], problem["der"], k0=k0, num_modes=4, neff_guess=problem["guess"]
            )
            return neff[0] + 1j * keff[0]

        step = 1e-3
        derivative = (neff_at(frequency * (1 + step)) - neff_at(frequency * (1 - step))) / (2 * step * frequency)
        from_three_solves = neff_at(frequency) + frequency * derivative
        assert abs(from_three_solves - complex(dispersion.group_index)) / abs(from_three_solves) < 1e-6

    def test_the_difference_shrinks_as_the_square_of_the_frequency_step(self):
        problem = _phase_shifter_problem(cell_um=0.1)
        frequency = C_LIGHT / LAMBDA0
        dispersion = mode_dispersion(
            *problem["eps"],
            *problem["mu"],
            problem["der"],
            problem["steps"],
            frequency=frequency,
            num_modes=4,
            neff_guess=problem["guess"],
        )

        def group_from_difference(step):
            def neff_at(f):
                _, _, neff, keff = solve_modes_diagonal(
                    *problem["eps"],
                    *problem["mu"],
                    problem["der"],
                    k0=2 * np.pi * f / C_LIGHT,
                    num_modes=4,
                    neff_guess=problem["guess"],
                )
                return neff[0] + 1j * keff[0]

            derivative = (neff_at(frequency * (1 + step)) - neff_at(frequency * (1 - step))) / (2 * step * frequency)
            return neff_at(frequency) + frequency * derivative

        coarse = abs(group_from_difference(1e-2) - complex(dispersion.group_index))
        fine = abs(group_from_difference(1e-3) - complex(dispersion.group_index))
        assert coarse / fine == pytest.approx(100.0, rel=0.2)


class TestGroupIndexCostsOneSolve:
    """The point of the contraction: no second eigen-solve, and no second assembly."""

    def test_only_one_arpack_call(self, monkeypatch):
        calls = []
        original = spl.eigs

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(spl, "eigs", counted)
        problem = _phase_shifter_problem(cell_um=0.15)
        mode_dispersion(
            *problem["eps"],
            *problem["mu"],
            problem["der"],
            problem["steps"],
            frequency=C_LIGHT / LAMBDA0,
            num_modes=4,
            neff_guess=problem["guess"],
        )
        assert sum(calls) == 1


class TestGroupIndexAgainstItsClosedForm:
    """The contraction written out by hand, as a second derivation of the same quantity."""

    def test_the_closed_form_agrees_with_the_differentiated_one(self):
        problem = _phase_shifter_problem(cell_um=0.15)
        frequency = C_LIGHT / LAMBDA0
        k0_ref = 2 * np.pi * frequency / C_LIGHT
        operator = assemble_mode_operator_jax(*problem["eps"], *problem["mu"], problem["der"], k0_ref, problem["steps"])
        diagonal = operator_diagonal_part(operator, jnp.asarray(problem["mu"][0]), jnp.asarray(problem["mu"][1]))
        rows, cols, data = frequency_scaled_operator_data(operator, diagonal, k0_ref, k0_ref)
        spec = EigenSolveSpec(
            mat_rows=rows,
            mat_cols=cols,
            qmat_rows=operator.qmat.rows,
            qmat_cols=operator.qmat.cols,
            left_weights=operator.left_weights,
            num_cells=operator.num_cells,
            num_modes=4,
            sigma=complex(-(problem["guess"] ** 2)),
        )
        vals, vecs, left = solve_with_left_eigenvectors(operator, spec, data)
        closed = group_index_closed_form(operator, diagonal, vals, vecs, left)
        differentiated = mode_dispersion(
            *problem["eps"],
            *problem["mu"],
            problem["der"],
            problem["steps"],
            frequency=frequency,
            num_modes=4,
            neff_guess=problem["guess"],
        )
        assert complex(closed[0]) == pytest.approx(complex(differentiated.group_index), rel=1e-10)


class TestGroupIndexOfABentGuide:
    """The bend transform has no frequency dependence, so the same contraction serves it."""

    def test_a_bent_guide_has_a_group_index_near_the_straight_one(self):
        common = dict(
            frequency=C_LIGHT / LAMBDA0,
            inv_permittivities=_slab_cross_section(12.5e-9),
            inv_permeabilities=1.0,
            resolution=12.5e-9,
            filter_pol="te",
        )
        straight = group_index(**common)
        bent = group_index(**common, bend_radius=5e-6, bend_axis=2, target_neff=2.0957)
        # R = 5 um raises n_eff by 0.1 % and lowers n_g by 0.2 %: the bend is felt more strongly by
        # the derivative than by the index itself, which is the whole reason to compute it exactly.
        assert complex(bent.neff).real > complex(straight.neff).real
        assert complex(bent.group_index).real == pytest.approx(complex(straight.group_index).real, rel=1e-2)
        assert complex(bent.group_index).real < complex(straight.group_index).real


class TestGroupIndexRefusals:
    def test_a_magnetic_wall_is_refused(self):
        with pytest.raises(NotImplementedError, match="electric"):
            group_index(
                frequency=C_LIGHT / LAMBDA0,
                inv_permittivities=_slab_cross_section(50e-9),
                inv_permeabilities=1.0,
                resolution=50e-9,
                symmetry=(1, 0),
            )

    def test_a_fully_tensorial_cross_section_is_refused(self):
        with pytest.raises(NotImplementedError, match="isotropic or diagonally"):
            group_index(
                frequency=C_LIGHT / LAMBDA0,
                inv_permittivities=jnp.ones((9, 1, 8, 2)),
                inv_permeabilities=1.0,
                resolution=50e-9,
            )

    def test_single_precision_is_refused(self):
        jax.config.update("jax_enable_x64", False)
        try:
            with pytest.raises(ValueError, match="double precision"):
                group_index(
                    frequency=C_LIGHT / LAMBDA0,
                    inv_permittivities=jnp.ones((1, 1, 8, 2)),
                    inv_permeabilities=1.0,
                    resolution=50e-9,
                )
        finally:
            jax.config.update("jax_enable_x64", True)
