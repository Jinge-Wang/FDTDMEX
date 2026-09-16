"""The JAX-native mode operator: parity with numpy, a dense oracle, gradients, degeneracy.

Everything here needs float64. The mode eigenvalue is ``-(n_eff)^2``, so a complex64 operator moves
``n_eff`` at the 1e-7 level and neither the parity bar (1e-14) nor the gradient checks (1e-9) have
any signal left.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

import scipy.sparse.linalg as spl  # noqa: E402

from fdtdx.core.physics.mode_backend.jax_operator import (  # noqa: E402
    assemble_mode_operator_jax,
    left_eigenvectors_jax,
    reconstruct_fields_jax,
)
from fdtdx.core.physics.mode_backend.jax_solve import (  # noqa: E402
    EigenSolveSpec,
    degeneracy_mask,
    degenerate_groups,
    dense_mode_eigs,
    left_eigenvector_residual,
    orthonormalize_degenerate_blocks,
    solve_modes_diagonal_jax,
    sparse_mode_eigs,
)
from fdtdx.core.physics.mode_backend.operator import build_derivative_matrices, primal_dual_steps  # noqa: E402
from fdtdx.core.physics.mode_backend.solve import (  # noqa: E402
    assemble_mode_operator,
    reconstruct_fields,
    solve_modes_diagonal,
)

LAM = 1.55e-6
K0 = 2 * np.pi / LAM
N_SI, N_SIO2 = 3.48, 1.55
N_TIN, K_TIN = 3.1477, 5.8429
EPS_TIN = (N_TIN**2 - K_TIN**2) + 2j * N_TIN * K_TIN


@pytest.fixture
def float64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _phase_shifter_cross_section(cell_um: float, window=(6.0, 2.4), w_metal: float = 2.0):
    """Jokisch et al. 2024 section 2: Si guide, SiO2 cladding, TiN heater, as a flat eps array."""
    nx = round(window[0] / cell_um)
    ny = round(window[1] / cell_um)
    x = -window[0] / 2 + (np.arange(nx) + 0.5) * cell_um
    y = -window[1] / 2 + (np.arange(ny) + 0.5) * cell_um
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    eps = np.full((nx, ny), N_SIO2**2, dtype=np.complex128)
    eps[(np.abs(grid_x) < 2.0) & (np.abs(grid_y) < 0.25)] = N_SI**2
    metal = (grid_x >= -2.0) & (grid_x <= -2.0 + w_metal) & (grid_y >= 0.25) & (grid_y <= 0.5)
    eps[metal] = EPS_TIN
    return eps, nx, ny


class _Problem:
    """A cross-section plus everything both backends need to be handed."""

    def __init__(self, eps2d, coords_x, coords_y, mu_scale=None):
        nx, ny = eps2d.shape
        self.n = nx * ny
        self.nx, self.ny = nx, ny
        flat = eps2d.ravel().astype(np.complex128)
        self.eps = (flat, flat * 1.0, flat * 1.0)
        ones = np.ones(self.n, dtype=np.complex128)
        self.mu = (
            (ones, ones, ones) if mu_scale is None else (ones * mu_scale[0], ones * mu_scale[1], ones * mu_scale[2])
        )
        self.der = build_derivative_matrices(coords_x, coords_y)
        self.steps = (primal_dual_steps(coords_x), primal_dual_steps(coords_y))
        self.guess = float(np.sqrt(np.max(np.real(flat)))) * (1.0 + 1e-6) + 1e-6

    def numpy_operator(self):
        return assemble_mode_operator(*self.eps, *self.mu, self.der, K0)

    def jax_operator(self):
        return assemble_mode_operator_jax(
            *[jnp.asarray(c) for c in self.eps],
            *[jnp.asarray(c) for c in self.mu],
            self.der,
            K0,
            self.steps,
        )


def _uniform(eps2d, cell_m):
    nx, ny = eps2d.shape
    return _Problem(eps2d, np.arange(nx + 1) * cell_m, np.arange(ny + 1) * cell_m)


def _graded(eps2d, cell_m, seed=5):
    """A rectilinear grid whose cell widths vary by a factor of three across both axes."""
    rng = np.random.default_rng(seed)
    nx, ny = eps2d.shape
    cx = np.concatenate(([0.0], np.cumsum(cell_m * (0.5 + rng.random(nx)))))
    cy = np.concatenate(([0.0], np.cumsum(cell_m * (0.5 + rng.random(ny)))))
    return _Problem(eps2d, cx, cy)


def _strip(nx=24, ny=18):
    eps = np.full((nx, ny), N_SIO2**2)
    eps[nx // 3 : 2 * nx // 3, ny // 3 : 2 * ny // 3] = N_SI**2
    return eps


class TestAssemblyParity:
    """The JAX assembly is the numpy assembly, entry by entry."""

    @staticmethod
    def _compare(problem):
        reference = problem.numpy_operator()
        jax_operator = problem.jax_operator()
        for name, ref, got in (
            ("mat", reference.mat, jax_operator.mat.to_scipy()),
            ("qmat", reference.qmat, jax_operator.qmat.to_scipy()),
        ):
            difference = np.abs((got - ref).toarray()).max()
            scale = np.abs(ref.toarray()).max()
            assert difference / scale < 1e-14, f"{name} differs by {difference / scale:g}"
        return reference, jax_operator

    def test_phase_shifter_cross_section(self, float64):
        """The real device cross-section (Si guide + TiN heater + oxide) at 50 nm."""
        eps, _, _ = _phase_shifter_cross_section(0.05)
        self._compare(_uniform(eps, 50e-9))

    def test_non_uniform_grid(self, float64):
        """A graded rectilinear grid, where the primal and dual steps stop being equal."""
        eps, _, _ = _phase_shifter_cross_section(0.1)
        self._compare(_graded(eps, 100e-9))

    def test_anisotropic_permeability(self, float64):
        problem = _Problem(
            _strip(12, 10),
            np.arange(13) * 60e-9,
            np.arange(11) * 60e-9,
            mu_scale=(1.0, 1.3, 0.8),
        )
        self._compare(problem)

    def test_duplicate_entries_are_summed_not_dropped(self, float64):
        """The coordinate list carries one entry per contribution; scipy and BCOO both sum them."""
        problem = _uniform(_strip(10, 8), 60e-9)
        jax_operator = problem.jax_operator()
        assert len(jax_operator.mat.data) > problem.numpy_operator().mat.nnz
        dense_bcoo = np.asarray(jax_operator.mat.to_bcoo().todense())
        assert np.abs(dense_bcoo - jax_operator.mat.to_scipy().toarray()).max() < 1e-20


class TestReconstructionParity:
    """Stage 3 in JAX reproduces stage 3 in numpy on the same eigenpairs."""

    def test_fields_agree_on_the_same_eigenvectors(self, float64):
        problem = _uniform(_strip(), 50e-9)
        reference = problem.numpy_operator()
        values, vectors = spl.eigs(
            reference.mat.astype(np.complex128),
            k=4,
            sigma=-(problem.guess**2),
            v0=np.random.default_rng(0).random(2 * problem.n) + 1j * np.random.default_rng(1).random(2 * problem.n),
        )
        order = np.argsort(np.real(np.emath.sqrt(-values + 0j)))[::-1]
        values, vectors = values[order], vectors[:, order]

        field_e, field_h, neff, keff = reconstruct_fields(
            vectors,
            values,
            reference.qmat,
            reference.q_ep,
            reference.inv_eps_zz,
            reference.inv_mu_zz,
            reference.der_mats,
        )
        got_e, got_h, got_neff, got_keff = reconstruct_fields_jax(
            problem.jax_operator(), jnp.asarray(vectors), jnp.asarray(values)
        )
        assert np.abs(np.asarray(got_e) - field_e).max() / np.abs(field_e).max() < 1e-14
        assert np.abs(np.asarray(got_h) - field_h).max() / np.abs(field_h).max() < 1e-14
        assert np.allclose(np.asarray(got_neff), neff, rtol=1e-14, atol=0)
        assert np.allclose(np.asarray(got_keff), keff, rtol=0, atol=1e-14)

    def test_backward_direction_flips_the_same_components(self, float64):
        problem = _uniform(_strip(12, 10), 60e-9)
        operator = problem.jax_operator()
        values, vectors = dense_mode_eigs(operator)
        forward = reconstruct_fields_jax(operator, vectors[:, :2], values[:2], direction="+")
        backward = reconstruct_fields_jax(operator, vectors[:, :2], values[:2], direction="-")
        assert np.allclose(np.asarray(backward[0][:2]), np.asarray(forward[0][:2]))
        assert np.allclose(np.asarray(backward[0][2]), -np.asarray(forward[0][2]))
        assert np.allclose(np.asarray(backward[1][:2]), -np.asarray(forward[1][:2]))
        assert np.allclose(np.asarray(backward[1][2]), np.asarray(forward[1][2]))


class TestLeftEigenvector:
    """The reciprocity relation the backward rests on: u = [w_x hy; -w_y hx]."""

    def test_exact_on_a_uniform_grid(self, float64):
        problem = _uniform(_strip(12, 10), 60e-9)
        operator = problem.jax_operator()
        values, vectors = dense_mode_eigs(operator)
        residual = np.asarray(left_eigenvector_residual(operator, vectors[:, :4], values[:4]))
        assert residual.max() < 1e-10

    def test_exact_on_a_graded_grid(self, float64):
        """Without the staggered cell-area weights this residual is a few percent, not 1e-10."""
        problem = _graded(_strip(12, 10), 60e-9)
        operator = problem.jax_operator()
        values, vectors = dense_mode_eigs(operator)
        residual = np.asarray(left_eigenvector_residual(operator, vectors[:, :4], values[:4]))
        assert residual.max() < 1e-10

        unweighted = np.concatenate(
            (
                np.asarray(operator.qmat.matmul(vectors[:, :4]))[operator.num_cells :],
                -np.asarray(operator.qmat.matmul(vectors[:, :4]))[: operator.num_cells],
            ),
            axis=0,
        )
        matrix = operator.mat.to_scipy().toarray()
        naive = np.linalg.norm(unweighted.T @ matrix - np.asarray(values[:4])[:, None] * unweighted.T, axis=1)
        naive = naive / (np.linalg.norm(unweighted, axis=0) * np.abs(np.asarray(values[:4])))
        assert naive.max() > 1e-3


class TestDenseOracle:
    """jnp.linalg.eig on a small cross-section, as a fully differentiable reference."""

    def test_matches_the_sparse_solve(self, float64):
        problem = _uniform(_strip(14, 12), 60e-9)
        operator = problem.jax_operator()
        dense_values, _ = dense_mode_eigs(operator)
        _, _, neff, _ = solve_modes_diagonal(*problem.eps, *problem.mu, problem.der, K0, 6, problem.guess)
        dense_neff = np.real(np.sqrt(-np.asarray(dense_values[: len(neff)])))
        assert np.allclose(dense_neff, neff, rtol=1e-10, atol=0)

    def test_eigenvectors_match_up_to_phase(self, float64):
        problem = _uniform(_strip(14, 12), 60e-9)
        operator = problem.jax_operator()
        dense_values, dense_vectors = dense_mode_eigs(operator)
        spec = EigenSolveSpec(
            mat_rows=operator.mat.rows,
            mat_cols=operator.mat.cols,
            qmat_rows=operator.qmat.rows,
            qmat_cols=operator.qmat.cols,
            left_weights=operator.left_weights,
            num_cells=operator.num_cells,
            num_modes=4,
            sigma=complex(-(problem.guess**2)),
        )
        sparse_values, sparse_vectors = sparse_mode_eigs(spec, operator.mat.data, operator.qmat.data)
        assert np.allclose(np.asarray(sparse_values[:2]), np.asarray(dense_values[:2]), rtol=1e-10)
        for mode in range(2):
            a = np.asarray(dense_vectors[:, mode])
            b = np.asarray(sparse_vectors[:, mode])
            overlap = abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
            assert overlap == pytest.approx(1.0, abs=1e-8)

    def test_gradient_agrees_with_the_sparse_adjoint(self, float64):
        """The hand-written reciprocity adjoint against JAX's own eigenvalue derivative.

        14 x 12, not the smaller 12 x 10 used elsewhere in this file: the smaller strip does not
        guide and its two leading modes are an exactly degenerate wall pair, on which a per-mode
        derivative is basis-dependent. A uniform scaling happens to hide that (both members of the
        block scale alike), so the check would pass for the wrong reason.
        """
        problem = _uniform(_strip(14, 12), 60e-9)

        def dense_neff(scale):
            eps = [jnp.asarray(c) * scale for c in problem.eps]
            operator = assemble_mode_operator_jax(
                *eps, *[jnp.asarray(c) for c in problem.mu], problem.der, K0, problem.steps
            )
            values, _ = dense_mode_eigs(operator)
            return jnp.real(jnp.sqrt(-values[0]))

        def sparse_neff(scale):
            eps = [jnp.asarray(c) * scale for c in problem.eps]
            _, _, neff, _ = solve_modes_diagonal_jax(
                *eps, *[jnp.asarray(c) for c in problem.mu], problem.der, problem.steps, K0, 4, problem.guess
            )
            return neff[0]

        assert float(dense_neff(1.0)) == pytest.approx(float(sparse_neff(1.0)), rel=1e-10)
        oracle = float(jax.grad(dense_neff)(1.0))
        adjoint = float(jax.grad(sparse_neff)(1.0))
        assert adjoint == pytest.approx(oracle, rel=1e-8)

    def test_the_differentiable_path_refuses_a_magnetic_wall(self, float64):
        """The closed-form left eigenvector is a PEC statement; PMC has to be refused, not warned."""
        eps = _strip(10, 8)
        nx, ny = eps.shape
        problem = _Problem(eps, np.arange(nx + 1) * 60e-9, np.arange(ny + 1) * 60e-9)
        with pytest.raises(NotImplementedError, match="PEC"):
            solve_modes_diagonal_jax(
                *[jnp.asarray(c) for c in problem.eps],
                *[jnp.asarray(c) for c in problem.mu],
                problem.der,
                problem.steps,
                K0,
                4,
                problem.guess,
                dmin_pmc=(True, False),
            )

    def test_refuses_a_cross_section_that_is_too_large(self, float64):
        problem = _uniform(np.full((30, 30), 4.0), 60e-9)
        with pytest.raises(ValueError, match="test oracle"):
            dense_mode_eigs(problem.jax_operator())


class TestGradientThroughTheWholePipeline:
    """The permittivity gradient reaches n_eff through assembly, ARPACK and reconstruction."""

    @staticmethod
    def _neff_of_scale(problem, mode=0, num_modes=4):
        def f(scale):
            eps = [jnp.asarray(c) * scale for c in problem.eps]
            _, _, neff, _ = solve_modes_diagonal_jax(
                *eps, *[jnp.asarray(c) for c in problem.mu], problem.der, problem.steps, K0, num_modes, problem.guess
            )
            return neff[mode]

        return f

    def test_matches_a_central_finite_difference(self, float64):
        problem = _uniform(_strip(20, 16), 50e-9)
        f = self._neff_of_scale(problem)
        step = 1e-6
        finite_difference = (float(f(1.0 + step)) - float(f(1.0 - step))) / (2 * step)
        assert float(jax.grad(f)(1.0)) == pytest.approx(finite_difference, rel=1e-8)

    def test_matches_on_a_graded_grid(self, float64):
        problem = _graded(_strip(16, 14), 60e-9)
        f = self._neff_of_scale(problem)
        step = 1e-6
        finite_difference = (float(f(1.0 + step)) - float(f(1.0 - step))) / (2 * step)
        assert float(jax.grad(f)(1.0)) == pytest.approx(finite_difference, rel=1e-7)

    def test_a_lossy_cross_section_differentiates_both_parts(self, float64):
        """The TiN heater makes n_eff complex; one backward gives d Re and d Im together."""
        eps, _, _ = _phase_shifter_cross_section(0.1)
        problem = _uniform(eps, 100e-9)

        def parts(scale):
            eps_scaled = [jnp.asarray(c) * scale for c in problem.eps]
            _, _, neff, keff = solve_modes_diagonal_jax(
                *eps_scaled, *[jnp.asarray(c) for c in problem.mu], problem.der, problem.steps, K0, 4, problem.guess
            )
            return neff[0], keff[0]

        step = 1e-6
        grad_re = float(jax.grad(lambda s: parts(s)[0])(1.0))
        grad_im = float(jax.grad(lambda s: parts(s)[1])(1.0))
        up, down = parts(1.0 + step), parts(1.0 - step)
        assert grad_re == pytest.approx((float(up[0]) - float(down[0])) / (2 * step), rel=1e-6)
        assert grad_im == pytest.approx((float(up[1]) - float(down[1])) / (2 * step), rel=1e-6)

    def test_a_degenerate_modes_field_cotangent_is_refused(self, float64):
        """The 10 x 8 strip does not guide: its two leading modes are a degenerate wall pair."""
        problem = _uniform(_strip(10, 8), 60e-9)

        def field_norm(scale):
            eps = [jnp.asarray(c) * scale for c in problem.eps]
            field_e, _, _, _ = solve_modes_diagonal_jax(
                *eps, *[jnp.asarray(c) for c in problem.mu], problem.der, problem.steps, K0, 4, problem.guess
            )
            return jnp.sum(jnp.abs(field_e[:, :, 0]) ** 2)

        with pytest.raises(Exception, match="degenerate block"):
            jax.grad(field_norm)(1.0)

    def test_the_permittivity_reaches_the_fields_at_a_frozen_eigenvector(self, float64):
        """Stage 3 is traceable: with the eigenvector held, eps still moves Ez and H."""
        problem = _uniform(_strip(10, 8), 60e-9)
        operator = problem.jax_operator()
        values, vectors = dense_mode_eigs(operator)
        frozen_vectors = jax.lax.stop_gradient(vectors[:, :1])
        frozen_values = jax.lax.stop_gradient(values[:1])

        def field_norm(scale):
            eps = [jnp.asarray(c) * scale for c in problem.eps]
            scaled = assemble_mode_operator_jax(
                *eps, *[jnp.asarray(c) for c in problem.mu], problem.der, K0, problem.steps
            )
            field_e, _, _, _ = reconstruct_fields_jax(scaled, frozen_vectors, frozen_values)
            return jnp.sum(jnp.abs(field_e[2]) ** 2)

        step = 1e-6
        gradient = float(jax.grad(field_norm)(1.0))
        finite_difference = (float(field_norm(1.0 + step)) - float(field_norm(1.0 - step))) / (2 * step)
        assert gradient != 0.0
        assert gradient == pytest.approx(finite_difference, rel=1e-6)


class TestDegeneracy:
    """A square waveguide, whose two fundamental modes are exactly degenerate."""

    @staticmethod
    def _square_guide(cells=12, cell_m=60e-9):
        """A homogeneous square cross-section inside the solver's PEC walls.

        Its two lowest modes are the x- and y-polarised partners of one 90-degree rotation, so the
        discrete operator makes them degenerate to the last bit - which is what a degeneracy test
        needs. A *dielectric* square guide is degenerate only in the continuum: the staggered
        operator splits its two fundamentals by about 1e-4 relative (measured below), which is why
        the detection tolerance is a relative gap and not an absolute one.
        """
        return _uniform(np.full((cells, cells), N_SI**2), cell_m)

    def test_the_two_fundamentals_are_exactly_degenerate(self, float64):
        problem = self._square_guide()
        _, _, neff, _ = solve_modes_diagonal(*problem.eps, *problem.mu, problem.der, K0, 4, problem.guess)
        assert abs(neff[0] - neff[1]) / abs(neff[0]) < 1e-14
        assert abs(neff[1] - neff[2]) / abs(neff[1]) > 1e-3

    def test_a_dielectric_square_guide_splits_at_the_discretisation_level(self, float64):
        eps = np.full((20, 20), N_SIO2**2)
        eps[6:14, 6:14] = N_SI**2
        problem = _uniform(eps, 60e-9)
        _, _, neff, _ = solve_modes_diagonal(*problem.eps, *problem.mu, problem.der, K0, 4, problem.guess)
        split = abs(neff[0] - neff[1]) / abs(neff[0])
        assert 1e-6 < split < 1e-3

    def test_groups_are_detected(self, float64):
        problem = self._square_guide()
        operator = problem.jax_operator()
        values, _ = dense_mode_eigs(operator)
        groups = degenerate_groups(np.asarray(values[:4]))
        assert groups[0] == [0, 1]

    def test_the_block_is_orthonormalised(self, float64):
        problem = self._square_guide()
        operator = problem.jax_operator()
        values, vectors = dense_mode_eigs(operator)
        pair = np.asarray(vectors[:, :2])
        skewed = np.stack((pair[:, 0], pair[:, 0] + 0.3 * pair[:, 1]), axis=1)
        assert abs(np.vdot(skewed[:, 0], skewed[:, 1])) > 1e-3
        fixed = orthonormalize_degenerate_blocks(skewed, np.asarray(values[:2]))
        assert abs(np.vdot(fixed[:, 0], fixed[:, 1])) < 1e-12
        assert np.linalg.norm(fixed[:, 0]) == pytest.approx(1.0)
        assert np.linalg.norm(fixed[:, 1]) == pytest.approx(1.0)

    def test_the_subspace_gradient_matches_finite_differences(self, float64):
        problem = self._square_guide()

        def neff_of(scale, mode):
            eps = [jnp.asarray(c) * scale for c in problem.eps]
            _, _, neff, _ = solve_modes_diagonal_jax(
                *eps, *[jnp.asarray(c) for c in problem.mu], problem.der, problem.steps, K0, 4, problem.guess
            )
            return neff[mode]

        step = 1e-6
        finite_difference = (float(neff_of(1.0 + step, 0)) - float(neff_of(1.0 - step, 0))) / (2 * step)
        grad_first = float(jax.grad(neff_of, argnums=0)(1.0, 0))
        grad_second = float(jax.grad(neff_of, argnums=0)(1.0, 1))
        assert grad_first == pytest.approx(finite_difference, rel=1e-8)
        # Both members of the block carry the same subspace-averaged derivative, by construction.
        assert grad_second == pytest.approx(grad_first, rel=1e-12)

    def test_a_splitting_perturbation_is_what_the_average_is_for(self, float64):
        """What ``d n_eff / d eps`` *means* on a degenerate pair, measured.

        A uniform scaling keeps the pair together, so every definition agrees and the test above
        cannot tell them apart. A per-cell perturbation splits it - and then the individual branch
        ``n_eff[0]`` is no longer a smooth function of the permittivity: it is the lower root of a
        2x2 problem whose two roots separate linearly in the step. The block average is smooth, and
        it is what this backward returns. Measured on the 12 x 12 square guide with a random
        direction: the average of the two branches' central differences matches the adjoint to 7e-10,
        while either branch alone is off by 7e-5 at h = 1e-3 and 7e-6 at h = 1e-4, i.e. the
        discrepancy is the splitting term and not an error in the gradient.
        """
        problem = self._square_guide()

        def modes(delta):
            eps = [jnp.asarray(component) + delta.astype(jnp.complex128) for component in problem.eps]
            _, _, neff, _ = solve_modes_diagonal_jax(
                *eps, *[jnp.asarray(c) for c in problem.mu], problem.der, problem.steps, K0, 4, problem.guess
            )
            return neff

        zero = jnp.zeros(problem.n, dtype=jnp.float64)
        gradient = np.asarray(jax.grad(lambda d: modes(d)[0])(zero))
        rng = np.random.default_rng(0)
        direction = rng.normal(size=problem.n)
        direction /= np.linalg.norm(direction)
        adjoint = float(np.dot(gradient, direction))

        step = 1e-3
        moved = jnp.asarray(step * direction)
        up, down = np.asarray(modes(zero + moved)[:2]), np.asarray(modes(zero - moved)[:2])
        branches = (up - down) / (2 * step)
        assert abs(up[0] - up[1]) > 1e-7  # the perturbation really does split the pair
        assert adjoint == pytest.approx(float(branches.mean()), rel=1e-8)
        assert abs(adjoint - branches[0]) / abs(adjoint) > 1e-6

    def test_the_averaged_sensitivity_is_basis_independent(self, float64):
        """Rotate the degenerate pair: the per-mode contraction moves, the averaged one does not."""
        problem = self._square_guide()
        operator = problem.jax_operator()
        _, vectors = dense_mode_eigs(operator)
        pair = np.asarray(vectors[:, :2])
        matrix = operator.mat.to_scipy().toarray()
        rng = np.random.default_rng(0)
        direction = rng.normal(size=matrix.shape) + 1j * rng.normal(size=matrix.shape)

        def sensitivities(basis):
            left = np.asarray(left_eigenvectors_jax(operator, jnp.asarray(basis)))
            per_mode = [
                (left[:, k] @ direction @ basis[:, k]) / (left[:, k] @ basis[:, k]) for k in range(basis.shape[1])
            ]
            gram = left.T @ basis
            averaged = np.trace(np.linalg.solve(gram, left.T @ direction @ basis)) / basis.shape[1]
            return per_mode, averaged

        angle = 0.7
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated = pair @ rotation

        per_mode_a, averaged_a = sensitivities(pair)
        per_mode_b, averaged_b = sensitivities(rotated)
        assert abs(averaged_a - averaged_b) / abs(averaged_a) < 1e-10
        assert abs(per_mode_a[0] - per_mode_b[0]) / abs(per_mode_a[0]) > 1e-3


class TestPermittivityArrayGradient:
    """One number per cell: ``d n_eff / d eps(cell)`` out of one backward.

    The scalar-scale checks above prove the chain is connected; these prove the *map* is right,
    which is what an optimiser over a design region actually consumes. A per-cell finite difference
    is not the way to check it - moving one cell of a 1440-cell cross-section by 1e-5 moves ``n_eff``
    by 1e-11, i.e. at ARPACK's own convergence floor, and the check then measures the eigensolver's
    noise (measured: 4e-6 relative, all of it noise). A directional derivative along a random
    perturbation of *every* cell has the same information and a usable signal-to-noise ratio.
    """

    @staticmethod
    def _neff_of_delta(problem, part="neff", num_modes=4):
        """``delta -> n_eff`` for a real per-cell perturbation added to all three eps components."""

        def f(delta):
            eps = [jnp.asarray(component) + delta.astype(jnp.complex128) for component in problem.eps]
            _, _, neff, keff = solve_modes_diagonal_jax(
                *eps, *[jnp.asarray(c) for c in problem.mu], problem.der, problem.steps, K0, num_modes, problem.guess
            )
            return neff[0] if part == "neff" else keff[0]

        return f

    @staticmethod
    def _directional(f, n, step=1e-3, seed=0):
        """The adjoint's directional derivative and the matching central difference."""
        rng = np.random.default_rng(seed)
        direction = rng.normal(size=n)
        direction /= np.linalg.norm(direction)
        zero = jnp.zeros(n, dtype=jnp.float64)
        gradient = np.asarray(jax.grad(f)(zero))
        moved = jnp.asarray(step * direction)
        finite_difference = (float(f(zero + moved)) - float(f(zero - moved))) / (2 * step)
        return float(np.dot(gradient, direction)), finite_difference, gradient

    def test_the_phase_shifter_cross_section(self, float64):
        """The real device: Si guide, TiN heater, oxide, 60 x 24 cells at 100 nm."""
        eps, _, _ = _phase_shifter_cross_section(0.1)
        problem = _uniform(eps, 100e-9)
        adjoint, finite_difference, gradient = self._directional(self._neff_of_delta(problem), problem.n)
        assert adjoint == pytest.approx(finite_difference, rel=1e-8)
        assert gradient.shape == (problem.n,)
        # The sensitivity is the mode's own energy density: it lives on the guide, not on the walls.
        cells = gradient.reshape(problem.nx, problem.ny)
        assert np.abs(cells[[0, -1], :]).max() < 1e-3 * np.abs(cells).max()

    def test_the_loss_part_of_the_same_solve(self, float64):
        """``d k_eff / d eps`` comes off the same backward - the eigenvalue is holomorphic in eps."""
        eps, _, _ = _phase_shifter_cross_section(0.1)
        problem = _uniform(eps, 100e-9)
        adjoint, finite_difference, _ = self._directional(self._neff_of_delta(problem, part="keff"), problem.n)
        assert adjoint == pytest.approx(finite_difference, rel=1e-8)

    def test_a_non_uniform_grid(self, float64):
        """A graded mesh, where the primal and dual steps differ cell by cell."""
        problem = _graded(_strip(16, 14), 60e-9)
        adjoint, finite_difference, _ = self._directional(self._neff_of_delta(problem), problem.n)
        assert adjoint == pytest.approx(finite_difference, rel=1e-8)

    @staticmethod
    def _oracle_map(problem, mode):
        """``d n_eff[mode] / d eps(cell)`` from ``jnp.linalg.eig``, one number per cell."""

        def dense_neff(delta):
            eps = [jnp.asarray(component) + delta.astype(jnp.complex128) for component in problem.eps]
            operator = assemble_mode_operator_jax(
                *eps, *[jnp.asarray(c) for c in problem.mu], problem.der, K0, problem.steps
            )
            values, _ = dense_mode_eigs(operator)
            return jnp.real(jnp.sqrt(-values[mode]))

        return np.asarray(jax.grad(dense_neff)(jnp.zeros(problem.n, dtype=jnp.float64)))

    def test_the_whole_map_matches_the_dense_oracle(self, float64):
        """Every cell of the sparse adjoint against JAX's own eigenvalue derivative.

        On a cross-section whose fundamental is isolated, the per-mode map is well defined and the
        two must agree cell by cell. 14 x 12 is the smallest strip here that actually guides
        (``n_eff`` 1.995 above a 1.55 cladding); see the degenerate case below for why that matters.
        """
        problem = _uniform(_strip(14, 12), 60e-9)
        oracle = self._oracle_map(problem, 0)
        adjoint = np.asarray(jax.grad(self._neff_of_delta(problem))(jnp.zeros(problem.n, dtype=jnp.float64)))
        assert np.abs(adjoint - oracle).max() / np.abs(oracle).max() < 1e-10

    def test_on_a_degenerate_pair_only_the_block_average_is_comparable(self, float64):
        """The 12 x 10 strip does not guide: its two leading modes are wall artefacts.

        They sit at exactly the cladding index and are degenerate to 7e-15, so the *per-mode* map is
        a property of whichever basis the solver returned - the dense oracle's own basis differs from
        ARPACK's, and the two maps disagree by 50 % of full scale. The block-averaged map, which is
        what this backward returns, is basis-independent and agrees with the averaged oracle to 3e-14.
        This is the degeneracy statement measured on a case nobody constructed for it.
        """
        problem = _uniform(_strip(12, 10), 60e-9)
        _, _, neff, _ = solve_modes_diagonal(*problem.eps, *problem.mu, problem.der, K0, 4, problem.guess)
        assert neff[0] == pytest.approx(N_SIO2, abs=1e-9)
        assert abs(neff[0] - neff[1]) / abs(neff[0]) < 1e-12

        adjoint = np.asarray(jax.grad(self._neff_of_delta(problem))(jnp.zeros(problem.n, dtype=jnp.float64)))
        first, second = self._oracle_map(problem, 0), self._oracle_map(problem, 1)
        scale = np.abs(first).max()
        assert np.abs(first - adjoint).max() / scale > 0.1
        assert np.abs(0.5 * (first + second) - adjoint).max() / scale < 1e-10

    def test_it_reproduces_the_field_level_reciprocity_formula(self, float64):
        """The matrix contraction here and the field integral in ``mode_adjoint`` are one identity.

        ``fdtdx.core.physics.mode_adjoint`` differentiates the same eigenvalue by the Lorentz-reciprocity
        integral ``0.5 E_c^2 w / flux`` evaluated on the returned mode field; this module contracts
        ``u_i v_j / (u^T v)`` over the operator's nonzeros. They are the same statement written at
        two levels, and on the same cross-section they agree cell by cell - which is the check that
        the JAX assembly did not quietly change the operator the field formula was derived for.
        """
        from fdtdx.constants import c as speed_of_light
        from fdtdx.core.physics.mode_adjoint import ModeSolveSettings, mode_sensitivity

        eps2d = _strip(20, 16)
        problem = _uniform(eps2d, 50e-9)
        volume = np.broadcast_to(eps2d[None, :, :, None], (3, problem.nx, problem.ny, 1)).copy()
        settings = ModeSolveSettings.create(
            frequency=speed_of_light / LAM, resolution=50e-9, mode_index=0, mode_backend="fdtdmex"
        )
        neff_field, sensitivity = mode_sensitivity(jnp.asarray(volume), settings)
        # One shared perturbation drives all three diagonal components, so the three add.
        field_level = np.real(np.asarray(sensitivity)[:, :, :, 0].sum(axis=0)).ravel()

        f = self._neff_of_delta(problem)
        matrix_level = np.asarray(jax.grad(f)(jnp.zeros(problem.n, dtype=jnp.float64)))
        assert float(f(jnp.zeros(problem.n, dtype=jnp.float64))) == pytest.approx(
            float(jnp.real(neff_field)), rel=1e-11
        )
        assert np.abs(matrix_level - field_level).max() / np.abs(field_level).max() < 1e-12


class TestPrecision:
    """The operator is double precision or it does not exist."""

    def test_assembly_refuses_without_x64(self):
        previous = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", False)
        try:
            problem = _uniform(_strip(6, 6), 60e-9)
            with pytest.raises(ValueError, match="double precision"):
                problem.jax_operator()
        finally:
            jax.config.update("jax_enable_x64", previous)

    def test_reconstruction_refuses_without_x64(self, float64):
        """Assembly is not the only stage that has to be double: so is the field reconstruction."""
        problem = _uniform(_strip(8, 8), 60e-9)
        operator = problem.jax_operator()
        values, vectors = dense_mode_eigs(operator)
        previous = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", False)
        try:
            with pytest.raises(ValueError, match="double precision"):
                reconstruct_fields_jax(operator, vectors[:, :1], values[:1])
        finally:
            jax.config.update("jax_enable_x64", previous)

    def test_every_stage_stays_complex128(self, float64):
        problem = _uniform(_strip(8, 8), 60e-9)
        operator = problem.jax_operator()
        assert operator.mat.data.dtype == jnp.complex128
        assert operator.qmat.data.dtype == jnp.complex128
        values, vectors = dense_mode_eigs(operator)
        field_e, field_h, neff, _ = reconstruct_fields_jax(operator, vectors[:, :2], values[:2])
        assert field_e.dtype == jnp.complex128
        assert field_h.dtype == jnp.complex128
        assert neff.dtype == jnp.float64

    def test_a_float32_cross_section_is_lifted_before_assembly(self, float64):
        """A caller handing in complex64 gets a complex128 operator, not a rounded one."""
        problem = _uniform(_strip(8, 8), 60e-9)
        single = [jnp.asarray(c, dtype=jnp.complex64) for c in problem.eps]
        operator = assemble_mode_operator_jax(
            *single, *[jnp.asarray(c) for c in problem.mu], problem.der, K0, problem.steps
        )
        assert operator.mat.data.dtype == jnp.complex128


class TestUnderJit:
    """The seam has to survive ``jax.jit``, because that is where an optimisation loop runs it."""

    @staticmethod
    def _neff(problem):
        def f(delta):
            eps = [jnp.asarray(component) + delta.astype(jnp.complex128) for component in problem.eps]
            _, _, neff, _ = solve_modes_diagonal_jax(
                *eps, *[jnp.asarray(c) for c in problem.mu], problem.der, problem.steps, K0, 4, problem.guess
            )
            return neff[0]

        return f

    def test_the_forward_traces(self, float64):
        problem = _uniform(_strip(14, 12), 60e-9)
        f = self._neff(problem)
        zero = jnp.zeros(problem.n, dtype=jnp.float64)
        assert float(jax.jit(f)(zero)) == pytest.approx(float(f(zero)), rel=1e-12)

    def test_the_gradient_traces_and_matches_the_eager_one(self, float64):
        """The degeneracy grouping in the backward is array algebra, not a Python loop over values.

        Grouping the eigenvalues with numpy would raise ``TracerArrayConversionError`` here: the
        backward runs inside whatever transformation the caller applied, and under ``jit`` the
        eigenvalues it groups are tracers.
        """
        problem = _uniform(_strip(14, 12), 60e-9)
        f = self._neff(problem)
        zero = jnp.zeros(problem.n, dtype=jnp.float64)
        eager = np.asarray(jax.grad(f)(zero))
        jitted = np.asarray(jax.jit(jax.grad(f))(zero))
        assert np.abs(eager - jitted).max() / np.abs(eager).max() < 1e-13

    def test_a_degenerate_block_traces_too(self, float64):
        """The square guide, whose two fundamentals share one invariant subspace."""
        problem = _uniform(np.full((12, 12), N_SI**2), 60e-9)
        f = self._neff(problem)
        zero = jnp.zeros(problem.n, dtype=jnp.float64)
        eager = np.asarray(jax.grad(f)(zero))
        jitted = np.asarray(jax.jit(jax.grad(f))(zero))
        assert np.abs(eager - jitted).max() / np.abs(eager).max() < 1e-13

    def test_the_traceable_mask_is_the_python_grouping(self, float64):
        """``degeneracy_mask`` and ``degenerate_groups`` are one relation in two forms."""
        problem = _uniform(np.full((12, 12), N_SI**2), 60e-9)
        values, _ = dense_mode_eigs(problem.jax_operator())
        head = values[:4]
        mask = np.asarray(degeneracy_mask(head))
        expected = np.zeros((4, 4))
        for group in degenerate_groups(np.asarray(head)):
            for i in group:
                for j in group:
                    expected[i, j] = 1.0
        assert np.array_equal(mask, expected)
        assert mask[0, 1] == 1.0 and mask[0, 2] == 0.0
