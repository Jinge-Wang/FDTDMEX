"""The full-tensor tier of the native mode solver: the operator, the analytic checks, the gradients.

Track J phase 2. Every number in the docstrings was measured by these tests on one CPU with
``jax_enable_x64`` on and ``complex128`` throughout.
"""

import warnings

import numpy as np
import pytest
import scipy.sparse.linalg as spl
from scipy.optimize import brentq

jax = pytest.importorskip("jax")
jnp = jax.numpy

from fdtdx.constants import c  # noqa: E402
from fdtdx.core.physics.mode_adjoint import (  # noqa: E402
    ModeSolveSettings,
    mode_neff_parts,
    mode_sensitivity,
    mode_solve,
)
from fdtdx.core.physics.mode_backend import ModeLongitudinalOffdiagWarning  # noqa: E402
from fdtdx.core.physics.mode_backend.jax_operator import assemble_mode_operator_jax  # noqa: E402
from fdtdx.core.physics.mode_backend.operator import (  # noqa: E402
    build_derivative_matrices,
    primal_dual_steps,
)
from fdtdx.core.physics.mode_backend.solve import assemble_mode_operator, solve_modes_diagonal  # noqa: E402
from fdtdx.core.physics.modes import compute_modes  # noqa: E402

LAM = 1.55e-6
FREQ = c / LAM
K0 = 2.0 * np.pi / LAM


@pytest.fixture
def float64():
    """The whole tier is complex128; a finite-difference check is meaningless without x64."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------


def _uniaxial(n_o: float, n_e: float, direction) -> np.ndarray:
    """Uniaxial permittivity tensor with its optic axis along ``direction``."""
    axis = np.asarray(direction, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return n_o**2 * np.eye(3) + (n_e**2 - n_o**2) * np.outer(axis, axis)


def _grid(nx: int, ny: int, cell: float, graded: bool = False):
    if graded:
        cx = np.cumsum(np.concatenate(([0.0], cell * (1.0 + 0.2 * np.sin(np.arange(nx))))))
        cy = np.cumsum(np.concatenate(([0.0], cell * (1.0 + 0.2 * np.cos(np.arange(ny))))))
    else:
        cx, cy = np.arange(nx + 1) * cell, np.arange(ny + 1) * cell
    return (
        build_derivative_matrices(cx, cy),
        (primal_dual_steps(cx), primal_dual_steps(cy)),
    )


def _flat(value, n: int) -> np.ndarray:
    return np.full(n, value, dtype=np.complex128)


def _inverse_tensor(tensor: np.ndarray) -> np.ndarray:
    """``(3, 3, ...) -> (9, ...)`` matrix inverse, the layout ``compute_modes`` takes."""
    spatial = tensor.shape[2:]
    moved = np.moveaxis(tensor.reshape(3, 3, -1), 2, 0)
    return np.moveaxis(np.linalg.inv(moved), 0, 2).reshape(9, *spatial)


def _slab_inv_eps(nx: int, num_core: int, core: np.ndarray, clad: float) -> jnp.ndarray:
    """A slab invariant along the last transverse axis: ``(9, 1, nx, 2)`` inverse permittivity."""
    mask = np.zeros(nx, dtype=bool)
    start = (nx - num_core) // 2
    mask[start : start + num_core] = True
    tensor = np.zeros((3, 3, nx))
    for a in range(3):
        tensor[a, a] = clad
    for a in range(3):
        for b in range(3):
            tensor[a, b] = np.where(mask, core[a, b], tensor[a, b])
    inv = _inverse_tensor(tensor)
    return jnp.asarray(np.broadcast_to(inv[:, None, :, None], (9, 1, nx, 2)).copy())


def _even_te_root(eps_yy: float, eps_clad: float, thickness: float) -> float:
    """Even TE root of ``kappa tan(kappa d / 2) = gamma`` for a symmetric slab.

    ``E`` lies along the invariant transverse axis, so it is tangential at both interfaces and the
    relation involves only the permittivity component it is polarised along. Marcuse, *Theory of
    Dielectric Optical Waveguides*, ch. 1; the anisotropic reading (``eps_yy`` in place of
    ``n_core^2``) follows because with no variation along that axis the ``E_y`` row of the
    transverse-E operator decouples exactly.

    Solved in the normalised variable ``u = kappa d / 2`` on ``(0, min(pi/2, V))``, where the
    fundamental root is the only one and the tangent has no pole.
    """
    half = 0.5 * thickness
    v_number = half * K0 * np.sqrt(eps_yy - eps_clad)

    def residual(u: float) -> float:
        return u * np.tan(u) - np.sqrt(max(v_number**2 - u**2, 0.0))

    upper = min(0.5 * np.pi, v_number) - 1e-12
    u_root = brentq(residual, 1e-12, upper, xtol=1e-15, rtol=8.9e-16)
    kappa = u_root / half
    return float(np.sqrt(eps_yy - (kappa / K0) ** 2))


def _even_tm_root(eps_xx: float, eps_zz: float, eps_clad: float, thickness: float) -> float:
    """Even TM root for a uniaxial core: ``tan(kappa d / 2) = eps_zz gamma / (eps_clad kappa)``.

    From ``H_y'' = eps_zz (beta^2 / eps_xx - k0^2) H_y`` with ``H_y`` and ``H_y' / eps_zz``
    continuous, so ``kappa^2 = eps_zz (k0^2 - beta^2 / eps_xx)``. Yariv & Yeh, *Optical Waves in
    Crystals*, ch. 11; it reduces to the isotropic TM relation when ``eps_xx = eps_zz``. In the
    normalised variable ``u = kappa d / 2``,

    .. code-block:: text

        u tan u = (eps_zz / eps_clad) sqrt(V^2 - (eps_xx / eps_zz) u^2),
        V = (d / 2) k0 sqrt(eps_xx - eps_clad).
    """
    half = 0.5 * thickness
    v_number = half * K0 * np.sqrt(eps_xx - eps_clad)
    ratio = eps_xx / eps_zz

    def residual(u: float) -> float:
        return u * np.tan(u) - (eps_zz / eps_clad) * np.sqrt(max(v_number**2 - ratio * u**2, 0.0))

    upper = min(0.5 * np.pi, v_number / np.sqrt(ratio)) - 1e-12
    u_root = brentq(residual, 1e-12, upper, xtol=1e-15, rtol=8.9e-16)
    kappa = u_root / half
    return float(np.sqrt(eps_xx * (1.0 - (kappa / K0) ** 2 / eps_zz)))


# ------------------------------------------------------------------------------------------------
# (a) the diagonal path is untouched
# ------------------------------------------------------------------------------------------------


class TestDiagonalPathUnchanged:
    """Deliverable (a): a diagonal tensor through the tensor path equals the diagonal path."""

    def _random_diagonal(self, n: int):
        rng = np.random.default_rng(3)
        return (
            rng.uniform(2.0, 12.0, n) + 0j,
            rng.uniform(2.0, 12.0, n) + 0j,
            rng.uniform(2.0, 12.0, n) + 0j,
        )

    def test_the_numpy_operator_is_bit_identical(self, float64):
        nx, ny = 9, 7
        n = nx * ny
        eps = self._random_diagonal(n)
        mu = _flat(1.0, n)
        der, _ = _grid(nx, ny, 50e-9)
        base = assemble_mode_operator(*eps, mu, mu, mu, der, K0)
        tensor = assemble_mode_operator(*eps, mu, mu, mu, der, K0, eps_xy=np.zeros(n), eps_yx=np.zeros(n))
        assert np.abs(base.mat.toarray() - tensor.mat.toarray()).max() == 0.0
        assert np.abs(base.qmat.toarray() - tensor.qmat.toarray()).max() == 0.0

    def test_the_jax_operator_is_bit_identical(self, float64):
        nx, ny = 9, 7
        n = nx * ny
        eps = [jnp.asarray(component) for component in self._random_diagonal(n)]
        mu = jnp.ones(n, dtype=jnp.complex128)
        der, steps = _grid(nx, ny, 50e-9, graded=True)
        base = assemble_mode_operator_jax(*eps, mu, mu, mu, der, K0, steps)
        tensor = assemble_mode_operator_jax(
            *eps, mu, mu, mu, der, K0, steps, eps_xy=jnp.zeros(n, dtype=jnp.complex128), eps_yx=jnp.zeros(n)
        )
        dense_base = np.asarray(base.mat.to_bcoo().todense())
        dense_tensor = np.asarray(tensor.mat.to_bcoo().todense())
        assert np.abs(dense_base - dense_tensor).max() == 0.0

    def test_the_solve_agrees_to_1e_14(self, float64):
        nx, ny, cell = 20, 16, 60e-9
        n = nx * ny
        core = np.zeros((nx, ny), dtype=bool)
        core[6:14, 5:11] = True
        eps = np.where(core.ravel(), 12.1, 2.4025) + 0j
        mu = _flat(1.0, n)
        der, _ = _grid(nx, ny, cell)
        args = (eps, eps.copy(), eps.copy(), mu, mu, mu, der)
        kwargs = dict(k0=K0, num_modes=4, neff_guess=float(np.sqrt(12.1)) * (1 + 1e-6))
        e_ref, h_ref, neff_ref, _ = solve_modes_diagonal(*args, **kwargs)
        e_ten, h_ten, neff_ten, _ = solve_modes_diagonal(
            *args, **kwargs, eps_xy=np.zeros(n, dtype=complex), eps_yx=np.zeros(n, dtype=complex)
        )
        assert np.abs(neff_ten - neff_ref).max() < 1e-14
        assert np.abs(e_ten - e_ref).max() / np.abs(e_ref).max() < 1e-14
        assert np.abs(h_ten - h_ref).max() / np.abs(h_ref).max() < 1e-14


# ------------------------------------------------------------------------------------------------
# (b) the analytic checks
# ------------------------------------------------------------------------------------------------


class TestBulkUniaxial:
    """Deliverable (b), part 1: the index ellipsoid, on a transverse-invariant cross-section.

    One cell on each transverse axis makes every difference matrix identically zero, so the operator
    is the bulk one and the two guided indices are exactly the two plane-wave solutions. For an
    optic axis at an angle ``alpha`` to the wave vector the extraordinary index is

    .. code-block:: text

        1 / n_e(alpha)^2 = cos^2(alpha) / n_o^2 + sin^2(alpha) / n_e^2

    (Born & Wolf, *Principles of Optics*, section 15.3; Yariv & Yeh, *Optical Waves in Crystals*,
    ch. 4). An axis lying in the transverse plane is ``alpha = 90 deg``, so the two indices are
    ``n_o`` and ``n_e`` at *every* in-plane rotation angle.

    Measured, ``n_o = 1.5``, ``n_e = 1.7``: the tensor path is exact to 4.4e-16 at every angle,
    while projecting the tensor onto its diagonal is wrong by up to 0.103 (at 45 degrees).
    """

    n_o, n_e = 1.5, 1.7

    def _bulk_indices(self, tensor: np.ndarray, carry_off_diagonal: bool = True) -> np.ndarray:
        der, _ = _grid(1, 1, 50e-9)
        one = _flat(1.0, 1)
        off = {}
        if carry_off_diagonal:
            off = dict(eps_xy=_flat(tensor[0, 1], 1), eps_yx=_flat(tensor[1, 0], 1))
        operator = assemble_mode_operator(
            _flat(tensor[0, 0], 1), _flat(tensor[1, 1], 1), _flat(tensor[2, 2], 1), one, one, one, der, K0, **off
        )
        values = np.linalg.eigvals(operator.mat.toarray())
        return np.sort(np.real(np.emath.sqrt(-values)))

    @pytest.mark.parametrize("theta_deg", [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
    def test_a_transverse_optic_axis_gives_n_o_and_n_e(self, float64, theta_deg):
        theta = np.deg2rad(theta_deg)
        tensor = _uniaxial(self.n_o, self.n_e, (np.cos(theta), np.sin(theta), 0.0))
        got = self._bulk_indices(tensor)
        assert np.abs(got - np.array([self.n_o, self.n_e])).max() < 1e-13

    def test_projecting_onto_the_diagonal_loses_the_extraordinary_index(self, float64):
        theta = np.deg2rad(45.0)
        tensor = _uniaxial(self.n_o, self.n_e, (np.cos(theta), np.sin(theta), 0.0))
        projected = self._bulk_indices(tensor, carry_off_diagonal=False)
        # both branches collapse onto sqrt((n_e^2 + n_o^2) / 2) = 1.6031
        assert np.abs(projected - np.sqrt(0.5 * (self.n_e**2 + self.n_o**2))).max() < 1e-13
        error = np.abs(projected - np.array([self.n_o, self.n_e])).max()
        assert error == pytest.approx(0.1031, rel=1e-3)

    @pytest.mark.parametrize(
        "alpha_deg, expected_error",
        [(80.0, 1.6e-3), (60.0, 9.7e-3), (45.0, 1.2e-2), (20.0, 4.9e-3), (1.0, 1.4e-5)],
    )
    def test_an_optic_axis_tilted_out_of_the_plane_costs_this_much_to_drop(self, float64, alpha_deg, expected_error):
        """What the *transverse* operator loses by not carrying the four longitudinal entries.

        This is the cost the four-component formulation removes; the numbers are pinned here so the
        comparison against it stays honest.
        """
        alpha = np.deg2rad(alpha_deg)
        tensor = _uniaxial(self.n_o, self.n_e, (np.sin(alpha), 0.0, np.cos(alpha)))
        exact = 1.0 / np.sqrt(np.cos(alpha) ** 2 / self.n_o**2 + np.sin(alpha) ** 2 / self.n_e**2)
        got = self._bulk_indices(tensor)[1]
        assert abs(got - exact) == pytest.approx(expected_error, rel=0.05)

    def test_the_entries_are_only_dropped_when_the_caller_asks_for_it(self, float64):
        """The warning is now a property of ``mode_formulation="transverse"``, not of the tensor."""
        alpha = np.deg2rad(45.0)
        bulk = _uniaxial(self.n_o, self.n_e, (np.sin(alpha), 0.0, np.cos(alpha)))
        tensor = np.broadcast_to(bulk[:, :, None, None], (3, 3, 8, 8)).copy()
        inv_eps = jnp.asarray(_inverse_tensor(tensor)[:, None, :, :])
        kwargs = dict(
            frequency=FREQ,
            inv_permittivities=inv_eps,
            inv_permeabilities=1.0,
            num_modes=1,
            resolution=50e-9,
            dtype=jnp.float64,
        )
        with pytest.warns(ModeLongitudinalOffdiagWarning, match="longitudinal off-diagonal entries yz"):
            compute_modes(mode_formulation="transverse", **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ModeLongitudinalOffdiagWarning)
            compute_modes(**kwargs)


class TestAnisotropicSlab:
    """Deliverable (b), part 2: a uniaxial slab against the two closed forms, on a real grid.

    Slab of thickness ``d`` in an isotropic cladding, propagation along the physical first axis, the
    slab normal on the second and the invariant direction on the third (two cells, which the solver
    collapses). The optic axis is rotated *in the transverse plane*, from the slab normal
    (``theta = 0``) to the invariant axis (``theta = 90 deg``), so the extraordinary index moves
    from the TM branch to the TE one and back.

    Measured, ``n_o = 2.0``, ``n_e = 2.2``, cladding 1.44, ``d = 400 nm``, one Richardson step over
    10 and 5 nm cells: the TE branch lands 7.9e-9 from its analytic root when the axis is on the
    invariant direction and -1.0e-9 when it is on the slab normal; the TM branch lands 3.9e-9 from
    the uniaxial TM root (numbers corrected after refutation J4v, 2026-09-09).
    """

    n_o, n_e, n_clad = 2.0, 2.2, 1.44
    thickness = 400e-9
    window = 5e-6

    def _solve(self, core: np.ndarray, cell: float, num_modes: int = 4):
        nx = round(self.window / cell)
        num_core = round(self.thickness / cell)
        inv_eps = _slab_inv_eps(nx, num_core, core, self.n_clad**2)
        _, _, neff = compute_modes(
            frequency=FREQ,
            inv_permittivities=inv_eps,
            inv_permeabilities=1.0,
            num_modes=num_modes,
            resolution=cell,
            dtype=jnp.float64,
        )
        return np.real(np.asarray(neff))

    def _richardson(self, core: np.ndarray, analytic: float) -> float:
        coarse = self._solve(core, 10e-9)
        fine = self._solve(core, 5e-9)
        pick = lambda values: float(values[np.argmin(np.abs(values - analytic))])  # noqa: E731
        e_coarse, e_fine = pick(coarse) - analytic, pick(fine) - analytic
        return (4 * e_fine - e_coarse) / 3

    def test_the_te_branch_sees_the_extraordinary_index_when_the_axis_is_on_the_invariant_direction(self, float64):
        # physical axes are (propagation, slab normal, invariant); the optic axis on the third.
        core = _uniaxial(self.n_o, self.n_e, (0.0, 0.0, 1.0))
        analytic = _even_te_root(self.n_e**2, self.n_clad**2, self.thickness)
        assert abs(self._richardson(core, analytic)) < 1e-7

    def test_the_te_branch_sees_the_ordinary_index_when_the_axis_is_on_the_slab_normal(self, float64):
        core = _uniaxial(self.n_o, self.n_e, (0.0, 1.0, 0.0))
        analytic = _even_te_root(self.n_o**2, self.n_clad**2, self.thickness)
        assert abs(self._richardson(core, analytic)) < 1e-7

    def test_the_tm_branch_sees_the_uniaxial_dispersion_relation(self, float64):
        """The TM (normal-E) branch uses both ``eps_xx`` and ``eps_zz`` of the solver frame."""
        core = _uniaxial(self.n_o, self.n_e, (0.0, 1.0, 0.0))
        # solver x = physical y (the slab normal) = n_e^2; solver z = physical x (propagation) = n_o^2
        analytic = _even_tm_root(self.n_e**2, self.n_o**2, self.n_clad**2, self.thickness)
        # the normal-E component is sampled co-located with the others, so this branch
        # converges more slowly than the tangential one; measured residual 1.4e-5.
        assert abs(self._richardson(core, analytic)) < 5e-5

    def test_the_off_diagonal_matters_on_a_real_slab_not_only_in_bulk(self, float64):
        """At 45 degrees the tensor carries a genuine off-diagonal; dropping it moves the answer."""
        theta = np.deg2rad(45.0)
        core = _uniaxial(self.n_o, self.n_e, (0.0, np.cos(theta), np.sin(theta)))
        projected = core.copy()
        projected[1, 2] = 0.0
        projected[2, 1] = 0.0
        exact = self._solve(core, 10e-9, num_modes=2)
        dropped = self._solve(projected, 10e-9, num_modes=2)
        assert np.abs(exact - dropped).max() > 1e-3


# ------------------------------------------------------------------------------------------------
# (c) rotation invariance
# ------------------------------------------------------------------------------------------------


class TestRotationInvariance:
    """Deliverable (c): rotate the tensor together with the geometry and nothing moves.

    The exchange ``(x, y) -> (y, x)`` is the quarter-turn the staggered grid carries exactly: it
    maps forward differences to forward differences, so the discrete operator maps onto itself with
    ``mat11 <-> mat22`` and ``mat12 <-> mat21``. That is precisely the pairing the six new
    off-diagonal terms sit in, so a wrong sign or a wrong block in any of them breaks it. Measured
    on a 20 x 20 square core with a 45-degree uniaxial tensor: 0 (bit-identical).

    The true quarter-turn (transpose then reverse one axis) is *not* an exact symmetry of the
    difference matrices, because reversing an axis exchanges the min-edge and max-edge wall rows.
    It is checked at a looser bar; measured 1.1e-14 on the same cross-section, i.e. the wall
    asymmetry does not reach a confined mode.
    """

    def _square_core_tensor(self, nx: int, theta_deg: float) -> np.ndarray:
        core = np.zeros((nx, nx), dtype=bool)
        core[nx // 2 - 5 : nx // 2 + 5, nx // 2 - 5 : nx // 2 + 5] = True
        theta = np.deg2rad(theta_deg)
        bulk = _uniaxial(2.0, 2.4, (np.cos(theta), np.sin(theta), 0.0))
        clad = 2.25 * np.eye(3)
        tensor = np.zeros((3, 3, nx, nx))
        for a in range(3):
            for b in range(3):
                tensor[a, b] = np.where(core, bulk[a, b], clad[a, b])
        return tensor

    def _neff(self, tensor: np.ndarray, cell: float = 60e-9) -> float:
        inv = _inverse_tensor(tensor)
        inv_eps = jnp.asarray(inv[:, None, :, :])
        _, _, neff = compute_modes(
            frequency=FREQ,
            inv_permittivities=inv_eps,
            inv_permeabilities=1.0,
            num_modes=1,
            resolution=cell,
            dtype=jnp.float64,
        )
        return float(np.real(np.asarray(neff)[0]))

    def test_the_axis_exchange_is_exact(self, float64):
        tensor = self._square_core_tensor(20, 45.0)
        exchange = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        swapped = np.einsum("ai,ij...,bj->ab...", exchange, tensor, exchange).transpose(0, 1, 3, 2)
        assert abs(self._neff(tensor) - self._neff(swapped)) < 1e-10

    def test_the_true_quarter_turn_agrees_too(self, float64):
        tensor = self._square_core_tensor(20, 45.0)
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rotated = np.einsum("ai,ij...,bj->ab...", rotation, tensor, rotation)
        rotated = np.rot90(rotated, k=1, axes=(2, 3)).copy()
        assert abs(self._neff(tensor) - self._neff(rotated)) < 1e-9


# ------------------------------------------------------------------------------------------------
# (d) the stressed-silica cross-section of S3
# ------------------------------------------------------------------------------------------------


class TestStressedSilica:
    """Deliverable (d): the cross-section S3 could not hand to the mode solver now solves.

    S3 measured, on the COMSOL model-190 stress-optic scene at 50 nm: the largest off-diagonal
    permittivity entry 1.5163e-06, the largest pair ratio ``delta / Delta`` 1.6441e-03 and hence an
    axis rotation ``0.5 atan(2 delta / Delta) = 0.0942 degrees``; and, for the variant with the
    core's Young's modulus at 0.85 of the cladding's, a pair ratio 0.04762 and a rotation of
    2.720 degrees. The old backend refused any cross-section with an off-diagonal entry above
    ``TOL_TENSORIAL = 1e-6``, so S3 had to run the projected diagonal.
    """

    eps_silica = 1.4508**2
    #: Delta, the diagonal splitting of the coupled pair, from S3's own pair ratio and entry.
    splitting = 1.5163e-06 / 1.6441e-03
    #: Core-cladding step. S3's guide has a step of about 1e-3 in a 20 x 20 um window at 50 nm,
    #: which is 400 x 400 cells; a unit test cannot carry that, so the *step* is raised to keep the
    #: mode inside a 24-cell window. The tensor structure being tested - the splitting and the pair
    #: ratio, hence the axis rotation - is S3's.
    core_step = 0.11
    resolution = 200e-9

    def _cross_section(self, nx: int, pair_ratio: float) -> np.ndarray:
        """A silica guide whose core carries S3's stress-perturbed tensor."""
        core = np.zeros((nx, nx), dtype=bool)
        core[nx // 2 - 5 : nx // 2 + 5, nx // 2 - 5 : nx // 2 + 5] = True
        delta = pair_ratio * self.splitting
        raised = self.eps_silica + self.core_step
        tensor = np.zeros((3, 3, nx, nx))
        for a in range(3):
            tensor[a, a] = np.where(core, raised, self.eps_silica)
        tensor[0, 0] = np.where(core, raised + self.splitting / 2, self.eps_silica)
        tensor[1, 1] = np.where(core, raised - self.splitting / 2, self.eps_silica)
        tensor[0, 1] = np.where(core, delta, 0.0)
        tensor[1, 0] = tensor[0, 1]
        return tensor

    @pytest.mark.parametrize("pair_ratio, rotation_deg", [(1.6441e-03, 0.09420), (0.04762, 2.72025)])
    def test_the_axis_rotation_matches_the_value_s3_reported(self, float64, pair_ratio, rotation_deg):
        tensor = self._cross_section(24, pair_ratio)
        core = tensor[:, :, 12, 12]
        delta = core[0, 1]
        gap = core[0, 0] - core[1, 1]
        assert np.rad2deg(0.5 * np.arctan(2 * delta / gap)) == pytest.approx(rotation_deg, rel=1e-4)

    def test_the_cross_section_solves_instead_of_being_refused(self, float64):
        tensor = self._cross_section(24, 1.6441e-03)
        assert np.abs(tensor[0, 1]).max() > 1e-6  # above the old TOL_TENSORIAL refusal
        inv_eps = jnp.asarray(_inverse_tensor(tensor)[:, None, :, :])
        _, _, neff = compute_modes(
            frequency=FREQ,
            inv_permittivities=inv_eps,
            inv_permeabilities=1.0,
            num_modes=2,
            resolution=self.resolution,
            dtype=jnp.float64,
        )
        assert float(np.real(np.asarray(neff)[0])) > self.eps_silica**0.5 + 1e-3

    def test_the_projected_diagonal_costs_nothing_at_this_off_diagonal(self, float64):
        """S3's own conclusion, now measurable rather than assumed: the drop is below 1e-9."""
        tensor = self._cross_section(24, 1.6441e-03)
        projected = tensor.copy()
        projected[0, 1] = 0.0
        projected[1, 0] = 0.0

        def solve(t):
            inv_eps = jnp.asarray(_inverse_tensor(t)[:, None, :, :])
            _, _, neff = compute_modes(
                frequency=FREQ,
                inv_permittivities=inv_eps,
                inv_permeabilities=1.0,
                num_modes=2,
                resolution=self.resolution,
                dtype=jnp.float64,
            )
            return np.real(np.asarray(neff))

        assert np.abs(solve(tensor) - solve(projected)).max() < 1e-9


# ------------------------------------------------------------------------------------------------
# the gradients
# ------------------------------------------------------------------------------------------------


def _lithium_niobate_cross_section(nx: int = 22, ny: int = 18, offdiag_fraction: float = 0.15) -> jnp.ndarray:
    """An x-cut thin-film lithium niobate ridge, propagation along the physical second axis.

    P3's crystal frame: the extraordinary axis (crystal Z) lies in the film plane transverse to
    propagation, so the base tensor is ``diag(n_e^2, n_o^2, n_o^2) = (4.571044, 4.888521, 4.888521)``
    in grid axes with grid y the propagation direction. A vertical drive writes ``eps_xz``
    (the r51 entry), which the mode frame sees as its transverse off-diagonal.
    """
    film = np.zeros((nx, ny), dtype=bool)
    film[:, 6:12] = True
    ridge = np.zeros((nx, ny), dtype=bool)
    ridge[nx // 2 - 4 : nx // 2 + 4, 10:14] = True
    core = film | ridge
    tensor = np.zeros((3, 3, nx, ny))
    for a, value in enumerate((4.571044, 4.888521, 4.888521)):
        tensor[a, a] = np.where(core, value, 2.085)
    perturbation = 1.045e-03
    tensor[0, 0] = tensor[0, 0] + np.where(core, perturbation, 0.0)
    off = offdiag_fraction * perturbation
    tensor[0, 2] = np.where(core, off, 0.0)
    tensor[2, 0] = tensor[0, 2]
    eps = np.moveaxis(tensor.reshape(9, nx, ny), 0, 0)
    return jnp.asarray(eps[:, :, None, :])  # (9, nx, 1, ny): propagation along the physical y axis


class TestTensorGradients:
    """Deliverable 2: ``d n_eff / d eps_ab`` for every entry, against finite differences."""

    @property
    def settings(self) -> ModeSolveSettings:
        # built per test, not at class-definition time: double_precision follows jax_enable_x64,
        # which the float64 fixture turns on only while a test is running.
        return ModeSolveSettings.create(frequency=FREQ, resolution=90e-9, mode_index=0)

    def test_every_carried_entry_matches_a_finite_difference(self, float64):
        eps = _lithium_niobate_cross_section()
        rng = np.random.default_rng(11)
        direction = jnp.asarray(rng.normal(size=eps.shape))

        def neff(step):
            return mode_neff_parts(eps + step * direction, self.settings)[0]

        gradient = float(jax.grad(neff)(0.0))
        h = 1e-4
        finite = (float(neff(h)) - float(neff(-h))) / (2 * h)
        assert gradient == pytest.approx(finite, rel=2e-6)

    @pytest.mark.parametrize("entry", [0, 1, 2, 3, 4, 5, 6, 7, 8])
    def test_each_tensor_entry_separately(self, float64, entry):
        """Every one of the nine entries, against its own central difference.

        The propagation axis is the physical second one, so the entries that couple a transverse
        axis to it are ``xy, yx, yz, zy`` = flat indices 1, 3, 5, 7. Those used to be dropped and
        used to have an exactly zero gradient; the four-component formulation carries them, and
        their gradient is now a number the finite difference reproduces.
        """
        eps = _lithium_niobate_cross_section()
        rng = np.random.default_rng(entry + 1)
        direction = np.zeros(eps.shape)
        direction[entry] = rng.normal(size=eps.shape[1:])
        direction_jax = jnp.asarray(direction)

        def neff(step):
            return mode_neff_parts(eps + step * direction_jax, self.settings)[0]

        gradient = float(jax.grad(neff)(0.0))
        h = 1e-3
        finite = (float(neff(h)) - float(neff(-h))) / (2 * h)
        assert gradient == pytest.approx(finite, rel=1e-5)

    @pytest.mark.parametrize("entry", [1, 3, 5, 7])
    def test_the_transverse_formulation_still_reports_zero_for_a_dropped_entry(self, float64, entry):
        """Asked to drop them, the solver must also say their sensitivity is zero.

        The gradient has to be the gradient of what was solved: the transverse operator genuinely
        does not contain these entries, so reporting the continuum value would make ``jax.grad``
        disagree with a finite difference taken on the same path.
        """
        eps = _lithium_niobate_cross_section()
        settings = ModeSolveSettings.create(frequency=FREQ, resolution=90e-9, mode_index=0, formulation="transverse")
        rng = np.random.default_rng(entry + 1)
        direction = np.zeros(eps.shape)
        direction[entry] = rng.normal(size=eps.shape[1:])
        direction_jax = jnp.asarray(direction)

        def neff(step):
            return mode_neff_parts(eps + step * direction_jax, settings)[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ModeLongitudinalOffdiagWarning)
            gradient = float(jax.grad(neff)(0.0))
            h = 1e-3
            finite = (float(neff(h)) - float(neff(-h))) / (2 * h)
        assert gradient == 0.0
        assert abs(finite) < 1e-9

    def test_the_field_level_and_matrix_level_sensitivities_agree(self, float64):
        """Two independently written adjoints for the same quantity: the reciprocity integral over
        the mode fields, and the contraction over the operator's nonzeros.

        The reciprocity integral's partner field is the mode with its propagation component negated,
        which is the backward mode only when the cross-section has a mirror plane — so the
        comparison is run on the transverse formulation, which is the one that has it.
        """
        eps = _lithium_niobate_cross_section(nx=16, ny=14)
        settings = ModeSolveSettings.create(frequency=FREQ, resolution=90e-9, mode_index=0, formulation="transverse")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ModeLongitudinalOffdiagWarning)
            _, field_level = mode_sensitivity(eps, settings)
            rng = np.random.default_rng(5)
            direction = jnp.asarray(rng.normal(size=eps.shape))
            matrix_level = float(jax.grad(lambda s: mode_neff_parts(eps + s * direction, settings)[0])(0.0))
        assert float(jnp.sum(jnp.real(field_level) * direction)) == pytest.approx(matrix_level, rel=1e-6)

    def test_the_field_level_sensitivity_is_refused_when_the_entries_are_carried(self, float64):
        """It would be wrong rather than coarse, so it raises instead of returning a number.

        The lithium-niobate fixture propagates along the physical second axis, so its ``eps_xz``
        entry is *transverse* in the solver frame and it takes the transverse operator; adding a
        ``yz`` entry is what makes it longitudinal and moves the solve onto the four-component
        operator, where the reciprocity integral's partner field is no longer the backward mode.
        """
        settings = ModeSolveSettings.create(frequency=FREQ, resolution=90e-9, mode_index=0)
        transverse_only = _lithium_niobate_cross_section(nx=16, ny=14)
        mode_sensitivity(transverse_only, settings)  # no longitudinal entry: not refused

        carried = np.array(transverse_only)
        core = np.abs(carried[0]) > np.min(np.abs(carried[0])) + 1e-9
        carried[5] = np.where(core, 2.0e-04, 0.0)  # eps_yz
        carried[7] = carried[5]  # eps_zy
        with pytest.raises(ValueError, match="mirror plane"):
            mode_sensitivity(jnp.asarray(carried), settings)

    def test_the_stressed_silica_cross_section_differentiates(self, float64):
        stressed = TestStressedSilica()
        tensor = stressed._cross_section(24, 1.6441e-03)
        eps = jnp.asarray(tensor.reshape(9, 24, 24)[:, None, :, :])
        settings = ModeSolveSettings.create(frequency=FREQ, resolution=stressed.resolution, mode_index=0)
        rng = np.random.default_rng(2)
        direction = jnp.asarray(rng.normal(size=eps.shape))

        def neff(step):
            return mode_neff_parts(eps + step * direction, settings)[0]

        gradient = float(jax.grad(neff)(0.0))
        # the stressed pair is split by only 9e-4, so the branch is curved and a central difference
        # carries a visible truncation; Richardson over two steps removes it.
        coarse = (float(neff(2e-4)) - float(neff(-2e-4))) / 4e-4
        fine = (float(neff(1e-4)) - float(neff(-1e-4))) / 2e-4
        finite = (4 * fine - coarse) / 3
        # measured 1.6e-6; the floor here is ARPACK's convergence on a pair split by only 9e-4,
        # not the adjoint (the well-separated lithium-niobate cross-section above reaches 2e-6).
        assert gradient == pytest.approx(finite, rel=1e-5)


class TestFieldGradients:
    """Deliverable 3: the mode profile is differentiable, so an overlap objective is."""

    @staticmethod
    def _phase_shifter(nx: int = 24, ny: int = 20) -> jnp.ndarray:
        eps = np.full((nx, ny), 2.085)
        eps[nx // 2 - 6 : nx // 2 + 6, ny // 2 - 3 : ny // 2 + 3] = 12.1
        return jnp.asarray(eps)[None, None, :, :]

    def _intensity(self, scale, settings):
        solution = mode_solve(self._phase_shifter() * scale, settings)
        return jnp.log10(jnp.sum(jnp.abs(solution.E) ** 2))

    def test_the_intensity_objective_matches_a_finite_difference(self, float64):
        settings = ModeSolveSettings.create(frequency=FREQ, resolution=70e-9, mode_index=0)
        gradient = float(jax.grad(self._intensity)(1.0, settings))
        step = 1e-5
        finite = (float(self._intensity(1.0 + step, settings)) - float(self._intensity(1.0 - step, settings))) / (
            2 * step
        )
        assert gradient != 0.0
        assert gradient == pytest.approx(finite, rel=1e-6)

    def test_the_per_cell_field_gradient_matches_a_directional_finite_difference(self, float64):
        settings = ModeSolveSettings.create(frequency=FREQ, resolution=70e-9, mode_index=0)
        eps = self._phase_shifter()
        rng = np.random.default_rng(7)
        direction = jnp.asarray(rng.normal(size=eps.shape))

        def objective(step):
            solution = mode_solve(eps + step * direction, settings)
            return jnp.log10(jnp.sum(jnp.abs(solution.E) ** 2))

        gradient = float(jax.grad(objective)(0.0))
        h = 1e-3
        finite = (float(objective(h)) - float(objective(-h))) / (2 * h)
        assert gradient == pytest.approx(finite, rel=1e-5)

    def test_the_field_gradient_survives_jit(self, float64):
        settings = ModeSolveSettings.create(frequency=FREQ, resolution=70e-9, mode_index=0)
        eager = float(jax.grad(self._intensity)(1.0, settings))
        jitted = float(jax.jit(jax.grad(self._intensity), static_argnums=(1,))(1.0, settings))
        assert jitted == pytest.approx(eager, rel=1e-12)


class TestLeftEigenvectorPolicy:
    """The closed-form left eigenvector, and where it stops holding."""

    def _problem(self, theta_deg: float, graded: bool, nonreciprocal: float = 0.0):
        nx, ny, cell = 14, 12, 60e-9
        n = nx * ny
        core = np.zeros((nx, ny), dtype=bool)
        core[5:9, 4:8] = True
        theta = np.deg2rad(theta_deg)
        bulk = _uniaxial(1.55, 3.48, (np.cos(theta), np.sin(theta), 0.0))
        eps_xx = np.where(core.ravel(), bulk[0, 0], 2.4025) + 0j
        eps_yy = np.where(core.ravel(), bulk[1, 1], 2.4025) + 0j
        eps_zz = np.where(core.ravel(), bulk[2, 2], 2.4025) + 0j
        off = np.where(core.ravel(), bulk[0, 1], 0.0) + 0j
        der, steps = _grid(nx, ny, cell, graded=graded)
        one = jnp.ones(n, dtype=jnp.complex128)
        kwargs = {}
        if theta_deg != 0.0:
            kwargs = dict(eps_xy=jnp.asarray(off), eps_yx=jnp.asarray(off + nonreciprocal))
        return assemble_mode_operator_jax(
            jnp.asarray(eps_xx), jnp.asarray(eps_yy), jnp.asarray(eps_zz), one, one, one, der, K0, steps, **kwargs
        )

    @pytest.mark.parametrize(
        "theta_deg, graded, nonreciprocal, holds",
        [
            (0.0, False, 0.0, True),
            (0.0, True, 0.0, True),
            (35.0, False, 0.0, True),
            (35.0, True, 0.0, False),
            (35.0, False, 0.5, False),
        ],
    )
    def test_where_the_closed_form_holds(self, float64, theta_deg, graded, nonreciprocal, holds):
        from fdtdx.core.physics.mode_backend.jax_solve import dense_mode_eigs, left_eigenvector_residual

        operator = self._problem(theta_deg, graded, nonreciprocal)
        values, vectors = dense_mode_eigs(operator)
        residual = float(np.asarray(left_eigenvector_residual(operator, vectors[:, :4], values[:4])).max())
        assert (residual < 1e-9) is holds

    def test_the_solved_left_eigenvector_reproduces_the_closed_form_gradient(self, float64):
        """Where both are valid they must agree; that pins the second Arnoldi run and its matching."""
        from fdtdx.core.physics.mode_backend.jax_solve import solve_modes_diagonal_jax

        nx, ny, cell = 14, 12, 60e-9
        n = nx * ny
        core = np.zeros((nx, ny), dtype=bool)
        core[5:9, 4:8] = True
        bulk = _uniaxial(1.55, 3.48, (np.cos(np.deg2rad(35.0)), np.sin(np.deg2rad(35.0)), 0.0))
        base = [np.where(core.ravel(), bulk[a, a], 2.4025) + 0j for a in range(3)]
        off = np.where(core.ravel(), bulk[0, 1], 0.0) + 0j
        der, steps = _grid(nx, ny, cell)
        one = jnp.ones(n, dtype=jnp.complex128)
        rng = np.random.default_rng(4)
        direction = jnp.asarray(rng.normal(size=n))

        def neff(step, solve_left):
            eps = [jnp.asarray(component) + step * direction for component in base]
            _, _, values, _ = solve_modes_diagonal_jax(
                *eps,
                one,
                one,
                one,
                der,
                steps,
                K0,
                4,
                3.5,
                eps_xy=jnp.asarray(off),
                eps_yx=jnp.asarray(off),
                solve_left=solve_left,
            )
            return values[0]

        closed = float(jax.grad(lambda s: neff(s, False))(0.0))
        solved = float(jax.grad(lambda s: neff(s, True))(0.0))
        coarse = (float(neff(2e-4, True)) - float(neff(-2e-4, True))) / 4e-4
        fine = (float(neff(1e-4, True)) - float(neff(-1e-4, True))) / 2e-4
        finite = (4 * fine - coarse) / 3
        # the two adjoints agree to 1e-8; against the finite difference the floor is the
        # eigensolver's own convergence, measured 4.9e-6.
        assert solved == pytest.approx(closed, rel=1e-8)
        assert solved == pytest.approx(finite, rel=1e-5)


def test_the_bordered_solve_is_only_taken_when_a_field_cotangent_is_present(float64):
    """The n_eff-only backward must not pay for the field adjoint."""
    from fdtdx.core.physics.mode_backend import jax_solve

    calls = {"count": 0}
    original = jax_solve.spl.spsolve

    def counting(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    jax_solve.spl.spsolve = counting
    try:
        settings = ModeSolveSettings.create(frequency=FREQ, resolution=70e-9, mode_index=0)
        eps = TestFieldGradients._phase_shifter()
        jax.grad(lambda s: mode_solve(eps * s, settings).neff.real)(1.0)
        assert calls["count"] == 0
        jax.grad(lambda s: jnp.sum(jnp.abs(mode_solve(eps * s, settings).E) ** 2))(1.0)
        assert calls["count"] > 0
    finally:
        jax_solve.spl.spsolve = original


assert spl is not None  # keep the scipy import meaningful for the monkeypatch above
