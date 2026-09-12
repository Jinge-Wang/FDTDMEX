"""The four-component mode operator: the tier switch, the closed forms, MPB, the gradients.

Every number in these docstrings was measured by these tests on one Apple M4 Pro CPU with
``jax_enable_x64`` on and ``complex128`` throughout.
"""

import warnings
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

jax = pytest.importorskip("jax")
jnp = jax.numpy

from fdtdx.constants import c  # noqa: E402
from fdtdx.core.physics.mode_adjoint import (  # noqa: E402
    ModeSolveSettings,
    carries_longitudinal_entries,
    mode_neff_parts,
    mode_sensitivity,
    mode_solve,
)
from fdtdx.core.physics.mode_backend import (  # noqa: E402
    ModeLongitudinalOffdiagWarning,
    TensorComponents,
    transverse_index_bound,
)
from fdtdx.core.physics.mode_backend.full_tensor import (  # noqa: E402
    assemble_full_tensor_operator,
    solve_modes_full_tensor,
)
from fdtdx.core.physics.mode_backend.jax_full_tensor import solve_modes_full_tensor_jax  # noqa: E402
from fdtdx.core.physics.mode_backend.operator import (  # noqa: E402
    build_average_matrices,
    build_derivative_matrices,
)
from fdtdx.core.physics.mode_backend.solve import assemble_mode_operator  # noqa: E402
from fdtdx.core.physics.modes import compute_modes  # noqa: E402

LAM = 1.55e-6
FREQ = c / LAM
K0 = 2.0 * np.pi / LAM
ENTRIES = ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def float64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _uniaxial(n_o: float, n_e: float, axis) -> np.ndarray:
    """Permittivity of a uniaxial crystal whose optic axis is the unit vector ``axis``."""
    s = np.asarray(axis, dtype=float)
    s = s / np.linalg.norm(s)
    return n_o**2 * np.eye(3) + (n_e**2 - n_o**2) * np.outer(s, s)


def _inverse_tensor(tensor: np.ndarray) -> np.ndarray:
    t = np.asarray(tensor)
    return np.linalg.inv(t.transpose(2, 3, 0, 1)).transpose(2, 3, 0, 1).reshape(9, *t.shape[2:])


def _flat(tensor: np.ndarray, mask: np.ndarray, background: float) -> dict[str, np.ndarray]:
    out = {}
    for index, name in enumerate(ENTRIES):
        outside = background if index in (0, 4, 8) else 0.0
        out[name] = np.where(mask.ravel(), tensor.reshape(9)[index], outside).astype(np.complex128)
    return out


# ------------------------------------------------------------------------------------------------
# the tier switch
# ------------------------------------------------------------------------------------------------


class TestTheTransverseTierIsUnchanged:
    """Forcing the four-component operator onto a cross-section without longitudinal entries.

    The two operators are related by an exact algebraic identity — with those entries zero the
    four-component one is block anti-diagonal, ``M = [[0, A], [B, 0]]``, and ``mat_transverse =
    -A B`` entry for entry — so the only difference is floating-point rounding. Measured on the
    existing fixtures: ``n_eff`` agrees to 3.6e-15 and the fields to 9e-15 relative.
    """

    @staticmethod
    def _both(inv_eps, **kwargs):
        base = dict(
            frequency=FREQ,
            inv_permittivities=jnp.asarray(inv_eps),
            inv_permeabilities=1.0,
            num_modes=2,
            dtype=jnp.float64,
            **kwargs,
        )
        return compute_modes(mode_formulation="transverse", **base), compute_modes(mode_formulation="full", **base)

    def test_a_one_component_strip(self, float64):
        eps = np.ones((1, 1, 20, 15))
        eps[:, :, 7:13, 6:10] = 3.48**2
        (e_t, h_t, n_t), (e_f, h_f, n_f) = self._both(1.0 / eps, resolution=100e-9)
        assert np.max(np.abs(np.asarray(n_t) - np.asarray(n_f))) < 1e-13
        assert np.max(np.abs(np.asarray(e_t) - np.asarray(e_f))) < 1e-13 * np.max(np.abs(np.asarray(e_t)))
        assert np.max(np.abs(np.asarray(h_t) - np.asarray(h_f))) < 1e-13 * np.max(np.abs(np.asarray(h_t)))

    def test_a_three_component_strip_on_a_graded_grid(self, float64):
        rng = np.random.default_rng(2)
        nx, ny = 22, 18
        cx = np.concatenate(([0.0], np.cumsum(70e-9 * (1.0 + 0.4 * rng.random(nx)))))
        cy = np.concatenate(([0.0], np.cumsum(65e-9 * (1.0 + 0.4 * rng.random(ny)))))
        eps = np.ones((3, nx, ny, 1))
        eps[:, 7:15, 6:12, :] = 3.48**2
        (_, _, n_t), (_, _, n_f) = self._both(1.0 / eps, transverse_coords=[jnp.asarray(cx), jnp.asarray(cy)])
        assert np.max(np.abs(np.asarray(n_t) - np.asarray(n_f))) < 1e-13

    @pytest.mark.parametrize("reciprocal", [True, False])
    def test_a_transverse_off_diagonal_tensor(self, float64, reciprocal):
        core = np.zeros((24, 20), bool)
        core[8:16, 7:13] = True
        tensor = np.zeros((3, 3, 24, 20))
        for a in range(3):
            tensor[a, a] = np.where(core, 3.48**2, 1.44**2)
        tensor[0, 1] = np.where(core, 0.3, 0.0)
        tensor[1, 0] = tensor[0, 1] if reciprocal else np.where(core, 0.15, 0.0)
        (_, _, n_t), (_, _, n_f) = self._both(_inverse_tensor(tensor)[:, :, :, None], resolution=80e-9)
        assert np.max(np.abs(np.asarray(n_t) - np.asarray(n_f))) < 1e-13

    def test_the_two_cell_collapse(self, float64):
        eps = np.ones((3, 26, 2, 1))
        eps[:, 9:17, :, :] = 3.48**2
        (_, _, n_t), (_, _, n_f) = self._both(1.0 / eps, resolution=60e-9)
        assert np.max(np.abs(np.asarray(n_t) - np.asarray(n_f))) < 1e-13

    def test_the_operator_identity_itself(self, float64):
        """``mat_transverse = -A B``, where A and B are the two off-diagonal blocks."""
        rng = np.random.default_rng(3)
        nx, ny = 12, 10
        cx = np.concatenate(([0.0], np.cumsum(60e-9 * (1.0 + 0.3 * rng.random(nx)))))
        cy = np.concatenate(([0.0], np.cumsum(55e-9 * (1.0 + 0.3 * rng.random(ny)))))
        der = build_derivative_matrices(cx, cy)
        avg = build_average_matrices(cx, cy)
        n = nx * ny
        base = rng.uniform(2.0, 12.0, n).astype(np.complex128)
        eps = {name: np.zeros(n, dtype=np.complex128) for name in ENTRIES}
        eps["xx"], eps["yy"], eps["zz"] = base, base * 1.02, base * 0.98
        eps["xy"] = 0.05 * rng.random(n)
        eps["yx"] = eps["xy"].copy()
        mu = tuple(np.ones(n, dtype=np.complex128) for _ in range(3))
        full = assemble_full_tensor_operator(eps, *mu, der, avg, K0).mat
        transverse = assemble_mode_operator(
            eps["xx"], eps["yy"], eps["zz"], *mu, der, K0, eps_xy=eps["xy"], eps_yx=eps["yx"]
        ).mat
        a_block = full[: 2 * n, 2 * n :]
        b_block = full[2 * n :, : 2 * n]
        assert np.max(np.abs(full[: 2 * n, : 2 * n].toarray())) == 0.0
        assert np.max(np.abs(full[2 * n :, 2 * n :].toarray())) == 0.0
        residual = np.max(np.abs((transverse + a_block @ b_block).toarray()))
        assert residual < 1e-12 * np.max(np.abs(transverse.toarray()))


# ------------------------------------------------------------------------------------------------
# closed forms
# ------------------------------------------------------------------------------------------------


class TestBulkUniaxialTiltedOutOfThePlane:
    """The extraordinary index of a crystal whose optic axis leaves the cross-section plane.

    One cell on each transverse axis makes every difference matrix identically zero, so the operator
    is the bulk one and its eigenvalues are the two plane-wave indices. The closed form is the index
    ellipsoid, ``1/n^2 = cos^2(theta)/n_o^2 + sin^2(theta)/n_e^2`` with ``theta`` the angle between
    the propagation direction and the optic axis (Born & Wolf, *Principles of Optics*, section 15.3).
    Measured with ``n_o = 1.5``, ``n_e = 1.7``: the error is at most 1.1e-15 at every tilt, against
    1.4e-5 to 1.2e-2 for the transverse operator, which cannot represent the tilt at all.
    """

    n_o, n_e = 1.5, 1.7

    def _bulk(self, tensor):
        coords = np.array([0.0, 100e-9])
        der = build_derivative_matrices(coords, coords)
        avg = build_average_matrices(coords, coords)
        eps = {name: np.full(1, tensor.reshape(9)[i], dtype=np.complex128) for i, name in enumerate(ENTRIES)}
        mu = tuple(np.ones(1, dtype=np.complex128) for _ in range(3))
        operator = assemble_full_tensor_operator(eps, *mu, der, avg, K0)
        values = np.linalg.eigvals(operator.mat.toarray())
        return np.sort(np.real(values[np.real(values) > 0]))

    @pytest.mark.parametrize("theta_deg", [1.0, 20.0, 45.0, 60.0, 80.0])
    def test_the_extraordinary_index_is_exact(self, float64, theta_deg):
        theta = np.deg2rad(theta_deg)
        tensor = _uniaxial(self.n_o, self.n_e, (np.sin(theta), 0.0, np.cos(theta)))
        exact = 1.0 / np.sqrt(np.cos(theta) ** 2 / self.n_o**2 + np.sin(theta) ** 2 / self.n_e**2)
        branches = self._bulk(tensor)
        assert np.min(np.abs(branches - exact)) < 1e-13
        assert np.min(np.abs(branches - self.n_o)) < 1e-13

    def test_the_plane_wave_indices_are_the_schur_complement(self, float64):
        """With no transverse variation, ``n^2`` are the eigenvalues of ``eps_tt - eps_tz eps_zz^-1 eps_zt``."""
        rng = np.random.default_rng(9)
        tensor = rng.uniform(-0.4, 0.4, (3, 3)) + np.diag([4.0, 4.5, 5.0])
        tensor = 0.5 * (tensor + tensor.T)
        schur = tensor[:2, :2] - np.outer(tensor[:2, 2], tensor[2, :2]) / tensor[2, 2]
        expected = np.sort(np.sqrt(np.linalg.eigvals(schur).real))
        assert np.max(np.abs(np.sort(self._bulk(tensor)) - expected)) < 1e-12


# ------------------------------------------------------------------------------------------------
# an independent 4x4 transfer-matrix reference
# ------------------------------------------------------------------------------------------------


def _berreman_matrix(eps: np.ndarray, n: complex) -> np.ndarray:
    """``d psi / d(k0 x) = M psi`` for ``psi = (E_y, E_z, h_y, h_z)``, stratification along x.

    Written from Maxwell's equations rather than from the mode operator, so it is an independent
    reference and not a restatement of the thing under test. The two algebraic components are
    ``E_x = (n h_y - eps_xy E_y - eps_xz E_z) / eps_xx`` and ``h_x = -n E_y``.
    """
    e = np.asarray(eps, dtype=np.complex128)
    exx, exy, exz = e[0, 0], e[0, 1], e[0, 2]
    eyx, eyy, eyz = e[1, 0], e[1, 1], e[1, 2]
    ezx, ezy, ezz = e[2, 0], e[2, 1], e[2, 2]
    m = np.zeros((4, 4), dtype=np.complex128)
    m[0, 3] = 1j
    m[1, 0] = -1j * n * exy / exx
    m[1, 1] = -1j * n * exz / exx
    m[1, 2] = 1j * n * n / exx - 1j
    m[2, 0] = -1j * (ezy - ezx * exy / exx)
    m[2, 1] = -1j * (ezz - ezx * exz / exx)
    m[2, 2] = -1j * n * ezx / exx
    m[3, 0] = 1j * (eyy - eyx * exy / exx - n * n)
    m[3, 1] = 1j * (eyz - eyx * exz / exx)
    m[3, 2] = 1j * n * eyx / exx
    return m


def _half_space_basis(eps, n, decays_toward):
    values, vectors = np.linalg.eig(_berreman_matrix(eps, n))
    sign = 1.0 if decays_toward == "minus" else -1.0
    keep = np.argsort(-sign * np.real(values))[:2]
    basis = vectors[:, keep]
    return basis / np.linalg.norm(basis, axis=0, keepdims=True)


def _slab_determinant(n, core, thickness, eps_clad):
    left = _half_space_basis(eps_clad, n, "minus")
    right = _half_space_basis(eps_clad, n, "plus")
    transferred = expm(_berreman_matrix(core, n) * (K0 * thickness)) @ left
    transferred = transferred / np.linalg.norm(transferred, axis=0, keepdims=True)
    return complex(np.linalg.det(np.concatenate((transferred, right), axis=1)))


def berreman_neff(guess, core, thickness, eps_clad, iterations=60):
    """Secant iteration on the analytic determinant; its root is the guided index."""
    a = complex(guess)
    b = a * (1.0 + 1e-6)
    fa, fb = _slab_determinant(a, core, thickness, eps_clad), _slab_determinant(b, core, thickness, eps_clad)
    for _ in range(iterations):
        if fb == fa:
            break
        step = fb * (b - a) / (fb - fa)
        a, fa = b, fb
        b = b - step
        fb = _slab_determinant(b, core, thickness, eps_clad)
        if abs(step) < 1e-14:
            break
    return b


class TestTiltedUniaxialSlab:
    """A 400 nm uniaxial slab in a 1.44 cladding, against the Berreman 4x4 and against MPB.

    ``n_o = 2.0``, ``n_e = 2.2``, 6 um window, PEC walls, 1.55 um. The Berreman solver above is an
    independent implementation; on an isotropic slab its roots match the textbook TE and TM
    dispersion relations to 1e-12.

    Convergence, measured: the branch whose electric field is tangential at the interface converges
    at order 2.00 and a Richardson step lands 7e-11 from the reference. The branch whose field has a
    normal component converges as ``A h^2 + B h`` once the tensor is tilted — the ``eps_zx E_x``
    product averages a *discontinuous* field across the interface — and a two-term fit lands 2e-8
    from it. The first-order term is 3.4e-6 of ``n_eff`` at a 2.5 nm cell.
    """

    window = 6.0e-6
    thickness = 400e-9
    n_clad, n_o, n_e = 1.44, 2.0, 2.2

    def _core(self, theta_deg, phi_deg):
        t, p = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
        return _uniaxial(self.n_o, self.n_e, (np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)))

    def _solve(self, core, res, num_modes=8):
        nx = round(self.window / res)
        cx = np.arange(nx + 1) * res
        cy = np.array([0.0, res])
        der = build_derivative_matrices(cx, cy)
        avg = build_average_matrices(cx, cy)
        mid = 0.5 * (cx[:-1] + cx[1:])
        mask = np.abs(mid - self.window / 2) < self.thickness / 2
        eps = _flat(core, mask, self.n_clad**2)
        mu = tuple(np.ones(nx, dtype=np.complex128) for _ in range(3))
        _, _, neff, keff = solve_modes_full_tensor(eps, *mu, der, avg, K0, num_modes, self.n_e * 1.001)
        return neff, keff

    @pytest.mark.parametrize("theta_deg, phi_deg", [(45.0, 0.0), (55.0, 35.0)])
    def test_against_the_berreman_transfer_matrix(self, float64, theta_deg, phi_deg):
        core = self._core(theta_deg, phi_deg)
        eps_clad = self.n_clad**2 * np.eye(3)
        errors = {}
        for res in (20e-9, 10e-9, 5e-9):
            neff, _ = self._solve(core, res)
            for guess in (1.79, 1.70):
                reference = float(np.real(berreman_neff(guess, core, self.thickness, eps_clad)))
                if not self.n_clad < reference < self.n_e:
                    continue
                got = float(neff[int(np.argmin(np.abs(neff - reference)))])
                errors.setdefault(round(reference, 9), []).append(got - reference)
        assert errors
        for reference, series in errors.items():
            assert abs(series[-1]) < 5e-5, f"branch {reference}"
            # halving the cell must cut the error by at least 3.2x (order 1.7 or better)
            for coarse, fine in pairwise(series):
                assert abs(coarse / fine) > 3.2

    def _graded_edges(self, core_cell, stretch=3.0):
        """Core uniform, each cladding stretched so its first cell equals the core cell.

        A *smooth* grading: the ratio between neighbouring cells falls towards one as the mesh is
        refined, which is the case where a second-order scheme is supposed to stay second order.
        """
        c0, c1 = self.window / 2 - self.thickness / 2, self.window / 2 + self.thickness / 2
        core = np.linspace(c0, c1, round(self.thickness / core_cell) + 1)
        count = max(2, round(c0 * stretch / ((np.exp(stretch) - 1) * core_cell)))
        shape = np.linspace(0.0, 1.0, count + 1)
        stretched = (np.exp(stretch * shape) - 1.0) / (np.exp(stretch) - 1.0)
        return np.unique(np.concatenate((c0 - c0 * stretched[::-1], core, c1 + c0 * stretched)))

    def test_a_smoothly_graded_mesh_keeps_the_order(self, float64):
        """The averaging matrices use the plain 1/2 average, which is exact only on a uniform grid.

        Measured on a mesh whose largest cell is 20x its smallest: the order between consecutive
        grids is 2.10, 2.04, 2.00 for the tangential branch and 1.97, 1.93, 1.87 for the branch with
        a normal E — the same as on a uniform mesh, so the spacing-weighted interpolation would buy
        nothing here. An abrupt cell-size *jump* at the interface does cost an order, equally with
        and without the tilt, so that is a mesh-quality property and not a property of this operator.
        """
        core = self._core(45.0, 0.0)
        eps_clad = self.n_clad**2 * np.eye(3)
        reference = float(np.real(berreman_neff(1.7458, core, self.thickness, eps_clad)))
        errors = []
        for core_cell in (20e-9, 10e-9, 5e-9):
            edges = self._graded_edges(core_cell)
            der = build_derivative_matrices(edges, np.array([0.0, core_cell]))
            avg = build_average_matrices(edges, np.array([0.0, core_cell]))
            mid = 0.5 * (edges[:-1] + edges[1:])
            mask = np.abs(mid - self.window / 2) < self.thickness / 2
            eps = _flat(core, mask, self.n_clad**2)
            cells = len(edges) - 1
            mu = tuple(np.ones(cells, dtype=np.complex128) for _ in range(3))
            _, _, neff, _ = solve_modes_full_tensor(eps, *mu, der, avg, K0, 8, self.n_e * 1.001)
            errors.append(float(neff[int(np.argmin(np.abs(neff - reference)))]) - reference)
            steps = np.diff(edges)
            assert steps.max() / steps.min() > 10.0  # the mesh really is graded
        for coarse, fine in pairwise(errors):
            assert abs(coarse / fine) > 3.6  # order 1.85 or better

    def test_against_mpb(self, float64):
        """MPB on the same structure; see ``data/generate_mpb_tilted_uniaxial_slab.py`` for settings.

        MPB converges to the Berreman value from below and this solver from above, so at a matched
        cell size the two straddle it. Measured gap at 5 nm: 2.5e-5 to 6.3e-5; at 2.5 nm the cases
        that were run fall to 1.0e-5 to 1.8e-5.
        """
        data = np.load(FIXTURES / "mpb_tilted_uniaxial_slab.npz")
        for theta, phi in ((45.0, 0.0), (55.0, 35.0), (0.0, 0.0)):
            reference = data[f"theta{theta:g}_phi{phi:g}_res200"]
            neff, _ = self._solve(self._core(theta, phi), 5e-9)
            for band in reference:
                got = float(neff[int(np.argmin(np.abs(neff - band)))])
                assert abs(got - band) < 1e-4, f"theta={theta} phi={phi} band={band}"


# ------------------------------------------------------------------------------------------------
# structural properties
# ------------------------------------------------------------------------------------------------


def _tilted_ridge(nx=24, ny=18, theta_deg=40.0, phi_deg=25.0, res=80e-9):
    core = np.zeros((nx, ny), bool)
    core[nx // 2 - 4 : nx // 2 + 4, ny // 2 - 3 : ny // 2 + 3] = True
    t, p = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    tensor = _uniaxial(2.1, 2.3, (np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)))
    cx = np.arange(nx + 1) * res
    cy = np.arange(ny + 1) * res
    return _flat(tensor, core, 1.44**2), build_derivative_matrices(cx, cy), build_average_matrices(cx, cy), nx * ny


@pytest.mark.parametrize("res", [80e-9, 40e-9])
def test_a_lossless_reciprocal_tensor_keeps_a_real_effective_index(float64, res):
    """Reciprocity is a discrete property here, not only a continuum one.

    ``beta_backward = -beta_forward`` for a symmetric permittivity, which forces ``n_eff`` real for a
    bound mode of a lossless guide. The discretisation keeps it only because the right-column
    longitudinal entries are sampled on the node, inside the average: sampling them at the
    transverse location instead puts an imaginary part on ``n_eff`` that falls only as the first
    power of the cell size (measured 7.6e-4 at 100 nm, 1.5e-4 at 12.5 nm).
    """
    eps, der, avg, n = _tilted_ridge(res=res)
    mu = tuple(np.ones(n, dtype=np.complex128) for _ in range(3))
    _, _, neff, keff = solve_modes_full_tensor(eps, *mu, der, avg, K0, 4, 2.31)
    assert np.max(np.abs(keff)) < 1e-12 * np.max(np.abs(neff))


def test_the_backward_mode_is_the_mirrored_structure_solved_forward(float64):
    """The check that says the backward solve is the physical backward mode.

    Reversing the propagation axis negates exactly the four longitudinal permittivity entries. So
    the backward mode of a structure must be the forward mode of that mirrored structure with its
    longitudinal E and transverse H negated — and it is, to 1.1e-14, while the same transform
    applied to the *unmirrored* forward mode is off by 0.21.
    """
    eps, der, avg, n = _tilted_ridge()
    mirrored = dict(eps)
    for name in ("xz", "yz", "zx", "zy"):
        mirrored[name] = -eps[name]
    mu = tuple(np.ones(n, dtype=np.complex128) for _ in range(3))
    forward = solve_modes_full_tensor(eps, *mu, der, avg, K0, 4, 2.31, direction="+")
    backward = solve_modes_full_tensor(eps, *mu, der, avg, K0, 4, 2.31, direction="-")
    mirror_forward = solve_modes_full_tensor(mirrored, *mu, der, avg, K0, 4, 2.31, direction="+")
    assert np.max(np.abs(backward[2] - forward[2])) < 1e-12

    # Compare the two leading modes only: the solvers order by Re(n_eff), and further up the list
    # this cross-section has a near-degenerate pair whose individual eigenvectors are an arbitrary
    # basis of their subspace, so a field-by-field comparison there is not a statement about the
    # physics.
    def normalise(fields, count=2):
        e, h = fields[0][:, :, :count], fields[1][:, :, :count]
        scale = np.max(np.abs(e[:2]), axis=(0, 1))
        return e / scale, h / scale

    e_back, h_back = normalise(backward)
    e_mirror, h_mirror = normalise(mirror_forward)
    predicted_e, predicted_h = e_mirror.copy(), h_mirror.copy()
    predicted_e[2] *= -1
    predicted_h[0] *= -1
    predicted_h[1] *= -1
    assert np.max(np.abs(predicted_e - e_back)) < 1e-11
    assert np.max(np.abs(predicted_h - h_back)) < 1e-11

    e_forward, _ = normalise(forward)
    naive = e_forward.copy()
    naive[2] *= -1
    assert np.max(np.abs(naive - e_back)) > 1e-2


def test_the_jax_assembly_matches_the_numpy_one(float64):
    eps, der, avg, n = _tilted_ridge()
    mu = tuple(np.ones(n, dtype=np.complex128) for _ in range(3))
    e_np, h_np, n_np, _ = solve_modes_full_tensor(eps, *mu, der, avg, K0, 4, 2.31)
    e_jx, h_jx, n_jx, _ = solve_modes_full_tensor_jax(
        {name: jnp.asarray(value) for name, value in eps.items()},
        *(jnp.asarray(m) for m in mu),
        der,
        avg,
        K0,
        4,
        2.31,
    )
    assert np.max(np.abs(n_np - np.asarray(n_jx))) < 1e-12
    # Fields for the two leading modes: further up the list this cross-section has a near-degenerate
    # pair, whose individual eigenvectors are an arbitrary basis of one subspace.
    e_np, e_jx = e_np[:, :, :2], np.asarray(e_jx)[:, :, :2]
    h_np, h_jx = h_np[:, :, :2], np.asarray(h_jx)[:, :, :2]
    assert np.max(np.abs(e_np - e_jx)) < 1e-11 * np.max(np.abs(e_np))
    assert np.max(np.abs(h_np - h_jx)) < 1e-11 * np.max(np.abs(h_np))


@pytest.mark.parametrize("mu_xx, mu_yy", [(1.0, 1.0), (1.0, 3.0), (3.0, 1.0), (0.3, 0.3), (2.5, 0.4)])
def test_the_shift_invert_bound_accounts_for_an_anisotropic_permeability(float64, mu_xx, mu_yy):
    """The automatic shift must never aim below the mode the caller wants.

    With no transverse variation, ``n^2`` are the eigenvalues of ``diag(mu_yy, mu_xx) S`` with ``S``
    the Schur complement of the permittivity, so both factors move the bound. A diagonal anisotropic
    permeability is a supported case here (only an off-diagonal one is refused), and reading the
    bound off the permittivity alone put it 1.07 below the true index at ``mu_yy = 3``.
    """
    tensor = _uniaxial(1.6, 2.0, (np.sin(np.deg2rad(35.0)), 0.0, np.cos(np.deg2rad(35.0))))
    coords = np.array([0.0, 100e-9])
    der = build_derivative_matrices(coords, coords)
    avg = build_average_matrices(coords, coords)
    eps = {name: np.full(1, tensor.reshape(9)[i], dtype=np.complex128) for i, name in enumerate(ENTRIES)}
    mu = (
        np.full(1, mu_xx, dtype=np.complex128),
        np.full(1, mu_yy, dtype=np.complex128),
        np.ones(1, dtype=np.complex128),
    )
    values = np.linalg.eigvals(assemble_full_tensor_operator(eps, *mu, der, avg, K0).mat.toarray())
    true_index = float(np.max(np.real(values)))
    components = TensorComponents(
        xx=eps["xx"],
        yy=eps["yy"],
        zz=eps["zz"],
        xy=eps["xy"],
        yx=eps["yx"],
        longitudinal={name: eps[name] for name in ("xz", "yz", "zx", "zy")},
        magnitudes={},
    )
    bound = float(np.sqrt(transverse_index_bound(components, mu[0], mu[1])))
    assert bound >= true_index
    assert bound < 2.0 * true_index  # and it must not be uselessly loose either


def test_the_sensitivity_refusal_looks_at_the_values_not_only_the_shape(float64):
    """A nine-component tensor with no longitudinal entry takes the transverse operator.

    The field-level integral is exact for that solve, so refusing it would be refusing a tier it
    never uses. The chain-rule predicate stays shape-based, because a *perturbation* of one of those
    zeros does move the solve; the two predicates differ on exactly this cross-section, and that is
    deliberate.
    """
    nx, ny = 16, 14
    tensor = np.zeros((3, 3, nx, ny, 1))
    core = np.zeros((nx, ny, 1), dtype=bool)
    core[5:11, 4:10] = True
    for a in range(3):
        tensor[a, a] = np.where(core, 3.48**2, 1.44**2)
    tensor[0, 1] = np.where(core, 0.2, 0.0)
    tensor[1, 0] = tensor[0, 1]
    transverse_only = jnp.asarray(tensor.reshape(9, nx, ny, 1))
    settings = ModeSolveSettings.create(frequency=FREQ, resolution=90e-9, mode_index=0)

    assert carries_longitudinal_entries(transverse_only, settings) is True
    neff, sensitivity = mode_sensitivity(transverse_only, settings)
    assert sensitivity.shape == transverse_only.shape
    assert float(jnp.real(neff)) > 1.44

    with_longitudinal = tensor.copy()
    with_longitudinal[0, 2] = np.where(core, 0.2, 0.0)
    with_longitudinal[2, 0] = with_longitudinal[0, 2]
    with pytest.raises(ValueError, match="mirror plane"):
        mode_sensitivity(jnp.asarray(with_longitudinal.reshape(9, nx, ny, 1)), settings)


# ------------------------------------------------------------------------------------------------
# gradients
# ------------------------------------------------------------------------------------------------


def _tilted_cross_section(nx=18, ny=16, theta_deg=40.0, phi_deg=25.0):
    t, p = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    tensor = _uniaxial(2.1, 2.3, (np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)))
    core = np.zeros((nx, ny), bool)
    core[nx // 2 - 4 : nx // 2 + 4, ny // 2 - 3 : ny // 2 + 3] = True
    out = np.zeros((3, 3, nx, ny))
    for a in range(3):
        for b in range(3):
            out[a, b] = np.where(core, tensor[a, b], 1.44**2 if a == b else 0.0)
    return jnp.asarray(out.reshape(9, nx, ny, 1))


@pytest.mark.parametrize("entry", ["xz", "yz", "zx", "zy"])
def test_the_longitudinal_entries_have_a_gradient_now(float64, entry):
    """``d n_eff / d eps_xz`` and its three partners, against a central finite difference.

    Measured relative agreement: 2.8e-8 (xz), 5.1e-8 (yz), 2.5e-8 (zx), 2.0e-8 (zy).
    """
    eps = _tilted_cross_section()
    settings = ModeSolveSettings.create(frequency=FREQ, resolution=90e-9, mode_index=0)
    index = ENTRIES.index(entry)
    rng = np.random.default_rng(index + 1)
    direction = np.zeros(eps.shape)
    direction[index] = rng.normal(size=eps.shape[1:])
    step = jnp.asarray(direction)

    def value(scale):
        return mode_neff_parts(eps + scale * step, settings)[0]

    gradient = float(jax.grad(value)(0.0))
    h = 1e-3
    finite = (float(value(h)) - float(value(-h))) / (2 * h)
    assert abs(gradient) > 1e-6
    assert gradient == pytest.approx(finite, rel=1e-6)


def test_a_symmetric_tensor_gives_the_same_gradient_for_xz_and_zx(float64):
    """A structural check the field-level reciprocity integral fails and the exact adjoint passes.

    ``n_eff`` depends on ``eps_xz`` and ``eps_zx`` only through their product in the Schur
    complement, so for a symmetric permittivity the two derivatives must be equal. The field-level
    integral carries its sign on the *row* index and makes them equal and opposite instead
    (measured: -2.38e-3 against +2.38e-3, where the exact value is -1.93e-5 for both), which is why
    that integral is refused on this tier.
    """
    eps = _tilted_cross_section()
    settings = ModeSolveSettings.create(frequency=FREQ, resolution=90e-9, mode_index=0)
    rng = np.random.default_rng(99)
    shared = rng.normal(size=eps.shape[1:])
    gradients = {}
    for entry in ("xz", "zx", "yz", "zy"):
        direction = np.zeros(eps.shape)
        direction[ENTRIES.index(entry)] = shared
        step = jnp.asarray(direction)
        gradients[entry] = float(jax.grad(lambda s, step=step: mode_neff_parts(eps + s * step, settings)[0])(0.0))
    assert gradients["xz"] == pytest.approx(gradients["zx"], rel=1e-9)
    assert gradients["yz"] == pytest.approx(gradients["zy"], rel=1e-9)
    assert abs(gradients["xz"]) > 1e-6
    assert abs(gradients["yz"]) > 1e-6


def test_the_full_path_still_serves_the_cases_the_transverse_one_did(float64):
    """A lossy diagonal, a metal cell and a magnetic min-edge wall all still solve."""
    eps = np.asarray(_tilted_cross_section()).astype(np.complex128)

    def inverse(array):
        tensor = jnp.asarray(array).reshape(3, 3, *array.shape[1:]).transpose(2, 3, 4, 0, 1)
        return jnp.linalg.inv(tensor).transpose(3, 4, 0, 1, 2).reshape(9, *array.shape[1:])

    base = dict(frequency=FREQ, inv_permeabilities=1.0, resolution=90e-9, dtype=jnp.float64)
    _, _, lossless = compute_modes(inv_permittivities=inverse(eps), num_modes=1, **base)
    assert abs(complex(lossless[0]).imag) < 1e-12

    lossy = eps.copy()
    lossy[[0, 4, 8]] += 0.01j
    _, _, neff = compute_modes(inv_permittivities=inverse(lossy), num_modes=1, **base)
    assert complex(neff[0]).imag > 1e-4

    metal = eps.copy()
    metal[[0, 4, 8], 2:4, 2:4, :] = -8.0 + 1.0j
    _, _, neff = compute_modes(inv_permittivities=inverse(metal), num_modes=1, target_neff=2.2, **base)
    assert 1.0 < complex(neff[0]).real < 2.4

    _, _, neff = compute_modes(inv_permittivities=inverse(eps), num_modes=1, symmetry=(1, 0), **base)
    assert complex(neff[0]).real > 1.5


def test_the_field_gradient_crosses_the_longitudinal_entries(float64):
    """``d(log10 sum |E|^2) / d eps`` along a direction that moves only the longitudinal entries.

    Measured against the central difference at its own optimum step (``h = 1e-4``): 6.5e-10
    relative, and 8.7e-10 against a Richardson-extrapolated difference. At ``h = 1e-5`` the
    difference is round-off limited and the apparent disagreement grows to 5.9e-8, which is the
    finite difference's floor and not the adjoint's.
    """
    eps = _tilted_cross_section()
    settings = ModeSolveSettings.create(frequency=FREQ, resolution=90e-9, mode_index=0)
    rng = np.random.default_rng(17)
    direction = np.zeros(eps.shape)
    direction[[2, 5, 6, 7]] = rng.normal(size=(4, *eps.shape[1:]))
    step = jnp.asarray(direction)

    def objective(perturbed):
        return jnp.log10(jnp.sum(jnp.abs(mode_solve(perturbed, settings, differentiable_fields=True).E) ** 2))

    gradient = float(jax.grad(lambda s: objective(eps + s * step))(0.0))
    h = 1e-4
    finite = (float(objective(eps + h * step)) - float(objective(eps - h * step))) / (2 * h)
    assert abs(gradient) > 1e-6
    assert gradient == pytest.approx(finite, rel=1e-8)


# ------------------------------------------------------------------------------------------------
# the stress-optic cross-section
# ------------------------------------------------------------------------------------------------


class TestStressedSilicaOnTheFullPath:
    """S3's stress-optic tensor re-solved with the longitudinal entries carried.

    S3 measured, on the COMSOL model-190 scene at 50 nm: largest off-diagonal permittivity entry
    1.5163e-06, pair ratio 1.6441e-03, axis rotation 0.0942 degrees; and 3.9957e-05 / 0.04762 /
    2.720 degrees for the soft-core variant. Those off-diagonals are purely *transverse* — the
    scene is laterally uniform, so the shear cancels — and the full path reproduces the transverse
    tier on them to 1e-13.

    Turning the principal axes 45 degrees into the plane that contains the propagation direction
    moves 2.3057e-04 of the splitting into ``eps_xz``, which the transverse operator must drop.
    Measured cost of that drop: 3.7e-09 and 4.1e-09 in ``n_eff``, against the second-order bound
    ``eps_xz^2 / (2 n eps_zz) = 8.2e-09``. So J4's argument that dropping the entries was harmless
    at S3's magnitudes is now a measured number with its bound.
    """

    eps_silica = 1.4508**2
    splitting = 1.5163e-06 / 1.6441e-03
    core_step = 0.11
    resolution = 200e-9

    def _tensor(self, nx, pair_ratio, rotate=False):
        core = np.zeros((nx, nx), dtype=bool)
        core[nx // 2 - 5 : nx // 2 + 5, nx // 2 - 5 : nx // 2 + 5] = True
        raised = self.eps_silica + self.core_step
        tensor = np.zeros((3, 3, nx, nx))
        for a in range(3):
            tensor[a, a] = np.where(core, raised, self.eps_silica)
        tensor[0, 0] = np.where(core, raised + self.splitting / 2, self.eps_silica)
        tensor[1, 1] = np.where(core, raised - self.splitting / 2, self.eps_silica)
        tensor[0, 1] = np.where(core, pair_ratio * self.splitting, 0.0)
        tensor[1, 0] = tensor[0, 1]
        if rotate:
            theta = np.deg2rad(45.0)
            rotation = np.array(
                [[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]]
            )
            tensor = np.einsum("ap,pqxy,bq->abxy", rotation, tensor, rotation)
        return tensor

    def _both(self, tensor):
        kwargs = dict(
            frequency=FREQ,
            inv_permittivities=jnp.asarray(_inverse_tensor(tensor)[:, None, :, :]),
            inv_permeabilities=1.0,
            num_modes=2,
            resolution=self.resolution,
            dtype=jnp.float64,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ModeLongitudinalOffdiagWarning)
            _, _, transverse = compute_modes(mode_formulation="transverse", **kwargs)
        _, _, full = compute_modes(mode_formulation="full", **kwargs)
        return np.real(np.asarray(transverse)), np.real(np.asarray(full))

    @pytest.mark.parametrize("pair_ratio", [1.6441e-03, 0.04762])
    def test_the_measured_scene_is_unchanged(self, float64, pair_ratio):
        transverse, full = self._both(self._tensor(24, pair_ratio))
        assert np.max(np.abs(transverse - full)) < 1e-9

    @pytest.mark.parametrize("pair_ratio", [1.6441e-03, 0.04762])
    def test_turning_the_axes_into_the_plane_costs_the_transverse_tier_its_bound(self, float64, pair_ratio):
        tensor = self._tensor(24, pair_ratio, rotate=True)
        longitudinal = float(np.max(np.abs(tensor[0, 2])))
        assert longitudinal == pytest.approx(2.3057e-04, rel=1e-3)
        transverse, full = self._both(tensor)
        difference = float(np.max(np.abs(transverse - full)))
        bound = longitudinal**2 / (2.0 * float(full[0]) * float(np.max(tensor[2, 2])))
        assert 1e-10 < difference < bound
