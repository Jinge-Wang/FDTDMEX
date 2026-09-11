"""The symmetric gradient of a vector FEM field, against closed forms.

Needs DOLFINx; skipped where it is not installed. Two levels of check. On an interpolated
displacement the answer is exact arithmetic: the symmetric gradient of a degree-``p`` Lagrange
function is a degree-``p - 1`` polynomial per cell, so the discontinuous space of that degree holds
it with no projection error, a linear displacement gives a constant strain and a rigid rotation
gives none at all. On a solved one it is the elasticity problem a photoelastic case starts from: a
4 x 2 um strip in plane strain under 1 MPa of uniaxial traction, whose uniform strain has a
one-line closed form.

The last test closes the loop the coupling layer exists for: the strain is sampled on a Yee grid
through a cross-section transform, and both the sample positions and the tensor components have to
land in the grid frame together.
"""

import numpy as np
import pytest

dolfinx = pytest.importorskip("dolfinx")
import ufl  # noqa: E402
from dolfinx import fem, mesh  # noqa: E402
from dolfinx.fem.petsc import LinearProblem  # noqa: E402
from mpi4py import MPI  # noqa: E402

from fdtdx.coupling import FemField, FemScalarField, PointTransform, sample_on_yee_lattices  # noqa: E402
from fdtdx.coupling.tensors import voigt_from_tensor  # noqa: E402

# Silicon in the COMSOL stress-optical model; the strip is drawn in micrometres and the moduli are
# in pascals, which linear elasticity allows: displacements come out in micrometres, strains bare.
E_MODULUS, POISSON, TRACTION = 110e9, 0.19, 1e6
STRIP = (4.0, 2.0)
EXPECTED_EPS_XX = (1.0 - POISSON**2) * TRACTION / E_MODULUS
EXPECTED_EPS_YY = -POISSON * (1.0 + POISSON) * TRACTION / E_MODULUS

PROBE_POINTS = np.array([[1.0, 0.5, 0.0], [2.0, 1.0, 0.0], [3.0, 1.5, 0.0], [0.7, 1.7, 0.0]])


def _displacement_field(degree: int, expression, cells: int = 6) -> FemField:
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, cells, cells, cells)
    space = fem.functionspace(domain, ("Lagrange", degree, (3,)))
    function = fem.Function(space, name="u")
    function.interpolate(expression)
    return FemField(function, name="u", unit="m")


@pytest.fixture(scope="module")
def strip_displacement() -> FemField:
    """Plane-strain uniaxial tension of the 4 x 2 um strip, solved once for the whole module."""
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array(STRIP)],
        [80, 40],
        cell_type=mesh.CellType.triangle,
    )
    space = fem.functionspace(domain, ("Lagrange", 2, (2,)))
    lame_lambda = E_MODULUS * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))
    lame_mu = E_MODULUS / (2.0 * (1.0 + POISSON))

    def stress(field):
        strain = ufl.sym(ufl.grad(field))
        return 2.0 * lame_mu * strain + lame_lambda * ufl.tr(strain) * ufl.Identity(2)

    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    bilinear = ufl.inner(stress(trial), ufl.sym(ufl.grad(test))) * ufl.dx

    loaded = mesh.locate_entities_boundary(domain, 1, lambda x: np.isclose(x[0], STRIP[0]))
    tags = mesh.meshtags(domain, 1, np.sort(loaded), np.full(len(loaded), 1, dtype=np.int32))
    traction = fem.Constant(domain, np.array([TRACTION, 0.0]))
    linear = ufl.inner(traction, test) * ufl.Measure("ds", domain=domain, subdomain_data=tags)(1)

    # Rollers on the two faces through the origin: they remove the rigid-body modes without
    # constraining the strain, so the exact solution is the uniform state the closed form describes.
    left = mesh.locate_entities_boundary(domain, 1, lambda x: np.isclose(x[0], 0.0))
    bottom = mesh.locate_entities_boundary(domain, 1, lambda x: np.isclose(x[1], 0.0))
    bcs = []
    for component, facets in ((0, left), (1, bottom)):
        sub_space, _ = space.sub(component).collapse()
        dofs = fem.locate_dofs_topological((space.sub(component), sub_space), 1, facets)
        bcs.append(fem.dirichletbc(fem.Function(sub_space), dofs, space.sub(component)))

    problem = LinearProblem(
        bilinear,
        linear,
        bcs=bcs,
        petsc_options_prefix="fdtdx_strip_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    )
    solution = problem.solve()
    if isinstance(solution, tuple):
        solution = solution[0]
    return FemField(solution, name="u", unit="um")


def test_a_linear_displacement_gives_its_own_symmetric_part_exactly():
    operator = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0], [7.0, -8.0, 9.0]]) * 1e-3
    field = _displacement_field(1, lambda x: operator @ x)
    strain = FemField.symmetric_gradient_of(field)
    assert strain.value_size == 9
    assert strain.unit is None
    rng = np.random.default_rng(11)
    samples = strain.evaluate(rng.uniform(0.05, 0.95, size=(400, 3)))
    assert samples.covered.all()
    expected = np.broadcast_to(0.5 * (operator + operator.T), (400, 3, 3))
    np.testing.assert_allclose(samples.values.reshape(-1, 3, 3), expected, rtol=1e-13, atol=0.0)


def test_a_rigid_rotation_carries_no_strain():
    spin = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]) * 1e-3
    field = _displacement_field(1, lambda x: spin @ x)
    samples = FemField.symmetric_gradient_of(field).evaluate(np.array([[0.3, 0.4, 0.5], [0.7, 0.2, 0.9]]))
    np.testing.assert_allclose(samples.values, 0.0, atol=1e-16)


def test_a_quadratic_displacement_on_p2_gives_its_linear_strain_exactly():
    def expression(x):
        return np.stack([x[0] ** 2, x[1] * x[2], 2.0 * x[2] ** 2])

    field = _displacement_field(2, expression, cells=4)
    point = np.array([[0.31, 0.62, 0.47]])
    samples = FemField.symmetric_gradient_of(field).evaluate(point)
    x, y, z = point[0]
    expected = np.array(
        [
            [2.0 * x, 0.0, 0.0],
            [0.0, z, 0.5 * y],
            [0.0, 0.5 * y, 4.0 * z],
        ]
    )
    np.testing.assert_allclose(samples.values.reshape(3, 3), expected, rtol=0.0, atol=1e-14)


def test_the_plane_strain_strip_reproduces_the_closed_form(strip_displacement):
    strain = FemField.symmetric_gradient_of(strip_displacement)
    assert strain.value_size == 4
    samples = strain.evaluate(PROBE_POINTS)
    assert samples.covered.all()
    tensors = samples.values.reshape(-1, 2, 2)
    np.testing.assert_allclose(tensors[:, 0, 0], EXPECTED_EPS_XX, rtol=1e-12)
    np.testing.assert_allclose(tensors[:, 1, 1], EXPECTED_EPS_YY, rtol=1e-9)
    np.testing.assert_allclose(tensors[:, 0, 1], 0.0, atol=1e-15)
    np.testing.assert_allclose(tensors[:, 0, 1], tensors[:, 1, 0], atol=0.0)
    # the numbers this case is graded against, printed once so a change is visible in the diff
    assert EXPECTED_EPS_XX == pytest.approx(8.762727e-06, rel=1e-6)
    assert EXPECTED_EPS_YY == pytest.approx(-2.055455e-06, rel=1e-6)


def test_the_out_of_plane_entry_is_a_stated_modelling_choice(strip_displacement):
    plane_strain = FemField.symmetric_gradient_of(strip_displacement, out_of_plane=0.0)
    assert plane_strain.value_size == 9
    tensor = plane_strain.evaluate(PROBE_POINTS[:1]).values.reshape(3, 3)
    np.testing.assert_allclose(tensor[:2, :2], np.diag([EXPECTED_EPS_XX, EXPECTED_EPS_YY]), atol=1e-15)
    np.testing.assert_array_equal(tensor[2, :], np.zeros(3))
    np.testing.assert_array_equal(tensor[:, 2], np.zeros(3))

    generalized = 3.5e-06
    field = FemField.symmetric_gradient_of(strip_displacement, out_of_plane=generalized)
    tensor = field.evaluate(PROBE_POINTS[:1]).values.reshape(3, 3)
    assert tensor[2, 2] == pytest.approx(generalized, rel=1e-15)
    # in the grid frame of a cross-section case the out-of-plane axis is the propagation axis
    transform = PointTransform(collapse_axes=(1,), permute=(0, 2, 1))
    grid = transform.apply_values(tensor[None], rank=2)[0]
    assert grid[1, 1] == pytest.approx(generalized, rel=1e-15)
    np.testing.assert_allclose(voigt_from_tensor(grid)[1], generalized, rtol=1e-15)


def test_a_scaled_symmetric_gradient_is_just_scaled(strip_displacement):
    plain = FemField.symmetric_gradient_of(strip_displacement).evaluate(PROBE_POINTS).values
    scaled = FemField.symmetric_gradient_of(strip_displacement, scale=-2.5).evaluate(PROBE_POINTS).values
    np.testing.assert_allclose(scaled, -2.5 * plain, rtol=1e-14)


def test_positions_and_tensor_components_reach_the_grid_frame_together(strip_displacement):
    """The strip sampled by a cross-section grid: mesh x -> grid x, mesh y -> grid z, y collapsed."""
    strain = FemField.symmetric_gradient_of(strip_displacement)
    edges = (
        np.linspace(-1.8e-6, 1.8e-6, 19),
        np.linspace(-0.5e-6, 0.5e-6, 2),
        np.linspace(-0.9e-6, 0.9e-6, 10),
    )
    transform = PointTransform(scale=1e6, offset=(2.0, 0.0, 1.0), collapse_axes=(1,), permute=(0, 2, 1))
    samples = sample_on_yee_lattices(strain, edges, lattices=("E0", "E2"), transform=transform)
    for lattice in ("E0", "E2"):
        assert samples.covered[lattice].all()
        grid = transform.apply_values(samples.values[lattice], rank=2)
        assert grid.shape == (*samples.covered[lattice].shape, 3, 3)
        np.testing.assert_allclose(grid[..., 0, 0], EXPECTED_EPS_XX, rtol=1e-12)
        np.testing.assert_allclose(grid[..., 2, 2], EXPECTED_EPS_YY, rtol=1e-9)
        # the collapsed propagation axis carries nothing, which is plane strain
        np.testing.assert_array_equal(grid[..., 1, :], np.zeros_like(grid[..., 1, :]))
        np.testing.assert_array_equal(grid[..., :, 1], np.zeros_like(grid[..., :, 1]))


def test_the_wrong_input_is_refused(strip_displacement):
    scalar = FemScalarField(
        fem.Function(fem.functionspace(strip_displacement.mesh, ("Lagrange", 1)), name="T"), name="T"
    )
    with pytest.raises(ValueError, match="needs a vector field"):
        FemField.symmetric_gradient_of(scalar)
    cube = _displacement_field(1, lambda x: 1e-3 * x)
    with pytest.raises(ValueError, match="two-dimensional modelling choice"):
        FemField.symmetric_gradient_of(cube, out_of_plane=0.0)
