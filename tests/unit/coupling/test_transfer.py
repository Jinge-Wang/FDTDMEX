"""A Cartesian array wrapped as a FEM source callable, and the transpose of that map.

The pure-NumPy checks pin the interpolation itself: a trilinear field is reproduced exactly, the
weights sum to one, a thin axis is constant, the three out-of-grid policies do what they say, a
frame transform is applied in the stated direction, and — the property the inverse-design chain
depends on — the transpose is the exact adjoint of the forward map, measured with a dot-product
test.

The DOLFINx check closes the seam end to end: the array becomes the right-hand side of a real
Poisson solve and the temperature it produces is compared with the closed-form one-dimensional
conduction profile, for a uniform source and for a linear one. Skipped where DOLFINx is not
installed.
"""

import numpy as np
import pytest

from fdtdx.coupling import PointTransform
from fdtdx.coupling.frames import inverse_point_transform
from fdtdx.coupling.lattice import sample_axes
from fdtdx.coupling.transfer import (
    MultilinearTransfer,
    cartesian_to_fem_source,
    fem_to_cartesian_design,
    multilinear_weights,
)

# ---------------------------------------------------------------------------
# The interpolation
# ---------------------------------------------------------------------------


def _grid(nx=6, ny=5, nz=4):
    edges = (np.linspace(-1.0, 2.0, nx + 1), np.linspace(0.0, 1.0, ny + 1), np.linspace(-0.5, 0.5, nz + 1))
    return edges, (nx, ny, nz)


def test_the_sample_axes_are_the_cell_centres_or_the_named_lattice():
    edges, shape = _grid()
    centres = sample_axes(edges, "cell")
    for axis in range(3):
        np.testing.assert_allclose(centres[axis], 0.5 * (edges[axis][:-1] + edges[axis][1:]))
        assert centres[axis].size == shape[axis]
    e0 = sample_axes(edges, "E0")
    np.testing.assert_allclose(e0[0], centres[0])  # E0 is offset by half a cell on x only
    np.testing.assert_allclose(e0[1], edges[1][:-1])
    with pytest.raises(ValueError, match="strictly increasing"):
        sample_axes((np.array([1.0, 0.0]), edges[1], edges[2]))


def test_a_trilinear_field_is_reproduced_exactly_and_the_weights_sum_to_one():
    edges, shape = _grid()
    axes = sample_axes(edges, "cell")
    X, Y, Z = np.meshgrid(*axes, indexing="ij")

    def f(x, y, z):
        return 2.0 + 3.0 * x - 1.5 * y + 0.5 * z + 0.7 * x * y - 0.2 * y * z + 0.1 * x * z + 0.3 * x * y * z

    values = f(X, Y, Z)
    rng = np.random.default_rng(11)
    points = np.stack(
        [
            rng.uniform(axes[0][0], axes[0][-1], 500),
            rng.uniform(axes[1][0], axes[1][-1], 500),
            rng.uniform(axes[2][0], axes[2][-1], 500),
        ],
        axis=-1,
    )
    indices, weights = multilinear_weights(points, edges, shape=shape)
    assert indices.shape == (500, 8) and weights.shape == (500, 8)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-14)
    assert indices.min() >= 0 and indices.max() < values.size

    source = cartesian_to_fem_source(values, edges)
    got = source(points.T)
    np.testing.assert_allclose(got, f(points[:, 0], points[:, 1], points[:, 2]), rtol=0.0, atol=1e-12)


def test_a_constant_field_comes_back_constant_including_on_the_sample_points():
    edges, shape = _grid()
    values = np.full(shape, 7.25)
    axes = sample_axes(edges, "cell")
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    on_nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()])
    source = cartesian_to_fem_source(values, edges)
    np.testing.assert_allclose(source(on_nodes), 7.25, rtol=0.0, atol=1e-14)


def test_a_thin_axis_is_constant_and_never_puts_a_point_outside():
    """A quasi-2-D scene: one cell along z, a mesh that lives at any z at all."""
    edges = (np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 4), np.array([-0.1, 0.1]))
    axes = sample_axes(edges, "cell")
    X, Y, _ = np.meshgrid(*axes, indexing="ij")
    values = X + 2.0 * Y
    assert values.shape == (4, 3, 1)
    source = cartesian_to_fem_source(values, edges, outside="error")
    points = np.array([[0.3, 0.3, 0.3], [0.3, 0.3, -50.0], [0.3, 0.3, 0.0]]).T
    got = source(points)
    np.testing.assert_allclose(got, got[0], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("policy, expected", [("zero", 0.0), ("nearest", None)])
def test_the_out_of_grid_policies(policy, expected):
    edges, _ = _grid()
    axes = sample_axes(edges, "cell")
    X, _, _ = np.meshgrid(*axes, indexing="ij")
    values = 1.0 + X
    source = cartesian_to_fem_source(values, edges, outside=policy)
    far = np.array([[10.0, 0.5, 0.0]]).T
    got = source(far)
    if expected is None:
        assert got[0] == pytest.approx(1.0 + axes[0][-1])  # clamped to the last sample
    else:
        assert got[0] == expected


def test_a_point_outside_is_an_error_on_request():
    edges, shape = _grid()
    source = cartesian_to_fem_source(np.zeros(shape), edges, outside="error")
    with pytest.raises(ValueError, match="outside the Cartesian grid"):
        source(np.array([[10.0, 0.5, 0.0]]).T)
    with pytest.raises(ValueError, match="outside must be one of"):
        cartesian_to_fem_source(np.zeros(shape), edges, outside="clip")


def test_the_callable_refuses_the_wrong_coordinate_layout():
    edges, shape = _grid()
    source = cartesian_to_fem_source(np.zeros(shape), edges)
    with pytest.raises(ValueError, match=r"\(3, N\) coordinates"):
        source(np.zeros((5, 3)))
    with pytest.raises(ValueError, match="3-D Cartesian array"):
        cartesian_to_fem_source(np.zeros((4, 4)), edges)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def test_the_transform_maps_mesh_coordinates_onto_the_grid_and_inverts_the_sampler_direction():
    # a grid in metres centred on the origin; the mesh drawn in micrometres from its own corner
    edges = (np.linspace(-0.5e-6, 0.5e-6, 5), np.linspace(-0.5e-6, 0.5e-6, 5), np.array([-1e-6, 1e-6]))
    axes = sample_axes(edges, "cell")
    X, Y, _ = np.meshgrid(*axes, indexing="ij")
    values = 1e6 * X + 2e6 * Y

    yee_to_mesh = PointTransform(offset=(0.5, 0.5, 0.0), scale=1e6)
    mesh_to_yee = inverse_point_transform(yee_to_mesh)
    # the two are inverse on random points
    rng = np.random.default_rng(3)
    pts = rng.uniform(-0.4e-6, 0.4e-6, size=(50, 3))
    np.testing.assert_allclose(mesh_to_yee.apply(yee_to_mesh.apply(pts)), pts, rtol=1e-12, atol=1e-18)

    source = cartesian_to_fem_source(values, edges, transform=mesh_to_yee, outside="nearest")
    mesh_points = np.array([[0.5, 0.5, 0.0], [0.3, 0.7, 0.0]])  # micrometres, mesh frame
    got = source(mesh_points.T)
    yee = mesh_to_yee.apply(mesh_points)
    np.testing.assert_allclose(got, 1e6 * yee[:, 0] + 2e6 * yee[:, 1], rtol=1e-10, atol=1e-12)


def test_a_permuting_transform_inverts_and_a_collapsing_one_is_refused():
    forward = PointTransform(offset=(1.0, -2.0, 0.5), scale=3.0, permute=(0, 2, 1))
    back = inverse_point_transform(forward)
    rng = np.random.default_rng(7)
    pts = rng.uniform(-2.0, 2.0, size=(40, 3))
    np.testing.assert_allclose(back.apply(forward.apply(pts)), pts, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="not invertible"):
        inverse_point_transform(PointTransform(collapse_axes=(2,)))
    with pytest.raises(ValueError, match="not invertible"):
        inverse_point_transform(PointTransform(scale=0.0))


# ---------------------------------------------------------------------------
# The transpose (the contract the inverse-design chain relies on)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outside", ["zero", "nearest"])
def test_the_transpose_is_the_adjoint_of_the_forward_map(outside):
    """<y, W g> == <W^T y, g> for random g and y, including points outside the grid."""
    edges, shape = _grid()
    rng = np.random.default_rng(19)
    inside = rng.uniform([-1.0, 0.0, -0.5], [2.0, 1.0, 0.5], size=(400, 3))
    outside_points = rng.uniform([3.0, 2.0, 1.0], [5.0, 4.0, 3.0], size=(60, 3))
    points = np.concatenate([inside, outside_points], axis=0)

    transfer = MultilinearTransfer.build(points, edges, outside=outside, shape=shape)
    g = rng.normal(size=shape)
    y = rng.normal(size=points.shape[0])

    lhs = float(y @ transfer.forward(g))
    rhs = float((transfer.transpose(y) * g).sum())
    assert lhs == pytest.approx(rhs, rel=1e-13, abs=1e-13)

    # and through the public pair of functions, which must use the same weights
    source = cartesian_to_fem_source(g, edges, outside=outside)
    np.testing.assert_allclose(source(points.T), transfer.forward(g), rtol=0.0, atol=0.0)
    scattered = fem_to_cartesian_design(y, points, edges, outside=outside, shape=shape)
    np.testing.assert_allclose(scattered, transfer.transpose(y), rtol=0.0, atol=0.0)


def test_the_transpose_accepts_either_point_layout_and_conserves_the_total():
    edges, shape = _grid()
    axes = sample_axes(edges, "cell")
    rng = np.random.default_rng(23)
    lower = [float(a[0]) for a in axes]
    upper = [float(a[-1]) for a in axes]
    points = rng.uniform(lower, upper, size=(300, 3))
    y = rng.normal(size=300)
    a = fem_to_cartesian_design(y, points, edges, shape=shape)
    b = fem_to_cartesian_design(y, points.T, edges, shape=shape)
    np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0)
    # weights sum to one per interior point, so the scattered total is the point total
    assert a.sum() == pytest.approx(y.sum(), rel=1e-12)
    with pytest.raises(ValueError, match="values but"):
        fem_to_cartesian_design(y[:10], points, edges, shape=shape)


def test_the_transpose_of_a_transform_matches_the_forward_transform():
    edges, shape = _grid()
    transform = PointTransform(offset=(0.1, -0.2, 0.05), scale=0.5)
    rng = np.random.default_rng(29)
    points = rng.uniform(-1.0, 1.0, size=(200, 3))
    transfer = MultilinearTransfer.build(points, edges, transform=transform, shape=shape)
    g = rng.normal(size=shape)
    y = rng.normal(size=200)
    assert float(y @ transfer.forward(g)) == pytest.approx(
        float((fem_to_cartesian_design(y, points, edges, transform=transform, shape=shape) * g).sum()),
        rel=1e-13,
        abs=1e-13,
    )


# ---------------------------------------------------------------------------
# End to end: the array as a real Poisson right-hand side
# ---------------------------------------------------------------------------


def _solve_conduction(source_callable, k: float, size: float, n: int, degree: int = 2):
    """-k div grad T = q on [0, size]^2, T = 300 on y = 0, natural (adiabatic) elsewhere."""
    import ufl
    from dolfinx import default_scalar_type, fem, mesh
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI

    domain = mesh.create_rectangle(
        MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([size, size])], [n, n], mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", degree))
    q = fem.Function(V)
    q.interpolate(source_callable)

    T, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = fem.Constant(domain, default_scalar_type(k)) * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx
    L = ufl.inner(q, v) * ufl.dx

    def bottom(x):
        return np.isclose(x[1], 0.0)

    dofs = fem.locate_dofs_geometrical(V, bottom)
    bc = fem.dirichletbc(default_scalar_type(300.0), dofs, V)
    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="fdtdx_grid_transfer_",
    )
    solution = problem.solve()
    uh = solution[0] if isinstance(solution, tuple) else solution
    return uh, V


@pytest.mark.parametrize("linear_source", [False, True])
def test_the_cartesian_source_reproduces_the_analytic_conduction_profile(linear_source):
    """The 1-D heat balance -k T'' = q with T(0) = 300 and T'(L) = 0, through the transfer."""
    pytest.importorskip("dolfinx")
    k, size, n_grid = 1.3e-6, 2.0, 41
    q0, q1 = 1e-6, (0.4e-6 if linear_source else 0.0)

    axes = (np.linspace(0.0, size, n_grid), np.linspace(0.0, size, n_grid), np.array([0.0]))
    _, Y, _ = np.meshgrid(*axes, indexing="ij")
    values = q0 + q1 * Y
    source = cartesian_to_fem_source(values, axes, outside="nearest")

    # the analytic profile is quadratic in z for a uniform source and cubic for a linear one, so
    # the element degree is raised with it and the only error left is the transfer's own
    uh, _ = _solve_conduction(source, k=k, size=size, n=32, degree=3 if linear_source else 2)
    coords = uh.function_space.tabulate_dof_coordinates()
    z = coords[:, 1]
    # T = 300 + (1/k) [ (q0 L + q1 L^2/2) z - q0 z^2/2 - q1 z^3/6 ]
    analytic = 300.0 + ((q0 * size + q1 * size**2 / 2.0) * z - q0 * z**2 / 2.0 - q1 * z**3 / 6.0) / k
    got = uh.x.array.real

    error = np.abs(got - analytic).max() / np.abs(analytic - 300.0).max()
    assert error < 1e-9, f"max relative error {error:g}"
    expected_max = 300.0 + (q0 * size**2 / 2.0 + q1 * size**3 / 3.0) / k
    assert got.max() == pytest.approx(expected_max, rel=1e-9)


def test_a_uniform_source_through_the_transfer_matches_a_constant_ufl_source():
    """The transfer is not merely self-consistent: it matches the source written analytically."""
    pytest.importorskip("dolfinx")
    k, size, q0 = 1.3e-6, 2.0, 1e-6
    axes = (np.linspace(0.0, size, 21), np.linspace(0.0, size, 21), np.array([0.0]))
    from_grid, _ = _solve_conduction(
        cartesian_to_fem_source(np.full((21, 21, 1), q0), axes, outside="nearest"),
        k=k,
        size=size,
        n=24,
    )
    analytic_source, _ = _solve_conduction(lambda x: np.full(x.shape[1], q0), k=k, size=size, n=24)
    np.testing.assert_allclose(from_grid.x.array.real, analytic_source.x.array.real, rtol=1e-12, atol=1e-10)
