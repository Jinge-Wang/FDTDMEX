"""Point evaluation of a DOLFINx field with coverage flags, against analytic fields on known meshes.

Needs DOLFINx; skipped where it is not installed (the fork's CI has none). A Lagrange P1 space
reproduces a linear field exactly and P2 a quadratic one, so the operator's answer at any interior
point is checkable to round-off. Points outside the mesh must come back flagged, with ``NaN`` and
cell ``-1``, never as zeros. A planar mesh is sampled from a 3-D Yee grid with one cell along the
third axis by collapsing that coordinate.
"""

import numpy as np
import pytest

dolfinx = pytest.importorskip("dolfinx")
from dolfinx import fem, mesh  # noqa: E402
from mpi4py import MPI  # noqa: E402

from fdtdx.coupling import (  # noqa: E402
    FemScalarField,
    PointTransform,
    lattice_axes,
    lattice_points,
    sample_on_yee_lattices,
)


def _cube(n: int, degree: int) -> fem.Function:
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
    V = fem.functionspace(domain, ("Lagrange", degree))
    return fem.Function(V)


def _square(n: int, degree: int) -> fem.Function:
    domain = mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    V = fem.functionspace(domain, ("Lagrange", degree))
    return fem.Function(V)


def test_a_linear_field_on_p1_is_reproduced_exactly_inside_and_flagged_outside():
    f = _cube(6, 1)
    f.interpolate(lambda x: 300.0 + 10.0 * x[0] + 20.0 * x[1] + 30.0 * x[2])
    field = FemScalarField(f, name="T", unit="K")
    rng = np.random.default_rng(1)
    inside = rng.uniform(0.05, 0.95, size=(5000, 3))
    outside = rng.uniform(1.05, 2.0, size=(700, 3)) * rng.choice([-1.0, 1.0], size=(700, 1))
    points = np.concatenate([inside, outside], axis=0)
    samples = field.evaluate(points)
    assert samples.num_points == 5700
    assert samples.covered[:5000].all()
    assert not samples.covered[5000:].any()
    expected = 300.0 + 10.0 * inside[:, 0] + 20.0 * inside[:, 1] + 30.0 * inside[:, 2]
    np.testing.assert_allclose(samples.values[:5000], expected, rtol=0.0, atol=1e-10)
    assert np.isnan(samples.values[5000:]).all()
    assert (samples.cells[5000:] == -1).all()
    assert (samples.cells[:5000] >= 0).all()
    assert samples.num_uncovered == 700


def test_a_quadratic_field_on_p2_is_reproduced_exactly():
    f = _cube(4, 2)
    f.interpolate(lambda x: 1.0 + x[0] ** 2 - 2.0 * x[1] * x[2] + 3.0 * x[2] ** 2)
    field = FemScalarField(f)
    rng = np.random.default_rng(2)
    points = rng.uniform(0.02, 0.98, size=(3000, 3))
    samples = field.evaluate(points)
    assert samples.covered.all()
    expected = 1.0 + points[:, 0] ** 2 - 2.0 * points[:, 1] * points[:, 2] + 3.0 * points[:, 2] ** 2
    np.testing.assert_allclose(samples.values, expected, rtol=0.0, atol=1e-10)


def test_from_dofs_wraps_a_vector_the_way_thermalfem_stores_it():
    f = _cube(4, 1)
    f.interpolate(lambda x: 2.0 * x[0] - x[1])
    V = f.function_space
    dofs = np.array(f.x.array, copy=True)
    field = FemScalarField.from_dofs(V, dofs, name="T", unit="K")
    points = np.array([[0.25, 0.5, 0.5], [0.8, 0.1, 0.3]])
    samples = field.evaluate(points)
    np.testing.assert_allclose(samples.values, 2.0 * points[:, 0] - points[:, 1], atol=1e-12)
    with pytest.raises(ValueError, match="non-finite"):
        FemScalarField.from_dofs(V, np.full_like(dofs, np.nan))
    with pytest.raises(ValueError, match="entries"):
        FemScalarField.from_dofs(V, np.zeros(dofs.size + 1))


def test_from_thermal_sim_reads_the_private_space_and_the_dof_vector():
    """A stand-in for thermalFEM's thSim: it exposes ``_V``, ``T_dofs`` and ``status``."""
    f = _cube(3, 1)
    f.interpolate(lambda x: 5.0 + x[2])

    class Sim:
        _V = f.function_space
        T_dofs = np.array(f.x.array, copy=True)
        status = "ok"
        solution = None  # the field wraps T_dofs; the solution container is not read

    field = FemScalarField.from_thermal_sim(Sim())
    samples = field.evaluate(np.array([[0.5, 0.5, 0.25]]))
    np.testing.assert_allclose(samples.values, [5.25], atol=1e-12)

    class Unsolved:
        _V = f.function_space
        T_dofs = np.zeros_like(Sim.T_dofs)
        status = "not_run"
        solution = None

    with pytest.raises(RuntimeError, match="not solved"):
        FemScalarField.from_thermal_sim(Unsolved())

    class NoSpace:
        _V = None

    with pytest.raises(RuntimeError, match="function space"):
        FemScalarField.from_thermal_sim(NoSpace())


def test_a_planar_mesh_is_sampled_from_a_one_cell_thick_yee_grid_with_coverage():
    f = _square(8, 2)
    f.interpolate(lambda x: 300.0 + 40.0 * x[0] ** 2 + 10.0 * x[1])
    field = FemScalarField(f)
    # A Yee grid 1.2 wide starting at -0.1: the first and last cells poke outside the unit square.
    edges = (np.linspace(-0.1, 1.1, 13), np.linspace(0.0, 1.0, 11), np.array([0.0, 0.05]))
    samples = sample_on_yee_lattices(
        field, edges, lattices=("E0", "E1", "E2", "V"), transform=PointTransform(collapse_axes=(2,))
    )
    assert samples.lattices == ("E0", "E1", "E2", "V")
    assert samples.shape == (12, 10, 1)
    for lattice in samples.lattices:
        x, y, _ = lattice_axes(edges, lattice)
        X, Y = np.meshgrid(x, y, indexing="ij")
        covered = samples.covered[lattice][:, :, 0]
        # Points strictly inside are covered; strictly outside are not. A point on the boundary
        # within round-off (linspace puts an edge at 1 + 2e-16) may go either way.
        tol = 1e-9
        strictly_in = (X > tol) & (X < 1 - tol) & (Y > tol) & (Y < 1 - tol)
        strictly_out = (X < -tol) | (X > 1 + tol) | (Y < -tol) | (Y > 1 + tol)
        assert covered[strictly_in].all()
        assert not covered[strictly_out].any()
        values = samples.values[lattice][:, :, 0]
        expected = 300.0 + 40.0 * X**2 + 10.0 * Y
        np.testing.assert_allclose(values[covered], expected[covered], atol=1e-10)
        assert np.isnan(values[~covered]).all()
    report = samples.coverage_report()
    assert report["E0"]["num_uncovered"] > 0
    assert report["E0"]["min"] >= 300.0


def test_a_transform_offsets_scales_and_collapses_before_evaluation():
    f = _square(4, 1)
    f.interpolate(lambda x: x[0] + 2.0 * x[1])
    field = FemScalarField(f)
    # Yee points in metres around the origin; the mesh is the unit square in "micrometres".
    edges = (np.linspace(-0.5e-6, 0.5e-6, 5), np.linspace(-0.5e-6, 0.5e-6, 5), np.array([-1e-6, 1e-6]))
    transform = PointTransform(offset=(0.5, 0.5, 0.0), scale=1e6, collapse_axes=(2,), collapse_value=0.0)
    samples = sample_on_yee_lattices(field, edges, lattices=("E2",), transform=transform)
    pts, shape = lattice_points(edges, "E2")
    mapped = transform.apply(pts)
    expected = (mapped[:, 0] + 2.0 * mapped[:, 1]).reshape(shape)
    covered = samples.covered["E2"]
    assert covered.sum() >= 9  # the interior points
    np.testing.assert_allclose(samples.values["E2"][covered], expected[covered], atol=1e-12)


def test_many_points_in_one_call_and_bounds():
    f = _cube(5, 1)
    f.interpolate(lambda x: x[0])
    field = FemScalarField(f)
    lower, upper = field.bounds
    np.testing.assert_allclose(lower, [0, 0, 0])
    np.testing.assert_allclose(upper, [1, 1, 1])
    rng = np.random.default_rng(3)
    points = rng.uniform(-0.2, 1.2, size=(300_000, 3))
    samples = field.evaluate(points)
    inside = np.all((points > 0) & (points < 1), axis=1)
    assert samples.covered[inside].all()
    assert not samples.covered[~inside].any()
    np.testing.assert_allclose(samples.values[inside], points[inside, 0], atol=1e-10)
