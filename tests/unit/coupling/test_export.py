"""The loader's material arrays exported onto Cartesian lattices for an external solver.

The checks pin, in order: that the export is an exact round trip of the loader's own
inverse-permittivity arrays and carries the grid's edges unchanged; that the values sit on the
lattice :func:`fdtdx.coupling.lattice_points` describes, entry for entry, which is the per-slot
convention an FDFD consumer relies on; that the three lattices stay independent; that the vertex
off-diagonal tier comes out as the *inverse*-permittivity entries the loader stored and is refused
when the scene has none; that the drop ratio a diagonal-only consumer must gate on is reported; and
that the conductivity is folded into the imaginary part with the sign the ``exp(-i omega t)``
convention needs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.coupling import lattice_points
from fdtdx.coupling.export import (
    complex_permittivity_slots,
    offdiag_drop_ratio,
    yee_arrays_to_cartesian,
)
from fdtdx.materials import Material
from fdtdx.objects.static_material.cylinder import Cylinder
from fdtdx.objects.static_material.static import SimulationVolume

_D = 50e-9
_N = 16
_COUNTER = [0]

EPS_CORE = 12.1
EPS_BG = 2.085


@pytest.fixture
def float64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _tag() -> str:
    _COUNTER[0] += 1
    return f"_{_COUNTER[0]}"


def _scene(sampling: str = "yee_smooth", with_disk: bool = True, cells: int = _N):
    """A disk in a periodic 2-D box through the fork's own loader."""
    tag = _tag()
    config = SimulationConfig(
        time=1e-15,
        grid=UniformGrid(spacing=_D),
        dtype=jnp.float64,
        material_sampling=sampling,
        yee_smooth_full_tensor=True,
        yee_smooth_offdiag_placement="node",
    )
    materials = {"bg": Material(permittivity=EPS_BG), "core": Material(permittivity=EPS_CORE)}
    volume = SimulationVolume(partial_grid_shape=(cells, cells, 1), material=materials["bg"], name=f"vol{tag}")
    objects = [volume]
    if with_disk:
        objects.append(
            Cylinder(
                axis=2,
                radius=0.30 * cells * _D,
                material_name="core",
                materials=materials,
                partial_grid_shape=(None, None, 1),
                placement_order=1,
                name=f"disk{tag}",
            )
        )
    boundaries, constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="periodic"), volume
    )
    _, arrays, _, config, _ = fdtdx.place_objects([*objects, *boundaries.values()], config, constraints)
    return arrays, config


# ---------------------------------------------------------------------------
# The round trip and the grid it refers to
# ---------------------------------------------------------------------------


def test_the_export_round_trips_the_loader_arrays_and_keeps_the_grid_edges(float64):
    arrays, config = _scene()
    values, edges = yee_arrays_to_cartesian(arrays, config.grid)

    assert set(values) == {"E0", "E1", "E2"}
    inv = np.asarray(arrays.inv_permittivities, dtype=np.float64)
    for component, lattice in enumerate(("E0", "E1", "E2")):
        assert values[lattice].shape == inv.shape[1:]
        np.testing.assert_allclose(1.0 / values[lattice], inv[component], rtol=1e-15, atol=0.0)
        assert values[lattice].min() >= 1.0 / inv[component].max() - 1e-12
    for axis in range(3):
        np.testing.assert_allclose(edges[axis], np.asarray(config.grid.edges(axis)), rtol=0.0, atol=0.0)
    # every exported permittivity lies between the two the scene names
    for lattice in values:
        assert values[lattice].min() >= EPS_BG - 1e-9
        assert values[lattice].max() <= EPS_CORE + 1e-9


def test_the_exported_values_sit_on_the_lattice_points_the_sampler_describes(float64):
    """The per-slot convention: E<c> is offset by half a cell on axis c and on no other."""
    arrays, config = _scene()
    values, edges = yee_arrays_to_cartesian(arrays, config.grid)

    dx, dy, dz = (float(edges[a][1] - edges[a][0]) for a in range(3))
    expected_first = {
        "E0": (edges[0][0] + 0.5 * dx, edges[1][0], edges[2][0]),
        "E1": (edges[0][0], edges[1][0] + 0.5 * dy, edges[2][0]),
        "E2": (edges[0][0], edges[1][0], edges[2][0] + 0.5 * dz),
    }
    offsets = {"E0": (0.5, 0.0, 0.0), "E1": (0.0, 0.5, 0.0), "E2": (0.0, 0.0, 0.5)}
    for lattice, first in expected_first.items():
        points, shape = lattice_points(edges, lattice)
        assert shape == values[lattice].shape
        assert points.shape == (values[lattice].size, 3)
        np.testing.assert_allclose(points[0], first, rtol=0.0, atol=1e-18)
        # the points ravel in the array's own ``ij`` order, so the exported value at (i, j, k)
        # belongs to the point at the same flat index
        shift = np.array([offsets[lattice][a] * (dx, dy, dz)[a] for a in range(3)])
        axes = [np.asarray(edges[a][:-1]) + shift[a] for a in range(3)]
        expected = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], axis=-1)
        np.testing.assert_allclose(points, expected, rtol=0.0, atol=1e-18)


def test_a_uniform_scene_gives_the_same_permittivity_on_all_three_lattices(float64):
    arrays, config = _scene(with_disk=False)
    values, _ = yee_arrays_to_cartesian(arrays, config.grid)
    for lattice in ("E0", "E1", "E2"):
        np.testing.assert_allclose(values[lattice], EPS_BG, rtol=1e-12)


def test_a_subset_of_lattices_comes_back_with_exactly_those_keys(float64):
    arrays, config = _scene()
    values, _ = yee_arrays_to_cartesian(arrays, config.grid, lattices=("E2", "E0"))
    assert set(values) == {"E0", "E2"}
    with pytest.raises(ValueError, match="subset"):
        yee_arrays_to_cartesian(arrays, config.grid, lattices=("E0", "H1"))


def test_the_grid_can_be_given_as_a_config_or_as_three_edge_arrays(float64):
    arrays, config = _scene()
    from_grid, edges = yee_arrays_to_cartesian(arrays, config.grid)
    from_config, _ = yee_arrays_to_cartesian(arrays, config)
    from_edges, _ = yee_arrays_to_cartesian(arrays, edges)
    for lattice in from_grid:
        np.testing.assert_array_equal(from_grid[lattice], from_config[lattice])
        np.testing.assert_array_equal(from_grid[lattice], from_edges[lattice])


def test_a_grid_that_does_not_match_the_arrays_is_refused(float64):
    arrays, config = _scene()
    edges = tuple(np.asarray(config.grid.edges(a)) for a in range(3))
    wrong = (edges[0][:-1], edges[1], edges[2])
    with pytest.raises(ValueError, match="cells but the grid edges"):
        yee_arrays_to_cartesian(arrays, wrong)


# ---------------------------------------------------------------------------
# The off-diagonal tier
# ---------------------------------------------------------------------------


def test_the_vertex_offdiagonals_come_out_as_inverse_permittivity_entries(float64):
    arrays, config = _scene()
    assert arrays.inv_permittivity_offdiag is not None
    values, _ = yee_arrays_to_cartesian(arrays, config.grid, include_offdiag=True)
    assert set(values) == {"E0", "E1", "E2", "V"}
    stored = np.asarray(arrays.inv_permittivity_offdiag, dtype=np.float64)
    np.testing.assert_allclose(values["V"], stored, rtol=0.0, atol=0.0)
    assert values["V"].shape == (3, *values["E0"].shape)
    assert np.count_nonzero(values["V"]) > 0  # the curved rim wrote entries


def test_a_scene_without_an_offdiagonal_tier_refuses_the_request(float64):
    arrays, config = _scene(sampling="box")
    assert arrays.inv_permittivity_offdiag is None
    with pytest.raises(ValueError, match="no off-diagonal tier"):
        yee_arrays_to_cartesian(arrays, config.grid, include_offdiag=True)
    values, _ = yee_arrays_to_cartesian(arrays, config.grid)
    assert set(values) == {"E0", "E1", "E2"}


def test_the_drop_ratio_is_reported_for_a_diagonal_only_consumer(float64):
    arrays, _ = _scene()
    report = offdiag_drop_ratio(arrays)
    assert report["num_nonzero"] > 0
    assert report["max_abs_offdiag"] > 0.0
    assert report["max_diag_spread"] > 0.0
    assert report["ratio"] == pytest.approx(report["max_abs_offdiag"] / report["max_diag_spread"])
    assert 0.0 < report["ratio"] < 10.0

    uniform, _ = _scene(with_disk=False)
    flat = offdiag_drop_ratio(uniform)
    assert flat["num_nonzero"] == 0
    assert flat["ratio"] == 0.0

    box, _ = _scene(sampling="box")
    assert offdiag_drop_ratio(box) == {
        "max_abs_offdiag": 0.0,
        "max_diag_spread": 0.0,
        "ratio": 0.0,
        "num_nonzero": 0,
    }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def test_the_conductivity_folds_into_a_positive_imaginary_part(float64):
    arrays, config = _scene()
    values, _ = yee_arrays_to_cartesian(arrays, config.grid, include_offdiag=True)
    shape = values["E0"].shape
    sigma = np.zeros(shape)
    sigma[shape[0] // 2, shape[1] // 2, 0] = 3.0
    omega, eps0 = 2.0, 1.0

    complexed = complex_permittivity_slots(values, sigma, omega=omega, eps0=eps0)
    for lattice in ("E0", "E1", "E2"):
        assert complexed[lattice].dtype == np.complex128
        np.testing.assert_allclose(complexed[lattice].real, values[lattice], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(complexed[lattice].imag, sigma / (omega * eps0), rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(complexed["V"], values["V"])  # untouched, and still real

    per_slot = np.stack([sigma, 2.0 * sigma, 0.0 * sigma])
    anisotropic = complex_permittivity_slots(values, per_slot, omega=omega, eps0=eps0)
    np.testing.assert_allclose(anisotropic["E1"].imag, 2.0 * sigma / (omega * eps0))
    np.testing.assert_allclose(anisotropic["E2"].imag, 0.0)

    lossless = complex_permittivity_slots(values, None, omega=omega)
    np.testing.assert_allclose(lossless["E0"].imag, 0.0)

    with pytest.raises(ValueError, match="omega must be positive"):
        complex_permittivity_slots(values, sigma, omega=0.0)
    with pytest.raises(ValueError, match="electric_conductivity must be"):
        complex_permittivity_slots(values, np.zeros((2, 2)), omega=omega)
