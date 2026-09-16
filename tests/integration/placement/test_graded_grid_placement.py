"""Placing objects on a graded grid with a mesh override region.

The scene is a dielectric slab whose front face sits at z = 0 with a refinement region wrapped
around it, plus a plane source on the coarse side. The checks are that placement resolves the
policy from the volume's physical extent, that the refinement ends up around the interface, and
that the time step follows the finest cell.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx

_BACKGROUND = 50e-9
_TARGET = 12.5e-9
_DOMAIN_XY = 4 * _BACKGROUND
_DOMAIN_Z = 4e-6
_INTERFACE_Z = 0.0
_REGION_Z = (-200e-9, 600e-9)
_SOURCE_Z = -1.5e-6
_FLOAT32_SLACK = 1e-4


def _policy() -> fdtdx.GradedGrid:
    return fdtdx.GradedGrid(
        spacing=_BACKGROUND,
        regions=(fdtdx.RefinementRegion(spacing=(_BACKGROUND, _BACKGROUND, _TARGET), z=_REGION_Z),),
        max_ratio=1.4,
    )


def _scene(volume_kwargs=None):
    config = fdtdx.SimulationConfig(grid=_policy(), time=10e-15, dtype=jnp.float32)
    volume = fdtdx.SimulationVolume(**(volume_kwargs or {"partial_real_shape": (_DOMAIN_XY, _DOMAIN_XY, _DOMAIN_Z)}))
    objects, constraints = [volume], []

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=10,
        override_types={"min_x": "periodic", "max_x": "periodic", "min_y": "periodic", "max_y": "periodic"},
    )
    bound_dict, bound_constraints = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(bound_constraints)
    objects.extend(bound_dict.values())

    slab = fdtdx.UniformMaterialObject(name="slab", material=fdtdx.Material(permittivity=4.0))
    constraints.extend(
        [
            slab.same_size(volume, axes=(0, 1)),
            slab.place_at_center(volume, axes=(0, 1)),
            fdtdx.RealCoordinateConstraint(
                object="slab", axes=(2, 2), sides=("-", "+"), coordinates=(_INTERFACE_Z, _DOMAIN_Z / 2)
            ),
        ]
    )
    objects.append(slab)

    source = fdtdx.UniformPlaneSource(
        name="source",
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=1e-6),
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            fdtdx.RealCoordinateConstraint(object="source", axes=(2,), sides=("-",), coordinates=(_SOURCE_Z,)),
        ]
    )
    objects.append(source)
    return objects, constraints, config


@pytest.fixture(scope="module")
def placed():
    objects, constraints, config = _scene()
    obj_container, arrays, _params, config, info = fdtdx.place_objects(
        objects, config, constraints, jax.random.PRNGKey(0)
    )
    return obj_container, arrays, config, info


def _named(obj_container, name):
    return next(obj for obj in obj_container.objects if obj.name == name)


def test_placement_pins_a_rectilinear_grid_matching_the_volume(placed):
    """The policy is resolved and the volume's cell count comes from the resolved edges."""
    obj_container, _arrays, config, _info = placed
    grid = config.resolved_grid
    assert isinstance(grid, fdtdx.RectilinearGrid)
    assert grid.shape == obj_container.volume.grid_shape
    edges = np.asarray(grid.z_edges, dtype=np.float64)
    assert float(edges[-1] - edges[0]) == pytest.approx(_DOMAIN_Z, rel=_FLOAT32_SLACK)


def test_the_refinement_lands_on_the_interface(placed):
    """Cells around z = 0 are at the target width and the coarse background survives far away."""
    _obj_container, _arrays, config, _info = placed
    grid = config.resolved_grid
    edges = np.asarray(grid.z_edges, dtype=np.float64)
    widths = np.asarray(grid.dz, dtype=np.float64)
    inside = (edges[:-1] >= _REGION_Z[0] * (1 + _FLOAT32_SLACK)) & (edges[1:] <= _REGION_Z[1] * (1 + _FLOAT32_SLACK))
    assert widths[inside].max() <= _TARGET * (1 + _FLOAT32_SLACK)
    # The coarse cells are shrunk by one common factor so the extent comes out exact, so the widest
    # cell sits just below the background rather than exactly on it.
    assert _BACKGROUND * 0.97 <= widths.max() <= _BACKGROUND * (1 + _FLOAT32_SLACK)
    assert float(np.abs(edges - _INTERFACE_Z).min()) < _FLOAT32_SLACK * _TARGET


def test_the_slab_front_face_sits_on_the_requested_coordinate(placed):
    """A face given in metres lands on the grid edge at that coordinate, not near it."""
    obj_container, _arrays, config, _info = placed
    grid = config.resolved_grid
    lower, upper = _named(obj_container, "slab").grid_slice_tuple[2]
    assert float(grid.z_edges[lower]) == pytest.approx(_INTERFACE_Z, abs=_FLOAT32_SLACK * _TARGET)
    assert float(grid.z_edges[upper]) == pytest.approx(_DOMAIN_Z / 2, rel=_FLOAT32_SLACK)
    assert upper - lower > 0


def test_the_source_lands_in_the_coarse_region(placed):
    """The plane source is one cell thick at its requested coordinate, on the background mesh."""
    obj_container, _arrays, config, _info = placed
    grid = config.resolved_grid
    lower, upper = _named(obj_container, "source").grid_slice_tuple[2]
    assert upper - lower == 1
    assert float(grid.z_edges[lower]) == pytest.approx(_SOURCE_Z, abs=_BACKGROUND)
    assert float(grid.dz[lower]) == pytest.approx(_BACKGROUND, rel=5e-3)


def test_the_time_step_follows_the_finest_cell(placed):
    """CFL uses the realized minimum width, not the background."""
    _obj_container, _arrays, config, _info = placed
    grid = config.resolved_grid
    assert config.time_step_duration == pytest.approx(grid.cfl_time_step(config.courant_factor))
    coarse = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_BACKGROUND), time=10e-15)
    assert config.time_step_duration < coarse.time_step_duration
    assert min(grid.min_spacings) == pytest.approx(_TARGET, rel=_FLOAT32_SLACK)


def test_field_arrays_have_the_graded_shape(placed):
    """The initialized arrays follow the resolved grid, so the rest of the engine sees one shape."""
    _obj_container, arrays, config, _info = placed
    grid = config.resolved_grid
    assert arrays.fields.E.shape[1:] == grid.shape
    assert arrays.fields.H.shape[1:] == grid.shape
    assert arrays.inv_permittivities.shape[-3:] == grid.shape


def test_a_volume_without_a_physical_extent_raises(placed):
    """A graded policy cannot start from a cell count, and says so."""
    objects, constraints, config = _scene(volume_kwargs={"partial_grid_shape": (4, 4, 80)})
    with pytest.raises(ValueError, match="no partial_real_shape"):
        fdtdx.place_objects(objects, config, constraints, jax.random.PRNGKey(0))
