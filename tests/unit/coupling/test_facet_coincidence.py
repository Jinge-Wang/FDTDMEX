"""The facet-coincidence diagnostic, on a synthetic two-material scene.

A slab whose face sits exactly on a grid edge is the case the P1 design note measured on a
three-layer capacitor: the two lattices whose points sit on that edge read the field on whichever
side the evaluator listed first, while the lattice half a cell away reads the other side, and
nothing in the run says so. Moving the face half a cell swaps which lattices are exposed, and that
swap is the whole content of the diagnostic -- so both positions are pinned here, together with the
jump the tie-break chose between.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.coupling import facet_coincidence_report, samples_from_callable, uniform_samples
from fdtdx.materials import Material
from fdtdx.objects.static_material.static import SimulationVolume, UniformMaterialObject

EPS_BG = 2.085
EPS_CORE = 12.1
_D = 25e-9
_CELLS = 12
_COUNTER = [0]

# The two sides of the synthetic discontinuous field, and the value a point on the facet takes
# under a "first cell listed wins" tie-break with the strict comparison used below.
LOW, HIGH = 1.0, 7.0


@pytest.fixture
def float64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _tag() -> str:
    _COUNTER[0] += 1
    return f"fc{_COUNTER[0]}"


def _slab_scene(face_offset: float):
    """Background with a slab filling ``x >= face``; ``face_offset`` moves the face off the edge."""
    tag = _tag()
    config = SimulationConfig(
        time=1e-15,
        grid=UniformGrid(spacing=_D),
        dtype=jnp.float64,
        material_sampling="yee_smooth",
        yee_smooth_full_tensor=True,
        yee_smooth_offdiag_placement="node",
    )
    volume = SimulationVolume(partial_grid_shape=(_CELLS, 4, 4), material=Material(permittivity=EPS_BG), name=f"v{tag}")
    span = _CELLS * _D
    face = 6 * _D + face_offset
    slab = UniformMaterialObject(
        material=Material(permittivity=EPS_CORE),
        partial_real_shape=(span - face, None, None),
        partial_real_position=(0.5 * face, 0.0, 0.0),
        placement_order=1,
        name=f"s{tag}",
    )
    container, _arrays, _, resolved, info = fdtdx.place_objects([volume, slab], config, [])
    placed = {o.name: o for o in container.object_list}[slab.name]
    return resolved.resolved_grid, info, float(placed.metric_bounds[0][0])


def _names(material_map) -> dict[str, str]:
    """The loader's own name for each of the two materials, keyed "core" and "bg"."""
    table = material_map["material_table"]
    labels = material_map["material_names"]
    out = {}
    for index, material in enumerate(table):
        if float(material.permittivity[0]) == EPS_CORE:
            out["core"] = labels[index]
        elif float(material.permittivity[0]) == EPS_BG:
            out["bg"] = labels[index]
    return out


def _step_samples(grid, x_face: float):
    """A field that jumps across ``x_face``; a point exactly on it takes the high side."""
    return samples_from_callable(grid, lambda p: np.where(p[:, 0] < x_face, LOW, HIGH), name="T", unit="K")


def test_a_facet_on_a_grid_edge_is_reported_on_the_lattices_that_sit_on_that_edge(float64):
    grid, info, x_face = _slab_scene(0.0)
    samples = _step_samples(grid, x_face)
    report = facet_coincidence_report(samples, info["yee_material_map"])
    names = set(_names(info["yee_material_map"]).values())

    # E1 and E2 sit on the x edges; their smoothing boxes are centred on the facet.
    assert report.num_coincident["E1"] > 0
    assert report.num_coincident["E2"] > 0
    assert report.num_coincident["V"] > 0
    # E0 sits at the cell centre in x, so the facet lands on its box face, not through its point.
    assert report.num_coincident.get("E0", 0) == 0
    assert report.total_coincident == sum(report.num_coincident.values())

    listed = [p for p in report.points if p.lattice == "E1"]
    assert listed
    for point in listed:
        assert point.fill == pytest.approx(0.5, abs=1e-12)
        assert point.distance <= report.tol
        assert point.axis == 0
        assert point.point[0] == pytest.approx(x_face, abs=1e-15)
        assert {point.material_front, point.material_back} == names
        assert point.material_taken in names
        # The tie-break took the high side; the two neighbours straddle the facet, so the jump the
        # sampler chose between is visible without re-running the source solver.
        assert point.value == pytest.approx(HIGH)
        assert point.neighbour_values == (pytest.approx(LOW), pytest.approx(HIGH))
        assert point.neighbour_jump == pytest.approx(HIGH - LOW)
        assert point.relative_jump == pytest.approx((HIGH - LOW) / HIGH)
    assert report.max_relative_jump["E1"] == pytest.approx((HIGH - LOW) / HIGH)


def test_moving_the_facet_half_a_cell_swaps_which_lattices_are_exposed(float64):
    grid, info, x_face = _slab_scene(0.5 * _D)
    report = facet_coincidence_report(_step_samples(grid, x_face), info["yee_material_map"])
    assert report.num_coincident.get("E0", 0) > 0
    assert report.num_coincident.get("E1", 0) == 0
    assert report.num_coincident.get("E2", 0) == 0
    assert report.num_coincident.get("V", 0) == 0
    exposed = [p for p in report.points if p.lattice == "E0"]
    assert exposed and all(p.point[0] == pytest.approx(x_face, abs=1e-15) for p in exposed)


def test_a_continuous_field_on_the_same_facet_reports_the_points_with_no_jump(float64):
    """The diagnostic is geometry: it flags the points, and the field says whether it matters."""
    grid, info, _ = _slab_scene(0.0)
    report = facet_coincidence_report(uniform_samples(grid, 300.0), info["yee_material_map"])
    assert report.num_coincident["E1"] > 0
    assert report.max_relative_jump["E1"] == pytest.approx(0.0)
    assert all(p.neighbour_jump == pytest.approx(0.0) for p in report.points)


def test_the_report_counts_every_blended_pixel_it_examined_and_serialises(float64):
    grid, info, x_face = _slab_scene(0.0)
    report = facet_coincidence_report(_step_samples(grid, x_face), info["yee_material_map"], max_listed=4)
    for lattice, coincident in report.num_coincident.items():
        assert report.num_examined[lattice] >= coincident
    assert len(report.points) <= 4
    payload = json.loads(json.dumps(report.as_dict()))
    assert payload["total_coincident"] == report.total_coincident
    assert payload["points"][0]["material_taken"] in set(_names(info["yee_material_map"]).values())


def test_a_mapping_of_fields_is_accepted_and_a_mismatched_grid_is_refused(float64):
    grid, info, x_face = _slab_scene(0.0)
    samples = _step_samples(grid, x_face)
    report = facet_coincidence_report({"T": samples}, info["yee_material_map"])
    assert report.num_coincident["E1"] > 0
    other, _, other_face = _slab_scene(0.0)
    shifted = _step_samples(other, other_face)
    shifted.edges = tuple(e + 1e-3 for e in shifted.edges)
    with pytest.raises(ValueError, match="different grid"):
        facet_coincidence_report({"T": samples, "U": shifted}, info["yee_material_map"])


def test_a_scene_without_the_smoothing_record_is_refused_with_the_reason(float64):
    tag = _tag()
    config = SimulationConfig(time=1e-15, grid=UniformGrid(spacing=_D), dtype=jnp.float64, material_sampling="yee")
    volume = SimulationVolume(partial_grid_shape=(_CELLS, 4, 4), material=Material(permittivity=EPS_BG), name=f"v{tag}")
    slab = UniformMaterialObject(
        material=Material(permittivity=EPS_CORE),
        partial_real_shape=(6 * _D, None, None),
        partial_real_position=(3 * _D, 0.0, 0.0),
        placement_order=1,
        name=f"s{tag}",
    )
    _, _, _, resolved, info = fdtdx.place_objects([volume, slab], config, [])
    with pytest.raises(ValueError, match="yee_smooth"):
        facet_coincidence_report(uniform_samples(resolved.resolved_grid, 300.0), info["yee_material_map"])
