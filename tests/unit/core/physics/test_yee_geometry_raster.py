"""Per-Yee-point material sampling: lattices, continuous shapes, priority and the metric shadow.

Everything here is placement plus array assembly — no mode solve and no time loop.

The contrast the whole stage exists for: with ``material_sampling="box"`` an object's extent is first
rounded to whole cells and then drawn on the cell centres, so a 500 nm bus becomes 480 nm at a 40 nm
grid and 512 nm at a 32 nm grid, identically at every sub-cell offset. With
``material_sampling="yee"`` the object keeps its 500 nm extent and each field component samples it at
its own position, so a single offset is wrong by at most one sample and the average over sub-cell
offsets is unbiased.
"""

import numpy as np
import pytest

import fdtdx
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.core.physics.geometry_raster import yee_lattice_coordinates
from fdtdx.materials import Material
from fdtdx.objects.static_material.cylinder import Cylinder
from fdtdx.objects.static_material.gds_layer_stack import GDSLayerObject
from fdtdx.objects.static_material.polygon import ExtrudedPolygon
from fdtdx.objects.static_material.sphere import Sphere
from fdtdx.objects.static_material.static import SimulationVolume, UniformMaterialObject

EPS_CORE = 12.1104
EPS_BG = 2.085
EPS_MID = 0.5 * (EPS_CORE + EPS_BG)

_COUNTER = [0]


def _tag() -> str:
    _COUNTER[0] += 1
    return f"yr{_COUNTER[0]}"


def _config(d: float, sampling: str, **kwargs) -> SimulationConfig:
    return SimulationConfig(time=1e-15, grid=UniformGrid(spacing=d), material_sampling=sampling, **kwargs)


def _volume(shape: tuple[int, int, int], name: str, permittivity: float = EPS_BG) -> SimulationVolume:
    return SimulationVolume(
        partial_grid_shape=shape,
        material=Material(permittivity=permittivity),
        name=name,
    )


def _rectangle(width: float, height: float) -> np.ndarray:
    return np.array(
        [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
        ]
    )


def _core_mask(arrays, component: int) -> np.ndarray:
    """Boolean core occupancy of one permittivity component, as a 3D array."""
    eps = 1.0 / np.asarray(arrays.inv_permittivities)
    return eps[component] > EPS_MID


def _num_components(arrays) -> int:
    return int(arrays.inv_permittivities.shape[0])


# ---------------------------------------------------------------------------
# The lattices themselves
# ---------------------------------------------------------------------------


def test_yee_lattice_positions_match_the_component_offsets():
    """Each component's lattice is edges on the axes it is not offset along, centres on the one it is."""
    grid = UniformGrid(spacing=40e-9).resolve((10, 10, 10))
    edges = [np.asarray(grid.edges(axis), dtype=float) for axis in range(3)]
    centers = [0.5 * (e[:-1] + e[1:]) for e in edges]
    nodes = [e[:-1] for e in edges]

    expected_E = {
        0: (centers[0], nodes[1], nodes[2]),
        1: (nodes[0], centers[1], nodes[2]),
        2: (nodes[0], nodes[1], centers[2]),
    }
    expected_H = {
        0: (nodes[0], centers[1], centers[2]),
        1: (centers[0], nodes[1], centers[2]),
        2: (centers[0], centers[1], nodes[2]),
    }
    for component, expected in expected_E.items():
        got = yee_lattice_coordinates(grid, "E", component)
        for axis in range(3):
            np.testing.assert_array_equal(got[axis], expected[axis])
    for component, expected in expected_H.items():
        got = yee_lattice_coordinates(grid, "H", component)
        for axis in range(3):
            np.testing.assert_array_equal(got[axis], expected[axis])


def test_per_component_lattices_see_an_interface_at_different_indices():
    """A material face inside a cell lands on different indices for E_x than for E_y and E_z.

    This is the assertion the whole stage exists for: today one cell-centre mask feeds all three
    components, so E_y and E_z carry their x-interface half a cell away from where the update reads
    them.
    """
    d = 40e-9
    name = _tag()
    volume = _volume((20, 6, 6), f"vol_{name}")
    slab_length = 6 * d
    # Offset the slab by 0.3 of a cell, so its lower face sits in the lower half of a cell, where
    # the centre lattice and the node lattice cross it in different cells.
    slab = UniformMaterialObject(
        material=Material(permittivity=EPS_CORE),
        partial_real_shape=(slab_length, None, None),
        partial_real_position=(0.3 * d, 0.0, 0.0),
        name=f"slab_{name}",
        placement_order=1,
    )
    constraints = [
        slab.extend_to(None, axis=1, direction="+"),
        slab.extend_to(None, axis=1, direction="-"),
        slab.extend_to(None, axis=2, direction="+"),
        slab.extend_to(None, axis=2, direction="-"),
    ]
    objects, arrays, _, config, _ = fdtdx.place_objects([volume, slab], _config(d, "yee"), constraints)

    grid = config.resolved_grid
    edges_x = np.asarray(grid.edges(0), dtype=float)
    centers_x = 0.5 * (edges_x[:-1] + edges_x[1:])
    face_x = next(o for o in objects.objects if o.name == slab.name).metric_bounds[0][0]

    core = [_core_mask(arrays, c) for c in range(3)]
    first_ex = int(np.argmax(core[0][:, 0, 0]))
    first_ey = int(np.argmax(core[1][:, 0, 0]))
    first_ez = int(np.argmax(core[2][:, 0, 0]))

    assert first_ex == int(np.searchsorted(centers_x, face_x))
    assert first_ey == int(np.searchsorted(edges_x[:-1], face_x))
    assert first_ez == first_ey
    assert first_ex != first_ey, "a face inside a cell must land on different indices per component"


# ---------------------------------------------------------------------------
# Areas of continuous shapes
# ---------------------------------------------------------------------------


def _shape_area_sweep(kind: str, d: float, sampling: str, offsets: np.ndarray) -> list[float]:
    """Rasterised area of one shape, once per sub-cell offset, averaged over the components."""
    areas = []
    for offset in offsets:
        name = _tag()
        n = round(1.6e-6 / d)
        volume = _volume((n, n, 3), f"vol_{name}")
        objects = [volume]
        shift = (float(offset) * d, float(offset) * d, 0.0)
        if kind == "rectangle":
            objects.append(
                ExtrudedPolygon(
                    axis=2,
                    vertices=_rectangle(500e-9, 340e-9),
                    material_name="core",
                    materials={"core": Material(permittivity=EPS_CORE)},
                    partial_grid_shape=(None, None, 3),
                    partial_real_position=shift,
                    placement_order=1,
                    name=f"rect_{name}",
                )
            )
        else:
            objects.append(
                Cylinder(
                    axis=2,
                    radius=400e-9,
                    material_name="core",
                    materials={"core": Material(permittivity=EPS_CORE)},
                    partial_grid_shape=(None, None, 3),
                    partial_real_position=shift,
                    placement_order=1,
                    name=f"disk_{name}",
                )
            )
            if kind == "annulus":
                objects.append(
                    Cylinder(
                        axis=2,
                        radius=240e-9,
                        material_name="bg",
                        materials={"bg": Material(permittivity=EPS_BG)},
                        partial_grid_shape=(None, None, 3),
                        partial_real_position=shift,
                        placement_order=2,
                        name=f"hole_{name}",
                    )
                )
        _, arrays, _, _, _ = fdtdx.place_objects(objects, _config(d, sampling), [])
        per_component = [float(_core_mask(arrays, c)[:, :, 1].sum()) * d * d for c in range(_num_components(arrays))]
        areas.append(float(np.mean(per_component)))
    return areas


ANALYTIC_AREA = {
    "rectangle": 500e-9 * 340e-9,
    "disk": np.pi * 400e-9**2,
    "annulus": np.pi * (400e-9**2 - 240e-9**2),
}
PERIMETER = {
    "rectangle": 2 * (500e-9 + 340e-9),
    "disk": 2 * np.pi * 400e-9,
    "annulus": 2 * np.pi * (400e-9 + 240e-9),
}


@pytest.mark.parametrize("kind", ["rectangle", "disk", "annulus"])
def test_area_error_scales_with_the_cell_size(kind):
    """Point-sampled area converges like the cell size, at 10 random sub-cell offsets.

    Point sampling a smooth shape misses or gains a band roughly one cell wide along its boundary,
    so the area error is bounded by the perimeter times the cell size and halves when the cell does.
    """
    rng = np.random.default_rng(12345)
    offsets = rng.uniform(0.05, 0.95, size=10)
    errors = {}
    for d in (40e-9, 20e-9):
        areas = np.asarray(_shape_area_sweep(kind, d, "yee", offsets))
        bound = 1.2 * PERIMETER[kind] * d
        worst = float(np.max(np.abs(areas - ANALYTIC_AREA[kind])))
        assert worst < bound, f"{kind} at {d * 1e9:.0f} nm: worst area error {worst:.3e} exceeds {bound:.3e}"
        errors[d] = float(np.mean(np.abs(areas - ANALYTIC_AREA[kind])))
    assert errors[20e-9] < 0.75 * errors[40e-9], f"{kind}: mean area error did not fall with the cell size: {errors}"


def test_box_mode_clips_the_rectangle_and_yee_mode_does_not():
    """The 500 x 340 nm rectangle is rasterised as 480 x 320 nm by the box path, at every offset."""
    rng = np.random.default_rng(7)
    offsets = rng.uniform(0.05, 0.95, size=10)
    d = 40e-9
    box_areas = np.asarray(_shape_area_sweep("rectangle", d, "box", offsets))
    yee_areas = np.asarray(_shape_area_sweep("rectangle", d, "yee", offsets))
    analytic = ANALYTIC_AREA["rectangle"]

    # The box path freezes the object onto the grid: the same clipped area at every offset.
    assert np.ptp(box_areas) == 0.0
    assert box_areas[0] == pytest.approx(480e-9 * 320e-9, rel=1e-6)
    # The yee path keeps the requested extent; its offset average is unbiased.
    assert abs(float(np.mean(yee_areas)) - analytic) < abs(box_areas[0] - analytic)
    assert abs(float(np.mean(yee_areas)) - analytic) < 0.05 * analytic


# ---------------------------------------------------------------------------
# Priority, metric shadow, and the guards
# ---------------------------------------------------------------------------


def _two_box_scene(order_a: int, order_b: int, sampling: str):
    name = _tag()
    d = 40e-9
    volume = _volume((10, 10, 4), f"vol_{name}")
    common = dict(
        partial_real_shape=(200e-9, 200e-9, 160e-9),
        partial_real_position=(0.0, 0.0, 0.0),
    )
    first = UniformMaterialObject(
        material=Material(permittivity=EPS_CORE), name=f"a_{name}", placement_order=order_a, **common
    )
    second = UniformMaterialObject(
        material=Material(permittivity=EPS_BG), name=f"b_{name}", placement_order=order_b, **common
    )
    _, arrays, _, _, _ = fdtdx.place_objects([volume, first, second], _config(d, sampling), [])
    return arrays


def test_priority_is_the_write_order():
    """Where two objects overlap, the one written later (higher placement_order) wins."""
    carved = _two_box_scene(order_a=1, order_b=2, sampling="yee")
    filled = _two_box_scene(order_a=2, order_b=1, sampling="yee")
    assert not _core_mask(carved, 0).any(), "the later low-index object must carve the earlier one"
    assert _core_mask(filled, 0).any(), "swapping the order must swap which material survives"


def test_metric_shadow_records_the_request_and_the_report_records_both():
    """A 500 nm object on a 40 nm grid keeps its 500 nm metric extent beside a 480 nm placed box."""
    d = 40e-9
    name = _tag()
    volume = _volume((20, 20, 4), f"vol_{name}")
    core = ExtrudedPolygon(
        axis=2,
        vertices=_rectangle(500e-9, 340e-9),
        material_name="core",
        materials={"core": Material(permittivity=EPS_CORE)},
        partial_grid_shape=(None, None, 4),
        partial_real_position=(0.0, 0.0, 0.0),
        name=f"rect_{name}",
    )
    objects, _, _, config, info = fdtdx.place_objects([volume, core], _config(d, "box"), [])
    placed = next(o for o in objects.objects if o.name == core.name)
    grid = config.resolved_grid

    assert placed.metric_extent[0] == pytest.approx(500e-9)
    assert grid.axis_extent(0, placed.grid_slice_tuple[0]) == pytest.approx(480e-9, rel=1e-5)

    rows = [r for r in info["placement_report"] if r["name"] == core.name and r["axis"] == 0]
    assert len(rows) == 1
    assert rows[0]["requested_size"] == pytest.approx(500e-9)
    assert rows[0]["realised_box_size"] == pytest.approx(480e-9, rel=1e-5)
    assert rows[0]["size_source"] == "partial_real_shape"


def test_the_logged_placement_report_only_lists_objects_whose_extent_moved():
    """The INFO table is the exceptions, not an inventory: an exactly-placed object is not in it."""
    from fdtdx.fdtd.metric_shadow import format_placement_report

    d = 40e-9
    name = _tag()
    volume = _volume((20, 20, 4), f"vol_{name}")
    exact = UniformMaterialObject(
        material=Material(permittivity=EPS_CORE),
        partial_real_shape=(480e-9, 480e-9, 160e-9),  # 12 x 12 x 4 whole cells
        partial_real_position=(0.0, 0.0, 0.0),
        placement_order=1,
        name=f"exact_{name}",
    )
    rounded = UniformMaterialObject(
        material=Material(permittivity=EPS_CORE),
        partial_real_shape=(500e-9, 480e-9, 160e-9),  # 12.5 cells on x -> rounded
        partial_real_position=(0.0, 0.0, 0.0),
        placement_order=2,
        name=f"rounded_{name}",
    )
    _, _, _, _, info = fdtdx.place_objects([volume, exact, rounded], _config(d, "yee"), [])

    rows = info["placement_report"]
    # Every object and axis is in the machine-readable rows...
    assert {r["name"] for r in rows} >= {volume.name, exact.name, rounded.name}
    # ...but only the object whose extent actually moved reaches the logged table.
    table = format_placement_report(rows)
    assert rounded.name in table
    assert exact.name not in table
    assert volume.name not in table


def test_metric_shadow_follows_a_position_constraint_in_metres():
    """A layer placed by a metric margin sits at that exact metric coordinate, not at a cell edge."""
    d = 40e-9
    z_base = 130e-9
    name = _tag()
    volume = _volume((6, 6, 20), f"vol_{name}")
    layer = UniformMaterialObject(
        material=Material(permittivity=EPS_CORE),
        partial_real_shape=(None, None, 220e-9),
        name=f"layer_{name}",
        placement_order=1,
    )
    constraints = [
        layer.place_relative_to(volume, axes=(2,), own_positions=(-1,), other_positions=(-1,), margins=(z_base,)),
        layer.extend_to(None, axis=0, direction="+"),
        layer.extend_to(None, axis=0, direction="-"),
        layer.extend_to(None, axis=1, direction="+"),
        layer.extend_to(None, axis=1, direction="-"),
    ]
    objects, _, _, config, _ = fdtdx.place_objects([volume, layer], _config(d, "box"), constraints)
    placed = next(o for o in objects.objects if o.name == layer.name)
    edges_z = np.asarray(config.resolved_grid.edges(2), dtype=float)

    assert placed.metric_bounds[2][0] == pytest.approx(edges_z[0] + z_base)
    assert placed.metric_extent[2] == pytest.approx(220e-9)


def test_box_mode_arrays_match_a_direct_mask_reconstruction():
    """The default path is untouched: its arrays still equal the per-object voxel-mask writes."""
    d = 40e-9
    name = _tag()
    volume = _volume((20, 20, 4), f"vol_{name}")
    disk = Cylinder(
        axis=2,
        radius=300e-9,
        material_name="core",
        materials={"core": Material(permittivity=EPS_CORE)},
        partial_grid_shape=(None, None, 4),
        partial_real_position=(60e-9, -20e-9, 0.0),
        placement_order=1,
        name=f"disk_{name}",
    )
    objects, arrays, _, _, _ = fdtdx.place_objects([volume, disk], _config(d, "box"), [])

    expected = np.full((20, 20, 4), EPS_BG, dtype=np.float64)
    placed_disk = next(o for o in objects.objects if o.name == disk.name)
    mask = np.asarray(placed_disk.get_voxel_mask_for_shape(), dtype=bool)
    expected[placed_disk.grid_slice] = np.where(mask, EPS_CORE, expected[placed_disk.grid_slice])

    assert _num_components(arrays) == 1
    got = 1.0 / np.asarray(arrays.inv_permittivities)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_yee_mode_forces_the_diagonal_component_tiers():
    """Per-component sampling cannot be represented by a 1-component array, so the tiers widen."""
    d = 40e-9
    name = _tag()
    volume = _volume((10, 10, 4), f"vol_{name}")
    lossy = UniformMaterialObject(
        material=Material(permittivity=EPS_CORE, electric_conductivity=3.0),
        partial_real_shape=(220e-9, 220e-9, 160e-9),
        partial_real_position=(0.0, 0.0, 0.0),
        placement_order=1,
        name=f"lossy_{name}",
    )
    _, box_arrays, _, _, _ = fdtdx.place_objects([volume, lossy], _config(d, "box"), [])
    assert box_arrays.inv_permittivities.shape[0] == 1
    assert box_arrays.electric_conductivity.shape[0] == 1

    name = _tag()
    volume = _volume((10, 10, 4), f"vol_{name}")
    lossy = UniformMaterialObject(
        material=Material(permittivity=EPS_CORE, electric_conductivity=3.0),
        partial_real_shape=(220e-9, 220e-9, 160e-9),
        partial_real_position=(0.0, 0.0, 0.0),
        placement_order=1,
        name=f"lossy_{name}",
    )
    _, yee_arrays, _, _, info = fdtdx.place_objects(
        [volume, lossy], _config(d, "yee", yee_sampling_diagnostics=True), []
    )
    assert yee_arrays.inv_permittivities.shape[0] == 3
    assert yee_arrays.electric_conductivity.shape[0] == 3
    assert info["yee_sampling_difference"]["num_differing_E"] > 0


def test_yee_mode_rejects_symmetry():
    name = _tag()
    volume = _volume((10, 10, 4), f"vol_{name}")
    with pytest.raises(NotImplementedError, match="material_sampling='yee'"):
        fdtdx.place_objects([volume], _config(40e-9, "yee", symmetry=(1, 0, 0)), [])


def test_yee_mode_rejects_subpixel_smoothing():
    name = _tag()
    volume = _volume((10, 10, 4), f"vol_{name}")
    disk = Cylinder(
        axis=2,
        radius=200e-9,
        material_name="core",
        materials={"core": Material(permittivity=EPS_CORE)},
        partial_grid_shape=(None, None, 4),
        partial_real_position=(0.0, 0.0, 0.0),
        subpixel_smoothing=True,
        placement_order=1,
        name=f"disk_{name}",
    )
    with pytest.raises(NotImplementedError, match="subpixel_smoothing"):
        fdtdx.place_objects([volume, disk], _config(40e-9, "yee"), [])


def test_material_sampling_is_validated():
    with pytest.raises(ValueError, match="material_sampling"):
        SimulationConfig(time=1e-15, grid=UniformGrid(spacing=40e-9), material_sampling="nearest")


def test_the_sampling_predicates_cover_both_yee_modes():
    """One helper, not a string literal: ``"yee_smooth"`` must answer yes to the sampling question.

    Comparing ``material_sampling == "yee"`` literally is the bug that took the ring case down when
    ``"yee_smooth"`` landed — the source gates silently excluded it. These are the predicates that
    replaced every such comparison.
    """
    expected = {
        "box": (False, False),
        "yee": (True, False),
        "yee_smooth": (True, True),
    }
    for mode, (sampling, smoothing) in expected.items():
        config = _config(40e-9, mode)
        assert config.uses_yee_material_sampling is sampling, mode
        assert config.uses_yee_smoothing is smoothing, mode


def test_the_box_difference_diagnostic_is_off_by_default(monkeypatch):
    """The second, box-mode rasterisation only runs when it is asked for."""
    from fdtdx.config import YEE_DIAGNOSTICS_ENV_VAR

    monkeypatch.delenv(YEE_DIAGNOSTICS_ENV_VAR, raising=False)
    name = _tag()
    volume = _volume((10, 10, 4), f"vol_{name}")
    core = UniformMaterialObject(
        material=Material(permittivity=EPS_CORE),
        partial_real_shape=(220e-9, 220e-9, 160e-9),
        partial_real_position=(0.0, 0.0, 0.0),
        placement_order=1,
        name=f"core_{name}",
    )
    _, _, _, _, info = fdtdx.place_objects([volume, core], _config(40e-9, "yee"), [])
    difference = info["yee_sampling_difference"]
    assert difference["box_difference_reported"] is False
    assert "num_differing_E" not in difference

    monkeypatch.setenv(YEE_DIAGNOSTICS_ENV_VAR, "1")
    name = _tag()
    volume = _volume((10, 10, 4), f"vol_{name}")
    core = UniformMaterialObject(
        material=Material(permittivity=EPS_CORE),
        partial_real_shape=(220e-9, 220e-9, 160e-9),
        partial_real_position=(0.0, 0.0, 0.0),
        placement_order=1,
        name=f"core_{name}",
    )
    _, _, _, _, info = fdtdx.place_objects([volume, core], _config(40e-9, "yee"), [])
    assert info["yee_sampling_difference"]["box_difference_reported"] is True
    assert info["yee_sampling_difference"]["num_differing_E"] > 0


# ---------------------------------------------------------------------------
# MRM-like scene: the 500 nm bus and the 180 nm bus-ring gap
# ---------------------------------------------------------------------------

BUS_WIDTH = 500e-9
DESIGN_GAP = 180e-9
RING_RADIUS = 1.0e-6
RING_WIDTH = 450e-9
MRM_DOMAIN_X = 3.0e-6
MRM_DOMAIN_Y = 3.2e-6
BUS_CENTER_Y = -1.0e-6
RING_CENTER_Y = BUS_CENTER_Y + BUS_WIDTH / 2 + DESIGN_GAP + RING_RADIUS


def _mrm_arrays(d: float, sampling: str, shift: float):
    """A straight bus next to a silicon ring (a disk with an oxide core carved out of it)."""
    name = _tag()
    nx, ny = round(MRM_DOMAIN_X / d), round(MRM_DOMAIN_Y / d)
    volume = _volume((nx, ny, 3), f"vol_{name}")
    core_material = {"core": Material(permittivity=EPS_CORE)}
    bus = ExtrudedPolygon(
        axis=2,
        vertices=_rectangle(MRM_DOMAIN_X, BUS_WIDTH),
        material_name="core",
        materials=core_material,
        partial_grid_shape=(None, None, 3),
        partial_real_position=(0.0, BUS_CENTER_Y + shift, 0.0),
        placement_order=1,
        name=f"bus_{name}",
    )
    ring = Cylinder(
        axis=2,
        radius=RING_RADIUS,
        material_name="core",
        materials=core_material,
        partial_grid_shape=(None, None, 3),
        partial_real_position=(0.0, RING_CENTER_Y + shift, 0.0),
        placement_order=2,
        name=f"ring_{name}",
    )
    hole = Cylinder(
        axis=2,
        radius=RING_RADIUS - RING_WIDTH,
        material_name="bg",
        materials={"bg": Material(permittivity=EPS_BG)},
        partial_grid_shape=(None, None, 3),
        partial_real_position=(0.0, RING_CENTER_Y + shift, 0.0),
        placement_order=3,
        name=f"hole_{name}",
    )
    return fdtdx.place_objects([volume, bus, ring, hole], _config(d, sampling), [])


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    """Start/stop index pairs of every contiguous True run."""
    padded = np.concatenate(([0], mask_1d.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[0::2], edges[1::2]))


def _bus_width_and_gap(arrays, config, component: int) -> tuple[float, float]:
    """Measure the bus width and the bus-ring gap on the lattice column at the ring's tangent."""
    grid = config.resolved_grid
    d = float(grid.cell_widths(1)[0])
    edges_x = np.asarray(grid.edges(0), dtype=float)
    lattice_x = 0.5 * (edges_x[:-1] + edges_x[1:]) if component == 0 else edges_x[:-1]
    column = int(np.argmin(np.abs(lattice_x - 0.5 * (edges_x[0] + edges_x[-1]))))
    occupancy = _core_mask(arrays, component)[column, :, 1]
    runs = _runs(occupancy)
    assert len(runs) >= 2, "expected a bus run and a ring run along the transverse line"
    return float(runs[0][1] - runs[0][0]) * d, float(runs[1][0] - runs[0][1]) * d


MRM_SHIFTS = (0.125, 0.375, 0.625, 0.875)


def _mrm_sweep(d: float, sampling: str) -> tuple[float, float, list[float]]:
    widths, gaps = [], []
    for shift in MRM_SHIFTS:
        _, arrays, _, config, _ = _mrm_arrays(d, sampling, shift * d)
        component = 2 if _num_components(arrays) == 3 else 0
        width, gap = _bus_width_and_gap(arrays, config, component)
        widths.append(width)
        gaps.append(gap)
    return float(np.mean(widths)), float(np.mean(gaps)), widths


@pytest.mark.parametrize("d", [40e-9, 32e-9, 25e-9, 20e-9])
def test_mrm_bus_width_and_gap_survive_the_grid(d):
    """The 500 nm bus and the 180 nm bus-ring gap, averaged over four sub-cell offsets.

    Measured on this machine (JAX CPU), effective bus width / bus-ring gap in nm:

    ======  ===============  ===============
    cell    box              yee
    ======  ===============  ===============
    40 nm   480.0 / 190.0    500.0 / 180.0
    32 nm   512.0 / 184.0    496.0 / 184.0
    25 nm   500.0 / 181.2    500.0 / 181.2
    20 nm   500.0 / 180.0    500.0 / 180.0
    ======  ===============  ===============

    The box path draws a different device at every resolution because the extent is rounded before
    the shape is drawn; the yee path draws the same 500 nm bus at all four.
    """
    yee_width, yee_gap, _ = _mrm_sweep(d, "yee")
    box_width, _box_gap, box_widths = _mrm_sweep(d, "box")

    assert abs(yee_width - BUS_WIDTH) <= 0.15 * d, f"yee width {yee_width * 1e9:.1f} nm at {d * 1e9:.0f} nm"
    assert abs(yee_gap - DESIGN_GAP) <= 0.25 * d, f"yee gap {yee_gap * 1e9:.1f} nm at {d * 1e9:.0f} nm"
    assert abs(yee_width - BUS_WIDTH) <= abs(box_width - BUS_WIDTH) + 1e-12
    # The box path is frozen onto the grid: the same rasterised width at every sub-cell offset.
    assert np.ptp(box_widths) == 0.0
    if d == 40e-9:
        assert box_width == pytest.approx(480e-9, rel=1e-5)
    if d == 32e-9:
        assert box_width == pytest.approx(512e-9, rel=1e-5)


# ---------------------------------------------------------------------------
# GDS layers and spheres
# ---------------------------------------------------------------------------


def _gds_layer_scene(d: float, sampling: str, z_base: float, thickness: float, sidewall_angle: float = 90.0):
    """One square GDS polygon extruded along z, placed by a metric z_base margin."""
    name = _tag()
    volume = _volume((20, 20, 20), f"vol_{name}")
    square = np.array([[-250e-9, -250e-9], [250e-9, -250e-9], [250e-9, 250e-9], [-250e-9, 250e-9]])
    layer = GDSLayerObject(
        polygons=[square],
        gds_center=(0.0, 0.0),
        material_name="core",
        materials={"core": Material(permittivity=EPS_CORE)},
        axis=2,
        thickness=thickness,
        sidewall_angle=sidewall_angle,
        reference_plane="bottom",
        partial_real_shape=(None, None, thickness),
        name=f"gds_{name}",
    )
    constraints = [
        layer.place_relative_to(volume, axes=(2,), own_positions=(-1.0,), other_positions=(-1.0,), margins=(z_base,)),
        layer.size_relative_to(volume, axes=(0, 1), other_axes=(0, 1)),
    ]
    return fdtdx.place_objects([volume, layer], _config(d, sampling), constraints)


def test_gds_layer_keeps_its_metric_thickness_and_base():
    """The layer spans [z_base, z_base + thickness) in metres, not the rounded cell count."""
    d = 40e-9
    z_base, thickness = 130e-9, 220e-9
    objects, arrays, _, config, _ = _gds_layer_scene(d, "yee", z_base, thickness)
    placed = next(o for o in objects.objects if o.name.startswith("gds_"))
    edges_z = np.asarray(config.resolved_grid.edges(2), dtype=float)

    assert placed.metric_bounds[2][0] == pytest.approx(edges_z[0] + z_base)
    assert placed.metric_extent[2] == pytest.approx(thickness)

    # E_x samples z at the nodes; every node inside [z_base, z_base + thickness) must be core.
    nodes_z = edges_z[:-1]
    expected = (nodes_z >= edges_z[0] + z_base) & (nodes_z < edges_z[0] + z_base + thickness)
    got = _core_mask(arrays, 0)[10, 10, :]
    np.testing.assert_array_equal(got, expected)


def test_gds_layer_contains_follows_the_sidewall_taper():
    """A sidewall angle below 90 degrees narrows the footprint with height, continuously."""
    d = 40e-9
    z_base, thickness = 130e-9, 400e-9
    objects, _, _, _, _ = _gds_layer_scene(d, "yee", z_base, thickness, sidewall_angle=75.0)
    layer = next(o for o in objects.objects if o.name.startswith("gds_"))
    center = layer.metric_center
    z_lower = layer.metric_bounds[2][0]

    line = np.linspace(-400e-9, 400e-9, 801)

    def half_width(height: float) -> float:
        points = np.stack([center[0] + line, np.full_like(line, center[1]), np.full_like(line, height)], axis=-1)
        inside = layer.contains(points)
        return float(line[inside].max())

    tan = float(np.tan(np.deg2rad(90.0 - 75.0)))
    bottom = half_width(z_lower + 0.05 * thickness)
    top = half_width(z_lower + 0.95 * thickness)
    # offset(z) = (z - z_base) * tan(90deg - angle), measured from the bottom reference face.
    assert bottom == pytest.approx(250e-9 - 0.05 * thickness * tan, abs=2e-9)
    assert top == pytest.approx(250e-9 - 0.95 * thickness * tan, abs=2e-9)
    assert top < bottom - 50e-9, "a 75 degree sidewall must erode the footprint towards the top"


def test_sphere_contains_matches_the_ellipsoid_equation():
    """The continuous ellipsoid test agrees with the analytic form at the object's metric centre."""
    d = 40e-9
    name = _tag()
    volume = _volume((20, 20, 20), f"vol_{name}")
    sphere = Sphere(
        radius=200e-9,
        radius_z=120e-9,
        material_name="core",
        materials={"core": Material(permittivity=EPS_CORE)},
        partial_real_position=(0.0, 0.0, 0.0),
        placement_order=1,
        name=f"sph_{name}",
    )
    objects, _, _, _, _ = fdtdx.place_objects([volume, sphere], _config(d, "yee"), [])
    placed = next(o for o in objects.objects if o.name == sphere.name)
    center = np.asarray(placed.metric_center)

    rng = np.random.default_rng(3)
    points = center + rng.uniform(-300e-9, 300e-9, size=(500, 3))
    radii = np.array([200e-9, 200e-9, 120e-9])
    expected = (((points - center) / radii) ** 2).sum(axis=-1) < 1.0
    np.testing.assert_array_equal(placed.contains(points), expected)


# ---------------------------------------------------------------------------
# Ties: faces exactly on lattice points
# ---------------------------------------------------------------------------


def test_faces_on_lattice_points_are_half_open_for_every_component():
    """A 500 nm square on a 25 nm grid, one cell thick, has faces exactly on lattice points.

    Every component must count exactly 20 samples across the square, and the components sampled on
    the z edge plane (E_x, E_y) must see the one-cell-thick object at all, independent of how the
    float32 grid edges round against the float64 metric shadow (the 2-D ring case lost its whole
    waveguide at 25 nm before the tie rule).
    """
    d = 25e-9
    name = _tag()
    volume = _volume((40, 40, 1), f"vol_{name}")
    core = ExtrudedPolygon(
        axis=2,
        vertices=_rectangle(500e-9, 500e-9),
        material_name="core",
        materials={"core": Material(permittivity=EPS_CORE)},
        partial_real_position=(0.0, 0.0, 0.0),
        name=f"sq_{name}",
    )
    object.__setattr__(core, "partial_real_shape", (*core.partial_real_shape[:2], d))
    _, arrays, _, _, _ = fdtdx.place_objects([volume, core], _config(d, "yee"), [])
    assert _num_components(arrays) == 3
    for component in range(3):
        mask = _core_mask(arrays, component)
        assert mask.any(), f"component {component} does not see the one-cell-thick object"
        along_x = int(mask[:, mask.shape[1] // 2, 0].sum())
        along_y = int(mask[mask.shape[0] // 2, :, 0].sum())
        assert along_x == 20, f"component {component}: {along_x} samples across x, expected 20"
        assert along_y == 20, f"component {component}: {along_y} samples across y, expected 20"
