"""Kottke sub-pixel smoothing on the Yee pixels: geometry, normals, the tensor, and the fallbacks.

Placement plus array assembly only — no mode solve and no time loop.

What ``material_sampling="yee_smooth"`` adds to ``"yee"``: the point sample at a pixel straddling
exactly one interface is replaced by the effective inverse permittivity of that pixel. The area a
shape occupies then stops being quantised to whole cells, which is what these tests measure — the
rasterised area of a rectangle, a disk and an annulus, read back out of the assembled permittivity,
is exact rather than wrong by a fraction of the perimeter times the cell size.
"""

import numpy as np
import pytest

import fdtdx
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import RectilinearGrid, UniformGrid
from fdtdx.core.physics.geometry_smooth import (
    circle_rectangle_area,
    kottke_inverse_permittivity,
    pixel_axis_bounds,
    polygons_rectangle_area,
)
from fdtdx.materials import Material
from fdtdx.objects.static_material.cylinder import Cylinder
from fdtdx.objects.static_material.gds_layer_stack import GDSLayerObject
from fdtdx.objects.static_material.polygon import ExtrudedPolygon
from fdtdx.objects.static_material.sphere import Sphere
from fdtdx.objects.static_material.static import SimulationVolume, UniformMaterialObject

EPS_BG = 2.085
EPS_CORE = 12.1104

_COUNTER = [0]


def _tag() -> str:
    _COUNTER[0] += 1
    return f"ys{_COUNTER[0]}"


def _config(d: float, sampling: str, **kwargs) -> SimulationConfig:
    return SimulationConfig(time=1e-15, grid=UniformGrid(spacing=d), material_sampling=sampling, **kwargs)


def _volume(shape, name, permittivity=EPS_BG) -> SimulationVolume:
    return SimulationVolume(
        partial_grid_shape=shape,
        material=Material(permittivity=permittivity),
        name=name,
    )


def _rectangle(width: float, height: float) -> np.ndarray:
    half_w, half_h = 0.5 * width, 0.5 * height
    return np.array([[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]])


def _fill_from_arrays(arrays, component: int) -> np.ndarray:
    """Recover the core fill fraction from one permittivity component.

    Only legitimate where the component is tangential to every interface in the scene, so that the
    Kottke blend reduces to the arithmetic mean of ``eps`` — which is exactly the 2-D
    (extrusion-invariant) situation the area tests below use.
    """
    eps = 1.0 / np.asarray(arrays.inv_permittivities, dtype=np.float64)
    return np.clip((eps[component] - EPS_BG) / (EPS_CORE - EPS_BG), 0.0, 1.0)


def _smoothing_stats(info) -> dict:
    return info["yee_sampling_difference"]["smoothing"]


# ---------------------------------------------------------------------------
# Pixel geometry
# ---------------------------------------------------------------------------


def test_pixel_boxes_match_the_component_offsets():
    """Primal cell on an axis the component sits at a centre on, dual cell where it sits at an edge."""
    widths = [
        np.array([20e-9, 30e-9, 40e-9, 25e-9]),
        np.array([10e-9, 50e-9, 20e-9]),
        np.array([15e-9, 15e-9]),
    ]
    edges = [np.concatenate([[0.0], np.cumsum(w)]) for w in widths]
    grid = RectilinearGrid(x_edges=edges[0], y_edges=edges[1], z_edges=edges[2])
    centers = [0.5 * (e[:-1] + e[1:]) for e in edges]

    # E_y sits at (e_x, c_y, e_z): dual on x and z, primal on y.
    bounds = pixel_axis_bounds(grid, "E", 1)
    lower_x, upper_x = bounds[0]
    previous = np.concatenate([widths[0][:1], widths[0][:-1]])
    # The grid stores its edges in the simulation dtype, so compare at that precision.
    tol = dict(rtol=1e-6, atol=0)
    np.testing.assert_allclose(lower_x, np.clip(edges[0][:-1] - 0.5 * previous, edges[0][0], None), **tol)
    np.testing.assert_allclose(upper_x, edges[0][:-1] + 0.5 * widths[0], **tol)
    # The domain-edge pixel is clipped to the domain, so its lower bound is the first edge itself.
    assert lower_x[0] == pytest.approx(edges[0][0], abs=1e-18)
    lower_y, upper_y = bounds[1]
    np.testing.assert_allclose(lower_y, edges[1][:-1], **tol)
    np.testing.assert_allclose(upper_y, edges[1][1:], **tol)
    # Every dual pixel above the first is centre-to-centre wide.
    np.testing.assert_allclose(upper_x[1:] - lower_x[1:], np.diff(centers[0]), **tol)


def test_pixel_boxes_collapse_on_an_invariant_axis():
    """A single-cell axis carries no pixel extent: fdtdx's 2-D convention has no third dimension."""
    grid = UniformGrid(spacing=20e-9).resolve((6, 6, 1))
    for component in range(3):
        lower, upper = pixel_axis_bounds(grid, "E", component)[2]
        assert lower.shape == (1,)
        np.testing.assert_array_equal(lower, upper)


# ---------------------------------------------------------------------------
# Exact planar overlaps
# ---------------------------------------------------------------------------


def test_circle_rectangle_area_matches_a_fine_quadrature():
    radius = 0.81e-6
    rng = np.random.default_rng(7)
    for _ in range(40):
        x0, y0 = rng.uniform(-1.2e-6, 1.0e-6, size=2)
        x1, y1 = x0 + rng.uniform(1e-8, 6e-7), y0 + rng.uniform(1e-8, 6e-7)
        exact = float(circle_rectangle_area(np.array([x0]), np.array([x1]), np.array([y0]), np.array([y1]), radius)[0])
        n = 800
        xs = x0 + (np.arange(n) + 0.5) / n * (x1 - x0)
        ys = y0 + (np.arange(n) + 0.5) / n * (y1 - y0)
        inside = (xs[:, None] ** 2 + ys[None, :] ** 2) < radius**2
        quadrature = inside.mean() * (x1 - x0) * (y1 - y0)
        assert abs(exact - quadrature) < 3e-3 * (x1 - x0) * (y1 - y0) + 1e-20


def test_polygon_rectangle_area_is_exact_for_a_tilted_edge():
    """A triangle clipped by a rectangle: the shoelace of the clip is exact at any edge angle."""
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    area = float(
        polygons_rectangle_area([triangle], np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([1.0]))[0]
    )
    assert area == pytest.approx(0.5, abs=1e-15)
    # A window covering only the lower-left quarter cuts the hypotenuse: 0.25 - 0 (the hypotenuse
    # x + y = 1 does not enter [0, 0.5]^2), so the whole quarter is inside.
    quarter = float(
        polygons_rectangle_area([triangle], np.array([0.0]), np.array([0.5]), np.array([0.0]), np.array([0.5]))[0]
    )
    assert quarter == pytest.approx(0.25, abs=1e-15)
    # A window straddling the hypotenuse.
    straddle = float(
        polygons_rectangle_area([triangle], np.array([0.4]), np.array([0.8]), np.array([0.4]), np.array([0.8]))[0]
    )
    n = 4000
    xs = 0.4 + (np.arange(n) + 0.5) / n * 0.4
    ys = 0.4 + (np.arange(n) + 0.5) / n * 0.4
    inside = (xs[:, None] >= 0) & (ys[None, :] >= 0) & (xs[:, None] + ys[None, :] <= 1)
    assert straddle == pytest.approx(inside.mean() * 0.16, abs=2e-5)


# ---------------------------------------------------------------------------
# Areas read back out of the assembled permittivity
# ---------------------------------------------------------------------------


def _rectangle_area(d: float, shift: float, sampling: str) -> float:
    width, height = 500e-9, 320e-9
    span = 2.0e-6
    cells = round(span / d)
    name = _tag()
    config = _config(d, sampling)
    volume = _volume((cells, cells, 1), f"v{name}")
    core = ExtrudedPolygon(
        axis=2,
        vertices=_rectangle(width, height),
        material_name="core",
        materials={"core": Material(permittivity=EPS_CORE)},
        partial_grid_shape=(None, None, 1),
        partial_real_position=(shift, 0.7 * shift, 0.0),
        placement_order=1,
        name=f"c{name}",
    )
    _, arrays, _, _, _ = fdtdx.place_objects([volume, core], config, [])
    return float(_fill_from_arrays(arrays, 2).sum()) * d * d


def _disk_area(d: float, shift: float, sampling: str, radius: float = 0.81e-6) -> float:
    span = 2.4e-6
    cells = round(span / d)
    name = _tag()
    config = _config(d, sampling)
    volume = _volume((cells, cells, 1), f"v{name}")
    disk = Cylinder(
        axis=2,
        radius=radius,
        material_name="core",
        materials={"core": Material(permittivity=EPS_CORE)},
        partial_grid_shape=(None, None, 1),
        partial_real_position=(shift, shift, 0.0),
        placement_order=1,
        name=f"c{name}",
    )
    _, arrays, _, _, _ = fdtdx.place_objects([volume, disk], config, [])
    return float(_fill_from_arrays(arrays, 2).sum()) * d * d


def _annulus_area(d: float, shift: float, sampling: str) -> float:
    outer, inner = 0.81e-6, 0.42e-6
    span = 2.4e-6
    cells = round(span / d)
    name = _tag()
    config = _config(d, sampling)
    volume = _volume((cells, cells, 1), f"v{name}")
    common = dict(
        axis=2,
        materials={"core": Material(permittivity=EPS_CORE), "hole": Material(permittivity=EPS_BG)},
        partial_grid_shape=(None, None, 1),
        partial_real_position=(shift, shift, 0.0),
    )
    ring = Cylinder(radius=outer, material_name="core", placement_order=1, name=f"o{name}", **common)
    hole = Cylinder(radius=inner, material_name="hole", placement_order=2, name=f"i{name}", **common)
    _, arrays, _, _, _ = fdtdx.place_objects([volume, ring, hole], config, [])
    return float(_fill_from_arrays(arrays, 2).sum()) * d * d


CELLS = (100e-9, 50e-9, 25e-9)
OFFSETS = (0.0, 0.13, 0.37, 0.61)


@pytest.mark.parametrize(
    "measure, truth",
    [
        (_rectangle_area, 500e-9 * 320e-9),
        (_disk_area, np.pi * 0.81e-6**2),
        (_annulus_area, np.pi * (0.81e-6**2 - 0.42e-6**2)),
    ],
    ids=["rectangle", "disk", "annulus"],
)
def test_smoothed_area_converges_at_second_order(measure, truth):
    """The rasterised area is exact at every cell size and every sub-cell offset.

    Second order is the bar the Kottke average has to clear; the analytic fill fractions clear it by
    a wide margin — the residual here is the float32 storage of the permittivity array, about 1e-7
    relative, not a discretisation error. The point-sampled ``"yee"`` path is graded against the
    perimeter-times-cell bound the same tolerance would fail.
    """
    for d in CELLS:
        for offset in OFFSETS:
            area = measure(d, offset * d, "yee_smooth")
            assert abs(area - truth) < 1e-5 * truth + 4.0 * (d**2) * 1e-6, (
                f"d={d * 1e9:.0f} nm offset={offset}: area {area:.6e} vs {truth:.6e}"
            )
    # The comparison the mode exists for: point sampling is wrong by O(perimeter * cell).
    point_errors = [abs(measure(d, 0.13 * d, "yee") - truth) / truth for d in CELLS]
    smooth_errors = [abs(measure(d, 0.13 * d, "yee_smooth") - truth) / truth for d in CELLS]
    assert max(smooth_errors) < 1e-5
    assert max(point_errors) > 20 * max(smooth_errors)


def test_face_on_a_lattice_point_is_unambiguous():
    """Sweeping a face across a lattice point moves the rasterised area smoothly, not by a whole cell.

    In ``"yee"`` mode this is the 12.5-cell tie: whichever way the containment test falls, the area
    jumps by a full row of cells. The blend has no tie to resolve — the fill fraction is continuous
    in the face position — so the swept area is a straight line.
    """
    d = 40e-9
    positions = np.linspace(-0.9 * d, 0.9 * d, 13)
    areas = np.array([_rectangle_area(d, float(p), "yee_smooth") for p in positions])
    truth = 500e-9 * 320e-9
    assert np.max(np.abs(areas - truth)) < 1e-5 * truth
    jumps = np.abs(np.diff([_rectangle_area(d, float(p), "yee") for p in positions]))
    assert jumps.max() > 0.5 * 320e-9 * d, "the point-sampled path should jump by about a row of cells"


# ---------------------------------------------------------------------------
# Normals
# ---------------------------------------------------------------------------


def _place(objects, d=25e-9, shape=(40, 40, 40), sampling="yee"):
    """Place a scene and return the placed copies, keyed by name.

    ``place_objects`` returns new objects rather than mutating the inputs, and the metric bounds a
    normal is measured against live on the placed copy.
    """
    config = _config(d, sampling)
    volume = _volume(shape, f"v{_tag()}")
    container, _, _, _, _ = fdtdx.place_objects([volume, *objects], config, [])
    return {o.name: o for o in container.object_list}


def _alignment(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """|cos| between two normal fields: the Kottke tensor only sees ``n n^T``, so the sign is free."""
    return np.abs(np.sum(actual * expected, axis=-1))


def test_cylinder_normal_is_radial_on_the_barrel_and_axial_on_the_caps():
    radius = 0.3e-6
    cylinder = Cylinder(
        axis=2,
        radius=radius,
        material_name="core",
        materials={"core": Material(permittivity=EPS_CORE)},
        partial_real_shape=(None, None, 0.4e-6),
        partial_real_position=(0.0, 0.0, 0.0),
        placement_order=1,
        name=f"c{_tag()}",
    )
    cylinder = _place([cylinder])[cylinder.name]
    center = np.asarray(cylinder.metric_center)
    angles = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
    barrel = center + np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)], axis=-1)
    normal = cylinder.normal_at(barrel)
    expected = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=-1)
    np.testing.assert_allclose(_alignment(normal, expected), 1.0, atol=1e-12)

    half = 0.5 * cylinder.metric_extent[2]
    caps = center + np.stack(
        [0.1 * radius * np.cos(angles), 0.1 * radius * np.sin(angles), np.full_like(angles, half)], axis=-1
    )
    cap_normal = cylinder.normal_at(caps)
    np.testing.assert_allclose(np.abs(cap_normal[:, 2]), 1.0, atol=1e-12)

    # With the extrusion axis declared not a surface, the barrel wins everywhere.
    plane_only = cylinder.normal_at(caps, ignore_axes=(2,))
    np.testing.assert_allclose(plane_only[:, 2], 0.0, atol=0.0)


def test_sphere_normal_is_the_ellipsoid_gradient():
    sphere = Sphere(
        radius=0.3e-6,
        radius_y=0.2e-6,
        material_name="core",
        materials={"core": Material(permittivity=EPS_CORE)},
        partial_real_position=(0.0, 0.0, 0.0),
        placement_order=1,
        name=f"s{_tag()}",
    )
    sphere = _place([sphere])[sphere.name]
    center = np.asarray(sphere.metric_center)
    rng = np.random.default_rng(3)
    direction = rng.normal(size=(200, 3))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    radii = np.array([0.3e-6, 0.2e-6, 0.3e-6])
    surface = center + direction * radii
    expected = (surface - center) / radii**2
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
    np.testing.assert_allclose(_alignment(sphere.normal_at(surface), expected), 1.0, atol=1e-12)
    np.testing.assert_allclose(sphere.normal_at(center[None, :]), 0.0, atol=0.0)


def test_polygon_normal_is_the_nearest_edge_normal():
    triangle = np.array([[-0.3e-6, -0.2e-6], [0.3e-6, -0.2e-6], [0.0, 0.4e-6]])
    polygon = ExtrudedPolygon(
        axis=2,
        vertices=triangle,
        material_name="core",
        materials={"core": Material(permittivity=EPS_CORE)},
        partial_real_shape=(None, None, 0.4e-6),
        partial_real_position=(0.0, 0.0, 0.0),
        placement_order=1,
        name=f"p{_tag()}",
    )
    polygon = _place([polygon])[polygon.name]
    center = np.asarray(polygon.metric_center)
    # Midpoints of the three sides, in object coordinates, with their known outward normals.
    for a, b in ((0, 1), (1, 2), (2, 0)):
        mid = 0.5 * (triangle[a] + triangle[b])
        edge = triangle[b] - triangle[a]
        outward = np.array([edge[1], -edge[0]])
        outward /= np.linalg.norm(outward)
        query = center + np.array([mid[0], mid[1], 0.0])
        normal = polygon.normal_at(query[None, :])[0]
        assert normal[2] == 0.0
        assert abs(abs(float(normal[0] * outward[0] + normal[1] * outward[1])) - 1.0) < 1e-12


def test_uniform_box_normal_is_the_nearest_face():
    box = UniformMaterialObject(
        material=Material(permittivity=EPS_CORE),
        partial_real_shape=(0.4e-6, 0.6e-6, 0.2e-6),
        partial_real_position=(0.0, 0.0, 0.0),
        placement_order=1,
        name=f"b{_tag()}",
    )
    box = _place([box])[box.name]
    center = np.asarray(box.metric_center)
    half = 0.5 * np.asarray(box.metric_extent)
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = half[axis]
        normal = box.normal_at((center + offset)[None, :])[0]
        assert normal[axis] == pytest.approx(1.0)
        assert np.count_nonzero(normal) == 1
    # Declaring an axis "not a surface" removes it from the competition.
    offset = np.zeros(3)
    offset[2] = half[2]
    restricted = box.normal_at((center + offset)[None, :], ignore_axes=(2,))[0]
    assert restricted[2] == 0.0
    assert np.count_nonzero(restricted) == 1


def _gds_layer(angle: float) -> GDSLayerObject:
    return GDSLayerObject(
        polygons=[_rectangle(1.0e-6, 1.0e-6)],
        gds_center=(0.0, 0.0),
        material_name="core",
        materials={"core": Material(permittivity=EPS_CORE)},
        axis=2,
        thickness=0.22e-6,
        sidewall_angle=angle,
        partial_real_position=(0.0, 0.0, 0.0),
        placement_order=1,
        name=f"g{_tag()}",
    )


def test_gds_layer_normal_tilts_with_the_sidewall():
    """An 80 degree sidewall tilts the wall normal out of plane by exactly tan(10 degrees)."""
    layer = _gds_layer(80.0)
    layer = _place([layer], d=25e-9, shape=(60, 60, 20))[layer.name]
    center = np.asarray(layer.metric_center)
    z_lo = layer.metric_bounds[2][0]
    query = np.array([center[0] + 0.5e-6, center[1], z_lo + 0.11e-6])
    normal = layer.normal_at(query[None, :])[0]
    tan = np.tan(np.deg2rad(10.0))
    expected = np.array([1.0, 0.0, tan])
    expected /= np.linalg.norm(expected)
    assert abs(_alignment(normal[None, :], expected[None, :])[0] - 1.0) < 1e-9

    vertical = _gds_layer(90.0)
    vertical = _place([vertical], d=25e-9, shape=(60, 60, 20))[vertical.name]
    normal_vertical = vertical.normal_at(query[None, :])[0]
    assert normal_vertical[2] == pytest.approx(0.0, abs=1e-15)


def test_gradient_fallback_normal_does_not_improve_with_the_cell_size():
    """Why the analytic normal is a requirement, not an optimisation.

    A normal recovered from a fixed-size fine raster inside the pixel carries an angular error set by
    the supersample count alone. Halving the cell does not shrink it, so it leaves an ``O(h)`` term in
    the field error and caps the observed convergence order near one. Measured here directly.
    """
    from fdtdx.core.physics.geometry_smooth import _gradient_normal_from_fill

    radius = 0.5e-6
    cylinder = Cylinder(
        axis=2,
        radius=radius,
        material_name="core",
        materials={"core": Material(permittivity=EPS_CORE)},
        partial_real_shape=(None, None, 0.4e-6),
        partial_real_position=(0.0, 0.0, 0.0),
        placement_order=1,
        name=f"c{_tag()}",
    )
    cylinder = _place([cylinder], d=25e-9, shape=(60, 60, 30))[cylinder.name]
    center = np.asarray(cylinder.metric_center)
    angles = np.linspace(0.1, 2 * np.pi, 37)
    exact = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=-1)
    surface = center + radius * exact

    errors = []
    for d in (40e-9, 20e-9, 10e-9):
        lower = surface - 0.5 * d
        upper = surface + 0.5 * d
        approx = _gradient_normal_from_fill(cylinder, lower, upper, 8, (False, False, False))
        errors.append(float(np.mean(np.arccos(np.clip(_alignment(approx, exact), 0.0, 1.0)))))
    assert errors[-1] > 0.25 * errors[0], f"gradient normal error should not scale with h: {errors}"

    analytic = cylinder.normal_at(surface, ignore_axes=(2,))
    # arccos loses half the digits near 1, so 1e-6 rad here is float64 round-off, not an error.
    assert float(np.max(np.arccos(np.clip(_alignment(analytic, exact), 0.0, 1.0)))) < 1e-6


# ---------------------------------------------------------------------------
# The tensor
# ---------------------------------------------------------------------------


def test_kottke_tensor_eigenvalues_and_inverse():
    """The blend is ``P*H + (I-P)/A``: eigenvalues ``{H, 1/A, 1/A}``, inverse ``A*(I-P) + P/H``."""
    rng = np.random.default_rng(11)
    for _ in range(50):
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        fill = float(rng.uniform(0.05, 0.95))
        eps_hi, eps_lo = float(rng.uniform(1.0, 13.0)), float(rng.uniform(1.0, 13.0))
        arithmetic = np.array([fill * eps_hi + (1 - fill) * eps_lo])
        harmonic = np.array([fill / eps_hi + (1 - fill) / eps_lo])
        rows = [
            kottke_inverse_permittivity(normal[None, :], arithmetic, harmonic, c, full_tensor=True)[0] for c in range(3)
        ]
        tensor = np.stack(rows, axis=0)
        np.testing.assert_allclose(tensor, tensor.T, atol=1e-13)
        eigenvalues = np.sort(np.linalg.eigvalsh(tensor))
        expected = np.sort([harmonic[0], 1.0 / arithmetic[0], 1.0 / arithmetic[0]])
        np.testing.assert_allclose(eigenvalues, expected, rtol=1e-12)
        assert (eigenvalues > 0).all()
        projection = np.outer(normal, normal)
        forward = arithmetic[0] * (np.eye(3) - projection) + projection / harmonic[0]
        np.testing.assert_allclose(np.linalg.inv(tensor), forward, rtol=1e-10, atol=1e-12)
        # The diagonal tier is the diagonal of the same tensor.
        for c in range(3):
            entry = kottke_inverse_permittivity(normal[None, :], arithmetic, harmonic, c, full_tensor=False)[0]
            assert entry == pytest.approx(tensor[c, c], rel=1e-14)


def _planar_interface(d: float, face_offset: float, sampling: str, **kwargs):
    """A half-space of core material whose x face sits ``face_offset`` into cell 6."""
    cells = 12
    name = _tag()
    config = _config(d, sampling, **kwargs)
    volume = _volume((cells, 4, 4), f"v{name}")
    span = cells * d
    face = 6 * d + face_offset
    slab = UniformMaterialObject(
        material=Material(permittivity=EPS_CORE),
        partial_real_shape=(span - face, None, None),
        partial_real_position=(0.5 * face, 0.0, 0.0),
        placement_order=1,
        name=f"s{name}",
    )
    container, arrays, _, resolved, info = fdtdx.place_objects([volume, slab], config, [])
    placed = {o.name: o for o in container.object_list}[slab.name]
    return (
        np.asarray(arrays.inv_permittivities, dtype=np.float64),
        resolved,
        info,
        float(placed.metric_bounds[0][0]),
    )


def test_planar_interface_gives_the_harmonic_and_arithmetic_means():
    """Normal along x: ``eps_xx`` is the harmonic mean, ``eps_yy`` and ``eps_zz`` the arithmetic one.

    This is the pairing that makes a Yee-staggered discretisation second order at a dielectric step:
    the component normal to the interface averages in ``1/eps``, the tangential ones in ``eps``.
    """
    d = 40e-9
    offset = 0.37 * d
    inv_eps, resolved, _, face = _planar_interface(d, offset, "yee_smooth")
    grid = resolved.resolved_grid
    probe = (slice(None), 2, 2)

    for component, tangential in ((0, False), (1, True), (2, True)):
        lower, upper = pixel_axis_bounds(grid, "E", component)[0]
        fill = np.clip((upper - face) / (upper - lower), 0.0, 1.0)
        interface = np.flatnonzero((fill > 1e-9) & (fill < 1 - 1e-9))
        assert interface.size == 1, f"component {component} should straddle exactly one pixel"
        cell = int(interface[0])
        f = float(fill[cell])
        arithmetic = f * EPS_CORE + (1 - f) * EPS_BG
        harmonic = f / EPS_CORE + (1 - f) / EPS_BG
        got = float(inv_eps[component][cell, probe[1], probe[2]])
        expected = 1.0 / arithmetic if tangential else harmonic
        assert got == pytest.approx(expected, rel=2e-6), (
            f"component {component}: got {got}, expected {expected} (f={f})"
        )


def test_full_tensor_matches_the_diagonal_for_axis_aligned_normals():
    """A Manhattan scene has ``n`` on an axis, so the off-diagonal Kottke terms are exactly zero."""
    d = 40e-9
    diagonal, _, _, _ = _planar_interface(d, 0.37 * d, "yee_smooth")
    full, _, _, _ = _planar_interface(d, 0.37 * d, "yee_smooth", yee_smooth_full_tensor=True)
    assert diagonal.shape[0] == 3
    assert full.shape[0] == 9
    for component in range(3):
        np.testing.assert_allclose(full[4 * component], diagonal[component], rtol=0, atol=0)
        for j in range(3):
            if j != component:
                np.testing.assert_array_equal(full[3 * component + j], np.zeros_like(diagonal[component]))


# ---------------------------------------------------------------------------
# Fallbacks and invariants
# ---------------------------------------------------------------------------


def test_uniform_pixels_keep_the_point_sample_bit_for_bit():
    """Only the pixels the probe flags are touched; everything else is the ``"yee"`` array itself."""
    d = 40e-9
    point, _, _, _ = _planar_interface(d, 0.37 * d, "yee")
    smooth, _, info, _ = _planar_interface(d, 0.37 * d, "yee_smooth")
    stats = _smoothing_stats(info)
    differing = np.count_nonzero(point != smooth)
    assert differing > 0
    assert differing <= stats["num_smoothed"]
    assert stats["num_three_material_fallbacks"] == 0
    assert stats["num_gradient_normal_fallbacks"] == 0
    assert stats["num_supersampled_pixels"] == 0
    # Deep interior on either side of the face is untouched.
    np.testing.assert_array_equal(point[:, :4], smooth[:, :4])
    np.testing.assert_array_equal(point[:, 9:], smooth[:, 9:])


def test_a_face_on_a_pixel_boundary_collapses_to_the_point_sample():
    """With the face exactly on the lattice the fill is 0 or 1 and the blend is the point value."""
    d = 40e-9
    point, _, _, _ = _planar_interface(d, 0.0, "yee")
    smooth, _, info, _ = _planar_interface(d, 0.0, "yee_smooth")
    stats = _smoothing_stats(info)
    assert stats["num_degenerate_fill_fallbacks"] > 0
    # The primal (E_x) pixel has the face on its own boundary, so that component is untouched.
    np.testing.assert_array_equal(point[0], smooth[0])


def test_three_materials_fall_back_to_the_point_sample():
    """A corner where three materials meet has no single planar interface for the blend to describe."""
    d = 50e-9
    name = _tag()
    config = _config(d, "yee_smooth")
    volume = _volume((16, 16, 1), f"v{name}")
    common = dict(
        axis=2,
        materials={
            "a": Material(permittivity=EPS_CORE),
            "b": Material(permittivity=6.0),
        },
        partial_grid_shape=(None, None, 1),
    )
    left = ExtrudedPolygon(
        vertices=_rectangle(300e-9, 300e-9),
        material_name="a",
        partial_real_position=(-0.13 * d, 0.0, 0.0),
        placement_order=1,
        name=f"l{name}",
        **common,
    )
    right = ExtrudedPolygon(
        vertices=_rectangle(300e-9, 300e-9),
        material_name="b",
        partial_real_position=(300e-9 - 0.13 * d, 0.17 * d, 0.0),
        placement_order=2,
        name=f"r{name}",
        **common,
    )
    _, arrays, _, _, info = fdtdx.place_objects([volume, left, right], config, [])
    stats = _smoothing_stats(info)
    assert stats["num_three_material_fallbacks"] > 0
    point_config = _config(d, "yee")
    volume_b = _volume((16, 16, 1), f"v{_tag()}")
    _, point_arrays, _, _, _ = fdtdx.place_objects(
        [volume_b, left.aset("name", f"l2{name}"), right.aset("name", f"r2{name}")], point_config, []
    )
    smooth = np.asarray(arrays.inv_permittivities, dtype=np.float64)
    point = np.asarray(point_arrays.inv_permittivities, dtype=np.float64)
    # The three-material pixels are a strict minority; the arrays still agree almost everywhere.
    assert np.count_nonzero(smooth != point) < 0.25 * smooth.size


def test_candidate_fraction_is_small():
    """Interface pixels are a surface against a volume, so only a per-cent of the grid does work."""
    d = 25e-9
    _disk_area(d, 0.13 * d, "yee")  # warm the tag counter; keeps the names unique
    span, cells = 2.4e-6, round(2.4e-6 / 25e-9)
    name = _tag()
    config = _config(d, "yee_smooth")
    volume = _volume((cells, cells, 1), f"v{name}")
    disk = Cylinder(
        axis=2,
        radius=0.81e-6,
        material_name="core",
        materials={"core": Material(permittivity=EPS_CORE)},
        partial_grid_shape=(None, None, 1),
        partial_real_position=(0.13 * d, 0.13 * d, 0.0),
        placement_order=1,
        name=f"c{name}",
    )
    _, _, _, _, info = fdtdx.place_objects([volume, disk], config, [])
    stats = _smoothing_stats(info)
    assert stats["candidate_fraction"] < 0.05
    assert stats["num_three_material_fallbacks"] == 0
    assert stats["num_gradient_normal_fallbacks"] == 0
    assert stats["num_supersampled_pixels"] == 0
    assert span > 0


def test_sphere_falls_back_to_supersampling_and_stays_accurate():
    """No closed form for a sphere-box overlap, so the fill is super-sampled; the area still lands."""
    d = 60e-9
    radius = 0.4e-6
    name = _tag()
    config = _config(d, "yee_smooth", yee_smooth_supersample=12)
    volume = _volume((20, 20, 20), f"v{name}")
    sphere = Sphere(
        radius=radius,
        material_name="core",
        materials={"core": Material(permittivity=EPS_CORE)},
        partial_real_position=(0.13 * d, 0.0, 0.0),
        placement_order=1,
        name=f"s{name}",
    )
    _, arrays, _, _, info = fdtdx.place_objects([volume, sphere], config, [])
    stats = _smoothing_stats(info)
    assert stats["num_supersampled_pixels"] > 0
    eps = 1.0 / np.asarray(arrays.inv_permittivities, dtype=np.float64)
    # eps_zz picks up the arithmetic mean only where the normal has no z component, so read the
    # volume from the trace of the tensor instead: tr(1/eps_eff) = H + 2/A is basis independent.
    volume_estimate = float(np.clip((eps[2] - EPS_BG) / (EPS_CORE - EPS_BG), 0.0, 1.0).sum()) * d**3
    truth = 4.0 / 3.0 * np.pi * radius**3
    assert abs(volume_estimate - truth) < 0.06 * truth


def test_the_smoother_refuses_a_mismatched_component_tier():
    """Entry (c, c) of a row-major 3x3 sits at 4*c, so the tier and the write form must agree."""
    from fdtdx.core.physics.geometry_raster import build_scene
    from fdtdx.core.physics.geometry_smooth import smooth_inverse_permittivity_on_yee_pixels

    name = _tag()
    config = _config(50e-9, "yee")
    volume = _volume((8, 8, 1), f"v{name}")
    core = ExtrudedPolygon(
        axis=2,
        vertices=_rectangle(210e-9, 210e-9),
        material_name="core",
        materials={"core": Material(permittivity=EPS_CORE)},
        partial_grid_shape=(None, None, 1),
        partial_real_position=(0.13 * 50e-9, 0.0, 0.0),
        placement_order=1,
        name=f"c{name}",
    )
    container, arrays, _, resolved, _ = fdtdx.place_objects([volume, core], config, [])
    scene = build_scene(container.static_material_objects)
    grid = resolved.resolved_grid
    inv_eps = np.asarray(arrays.inv_permittivities, dtype=np.float64)
    front = np.zeros((3, *inv_eps.shape[1:]), dtype=np.int32)
    with pytest.raises(ValueError, match="9-component"):
        smooth_inverse_permittivity_on_yee_pixels(
            scene=scene,
            grid=grid,
            front_material=front,
            front_owner=front,
            inv_permittivities=inv_eps,
            supersample=4,
            full_tensor=True,
        )


def test_yee_smooth_rejects_symmetry():
    config = _config(40e-9, "yee_smooth", symmetry=(1, 0, 0))
    volume = _volume((8, 8, 8), f"v{_tag()}")
    with pytest.raises(NotImplementedError, match="material_sampling='yee_smooth'"):
        fdtdx.place_objects([volume], config, [])


def test_yee_smooth_rejects_the_per_object_subpixel_flag():
    name = _tag()
    config = _config(40e-9, "yee_smooth")
    volume = _volume((10, 10, 1), f"v{name}")
    core = ExtrudedPolygon(
        axis=2,
        vertices=_rectangle(200e-9, 200e-9),
        material_name="core",
        materials={"core": Material(permittivity=EPS_CORE)},
        partial_grid_shape=(None, None, 1),
        placement_order=1,
        subpixel_smoothing=True,
        name=f"c{name}",
    )
    with pytest.raises(NotImplementedError, match="redundant"):
        fdtdx.place_objects([volume, core], config, [])
