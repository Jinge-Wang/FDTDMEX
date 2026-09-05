"""Per-Yee-point sampling and Kottke smoothing on a non-uniform (rectilinear) grid.

Placement and array assembly only — no mode solve and no time loop.

Everything the two Yee modes measure a length with has to come from the local cell edges, not from
a single scalar spacing. Three places could have hidden a uniform-grid assumption:

1. the sample lattice — ``yee_lattice_coordinates`` puts a component on the cell edge or the cell
   centre per axis, and on a graded grid the centre is not the midpoint of a constant step;
2. the smoothing pixel — the dual box around an edge-sitting component is
   ``[e[i] - w[i-1]/2, e[i] + w[i]/2]`` with *two different* widths on a graded grid, which is the
   metric the backward difference already divides by;
3. the area a fill fraction is weighted with, when the rasterised shape is read back out.

These tests pin all three, and then repeat the rectangle- and disk-area convergence study of
``test_yee_smooth.py`` on a family of graded grids whose cell widths vary by a factor of three
across the domain.
"""

import numpy as np
import pytest

import fdtdx
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import QuasiUniformGrid, RectilinearGrid
from fdtdx.core.physics.geometry_raster import cell_center_coordinates, yee_lattice_coordinates
from fdtdx.core.physics.geometry_smooth import pixel_axis_bounds
from fdtdx.materials import Material
from fdtdx.objects.static_material.cylinder import Cylinder
from fdtdx.objects.static_material.polygon import ExtrudedPolygon
from fdtdx.objects.static_material.static import SimulationVolume

EPS_BG = 2.085
EPS_CORE = 12.1104

_COUNTER = [0]


def _tag() -> str:
    _COUNTER[0] += 1
    return f"yn{_COUNTER[0]}"


def _graded_widths(span: float, n: int, strength: float = 0.5) -> np.ndarray:
    """Smoothly graded cell widths summing to ``span``.

    ``w(s) = 1 + strength * cos(2 pi s)`` sampled at the cell midpoints, so the same continuous
    width profile is resampled at every refinement level and the local width halves when ``n``
    doubles. With ``strength = 0.5`` the widest cell is three times the narrowest.
    """
    s = (np.arange(n) + 0.5) / n
    widths = 1.0 + strength * np.cos(2.0 * np.pi * s)
    return widths * (span / widths.sum())


def _uniform_grid(dx: float, dy: float, dz: float, shape: tuple[int, int, int]) -> RectilinearGrid:
    """A rectilinear grid with a different (constant) spacing on each axis, centred on the origin.

    ``QuasiUniformGrid`` would say the same thing but insists on an even cell count on every axis,
    which rules out the single-cell z these 2-D scenes use.
    """
    spacings = (dx, dy, dz)
    edges = [np.arange(shape[a] + 1) * spacings[a] - 0.5 * shape[a] * spacings[a] for a in range(3)]
    return RectilinearGrid.custom(x_edges=edges[0], y_edges=edges[1], z_edges=edges[2])


def _graded_grid(span: float, n: int, thickness: float, strength: float = 0.5) -> RectilinearGrid:
    """A 2-D (one cell thick in z) rectilinear grid, graded in x and y, centred on the origin."""
    widths = _graded_widths(span, n, strength)
    edges = np.concatenate([[0.0], np.cumsum(widths)]) - 0.5 * span
    return RectilinearGrid.custom(
        x_edges=np.asarray(edges),
        y_edges=np.asarray(edges),
        z_edges=np.asarray([-0.5 * thickness, 0.5 * thickness]),
    )


# ---------------------------------------------------------------------------
# 1. The sample lattice
# ---------------------------------------------------------------------------


def test_yee_sample_positions_follow_the_local_edges():
    """Every component's lattice is built from this grid's own edges, per axis, not from a spacing."""
    grid = QuasiUniformGrid(dx=10e-9, dy=25e-9, dz=40e-9).resolve((8, 6, 4))
    edges = [np.asarray(grid.edges(axis), dtype=float) for axis in range(3)]
    centers = [0.5 * (e[:-1] + e[1:]) for e in edges]
    # The three axes really do have three different spacings, so a scalar-spacing bug shows up.
    assert not np.isclose(np.diff(edges[0])[0], np.diff(edges[1])[0])

    offsets = {"E": ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5))}
    offsets["H"] = ((0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))
    for field, table in offsets.items():
        for component in range(3):
            coords = yee_lattice_coordinates(grid, field, component)
            for axis in range(3):
                expected = centers[axis] if table[component][axis] == 0.5 else edges[axis][:-1]
                np.testing.assert_allclose(np.asarray(coords[axis]), expected, rtol=0, atol=0)


def test_graded_cell_centres_are_not_a_constant_step():
    """A graded grid's centres drift away from any uniform lattice — the case the test above covers."""
    grid = _graded_grid(2.0e-6, 32, 40e-9)
    centers = np.asarray(cell_center_coordinates(grid)[0], dtype=float)
    steps = np.diff(centers)
    assert steps.max() / steps.min() > 2.0


# ---------------------------------------------------------------------------
# 2. The smoothing pixel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("component", [0, 1, 2])
def test_pixel_bounds_use_the_local_widths(component):
    """Primal cell where the component sits at a centre, local dual cell where it sits at an edge."""
    grid = _graded_grid(2.0e-6, 24, 40e-9)
    bounds = pixel_axis_bounds(grid, "E", component)
    offsets = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5))[component]

    for axis in range(2):  # z is a single cell here and is covered by its own test below
        edges = np.asarray(grid.edges(axis), dtype=float)
        widths = np.diff(edges)
        lower, upper = bounds[axis]
        if offsets[axis] == 0.5:
            np.testing.assert_allclose(lower, edges[:-1], rtol=0, atol=0)
            np.testing.assert_allclose(upper, edges[1:], rtol=0, atol=0)
        else:
            previous = np.concatenate([widths[:1], widths[:-1]])
            expected_lower = np.clip(edges[:-1] - 0.5 * previous, edges[0], None)
            np.testing.assert_allclose(lower, expected_lower, rtol=1e-14, atol=0)
            np.testing.assert_allclose(upper, edges[:-1] + 0.5 * widths, rtol=1e-14, atol=0)
            # The two halves of an interior dual pixel are genuinely different on a graded grid.
            interior = slice(1, -1)
            left = (edges[:-1] - lower)[interior]
            right = (upper - edges[:-1])[interior]
            assert np.max(np.abs(left - right)) > 1e-12


def test_the_invariant_axis_collapses_on_a_graded_grid_too():
    """One cell in z means a degenerate pixel there, independently of how x and y are graded."""
    grid = _graded_grid(2.0e-6, 16, 40e-9)
    for component in range(3):
        lower, upper = pixel_axis_bounds(grid, "E", component)[2]
        np.testing.assert_allclose(lower, upper, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 3. Areas read back out of the assembled permittivity, on graded grids
# ---------------------------------------------------------------------------


def _pixel_areas(grid: RectilinearGrid) -> np.ndarray:
    """In-plane area of every E_z pixel: the product of the two local dual-cell widths."""
    bounds = pixel_axis_bounds(grid, "E", 2)
    widths = [np.asarray(upper - lower, dtype=float) for lower, upper in bounds[:2]]
    return widths[0][:, None] * widths[1][None, :]


def _rasterised_fill(grid: RectilinearGrid, sampling: str, shape: str) -> np.ndarray:
    """Place one shape on ``grid`` and return its E_z fill fraction as an ``(Nx, Ny)`` array.

    Legitimate for these shapes because every interface is in-plane, so E_z is tangential to all of
    them and the Kottke blend reduces to the arithmetic mean of eps — the fill fraction is then a
    linear read-back of the stored permittivity.
    """
    name = _tag()
    config = SimulationConfig(time=1e-15, grid=grid, material_sampling=sampling)
    volume = SimulationVolume(
        partial_grid_shape=grid.shape,
        material=Material(permittivity=EPS_BG),
        name=f"v{name}",
    )
    common = dict(
        axis=2,
        materials={"core": Material(permittivity=EPS_CORE)},
        material_name="core",
        partial_grid_shape=(None, None, 1),
        placement_order=1,
        name=f"c{name}",
    )
    if shape == "rectangle":
        half_w, half_h = 0.5 * RECT[0], 0.5 * RECT[1]
        core = ExtrudedPolygon(
            vertices=np.array([[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]]),
            **common,
        )
    else:
        core = Cylinder(radius=DISK_RADIUS, **common)
    _, arrays, _, _, _ = fdtdx.place_objects([volume, core], config, [])
    eps = 1.0 / np.asarray(arrays.inv_permittivities, dtype=np.float64)
    # E_z in the two Yee modes (3-component tier); the box path keeps one isotropic component.
    component = 2 if eps.shape[0] > 1 else 0
    return np.clip((eps[component, :, :, 0] - EPS_BG) / (EPS_CORE - EPS_BG), 0.0, 1.0)


def _rasterised_area(grid: RectilinearGrid, sampling: str, shape: str) -> float:
    """Fill fraction integrated over the local pixel areas."""
    return float((_rasterised_fill(grid, sampling, shape) * _pixel_areas(grid)).sum())


def _marginal_extents(grid: RectilinearGrid, sampling: str, shape: str) -> tuple[float, float]:
    """Per-axis extent of a rectangle, each measured by integrating along the *other* axis.

    ``sum_j fill[i, j] * w_y[j]`` is the height of the rectangle times the x-overlap fraction of
    pixel ``i``, so its maximum over ``i`` is the height alone — read off the y widths only. The
    transpose gives the width from the x widths only. A scalar-spacing assumption on either axis
    moves exactly one of the two numbers.
    """
    fill = _rasterised_fill(grid, sampling, shape)
    bounds = pixel_axis_bounds(grid, "E", 2)
    w_x = np.asarray(bounds[0][1] - bounds[0][0], dtype=float)
    w_y = np.asarray(bounds[1][1] - bounds[1][0], dtype=float)
    height = float(np.max(fill @ w_y))
    width = float(np.max(w_x @ fill))
    return width, height


SPAN = 2.4e-6
THICKNESS = 40e-9
RECT = (0.5e-6, 0.32e-6)
DISK_RADIUS = 0.81e-6
#: Three refinement levels of the same continuous grading; the local width halves each step.
GRADED_CELLS = (24, 48, 96)


@pytest.mark.parametrize(
    "shape, truth",
    [("rectangle", RECT[0] * RECT[1]), ("disk", np.pi * DISK_RADIUS**2)],
)
def test_smoothed_area_converges_at_second_order_on_a_graded_grid(shape, truth):
    """The rasterised area is exact at every grading level, which clears the second-order bar.

    The bound is the same one the uniform-grid test uses, with the *largest* local cell width in
    place of the spacing: ``1e-5`` relative plus ``4e-6 * h_max^2``. The smoothed area sits orders
    of magnitude inside it — the residual is the float32 storage of the permittivity array, not a
    discretisation error — so an order fit on it would be fitting noise. What the fit would show is
    in the second half of the test: the point sample's error on the same grids is more than twenty
    times larger and falls only like ``h``.
    """
    smooth_errors, point_errors = [], []
    for n in GRADED_CELLS:
        grid = _graded_grid(SPAN, n, THICKNESS)
        h_max = float(np.max(np.diff(np.asarray(grid.edges(0), dtype=float))))
        smoothed = _rasterised_area(grid, "yee_smooth", shape)
        assert abs(smoothed - truth) < 1e-5 * truth + 4e-6 * h_max**2, (
            f"{shape} on {n} graded cells: area {smoothed:.6e} vs {truth:.6e}"
        )
        smooth_errors.append(abs(smoothed - truth) / truth)
        point_errors.append(abs(_rasterised_area(grid, "yee", shape) - truth) / truth)

    assert max(smooth_errors) < 1e-5
    assert min(point_errors) > 20 * max(smooth_errors)
    # The point sample is first order in the local cell size: halving the grid roughly halves it.
    assert point_errors[-1] < 0.75 * point_errors[0]


def test_a_uniform_grading_reproduces_the_uniform_grid_answer():
    """``strength = 0`` makes the graded family uniform again; the two paths must agree exactly."""
    n = 48
    graded = _graded_grid(SPAN, n, THICKNESS, strength=0.0)
    uniform = _uniform_grid(SPAN / n, SPAN / n, THICKNESS, (n, n, 1))
    np.testing.assert_allclose(np.asarray(graded.edges(0)), np.asarray(uniform.edges(0)), rtol=1e-5, atol=1e-18)
    a_graded = _rasterised_area(graded, "yee_smooth", "rectangle")
    a_uniform = _rasterised_area(uniform, "yee_smooth", "rectangle")
    assert abs(a_graded - a_uniform) < 1e-9 * RECT[0] * RECT[1]


# ---------------------------------------------------------------------------
# 4. Placement itself on a non-uniform grid
# ---------------------------------------------------------------------------


def test_each_axis_is_quantised_by_its_own_cell_width():
    """Anisotropic cells: 30 nm in x, 50 nm in y, and each axis must use its own.

    The point-sampled mode makes this visible without any tolerance argument. It quantises the
    rectangle to whole *lattice steps* of each axis separately, so a 500 x 320 nm rectangle comes
    out 480 x 300 nm — 16 steps of 30 nm by 6 steps of 50 nm. A single scalar spacing anywhere in
    the sampling would quantise both axes by the same number and could not produce that pair.
    Smoothing then recovers the drawn extent on both axes.
    """
    grid = _uniform_grid(30e-9, 50e-9, THICKNESS, (64, 64, 1))

    width, height = _marginal_extents(grid, "yee", "rectangle")
    assert width == pytest.approx(16 * 30e-9, rel=1e-6)
    assert height == pytest.approx(6 * 50e-9, rel=1e-6)

    width, height = _marginal_extents(grid, "yee_smooth", "rectangle")
    assert width == pytest.approx(RECT[0], rel=1e-4)
    assert height == pytest.approx(RECT[1], rel=1e-4)


@pytest.mark.parametrize("sampling", ["yee", "yee_smooth"])
def test_the_scene_places_and_assembles_on_anisotropic_cells(sampling):
    """Both Yee modes assemble a finite 3-component array on a grid with three different spacings."""
    grid = _uniform_grid(30e-9, 50e-9, THICKNESS, (64, 64, 1))
    fill = _rasterised_fill(grid, sampling, "rectangle")
    assert np.all(np.isfinite(fill))
    assert 0.0 <= fill.min() and fill.max() <= 1.0
    area = _rasterised_area(grid, sampling, "rectangle")
    truth = RECT[0] * RECT[1]
    bound = 1e-4 * truth if sampling == "yee_smooth" else 1.2 * 2 * (RECT[0] + RECT[1]) * 50e-9
    assert abs(area - truth) < bound
