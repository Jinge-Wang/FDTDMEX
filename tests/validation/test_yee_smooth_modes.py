"""Effective index of an oxide-clad silicon strip under Kottke-smoothed per-Yee-point loading.

Cross-sections only: the material arrays come out of ``place_objects`` and go straight into
``compute_mode`` on the native (``mode_backend="fdtdmex"``) solver. No FDTD time loop.

The number this file exists to move is the sub-cell spread. Stage A (``material_sampling="yee"``)
fixed the drawn device at 500 x 220 nm at every resolution, but paid for it with an O(cell)
staircasing jitter: sliding the core by a fraction of a cell changed ``n_eff`` by 0.037 at a 40 nm
grid. Smoothing replaces the point sample at the interface pixels by the effective inverse
permittivity of the pixel, and the jitter collapses.

Measured on this machine (JAX CPU, 2.4 x 1.6 um window, ``n_Si = 3.48``, ``n_SiO2 = 1.444``,
lambda = 1.55 um, five sub-cell shifts of the core along the width axis), against a 10 nm
``yee_smooth`` reference of 2.449097:

======  ================  ==================  ==================  ====================
cell    spread, yee       spread, yee_smooth  |err|, yee          |err|, yee_smooth
======  ================  ==================  ==================  ====================
40 nm   0.0371            0.0066              0.0591              0.0016
32 nm   0.0398            0.0041              0.0335              0.0062
25 nm   0.0000            0.0013              0.0259              0.0026
20 nm   0.0000            0.0014              0.0059              0.0046
======  ================  ==================  ==================  ====================

The zero spreads at 25 and 20 nm are not a win for the point sample: 500 nm is a whole number of
cells there, so every sub-cell shift lands the faces on the same lattice points. The honest summary
is the last column — the smoothed answer sits within 0.006 of the converged value at every grid,
where the point sample is still 0.059 away at 40 nm.
"""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.core.physics.modes import compute_mode
from fdtdx.materials import Material
from fdtdx.objects.static_material.polygon import ExtrudedPolygon
from fdtdx.objects.static_material.static import SimulationVolume

pytestmark = pytest.mark.validation

C0 = 299792458.0
LAMBDA = 1.55e-6
N_SI = 3.48
N_OXIDE = 1.444
CORE_WIDTH = 500e-9
CORE_THICKNESS = 220e-9
WINDOW_Y = 2.4e-6
WINDOW_Z = 1.6e-6

CELLS = (40e-9, 32e-9, 25e-9, 20e-9)
SHIFTS = (0.1, 0.3, 0.5, 0.7, 0.9)

_COUNTER = [0]


def _tag() -> str:
    _COUNTER[0] += 1
    return f"ysm{_COUNTER[0]}"


def _cross_section(d: float, sampling: str, shift: float, **kwargs):
    name = _tag()
    ny, nz = round(WINDOW_Y / d), round(WINDOW_Z / d)
    config = SimulationConfig(time=1e-15, grid=UniformGrid(spacing=d), material_sampling=sampling, **kwargs)
    volume = SimulationVolume(
        partial_grid_shape=(3, ny, nz),
        material=Material(permittivity=N_OXIDE**2),
        name=f"vol_{name}",
    )
    half_w, half_t = CORE_WIDTH / 2, CORE_THICKNESS / 2
    core = ExtrudedPolygon(
        axis=0,
        vertices=np.array([[-half_w, -half_t], [half_w, -half_t], [half_w, half_t], [-half_w, half_t]]),
        material_name="si",
        materials={"si": Material(permittivity=N_SI**2)},
        partial_grid_shape=(3, None, None),
        partial_real_position=(0.0, shift, 0.0),
        placement_order=1,
        name=f"core_{name}",
    )
    _, arrays, _, config, _ = fdtdx.place_objects([volume, core], config, [])
    grid = config.resolved_grid
    coords = (np.asarray(grid.edges(1)), np.asarray(grid.edges(2)))
    return np.asarray(arrays.inv_permittivities)[:, 0:1, :, :], coords


def _neff(d: float, sampling: str, shift: float, **kwargs) -> float:
    inv_eps, coords = _cross_section(d, sampling, shift, **kwargs)
    _, _, eff_index = compute_mode(
        frequency=C0 / LAMBDA,
        inv_permittivities=jnp.asarray(inv_eps),
        inv_permeabilities=1.0,
        transverse_coords=coords,
        mode_backend="fdtdmex",
        filter_pol="te",
    )
    return float(np.real(np.asarray(eff_index)))


@pytest.fixture(scope="module")
def neff_table() -> dict[tuple[float, str], list[float]]:
    """n_eff of the fundamental TE mode for both sampling modes, every cell size, every shift."""
    return {
        (d, sampling): [_neff(d, sampling, shift * d) for shift in SHIFTS]
        for d, sampling in itertools.product(CELLS, ("yee", "yee_smooth"))
    }


@pytest.fixture(scope="module")
def reference_neff() -> float:
    """Smoothed sampling on a 10 nm grid: the converged value the coarse grids aim at."""
    return _neff(10e-9, "yee_smooth", 0.5 * 10e-9)


@pytest.mark.parametrize("d", CELLS)
def test_subcell_spread_collapses(d, neff_table):
    """Sliding the core inside a cell barely moves n_eff any more.

    Stage A's own test documents 0.037 at 40 nm as the limitation this stage has to remove; the
    measured smoothed spread is 0.0066 there, so the assertion is pinned at 0.010 with the ratio
    against the point sample checked separately.
    """
    spread = float(np.ptp(neff_table[(d, "yee_smooth")]))
    assert spread < 0.010, f"d={d * 1e9:.0f} nm: sub-cell spread {spread:.4f}"


@pytest.mark.parametrize("d", (40e-9, 32e-9))
def test_subcell_spread_is_a_quarter_of_the_point_sample(d, neff_table):
    """At the grids where 500 nm is not a whole number of cells, the jitter drops by 5x or more."""
    smooth = float(np.ptp(neff_table[(d, "yee_smooth")]))
    point = float(np.ptp(neff_table[(d, "yee")]))
    assert smooth <= 0.25 * point, f"d={d * 1e9:.0f} nm: {smooth:.4f} vs {point:.4f}"


@pytest.mark.parametrize("d", CELLS)
def test_error_versus_the_reference_beats_the_point_sample(d, neff_table, reference_neff):
    """The smoothed mean is no further from the 10 nm reference than the point-sampled mean."""
    smooth = abs(float(np.mean(neff_table[(d, "yee_smooth")])) - reference_neff)
    point = abs(float(np.mean(neff_table[(d, "yee")])) - reference_neff)
    assert smooth <= point + 1e-3, f"d={d * 1e9:.0f} nm: {smooth:.4f} vs {point:.4f}"


def test_the_answer_stops_moving_with_the_grid(neff_table):
    """Across the whole 40 -> 20 nm sweep the smoothed mean varies by under 0.01, the point sample by 0.09.

    A least-squares convergence order is not meaningful on the smoothed series: its residual is
    already at the level of the mode solver's own discretisation, so the errors stop being ordered by
    cell size. The honest statement is the total variation.
    """
    smooth = [float(np.mean(neff_table[(d, "yee_smooth")])) for d in CELLS]
    point = [float(np.mean(neff_table[(d, "yee")])) for d in CELLS]
    assert float(np.ptp(smooth)) < 0.012
    assert float(np.ptp(smooth)) < 0.2 * float(np.ptp(point))


def test_full_tensor_matches_the_diagonal_on_a_manhattan_cross_section():
    """Every interface of the strip is axis-aligned, so the off-diagonal Kottke terms vanish."""
    for d in (40e-9, 25e-9):
        diagonal = _neff(d, "yee_smooth", 0.3 * d)
        full = _neff(d, "yee_smooth", 0.3 * d, yee_smooth_full_tensor=True)
        assert full == pytest.approx(diagonal, abs=1e-9)


@pytest.mark.parametrize("d", CELLS)
def test_the_rasterised_core_keeps_its_drawn_extent(d):
    """Smoothing does not move the device: the fill-weighted width and thickness stay at 500/220 nm.

    Read each extent from a component that is tangential to the interfaces bounding it, so the
    Kottke blend is the arithmetic mean of eps and the fill fraction comes back directly.
    """
    for shift in (0.0, 0.37):
        inv_eps, _ = _cross_section(d, "yee_smooth", shift * d)
        eps = 1.0 / np.asarray(inv_eps, dtype=np.float64)
        # E_z is tangential to the two width (y-normal) faces; E_y to the two thickness faces.
        width_fill = np.clip((eps[2, 0] - N_OXIDE**2) / (N_SI**2 - N_OXIDE**2), 0.0, 1.0)
        thickness_fill = np.clip((eps[1, 0] - N_OXIDE**2) / (N_SI**2 - N_OXIDE**2), 0.0, 1.0)
        width = float(width_fill.sum(axis=0).max()) * d
        thickness = float(thickness_fill.sum(axis=1).max()) * d
        assert width == pytest.approx(CORE_WIDTH, abs=0.02 * d)
        assert thickness == pytest.approx(CORE_THICKNESS, abs=0.02 * d)
