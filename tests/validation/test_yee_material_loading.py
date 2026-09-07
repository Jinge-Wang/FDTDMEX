"""Effective index of an oxide-clad silicon strip under per-Yee-point material loading.

Cross-sections only: the material arrays come out of ``place_objects`` and go straight into
``compute_mode`` on the native (``mode_backend="fdtdmex"``) solver. No FDTD time loop.

What the two sampling modes do to a 500 x 220 nm strip, measured on this machine (JAX CPU,
2.4 x 1.6 um window, ``n_Si = 3.48``, ``n_SiO2 = 1.444``, lambda = 1.55 um, five sub-cell shifts of
the core along the width axis):

======  =====================  =====================  =========================  =========================
cell    mean n_eff, box        spread, box            mean n_eff, yee            spread, yee
======  =====================  =====================  =========================  =========================
40 nm   2.3592                 0.0000                 2.3900                     0.0371
32 nm   2.4956                 0.0000                 2.4826                     0.0398
25 nm   2.4759                 0.0000                 2.4750                     0.0000
20 nm   2.4550                 0.0000                 2.4550                     0.0000
======  =====================  =====================  =========================  =========================

Read honestly. The box path has *zero* sub-cell spread because the object is frozen onto the grid —
and it simulates a different device at every resolution (its rasterised width/thickness is
480/200, 512/224, 500/225, 500/220 nm). The yee path fixes the device at 500/220 nm and pays for it
with O(cell) staircasing jitter, which is what Stage B (fill-fraction smoothing) has to remove. The
spread vanishes at 25 and 20 nm only because 500 nm is a whole number of cells there.
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
#: Sub-cell shifts of the core along the width axis. Deliberately off the cell edges: a face landing
#: exactly on a lattice point is a tie whose outcome float32 grid coordinates decide.
SHIFTS = (0.1, 0.3, 0.5, 0.7, 0.9)

_COUNTER = [0]


def _tag() -> str:
    _COUNTER[0] += 1
    return f"ym{_COUNTER[0]}"


def _cross_section(d: float, sampling: str, shift: float):
    """Place the strip and return its assembled material cross-section plus the transverse edges."""
    name = _tag()
    ny, nz = round(WINDOW_Y / d), round(WINDOW_Z / d)
    config = SimulationConfig(time=1e-15, grid=UniformGrid(spacing=d), material_sampling=sampling)
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


def _neff(d: float, sampling: str, shift: float) -> float:
    inv_eps, coords = _cross_section(d, sampling, shift)
    _, _, eff_index = compute_mode(
        frequency=C0 / LAMBDA,
        inv_permittivities=jnp.asarray(inv_eps),
        inv_permeabilities=1.0,
        transverse_coords=coords,
        mode_backend="fdtdmex",
        filter_pol="te",
    )
    return float(np.real(np.asarray(eff_index)))


def _rasterised_extent(d: float, sampling: str, shift: float) -> tuple[float, float]:
    """Core width and thickness read from each component at its own position, in metres."""
    inv_eps, _ = _cross_section(d, sampling, shift)
    eps = 1.0 / inv_eps
    threshold = 0.5 * (N_SI**2 + N_OXIDE**2)
    core = eps > threshold
    # Each extent must be read from a component whose lattice is dense along that axis: E_z samples
    # the width axis (y) at the nodes, E_x samples the thickness axis (z) at the nodes. In box mode
    # there is only one component and it carries the same cell-centre mask for every field.
    width_component = 2 if core.shape[0] == 3 else 0
    thickness_component = 0
    width = float(core[width_component, 0].sum(axis=0).max()) * d
    thickness = float(core[thickness_component, 0].sum(axis=1).max()) * d
    return width, thickness


@pytest.fixture(scope="module")
def neff_table() -> dict[tuple[float, str], list[float]]:
    """n_eff of the fundamental TE mode for both modes, every cell size, every sub-cell shift."""
    table: dict[tuple[float, str], list[float]] = {}
    for d in CELLS:
        for sampling in ("box", "yee"):
            table[(d, sampling)] = [_neff(d, sampling, shift * d) for shift in SHIFTS]
    return table


@pytest.fixture(scope="module")
def reference_neff() -> float:
    """Per-component sampling on a 10 nm grid, the converged value the coarse grids aim at."""
    return _neff(10e-9, "yee", 0.5 * 10e-9)


@pytest.mark.parametrize("d", CELLS)
def test_yee_keeps_the_core_extent_at_every_resolution(d):
    """The rasterised core stays within one sample of 500 x 220 nm; the box path does not.

    Box-path rasterised width/thickness for contrast: 480/200 at 40 nm, 512/224 at 32 nm,
    500/225 at 25 nm, 500/220 at 20 nm.
    """
    widths = []
    for shift in SHIFTS:
        width, thickness = _rasterised_extent(d, "yee", shift * d)
        widths.append(width)
        assert abs(width - CORE_WIDTH) <= d, f"width {width * 1e9:.1f} nm at cell {d * 1e9:.0f} nm"
        assert abs(thickness - CORE_THICKNESS) <= d
    assert abs(float(np.mean(widths)) - CORE_WIDTH) <= 0.3 * d


def test_box_path_simulates_a_different_device_at_every_resolution():
    """The evidence that motivates the change: the box path's core is a different size each time."""
    rasterised = {d: _rasterised_extent(d, "box", 0.5 * d) for d in CELLS}
    assert rasterised[40e-9][0] == pytest.approx(480e-9, rel=1e-5)
    assert rasterised[32e-9][0] == pytest.approx(512e-9, rel=1e-5)
    assert rasterised[25e-9][0] == pytest.approx(500e-9, rel=1e-5)
    assert rasterised[20e-9][0] == pytest.approx(500e-9, rel=1e-5)


@pytest.mark.parametrize("d", CELLS)
def test_yee_neff_is_shift_invariant_within_the_staircase_bound(d, neff_table):
    """Stage A leaves an O(cell) sub-cell jitter; this pins how large it is allowed to be.

    Measured spread over the five shifts: 0.0371 at 40 nm, 0.0398 at 32 nm, 0.0000 at 25 nm and
    0.0000 at 20 nm (the last two are exact because 500 nm is a whole number of cells there). This
    test documents a Stage A limitation, not a win: it is the number fill-fraction smoothing has to
    drive down.
    """
    values = np.asarray(neff_table[(d, "yee")])
    bound = 0.06 if d >= 32e-9 else 0.02
    assert float(np.ptp(values)) < bound, f"yee n_eff spread {np.ptp(values):.4f} at cell {d * 1e9:.0f} nm"


def test_yee_neff_converges_monotonically(neff_table, reference_neff):
    """The offset-averaged n_eff moves monotonically towards the 10 nm reference as the cell shrinks.

    Measured errors against the 10 nm reference (2.45098): 0.0610, 0.0316, 0.0240, 0.0040 for
    40, 32, 25 and 20 nm. The 32 -> 25 nm step is small (0.008), hence the 1e-3 slack.
    """
    errors = [abs(float(np.mean(neff_table[(d, "yee")])) - reference_neff) for d in CELLS]
    for coarse, fine in itertools.pairwise(errors):
        assert fine <= coarse + 1e-3, f"n_eff error did not fall with the cell size: {errors}"
    assert errors[-1] < 0.01


@pytest.mark.parametrize("d", CELLS)
def test_yee_neff_is_no_further_from_the_reference_than_the_box_path(d, neff_table, reference_neff):
    """Fixing the device geometry does not cost accuracy at any resolution.

    Measured errors against the 10 nm reference, yee vs box: 0.0610 vs 0.0918 at 40 nm,
    0.0316 vs 0.0446 at 32 nm, 0.0240 vs 0.0249 at 25 nm, 0.0040 vs 0.0040 at 20 nm.
    """
    yee_error = abs(float(np.mean(neff_table[(d, "yee")])) - reference_neff)
    box_error = abs(float(np.mean(neff_table[(d, "box")])) - reference_neff)
    assert yee_error <= box_error + 5e-3


# ---------------------------------------------------------------------------
# The permeability on the H lattices
# ---------------------------------------------------------------------------

MU_CORE = 2.4


def _magnetic_cross_section(d: float, sampling: str, shift: float):
    """The same strip with a magnetic core, returning the assembled arrays and the loader report."""
    name = _tag()
    ny, nz = round(WINDOW_Y / d), round(WINDOW_Z / d)
    config = SimulationConfig(time=1e-15, grid=UniformGrid(spacing=d), material_sampling=sampling)
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
        materials={"si": Material(permittivity=N_SI**2, permeability=MU_CORE)},
        partial_grid_shape=(3, None, None),
        partial_real_position=(0.0, shift, 0.0),
        placement_order=1,
        name=f"core_{name}",
    )
    _, arrays, _, _, info = fdtdx.place_objects([volume, core], config, [])
    return arrays, info


def test_a_magnetic_core_is_smoothed_on_the_h_lattices():
    """End to end: a mu = 2.4 core in a mu = 1 cladding loads a smoothed 3-component permeability.

    The interior and the cladding keep their exact point values, the interface cells sit strictly
    between them, and the point-sampled load of the same scene differs from the smoothed one only
    where the core's faces cut a cell. The electric side is unaffected: the permittivity arrays of
    the magnetic and non-magnetic strips are identical, because the permeability contrast is not a
    permittivity contrast.
    """
    d, shift = 32e-9, 0.3 * 32e-9
    smooth, info = _magnetic_cross_section(d, "yee_smooth", shift)
    point, point_info = _magnetic_cross_section(d, "yee", shift)

    inv_mu = np.asarray(smooth.inv_permeabilities, dtype=np.float64)
    assert inv_mu.shape[0] == 3
    assert "smoothing_H" not in point_info["yee_sampling_difference"]
    stats = info["yee_sampling_difference"]["smoothing_H"]
    assert stats["num_smoothed"] > 0
    assert stats["num_three_material_fallbacks"] == 0
    assert stats["num_metal_skips"] == 0

    mu = 1.0 / inv_mu
    assert float(mu.min()) == pytest.approx(1.0, rel=1e-6)
    assert float(mu.max()) == pytest.approx(MU_CORE, rel=1e-6)
    between = (mu > 1.0 + 1e-6) & (mu < MU_CORE - 1e-6)
    assert int(between.sum()) > 0
    assert int(between.sum()) <= stats["num_smoothed"]

    point_mu = np.asarray(point.inv_permeabilities, dtype=np.float64)
    differing = inv_mu != point_mu
    assert int(differing.sum()) > 0
    assert int(differing.sum()) <= stats["num_smoothed"]
    # Every cell that moved is an interface cell: the point sample there is one of the two extremes.
    # The comparison is at float32, which is what the arrays are stored in.
    moved = point_mu[differing]
    assert np.all((np.abs(moved - 1.0) < 1e-6) | (np.abs(moved - 1.0 / MU_CORE) < 1e-6))

    # The electric side does not notice: the same strip without the magnetic core loads the same
    # permittivity, because a permeability contrast is not a permittivity contrast.
    dielectric, _ = _cross_section(d, "yee_smooth", shift)
    np.testing.assert_array_equal(np.asarray(smooth.inv_permittivities)[:, 0:1, :, :], dielectric)


def test_the_magnetic_strip_keeps_its_extent_like_the_dielectric_one():
    """The smoothed permeability recovers the core's true 500 x 220 nm cross-section, not a raster.

    Read from ``mu_xx`` on the ``H_x`` lattice: summing the recovered fill fraction along one axis
    and taking the maximum gives the extent along the other, which is the same measurement the
    permittivity tests above make on the E lattices.
    """
    d, shift = 32e-9, 0.3 * 32e-9
    smooth, _ = _magnetic_cross_section(d, "yee_smooth", shift)
    mu = 1.0 / np.asarray(smooth.inv_permeabilities, dtype=np.float64)
    # H_x sits at (e_x, c_y, c_z): primal in y and z, so its pixel tiles the cross-section exactly.
    fill = np.clip((mu[0, 0] - 1.0) / (MU_CORE - 1.0), 0.0, 1.0)
    width = float(np.max(fill.sum(axis=0))) * d
    thickness = float(np.max(fill.sum(axis=1))) * d
    assert width == pytest.approx(CORE_WIDTH, abs=0.05 * d)
    assert thickness == pytest.approx(CORE_THICKNESS, abs=0.05 * d)
