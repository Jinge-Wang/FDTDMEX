"""The off-diagonal Kottke entries on the cell-vertex lattice, and the stencil that applies them.

``material_sampling="yee_smooth"`` with ``yee_smooth_full_tensor=True`` used to write the whole
Kottke row at each component's own pixel, into a dense 9-component tensor. The two coupled rows then
read two different boxes, and the assembled D-to-E map is asymmetric — measurably, 1-5% in Frobenius
norm — which puts complex pairs into the spectrum of ``M K`` and lets a mode grow on a curved
interface. ``yee_smooth_offdiag_placement="node"`` instead leaves the diagonal entries exactly where
they were and puts the three off-diagonal entries on the primary-grid vertices, half a cell back
along each row's own axis, where both coupled rows read the same number. The map is then its own
transpose.

These tests pin, in order: the stencil against an explicit index-by-index transcription of Meep's
``OFFDIAG`` macro; the assembled map's symmetry and spectrum on a tilted slab and a disk; that a
scene without the entries is bit-identical to the diagonal tier; that an axis-aligned scene produces
no entries at all; that the reverse update inverts the forward one; and that the 2-D ``E_z``
polarization cannot be touched by the correction at all.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.core.physics.geometry_smooth import (
    OFFDIAGONAL_ENTRIES,
    apply_dtoe_map,
    min_eigenvalue_of_symmetric_part,
    pixel_diagonals_from_vertex_tensor,
    vertex_offdiagonals_from_pixel_rows,
)
from fdtdx.fdtd.misc import OFFDIAG_ROW_PARTNERS, add_offdiag_correction
from fdtdx.fdtd.update import offdiag_correction_terms, update_E, update_E_reverse
from fdtdx.materials import Material
from fdtdx.objects.static_material.cylinder import Cylinder
from fdtdx.objects.static_material.polygon import ExtrudedPolygon
from fdtdx.objects.static_material.static import SimulationVolume, UniformMaterialObject

_COUNTER = [0]


def _mlx_available() -> bool:
    """Whether the MLX twin of the stencil can be exercised here."""
    try:
        import mlx.core  # noqa: F401
    except Exception:  # pragma: no cover - depends on the machine
        return False
    return True


def _tag() -> str:
    _COUNTER[0] += 1
    return f"nd{_COUNTER[0]}"


@pytest.fixture
def float64():
    """Run one test in double precision, restoring the global flag afterwards.

    The stencil is exact arithmetic on a handful of terms, so its agreement with an explicit
    assembly is limited only by round-off; asserting that at 1e-12 needs more than float32.
    """
    previous = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


# ---------------------------------------------------------------------------
# Reference implementations, written independently of the production code
# ---------------------------------------------------------------------------


def _pad_field(field: np.ndarray, periodic: tuple[bool, bool, bool]) -> np.ndarray:
    """Zero halo, wrapped on the periodic axes — the field padding the updates already use."""
    padded = np.zeros((3, *[n + 2 for n in field.shape[1:]]), dtype=float)
    padded[:, 1:-1, 1:-1, 1:-1] = field
    for axis, wrap in enumerate(periodic):
        if not wrap:
            continue
        keep = [slice(None)] * 4
        keep[axis + 1] = slice(1, -1)
        width = [(0, 0)] * 4
        width[axis + 1] = (1, 1)
        padded = np.pad(padded[tuple(keep)], width, mode="wrap")
    return padded


def _pad_entries(entries: np.ndarray, periodic: tuple[bool, bool, bool]) -> np.ndarray:
    """Edge-replicated halo, wrapped on the periodic axes — the coefficient padding."""
    padded = np.pad(entries, [(0, 0)] + [(1, 1)] * 3, mode="edge")
    for axis, wrap in enumerate(periodic):
        if not wrap:
            continue
        keep = [slice(None)] * 4
        keep[axis + 1] = slice(1, -1)
        width = [(0, 0)] * 4
        width[axis + 1] = (1, 1)
        padded = np.pad(padded[tuple(keep)], width, mode="wrap")
    return padded


def _explicit_offdiag(entries: np.ndarray, increment: np.ndarray, periodic) -> np.ndarray:
    """Meep's ``OFFDIAG`` macro, transcribed index by index with Python loops.

    ``OFFDIAG(u, g, sx) = 0.25 * ((g[i] + g[i-sx]) * u[i] + (g[i+s] + g[(i+s)-sx]) * u[i+s])`` with
    ``s`` the step along the row's own axis and ``sx`` the step along the partner's. Deliberately
    written as scalar lookups with explicit out-of-range handling, so it shares no code with the
    vectorised production stencil.
    """
    shape = increment.shape[1:]
    out = np.zeros_like(increment)

    def sample(component: int, index) -> float:
        wrapped = list(index)
        for axis in range(3):
            if 0 <= wrapped[axis] < shape[axis]:
                continue
            if periodic[axis]:
                wrapped[axis] %= shape[axis]
            else:
                return 0.0  # the field halo is zero at a PML/PEC face
        return float(increment[component][tuple(wrapped)])

    def entry(which: int, index) -> float:
        clamped = list(index)
        for axis in range(3):
            if 0 <= clamped[axis] < shape[axis]:
                continue
            if periodic[axis]:
                clamped[axis] %= shape[axis]
            else:
                clamped[axis] = min(max(clamped[axis], 0), shape[axis] - 1)
        return float(entries[which][tuple(clamped)])

    for component in range(3):
        for partner, which in OFFDIAG_ROW_PARTNERS[component]:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        for near in (0, 1):
                            vertex = [i, j, k]
                            vertex[component] += near
                            lower = list(vertex)
                            lower[partner] -= 1
                            out[component][i, j, k] += (
                                0.25 * entry(which, vertex) * (sample(partner, vertex) + sample(partner, lower))
                            )
    return out


# ---------------------------------------------------------------------------
# (a) The stencil itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "periodic",
    [(False, False, False), (True, True, True), (True, False, True)],
    ids=["terminated", "periodic", "mixed"],
)
def test_the_stencil_matches_an_explicit_transcription_of_meeps_macro(float64, periodic):
    """The vectorised helper equals the index-by-index assembly to round-off, on random arrays."""
    rng = np.random.default_rng(11)
    shape = (5, 4, 3)
    entries = rng.standard_normal((3, *shape))
    increment = rng.standard_normal((3, *shape))
    field = rng.standard_normal((3, *shape))

    expected = field + _explicit_offdiag(entries, increment, periodic)
    obtained = np.asarray(
        add_offdiag_correction(
            jnp.asarray(field),
            jnp.asarray(_pad_field(increment, periodic)),
            jnp.asarray(_pad_entries(entries, periodic)),
        )
    )
    assert np.max(np.abs(expected - obtained)) < 1e-12


@pytest.mark.parametrize(
    "periodic",
    [(False, False, False), (True, True, True), (True, False, True)],
    ids=["terminated", "periodic", "mixed"],
)
def test_the_stencil_is_its_own_transpose(float64, periodic):
    """The whole point of the placement: both coupled rows read one shared vertex entry.

    Assembled column by column on a small grid, the correction operator equals its transpose
    exactly. The counter-case is the pixel placement, where the ``xy`` entry of row x is sampled on
    the ``E_x`` pixel and the ``yx`` entry of row y on the ``E_y`` pixel — two different boxes, and
    an operator 1-5% away from symmetric.
    """
    rng = np.random.default_rng(12)
    shape = (4, 5, 3)
    entries = rng.standard_normal((3, *shape))
    padded_entries = jnp.asarray(_pad_entries(entries, periodic))
    size = 3 * int(np.prod(shape))
    zero = jnp.zeros((3, *shape), dtype=jnp.float64)

    operator = np.zeros((size, size))
    for column in range(size):
        basis = np.zeros(size)
        basis[column] = 1.0
        operator[:, column] = np.asarray(
            add_offdiag_correction(zero, jnp.asarray(_pad_field(basis.reshape((3, *shape)), periodic)), padded_entries)
        ).reshape(-1)

    assert np.max(np.abs(operator - operator.T)) == 0.0


def test_the_partner_average_is_dual_cell_weighted_on_a_graded_axis(float64):
    """On a non-uniform grid the two samples straddling a vertex are weighted by the far widths.

    The vertex is the edge between two neighbouring cell centres, so the interpolation weight of the
    sample at ``j`` is the *other* cell's width: ``(v[j] w[j-1] + v[j-1] w[j]) / (w[j-1] + w[j])``.
    Along the row's own axis the component point is the exact midpoint of its own two edges on any
    grid, so that half stays 1/2 and 1/2 — which is why only one weight appears here.
    """
    shape = (3, 4, 1)
    widths = np.array([1.0, 3.0, 1.0, 2.0])  # graded on y, the partner axis of row x

    def _padded_widths(axis_widths, axis):
        padded = np.concatenate([axis_widths[:1], axis_widths, axis_widths[-1:]])
        broadcast = [1, 1, 1]
        broadcast[axis] = padded.size
        return jnp.asarray(padded.reshape(broadcast))

    # The shape ``get_anisotropic_averaging_widths`` returns: one length ``N + 2`` array per axis,
    # edge-replicated to line up with the field halo, broadcasting along its own axis.
    aniso = [
        _padded_widths(np.ones(shape[0]), 0),
        _padded_widths(widths, 1),
        _padded_widths(np.ones(shape[2]), 2),
    ]

    entries = np.zeros((3, *shape))
    entries[0, 1, 2, 0] = 1.0  # a single xy entry at vertex (1, 2, 0)
    increment = np.zeros((3, *shape))
    increment[1, 1, 2, 0] = 5.0  # D_y just above the vertex, in cell j = 2
    increment[1, 1, 1, 0] = 7.0  # D_y just below it, in cell j = 1

    periodic = (False, False, False)
    result = np.asarray(
        add_offdiag_correction(
            jnp.zeros((3, *shape), dtype=jnp.float64),
            jnp.asarray(_pad_field(increment, periodic)),
            jnp.asarray(_pad_entries(entries, periodic)),
            tuple(aniso),
        )
    )
    # Row x, cell (0, 2, 0) and (1, 2, 0) both bracket vertex (1, 2, 0); each takes half of the
    # weighted partner average, with weights w[1] = 3 on the lower sample and w[2] = 1 on the upper.
    expected = 0.5 * (5.0 * widths[1] + 7.0 * widths[2]) / (widths[1] + widths[2])
    assert result[0, 0, 2, 0] == pytest.approx(expected, rel=1e-14)
    assert result[0, 1, 2, 0] == pytest.approx(expected, rel=1e-14)


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

_D = 40e-9
_N = 16


def _config(eps_placement: str = "node", full_tensor: bool = True, **kwargs) -> SimulationConfig:
    return SimulationConfig(
        time=1e-15,
        grid=UniformGrid(spacing=_D),
        material_sampling="yee_smooth",
        yee_smooth_full_tensor=full_tensor,
        yee_smooth_offdiag_placement=eps_placement,
        **kwargs,
    )


def _disk(eps_core: float, tag: str, cells: int = _N):
    return Cylinder(
        axis=2,
        radius=0.30 * cells * _D,
        material_name="core",
        materials={"core": Material(permittivity=eps_core)},
        partial_grid_shape=(None, None, 1),
        placement_order=1,
        name=f"disk{tag}",
    )


def _tilted_slab(eps_core: float, tag: str, cells: int = _N):
    """A rectangle rotated 30 degrees, fully inside the domain: two long faces with a tilted normal."""
    half_long, half_short = 0.32 * cells * _D, 0.11 * cells * _D
    corners = np.array(
        [[-half_long, -half_short], [half_long, -half_short], [half_long, half_short], [-half_long, half_short]]
    )
    angle = np.radians(30.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return ExtrudedPolygon(
        axis=2,
        vertices=corners @ rotation.T,
        material_name="core",
        materials={"core": Material(permittivity=eps_core)},
        partial_grid_shape=(None, None, 1),
        placement_order=1,
        name=f"slab{tag}",
    )


def _silicon_air_edge_slab(eps_core: float, tag: str, cells: int = _N):
    """A tilted rectangle whose long edge is one straight interface, for F5's hand-computed check.

    Sized so a handful of vertices near the middle of the long edge sit several cells from the
    short edges and from the domain's periodic seam: the fill fraction there is exactly what one
    straight line cuts off a rectangle, checkable against the shape's own ``box_fill_fraction``
    independently of the vertex-vs-pixel choice ``smooth_offdiagonal_on_vertex_lattice`` makes.
    """
    half_long = 0.42 * cells * _D
    half_short = 0.12 * cells * _D
    corners = np.array(
        [[-half_long, -half_short], [half_long, -half_short], [half_long, half_short], [-half_long, half_short]]
    )
    angle = np.radians(20.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return ExtrudedPolygon(
        axis=2,
        vertices=corners @ rotation.T,
        material_name="core",
        materials={"core": Material(permittivity=eps_core)},
        partial_grid_shape=(None, None, 1),
        placement_order=1,
        name=f"edge{tag}",
    )


def _periodic_scene(shape_fn, eps_core: float, cells: int = _N, **config_kwargs):
    """One shape in a fully periodic 2-D box, loaded through the fork's own material loader."""
    tag = _tag()
    config = _config(**config_kwargs)
    volume = SimulationVolume(
        partial_grid_shape=(cells, cells, 1), material=Material(permittivity=1.0), name=f"vol{tag}"
    )
    boundaries, constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="periodic"), volume
    )
    objects = [volume, shape_fn(eps_core, tag, cells), *boundaries.values()]
    container, arrays, _, config, info = fdtdx.place_objects(objects, config, constraints)
    return container, arrays, config, info


# ---------------------------------------------------------------------------
# (b) The assembled D-to-E map
# ---------------------------------------------------------------------------


def _dense_dtoe(arrays, periodic, shape) -> np.ndarray:
    """Assemble the D-to-E map column by column from the loader's own arrays."""
    inv_eps = np.asarray(arrays.inv_permittivities, dtype=np.float64)
    offdiag = np.asarray(arrays.inv_permittivity_offdiag, dtype=np.float64)
    size = 3 * int(np.prod(shape))
    dense = np.zeros((size, size))
    for column in range(size):
        basis = np.zeros(size)
        basis[column] = 1.0
        dense[:, column] = apply_dtoe_map(inv_eps, offdiag, basis.reshape((3, *shape)), periodic).reshape(-1)
    return dense


def _inplane_double_curl(cells: int) -> np.ndarray:
    """``K = C^T C`` for the in-plane ``(E_x, E_y)`` block on a periodic square, ``h = 1``.

    ``(C E)[i,j] = E_y[i+1,j] - E_y[i,j] - E_x[i,j+1] + E_x[i,j]`` is the discrete curl that produces
    ``H_z`` at the cell centre, which is where fdtdx's ``H_z`` sits. ``K`` is symmetric positive
    semi-definite for any real ``C``, so the squared frequencies of the semi-discrete system
    ``d2E/dt2 = -M K E`` are the eigenvalues of ``M K``.
    """
    unknowns = 2 * cells * cells

    def index(component, i, j):
        return component * cells * cells + (i % cells) * cells + (j % cells)

    curl = np.zeros((cells * cells, unknowns))
    for i in range(cells):
        for j in range(cells):
            row = i * cells + j
            curl[row, index(1, i + 1, j)] += 1.0
            curl[row, index(1, i, j)] -= 1.0
            curl[row, index(0, i, j + 1)] -= 1.0
            curl[row, index(0, i, j)] += 1.0
    return curl.T @ curl


@pytest.mark.parametrize("eps_core", [6.25, 12.1, 20.0, 30.0])
@pytest.mark.parametrize("shape_fn, geometry", [(_disk, "disk"), (_tilted_slab, "tilted slab")])
def test_the_assembled_map_is_symmetric_and_its_spectrum_is_real(eps_core, shape_fn, geometry):
    """Symmetry to round-off, and no growing mode, at every contrast up to 30.

    This is the static test that decided the placement: assemble the constitutive map ``M`` and the
    discrete double curl ``K``, then read ``||M - M^T||`` and the spectrum of ``M K``. A complex
    ``omega^2`` is a mode that grows and oscillates; a negative real one grows without oscillating.
    Both are absent here. Under the pixel placement the same disk puts complex pairs into the
    spectrum already at ``eps = 6.25``.
    """
    _, arrays, _, _ = _periodic_scene(shape_fn, eps_core)
    shape = tuple(int(n) for n in np.asarray(arrays.inv_permittivities).shape[1:])
    dense = _dense_dtoe(arrays, (True, True, True), shape)

    asymmetry = np.linalg.norm(dense - dense.T, "fro") / np.linalg.norm(dense, "fro")
    assert asymmetry < 1e-14, f"{geometry} eps={eps_core}: relative asymmetry {asymmetry:.3e}"

    # The 2-D layout leaves n_z = 0, so the E_z row decouples; the in-plane block carries the physics.
    inplane = np.arange(2 * shape[0] * shape[1])
    block = dense[np.ix_(inplane, inplane)]
    spectrum = np.linalg.eigvals(block @ _inplane_double_curl(shape[0]))
    scale = float(np.max(np.abs(spectrum)))
    tolerance = 1e-9 * scale
    complex_pairs = int(np.count_nonzero(np.abs(spectrum.imag) > tolerance))
    negative_real = int(np.count_nonzero((spectrum.real < -tolerance) & (np.abs(spectrum.imag) <= tolerance)))
    assert complex_pairs == 0, f"{geometry} eps={eps_core}: {complex_pairs} complex omega^2"
    assert negative_real == 0, f"{geometry} eps={eps_core}: {negative_real} negative real omega^2"


def test_the_definiteness_check_separates_silicon_from_the_high_contrast_failure():
    """The build-time Lanczos estimate: positive at silicon contrast, negative at eps = 60.

    Symmetry is necessary and not sufficient. The map stays exactly symmetric at every contrast, but
    at ``eps = 60`` on a curved rim its symmetric part loses positive definiteness and a purely
    growing mode becomes possible. The check is what turns that from a surprise into a warning.

    Run on a 24-cell box rather than the 16 the spectrum tests use: the sign of the smallest
    eigenvalue at ``eps = 60`` is a property of the formula and not of the mesh — it was measured
    negative at 32, 48, 64, 96 and 128 cells — but the rim of a disk 4.8 cells in radius is too
    coarse to resolve the configuration that produces it.
    """
    cells = 24
    _, silicon, _, info_silicon = _periodic_scene(_disk, 12.1, cells=cells, yee_smooth_check_definiteness=True)
    with pytest.warns(UserWarning, match="smallest eigenvalue"):
        _, high, _, info_high = _periodic_scene(_disk, 60.0, cells=cells, yee_smooth_check_definiteness=True)

    low = info_silicon["yee_sampling_difference"]["smoothing"]
    top = info_high["yee_sampling_difference"]["smoothing"]
    assert low["min_eig_sym_dtoe"] > 0.0
    assert low["min_eig_sym_dtoe_positive"] is True
    assert top["min_eig_sym_dtoe"] < 0.0
    assert top["min_eig_sym_dtoe_positive"] is False
    # Both maps are exactly symmetric — the difference is definiteness, not symmetry.
    assert low["asym_rel_dtoe"] < 1e-12
    assert top["asym_rel_dtoe"] < 1e-12

    # And the estimate is the real smallest eigenvalue, not an artefact of the Lanczos solve.
    shape = tuple(int(n) for n in np.asarray(high.inv_permittivities).shape[1:])
    dense = _dense_dtoe(high, (True, True, True), shape)
    exact = float(np.linalg.eigvalsh(0.5 * (dense + dense.T)).min())
    assert top["min_eig_sym_dtoe"] == pytest.approx(exact, rel=1e-6, abs=1e-9)
    del silicon


def test_the_min_eigenvalue_helper_reports_a_positive_value_without_the_correction():
    """With no vertex entries the map is the diagonal multiply, whose eigenvalues are its entries."""
    inv_eps = np.full((3, 4, 4, 2), 0.25)
    inv_eps[1, 2, 2, 1] = 0.125
    estimate = min_eigenvalue_of_symmetric_part(inv_eps, np.zeros_like(inv_eps), (False, False, False))
    assert estimate["min_eig_sym_dtoe"] == pytest.approx(0.125, rel=1e-6)
    assert estimate["max_eig_sym_dtoe"] == pytest.approx(0.25, rel=1e-6)
    assert estimate["asym_rel_dtoe"] < 1e-12


def test_the_vertex_lattice_matches_a_hand_computed_fill_and_normal_at_silicon_contrast(float64):
    """Pins the vertex dual box directly, at eps = 12.1 (silicon/air), where F5 found a gap.

    ``test_the_assembled_map_is_symmetric_and_its_spectrum_is_real`` would still pass if
    ``smooth_offdiagonal_on_vertex_lattice`` sampled the ``E_x`` component pixel instead of the
    cell-vertex dual box: both coupled rows would still read one *consistent* box, so the map
    stays exactly symmetric, and at eps = 12.1 it stays positive definite too (the mutation needs
    eps = 20 before the spectrum test notices). Only the number actually written at a vertex tells
    the two boxes apart, so this test computes that number by hand and checks it directly.

    The scene is a tilted rectangle of eps = 12.1 in eps = 1, one straight interface. At a handful
    of vertices near the middle of its long edge -- several cells from any other edge, corner, or
    the domain's periodic seam -- the fill fraction is exactly what one straight line cuts off a
    rectangle, so it is read here from the shape's own ``box_fill_fraction`` / ``normal_at``
    (simple, independently trustworthy primitives) against box bounds built from nothing but the
    grid edges and dx -- NOT from the loader's own ``pixel_axis_bounds``, so a wrong field-code
    choice inside the function under test has nowhere to hide. The ``E_x`` component box (same as
    the vertex box on every axis except a dx/2 shift along x, matching ``pixel_axis_bounds``'s own
    ``offsets[axis] == 0.5`` branch) is computed the same way, to confirm it would give a clearly
    different, not merely rounded, answer.
    """
    container, arrays, config, _ = _periodic_scene(_silicon_air_edge_slab, 12.1, cells=_N, dtype=jnp.float64)
    obj = next(o for o in container.object_list if o.name.startswith("edge"))
    grid = config.resolved_grid
    edges_x = np.asarray(grid.edges(0))
    edges_y = np.asarray(grid.edges(1))
    edges_z = np.asarray(grid.edges(2))
    z_lo, z_hi = float(edges_z[0]), float(edges_z[-1])
    eps_core, eps_bg = 12.1, 1.0

    def fill_of(lower, upper):
        result = obj.box_fill_fraction(np.asarray([lower]), np.asarray([upper]))
        assert result is not None
        return float(result[0])

    def offdiag_xy(fill: float, normal: np.ndarray) -> float:
        arithmetic = fill * eps_core + (1.0 - fill) * eps_bg  # <eps>
        harmonic = fill / eps_core + (1.0 - fill) / eps_bg  # <1/eps>
        return float(normal[0] * normal[1] * (harmonic - 1.0 / arithmetic))

    entries = np.asarray(arrays.inv_permittivity_offdiag, dtype=np.float64)
    # Three vertices along the tilted edge (checked once, offline, against this same scene): the
    # vertex box and the E_x component box give clearly different fills at every one of them.
    for i, j in ((4, 5), (7, 6), (10, 7)):
        xi, yj = float(edges_x[i]), float(edges_y[j])
        fill_vertex = fill_of((xi - 0.5 * _D, yj - 0.5 * _D, z_lo), (xi + 0.5 * _D, yj + 0.5 * _D, z_hi))
        fill_component = fill_of((xi, yj - 0.5 * _D, z_lo), (xi + _D, yj + 0.5 * _D, z_hi))
        assert 0.05 < fill_vertex < 0.95, "the probe should be a genuine interface vertex, not a corner"
        assert abs(fill_vertex - fill_component) > 0.15, "the two boxes must disagree substantially here"

        normal = np.asarray(obj.normal_at(np.asarray([[xi, yj, 0.5 * (z_lo + z_hi)]]), ignore_axes=(2,)))[0]
        normal = normal / np.linalg.norm(normal)
        assert normal[2] == 0.0  # extruded along z, invariant there

        expected_xy = offdiag_xy(fill_vertex, normal)
        wrong_xy = offdiag_xy(fill_component, normal)  # what the F5 mutation would write instead
        assert abs(wrong_xy - expected_xy) > 0.03, "the wrong box should not be a rounding-level change"

        assert entries[0, i, j, 0] == pytest.approx(expected_xy, abs=1e-9), f"xy at vertex ({i}, {j})"
        assert entries[1, i, j, 0] == 0.0  # xz: normal has no z component
        assert entries[2, i, j, 0] == 0.0  # yz


# ---------------------------------------------------------------------------
# (d) An axis-aligned scene produces no entries
# ---------------------------------------------------------------------------


def test_an_axis_aligned_box_yields_an_all_zero_vertex_array():
    """``A_ij = n_i n_j (<1/eps> - 1/<eps>)`` vanishes exactly when the normal lies on an axis.

    So the whole hybrid route's snapping half costs nothing here: a Manhattan scene allocates the
    vertex array and leaves every entry at zero, and the update's correction is a no-op even though
    it runs.
    """
    tag = _tag()
    config = _config()
    volume = SimulationVolume(partial_grid_shape=(12, 12, 4), material=Material(permittivity=1.0), name=f"vol{tag}")
    slab = UniformMaterialObject(
        material=Material(permittivity=6.25),
        partial_real_shape=(5.4 * _D, None, None),  # a deliberately sub-cell width
        partial_real_position=(0.0, 0.0, 0.0),
        placement_order=1,
        name=f"slab{tag}",
    )
    _, arrays, _, _, info = fdtdx.place_objects([volume, slab], config, [])

    entries = np.asarray(arrays.inv_permittivity_offdiag, dtype=np.float64)
    assert entries.shape == (3, 12, 12, 4)
    assert np.count_nonzero(entries) == 0
    # The pass really ran: it found the interface and blended it, it just had nothing to write.
    assert info["yee_sampling_difference"]["smoothing_offdiag"]["num_smoothed"] > 0


# ---------------------------------------------------------------------------
# (c) Bit-identity when the entries are absent
# ---------------------------------------------------------------------------


def _random_fields(arrays, seed: int):
    E = jax.random.normal(jax.random.PRNGKey(seed), arrays.fields.E.shape, dtype=arrays.fields.E.dtype)
    H = jax.random.normal(jax.random.PRNGKey(seed + 1), arrays.fields.H.shape, dtype=arrays.fields.H.dtype)
    return arrays.aset("fields->E", E).aset("fields->H", H)


def test_the_diagonal_tier_and_a_none_array_give_the_same_update_bit_for_bit():
    """Nothing moves unless the loader wrote the entries.

    Two checks in one: ``yee_smooth_full_tensor=False`` never allocates the vertex array, and
    blanking the array on a scene that has one reproduces the diagonal branch exactly. Both compare
    against the same diagonal-tier scene on the same random fields, with ``jnp.array_equal`` rather
    than a tolerance.
    """
    container, node_arrays, config, _ = _periodic_scene(_disk, 6.25)
    diagonal_container, diagonal_arrays, diagonal_config, _ = _periodic_scene(_disk, 6.25, full_tensor=False)

    assert diagonal_arrays.inv_permittivity_offdiag is None
    assert node_arrays.inv_permittivity_offdiag is not None
    # The diagonal entries themselves are untouched by the placement.
    assert bool(jnp.array_equal(node_arrays.inv_permittivities, diagonal_arrays.inv_permittivities))

    node_arrays = _random_fields(node_arrays, 5)
    diagonal_arrays = _random_fields(diagonal_arrays, 5)
    step = jnp.asarray(0)
    diagonal_step = update_E(step, diagonal_arrays, diagonal_container, diagonal_config, simulate_boundaries=False)
    blanked = update_E(
        step,
        node_arrays.aset("inv_permittivity_offdiag", None),
        container,
        config,
        simulate_boundaries=False,
    )
    assert bool(jnp.array_equal(blanked.fields.E, diagonal_step.fields.E))

    # ... and with the entries present it is a different field, so the comparison above has teeth.
    corrected = update_E(step, node_arrays, container, config, simulate_boundaries=False)
    assert not bool(jnp.array_equal(corrected.fields.E, diagonal_step.fields.E))


# ---------------------------------------------------------------------------
# (e) The reverse update
# ---------------------------------------------------------------------------


def test_the_reverse_update_restores_the_field_with_the_correction_present():
    """Forward then reverse returns ``E`` to float32 round-off, correction included.

    The correction is a linear term added to the same increment before the lossy divide, so the
    reverse step subtracts it in the mirrored place. If the reverse branch were left out, the
    residual would be the size of the correction itself, which the second assertion measures.
    """
    container, arrays, config, _ = _periodic_scene(_disk, 12.1)
    arrays = _random_fields(arrays, 9)
    original = arrays.fields.E
    step = jnp.asarray(0)

    forward = update_E(step, arrays, container, config, simulate_boundaries=False)
    restored = update_E_reverse(step, forward, container, config)
    residual = float(jnp.max(jnp.abs(restored.fields.E - original)))
    scale = float(jnp.max(jnp.abs(original)))
    assert residual / scale < 1e-5

    # Size of the term the reverse step had to undo, for contrast with the residual above.
    increment_pad, offdiag_pad, widths = offdiag_correction_terms(
        jnp.zeros_like(original) + config.courant_number, arrays.inv_permittivity_offdiag, container, config
    )
    del increment_pad, offdiag_pad, widths
    blanked = update_E(
        step, arrays.aset("inv_permittivity_offdiag", None), container, config, simulate_boundaries=False
    )
    correction = float(jnp.max(jnp.abs(forward.fields.E - blanked.fields.E)))
    assert correction / scale > 1e-3


# ---------------------------------------------------------------------------
# (g) The 2-D E_z control
# ---------------------------------------------------------------------------


def test_the_two_dimensional_ez_polarization_cannot_be_touched(float64):
    """The ``E_z`` (TM) case is bit-identical between the diagonal tier and the node placement.

    Two independent reasons, and the test pins both.

    First, the entries. fdtdx's 2-D convention is one cell on ``z``, and the loader zeroes the
    normal on every invariant axis, so ``n_z = 0`` at every smoothed vertex. The off-diagonal entry
    is ``n_i n_j (<1/eps> - 1/<eps>)``, so ``xz`` and ``yz`` are exactly zero and only ``xy``
    survives. Row ``z`` reads ``xz`` and ``yz`` alone, so it receives nothing whatever the fields do.

    Second, the fields. In the TM polarization ``H_z`` vanishes and nothing varies along ``z``, so
    ``curl(H)_x = dH_z/dy - dH_y/dz`` and ``curl(H)_y = dH_x/dz - dH_z/dx`` are both zero: the two
    partner increments the surviving ``xy`` entry would multiply are zero too, and ``E_x``/``E_y``
    stay at zero as well. The correction is a no-op on the whole field, not only on its ``z`` row.
    """
    container, arrays, config, _ = _periodic_scene(_disk, 12.1)
    entries = np.asarray(arrays.inv_permittivity_offdiag, dtype=np.float64)
    assert np.count_nonzero(entries[0]) > 0  # xy is alive
    assert np.count_nonzero(entries[1]) == 0  # xz
    assert np.count_nonzero(entries[2]) == 0  # yz

    diagonal_container, diagonal_arrays, diagonal_config, _ = _periodic_scene(_disk, 12.1, full_tensor=False)

    # A TM state: E_z and the in-plane H only.
    shape = arrays.fields.E.shape
    key = jax.random.PRNGKey(21)
    Ez = jax.random.normal(key, shape[1:], dtype=arrays.fields.E.dtype)
    Hx = jax.random.normal(jax.random.PRNGKey(22), shape[1:], dtype=arrays.fields.H.dtype)
    Hy = jax.random.normal(jax.random.PRNGKey(23), shape[1:], dtype=arrays.fields.H.dtype)
    zero = jnp.zeros(shape[1:], dtype=arrays.fields.E.dtype)
    E = jnp.stack([zero, zero, Ez], axis=0)
    H = jnp.stack([Hx, Hy, zero], axis=0)

    step = jnp.asarray(0)
    node = update_E(
        step,
        arrays.aset("fields->E", E).aset("fields->H", H),
        container,
        config,
        simulate_boundaries=False,
    )
    diagonal = update_E(
        step,
        diagonal_arrays.aset("fields->E", E).aset("fields->H", H),
        diagonal_container,
        diagonal_config,
        simulate_boundaries=False,
    )
    assert bool(jnp.array_equal(node.fields.E, diagonal.fields.E))
    # The in-plane rows stayed at zero, which is the second half of the mechanism.
    assert float(jnp.max(jnp.abs(node.fields.E[0]))) == 0.0
    assert float(jnp.max(jnp.abs(node.fields.E[1]))) == 0.0


# ---------------------------------------------------------------------------
# The MLX twin of the stencil
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _mlx_available(), reason="the MLX helper needs mlx installed")
@pytest.mark.parametrize(
    "periodic",
    [(False, False, False), (True, True, True), (True, False, True)],
    ids=["terminated", "periodic", "mixed"],
)
def test_the_mlx_helper_agrees_with_the_jax_one(periodic):
    """The MLX-op core's correction equals the JAX one on random arrays, to float32 round-off.

    Run on the MLX CPU device: this box has no Metal available to the test process, and the
    arithmetic is the same either way. The custom Metal kernels do not carry this term at all —
    ``kernel_eligible`` refuses a run whose state has the vertex array, which the companion
    assertion below pins — so the MLX path this exercises is the one such a run actually takes.
    """
    import mlx.core as mx

    from fdtdx.mlx.update import add_offdiag_correction_mlx

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        rng = np.random.default_rng(31)
        shape = (5, 4, 3)
        entries = rng.standard_normal((3, *shape)).astype(np.float32)
        increment = rng.standard_normal((3, *shape)).astype(np.float32)
        field = rng.standard_normal((3, *shape)).astype(np.float32)

        reference = np.asarray(
            add_offdiag_correction(
                jnp.asarray(field),
                jnp.asarray(_pad_field(increment, periodic)),
                jnp.asarray(_pad_entries(entries, periodic)),
            )
        )
        obtained = np.asarray(
            add_offdiag_correction_mlx(mx.array(field), mx.array(increment), mx.array(entries), periodic)
        )
        assert np.max(np.abs(reference - obtained)) < 2e-6 * max(float(np.max(np.abs(reference))), 1.0)
    finally:
        mx.set_default_device(previous)


@pytest.mark.skipif(not _mlx_available(), reason="the kernel eligibility gate needs mlx installed")
def test_the_metal_kernel_refuses_a_run_carrying_the_vertex_entries():
    """A state with ``inv_eps_offdiag`` drops to the MLX-op cores, which do carry the correction."""
    import mlx.core as mx

    from fdtdx.mlx.kernels import kernel_eligible
    from fdtdx.mlx.state import MLXState

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        shape = (4, 4, 4)
        state = MLXState(
            E=mx.zeros((3, *shape)),
            H=mx.zeros((3, *shape)),
            psi_E=(),
            psi_H=(),
            inv_eps=mx.ones((3, *shape)),
            inv_mu=1.0,
            cpml_a=mx.zeros((6, *shape)),
            cpml_b=mx.zeros((6, *shape)),
            inv_kappa=mx.ones((6, *shape)),
        )
        assert kernel_eligible(state) is True
        state.inv_eps_offdiag = mx.zeros((3, *shape))
        assert kernel_eligible(state) is False
    finally:
        mx.set_default_device(previous)


# ---------------------------------------------------------------------------
# The gates that send a scene back to the dense placement
# ---------------------------------------------------------------------------


def test_a_bulk_tensor_material_falls_back_to_the_dense_placement():
    """The vertex array holds smoothing-induced off-diagonals only.

    A material whose own permittivity tensor has off-diagonal entries has a bulk term that belongs
    at its own cells, and separating the two on one shared array is not defined. Such a scene keeps
    the recorded dense 9-component path, and says so.
    """
    tag = _tag()
    tilted = (6.0, 0.8, 0.0, 0.8, 5.0, 0.0, 0.0, 0.0, 4.0)
    config = _config()
    volume = SimulationVolume(partial_grid_shape=(12, 12, 1), material=Material(permittivity=1.0), name=f"vol{tag}")
    disk = Cylinder(
        axis=2,
        radius=3.0 * _D,
        material_name="core",
        materials={"core": Material(permittivity=tilted)},
        partial_grid_shape=(None, None, 1),
        placement_order=1,
        name=f"disk{tag}",
    )
    with pytest.warns(UserWarning, match="off-diagonal permittivity entries of its own"):
        _, arrays, _, _, _ = fdtdx.place_objects([volume, disk], config, [])
    assert arrays.inv_permittivities.shape[0] == 9
    assert arrays.inv_permittivity_offdiag is None


def test_an_oriented_dispersive_pole_falls_back_to_the_dense_placement():
    """Anything else that widens the permittivity array takes the vertex entries with it.

    Oriented poles force the 9-component tier further down in initialization, and that update reads
    the tensor rows straight out of the array — it never looks at a vertex entry. Writing them would
    allocate an array nothing applies, so the placement follows the tier.
    """
    from fdtdx.dispersion import DispersionModel, LorentzPole

    tag = _tag()
    pole = LorentzPole(resonance_frequency=2e14, damping=1e13, delta_epsilon=2.0, orientation=(1.0, 1.0, 0.0))
    config = _config()
    volume = SimulationVolume(partial_grid_shape=(12, 12, 1), material=Material(permittivity=1.0), name=f"vol{tag}")
    disk = Cylinder(
        axis=2,
        radius=3.0 * _D,
        material_name="core",
        materials={"core": Material(permittivity=6.25, dispersion=DispersionModel(poles=(pole,)))},
        partial_grid_shape=(None, None, 1),
        placement_order=1,
        name=f"disk{tag}",
    )
    with pytest.warns(UserWarning, match="widened to the 9-component tier"):
        _, arrays, _, _, _ = fdtdx.place_objects([volume, disk], config, [])
    assert arrays.inv_permittivities.shape[0] == 9
    assert arrays.inv_permittivity_offdiag is None


def test_a_lossy_vertex_keeps_a_zero_entry_and_is_counted():
    """A conductive interface gets no off-diagonal term, and the loader says how many it skipped.

    The diagonal branch divides by ``1 + c sigma eta0 inv_eps / 2``, a per-component scalar. There
    is no off-diagonal form of that factor without writing the update on ``D``, so rather than scale
    the correction by a factor that does not describe it, the vertex is left at zero and counted.
    """
    tag = _tag()
    config = _config()
    volume = SimulationVolume(partial_grid_shape=(16, 16, 1), material=Material(permittivity=1.0), name=f"vol{tag}")
    disk = Cylinder(
        axis=2,
        radius=0.30 * 16 * _D,
        material_name="core",
        materials={"core": Material(permittivity=6.25, electric_conductivity=0.5)},
        partial_grid_shape=(None, None, 1),
        placement_order=1,
        name=f"disk{tag}",
    )
    _, arrays, _, _, info = fdtdx.place_objects([volume, disk], config, [])
    stats = info["yee_sampling_difference"]["smoothing_offdiag"]
    assert stats["num_lossy_offdiag_skips"] > 0
    assert stats["num_lossy_offdiag_skips"] == stats["num_candidates"]
    assert np.count_nonzero(np.asarray(arrays.inv_permittivity_offdiag)) == 0

    # The lossless twin of the same scene does write entries, so the counter above is not vacuous.
    _, lossless, _, _ = _periodic_scene(_disk, 6.25)
    assert np.count_nonzero(np.asarray(lossless.inv_permittivity_offdiag)) > 0


# ---------------------------------------------------------------------------
# (i) The two placements that take every entry from one family of boxes
# ---------------------------------------------------------------------------


def _explicit_vertex_offdiagonals(rows: np.ndarray, periodic) -> np.ndarray:
    """``vertex_offdiagonals_from_pixel_rows``, written as scalar lookups with Python loops.

    The mean of the four component pixels adjacent to a vertex in the entry's own plane. Deliberately
    index by index, with the out-of-range rule spelled out, so it shares no code with the vectorised
    helper.
    """
    shape = rows.shape[1:]
    out = np.zeros((3, *shape))

    def sample(plane: int, index) -> float:
        wrapped = list(index)
        for axis in range(3):
            if 0 <= wrapped[axis] < shape[axis]:
                continue
            wrapped[axis] = (
                wrapped[axis] % shape[axis] if periodic[axis] else min(max(wrapped[axis], 0), shape[axis] - 1)
            )
        return float(rows[(plane, *wrapped)])

    for entry, (i, j) in enumerate(OFFDIAGONAL_ENTRIES):
        for a in range(shape[0]):
            for b in range(shape[1]):
                for c in range(shape[2]):
                    here = [a, b, c]
                    back_i = list(here)
                    back_i[i] -= 1
                    back_j = list(here)
                    back_j[j] -= 1
                    out[(entry, *here)] = 0.25 * (
                        sample(3 * i + j, here)
                        + sample(3 * i + j, back_i)
                        + sample(3 * j + i, here)
                        + sample(3 * j + i, back_j)
                    )
    return out


def _explicit_pixel_diagonals(vertex: np.ndarray, periodic) -> np.ndarray:
    """``pixel_diagonals_from_vertex_tensor``, written as scalar lookups with Python loops."""
    shape = vertex.shape[1:]
    out = np.zeros((3, *shape))

    def sample(axis: int, index) -> float:
        wrapped = list(index)
        for a in range(3):
            if 0 <= wrapped[a] < shape[a]:
                continue
            wrapped[a] = wrapped[a] % shape[a] if periodic[a] else min(max(wrapped[a], 0), shape[a] - 1)
        return float(vertex[(axis, *wrapped)])

    for axis in range(3):
        for a in range(shape[0]):
            for b in range(shape[1]):
                for c in range(shape[2]):
                    here = [a, b, c]
                    ahead = list(here)
                    ahead[axis] += 1
                    out[(axis, *here)] = 0.5 * (sample(axis, here) + sample(axis, ahead))
    return out


@pytest.mark.parametrize(
    "periodic",
    [(False, False, False), (True, True, True), (True, False, True)],
    ids=["terminated", "periodic", "mixed"],
)
def test_the_two_re_placement_helpers_are_plain_means_of_their_neighbours(periodic):
    """Both helpers equal an index-by-index transcription on random arrays, at every boundary.

    They are the whole difference between the three vertex placements: ``node_avg`` averages four
    component pixels onto a vertex, ``vertex_all`` averages two vertices onto a component point. A
    missing neighbour at a terminated face replicates the edge value, which is the same halo rule the
    update's coefficient padding uses.
    """
    rng = np.random.default_rng(1401)
    shape = (5, 4, 3)
    rows = rng.standard_normal((9, *shape))
    vertex = rng.standard_normal((6, *shape))

    obtained = vertex_offdiagonals_from_pixel_rows(rows, periodic)
    assert np.max(np.abs(obtained - _explicit_vertex_offdiagonals(rows, periodic))) < 1e-14

    obtained = pixel_diagonals_from_vertex_tensor(vertex, periodic)
    assert np.max(np.abs(obtained - _explicit_pixel_diagonals(vertex, periodic))) < 1e-14


@pytest.mark.parametrize("placement", ["node_avg", "vertex_all"])
@pytest.mark.parametrize("eps_core", [6.25, 12.1, 30.0])
@pytest.mark.parametrize("shape_fn, geometry", [(_disk, "disk"), (_tilted_slab, "tilted slab")])
def test_the_two_variants_assemble_a_symmetric_map_with_a_real_spectrum(placement, eps_core, shape_fn, geometry):
    """The reason they exist at all: one shared vertex array, so the map is its own transpose.

    Same assertion as the ``"node"`` test above. Both variants keep the permittivity on the
    3-component tier and write one shared ``(xy, xz, yz)`` array, so nothing about the symmetry
    argument changes when the entries are computed from different boxes.
    """
    _, arrays, _, _ = _periodic_scene(shape_fn, eps_core, eps_placement=placement)
    assert arrays.inv_permittivities.shape[0] == 3
    assert arrays.inv_permittivity_offdiag is not None
    shape = tuple(int(n) for n in np.asarray(arrays.inv_permittivities).shape[1:])
    dense = _dense_dtoe(arrays, (True, True, True), shape)

    asymmetry = np.linalg.norm(dense - dense.T, "fro") / np.linalg.norm(dense, "fro")
    assert asymmetry < 1e-14, f"{placement} {geometry} eps={eps_core}: asymmetry {asymmetry:.3e}"

    inplane = np.arange(2 * shape[0] * shape[1])
    block = dense[np.ix_(inplane, inplane)]
    spectrum = np.linalg.eigvals(block @ _inplane_double_curl(shape[0]))
    scale = float(np.max(np.abs(spectrum)))
    tolerance = 1e-9 * scale
    assert int(np.count_nonzero(np.abs(spectrum.imag) > tolerance)) == 0
    assert int(np.count_nonzero((spectrum.real < -tolerance) & (np.abs(spectrum.imag) <= tolerance))) == 0


@pytest.mark.parametrize("eps_core", [6.25, 12.1])
def test_node_avg_leaves_the_diagonal_entries_exactly_where_node_leaves_them(eps_core):
    """``node_avg`` moves only the off-diagonal entries; the diagonal array does not move at all.

    It reaches the diagonal through the 9-component Kottke row rather than the diagonal-entry call,
    so this pins that the two really are the same number and not merely close: entry ``(c, c)`` of
    the row is ``n_c n_c <1/eps> + (1 - n_c n_c) / <eps>``, which is what the diagonal tier writes.
    """
    _, node, _, _ = _periodic_scene(_disk, eps_core, eps_placement="node")
    _, averaged, _, _ = _periodic_scene(_disk, eps_core, eps_placement="node_avg")
    assert np.array_equal(np.asarray(node.inv_permittivities), np.asarray(averaged.inv_permittivities))
    # The entries themselves do move: the vertex value is a mean over four pixels, so its peak is
    # smaller than the value the vertex box produces on its own.
    node_entries = np.asarray(node.inv_permittivity_offdiag)
    averaged_entries = np.asarray(averaged.inv_permittivity_offdiag)
    assert np.max(np.abs(averaged_entries)) < np.max(np.abs(node_entries))
    assert np.count_nonzero(averaged_entries) > 0


@pytest.mark.parametrize("eps_core", [6.25, 12.1])
def test_vertex_all_keeps_nodes_entries_and_rebuilds_the_diagonal_from_them(eps_core):
    """``vertex_all`` computes the same vertex box, so its off-diagonal entries are ``node``'s.

    What changes is the diagonal: it is the mean of the two vertices bracketing each component point
    along that component's own axis, and no longer the value of the component's own pixel. On a
    2-D scene the ``E_z`` pixel *is* the vertex box on both in-plane axes and degenerate on the third,
    so that one component comes out unmoved — a check that the box bookkeeping is right.
    """
    _, node, _, _ = _periodic_scene(_disk, eps_core, eps_placement="node")
    _, whole, _, _ = _periodic_scene(_disk, eps_core, eps_placement="vertex_all")
    assert np.array_equal(np.asarray(node.inv_permittivity_offdiag), np.asarray(whole.inv_permittivity_offdiag))

    node_diagonal = np.asarray(node.inv_permittivities)
    whole_diagonal = np.asarray(whole.inv_permittivities)
    assert np.array_equal(node_diagonal[2], whole_diagonal[2])
    assert not np.array_equal(node_diagonal[0], whole_diagonal[0])
    assert not np.array_equal(node_diagonal[1], whole_diagonal[1])

    # The tent-shaped average spreads the rim over one more cell on each in-plane axis, so more
    # pixels differ from vacuum than under the pixel-placed diagonal, and the extremes stay inside
    # the two bulk values.
    for axis in (0, 1):
        assert np.count_nonzero(whole_diagonal[axis] != 1.0) > np.count_nonzero(node_diagonal[axis] != 1.0)
        assert whole_diagonal[axis].min() >= node_diagonal[axis].min() - 1e-12
        assert whole_diagonal[axis].max() <= 1.0 + 1e-12


def test_every_placement_name_is_accepted_and_an_unknown_one_is_not():
    """The four values the config takes, and the error for anything else."""
    for placement in ("node", "node_avg", "vertex_all", "pixel"):
        assert _config(eps_placement=placement).yee_smooth_offdiag_placement == placement
    with pytest.raises(ValueError, match="yee_smooth_offdiag_placement"):
        _config(eps_placement="vertex")
