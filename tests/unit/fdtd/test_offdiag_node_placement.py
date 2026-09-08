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
from fdtdx.core.physics.geometry_smooth import apply_dtoe_map, min_eigenvalue_of_symmetric_part
from fdtdx.fdtd.misc import OFFDIAG_ROW_PARTNERS, add_offdiag_correction
from fdtdx.fdtd.update import offdiag_correction_terms, update_E, update_E_reverse
from fdtdx.materials import Material
from fdtdx.objects.static_material.cylinder import Cylinder
from fdtdx.objects.static_material.polygon import ExtrudedPolygon
from fdtdx.objects.static_material.static import SimulationVolume, UniformMaterialObject

_COUNTER = [0]


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
