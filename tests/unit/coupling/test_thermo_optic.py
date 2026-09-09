"""The thermo-optic perturbation applied after the interface blend.

The checks pin, in order: that a uniform temperature is indistinguishable from writing the
perturbed indices into the materials before loading (bulk points *and* every blended pixel and
vertex, so the re-blend reproduces the loader's own blend); that a spatially varying temperature
gives the closed form at bulk points and the Kottke formulas with local permittivities at the
recorded pixels; that an uncovered point is an error by default and a counted no-op on request;
that the unsupported tiers and material kinds are refused rather than approximated; that the
record the loader exposes agrees with its own counters; and that the sampled-field artefact
round-trips through its file format.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.core.physics.geometry_smooth import OFFDIAGONAL_ENTRIES
from fdtdx.coupling import (
    ThermoOpticCoefficients,
    YeeLatticeSamples,
    apply_thermo_optic_perturbation,
    perturb_arrays,
    perturbed_permittivity,
    samples_from_callable,
    uniform_samples,
)
from fdtdx.materials import Material
from fdtdx.objects.static_material.cylinder import Cylinder
from fdtdx.objects.static_material.static import SimulationVolume

_D = 25e-9
_N = 24
_COUNTER = [0]

T_REF = 293.15
DN_CORE = 1.8e-4  # silicon-like, 1/K
DN_BG = 1.0e-5  # oxide-like, 1/K


@pytest.fixture
def float64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _tag() -> str:
    _COUNTER[0] += 1
    return f"_{_COUNTER[0]}"


def _config(placement: str = "node", full_tensor: bool = True, sampling: str = "yee_smooth") -> SimulationConfig:
    return SimulationConfig(
        time=1e-15,
        grid=UniformGrid(spacing=_D),
        dtype=jnp.float64,
        material_sampling=sampling,
        yee_smooth_full_tensor=full_tensor,
        yee_smooth_offdiag_placement=placement,
    )


def _scene(eps_core: float, eps_bg: float, config: SimulationConfig, cells: int = _N):
    """A disk in a periodic 2-D box, through the fork's loader; returns everything a test needs."""
    tag = _tag()
    materials = {"bg": Material(permittivity=eps_bg), "core": Material(permittivity=eps_core)}
    volume = SimulationVolume(partial_grid_shape=(cells, cells, 1), material=materials["bg"], name=f"vol{tag}")
    disk = Cylinder(
        axis=2,
        radius=0.30 * cells * _D,
        material_name="core",
        materials=materials,
        partial_grid_shape=(None, None, 1),
        placement_order=1,
        name=f"disk{tag}",
    )
    boundaries, constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="periodic"), volume
    )
    container, arrays, _, config, info = fdtdx.place_objects([volume, disk, *boundaries.values()], config, constraints)
    return container, arrays, config, info, materials


def _coefficients(dn_core: float = DN_CORE, dn_bg: float = DN_BG) -> ThermoOpticCoefficients:
    return ThermoOpticCoefficients(dn_dT={"core": dn_core, "bg": dn_bg}, reference_temperature=T_REF)


def _index(eps: float, dn: float, dT: float) -> float:
    return float((np.sqrt(eps) + dn * dT) ** 2)


# ---------------------------------------------------------------------------
# (a) A uniform temperature is a material change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("placement, full_tensor", [("node", True), ("node", False)])
def test_a_uniform_temperature_reproduces_the_pre_perturbed_scene_at_every_entry(float64, placement, full_tensor):
    eps_core, eps_bg = 12.1, 2.085
    dT = 47.0
    _, arrays, config, info, materials = _scene(eps_core, eps_bg, _config(placement, full_tensor))
    grid = config.resolved_grid
    samples = uniform_samples(grid, T_REF + dT)
    perturbed, report = perturb_arrays(arrays, info, materials, samples, _coefficients())

    # The same scene drawn with the perturbed indices: the loader blends it itself.
    _, reference, _, _, _ = _scene(
        _index(eps_core, DN_CORE, dT), _index(eps_bg, DN_BG, dT), _config(placement, full_tensor)
    )

    got = np.asarray(perturbed.inv_permittivities, dtype=np.float64)
    want = np.asarray(reference.inv_permittivities, dtype=np.float64)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)
    assert not np.allclose(got, np.asarray(arrays.inv_permittivities, dtype=np.float64))
    if full_tensor:
        assert perturbed.inv_permittivity_offdiag is not None
        got_off = np.asarray(perturbed.inv_permittivity_offdiag, dtype=np.float64)
        want_off = np.asarray(reference.inv_permittivity_offdiag, dtype=np.float64)
        np.testing.assert_allclose(got_off, want_off, rtol=1e-12, atol=1e-18)
        assert np.count_nonzero(got_off) > 0
        assert report.num_reblended["V"] == info["yee_sampling_difference"]["smoothing_offdiag"]["num_smoothed"]
    assert (
        sum(report.num_reblended.get(f"E{c}", 0) for c in range(3))
        == info["yee_sampling_difference"]["smoothing"]["num_smoothed"]
    )
    assert report.num_uncovered == {} or all(v == 0 for v in report.num_uncovered.values())
    assert report.max_delta_T == pytest.approx(dT)
    assert report.max_delta_n == pytest.approx(DN_CORE * dT)


def test_a_zero_temperature_change_is_the_identity(float64):
    _, arrays, config, info, materials = _scene(12.1, 2.085, _config())
    samples = uniform_samples(config.resolved_grid, T_REF)
    perturbed, _ = perturb_arrays(arrays, info, materials, samples, _coefficients())
    np.testing.assert_array_equal(np.asarray(perturbed.inv_permittivities), np.asarray(arrays.inv_permittivities))
    np.testing.assert_array_equal(
        np.asarray(perturbed.inv_permittivity_offdiag), np.asarray(arrays.inv_permittivity_offdiag)
    )


# ---------------------------------------------------------------------------
# (b) A spatially varying temperature: bulk closed form, local Kottke at the interface
# ---------------------------------------------------------------------------


def test_a_gradient_field_gives_the_closed_form_in_the_bulk_and_local_kottke_at_the_interface(float64):
    eps_core, eps_bg = 12.1, 2.085
    _, arrays, config, info, materials = _scene(eps_core, eps_bg, _config())
    grid = config.resolved_grid
    span = _N * _D
    field = lambda p: T_REF + 60.0 * (p[:, 0] / span) + 25.0 * (p[:, 1] / span) ** 2  # noqa: E731
    samples = samples_from_callable(grid, field)
    inv_eps, offdiag, report = apply_thermo_optic_perturbation(
        np.asarray(arrays.inv_permittivities),
        np.asarray(arrays.inv_permittivity_offdiag),
        info["yee_material_map"],
        materials,
        samples,
        _coefficients(),
    )
    front_E = info["yee_material_map"]["front_E"]
    table = info["yee_material_map"]["material_table"]
    record = info["yee_material_map"]["smoothing_record"]
    # Table entries by value: the volume's copy of the background carries a synthetic name.
    names = tuple("core" if m.permittivity[0] == eps_core else "bg" for m in table)
    eps_of = {"core": eps_core, "bg": eps_bg}
    dn_of = {"core": DN_CORE, "bg": DN_BG}

    for c in range(3):
        blended = np.zeros(front_E.shape[1:], dtype=bool)
        entry = record.lattice("E", c)
        if entry is not None:
            blended[tuple(entry.cells.T)] = True
        bulk = ~blended
        T = samples.values[f"E{c}"]
        for index, name in enumerate(names):
            sel = bulk & (front_E[c] == index)
            if not sel.any():
                continue
            expected = 1.0 / perturbed_permittivity(eps_of[name], dn_of[name], T[sel] - T_REF)
            np.testing.assert_allclose(inv_eps[c][sel], expected, rtol=1e-13)
        if entry is None:
            continue
        # The interface pixels, from the documented formula n n^T <1/eps> + (I - n n^T) / <eps>
        # with both permittivities taken at the pixel's temperature.
        idx = tuple(entry.cells.T)
        dT = T[idx] - T_REF
        e_hi = perturbed_permittivity(
            np.array([eps_of[names[m]] for m in entry.material_hi]),
            np.array([dn_of[names[m]] for m in entry.material_hi]),
            dT,
        )
        e_lo = perturbed_permittivity(
            np.array([eps_of[names[m]] for m in entry.material_lo]),
            np.array([dn_of[names[m]] for m in entry.material_lo]),
            dT,
        )
        f = entry.fill
        mean = f * e_hi + (1 - f) * e_lo
        mean_inv = f / e_hi + (1 - f) / e_lo
        n_c = entry.normal[:, c]
        expected = n_c**2 * mean_inv + (1 - n_c**2) / mean
        np.testing.assert_allclose(inv_eps[c][idx], expected, rtol=1e-13)
        # A pixel sitting exactly at the reference temperature is skipped (identity), not counted.
        assert report.num_reblended[f"E{c}"] == int(np.count_nonzero(dT != 0.0))

    vertex = record.lattice("V", 0)
    assert vertex is not None and vertex.num_pixels > 0
    idx = tuple(vertex.cells.T)
    dT = samples.values["V"][idx] - T_REF
    e_hi = perturbed_permittivity(
        np.array([eps_of[names[m]] for m in vertex.material_hi]),
        np.array([dn_of[names[m]] for m in vertex.material_hi]),
        dT,
    )
    e_lo = perturbed_permittivity(
        np.array([eps_of[names[m]] for m in vertex.material_lo]),
        np.array([dn_of[names[m]] for m in vertex.material_lo]),
        dT,
    )
    f = vertex.fill
    gap = (f / e_hi + (1 - f) / e_lo) - 1.0 / (f * e_hi + (1 - f) * e_lo)
    for q, (i, j) in enumerate(OFFDIAGONAL_ENTRIES):
        np.testing.assert_allclose(
            offdiag[q][idx], vertex.normal[:, i] * vertex.normal[:, j] * gap, rtol=1e-12, atol=1e-18
        )
    # Nothing else moved on the vertex lattice.
    untouched = np.ones(offdiag.shape[1:], dtype=bool)
    untouched[idx] = False
    np.testing.assert_array_equal(offdiag[:, untouched], np.asarray(arrays.inv_permittivity_offdiag)[:, untouched])


# ---------------------------------------------------------------------------
# (c) Coverage
# ---------------------------------------------------------------------------


def test_an_uncovered_perturbed_point_is_an_error_by_default_and_a_counted_no_op_on_request(float64):
    _, arrays, config, info, materials = _scene(12.1, 2.085, _config())
    samples = uniform_samples(config.resolved_grid, T_REF + 30.0)
    # Blank a stripe through the disk on every lattice.
    for lattice in samples.lattices:
        samples.covered[lattice][:, _N // 2 - 1 : _N // 2 + 1, :] = False
        samples.values[lattice][~samples.covered[lattice]] = np.nan

    with pytest.raises(ValueError, match="outside the temperature mesh"):
        perturb_arrays(arrays, info, materials, samples, _coefficients())

    perturbed, report = perturb_arrays(arrays, info, materials, samples, _coefficients(), uncovered="unperturbed")
    got = np.asarray(perturbed.inv_permittivities)
    before = np.asarray(arrays.inv_permittivities)
    stripe = np.zeros(before.shape[1:], dtype=bool)
    stripe[:, _N // 2 - 1 : _N // 2 + 1, :] = True
    np.testing.assert_array_equal(got[:, stripe], before[:, stripe])
    assert not np.allclose(got[:, ~stripe], before[:, ~stripe])
    assert all(v > 0 for v in report.num_uncovered.values())
    assert report.num_uncovered["E0"] + report.num_uncovered["E1"] + report.num_uncovered["E2"] >= 3 * 2 * _N
    assert not np.isnan(got).any()
    assert not np.isnan(np.asarray(perturbed.inv_permittivity_offdiag)).any()


def test_a_material_without_a_coefficient_needs_no_coverage(float64):
    _, arrays, config, info, materials = _scene(12.1, 2.085, _config())
    samples = uniform_samples(config.resolved_grid, T_REF + 30.0)
    front_E = info["yee_material_map"]["front_E"]
    names = info["yee_material_map"]["material_names"]
    bg = names.index("bg")
    # Coverage only where the core was sampled: the background carries no coefficient.
    for c in range(3):
        samples.covered[f"E{c}"][:] = front_E[c] != bg
    # ... and the vertices the record lists (those involve the core).
    record = info["yee_material_map"]["smoothing_record"]
    vertex = record.lattice("V", 0)
    samples.covered["V"][:] = False
    samples.covered["V"][tuple(vertex.cells.T)] = True
    _, report = perturb_arrays(arrays, info, materials, samples, ThermoOpticCoefficients({"core": DN_CORE}, T_REF))
    assert all(v == 0 for v in report.num_uncovered.values())
    assert "bg" not in report.coefficients


# ---------------------------------------------------------------------------
# (d) Refusals
# ---------------------------------------------------------------------------


def test_the_nine_component_tier_is_refused(float64):
    _, arrays, config, info, materials = _scene(12.1, 2.085, _config(placement="pixel", full_tensor=True))
    assert info["yee_material_map"]["num_perm_components"] == 9
    with pytest.raises(NotImplementedError, match="3-component"):
        perturb_arrays(arrays, info, materials, uniform_samples(config.resolved_grid, T_REF + 1.0), _coefficients())


def test_an_anisotropic_perturbed_material_is_refused(float64):
    tag = _tag()
    config = _config()
    materials = {"bg": Material(permittivity=2.085), "core": Material(permittivity=(12.1, 12.1, 11.0))}
    volume = SimulationVolume(partial_grid_shape=(_N, _N, 1), material=materials["bg"], name=f"vol{tag}")
    disk = Cylinder(
        axis=2,
        radius=0.3 * _N * _D,
        material_name="core",
        materials=materials,
        partial_grid_shape=(None, None, 1),
        placement_order=1,
        name=f"disk{tag}",
    )
    boundaries, constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="periodic"), volume
    )
    _, arrays, _, config, info = fdtdx.place_objects([volume, disk, *boundaries.values()], config, constraints)
    with pytest.raises(NotImplementedError, match="anisotropic"):
        perturb_arrays(arrays, info, materials, uniform_samples(config.resolved_grid, T_REF + 1.0), _coefficients())


def test_two_materials_of_one_value_class_must_share_a_coefficient(float64):
    """Two names, one permittivity: the loader blends them as one class, so one dn/dT."""
    tag = _tag()
    config = _config()
    materials = {"bg": Material(permittivity=2.085), "core": Material(permittivity=2.085)}
    volume = SimulationVolume(partial_grid_shape=(_N, _N, 1), material=materials["bg"], name=f"vol{tag}")
    disk = Cylinder(
        axis=2,
        radius=0.3 * _N * _D,
        material_name="core",
        materials=materials,
        partial_grid_shape=(None, None, 1),
        placement_order=1,
        name=f"disk{tag}",
    )
    boundaries, constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="periodic"), volume
    )
    _, arrays, _, config, info = fdtdx.place_objects([volume, disk, *boundaries.values()], config, constraints)
    with pytest.raises(ValueError, match="same value"):
        perturb_arrays(
            arrays,
            info,
            materials,
            uniform_samples(config.resolved_grid, T_REF + 1.0),
            ThermoOpticCoefficients({"core": 1e-4, "bg": 2e-4}, T_REF),
        )
    # The same coefficient is fine (and the class carries no interface, so nothing is re-blended).
    _, report = perturb_arrays(
        arrays,
        info,
        materials,
        uniform_samples(config.resolved_grid, T_REF + 1.0),
        ThermoOpticCoefficients({"core": 1e-4, "bg": 1e-4}, T_REF),
    )
    assert report.num_reblended == {}


def test_an_unknown_material_name_is_an_error(float64):
    _, arrays, config, info, materials = _scene(12.1, 2.085, _config())
    with pytest.raises(KeyError, match="absent from the scene"):
        perturb_arrays(
            arrays,
            info,
            materials,
            uniform_samples(config.resolved_grid, T_REF + 1.0),
            ThermoOpticCoefficients({"silicon": 1e-4}, T_REF),
        )


def test_the_box_sampling_mode_has_no_material_map():
    tag = _tag()
    config = SimulationConfig(time=1e-15, grid=UniformGrid(spacing=_D), material_sampling="box")
    materials = {"bg": Material(permittivity=2.085)}
    volume = SimulationVolume(partial_grid_shape=(8, 8, 1), material=materials["bg"], name=f"vol{tag}")
    boundaries, constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="periodic"), volume
    )
    _, arrays, _, config, info = fdtdx.place_objects([volume, *boundaries.values()], config, constraints)
    assert "yee_material_map" not in info
    with pytest.raises(ValueError, match="yee_material_map"):
        perturb_arrays(
            arrays,
            info,
            materials,
            uniform_samples(config.resolved_grid, T_REF + 1.0),
            ThermoOpticCoefficients({"bg": 1e-5}, T_REF),
        )


# ---------------------------------------------------------------------------
# (e) The record the loader exposes, and the artefact's file format
# ---------------------------------------------------------------------------


def test_the_record_agrees_with_the_loader_counters_and_lists_only_two_material_pixels(float64):
    _, _, _, info, _ = _scene(12.1, 2.085, _config())
    record = info["yee_material_map"]["smoothing_record"]
    stats = info["yee_sampling_difference"]
    e_passes = [p for p in record.passes if p.field == "E"]
    assert sum(p.num_pixels for p in e_passes) == stats["smoothing"]["num_smoothed"]
    assert record.lattice("V", 0).num_pixels == stats["smoothing_offdiag"]["num_smoothed"]
    for entry in record.passes:
        assert entry.cells.shape == (entry.num_pixels, 3)
        assert np.all((entry.fill > 0.0) & (entry.fill < 1.0))
        np.testing.assert_allclose(np.linalg.norm(entry.normal, axis=1), 1.0, rtol=1e-12)
        assert np.all(entry.material_hi != entry.material_lo)
        assert entry.isotropic_pair.all()
    # E_z is invariant along z here: its lattice sees the same in-plane interface as E_x/E_y do.
    assert {p.component for p in e_passes} == {0, 1, 2}


def test_yee_lattice_samples_round_trip_through_npz(float64, tmp_path):
    _, arrays, config, info, materials = _scene(12.1, 2.085, _config())
    grid = config.resolved_grid
    samples = samples_from_callable(grid, lambda p: 300.0 + 1e6 * p[:, 0])
    samples.covered["E1"][0, 0, 0] = False
    samples.values["E1"][0, 0, 0] = np.nan
    samples.provenance["solver"] = "test"
    path = samples.save(tmp_path / "T.npz")
    back = YeeLatticeSamples.load(path)
    assert back.lattices == samples.lattices
    assert back.matches_edges([grid.edges(a) for a in range(3)])
    for lattice in samples.lattices:
        np.testing.assert_array_equal(back.covered[lattice], samples.covered[lattice])
        np.testing.assert_array_equal(back.values[lattice], samples.values[lattice])
    assert back.provenance == {"solver": "test"}
    assert back.coverage_report()["E1"]["num_uncovered"] == 1
    # The loaded artefact drives the perturbation exactly like the in-memory one.
    a, _ = perturb_arrays(arrays, info, materials, samples, _coefficients(), uncovered="unperturbed")
    b, _ = perturb_arrays(arrays, info, materials, back, _coefficients(), uncovered="unperturbed")
    np.testing.assert_array_equal(np.asarray(a.inv_permittivities), np.asarray(b.inv_permittivities))
