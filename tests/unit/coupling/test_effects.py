"""The object-oriented coupling front end: declaration, sampling, perturbation, composition.

The pins, in order: a coupling declares one effect (field name, rank, unit, per-material response)
and refuses samples whose label contradicts it; a uniform field through each concrete coupling
reproduces, entry for entry, a scene drawn with the perturbed material, which is the only way to
tell a coupling apart from an arbitrary rewrite of the arrays; a vector field's components are
turned into the Yee frame by the same transform that moved its positions; ``MultiCoupling``
reduces to its single part bit for bit, is exactly order-independent when one effect sits at its
own null value and differs only at second order otherwise; and the class and the engine underneath it
(``perturb_arrays_with_model`` on the coupling's own model) write bit-identical arrays, on the
disk-in-oxide test scene and on the phase-shifter cross-section at 40 nm.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.coupling import (
    CompositeResponse,
    CouplingReport,
    MultiCoupling,
    Photoelastic,
    Pockels,
    PointTransform,
    ThermoOptic,
    UniformFieldSource,
    YeeLatticeSamples,
    as_field_source,
    cubic_photoelastic_matrix,
    perturb_arrays_with_model,
    samples_from_callable,
    uniform_samples,
)
from fdtdx.materials import Material
from fdtdx.objects.static_material.cylinder import Cylinder
from fdtdx.objects.static_material.static import SimulationVolume

_D = 25e-9
_N = 24
_COUNTER = [0]
T_REF = 300.0

# Lithium-niobate-like numbers [general knowledge]: n_o 2.21, n_e 2.14 at 1550 nm; r33 30.8 pm/V,
# r13 8.6 pm/V, r51 28 pm/V, r22 3.4 pm/V (3m point group, optic axis along z).
N_O, N_E = 2.21, 2.14
R33, R13, R51, R22 = 30.8e-12, 8.6e-12, 28.0e-12, 3.4e-12
# silicon-like photoelastic constants [general knowledge]
P11, P12, P44 = -0.094, 0.017, -0.051

# the notebook's phase-shifter cross-section, micrometres (MetalHeaterPhaseShifter.ipynb)
SI_N, SIO2_N = 3.4777, 1.444
SI_DNDT, SIO2_DNDT = 1.86e-4, 1e-5
W_CORE_UM, H_CORE_UM = 0.5, 0.22
WIN_X, WIN_Z = 4.0e-6, 3.0e-6


def r_3m() -> list[list[float]]:
    """Contracted (6, 3) electro-optic matrix of the 3m class, rows (xx, yy, zz, yz, xz, xy)."""
    return [
        [0.0, -R22, R13],
        [0.0, R22, R13],
        [0.0, 0.0, R33],
        [0.0, R51, 0.0],
        [R51, 0.0, 0.0],
        [-R22, 0.0, 0.0],
    ]


@pytest.fixture
def float64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _tag() -> str:
    _COUNTER[0] += 1
    return f"_r{_COUNTER[0]}"


def _config() -> SimulationConfig:
    return SimulationConfig(
        time=1e-15,
        grid=UniformGrid(spacing=_D),
        dtype=jnp.float64,
        material_sampling="yee_smooth",
        yee_smooth_full_tensor=True,
        yee_smooth_offdiag_placement="node",
    )


def _scene(core: Material, bg: Material, cells: int = _N):
    """A disk of ``core`` in ``bg``: curved interface pixels and non-zero vertex off-diagonals."""
    tag = _tag()
    materials = {"bg": bg, "core": core}
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
    _, arrays, _, config, info = fdtdx.place_objects([volume, disk, *boundaries.values()], _config(), constraints)
    return arrays, config, info, materials


# ---------------------------------------------------------------------------
# (a) Declaration
# ---------------------------------------------------------------------------
def test_each_coupling_declares_its_field_its_rank_its_unit_and_its_responses():
    thermo = ThermoOptic(dn_dT={"si": SI_DNDT, "sio2": SIO2_DNDT, "air": 0.0}, reference_temperature=T_REF)
    assert (thermo.field_name, thermo.field_rank, thermo.expected_unit) == ("T", 0, "K")
    assert thermo.sample_unit == "K"
    assert set(thermo.responses()) == {"si", "sio2"}  # a zero coefficient does not respond
    assert thermo.material_names() == ("si", "sio2", "air")  # but it is still declared
    assert thermo.responses()["si"].dn_dT == SI_DNDT
    assert thermo.responses()["si"].reference_temperature == T_REF
    assert thermo.null_value() == T_REF

    pockels = Pockels(r={"ln": r_3m()}, field_scale=1e6)
    assert (pockels.field_name, pockels.field_rank, pockels.expected_unit) == ("E", 1, "V/m")
    assert pockels.responses()["ln"].field_scale == 1e6
    assert pockels.null_value() == 0.0

    photo = Photoelastic(p={"si": cubic_photoelastic_matrix(P11, P12, P44)})
    assert (photo.field_name, photo.field_rank, photo.expected_unit) == ("S", 2, "1")
    assert photo.responses()["si"].p is not None

    stack = MultiCoupling([thermo, photo])
    assert stack.model().fields == ("T", "S")
    assert isinstance(stack.responses()["si"], CompositeResponse)
    assert stack.responses()["sio2"] is not None
    assert not isinstance(stack.responses()["sio2"], CompositeResponse)  # only one effect touches it


def test_two_couplings_reading_one_field_name_are_refused():
    with pytest.raises(ValueError, match="two couplings read the field"):
        MultiCoupling([ThermoOptic(dn_dT={"a": 1e-4}), ThermoOptic(dn_dT={"b": 1e-4})])


def test_an_unknown_material_is_named_before_anything_is_sampled(float64):
    arrays, config, info, materials = _scene(Material(permittivity=12.1), Material(permittivity=2.085))
    with pytest.raises(KeyError, match="absent from the scene"):
        ThermoOptic(dn_dT={"silicon": 1e-4}, reference_temperature=T_REF).apply(
            T_REF + 1.0, arrays, info, materials, config.resolved_grid
        )


def test_an_anisotropic_material_is_still_refused_by_the_thermo_optic_class(float64):
    arrays, config, info, materials = _scene(Material(permittivity=(12.1, 12.1, 11.0)), Material(permittivity=2.085))
    with pytest.raises(NotImplementedError, match="anisotropic"):
        ThermoOptic(dn_dT={"core": 1e-4}, reference_temperature=T_REF).apply(
            T_REF + 1.0, arrays, info, materials, config.resolved_grid
        )


def test_a_sample_label_that_contradicts_the_declared_unit_is_refused(float64):
    """The 10^6 slip: a field solved in volts per micrometre reaching a response written in V/m."""
    arrays, config, info, materials = _scene(
        Material(permittivity=(N_O**2, N_O**2, N_E**2)), Material(permittivity=2.085)
    )
    grid = config.resolved_grid
    in_um = samples_from_callable(grid, lambda p: np.tile([0.0, 0.0, 5.0], (p.shape[0], 1)), name="E", unit="V/um")
    with pytest.raises(ValueError, match="field_scale"):
        Pockels(r={"core": r_3m()}).perturb(arrays, info, materials, in_um)
    # the same samples with the scale that converts them go through
    perturbed, report = Pockels(r={"core": r_3m()}, field_scale=1e6).perturb(arrays, info, materials, in_um)
    assert report.units == {"E": "V/um"}
    assert not np.array_equal(np.asarray(perturbed.inv_permittivities), np.asarray(arrays.inv_permittivities))


# ---------------------------------------------------------------------------
# (b) A uniform field is a pre-perturbed scene, for every subclass
# ---------------------------------------------------------------------------
def test_a_uniform_temperature_through_the_class_matches_the_pre_perturbed_scene(float64):
    dT = 45.0
    n_core, n_bg = np.sqrt(12.1), np.sqrt(2.085)
    arrays, config, info, materials = _scene(Material(permittivity=12.1), Material(permittivity=2.085))
    coupling = ThermoOptic(dn_dT={"core": 1.8e-4, "bg": 1e-5}, reference_temperature=T_REF)
    perturbed, report = coupling.apply(T_REF + dT, arrays, info, materials, config.resolved_grid)
    reference, _, _, _ = _scene(
        Material(permittivity=(n_core + 1.8e-4 * dT) ** 2), Material(permittivity=(n_bg + 1e-5 * dT) ** 2)
    )
    np.testing.assert_allclose(
        np.asarray(perturbed.inv_permittivities, dtype=np.float64),
        np.asarray(reference.inv_permittivities, dtype=np.float64),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(perturbed.inv_permittivity_offdiag, dtype=np.float64),
        np.asarray(reference.inv_permittivity_offdiag, dtype=np.float64),
        rtol=1e-12,
        atol=1e-18,
    )
    assert report.extras["max_delta_T"] == pytest.approx(dT)
    assert report.extras["max_delta_n"] == pytest.approx(1.8e-4 * dT)
    assert report.perturbation is not None and sum(report.perturbation.num_reblended.values()) > 0


def test_a_uniform_field_through_the_pockels_class_matches_the_pre_perturbed_scene(float64):
    """z-cut lithium niobate: E along z perturbs n_o and n_e diagonally, nothing off-diagonal."""
    bg = Material(permittivity=2.085)
    arrays, config, info, materials = _scene(Material(permittivity=(N_O**2, N_O**2, N_E**2)), bg)
    E_z = 5.0e6  # V/m, 5 V across 1 um
    perturbed, report = Pockels(r={"core": r_3m()}).apply(
        [0.0, 0.0, E_z], arrays, info, materials, config.resolved_grid
    )
    n_x = (1 / N_O**2 + R13 * E_z) ** -0.5
    n_z = (1 / N_E**2 + R33 * E_z) ** -0.5
    reference, _, _, _ = _scene(Material(permittivity=(n_x**2, n_x**2, n_z**2)), bg)
    np.testing.assert_allclose(
        np.asarray(perturbed.inv_permittivities, dtype=np.float64),
        np.asarray(reference.inv_permittivities, dtype=np.float64),
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        np.asarray(perturbed.inv_permittivity_offdiag, dtype=np.float64),
        np.asarray(reference.inv_permittivity_offdiag, dtype=np.float64),
        rtol=1e-9,
        atol=1e-16,
    )
    assert report.coupling == "Pockels"
    assert report.perturbation is not None
    assert report.perturbation.responding_materials == {"core": "PockelsResponse"}


def test_a_uniform_strain_through_the_photoelastic_class_matches_the_pre_perturbed_scene(float64):
    """A uniaxial strain tensor handed in as (3, 3): the class contracts it into Voigt itself."""
    bg = Material(permittivity=2.085)
    eps = 12.1
    arrays, config, info, materials = _scene(Material(permittivity=eps), bg)
    S = 1e-3
    strain = np.diag([S, 0.0, 0.0])
    coupling = Photoelastic(p={"core": cubic_photoelastic_matrix(P11, P12, P44)})
    perturbed, report = coupling.apply(strain, arrays, info, materials, config.resolved_grid)
    n_x = (1 / eps + P11 * S) ** -0.5
    n_y = (1 / eps + P12 * S) ** -0.5
    reference, _, _, _ = _scene(Material(permittivity=(n_x**2, n_y**2, n_y**2)), bg)
    np.testing.assert_allclose(
        np.asarray(perturbed.inv_permittivities, dtype=np.float64),
        np.asarray(reference.inv_permittivities, dtype=np.float64),
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        np.asarray(perturbed.inv_permittivity_offdiag, dtype=np.float64),
        np.asarray(reference.inv_permittivity_offdiag, dtype=np.float64),
        rtol=1e-9,
        atol=1e-16,
    )
    assert report.fields == ("S",)


def test_a_zero_field_is_the_identity_for_every_coupling(float64):
    arrays, config, info, materials = _scene(Material(permittivity=12.1), Material(permittivity=2.085))
    grid = config.resolved_grid
    cases = [
        (ThermoOptic(dn_dT={"core": 1.8e-4, "bg": 1e-5}, reference_temperature=T_REF), T_REF),
        (Pockels(r={"core": r_3m()}), [0.0, 0.0, 0.0]),
        (Photoelastic(p={"core": cubic_photoelastic_matrix(P11, P12, P44)}), np.zeros((3, 3))),
    ]
    for coupling, null in cases:
        same, _ = coupling.apply(null, arrays, info, materials, grid)
        np.testing.assert_array_equal(np.asarray(same.inv_permittivities), np.asarray(arrays.inv_permittivities))
        np.testing.assert_array_equal(
            np.asarray(same.inv_permittivity_offdiag), np.asarray(arrays.inv_permittivity_offdiag)
        )


# ---------------------------------------------------------------------------
# (c) The transform moves positions and components together
# ---------------------------------------------------------------------------
def test_a_vector_fields_components_are_turned_into_the_yee_frame(float64):
    """A mesh drawn in (x, z) sampled by a grid whose second axis is the propagation axis."""
    _, config, _, _ = _scene(Material(permittivity=12.1), Material(permittivity=2.085))
    grid = config.resolved_grid
    transform = PointTransform(scale=1e6, collapse_axes=(1,), permute=(0, 2, 1))
    coupling = Pockels(r={"core": r_3m()})
    # a field that is (1, 2, 3) in the mesh frame everywhere: mesh axis i lands on Yee axis permute[i]
    samples = coupling.sample(lambda p: np.tile([1.0, 2.0, 3.0], (p.shape[0], 1)), grid, ("E0",), transform)
    got = samples.values["E0"].reshape(-1, 3)[0]
    np.testing.assert_array_equal(got, [1.0, 3.0, 2.0])
    # a scalar coupling leaves the values alone whatever the transform is
    scalar = ThermoOptic(dn_dT={"core": 1e-4}, reference_temperature=T_REF).sample(
        lambda p: p[:, 0] * 0.0 + 7.0, grid, ("E0",), transform
    )
    assert float(scalar.values["E0"].ravel()[0]) == 7.0


def test_samples_that_are_already_in_the_yee_frame_are_not_turned_twice(float64):
    _, config, _, _ = _scene(Material(permittivity=12.1), Material(permittivity=2.085))
    grid = config.resolved_grid
    transform = PointTransform(permute=(0, 2, 1))
    coupling = Pockels(r={"core": r_3m()})
    once = coupling.sample(lambda p: np.tile([1.0, 2.0, 3.0], (p.shape[0], 1)), grid, ("E0",), transform)
    twice = coupling.sample(once, grid, ("E0",), transform)
    np.testing.assert_array_equal(twice.values["E0"], once.values["E0"])


def test_a_field_source_can_be_a_constant_a_callable_or_samples_read_back(float64, tmp_path):
    _, config, _, _ = _scene(Material(permittivity=12.1), Material(permittivity=2.085))
    grid = config.resolved_grid
    coupling = ThermoOptic(dn_dT={"core": 1e-4}, reference_temperature=T_REF)
    constant = coupling.sample(T_REF + 3.0, grid, ("E0",))
    assert constant.unit == "K" and constant.values["E0"].min() == T_REF + 3.0
    assert isinstance(as_field_source(T_REF), UniformFieldSource)
    path = constant.save(tmp_path / "T.npz")
    back = coupling.sample(YeeLatticeSamples.load(path), grid, ("E0",))
    np.testing.assert_array_equal(back.values["E0"], constant.values["E0"])
    with pytest.raises(ValueError, match="different grid"):
        coupling.sample(constant, (grid.edges(0)[:-1], grid.edges(1), grid.edges(2)), ("E0",))


# ---------------------------------------------------------------------------
# (d) MultiCoupling
# ---------------------------------------------------------------------------
def _thermo_and_strain(float64_scene):
    arrays, config, info, materials = float64_scene
    thermo = ThermoOptic(dn_dT={"core": 1.8e-4, "bg": 1e-5}, reference_temperature=T_REF)
    photo = Photoelastic(p={"core": cubic_photoelastic_matrix(P11, P12, P44)})
    return arrays, config, info, materials, thermo, photo


def test_a_one_part_stack_is_the_part_itself_bit_for_bit(float64):
    scene = _scene(Material(permittivity=12.1), Material(permittivity=2.085))
    arrays, config, info, materials, thermo, _ = _thermo_and_strain(scene)
    grid = config.resolved_grid
    alone, _ = thermo.apply(T_REF + 30.0, arrays, info, materials, grid)
    stacked, report = MultiCoupling([thermo]).apply({"T": T_REF + 30.0}, arrays, info, materials, grid)
    np.testing.assert_array_equal(np.asarray(stacked.inv_permittivities), np.asarray(alone.inv_permittivities))
    np.testing.assert_array_equal(
        np.asarray(stacked.inv_permittivity_offdiag), np.asarray(alone.inv_permittivity_offdiag)
    )
    assert report.coupling == "MultiCoupling"
    assert [part.coupling for part in report.parts] == ["ThermoOptic"]
    assert report.parts[0].extras["max_delta_T"] == pytest.approx(30.0)


def test_the_order_of_a_stack_does_not_matter_when_one_effect_sits_at_its_null_value(float64):
    scene = _scene(Material(permittivity=12.1), Material(permittivity=2.085))
    arrays, config, info, materials, thermo, photo = _thermo_and_strain(scene)
    grid = config.resolved_grid
    sources = {"T": T_REF + 30.0, "S": np.zeros((3, 3))}
    a, _ = MultiCoupling([thermo, photo]).apply(sources, arrays, info, materials, grid)
    b, _ = MultiCoupling([photo, thermo]).apply(sources, arrays, info, materials, grid)
    alone, _ = thermo.apply(T_REF + 30.0, arrays, info, materials, grid)
    np.testing.assert_array_equal(np.asarray(a.inv_permittivities), np.asarray(b.inv_permittivities))
    np.testing.assert_array_equal(np.asarray(a.inv_permittivities), np.asarray(alone.inv_permittivities))
    np.testing.assert_array_equal(np.asarray(a.inv_permittivity_offdiag), np.asarray(alone.inv_permittivity_offdiag))


def test_a_stack_applies_both_effects_and_the_order_costs_only_second_order(float64):
    scene = _scene(Material(permittivity=12.1), Material(permittivity=2.085))
    arrays, config, info, materials, thermo, photo = _thermo_and_strain(scene)
    grid = config.resolved_grid
    dT, S = 40.0, 1e-3
    sources = {"T": T_REF + dT, "S": np.diag([S, 0.0, 0.0])}
    both, report = MultiCoupling([thermo, photo]).apply(sources, arrays, info, materials, grid)
    other, _ = MultiCoupling([photo, thermo]).apply(sources, arrays, info, materials, grid)
    hot, _ = thermo.apply(T_REF + dT, arrays, info, materials, grid)
    strained, _ = photo.apply(np.diag([S, 0.0, 0.0]), arrays, info, materials, grid)

    got = np.asarray(both.inv_permittivities, dtype=np.float64)
    only_hot = np.asarray(hot.inv_permittivities, dtype=np.float64)
    only_strained = np.asarray(strained.inv_permittivities, dtype=np.float64)
    cold = np.asarray(arrays.inv_permittivities, dtype=np.float64)
    # both effects are present: the stack is neither of the single ones
    assert not np.allclose(got, only_hot) and not np.allclose(got, only_strained)
    # and to first order the two changes add
    linear = only_hot + only_strained - cold
    np.testing.assert_allclose(got, linear, rtol=2e-4)
    # The two orders differ, but only at second order: the gap between them is a small fraction of
    # what either effect does on its own (dn/n ~ 2e-3 for the temperature, ~1e-3 for the strain).
    swapped = np.asarray(other.inv_permittivities, dtype=np.float64)
    assert not np.array_equal(got, swapped)
    gap = float(np.max(np.abs(got - swapped)))
    effect = float(np.max(np.abs(got - cold)))
    assert 0.0 < gap < 1e-2 * effect
    assert [part.coupling for part in report.parts] == ["ThermoOptic", "Photoelastic"]
    assert report.extras["order"] == ["ThermoOptic", "Photoelastic"]


def test_a_composite_response_refuses_a_stacked_base():
    composite = CompositeResponse(parts=(ThermoOptic(dn_dT={"a": 1e-4}).responses()["a"],))
    with pytest.raises(ValueError, match="one base tensor"):
        composite.tensor(np.broadcast_to(np.eye(3), (2, 3, 3)), {"T": np.zeros(2)})


# ---------------------------------------------------------------------------
# (e) The declaration and the engine underneath it agree bit for bit
# ---------------------------------------------------------------------------
def test_the_class_and_the_engine_it_calls_write_identical_arrays(float64):
    arrays, config, info, materials = _scene(Material(permittivity=12.1), Material(permittivity=2.085))
    grid = config.resolved_grid
    span = _N * _D
    field = samples_from_callable(grid, lambda p: T_REF + 40.0 * p[:, 0] / span + 10.0 * (p[:, 1] / span) ** 2)
    coupling = ThermoOptic(dn_dT={"core": 1.8e-4, "bg": 1e-5}, reference_temperature=T_REF)
    old, old_report = perturb_arrays_with_model(arrays, info, materials, {"T": field}, coupling.model())
    new, new_report = coupling.apply(field, arrays, info, materials, grid)
    assert np.array_equal(np.asarray(old.inv_permittivities), np.asarray(new.inv_permittivities))
    assert np.array_equal(np.asarray(old.inv_permittivity_offdiag), np.asarray(new.inv_permittivity_offdiag))
    # the coupling's report is a superset of the engine's own, entry for entry
    old_dict, new_dict = old_report.as_dict(), new_report.as_dict()
    assert set(old_dict) <= set(new_dict)
    for key, value in old_dict.items():
        assert new_dict[key] == value, key
    assert isinstance(new_report, CouplingReport)


def _phase_shifter_scene(dx_nm: float, materials: dict, y_cells: int = 8):
    """The notebook's phase-shifter cross-section: a Si core in oxide, faces snapped to grid lines.

    The 4 x 3 um optical window and the 0.5 x 0.22 um core of ``thermal_phase_shifter_fdtd3d``, at
    the case's smoke resolution. The propagation extent is cut to a few cells: the perturbation is
    per Yee point and the scene is invariant along that axis, so the arrays it writes there are the
    same ones the full-length placement writes, at a fraction of the placement cost.
    """
    tag = _tag()
    d = dx_nm * 1e-9
    nx, nz = round(WIN_X / d), round(WIN_Z / d)
    grid = fdtdx.RectilinearGrid.custom(
        x_edges=np.linspace(0.0, nx * d, nx + 1),
        y_edges=np.linspace(0.0, y_cells * d, y_cells + 1),
        z_edges=np.linspace(0.0, nz * d, nz + 1),
    )
    config = SimulationConfig(
        time=1e-15,
        grid=grid,
        backend="cpu",
        dtype=jnp.float64,
        gradient_config=None,
        material_sampling="yee_smooth",  # the diagonal tier, as the case places it
    )
    volume = fdtdx.SimulationVolume(partial_grid_shape=grid.shape, material=materials["sio2"], name=f"bg{tag}")
    core = fdtdx.UniformMaterialObject(
        material=materials["si"],
        partial_real_shape=(W_CORE_UM * 1e-6, None, H_CORE_UM * 1e-6),
        name=f"core{tag}",
        placement_order=1,
    )
    constraints = [
        core.same_size(volume, axes=(1,)),
        core.place_relative_to(
            volume,
            axes=(0, 2),
            own_positions=(0, 0),
            other_positions=(-1, -1),
            margins=(nx * d / 2, nz * d / 2),
        ),
    ]
    boundaries, bcons = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="periodic"), volume
    )
    _, arrays, _, config, info = fdtdx.place_objects([volume, core, *boundaries.values()], config, constraints + bcons)
    return arrays, config, info


def test_the_phase_shifter_scene_at_40_nm_is_bit_identical_through_class_and_engine(float64):
    """Placement only, no FDTD: the recorded numbers of the smoke run must not move."""
    materials = {"sio2": Material(permittivity=SIO2_N**2), "si": Material(permittivity=SI_N**2)}
    arrays, config, info = _phase_shifter_scene(40.0, materials)
    grid = config.resolved_grid
    coupling = ThermoOptic(dn_dT={"si": SI_DNDT, "sio2": SIO2_DNDT}, reference_temperature=T_REF)
    lattices = ("E0", "E1", "E2")

    # a temperature that varies the way the heater's does: hottest above the core, falling with z
    def profile(points: np.ndarray) -> np.ndarray:
        x, z = points[:, 0] - WIN_X / 2, points[:, 2] - WIN_Z / 2
        return T_REF + 60.0 * np.exp(-((x / 1.5e-6) ** 2)) * np.exp(-(((z - 1.0e-6) / 1.2e-6) ** 2))

    field = samples_from_callable(grid, profile, lattices=lattices)
    old, old_report = perturb_arrays_with_model(arrays, info, materials, {"T": field}, coupling.model())
    new, new_report = coupling.apply(field, arrays, info, materials, grid, lattices=lattices)
    assert np.array_equal(np.asarray(old.inv_permittivities), np.asarray(new.inv_permittivities))
    assert old_report.as_dict()["max_delta_eps"] == new_report.as_dict()["max_delta_eps"]
    assert old_report.num_bulk_points == new_report.perturbation.num_bulk_points

    # and the identity at the reference temperature holds through the class, as the case asserts
    identity, _ = coupling.apply(T_REF, arrays, info, materials, grid, lattices=lattices)
    assert np.array_equal(np.asarray(identity.inv_permittivities), np.asarray(arrays.inv_permittivities))
    # the engine underneath, given the same uniform field, said the same
    control, _ = perturb_arrays_with_model(
        arrays, info, materials, {"T": uniform_samples(grid, T_REF, lattices=lattices)}, coupling.model()
    )
    assert np.array_equal(np.asarray(control.inv_permittivities), np.asarray(identity.inv_permittivities))
