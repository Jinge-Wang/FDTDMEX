"""The off-diagonal bulk policy and the unit assertion at the response boundary.

An electrode geometry never puts the field exactly on the optic axis, and a shear strain never
produces a diagonal tensor, so a response that runs on a real device produces off-diagonal
permittivity entries at bulk points. What the loader's arrays do with them is
``offdiag_bulk``. Pinned here:

* ``"tensor"`` on x-cut lithium niobate in a uniform field reproduces, at every entry of the
  9-component tier, a scene drawn with the pre-perturbed anisotropic material;
* ``"project"`` on the same case keeps the diagonal and reports the entry it dropped, its ratio to
  the tensor's diagonal spread and the mixing angle -- the numbers section 7 of the P1 design note
  measured on the thin-film device;
* ``"error"`` still refuses, which is what a case gets if it does not choose;
* silicon under a pure shear strain (the mechanical track's acceptance case) is refused under
  ``"error"``, written symmetric and positive definite under ``"tensor"``, and reports a degenerate
  (infinite) ratio under ``"project"``, because an isotropic base has no diagonal spread for the
  dropped entry to be small against;
* the per-voxel physics validation runs on the full tensors, not only on their diagonals;
* a sample whose recorded unit disagrees with the unit the response's arithmetic assumes is
  refused before anything is written.
"""

import json
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.coupling import (
    PerturbationModel,
    PhotoelasticResponse,
    PockelsResponse,
    TensorConstraints,
    ThermoOpticResponse,
    apply_permittivity_perturbation,
    check_sample_units,
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

# Congruent lithium niobate at 1550 nm [general knowledge]: n_o 2.21, n_e 2.14; r33 30.8 pm/V,
# r13 8.6 pm/V, r51 28 pm/V, r22 3.4 pm/V (3m point group, optic axis along crystal Z).
N_O, N_E = 2.21, 2.14
R33, R13, R51, R22 = 30.8e-12, 8.6e-12, 28.0e-12, 3.4e-12

# Silicon [general knowledge]: p11 -0.094, p12 0.017, p44 -0.051; permittivity 12.1 as elsewhere
# in this test suite.
P11, P12, P44 = -0.094, 0.017, -0.051
EPS_SI = 12.1

# The field the P1 design note measured over the ridge of the x-cut thin-film device at 1 V drive
# (grid axes, volts per metre), so the numbers this file asserts are that device's numbers.
E_RIDGE = np.array([-1.121e5, 1.696e4, 1.721e3])


def r_xcut() -> list[list[float]]:
    """The 3m electro-optic matrix in grid axes for an x-cut film: crystal Z on grid x.

    Rows in Voigt order (xx, yy, zz, yz, xz, xy), columns the grid field components. r33 lands on
    the lateral field (the working term) and r51 on the vertical one (the off-diagonal leak).
    """
    return [
        [R33, 0.0, 0.0],
        [R13, R22, 0.0],
        [R13, -R22, 0.0],
        [0.0, 0.0, -R22],
        [0.0, 0.0, R51],
        [0.0, R51, 0.0],
    ]


def p_cubic() -> list[list[float]]:
    """The (6, 6) photoelastic matrix of a cubic crystal, Voigt order, engineering shears."""
    return [
        [P11, P12, P12, 0.0, 0.0, 0.0],
        [P12, P11, P12, 0.0, 0.0, 0.0],
        [P12, P12, P11, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, P44, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, P44, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, P44],
    ]


@pytest.fixture
def float64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _tag() -> str:
    _COUNTER[0] += 1
    return f"_o{_COUNTER[0]}"


def _config(placement: str) -> SimulationConfig:
    return SimulationConfig(
        time=1e-15,
        grid=UniformGrid(spacing=_D),
        dtype=jnp.float64,
        material_sampling="yee_smooth",
        yee_smooth_full_tensor=True,
        yee_smooth_offdiag_placement=placement,
    )


def _scene(core: Material, bg: Material, placement: str, cells: int = _N):
    """A disk of ``core`` in ``bg``: a curved rim, so every lattice carries blended pixels."""
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
    with warnings.catch_warnings():
        # The dense pixel placement warns about its asymmetric D-to-E map; that warning is the
        # loader's, is asserted where it belongs, and is not what this file is about.
        warnings.simplefilter("ignore", UserWarning)
        _, arrays, _, config, info = fdtdx.place_objects(
            [volume, disk, *boundaries.values()], _config(placement), constraints
        )
    return arrays, config, info, materials


def _uniform_field(config, vector: np.ndarray):
    return samples_from_callable(
        config.resolved_grid,
        lambda p: np.tile(np.asarray(vector, dtype=np.float64), (p.shape[0], 1)),
        name="E",
        unit="V/m",
    )


def _uniform_strain(config, voigt: np.ndarray):
    return samples_from_callable(
        config.resolved_grid,
        lambda p: np.tile(np.asarray(voigt, dtype=np.float64), (p.shape[0], 1)),
        name="S",
        unit="1",
    )


# --- (a) the tensor tier reproduces a pre-perturbed anisotropic scene ---------------------------


def test_x_cut_lithium_niobate_under_tensor_matches_the_pre_perturbed_anisotropic_scene(float64):
    """Every entry, every lattice: the 9-component tier carries the full perturbed tensor."""
    base_core = Material(permittivity=(N_E**2, N_O**2, N_O**2))  # optic axis on grid x
    bg = Material(permittivity=2.085)
    arrays, config, info, materials = _scene(base_core, bg, placement="pixel")
    assert info["yee_material_map"]["num_perm_components"] == 9
    assert arrays.inv_permittivity_offdiag is None

    response = PockelsResponse(r=r_xcut())
    model = PerturbationModel({"core": response})
    samples = _uniform_field(config, E_RIDGE)
    perturbed, report = perturb_arrays_with_model(arrays, info, materials, {"E": samples}, model, offdiag_bulk="tensor")

    perturbed_tensor = response.tensor(np.diag([N_E**2, N_O**2, N_O**2]), {"E": E_RIDGE.reshape(1, 3)})[0]
    assert np.max(np.abs(perturbed_tensor - perturbed_tensor.T)) < 1e-15 * np.max(np.abs(perturbed_tensor))
    pre = Material(permittivity=tuple(perturbed_tensor.reshape(-1)))
    reference, _, _, _ = _scene(pre, bg, placement="pixel")

    got = np.asarray(perturbed.inv_permittivities, dtype=np.float64)
    want = np.asarray(reference.inv_permittivities, dtype=np.float64)
    assert got.shape[0] == 9
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-16)
    # The off-diagonal rows are not trivially zero, and the base scene did not already hold them.
    off_rows = [1, 2, 3, 5, 6, 7]
    assert np.max(np.abs(got[off_rows])) > 1e-8
    base_arr = np.asarray(arrays.inv_permittivities, dtype=np.float64)
    assert not np.allclose(got, base_arr)
    assert report.offdiag_bulk == "tensor"
    assert report.num_bulk_points["E0"] > 0
    assert sum(report.num_reblended.values()) > 0
    # The policy still records how large the off-diagonal term was, even though it kept it.
    assert report.offdiag["E0"]["core"].max_offdiag == pytest.approx(1.0621962510e-05, rel=1e-9)


def test_tensor_is_refused_on_the_three_component_tier_with_a_message_that_says_how_to_get_there(float64):
    arrays, config, info, materials = _scene(
        Material(permittivity=(N_E**2, N_O**2, N_O**2)), Material(permittivity=2.085), placement="node"
    )
    assert info["yee_material_map"]["num_perm_components"] == 3
    model = PerturbationModel({"core": PockelsResponse(r=r_xcut())})
    with pytest.raises(NotImplementedError, match="9-component"):
        perturb_arrays_with_model(
            arrays, info, materials, {"E": _uniform_field(config, E_RIDGE)}, model, offdiag_bulk="tensor"
        )


# --- (b) project keeps the diagonal and reports what it dropped ---------------------------------


def test_project_keeps_the_diagonal_and_reports_the_dropped_entry_and_the_ratio(float64):
    """The numbers of P1 section 7, on the 3-component tier the device actually runs on."""
    base = np.diag([N_E**2, N_O**2, N_O**2])
    arrays, config, info, materials = _scene(
        Material(permittivity=(N_E**2, N_O**2, N_O**2)), Material(permittivity=2.085), placement="node"
    )
    response = PockelsResponse(r=r_xcut())
    model = PerturbationModel({"core": response})
    perturbed, report = perturb_arrays_with_model(
        arrays, info, materials, {"E": _uniform_field(config, E_RIDGE)}, model, offdiag_bulk="project"
    )
    tensor = response.tensor(base, {"E": E_RIDGE.reshape(1, 3)})[0]

    record = report.offdiag["E0"]["core"]
    assert record.max_offdiag == pytest.approx(1.0621962510e-05, rel=1e-9)
    assert record.max_diagonal_change == pytest.approx(7.2413315015e-05, rel=1e-9)
    # The dropped entry is 15 % of the perturbation as a tensor entry ...
    assert record.ratio_to_change == pytest.approx(0.1467, abs=5e-4)
    # ... and 3.5e-5 of the birefringence it has to be small against, which is the number that
    # decides whether dropping it is second order.
    assert record.max_diagonal_spread == pytest.approx(N_O**2 - N_E**2, rel=1e-3)
    assert record.ratio == pytest.approx(3.4889e-05, rel=1e-3)
    # Weighed against the splitting of the two axes it actually mixes, the worst entry is not the
    # r51 leak but the r22 one on the nearly degenerate ordinary pair: 5 % and a 2.9 degree
    # rotation of axes whose eigenvalues still move by only 7e-9.
    assert record.max_pair_ratio == pytest.approx(0.0507, rel=2e-2)
    assert record.rotation_deg == pytest.approx(2.9, rel=5e-2)
    assert report.max_offdiag_ratio() == record.max_pair_ratio
    assert record.num_points == report.num_bulk_points["E0"]

    # What was written is the diagonal of the perturbed tensor, at the bulk points of the core.
    front = info["yee_material_map"]["front_E"]
    table = info["yee_material_map"]["material_table"]
    core = [k for k, m in enumerate(table) if m.permittivity[0] == N_E**2]
    got = np.asarray(perturbed.inv_permittivities, dtype=np.float64)
    for c in range(3):
        bulk = np.isin(front[c], core)
        bulk[tuple(info["yee_material_map"]["smoothing_record"].lattice("E", c).cells.T)] = False
        assert bulk.any()
        np.testing.assert_allclose(got[c][bulk], 1.0 / tensor[c, c], rtol=1e-12)


def test_error_is_still_the_default_and_names_the_policies(float64):
    arrays, config, info, materials = _scene(
        Material(permittivity=(N_E**2, N_O**2, N_O**2)), Material(permittivity=2.085), placement="node"
    )
    model = PerturbationModel({"core": PockelsResponse(r=r_xcut())})
    with pytest.raises(NotImplementedError, match="offdiag_bulk='project'"):
        perturb_arrays_with_model(arrays, info, materials, {"E": _uniform_field(config, E_RIDGE)}, model)
    with pytest.raises(ValueError, match="offdiag_bulk must be one of"):
        perturb_arrays_with_model(
            arrays, info, materials, {"E": _uniform_field(config, E_RIDGE)}, model, offdiag_bulk="vertex"
        )


def test_a_field_on_the_optic_axis_alone_still_runs_under_error(float64):
    """The policy changes nothing when the response stays diagonal: the default keeps working."""
    arrays, config, info, materials = _scene(
        Material(permittivity=(N_E**2, N_O**2, N_O**2)), Material(permittivity=2.085), placement="node"
    )
    model = PerturbationModel({"core": PockelsResponse(r=r_xcut())})
    axis_only = _uniform_field(config, np.array([E_RIDGE[0], 0.0, 0.0]))
    strict, report_strict = perturb_arrays_with_model(arrays, info, materials, {"E": axis_only}, model)
    projected, report_project = perturb_arrays_with_model(
        arrays, info, materials, {"E": axis_only}, model, offdiag_bulk="project"
    )
    np.testing.assert_array_equal(np.asarray(strict.inv_permittivities), np.asarray(projected.inv_permittivities))
    assert report_strict.offdiag == {} and report_project.offdiag == {}


# --- (c) the mechanical track's acceptance case: silicon under a pure shear strain ---------------


def _sheared_silicon(placement: str, shear: float = 1e-3):
    arrays, config, info, materials = _scene(
        Material(permittivity=EPS_SI), Material(permittivity=2.085), placement=placement
    )
    model = PerturbationModel({"core": PhotoelasticResponse(p=p_cubic())})
    samples = _uniform_strain(config, np.array([0.0, 0.0, 0.0, 0.0, 0.0, shear]))
    return arrays, config, info, materials, model, samples


def test_sheared_silicon_is_refused_under_error(float64):
    arrays, _, info, materials, model, samples = _sheared_silicon("node")
    with pytest.raises(NotImplementedError, match="off-diagonal"):
        perturb_arrays_with_model(arrays, info, materials, {"S": samples}, model)


def test_sheared_silicon_writes_a_symmetric_positive_definite_tensor_under_tensor(float64):
    """M's acceptance case: p44 on a pure shear, the off-diagonal entry is the signal."""
    shear = 1e-3
    arrays, _config_unused, info, materials, model, samples = _sheared_silicon("pixel", shear)
    assert info["yee_material_map"]["num_perm_components"] == 9
    perturbed, report = perturb_arrays_with_model(arrays, info, materials, {"S": samples}, model, offdiag_bulk="tensor")

    response = PhotoelasticResponse(p=p_cubic())
    tensor = response.tensor(np.eye(3) * EPS_SI, {"S": np.array([[0.0, 0.0, 0.0, 0.0, 0.0, shear]])})[0]
    assert np.max(np.abs(tensor - tensor.T)) < 1e-15 * np.max(np.abs(tensor))
    assert np.linalg.eigvalsh(tensor)[0] > 0.0
    # The shear splits the two in-plane indices and leaves the third alone: this is birefringence
    # the diagonal tier cannot represent at all.
    assert abs(tensor[0, 1]) > 0.0
    assert tensor[0, 0] == pytest.approx(tensor[1, 1])

    inverse = np.linalg.inv(tensor)
    got = np.asarray(perturbed.inv_permittivities, dtype=np.float64)
    front = info["yee_material_map"]["front_E"]
    table = info["yee_material_map"]["material_table"]
    core = [k for k, m in enumerate(table) if m.permittivity[0] == EPS_SI]
    for c in range(3):
        bulk = np.isin(front[c], core)
        bulk[tuple(info["yee_material_map"]["smoothing_record"].lattice("E", c).cells.T)] = False
        assert bulk.any()
        for j in range(3):
            np.testing.assert_allclose(got[3 * c + j][bulk], inverse[c, j], rtol=1e-12, atol=1e-18)
    assert np.max(np.abs(got[1][np.isin(front[0], core)])) > 0.0  # the xy entry was written
    assert report.offdiag["E0"]["core"].max_offdiag > 0.0


def test_project_reports_a_degenerate_ratio_for_sheared_silicon(float64):
    """An isotropic base has no diagonal spread, so the dropped entry is first order, not second."""
    arrays, _, info, materials, model, samples = _sheared_silicon("node")
    _, report = perturb_arrays_with_model(arrays, info, materials, {"S": samples}, model, offdiag_bulk="project")
    record = report.offdiag["E0"]["core"]
    assert record.max_offdiag > 0.0
    # The shear couples x and y, whose perturbed diagonal entries stay exactly equal, so the entry
    # sets the axes rather than perturbing them: the ratio is infinite and the rotation 45 degrees.
    assert record.max_pair_ratio == np.inf
    assert record.degenerate
    assert record.rotation_deg == 45.0
    assert report.max_offdiag_ratio() == np.inf
    # The report still has to land in a results file, so the infinity travels as a flag.
    payload = json.loads(json.dumps(report.as_dict(), allow_nan=False))
    assert payload["offdiag"]["E0"]["core"]["degenerate"] is True
    assert payload["offdiag"]["E0"]["core"]["max_pair_ratio"] is None


# --- (d) the physics validation runs on the full tensors ------------------------------------------


def test_the_tensor_constraints_run_on_the_full_tensor_not_only_its_diagonal(float64):
    shear = 1e-3
    arrays, _, info, materials, model, samples = _sheared_silicon("pixel", shear)
    _, report = perturb_arrays_with_model(arrays, info, materials, {"S": samples}, model, offdiag_bulk="tensor")
    assert report.validation["num_checked"] > 0
    assert report.validation["constraints"] == {"real": True, "symmetric": True, "positive_definite": True}
    # The smallest eigenvalue of a sheared tensor is below its diagonal entries, so a check that
    # read the diagonal only would report a different number.
    tensor = PhotoelasticResponse(p=p_cubic()).tensor(
        np.eye(3) * EPS_SI, {"S": np.array([[0.0, 0.0, 0.0, 0.0, 0.0, shear]])}
    )[0]
    counts = TensorConstraints().check(tensor[None])
    assert counts["num_not_positive_definite"] == 0 and counts["num_asymmetric"] == 0
    assert counts["min_eigenvalue"] == pytest.approx(np.linalg.eigvalsh(tensor)[0], rel=1e-12)
    assert counts["min_eigenvalue"] < tensor[0, 0]
    # The run's own minimum is over every tensor it checked, the background included.
    assert report.validation["min_eigenvalue"] <= counts["min_eigenvalue"]


def test_a_shear_large_enough_to_lose_positive_definiteness_is_caught_under_tensor(float64):
    """d(1/eps)_xy = p44 * shear; past 1/eps the tensor stops being positive definite."""
    arrays, _, info, materials, model, samples = _sheared_silicon("pixel", shear=2.0)
    with pytest.raises(ValueError, match="physical constraints"):
        perturb_arrays_with_model(arrays, info, materials, {"S": samples}, model, offdiag_bulk="tensor")
    # ... and the flags still switch it off explicitly rather than silently.
    inv, _, rep = apply_permittivity_perturbation(
        np.asarray(arrays.inv_permittivities),
        None,
        info["yee_material_map"],
        materials,
        {"S": samples},
        model,
        constraints=TensorConstraints(positive_definite=False),
        offdiag_bulk="tensor",
    )
    assert rep.validation["constraints"]["positive_definite"] is False
    assert np.isfinite(inv).all()


# --- the unit assertion at the response boundary ---------------------------------------------------


def test_a_field_in_volts_per_micrometre_against_a_response_in_volts_per_metre_is_refused(float64):
    arrays, config, info, materials = _scene(
        Material(permittivity=(N_E**2, N_O**2, N_O**2)), Material(permittivity=2.085), placement="node"
    )
    model = PerturbationModel({"core": PockelsResponse(r=r_xcut())})  # field_scale 1.0
    samples = samples_from_callable(
        config.resolved_grid,
        lambda p: np.tile([0.11, 0.0, 0.0], (p.shape[0], 1)),
        name="E",
        unit="V/um",
    )
    with pytest.raises(ValueError, match="field_scale"):
        perturb_arrays_with_model(arrays, info, materials, {"E": samples}, model, offdiag_bulk="project")
    # The same samples with the scale that converts them are accepted.
    scaled = PerturbationModel({"core": PockelsResponse(r=r_xcut(), field_scale=1e6)})
    assert check_sample_units(scaled, {"E": samples}) == {"E": "V/um"}
    perturb_arrays_with_model(arrays, info, materials, {"E": samples}, scaled, offdiag_bulk="project")


def test_a_temperature_label_on_an_electric_field_is_refused(float64):
    """The default sample label is "K"; on a Pockels response that is a mislabelled field."""
    arrays, config, info, materials = _scene(
        Material(permittivity=(N_E**2, N_O**2, N_O**2)), Material(permittivity=2.085), placement="node"
    )
    model = PerturbationModel({"core": PockelsResponse(r=r_xcut())})
    samples = samples_from_callable(config.resolved_grid, lambda p: np.tile(E_RIDGE, (p.shape[0], 1)), name="E")
    assert samples.unit == "K"
    with pytest.raises(ValueError, match="not a unit of 'V/m'"):
        perturb_arrays_with_model(arrays, info, materials, {"E": samples}, model, offdiag_bulk="project")


def test_an_unlabelled_sample_is_not_checked_and_a_thermo_optic_one_passes(float64):
    arrays, config, info, materials = _scene(
        Material(permittivity=EPS_SI), Material(permittivity=2.085), placement="node"
    )
    pockels = PerturbationModel({"core": PockelsResponse(r=r_xcut())})
    blank = samples_from_callable(config.resolved_grid, lambda p: np.tile(E_RIDGE, (p.shape[0], 1)), name="E", unit="")
    assert check_sample_units(pockels, {"E": blank}) == {"E": ""}
    thermal = PerturbationModel({"core": ThermoOpticResponse(1.8e-4, T_REF)})
    assert check_sample_units(thermal, {"T": uniform_samples(config.resolved_grid, T_REF + 1.0)}) == {"T": "K"}
    _, report = perturb_arrays_with_model(
        arrays, info, materials, {"T": uniform_samples(config.resolved_grid, T_REF + 1.0)}, thermal
    )
    assert report.num_bulk_points["E0"] > 0


def test_a_strain_labelled_in_a_unit_the_table_knows_is_converted_not_refused(float64):
    """Microstrain is a legal strain label as long as the response's field_scale converts it."""
    arrays, config, info, materials = _scene(
        Material(permittivity=EPS_SI), Material(permittivity=2.085), placement="node"
    )
    micro = samples_from_callable(
        config.resolved_grid, lambda p: np.tile([1e3, 0, 0, 0, 0, 0], (p.shape[0], 1)), name="S", unit="ustrain"
    )
    wrong = PerturbationModel({"core": PhotoelasticResponse(p=p_cubic())})
    with pytest.raises(ValueError, match="field_scale"):
        perturb_arrays_with_model(arrays, info, materials, {"S": micro}, wrong)
    right = PerturbationModel({"core": PhotoelasticResponse(p=p_cubic(), field_scale=1e-6)})
    moved, _ = perturb_arrays_with_model(arrays, info, materials, {"S": micro}, right)
    direct = PerturbationModel({"core": PhotoelasticResponse(p=p_cubic())})
    plain = _uniform_strain(config, np.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0]))
    same, _ = perturb_arrays_with_model(arrays, info, materials, {"S": plain}, direct)
    np.testing.assert_allclose(np.asarray(moved.inv_permittivities), np.asarray(same.inv_permittivities), rtol=1e-12)
