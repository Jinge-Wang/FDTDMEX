"""The general field-driven perturbation engine: tensor responses, re-blend, validation.

Pins that the thermo-optic front end and the general engine agree bit for bit; that a uniform
electric field through a Pockels response reproduces, at every entry, a scene drawn with the
perturbed (diagonally anisotropic) material, including the vertex off-diagonal entries that the
loader's tensor blend produces there; that an off-diagonal bulk tensor is refused on the
3-component tier; that the per-voxel physics validation catches a non-positive tensor; and that the
photoelastic response is the identity at zero strain.
"""

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
    ThermoOpticCoefficients,
    ThermoOpticResponse,
    apply_permittivity_perturbation,
    apply_thermo_optic_perturbation,
    perturb_arrays,
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
    return f"_p{_COUNTER[0]}"


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


def test_the_thermo_optic_front_end_and_the_general_engine_agree_bit_for_bit(float64):
    arrays, config, info, materials = _scene(Material(permittivity=12.1), Material(permittivity=2.085))
    grid = config.resolved_grid
    span = _N * _D
    samples = samples_from_callable(grid, lambda p: T_REF + 40.0 * p[:, 0] / span + 10.0 * (p[:, 1] / span) ** 2)
    coefficients = ThermoOpticCoefficients({"core": 1.8e-4, "bg": 1e-5}, T_REF)
    a_inv, a_off, _ = apply_thermo_optic_perturbation(
        np.asarray(arrays.inv_permittivities),
        np.asarray(arrays.inv_permittivity_offdiag),
        info["yee_material_map"],
        materials,
        samples,
        coefficients,
    )
    model = PerturbationModel({"core": ThermoOpticResponse(1.8e-4, T_REF), "bg": ThermoOpticResponse(1e-5, T_REF)})
    b_inv, b_off, report = apply_permittivity_perturbation(
        np.asarray(arrays.inv_permittivities),
        np.asarray(arrays.inv_permittivity_offdiag),
        info["yee_material_map"],
        materials,
        {"T": samples},
        model,
    )
    np.testing.assert_array_equal(a_inv, b_inv)
    np.testing.assert_array_equal(a_off, b_off)
    assert report.num_tensor_reblended == {}  # isotropic pairs take the scalar path
    assert report.validation["num_checked"] > 0
    assert report.validation["min_eigenvalue"] > 1.0


def test_a_uniform_field_through_a_pockels_response_matches_the_pre_perturbed_anisotropic_scene(float64):
    """z-cut lithium niobate on the 3-component tier: E along z perturbs n_o and n_e diagonally."""
    base_core = Material(permittivity=(N_O**2, N_O**2, N_E**2))
    bg = Material(permittivity=2.085)
    arrays, config, info, materials = _scene(base_core, bg)
    grid = config.resolved_grid
    E_z = 5.0e6  # V/m (5 V across 1 um)
    samples = samples_from_callable(grid, lambda p: np.tile([0.0, 0.0, E_z], (p.shape[0], 1)), name="E", unit="V/m")
    model = PerturbationModel({"core": PockelsResponse(r=r_3m())})
    perturbed, report = perturb_arrays_with_model(arrays, info, materials, {"E": samples}, model)

    # Closed form for the diagonal entries: d(1/n^2)_i = r_i3 E_z  ->  n_i' = (1/n_i^2 + r_i3 E_z)^(-1/2)
    n_x = (1 / N_O**2 + R13 * E_z) ** -0.5
    n_z = (1 / N_E**2 + R33 * E_z) ** -0.5
    pre = Material(permittivity=(n_x**2, n_x**2, n_z**2))
    reference, _, _, _ = _scene(pre, bg)
    got = np.asarray(perturbed.inv_permittivities, dtype=np.float64)
    want = np.asarray(reference.inv_permittivities, dtype=np.float64)
    np.testing.assert_allclose(got, want, rtol=1e-11, atol=0.0)
    got_off = np.asarray(perturbed.inv_permittivity_offdiag, dtype=np.float64)
    want_off = np.asarray(reference.inv_permittivity_offdiag, dtype=np.float64)
    np.testing.assert_allclose(got_off, want_off, rtol=1e-9, atol=1e-16)
    assert np.count_nonzero(got_off) > 0
    assert not np.allclose(got, np.asarray(arrays.inv_permittivities, dtype=np.float64))
    # The tensor pair (anisotropic core, isotropic background) went through the tensor blend.
    assert sum(report.num_tensor_reblended.values()) == sum(report.num_reblended.values())
    assert report.responding_materials == {"core": "PockelsResponse"}
    assert report.validation["min_eigenvalue"] > 1.0


def test_an_off_diagonal_bulk_tensor_is_refused_on_the_diagonal_tier(float64):
    """E along x on a 3m crystal produces an xz entry through r51: not representable per component."""
    arrays, config, info, materials = _scene(
        Material(permittivity=(N_O**2, N_O**2, N_E**2)), Material(permittivity=2.085)
    )
    samples = samples_from_callable(
        config.resolved_grid, lambda p: np.tile([5.0e6, 0.0, 0.0], (p.shape[0], 1)), name="E"
    )
    model = PerturbationModel({"core": PockelsResponse(r=r_3m())})
    with pytest.raises(NotImplementedError, match="off-diagonal"):
        perturb_arrays_with_model(arrays, info, materials, {"E": samples}, model)


def test_the_physics_validation_catches_a_tensor_that_stops_being_positive_definite(float64):
    arrays, config, info, materials = _scene(Material(permittivity=12.1), Material(permittivity=2.085))
    # A temperature so far below the reference that n = sqrt(12.1) - 4 < 0 in the core: n^2 is still
    # positive, so the thermo-optic closed form is well defined; make the check bite with a response
    # that drives the permittivity negative instead.
    from fdtdx.coupling.perturbation import MaterialResponse

    class Negative(MaterialResponse):
        fields = ("T",)

        def tensor(self, base, values):
            K = np.asarray(values["T"]).shape[0]
            out = np.broadcast_to(np.asarray(base).reshape(3, 3), (K, 3, 3)).copy()
            out[:, 2, 2] = -1.0
            return out

        def unchanged(self, values):
            return np.zeros(np.asarray(values["T"]).shape[0], dtype=bool)

        def __eq__(self, other):
            return isinstance(other, Negative)

        def __hash__(self):
            return 1

    samples = uniform_samples(config.resolved_grid, T_REF + 1.0)
    with pytest.raises(ValueError, match="physical constraints"):
        perturb_arrays_with_model(arrays, info, materials, {"T": samples}, PerturbationModel({"core": Negative()}))
    # With the constraint switched off the write goes through (the tier holds the diagonal).
    inv, _, rep = apply_permittivity_perturbation(
        np.asarray(arrays.inv_permittivities),
        np.asarray(arrays.inv_permittivity_offdiag),
        info["yee_material_map"],
        materials,
        {"T": samples},
        PerturbationModel({"core": Negative()}),
        constraints=TensorConstraints(positive_definite=False),
    )
    assert rep.validation["constraints"]["positive_definite"] is False
    assert np.any(inv[2] < 0.0)


def test_the_photoelastic_response_is_the_identity_at_zero_strain_and_moves_with_strain(float64):
    arrays, config, info, materials = _scene(Material(permittivity=12.1), Material(permittivity=2.085))
    grid = config.resolved_grid
    # silicon-like p11 = -0.094, p12 = 0.017, p44 = -0.051 [general knowledge], cubic matrix
    p11, p12, p44 = -0.094, 0.017, -0.051
    p = [
        [p11, p12, p12, 0, 0, 0],
        [p12, p11, p12, 0, 0, 0],
        [p12, p12, p11, 0, 0, 0],
        [0, 0, 0, p44, 0, 0],
        [0, 0, 0, 0, p44, 0],
        [0, 0, 0, 0, 0, p44],
    ]
    model = PerturbationModel({"core": PhotoelasticResponse(p=p)})
    zero = samples_from_callable(grid, lambda q: np.zeros((q.shape[0], 6)), name="S")
    same, report = perturb_arrays_with_model(arrays, info, materials, {"S": zero}, model)
    np.testing.assert_array_equal(np.asarray(same.inv_permittivities), np.asarray(arrays.inv_permittivities))
    assert report.num_bulk_points == {"E0": 0, "E1": 0, "E2": 0}
    # uniaxial strain S_xx = 1e-3: diagonal change only, n_x moves by -n^3 p11 S / 2 to first order
    S = 1e-3
    strained = samples_from_callable(grid, lambda q: np.tile([S, 0, 0, 0, 0, 0], (q.shape[0], 1)), name="S")
    moved, _ = perturb_arrays_with_model(arrays, info, materials, {"S": strained}, model)
    front = info["yee_material_map"]["front_E"]
    table = info["yee_material_map"]["material_table"]
    core = [k for k, m in enumerate(table) if m.permittivity[0] == 12.1]
    bulk = np.isin(front[0], core)
    record = info["yee_material_map"]["smoothing_record"]
    bulk[tuple(record.lattice("E", 0).cells.T)] = False
    n0 = np.sqrt(12.1)
    n_x = (1 / n0**2 + p11 * S) ** -0.5
    np.testing.assert_allclose(np.asarray(moved.inv_permittivities)[0][bulk], 1 / n_x**2, rtol=1e-12)
    assert abs((n_x - n0) - (-0.5 * n0**3 * p11 * S)) < 1e-5  # first-order check; the second-order term is ~2e-6


def test_the_front_end_still_refuses_an_anisotropic_thermo_optic_material(float64):
    arrays, config, info, materials = _scene(Material(permittivity=(12.1, 12.1, 11.0)), Material(permittivity=2.085))
    with pytest.raises(NotImplementedError, match="anisotropic"):
        perturb_arrays(
            arrays,
            info,
            materials,
            uniform_samples(config.resolved_grid, T_REF + 1.0),
            ThermoOpticCoefficients({"core": 1e-4}, T_REF),
        )
    # ... while the general engine carries a diagonal one.
    model = PerturbationModel({"core": ThermoOpticResponse(1e-4, T_REF)})
    _, report = perturb_arrays_with_model(
        arrays, info, materials, {"T": uniform_samples(config.resolved_grid, T_REF + 10.0)}, model
    )
    assert report.num_bulk_points["E0"] > 0
    assert sum(report.num_tensor_reblended.values()) > 0
