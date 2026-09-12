"""The conductivity write: the second of the loader's two arrays, end to end through a coupling.

The gates, in the order a reader should care about them:

* a uniform carrier field through the coupling produces *both* of the loader's arrays bit for bit
  equal to a scene drawn with the perturbed complex index -- the identity that says the write
  matches the loader instead of inventing a blend;
* a zero carrier change leaves both arrays bit for bit alone, and a lossless response leaves no
  conductivity block in the record at all;
* the fork's own FDTD, run through a medium this write made lossy, decays like ``exp(-alpha z)``
  with the ``alpha`` of the complex index that was asked for. That is the sign convention proved by
  measurement rather than by reading the update formula;
* the physical constraints still hold: the permittivity stays real, symmetric and positive definite
  and the conductivity stays non-negative unless a response says otherwise.

Every test here drives ``PlasmaDispersion.perturb``, so it crosses the coupling in
:mod:`fdtdx.coupling.effects`, the law in :mod:`fdtdx.coupling.responses` and the write in
:mod:`fdtdx.coupling.perturb` in one call; the laws' own arithmetic is pinned without a scene in
``test_responses.py``.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.coupling import (
    SOREF_BENNETT_1550,
    LossyResponse,
    PlasmaDispersion,
    SorefBennett,
    TensorConstraints,
    samples_from_callable,
    sigma_from_extinction,
)

WAVELENGTH = 1.55e-6
N_SI, K_SI = 3.4757, 3.0836e-05  # registry 148411: nSi0 and k0 = lam0 alpha0 / (4 pi)
N_OX, K_OX = 1.443, 2e-05
RESOLUTION = 20e-9


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def _materials(index_si: complex = complex(N_SI, K_SI)) -> dict:
    return {
        "si": fdtdx.Material.from_refractive_index(index_si, wavelength=WAVELENGTH),
        "sio2": fdtdx.Material.from_refractive_index(complex(N_OX, K_OX), wavelength=WAVELENGTH),
    }


def _grid():
    axes = [
        np.arange(0.0, 0.6e-6 + 1e-12, RESOLUTION),
        np.arange(0.0, 0.2e-6 + 1e-12, RESOLUTION),
        np.arange(0.0, 0.4e-6 + 1e-12, RESOLUTION),
    ]
    return fdtdx.RectilinearGrid.custom(x_edges=axes[0], y_edges=axes[1], z_edges=axes[2])


def _place(materials: dict):
    """A silicon block in oxide, on the smoothed Yee tier, so interface pixels exist."""
    grid = _grid()
    config = fdtdx.SimulationConfig(
        time=10e-15,
        grid=grid,
        backend="cpu",
        dtype=jnp.float32,
        gradient_config=None,
        material_sampling="yee_smooth",
    )
    volume = fdtdx.SimulationVolume(partial_grid_shape=grid.shape, material=materials["sio2"], name="bg")
    core = fdtdx.UniformMaterialObject(
        material=materials["si"], partial_real_shape=(0.22e-6, None, 0.14e-6), name="core"
    )
    constraints = [core.place_relative_to(volume, axes=(0, 1, 2), own_positions=(0, 0, 0), other_positions=(0, 0, 0))]
    _, arrays, _, config, info = fdtdx.place_objects(
        object_list=[volume, core], config=config, constraints=constraints, key=jax.random.PRNGKey(0)
    )
    return arrays, info, grid


def _carrier_samples(grid, electrons: float, holes: float, unit: str = "1/cm^3"):
    return samples_from_callable(
        grid,
        lambda points: np.broadcast_to(np.array([electrons, holes], dtype=np.float64), (points.shape[0], 2)),
        ("E0", "E1", "E2"),
        name="C",
        unit=unit,
    )


def _deltas(electrons: float, holes: float, coefficients: SorefBennett = SOREF_BENNETT_1550):
    dN, dP = np.array([electrons]), np.array([holes])
    return (
        float(coefficients.delta_index(dN, dP)[0]),
        float(coefficients.delta_extinction(dN, dP, WAVELENGTH)[0]),
    )


# ------------------------------------------------------------------------------------------------
# the identity: a uniform field equals a scene drawn with the perturbed material
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("electrons, holes", [(3.0e17, 7.0e17), (0.0, 5.0e17), (2.0e18, 0.0)])
def test_uniform_carrier_field_equals_a_prebuilt_scene(electrons, holes):
    """Both arrays, every entry: the perturbation and the pre-built scene must agree bit for bit."""
    materials = _materials()
    arrays, info, grid = _place(materials)
    coupling = PlasmaDispersion(index={"si": complex(N_SI, K_SI)}, wavelength=WAVELENGTH)
    perturbed, report = coupling.perturb(arrays, info, materials, _carrier_samples(grid, electrons, holes))

    dn, dk = _deltas(electrons, holes)
    reference_materials = {
        **materials,
        "si": fdtdx.Material.from_refractive_index(complex(N_SI + dn, K_SI + dk), wavelength=WAVELENGTH),
    }
    reference, _, _ = _place(reference_materials)

    assert np.array_equal(np.asarray(perturbed.inv_permittivities), np.asarray(reference.inv_permittivities)), (
        "the perturbed inverse permittivity is not the pre-built scene's"
    )
    assert np.array_equal(np.asarray(perturbed.electric_conductivity), np.asarray(reference.electric_conductivity)), (
        "the perturbed conductivity is not the pre-built scene's"
    )
    # The permittivity pass re-blends interface pixels; the conductivity pass must not, because the
    # loader does not blend the conductivity either. Every point of the responding material is
    # written, blended pixels included.
    front = np.asarray(info["yee_material_map"]["front_E"])
    table = info["yee_material_map"]["material_table"]
    si_index = next(i for i, m in enumerate(table) if m == materials["si"])
    for c in range(3):
        assert report.perturbation.conductivity.num_points[f"E{c}"] == int(np.count_nonzero(front[c] == si_index))
    assert report.perturbation.num_reblended, "the scene has no blended interface pixels to test against"


def test_zero_carrier_change_is_bit_identical():
    materials = _materials()
    arrays, info, grid = _place(materials)
    coupling = PlasmaDispersion(index={"si": complex(N_SI, K_SI)}, wavelength=WAVELENGTH)
    out, report = coupling.perturb(arrays, info, materials, _carrier_samples(grid, 0.0, 0.0))
    assert np.array_equal(np.asarray(out.inv_permittivities), np.asarray(arrays.inv_permittivities))
    assert np.array_equal(np.asarray(out.electric_conductivity), np.asarray(arrays.electric_conductivity))
    assert report.perturbation.conductivity.max_delta_sigma == 0.0


def test_conductivity_is_written_at_the_loaders_scale():
    """The stored array is ``sigma * conductivity_spacing``, which is the grid resolution in metres."""
    materials = _materials()
    arrays, info, grid = _place(materials)
    coupling = PlasmaDispersion(index={"si": complex(N_SI, K_SI)}, wavelength=WAVELENGTH)
    out, report = coupling.perturb(arrays, info, materials, _carrier_samples(grid, 1.0e18, 1.0e18))
    spacing = report.perturbation.conductivity.conductivity_spacing
    assert spacing == pytest.approx(RESOLUTION, rel=1e-12)

    dn, dk = _deltas(1.0e18, 1.0e18)
    expected = float(np.asarray(sigma_from_extinction(N_SI + dn, K_SI + dk, WAVELENGTH)))
    stored = np.asarray(out.electric_conductivity, dtype=np.float64)
    front = np.asarray(info["yee_material_map"]["front_E"])
    table = info["yee_material_map"]["material_table"]
    si_index = next(i for i, m in enumerate(table) if m == materials["si"])
    inside = stored[0][front[0] == si_index]
    # The scene is placed in float32, so the stored array carries the value to about 1e-7 relative;
    # the bit-identity test above is what pins the arithmetic exactly.
    assert np.allclose(inside / spacing, expected, rtol=1e-6)


# ------------------------------------------------------------------------------------------------
# physics guards
# ------------------------------------------------------------------------------------------------
def test_added_carriers_absorb_more_than_the_bare_material():
    """One voxel, explicitly: more carriers must mean a larger conductivity, not a smaller one."""
    materials = _materials()
    arrays, info, grid = _place(materials)
    coupling = PlasmaDispersion(index={"si": complex(N_SI, K_SI)}, wavelength=WAVELENGTH)
    base = float(materials["si"].electric_conductivity[0])
    _, report = coupling.perturb(arrays, info, materials, _carrier_samples(grid, 1.0e18, 1.0e18))
    assert report.perturbation.conductivity.min_sigma > base
    assert report.perturbation.conductivity.num_gain_points == 0


def test_removing_more_absorption_than_the_material_has_is_refused_as_gain():
    """A carrier *loss* below the material's own ``kappa_0`` would make it amplify; that is refused."""
    materials = _materials()
    arrays, info, grid = _place(materials)
    coupling = PlasmaDispersion(index={"si": complex(N_SI, K_SI)}, wavelength=WAVELENGTH)
    with pytest.raises(ValueError, match="negative electric conductivity"):
        coupling.perturb(arrays, info, materials, _carrier_samples(grid, -1.0e18, -1.0e18))


def test_gain_is_allowed_when_the_response_says_so():
    materials = _materials()
    arrays, info, grid = _place(materials)
    coupling = PlasmaDispersion(index={"si": complex(N_SI, K_SI)}, wavelength=WAVELENGTH, allow_gain=True)
    _, report = coupling.perturb(arrays, info, materials, _carrier_samples(grid, -1.0e18, -1.0e18))
    assert report.perturbation.conductivity.num_gain_points > 0
    assert report.perturbation.conductivity.min_sigma < 0.0


def test_tensor_constraints_still_run_on_the_real_part():
    """The loss lives in the conductivity, so the permittivity must still pass the lossless set."""
    materials = _materials()
    arrays, info, grid = _place(materials)
    coupling = PlasmaDispersion(index={"si": complex(N_SI, K_SI)}, wavelength=WAVELENGTH)
    _, report = coupling.perturb(
        arrays, info, materials, _carrier_samples(grid, 1.0e18, 1.0e18), constraints=TensorConstraints()
    )
    validation = report.perturbation.validation
    assert validation["constraints"] == {"real": True, "symmetric": True, "positive_definite": True}
    assert validation["num_checked"] > 0
    assert validation["min_eigenvalue"] > 0.0


class _IndefiniteResponse(LossyResponse):
    """A response that breaks positive definiteness, to prove the constraint still runs."""

    fields = ("C",)
    expects_unit = None

    def tensor(self, base, values):
        count = np.asarray(values["C"]).shape[0]
        out = np.broadcast_to(np.asarray(base, dtype=np.float64).reshape(3, 3), (count, 3, 3)).copy()
        out[:, 0, 0] = -1.0
        return out

    def conductivity(self, base, base_sigma, values):
        count = np.asarray(values["C"]).shape[0]
        return np.broadcast_to(np.asarray(base_sigma, dtype=np.float64), (count, 3)).copy()

    def unchanged(self, values):
        return np.zeros(np.asarray(values["C"]).shape[0], dtype=bool)


def test_tensor_constraints_refuse_an_indefinite_tensor_from_a_lossy_response():
    """Adding the conductivity channel must not let a response past the per-voxel physics check."""
    from fdtdx.coupling import PerturbationModel, perturb_arrays_with_model

    materials = _materials()
    arrays, info, grid = _place(materials)
    model = PerturbationModel(responses={"si": _IndefiniteResponse()})
    with pytest.raises(ValueError, match="physical constraints"):
        perturb_arrays_with_model(arrays, info, materials, {"C": _carrier_samples(grid, 1.0, 1.0)}, model)


# ------------------------------------------------------------------------------------------------
# declarations the coupling checks before it touches anything
# ------------------------------------------------------------------------------------------------
def test_a_material_that_is_not_the_declared_complex_index_is_refused():
    materials = _materials()
    coupling = PlasmaDispersion(index={"si": complex(N_SI, 10.0 * K_SI)}, wavelength=WAVELENGTH)
    with pytest.raises(ValueError, match="not the complex index"):
        coupling.check_materials(materials)


def test_the_carrier_unit_must_match_the_field_scale():
    materials = _materials()
    arrays, info, grid = _place(materials)
    coupling = PlasmaDispersion(index={"si": complex(N_SI, K_SI)}, wavelength=WAVELENGTH, field_scale=1.0)
    with pytest.raises(ValueError, match="field_scale"):
        coupling.perturb(arrays, info, materials, _carrier_samples(grid, 1.0e24, 1.0e24, unit="1/m^3"))


def test_carriers_in_per_cubic_metre_with_the_right_scale_agree_with_per_cubic_centimetre():
    materials = _materials()
    arrays, info, grid = _place(materials)
    per_cm3 = PlasmaDispersion(index={"si": complex(N_SI, K_SI)}, wavelength=WAVELENGTH)
    per_m3 = PlasmaDispersion(index={"si": complex(N_SI, K_SI)}, wavelength=WAVELENGTH, field_scale=1e-6)
    a, _ = per_cm3.perturb(arrays, info, materials, _carrier_samples(grid, 1.0e18, 1.0e18))
    b, _ = per_m3.perturb(arrays, info, materials, _carrier_samples(grid, 1.0e24, 1.0e24, unit="1/m^3"))
    assert np.array_equal(np.asarray(a.inv_permittivities), np.asarray(b.inv_permittivities))
    assert np.array_equal(np.asarray(a.electric_conductivity), np.asarray(b.electric_conductivity))


def test_a_lossless_scene_has_no_conductivity_array_to_write_into():
    """The loader allocates the array only for a scene that already absorbs; say so plainly."""
    materials = {
        "si": fdtdx.Material(permittivity=N_SI**2),
        "sio2": fdtdx.Material(permittivity=N_OX**2),
    }
    arrays, info, grid = _place(materials)
    assert arrays.electric_conductivity is None
    coupling = PlasmaDispersion(index={"si": complex(N_SI, 0.0)}, wavelength=WAVELENGTH)
    with pytest.raises(ValueError, match="no conductivity array"):
        coupling.perturb(arrays, info, materials, _carrier_samples(grid, 1.0e18, 1.0e18))


def test_a_lossless_response_leaves_no_conductivity_block_in_the_report():
    """Every response outside this module must leave a run's record exactly as it was."""
    materials = {"si": fdtdx.Material(permittivity=N_SI**2), "sio2": fdtdx.Material(permittivity=N_OX**2)}
    arrays, info, grid = _place(materials)
    coupling = fdtdx.coupling.ThermoOptic(dn_dT={"si": 1.8e-4}, reference_temperature=300.0)
    samples = samples_from_callable(
        grid, lambda points: np.full(points.shape[0], 310.0), ("E0", "E1", "E2"), name="T", unit="K"
    )
    _, report = coupling.perturb(arrays, info, materials, samples)
    assert report.perturbation.conductivity is None
    assert "conductivity" not in report.perturbation.as_dict()


# ------------------------------------------------------------------------------------------------
# the sign convention, measured: a wave through the perturbed medium decays like exp(-alpha z)
# ------------------------------------------------------------------------------------------------
_DECAY_RES = 20e-9
_DECAY_PML = 10
_DECAY_SOURCE_Z = _DECAY_PML + 4
_DECAY_DET1_Z = 40
_DECAY_DET2_Z = 140
_DECAY_CELLS_Z = _DECAY_DET2_Z + 20 + _DECAY_PML
#: Holes chosen so the Soref-Bennett absorption is about 3000 cm^-1: enough decay over the two
#: microns between the detectors (a factor of about 0.55) that a 10 % error in alpha is obvious.
_DECAY_HOLES = 3.011e20


def _decay_scene(materials: dict, wave):
    grid = fdtdx.RectilinearGrid.custom(
        x_edges=np.arange(0.0, 3 * _DECAY_RES + 1e-12, _DECAY_RES),
        y_edges=np.arange(0.0, 3 * _DECAY_RES + 1e-12, _DECAY_RES),
        z_edges=np.arange(0.0, _DECAY_CELLS_Z * _DECAY_RES + 1e-12, _DECAY_RES),
    )
    config = fdtdx.SimulationConfig(
        grid=grid, time=120e-15, backend="cpu", dtype=jnp.float32, material_sampling="yee_smooth"
    )
    volume = fdtdx.SimulationVolume(partial_grid_shape=grid.shape, material=materials["si"], name="bulk")
    objects, constraints = [volume], []
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_DECAY_PML,
        override_types={"min_x": "periodic", "max_x": "periodic", "min_y": "periodic", "max_y": "periodic"},
    )
    bound_dict, bound_constraints = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(bound_constraints)
    objects.extend(bound_dict.values())

    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_DECAY_SOURCE_Z,)),
        ]
    )
    objects.append(source)
    for name, z_index in (("d1", _DECAY_DET1_Z), ("d2", _DECAY_DET2_Z)):
        detector = fdtdx.PhasorDetector(
            name=name,
            partial_grid_shape=(None, None, 1),
            wave_characters=(wave,),
            reduce_volume=True,
            components=("Ex",),
            exact_interpolation=True,
            plot=False,
        )
        constraints.extend(
            [
                detector.same_size(volume, axes=(0, 1)),
                detector.place_at_center(volume, axes=(0, 1)),
                detector.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(z_index,)),
            ]
        )
        objects.append(detector)
    return objects, constraints, config, grid


def test_conductivity_write_decays_like_the_analytic_wave():
    """The convention, measured: the fork's FDTD through a medium this module made lossy.

    A uniform carrier field raises silicon's extinction coefficient from 3.1e-5 to about 0.037.
    Two phasor detectors two micrometres apart in that medium must see the amplitude ratio
    ``exp(-alpha d)`` with ``alpha = 4 pi kappa / lambda``, the same ``kappa`` the response asked
    for. A sign slip in ``sigma = omega eps0 Im(eps)`` would grow the wave instead; a factor error
    in the loader's ``conductivity_spacing`` would move ``alpha`` by the resolution in metres.
    """
    materials = _materials()
    wave = fdtdx.WaveCharacter(wavelength=WAVELENGTH)
    objects, constraints, config, grid = _decay_scene(materials, wave)
    key = jax.random.PRNGKey(0)
    object_container, arrays, params, config, info = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, object_container, _ = fdtdx.apply_params(arrays, object_container, params, key)

    coupling = PlasmaDispersion(index={"si": complex(N_SI, K_SI)}, wavelength=WAVELENGTH)
    samples = samples_from_callable(
        grid,
        lambda points: np.broadcast_to(np.array([0.0, _DECAY_HOLES]), (points.shape[0], 2)),
        ("E0", "E1", "E2"),
        name="C",
        unit="1/cm^3",
    )
    arrays, report = coupling.perturb(arrays, info, materials, samples)

    dk = _deltas(0.0, _DECAY_HOLES)[1]
    kappa = K_SI + dk
    alpha_analytic = 4.0 * math.pi * kappa / WAVELENGTH
    assert alpha_analytic == pytest.approx(3.0e5, rel=0.1), "the test's own carrier level drifted"

    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=object_container, config=config, key=key)
    p1 = complex(arrays.detector_states["d1"]["phasor"][0, 0, 0])
    p2 = complex(arrays.detector_states["d2"]["phasor"][0, 0, 0])
    assert abs(p1) > 0.0 and abs(p2) > 0.0
    assert abs(p2) < abs(p1), f"the perturbed medium amplifies: |p2| = {abs(p2):.4e} >= |p1| = {abs(p1):.4e}"

    separation = (_DECAY_DET2_Z - _DECAY_DET1_Z) * _DECAY_RES
    alpha_measured = -math.log(abs(p2) / abs(p1)) / separation
    assert alpha_measured == pytest.approx(alpha_analytic, rel=0.10), (
        f"alpha measured {alpha_measured:.4e} 1/m against {alpha_analytic:.4e} 1/m "
        f"(kappa {kappa:.6g}, sigma {report.perturbation.conductivity.max_sigma:.6g} S/m)"
    )
