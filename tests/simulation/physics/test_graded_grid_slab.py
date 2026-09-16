"""Plane wave through a dielectric slab on a graded grid with a mesh override region.

Three z discretizations of the same physical problem, all with the same transverse grid and the
same periodic/PML boundaries:

===========  =====================================================================
``coarse``   50 nm everywhere (20 cells per vacuum wavelength, 10 in the slab)
``fine``     12.5 nm everywhere along z
``graded``   50 nm background with a 12.5 nm refinement region over the slab and
             a 300 nm margin in front of it, graded back to 50 nm on the vacuum
             side
===========  =====================================================================

Each discretization is run twice - once with the slab and once in vacuum - and the transmission is
the ratio of the two time-averaged Poynting fluxes at the same detector, which removes the source
amplitude and the detector's own grid weighting.

The refinement covers the whole slab, not only the interface: the wavelength inside eps = 4 is half
the vacuum one, so the background 50 nm leaves only 10 cells per wavelength there. Ending the
refinement inside the slab instead (at 700 nm, just past the detector) measured T = 0.8959 against
the uniformly fine 0.8879 - a 0.8 % bias from the grading transition sitting in the high-index
medium. Sizing cells per material is issue #34; here the region is placed by hand.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.constants import c as c0

_WAVELENGTH = 1.0e-6
_COARSE = 50e-9
_FINE = 12.5e-9
_PERMITTIVITY = 4.0

_DOMAIN_XY = 4 * _COARSE
_DOMAIN_Z = 4.0e-6
_PML_THICKNESS = 500e-9

_INTERFACE_Z = 0.0
_SOURCE_Z = -1.5e-6
_DETECTOR_Z = 0.5e-6
_REFINED_MARGIN = 0.3e-6

_SIM_TIME = 120e-15
_AVERAGED_PERIODS = 10


def _grid(kind: str):
    if kind == "coarse":
        return fdtdx.UniformGrid(spacing=_COARSE)
    if kind == "fine":
        return fdtdx.QuasiUniformGrid(dx=_COARSE, dy=_COARSE, dz=_FINE)
    if kind == "graded":
        return fdtdx.GradedGrid(
            spacing=_COARSE,
            regions=(
                fdtdx.RefinementRegion(
                    spacing=(_COARSE, _COARSE, _FINE),
                    z=(_INTERFACE_Z - _REFINED_MARGIN, _DOMAIN_Z / 2),
                ),
            ),
            max_ratio=1.4,
        )
    raise ValueError(f"Unknown grid kind: {kind}")


def _pml_cells(kind: str) -> int:
    """PML thickness in cells that gives the same physical thickness on every grid."""
    return round(_PML_THICKNESS / (_FINE if kind == "fine" else _COARSE))


def _build(kind: str, with_slab: bool):
    config = fdtdx.SimulationConfig(grid=_grid(kind), time=_SIM_TIME, dtype=jnp.float32)
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(partial_real_shape=(_DOMAIN_XY, _DOMAIN_XY, _DOMAIN_Z))
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_pml_cells(kind),
        override_types={
            "min_x": "periodic",
            "max_x": "periodic",
            "min_y": "periodic",
            "max_y": "periodic",
        },
    )
    bound_dict, bound_constraints = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(bound_constraints)
    objects.extend(bound_dict.values())

    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            fdtdx.RealCoordinateConstraint(object=source.name, axes=(2,), sides=("-",), coordinates=(_SOURCE_Z,)),
        ]
    )
    objects.append(source)

    if with_slab:
        slab = fdtdx.UniformMaterialObject(
            name="slab",
            material=fdtdx.Material(permittivity=_PERMITTIVITY),
        )
        constraints.extend(
            [
                slab.same_size(volume, axes=(0, 1)),
                slab.place_at_center(volume, axes=(0, 1)),
                fdtdx.RealCoordinateConstraint(
                    object="slab",
                    axes=(2, 2),
                    sides=("-", "+"),
                    coordinates=(_INTERFACE_Z, _DOMAIN_Z / 2),
                ),
            ]
        )
        objects.append(slab)

    detector = fdtdx.PoyntingFluxDetector(
        name="flux",
        partial_grid_shape=(None, None, 1),
        direction="+",
        reduce_volume=True,
        plot=False,
    )
    constraints.extend(
        [
            detector.same_size(volume, axes=(0, 1)),
            detector.place_at_center(volume, axes=(0, 1)),
            fdtdx.RealCoordinateConstraint(object="flux", axes=(2,), sides=("-",), coordinates=(_DETECTOR_Z,)),
        ]
    )
    objects.append(detector)

    return objects, constraints, config


def _mean_flux(kind: str, with_slab: bool) -> float:
    objects, constraints, config = _build(kind, with_slab)
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj_container, config=config, key=key)
    flux = np.asarray(arrays.detector_states["flux"]["poynting_flux"][:, 0])
    steps_per_period = round(_WAVELENGTH / (c0 * config.time_step_duration))
    return float(np.mean(flux[-_AVERAGED_PERIODS * steps_per_period :]))


def _transmission(kind: str) -> float:
    reference = _mean_flux(kind, with_slab=False)
    assert reference > 0, f"Reference flux on the {kind} grid is not positive: {reference}"
    return _mean_flux(kind, with_slab=True) / reference


@pytest.fixture(scope="module")
def transmissions() -> dict[str, float]:
    return {kind: _transmission(kind) for kind in ("coarse", "fine", "graded")}


def test_graded_grid_matches_the_uniformly_fine_transmission(transmissions):
    """Refining only around the interface and the detector reproduces the fine-grid answer."""
    difference = abs(transmissions["graded"] - transmissions["fine"])
    assert difference < 1e-3, (
        f"graded T={transmissions['graded']:.6f}, fine T={transmissions['fine']:.6f}, difference={difference:.2e}"
    )


def test_graded_grid_is_closer_to_the_fine_answer_than_the_coarse_grid(transmissions):
    """The refinement has to buy something: the coarse background alone is further off."""
    graded_error = abs(transmissions["graded"] - transmissions["fine"])
    coarse_error = abs(transmissions["coarse"] - transmissions["fine"])
    assert graded_error < coarse_error, (
        f"graded T={transmissions['graded']:.6f} (error {graded_error:.2e}) is not closer to "
        f"fine T={transmissions['fine']:.6f} than coarse T={transmissions['coarse']:.6f} "
        f"(error {coarse_error:.2e})"
    )


def test_transmission_matches_the_fresnel_coefficient(transmissions):
    """All three grids stay within 5 % of T = 4 n1 n2 / (n1 + n2)^2."""
    n2 = float(np.sqrt(_PERMITTIVITY))
    analytic = 4.0 * n2 / (1.0 + n2) ** 2
    for kind, measured in transmissions.items():
        assert abs(measured - analytic) / analytic < 0.05, f"{kind}: T={measured:.4f}, analytic={analytic:.4f}"


def test_graded_grid_uses_fewer_cells_than_the_fine_grid():
    """The point of the override region: the fine answer without a fine grid everywhere."""
    extent = (_DOMAIN_XY, _DOMAIN_XY, _DOMAIN_Z)
    graded_cells = _grid("graded").resolve_extent(extent).shape[2]
    fine_cells = round(_DOMAIN_Z / _FINE)
    assert graded_cells < fine_cells, f"graded {graded_cells} cells along z, fine {fine_cells}"
