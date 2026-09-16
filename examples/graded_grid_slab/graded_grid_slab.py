"""Mesh override regions: a graded grid for a plane wave through a dielectric slab.

Runs the same problem - a normally incident plane wave hitting an eps = 4 half space - on three z
discretizations and prints the transmission each one measures:

* ``coarse``  50 nm everywhere,
* ``fine``    12.5 nm everywhere along z,
* ``graded``  50 nm background with a 12.5 nm ``RefinementRegion`` over the slab.

Every discretization is run twice, once with the slab and once in vacuum, and the transmission is
the ratio of the two time-averaged Poynting fluxes at the same detector. That two-run normalization
removes the source amplitude and the detector's own grid weighting, so the three numbers are
directly comparable with each other and with the Fresnel value 4 n / (1 + n)^2 = 8/9.

Run: python examples/graded_grid_slab/graded_grid_slab.py
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx
from fdtdx.constants import c as c0

WAVELENGTH = 1.0e-6
COARSE = 50e-9
FINE = 12.5e-9
PERMITTIVITY = 4.0

DOMAIN_XY = 4 * COARSE
DOMAIN_Z = 4.0e-6
PML_THICKNESS = 500e-9

INTERFACE_Z = 0.0
SOURCE_Z = -1.5e-6
DETECTOR_Z = 0.5e-6
REFINED_MARGIN = 0.3e-6

SIM_TIME = 120e-15
AVERAGED_PERIODS = 10


def grid_policy(kind: str):
    """The three discretizations. Only the z axis differs; x and y stay at 50 nm everywhere."""
    if kind == "coarse":
        return fdtdx.UniformGrid(spacing=COARSE)
    if kind == "fine":
        return fdtdx.QuasiUniformGrid(dx=COARSE, dy=COARSE, dz=FINE)
    if kind == "graded":
        return fdtdx.GradedGrid(
            spacing=COARSE,
            regions=(
                fdtdx.RefinementRegion(
                    spacing=(COARSE, COARSE, FINE),
                    z=(INTERFACE_Z - REFINED_MARGIN, DOMAIN_Z / 2),
                ),
            ),
            max_ratio=1.4,
        )
    raise ValueError(f"Unknown grid kind: {kind}")


def build(kind: str, with_slab: bool):
    config = fdtdx.SimulationConfig(grid=grid_policy(kind), time=SIM_TIME, dtype=jnp.float32)
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(partial_real_shape=(DOMAIN_XY, DOMAIN_XY, DOMAIN_Z))
    objects.append(volume)

    # The same physical PML thickness on every grid, so the three runs differ only in cell size.
    pml_cells = round(PML_THICKNESS / (FINE if kind == "fine" else COARSE))
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=pml_cells,
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
        wave_character=fdtdx.WaveCharacter(wavelength=WAVELENGTH),
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            fdtdx.RealCoordinateConstraint(object=source.name, axes=(2,), sides=("-",), coordinates=(SOURCE_Z,)),
        ]
    )
    objects.append(source)

    if with_slab:
        slab = fdtdx.UniformMaterialObject(name="slab", material=fdtdx.Material(permittivity=PERMITTIVITY))
        constraints.extend(
            [
                slab.same_size(volume, axes=(0, 1)),
                slab.place_at_center(volume, axes=(0, 1)),
                # Both faces in metres. On a graded grid a single metric length would be converted
                # with the cells at the domain's lower corner, which are not the cells the object
                # sits on; pinning the two faces is exact everywhere.
                fdtdx.RealCoordinateConstraint(
                    object="slab",
                    axes=(2, 2),
                    sides=("-", "+"),
                    coordinates=(INTERFACE_Z, DOMAIN_Z / 2),
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
            fdtdx.RealCoordinateConstraint(object="flux", axes=(2,), sides=("-",), coordinates=(DETECTOR_Z,)),
        ]
    )
    objects.append(detector)
    return objects, constraints, config


def mean_flux(kind: str, with_slab: bool) -> tuple[float, tuple[int, int, int]]:
    objects, constraints, config = build(kind, with_slab)
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = fdtdx.place_objects(objects, config, constraints, key)
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj_container, config=config, key=key)
    flux = np.asarray(arrays.detector_states["flux"]["poynting_flux"][:, 0])
    steps_per_period = round(WAVELENGTH / (c0 * config.time_step_duration))
    return float(np.mean(flux[-AVERAGED_PERIODS * steps_per_period :])), config.resolved_grid.shape


def main() -> None:
    policy = grid_policy("graded")
    print(policy.summary((DOMAIN_XY, DOMAIN_XY, DOMAIN_Z)))
    print()

    analytic = 4.0 * np.sqrt(PERMITTIVITY) / (1.0 + np.sqrt(PERMITTIVITY)) ** 2
    results: dict[str, tuple[int, float]] = {}
    for kind in ("coarse", "fine", "graded"):
        reference, _ = mean_flux(kind, with_slab=False)
        transmitted, shape = mean_flux(kind, with_slab=True)
        results[kind] = (shape[2], transmitted / reference)

    print(f"{'grid':>8}  {'z cells':>8}  {'T':>9}  {'vs fine':>9}")
    for kind, (cells, transmission) in results.items():
        difference = "" if kind == "fine" else f"{abs(transmission - results['fine'][1]):9.2e}"
        print(f"{kind:>8}  {cells:>8}  {transmission:9.6f}  {difference:>9}")
    print(f"{'analytic':>8}  {'':>8}  {analytic:9.6f}")


if __name__ == "__main__":
    main()
