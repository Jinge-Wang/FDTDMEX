"""One-way multiphysics coupling into the material loader.

``fem_field`` evaluates a finite-element scalar field (a DOLFINx function, e.g. the temperature a
Kronos ``thermalFEM`` run solved for) at the Yee lattice points with a coverage flag per point.
``thermo_optic`` applies a per-material ``dn/dT`` to the assembled inverse-permittivity arrays after
the interface blend, re-blending the recorded interface pixels at their own temperature. The
fields do not act back on the thermal problem.
"""

from fdtdx.coupling.fem_field import (
    LATTICE_NAMES,
    FemField,
    FemScalarField,
    PointSamples,
    PointTransform,
    RadialPlaneTransform,
    YeeLatticeSamples,
    lattice_axes,
    lattice_points,
    sample_on_yee_lattices,
    samples_from_callable,
    uniform_samples,
)
from fdtdx.coupling.perturbation import (
    MaterialResponse,
    PerturbationModel,
    PerturbationReport,
    PhotoelasticResponse,
    PockelsResponse,
    TensorConstraints,
    ThermoOpticResponse,
    apply_permittivity_perturbation,
    perturb_arrays_with_model,
)
from fdtdx.coupling.thermo_optic import (
    ThermoOpticCoefficients,
    ThermoOpticReport,
    apply_thermo_optic_perturbation,
    perturb_arrays,
    perturbed_permittivity,
)

__all__ = [
    "LATTICE_NAMES",
    "FemField",
    "FemScalarField",
    "MaterialResponse",
    "PerturbationModel",
    "PerturbationReport",
    "PhotoelasticResponse",
    "PockelsResponse",
    "PointSamples",
    "PointTransform",
    "RadialPlaneTransform",
    "TensorConstraints",
    "ThermoOpticCoefficients",
    "ThermoOpticReport",
    "ThermoOpticResponse",
    "YeeLatticeSamples",
    "apply_permittivity_perturbation",
    "apply_thermo_optic_perturbation",
    "lattice_axes",
    "lattice_points",
    "perturb_arrays",
    "perturb_arrays_with_model",
    "perturbed_permittivity",
    "sample_on_yee_lattices",
    "samples_from_callable",
    "uniform_samples",
]
