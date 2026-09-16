"""Multiphysics coupling: a field solved by another engine, into and out of the material loader.

One module per stage of the pipeline, named for the stage. Read top to bottom, the package is the
data flow::

    source field  ->  frames  ->  Yee lattice samples  ->  response  ->  perturbed arrays  ->  solve
       (fem)         (frames)        (lattice)          (responses)       (perturb)
       (kronos)                                          (tensors)        (deform: the shape channel)
       (sources)

    two way:   EM field  ->  heat  ->  transfer back to the FEM  ->  loop
    out:       perturbed arrays  ->  export (FDFD, a mode solver, a reference engine)
    end:       readout

The physics lives in classes, not in module names: a case declares one
:class:`~fdtdx.coupling.effects.Coupling` per physical effect — the field it reads, the unit that
field must be in, its rank, and how each material answers — and calls ``sample``, ``perturb`` or
``apply``. The fields do not act back on the source problem unless a case drives the outer loop
itself (:mod:`fdtdx.coupling.loop`).
"""

# -- 1. the field comes in: the Yee lattices, the frames between the meshes, the FEM evaluation --
from fdtdx.coupling.fem import (
    DEFAULT_TOL_CELLS,
    FacetCoincidence,
    FacetCoincidenceReport,
    FemField,
    FemScalarField,
    facet_coincidence_report,
    sample_on_yee_lattices,
)
from fdtdx.coupling.frames import (
    PointTransform,
    RadialPlaneTransform,
    inverse_point_transform,
)
from fdtdx.coupling.lattice import (
    LATTICE_NAMES,
    PointSamples,
    YeeLatticeSamples,
    grid_edges,
    lattice_axes,
    lattice_coordinates,
    lattice_points,
    sample_axes,
    samples_from_callable,
    uniform_samples,
)

# isort: split
# -- 2. the adapters to the Kronos FEM engines (the only module that names their attributes) -----
from fdtdx.coupling.kronos import (
    electrostatic_field,
    electrostatic_potential,
    mechanical_displacement,
    thermal_temperature,
)

# isort: split
# -- 3. what a case declares: one object per coupled effect, and the field adapters they read ----
from fdtdx.coupling.effects import (
    PERMITTIVITY_LATTICES,
    Coupling,
    CouplingReport,
    MultiCoupling,
    Photoelastic,
    PlasmaDispersion,
    Pockels,
    ThermoOptic,
    check_coupling_units,
)
from fdtdx.coupling.sources import (
    CallableFieldSource,
    FemFieldSource,
    FieldSource,
    SamplesFieldSource,
    UniformFieldSource,
    as_field_source,
)

# isort: split
# -- 4. the constitutive laws, the sigma <-> Im(eps) map, and the tensor notation ----------------
from fdtdx.coupling.responses import (
    SOREF_BENNETT_1550,
    CompositeResponse,
    LossyResponse,
    MaterialResponse,
    PhotoelasticResponse,
    PlasmaDispersionResponse,
    PockelsResponse,
    SorefBennett,
    TensorConstraints,
    ThermoOpticResponse,
    extinction_from_sigma,
    sigma_from_extinction,
    sigma_from_im_permittivity,
)
from fdtdx.coupling.tensors import (
    VOIGT_LABELS,
    VOIGT_ORDER,
    cubic_photoelastic_matrix,
    isotropic_photoelastic_matrix,
    permute_tensor,
    photoelastic_from_stress_optic,
    stress_optic_from_photoelastic,
    tensor_from_voigt,
    voigt_from_tensor,
    voigt_index_permutation,
    voigt_permute,
    voigt_samples_from_tensor,
)

# isort: split
# -- 5. the engine: the loader's arrays rewritten after the interface blend ----------------------
from fdtdx.coupling.perturb import (
    OFFDIAG_BULK_POLICIES,
    ConductivityReport,
    OffdiagRecord,
    PerturbationModel,
    PerturbationReport,
    apply_conductivity_perturbation,
    apply_permittivity_perturbation,
    check_sample_units,
    perturb_arrays_with_model,
    perturb_conductivity_with_model,
)

# isort: split
# -- 6. the shape channel: a deformation moves the interfaces themselves -------------------------
from fdtdx.coupling.deform import (
    displace_polygon,
)

# isort: split
# -- 7. the material arrays out, to a frequency-domain solver or a reference engine --------------
from fdtdx.coupling.export import (
    complex_permittivity_slots,
    offdiag_drop_ratio,
    yee_arrays_to_cartesian,
)

# isort: split
# -- 8. the return direction: a grid array as a FEM source callable, and its exact transpose -----
from fdtdx.coupling.transfer import (
    MultilinearTransfer,
    cartesian_to_fem_source,
    fem_to_cartesian_design,
    multilinear_weights,
)

# isort: split
# -- 9. two-way: the field as a heat source, and the outer loop that closes it -------------------
from fdtdx.coupling.heat import (
    absorbed_power_density,
    assert_no_lossy_pml_overlap,
    heater_power,
    normalise_to_flux,
    total_absorbed_power,
    volumetric_heat_rate,
)
from fdtdx.coupling.loop import (
    CouplingConvergenceReport,
    alternating_solver,
    continuation_schedule,
    continuation_sweep,
    coupling_convergence_report,
)

# isort: split
# -- 10. what a case reads off the solve --------------------------------------------------------
from fdtdx.coupling.readout import (
    NEPER_TO_DB,
    delta_neff_from_phase,
    im_neff_from_loss_db_per_cm,
    intensity_fom,
    loss_db_per_cm,
    phase_per_power,
    phase_shift,
    pi_power,
    pi_power_length_product,
    plane_overlap_phase,
    reference_neff_between_planes,
    two_plane_delta_neff,
)

# The exports are grouped by pipeline stage, in the order above, rather than sorted as one list, so
# that the package's shape is visible here as well as in the tree (RUF022 is off for that reason).
__all__ = [  # noqa: RUF022
    # -- 1. the field comes in --------------------------------------------------------------
    "LATTICE_NAMES",
    "PointSamples",
    "YeeLatticeSamples",
    "grid_edges",
    "lattice_axes",
    "lattice_coordinates",
    "lattice_points",
    "sample_axes",
    "uniform_samples",
    "samples_from_callable",
    "PointTransform",
    "RadialPlaneTransform",
    "inverse_point_transform",
    "FemField",
    "FemScalarField",
    "sample_on_yee_lattices",
    "DEFAULT_TOL_CELLS",
    "FacetCoincidence",
    "FacetCoincidenceReport",
    "facet_coincidence_report",
    # -- 2. the Kronos FEM engines ----------------------------------------------------------
    "thermal_temperature",
    "electrostatic_potential",
    "electrostatic_field",
    "mechanical_displacement",
    # -- 3. what a case declares ------------------------------------------------------------
    "PERMITTIVITY_LATTICES",
    "Coupling",
    "CouplingReport",
    "ThermoOptic",
    "Pockels",
    "Photoelastic",
    "PlasmaDispersion",
    "MultiCoupling",
    "check_coupling_units",
    "FieldSource",
    "FemFieldSource",
    "UniformFieldSource",
    "CallableFieldSource",
    "SamplesFieldSource",
    "as_field_source",
    # -- 4. the constitutive laws, the sigma <-> Im(eps) map, their tensor notation ---------
    "MaterialResponse",
    "ThermoOpticResponse",
    "PockelsResponse",
    "PhotoelasticResponse",
    "LossyResponse",
    "PlasmaDispersionResponse",
    "CompositeResponse",
    "TensorConstraints",
    "SorefBennett",
    "SOREF_BENNETT_1550",
    "sigma_from_im_permittivity",
    "sigma_from_extinction",
    "extinction_from_sigma",
    "VOIGT_ORDER",
    "VOIGT_LABELS",
    "voigt_from_tensor",
    "tensor_from_voigt",
    "voigt_index_permutation",
    "permute_tensor",
    "voigt_permute",
    "cubic_photoelastic_matrix",
    "isotropic_photoelastic_matrix",
    "photoelastic_from_stress_optic",
    "stress_optic_from_photoelastic",
    "voigt_samples_from_tensor",
    # -- 5. the perturbation engine ---------------------------------------------------------
    "PerturbationModel",
    "PerturbationReport",
    "ConductivityReport",
    "OffdiagRecord",
    "OFFDIAG_BULK_POLICIES",
    "apply_permittivity_perturbation",
    "perturb_arrays_with_model",
    "apply_conductivity_perturbation",
    "perturb_conductivity_with_model",
    "check_sample_units",
    # -- 6. the shape channel ---------------------------------------------------------------
    "displace_polygon",
    # -- 7. the material arrays out ---------------------------------------------------------
    "yee_arrays_to_cartesian",
    "complex_permittivity_slots",
    "offdiag_drop_ratio",
    # -- 8. the return direction ------------------------------------------------------------
    "MultilinearTransfer",
    "multilinear_weights",
    "cartesian_to_fem_source",
    "fem_to_cartesian_design",
    # -- 9. two way: heat, and the outer loop -----------------------------------------------
    "absorbed_power_density",
    "total_absorbed_power",
    "normalise_to_flux",
    "assert_no_lossy_pml_overlap",
    "volumetric_heat_rate",
    "heater_power",
    "alternating_solver",
    "continuation_sweep",
    "continuation_schedule",
    "CouplingConvergenceReport",
    "coupling_convergence_report",
    # -- 10. the readout --------------------------------------------------------------------
    "plane_overlap_phase",
    "two_plane_delta_neff",
    "reference_neff_between_planes",
    "phase_shift",
    "delta_neff_from_phase",
    "phase_per_power",
    "pi_power",
    "pi_power_length_product",
    "loss_db_per_cm",
    "im_neff_from_loss_db_per_cm",
    "NEPER_TO_DB",
    "intensity_fom",
]
