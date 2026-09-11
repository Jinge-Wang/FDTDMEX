=================================
Multiphysics coupling
=================================

``fdtdx.coupling`` feeds a field solved by another engine into the material loader, and hands the
assembled material back out. The first use is thermo-optic tuning: a temperature field from a
finite-element heat solver (Kronos thermalFEM, built on DOLFINx) changes each material's refractive
index, ``n(T) = n + dn/dT (T - T_ref)``, and the resonance of a device moves accordingly. A case
that wants the fields to act back on the source problem drives the outer loop itself.

One module per stage of the pipeline
====================================

The package is the data flow, read top to bottom:

.. code-block:: text

    source field  ->  frames  ->  Yee lattice samples  ->  response  ->  perturbed arrays  ->  solve
       (fem)         (frames)        (lattice)          (responses)       (perturb)
       (kronos)                                          (tensors)        (deform: the shape channel)
       (sources)

    two way:   EM field  ->  heat  ->  transfer back to the FEM  ->  loop
    out:       perturbed arrays  ->  export (FDFD, a mode solver, a reference engine)
    end:       readout

.. list-table::
   :header-rows: 1
   :widths: 14 24 62

   * - module
     - stage
     - what is in it
   * - ``lattice``
     - the Yee lattices and the sample containers
     - ``lattice_points``, ``lattice_axes``, ``lattice_coordinates``, ``sample_axes``,
       ``grid_edges``, ``PointSamples``, ``YeeLatticeSamples``, ``uniform_samples``,
       ``samples_from_callable``
   * - ``frames``
     - coordinate frames between a source mesh and the grid
     - ``PointTransform``, ``RadialPlaneTransform``, ``inverse_point_transform``
   * - ``fem``
     - a finite-element field in, plus the seam diagnostic
     - ``FemField``, ``FemScalarField``, ``sample_on_yee_lattices``, ``facet_coincidence_report``
   * - ``kronos``
     - the adapters to the Kronos FEM engines
     - ``thermal_temperature``, ``electrostatic_potential``, ``electrostatic_field``,
       ``mechanical_displacement``
   * - ``sources``
     - field adapters
     - ``FieldSource``, ``FemFieldSource``, ``UniformFieldSource``, ``CallableFieldSource``,
       ``SamplesFieldSource``, ``as_field_source``
   * - ``responses``
     - the constitutive laws
     - ``MaterialResponse``, ``ThermoOpticResponse``, ``PockelsResponse``,
       ``PhotoelasticResponse``, ``CompositeResponse``, ``TensorConstraints``;
       ``LossyResponse``, ``PlasmaDispersionResponse``, ``SorefBennett``,
       ``sigma_from_extinction``, ``extinction_from_sigma``
   * - ``tensors``
     - Voigt notation and the crystal cut
     - ``voigt_from_tensor``, ``voigt_permute``, ``permute_tensor``,
       ``photoelastic_from_stress_optic``, ``voigt_samples_from_tensor``
   * - ``effects``
     - the API a case declares
     - ``Coupling``, ``ThermoOptic``, ``Pockels``, ``Photoelastic``, ``PlasmaDispersion``,
       ``MultiCoupling``, ``CouplingReport``
   * - ``perturb``
     - the engine that rewrites the loader's arrays
     - ``PerturbationModel``, ``apply_permittivity_perturbation``, ``perturb_arrays_with_model``,
       ``PerturbationReport``, ``check_sample_units``; ``apply_conductivity_perturbation``,
       ``perturb_conductivity_with_model``, ``ConductivityReport``
   * - ``deform``
     - the shape channel
     - ``displace_polygon``
   * - ``export``
     - material arrays out
     - ``yee_arrays_to_cartesian``, ``complex_permittivity_slots``, ``offdiag_drop_ratio``
   * - ``transfer``
     - grid arrays back to a FEM source
     - ``MultilinearTransfer``, ``multilinear_weights``, ``cartesian_to_fem_source``,
       ``fem_to_cartesian_design``
   * - ``heat``
     - the EM field as a heat source
     - ``absorbed_power_density``, ``total_absorbed_power``, ``normalise_to_flux``,
       ``assert_no_lossy_pml_overlap``, ``volumetric_heat_rate``, ``heater_power``
   * - ``loop``
     - two-way drivers
     - ``alternating_solver``, ``continuation_sweep``, ``continuation_schedule``,
       ``CouplingConvergenceReport``
   * - ``readout``
     - what a case reads off a solve
     - ``plane_overlap_phase``, ``two_plane_delta_neff``, ``phase_shift``, ``pi_power``,
       ``loss_db_per_cm``, ``intensity_fom``

Every public name is re-exported from ``fdtdx.coupling`` itself, which is what a case imports.

One class per coupled effect
============================

A :class:`~fdtdx.coupling.Coupling` is one physical effect, declared once: the field it reads, the
unit that field must be in, whether the field is a scalar, a vector or a second-rank tensor, and
how each of the user's materials answers to it. It then does the three steps a case needs --
``sample`` the field on the Yee lattices, ``perturb`` the placed arrays, ``apply`` for both.

.. code-block:: text

    FieldSource (protocol)          sample_on(grid, lattices, transform) -> YeeLatticeSamples
      FemFieldSource                a DOLFINx field, evaluated with a coverage flag per point
      UniformFieldSource            one constant everywhere, fully covered: the control run
      CallableFieldSource           an analytic fn(points) -> values
      SamplesFieldSource            samples already taken (an .npz artefact), grid checked

    Coupling                        field name + unit + rank; responses(); sample/perturb/apply;
                                    CouplingReport (the engine's counters + its own excursions)
      ThermoOptic(dn_dT, T_ref)     scalar field T [K]; n(T) = n + dn/dT (T - T_ref); refuses an
                                    anisotropic material, so an isotropic pair keeps the scalar blend
      Pockels(r, field_scale)       vector field E [V/m]; d(1/eps)_I = r_Ik E_k, contracted (6, 3);
                                    makes off-diagonal bulk entries, so offdiag_bulk decides
      Photoelastic(p, field_scale)  tensor field S [1]; d(1/eps)_I = p_IJ S_J, contracted (6, 6);
                                    turns the tensor into the Yee frame, then contracts to Voigt
      PlasmaDispersion(index, lam)  two-component field (N, P) [1/cm^3]; Soref-Bennett dn and
                                    dalpha through one complex index, so it writes the loader's
                                    conductivity as well as its permittivity
      MultiCoupling([...])          several of the above on one scene, in the order given

.. code-block:: python

    from fdtdx.coupling import PointTransform, ThermoOptic, thermal_temperature

    objects, arrays, params, config, info = fdtdx.place_objects(...)
    arrays = fdtdx.extend_material_to_pml(objects=objects, arrays=arrays)
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)

    coupling = ThermoOptic(dn_dT={"core": 1.86e-4, "bg": 1e-5}, reference_temperature=300.0)
    field = thermal_temperature(sim)                       # a solved thermalFEM thSim
    arrays, report = coupling.apply(
        field, arrays, info, materials, config.resolved_grid,
        transform=PointTransform(collapse_axes=(2,)),      # a 2-D mesh read by a one-cell-thick grid
        uncovered="error",
    )
    report.as_dict()      # points rewritten, pixels re-blended, uncovered counts, largest dT and dn

The same call takes a constant instead of a field -- ``coupling.apply(300.0, ...)`` is the control
run every coupled case checks first, and it must leave the loader's arrays bit for bit. When the
case wants the samples themselves (to save them, or to report their coverage), it splits the call::

    samples = coupling.sample(field, config.resolved_grid, ("E0", "E1", "E2", "V"), transform)
    samples.coverage_report()                              # per lattice: points, uncovered, min, max
    samples.save("temperature_on_yee.npz")
    arrays, report = coupling.perturb(arrays, info, materials, samples, uncovered="error")

Step 1: point evaluation with coverage flags
--------------------------------------------

(``fem``, ``lattice``, ``frames``.) A DOLFINx function is evaluated at the points of the Yee
lattices: one bounding-box tree, one collision query, one basis evaluation per call, no Python loop
over points. Every point carries a flag saying whether it lies inside the mesh; an uncovered point
is ``NaN``, never a silent zero. The result, :class:`~fdtdx.coupling.YeeLatticeSamples`, holds one
value array and one coverage mask per lattice (``E0``, ``E1``, ``E2``, ``H0``, ``H1``, ``H2`` and
the cell-vertex lattice ``V``) together with the grid edges, and it round-trips through a ``.npz``
file, so it is also the artefact that crosses a process boundary when the two solvers do not share
an interpreter.

A ``PointTransform`` (offset, scale, collapsed axes, axis permutation) or a
``RadialPlaneTransform`` (distance from a ring axis, fixed height, for an axisymmetric thermal
solve) maps the loader's coordinates onto the mesh's frame explicitly. The coupling applies that
transform twice: to the *positions* before evaluation, and -- for a vector or tensor field -- to
the sampled *components* afterwards (``PointTransform.apply_values``), because a value solved on
somebody else's mesh carries its components in that mesh's frame. Doing only the first half is a
silent error, which is why the class does both and records ``component_frame="yee"`` in the
provenance so an artefact read back is not turned twice.

A field recovered exactly per cell (``-grad(V)``, ``sym(grad u)``) is **discontinuous** at every
material interface of the source mesh, and a lattice point that lands on one of those takes
whichever cell the evaluator listed first. ``facet_coincidence_report`` walks the loader's own
smoothing record and lists the sample points a material facet passes through, with the jump each
tie-break chose between.

Step 2: perturbation after the interface blend
----------------------------------------------

(``perturb``, ``responses``.) The loader assembles inverse permittivities per Yee point and, under
``material_sampling="yee_smooth"``, replaces every two-material pixel by its Kottke blend. The
perturbation is applied to those arrays:

* a bulk point gets the inverse of the perturbed tensor of the material it sampled (for the
  thermo-optic response, the closed form ``1 / n(T)^2``);
* a blended pixel or vertex is **re-blended**: the smoother records the fill fraction, the unit
  normal and the material pair of every pixel it wrote (``SmoothingRecord``), and the same Kottke
  formulas are evaluated with both materials' tensors at that point's own field value. The
  geometry is untouched, the smooth field is sampled once per pixel, and the diagonal entries and
  the vertex off-diagonal entries are treated alike. Two isotropic tensors go through the scalar
  formulas (bit for bit what the loader wrote); anything else goes through the tensor blend.

``info["yee_material_map"]`` (from ``place_objects`` under any ``yee`` sampling mode) carries the
material index sampled at every E point, the material table and the smoothing record. Materials
are keyed by the names of the user's material dictionary and matched to the loader's table by
material value, never by name.

Step 3 (only for an absorbing effect): the conductivity, written the loader's way
---------------------------------------------------------------------------------

(``perturb``, ``responses``.) The inverse-permittivity arrays hold a **real** tensor, so an effect
that changes how much a medium *absorbs* has nowhere to write there. The loader keeps that in a
second array, ``electric_conductivity`` in siemens per metre, which the time-domain update reads
separately, and :class:`~fdtdx.coupling.LossyResponse` is :class:`~fdtdx.coupling.MaterialResponse`
plus one more method returning what goes into it at the same points. ``perturb_arrays_with_model``
runs that second write after its own pass; it is a no-op for every response that is not lossy, so a
lossless run's record is exactly what it was.

That second write follows a different rule from the first, because the loader does. The loader does
**not** smooth the conductivity: ``load_scene_on_yee_lattices`` gives every Yee point the
conductivity of the single material recorded there, including the pixels whose permittivity the
sub-pixel blend mixed, and scales the whole array by the grid resolution in metres
(``conductivity_spacing``, recorded in ``info["yee_material_map"]``). The perturbation does the
same -- one bulk lookup per point of a responding material, no interface blend, the same scale
factor -- because that is what makes a perturbed scene *equal* a scene drawn with the perturbed
complex index, which is what the unit tests gate on, in both arrays at once.

The convention is the fork's own and is used in both directions in one place:
``Material.from_complex_permittivity`` splits ``eps' + i eps''`` into the real part it stores and
``sigma = omega eps0 eps''`` under ``exp(-i omega t)``, so a positive extinction coefficient is
loss; ``sigma_from_extinction`` and ``extinction_from_sigma`` are the two halves, and
``complex_permittivity_slots`` folds the array back the same way on the way out to a mode solver.
That the sign absorbs rather than amplifies is measured, not asserted: a unit test runs the fork's
own FDTD through a slab this channel made lossy and fits ``exp(-alpha z)``.

:class:`~fdtdx.coupling.PlasmaDispersion` is the effect built on it. Free carriers lower silicon's
index and raise its absorption; the coupling reads one two-component ``(N, P)`` carrier field in
cm\ :sup:`-3`, turns it into ``dn`` and ``dalpha`` through the Soref-Bennett power laws
(:class:`~fdtdx.coupling.SorefBennett`), and writes both arrays from the one complex index
``(n0 + dn) + i (kappa0 + dk)``. It is declared with each responding material's *complex* index at
the operating wavelength, because a response is handed the permittivity only and cannot recover the
unperturbed ``kappa0`` from it; ``check_materials`` rebuilds that declaration with
``Material.from_refractive_index`` and refuses a scene whose material is not it. A perturbed
conductivity that comes out negative is refused by default -- an amplifying medium is a sign slip
until a response says ``allow_gain=True`` -- and the count of such points is reported either way::

    coupling = PlasmaDispersion(index={"si": 3.4757 + 3.0836e-5j}, wavelength=1.55e-6)
    arrays, report = coupling.perturb(arrays, info, materials, carrier_samples)
    report.perturbation.conductivity.as_dict()   # points written, max |dsigma|, min/max sigma,
                                                 # gain points, the loader's scale factor

Several effects at once
=======================

:class:`~fdtdx.coupling.MultiCoupling` puts a temperature and a strain (or a voltage) on one
scene. It is deliberately **not** a loop over ``apply``: the engine writes each perturbed tensor
from the *material table*, not from the array it is handed, so a second pass would overwrite the
first rather than compose with it. The stack composes at the response level instead -- one pass,
one report, and a :class:`~fdtdx.coupling.CompositeResponse` for every material that answers to
more than one field, which applies each effect in turn to the running per-point tensor::

    stack = MultiCoupling([ThermoOptic(dn_dT=..., reference_temperature=300.0),
                           Photoelastic(p=...)])
    arrays, report = stack.apply({"T": temperature, "S": strain}, arrays, info, materials, grid)
    [part.coupling for part in report.parts]      # ['ThermoOptic', 'Photoelastic']

A part whose field sits at its own null value (a reference temperature, a zero strain) is skipped
exactly, so the stack is bit for bit the other effect alone; that is also why the order is exactly
irrelevant in that case. With both effects live the order does matter -- a temperature acts on the
index and a strain on the impermeability, and those two operations do not commute -- and the
difference is the product of the two small parameters, which the unit test bounds rather than
assumes.

The Kronos engine seam
======================

(``kronos``.) One module knows how each engine exposes the field it solved for -- ``sim._V`` and
``sim.T_dofs`` for thermalFEM, ``sim.V_dofs`` for electrostatFEM, ``sim.solution["u"]`` for
mechFEM -- and nothing else in the package names those attributes. ``fem`` holds DOLFINx-level code
only (a function space plus a dof vector, evaluated at points), so it survives a change of FEM
engine unchanged and the switch costs one adapter file.

.. code-block:: python

    from fdtdx.coupling import electrostatic_field, mechanical_displacement, thermal_temperature

    temperature = thermal_temperature(th_sim)                     # K
    e_field = electrostatic_field(es_sim, unit="V/um")            # -grad(V), exact per cell
    strain = FemField.symmetric_gradient_of(mechanical_displacement(mech_sim), out_of_plane=0.0)

The unit is the caller's claim, not the engine's: a Kronos scene is drawn in whatever length unit
its author chose, the engine does not record which, and a response's ``field_scale`` is checked
against the sample's label at the perturbation boundary.

Unit and physics assertions
===========================

Each response states the unit its arithmetic is written in (``expects_unit``: ``"K"``, ``"V/m"``,
``"1"``) and the scale that reaches it (``field_scale``), and the samples' own label is checked
against the pair, so a field solved in volts per micrometre cannot reach a response written in
volts per metre without the ``1e6`` that converts it. A composite response states one requirement
per part.

The 3-component permittivity tier holds diagonal tensors, so what happens to an off-diagonal entry
at a bulk point is an explicit choice, ``offdiag_bulk``: ``"error"`` (the default) refuses with the
material named, ``"project"`` keeps the diagonal and records the dropped entry, the ratio it bears
to the splitting of the two axes it mixes and the mixing angle discarded, and ``"tensor"`` writes
the loader's 9-component tier instead, which drops nothing but needs the scene placed on that tier.

Every perturbed tensor is validated per voxel against the physics it models before it is written
(:class:`~fdtdx.coupling.TensorConstraints`): a lossless dielectric response keeps the
permittivity real, symmetric and positive definite, and a violation raises with the material and
lattice named. A lossy reciprocal medium (complex symmetric, passive) or a gyrotropic one
(Hermitian, antisymmetric imaginary part) would relax those flags explicitly; the static loader
carries the real part only. The report records the constraints checked, how many tensors, and the
smallest eigenvalue seen.

Vector-valued fields are sampled by the same :class:`~fdtdx.coupling.FemField`;
``FemField.gradient_of(potential, scale=-1)`` gives the electrostatic field of a solved potential
exactly, on a discontinuous vector space one degree lower, and
``FemField.symmetric_gradient_of(displacement)`` gives the strain a photoelastic coupling reads.

What the first version refuses
==============================

An error, never an approximation: the 9-component permittivity tier under any policy but
``offdiag_bulk="tensor"``; off-diagonal placements other than ``"node"``; a dispersive responding
material; an anisotropic material under ``ThermoOptic``, whose one ``dn/dT`` per material has
nothing to say about a birefringent base (``Pockels`` and ``Photoelastic`` take one); two names of
one material value with different responses; and, under the default ``uncovered="error"``, a point
that needs a field value and lies outside the mesh (``uncovered="unperturbed"`` leaves such points
alone and counts them). Permeability and dispersive poles are not perturbed; the electric
conductivity is, but only by a ``LossyResponse`` and only on the 3-component tier -- the
1-component tier holds one row for all three E lattices and the 9-component tier is a
conductivity tensor, which no response here produces, so both are refused by name.

Order of operations, and Tidy3D
===============================

Tidy3D's ``perturbed_mediums_copy`` turns each perturbation medium into a spatially varying custom
medium first and lets the solver's sub-pixel averaging run on it afterwards. Re-blending the
recorded pixels is the same operation once the temperature is smooth across a pixel, and it costs
one pass over the interface set instead of a second geometry pass. A uniform temperature through
this layer reproduces, entry for entry, a scene drawn with the perturbed indices (this is one of
the unit tests, at ``rtol=1e-12`` including the vertex entries).

The material arrays out, and back
=================================

(``export``, ``transfer``.) ``yee_arrays_to_cartesian`` hands the assembled permittivity -- sub-pixel
smoothing, interface blends and all -- to a frequency-domain solver, a mode solver or a reference
engine, so it is given exactly the structure the loader built rather than re-rasterising the same
geometry with its own conventions. It does not invert the off-diagonal tier (those entries are
entries of ``eps^-1``; inverting a tensor entry by entry is wrong) and it does not fold the
conductivity in by itself: ``complex_permittivity_slots`` is the explicit one-line step that
combines them at a stated angular frequency.

The return direction is one multilinear interpolation, written as explicit weights rather than
hidden in a closure, because an adjoint chain needs the exact transpose of the forward map and a
transpose is only exact if it uses the same weights. ``MultilinearTransfer`` holds them and exposes
``forward`` and ``transpose``; ``cartesian_to_fem_source`` wraps an array as the ``(3, N) -> (N,)``
callable a Kronos solver takes.

Two-way coupling
================

(``heat``, ``loop``.) A two-way coupling is a fixed point: one outer iteration composes an
electromagnetic solve, the absorbed-power assembly, a thermal solve and the thermo-optic map, and
the coupled solution is the state that maps to itself. Two things the drivers keep straight, both
measured on an independent one-dimensional etalon model:

* **Damped Picard tracks a branch; Anderson does not.** Plain Picard failed to converge in 500
  iterations at a strongly driven operating point, at every damping from 1.0 down to 0.3, while
  Anderson of depth 3 converged in 14 -- but Anderson started cold and Anderson started hot
  converged to *different* fixed points at the same drive (peak temperature rise 63.93 K against
  24.91 K). A continuation that follows one branch takes small steps with damped Picard from the
  previous solution; Anderson polishes a residual once the branch is fixed.
* **Bistability is the observable, not a convergence nuisance.** Two coexisting states at one drive
  differed by a factor 2.56 in absorbed fraction. ``continuation_sweep`` sweeps a control parameter
  up and then down, each point started from its neighbour's solution, and reports where the two
  directions disagree -- the hysteresis window.

``absorbed_power_density`` evaluates ``q = 1/2 omega eps0 sum_c Im(eps_c) |E_c|^2`` with each
component at its own Yee point and its own slot permittivity. A lossy cell inside a perfectly
matched layer is refused rather than approximated: the formula uses the physical permittivity while
the operator inside the layer uses the stretched one, and the two were measured 60-76 % apart. The
identity holds over the whole non-PML domain and never over "the lossy cells" alone, which was
measured 6.7 % low.

Gradients
=========

The loader is host-side NumPy and ``place_objects`` wraps the static permittivity in
``stop_gradient``, so no gradient flows from the temperature to the fields. The evaluation is
linear in the FEM degrees of freedom (a fixed sparse matrix once the sample points are fixed) and
the perturbation is elementwise plus a gather/scatter over the recorded pixels, so a traced
version is possible; it is not implemented.

Phase readout and figures of merit
==================================

(``readout``.) The readout every tuned-waveguide case ends with: the phase of the overlap of the
perturbed and the reference E fields on a monitor plane (``plane_overlap_phase``), and the
effective-index change from two planes along the waveguide (``two_plane_delta_neff``), in which the
source's mode mismatch and any fixed phase offset cancel. ``reference_neff_between_planes`` reads
one run's own effective index from its phase advance, numerical dispersion included; it is
informational, since the mode mismatch does not cancel there.

The same module holds the arithmetic on top: phase over a length from an index change, phase per
watt, ``P_pi`` and ``P_pi L``, propagation loss in dB/cm from the attenuation index, and the
``log10`` integrated intensity a density-based topology optimization maximizes.

Differentiating the mode solver
===============================

The differentiable effective index is a feature of the mode solver, not of the coupling, and lives
next to it in :mod:`fdtdx.core.physics.mode_adjoint`. ``compute_mode`` reaches its backend through
``jax.pure_callback`` and wraps every array argument in ``jax.lax.stop_gradient``, so ``jax.grad``
through it returns **zero without raising**: on a 40 x 30 cross-section at 40 nm, scaling the whole
permittivity by a scalar, ``jax.grad`` gives ``0.000000`` where a central finite difference gives
``2.040029``.

``mode_neff`` is a ``jax.custom_vjp`` around the same solve. Its backward is the first-order
reciprocity sensitivity evaluated on the field the forward already returned, so it needs no extra
solve:

.. math::

    \frac{\partial n_{\mathrm{eff}}}{\partial \varepsilon_c}
      = \frac{1}{2}\, s_c\, \frac{E_c^2\, w}{\int (\mathbf{E}_t \times \mathbf{H}_t)\cdot\hat{e}_p\,\mathrm{d}A}

with ``s_c = -1`` on the propagation axis and ``+1`` on the two transverse axes, ``w`` the cell
area, and unconjugated products throughout. The unconjugated form is the one that stays correct for
a complex permittivity, and because the eigenvalue is holomorphic in the permittivity entries one
backward gives ``d Re(n_eff)/d eps`` and ``d Im(n_eff)/d eps`` together, i.e. phase and loss off one
solve.

.. code-block:: python

    from fdtdx.core.physics.mode_adjoint import ModeSolveSettings, mode_neff_parts

    settings = ModeSolveSettings.create(frequency=c / 1.55e-6, resolution=25e-9, mode_index=0)
    loss_index = jax.grad(lambda eps: mode_neff_parts(eps, settings)[1])(permittivity)

Scope: one, three or nine components, complex entries allowed (a metal cell has a negative real
part, which the material table cannot express but the array can). The native backend carries the two
transverse off-diagonal entries of a permittivity tensor exactly, so the nine-component tier has a
gradient for every entry, the reciprocity formula above generalising to the bilinear form
``0.5 s_a E_a E_b w / flux``. The four entries that couple a transverse axis to the propagation axis
cannot enter an eigenproblem linear in ``n_eff**2`` at all - the medium then has no mirror plane at
the cross-section and its dispersion relation gains odd powers of the propagation constant - so the
solver drops them with a ``ModeLongitudinalOffdiagWarning`` naming the entry and its magnitude, and
their sensitivity is reported as exactly zero, which is what the solved ``n_eff`` actually depends
on. The permeability is held constant.

The returned mode **fields** are differentiable too. ``mode_solve`` routes through the JAX-native
pipeline, where the tensor inversion, the axis rotation, the two-dimensional collapse, the field
reconstruction and the Poynting normalisation are all traced and only the eigen-solve is opaque; its
backward covers the eigenvectors with one bordered shifted solve per mode that carries a cotangent,
so an overlap-integral objective such as ``log10 sum |E|^2`` has a gradient. A setting that pipeline
cannot serve - the Tidy3D backend, ``filter_pol``, a magnetic wall, single precision - is named in a
warning rather than silently returning zeros, and ``differentiable_fields=False`` asks for the old
callback path deliberately.

``track_mode`` and ``tracked_mode_index`` follow one physical mode across a parameter sweep by field
overlap. Sorting by ``Re(n_eff)`` reorders the list the moment two indices cross, and a finite
difference taken at a fixed mode index across such a crossing returns the derivative of the wrong
mode. ``tracked_mode_index`` costs one eigen-solve regardless of how many candidates it compares: it
takes them from ``fdtdx.compute_modes``, which returns the sorted list the backend already computed
instead of throwing all but one entry away. Aiming the solve itself is ``target_neff``, an argument
of both ``compute_mode`` and ``compute_modes``: it becomes the shift-invert target and, when given,
the sort key, so ``mode_index=0`` selects the mode nearest that index rather than the one of highest
index.

A sweep should use ``ModeTracker`` rather than call ``track_mode`` by hand. It keeps the previous
step's field, selects the candidate with the largest overlap with it at every new parameter value,
and hands the selected index to the differentiable solve, so following the mode is the default and
not something the caller has to remember:

.. code-block:: python

    from fdtdx.core.physics.mode_adjoint import ModeSolveSettings, ModeTracker

    tracker = ModeTracker(ModeSolveSettings.create(frequency=c / 1.55e-6, resolution=60e-9))
    for temperature in sweep:
        neff = tracker.neff(permittivity_at(temperature))   # differentiable in the permittivity

Two robustness features sit under it. ``fdtdx.filter_spurious_modes`` (reachable from
``compute_mode`` and ``compute_modes`` as ``drop_spurious=True``, and on by default inside
``ModeTracker``) removes modes whose effective index exceeds the largest material index of the
cross-section, and modes that leave more than half their electric energy in the one-cell ring
against the solver's electric walls - the discretization's own wall solutions, which on a 40 x 30
strip at 40 nm are 7 of the first 22 entries. And a mode plane that reaches into the simulation's
PML is now reported: the mode solver has no PML, so it closes the cross-section with electric walls
and reflects the part of the mode the FDTD run would absorb. A diagonal permittivity is extended
into the PML region by the loader, so that case warns and names the overlap in cells; a fully
tensorial cross-section has no meaningful continuation there and raises.
