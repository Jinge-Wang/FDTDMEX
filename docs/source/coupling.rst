=================================
Multiphysics coupling (one-way)
=================================

``fdtdx.coupling`` feeds a field solved by another engine into the material loader. The first
use is thermo-optic tuning: a temperature field from a finite-element heat solver (Kronos
thermalFEM, built on DOLFINx) changes each material's refractive index, ``n(T) = n + dn/dT (T - T_ref)``,
and the resonance of a device moves accordingly. The coupling is one way: the fields do not act
back on the heat problem.

Two pieces
==========

**Point evaluation with coverage flags** (``fdtdx.coupling.fem_field``). A DOLFINx scalar
function is evaluated at the points of the Yee lattices: one bounding-box tree, one collision
query, one basis evaluation per call, no Python loop over points. Every point carries a flag
saying whether it lies inside the mesh; an uncovered point is ``NaN``, never a silent zero. The
result, :class:`~fdtdx.coupling.YeeLatticeSamples`, holds one value array and one coverage mask
per lattice (``E0``, ``E1``, ``E2``, ``H0``, ``H1``, ``H2`` and the cell-vertex lattice ``V``)
together with the grid edges, and it round-trips through a ``.npz`` file, so it is also the
artefact that crosses a process boundary when the two solvers do not share an interpreter.

.. code-block:: python

    from fdtdx.coupling import FemScalarField, PointTransform, sample_on_yee_lattices

    field = FemScalarField.from_thermal_sim(sim)          # a solved thermalFEM thSim
    # or: FemScalarField(dolfinx_function) / FemScalarField.from_dofs(V, T_dofs)
    samples = sample_on_yee_lattices(
        field, config.resolved_grid, lattices=("E0", "E1", "E2", "V"),
        transform=PointTransform(collapse_axes=(2,)),       # a 2-D mesh sampled from a one-cell-thick grid
    )
    samples.coverage_report()                              # per lattice: points, uncovered, min, max
    samples.save("temperature_on_yee.npz")

A ``PointTransform`` (offset, scale, collapsed axes) or a ``RadialPlaneTransform`` (distance from
a ring axis, fixed height, for an axisymmetric thermal solve) maps the loader's coordinates onto
the mesh's frame explicitly.

**Perturbation after the interface blend** (``fdtdx.coupling.thermo_optic``). The loader assembles
inverse permittivities per Yee point and, under ``material_sampling="yee_smooth"``, replaces every
two-material pixel by its Kottke blend. The perturbation is applied to those arrays:

* a bulk point gets the closed form ``1 / n(T)^2`` of the material it sampled;
* a blended pixel or vertex is **re-blended**: the smoother records the fill fraction, the unit
  normal and the material pair of every pixel it wrote (``SmoothingRecord``), and the same Kottke
  formulas are evaluated with both materials' permittivities at that point's temperature. The
  geometry is untouched, the smooth field is sampled once per pixel, and the diagonal entries and
  the vertex off-diagonal entries are treated alike.

.. code-block:: python

    from fdtdx.coupling import ThermoOpticCoefficients, perturb_arrays

    objects, arrays, params, config, info = fdtdx.place_objects(...)
    arrays = fdtdx.extend_material_to_pml(objects=objects, arrays=arrays)
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)
    coefficients = ThermoOpticCoefficients(dn_dT={"core": 1.86e-4, "bg": 1e-5}, reference_temperature=300.0)
    arrays, report = perturb_arrays(arrays, info, materials, samples, coefficients, uncovered="error")
    report.as_dict()   # points rewritten, pixels re-blended, uncovered counts, largest dT and dn

``info["yee_material_map"]`` (from ``place_objects`` under any ``yee`` sampling mode) carries the
material index sampled at every E point, the material table and the smoothing record the
perturbation needs. Coefficients are keyed by the names of the user's material dictionary and
matched to the loader's table by material value.

What the first version refuses
==============================

An error, never an approximation: the 9-component permittivity tier; off-diagonal placements other
than ``"node"``; an anisotropic or dispersive material with a coefficient; a tensor blend that
involves a perturbed material; two names of one material value with different coefficients; and,
under the default ``uncovered="error"``, a point that needs a temperature and lies outside the
mesh (``uncovered="unperturbed"`` leaves such points alone and counts them). Conductivity,
permeability and dispersive poles are not perturbed.

Order of operations, and Tidy3D
===============================

Tidy3D's ``perturbed_mediums_copy`` turns each perturbation medium into a spatially varying custom
medium first and lets the solver's sub-pixel averaging run on it afterwards. Re-blending the
recorded pixels is the same operation once the temperature is smooth across a pixel, and it costs
one pass over the interface set instead of a second geometry pass. A uniform temperature through
this layer reproduces, entry for entry, a scene drawn with the perturbed indices (this is one of
the unit tests, at ``rtol=1e-12`` including the vertex entries).

Gradients
=========

The loader is host-side NumPy and ``place_objects`` wraps the static permittivity in
``stop_gradient``, so no gradient flows from the temperature to the fields. The evaluation is
linear in the FEM degrees of freedom (a fixed sparse matrix once the sample points are fixed) and
the perturbation is elementwise plus a gather/scatter over the recorded pixels, so a traced
version is possible; it is not implemented.
