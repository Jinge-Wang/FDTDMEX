"""The engine seam: how each Kronos FEM simulator exposes the field it solved for.

Every other module of this package speaks DOLFINx — a function space and a degree-of-freedom
vector, evaluated at points (:mod:`fdtdx.coupling.fem`). This one file knows the rest: that a
thermal simulator keeps its Lagrange space on ``sim._V`` and its solved temperature on
``sim.T_dofs``, that an electrostatic one keeps the potential on ``sim.V_dofs``, that a mechanical
one keeps its displacement function in ``sim.solution["u"]``. None of those attributes appears
anywhere else in the package, and nothing here is modified or written to disk.

That containment is the point. Kronos today has five independent engines plus ``femCommon``, and a
combined FEM core with a split front end and back end is being built to replace them. When it lands
this file gains one adapter per physics, or is replaced outright, and no other module moves.

Each function returns a :class:`~fdtdx.coupling.fem.FemField` a coupling can sample directly::

    from fdtdx.coupling import ThermoOptic
    from fdtdx.coupling.kronos import thermal_temperature

    arrays, report = ThermoOptic(dn_dT=...).apply(thermal_temperature(sim), arrays, info, materials, grid)

**Units are the caller's claim, not the engine's.** A Kronos scene is drawn in whatever length unit
its author chose, so a potential gradient is volts per micrometre in a micrometre scene and volts
per metre in a metre one. The engine does not record which, so the unit is an argument here with the
micrometre scene as the default, and the response's own ``field_scale`` is checked against it at the
perturbation boundary (:func:`fdtdx.coupling.perturb.check_sample_units`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fdtdx.coupling.fem import FemField


def thermal_temperature(sim: Any, dofs: np.ndarray | None = None, unit: str = "K") -> FemField:
    """The temperature a Kronos ``thermalFEM.thSim`` has solved for, in process.

    Reads the private ``sim._V`` (the Lagrange space ``thAssembly.make_function_space`` built) and
    ``sim.T_dofs``; nothing in thermalFEM is modified and nothing is written to disk. ``dofs``
    overrides the vector, for a transient output row (``result["T_out"][i]``).

    Args:
        sim: A ``thSim`` after ``solve()``.
        dofs (np.ndarray | None): Optional vector to use instead of ``sim.T_dofs``.
        unit (str): Unit label.

    Returns:
        FemField: The wrapped temperature field, named ``"T"``.

    Raises:
        RuntimeError: If the simulator has no DOLFINx function space (no backend or no mesh) or
            has not been solved.
    """
    space = getattr(sim, "_V", None)
    if space is None:
        raise RuntimeError("thermalFEM simulator has no DOLFINx function space (no mesh built, or no backend)")
    vector = sim.T_dofs if dofs is None else dofs
    status = getattr(sim, "status", None)
    # thermalFEM reports "converged" after a solve, "failed" after a solver error (with a
    # zero-filled vector) and "not_run" before any solve; the last two carry no field.
    if dofs is None and status in ("failed", "not_run"):
        raise RuntimeError(f"thermalFEM simulator has not solved (status={status!r})")
    return FemField.from_dofs(space, np.asarray(vector), name="T", unit=unit)


def electrostatic_potential(sim: Any, dofs: np.ndarray | None = None, unit: str = "V") -> FemField:
    """The potential a Kronos ``electrostatFEM.esSim`` has solved for, in process.

    Reads the private ``sim._V`` and ``sim.V_dofs``, the electrostatic counterparts of the thermal
    pair. ``dofs`` overrides the vector.

    Args:
        sim: An ``esSim`` after ``solve()``.
        dofs (np.ndarray | None): Optional vector to use instead of ``sim.V_dofs``.
        unit (str): Unit label of the potential.

    Returns:
        FemField: The wrapped potential, named ``"V"``.

    Raises:
        RuntimeError: If the simulator has no DOLFINx function space or has not been solved.
    """
    space = getattr(sim, "_V", None)
    if space is None:
        raise RuntimeError("electrostatFEM simulator has no DOLFINx function space (no mesh built, or no backend)")
    vector = sim.V_dofs if dofs is None else dofs
    status = getattr(sim, "status", None)
    if dofs is None and status in ("failed", "not_run"):
        raise RuntimeError(f"electrostatFEM simulator has not solved (status={status!r})")
    return FemField.from_dofs(space, np.asarray(vector), name="V", unit=unit)


def electrostatic_field(sim: Any, dofs: np.ndarray | None = None, unit: str = "V/um") -> FemField:
    """``E = -grad(V)`` of a solved electrostatic potential, exactly, as a discontinuous field.

    The recovery is exact for the finite-element function rather than a projection: the gradient of
    a degree-``p`` Lagrange potential is a degree-``p - 1`` polynomial per cell and the
    discontinuous space of that degree holds it with no projection error
    (:meth:`fdtdx.coupling.fem.FemField.gradient_of`). It is therefore *discontinuous at every
    material interface of the mesh*, which is exactly where a Yee lattice point can land; run
    :func:`fdtdx.coupling.fem.facet_coincidence_report` on the samples before trusting them.

    Args:
        sim: An ``esSim`` after ``solve()``.
        dofs (np.ndarray | None): Optional potential vector to use instead of ``sim.V_dofs``.
        unit (str): Unit label of the field. The default is the micrometre scene's ``"V/um"``,
            because the gradient carries the mesh's own length unit and the engine does not record
            which that was; a scene drawn in metres passes ``"V/m"``.

    Returns:
        FemField: The wrapped field, named ``"E"`` — what
        :class:`~fdtdx.coupling.effects.Pockels` reads.
    """
    potential = electrostatic_potential(sim, dofs=dofs)
    return FemField.gradient_of(potential, scale=-1.0, name="E", unit=unit)


def mechanical_displacement(sim: Any, solution: Any | None = None, unit: str = "um") -> FemField:
    """The displacement a Kronos ``mechFEM.femSim`` has solved for, in process.

    mechFEM keeps the solved ``dolfinx.fem.Function`` itself rather than a dof vector, in
    ``sim.solution["u"]`` (the same object ``femPost.evaluate_displacement`` reads), so this wraps
    the function directly.

    The strain a photoelastic coupling reads is one step further:
    ``FemField.symmetric_gradient_of(u, out_of_plane=...)``, whose ``out_of_plane`` argument is the
    plane-strain modelling choice a two-dimensional solve cannot make for the caller.

    Args:
        sim: A ``femSim`` after its solve.
        solution: Optional ``dolfinx.fem.Function`` to use instead of ``sim.solution["u"]``.
        unit (str): Unit label of the displacement; the mesh's own length unit.

    Returns:
        FemField: The wrapped displacement, named ``"u"``.

    Raises:
        RuntimeError: If the simulator carries no displacement solution.
    """
    function = solution
    if function is None:
        store = getattr(sim, "solution", None) or {}
        function = store.get("u") if hasattr(store, "get") else None
    if function is None:
        raise RuntimeError("mechFEM simulator has no displacement solution (solution['u'] is empty)")
    return FemField(function, name="u", unit=unit)
