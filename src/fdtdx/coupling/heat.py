"""The electromagnetic field as a heat source: absorbed-power density, and the drive it stands for.

The heat a steady conduction problem consumes is a volumetric density, and a frequency-domain
electromagnetic solve gives it directly::

    q(i, j, k) = 1/2 omega eps0 sum_{c = x, y, z} Im(eps_c(i, j, k)) |E_c(i, j, k)|^2

Each component uses *its own* slot permittivity at *its own* Yee point: no interpolation to cell
centres, no scalar cell permittivity. That matters — a solver's own point-evaluation helper
typically interpolates all three components as if they sat at the cell centre, which carries a
half-cell error per component.

Two things this function refuses to paper over, both measured (report ``W1v_refutation.md``,
Apple M4 Pro, Kronos waveEMFDFD 1330c2a, SuperLU direct solves):

* **A lossy cell inside a perfectly matched layer (PML) is an error, not a caveat.** The formula
  uses the physical permittivity, while the operator inside the absorbing layer uses the stretched
  one, so the two no longer describe the same absorption: the discrepancy measured 60-76 %. Pass
  ``pml_mask`` and the overlap raises.
* **The identity holds over the whole non-PML domain, never over "the lossy cells".** Summing ``q``
  only where the material is lossy misses the node-averaged boundary slots and was measured 6.7 %
  off; summing over every non-PML cell agreed with the engine's own Poynting-flux difference to
  5e-16 - 5.8e-15 in one dimension.

**The multi-component discrepancy, and what it actually is.** W1v recorded that the identity slips
to 0.1-7 % "once all three electric-field components are live". Phase 2 reproduced it and narrowed
it: the number of live components is not the variable. What matters is whether the electric field
has a component along the flux-plane normal (the longitudinal component; ``E_x`` for a flux plane
at constant ``x``). Measured on waveEMFDFD 1330c2a, oblique incidence in the x-y plane with a
Bloch-periodic transverse axis, absorbing layers on x, a smooth loss profile with no material
interface, SuperLU direct solves, Apple M4 Pro:

* Every case with a zero longitudinal component agrees to machine precision — out-of-plane
  (s) polarisation at 10, 20, 30, 45 degrees, and any polarisation at normal incidence: the
  relative difference measured 4.4e-16 to 1.6e-15, at every resolution from 10 to 160 cells per
  wavelength. So the discrete power identity is exact, not approximate, in that whole family.
* With the longitudinal component live, ``sum q dV`` exceeds the flux difference. The excess is
  proportional to the longitudinal component's share of the absorbed power, and its **absolute
  size does not depend on the transverse period**: at 30 degrees and 40 cells per wavelength the
  excess was 1.35133e-4 in the run's own units for transverse periods of 1, 1.5, 2, 2.5, 3 and 4
  wavelengths, while the absorbed power itself grew from 1.07e-2 to 4.26e-2. It is a boundary term
  at the transverse periodic seam, not a bulk error in the density.
* So its *relative* size falls as the transverse period grows: 1.28e-2 (period 1 wavelength),
  6.4e-3 (2), 4.2e-3 (3), 3.2e-3 (4). It does not vanish with the cell size: 1.80e-2, 1.03e-2,
  6.4e-3, 4.4e-3, 3.4e-3 at 10, 20, 40, 80 and 160 cells per wavelength (period 2 wavelengths).
* Against the exact stratified-medium answer for the same smooth profile, the **flux difference**
  is the accurate number in that regime (2.3e-6 relative at 160 cells per wavelength) and this
  density is the one carrying the seam term. With a sharp material interface both numbers also
  carry the solver's interface rasterisation, which is what widened W1v's range to 7 %.

Practical consequence: report both numbers and their difference. Where a longitudinal component is
live, keep the transverse period at several wavelengths, and read the residual difference as the
seam diagnostic rather than as a convergence failure.

:func:`volumetric_heat_rate` and :func:`heater_power` are the other direction, for a scene whose
heat comes from a prescribed drive rather than from the optical field: a resistive heater's power
spread uniformly over its volume, and back.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from fdtdx.constants import eps0 as EPS0

#: Time convention of the field the density is computed from.
CONVENTIONS: tuple[str, str] = ("exp(-iwt)", "exp(+iwt)")

#: Slot order of a ``(3, Nx, Ny, Nz)`` field or permittivity.
SLOT_NAMES: tuple[str, str, str] = ("E0", "E1", "E2")


def _stack_slots(value: Any, what: str) -> Any:
    """Accept ``(3, ...)`` arrays or a mapping keyed by ``E0``/``E1``/``E2``; return a stack."""
    if isinstance(value, Mapping):
        missing = [name for name in SLOT_NAMES if name not in value]
        if missing:
            raise ValueError(f"{what} is missing the slots {missing}")
        return np.stack([np.asarray(value[name]) for name in SLOT_NAMES])
    if isinstance(value, (list, tuple)):
        return np.stack([np.asarray(p) for p in value])
    if getattr(value, "ndim", None) != 4 or int(value.shape[0]) != 3:
        raise ValueError(f"{what} must have shape (3, Nx, Ny, Nz), got {getattr(value, 'shape', type(value))}")
    return value


def assert_no_lossy_pml_overlap(
    eps_per_slot: Any,
    pml_mask: Any,
    tol: float = 0.0,
) -> None:
    """Raise if any cell that carries loss also lies inside an absorbing layer.

    Inside a perfectly matched layer the operator uses a coordinate-stretched permittivity while
    this formula uses the physical one, so a lossy material overlapping the layer makes the two
    disagree — measured 60-76 % on a one-dimensional slab whose absorber reached into the PML.
    The check is a separate function so a caller that traces
    :func:`absorbed_power_density` through a differentiation framework can run it once, outside the
    trace, on concrete arrays.

    Args:
        eps_per_slot: ``(3, Nx, Ny, Nz)`` complex permittivity, or a mapping keyed by slot.
        pml_mask: ``(Nx, Ny, Nz)`` boolean, true inside an absorbing layer.
        tol (float): Imaginary parts at or below this count as lossless.

    Raises:
        ValueError: If the shapes disagree, or if a lossy cell overlaps the mask.
    """
    eps = np.asarray(_stack_slots(eps_per_slot, "eps_per_slot"))
    mask = np.asarray(pml_mask, dtype=bool)
    if mask.shape != eps.shape[1:]:
        raise ValueError(f"pml_mask must be {eps.shape[1:]}, got {mask.shape}")
    lossy = np.abs(eps.imag) > tol
    overlap = lossy & mask[None]
    if not overlap.any():
        return
    count = int(np.count_nonzero(overlap.any(axis=0)))
    slots = [SLOT_NAMES[c] for c in range(3) if overlap[c].any()]
    where = np.argwhere(overlap.any(axis=0))[0]
    raise ValueError(
        f"{count} lossy cells lie inside the PML (slots {slots}, first at index {tuple(int(i) for i in where)}). "
        "The absorbed-power formula uses the physical permittivity while the operator inside the layer uses the "
        "stretched one; the two disagree by 60-76 % (W1v). Move the lossy material clear of the absorbing layer, "
        "or pass pml_mask=None and accept that the result is not the engine's absorption."
    )


def absorbed_power_density(
    E: Any,
    eps_per_slot: Any,
    omega: float,
    pml_mask: Any | None = None,
    eps0: float = EPS0,
    convention: str = "exp(-iwt)",
) -> Any:
    """Time-averaged absorbed power per unit volume, per cell.

    ``q = 1/2 omega eps0 sum_c Im(eps_c) |E_c|^2``, each component evaluated with its own slot
    permittivity at its own Yee point. Only ``abs``, ``*``, ``+`` and a reduction over the slot
    axis are used, so the same function runs on NumPy, JAX or MLX arrays and is differentiable
    where the array library is.

    **What the result is comparable with.** The discrete sum ``(q * cell_volume).sum()`` over the
    *whole non-PML domain* is the absorbed power the engine's own Poynting-flux difference reports;
    summing only over the cells whose material is lossy misses the node-averaged boundary slots and
    was measured 6.7 % low. See the module docstring for the one regime where the two numbers do
    not agree: a live longitudinal component (along the flux-plane normal) adds a periodic-seam
    term to this sum, 0.3-1.3 % of the absorbed power depending on the transverse period.

    Args:
        E: ``(3, Nx, Ny, Nz)`` complex electric field on the three E lattices, or a mapping keyed
            by ``"E0"``/``"E1"``/``"E2"``.
        eps_per_slot: The same layout, complex relative permittivity per slot. A real array is
            accepted and gives ``q = 0``.
        omega (float): Angular frequency, rad/s (or the run's nondimensional equivalent).
        pml_mask: ``(Nx, Ny, Nz)`` boolean, true inside an absorbing layer. When given, a lossy
            cell overlapping the layer raises (see :func:`assert_no_lossy_pml_overlap`) and the
            density is set to zero inside the layer, where this formula has no meaning. ``None``
            skips both, which is right only for a run with no absorbing layer or when the caller
            has already checked.
        eps0 (float): Vacuum permittivity; pass ``1.0`` for a nondimensional scene where
            ``c0 = eps0 = mu0 = 1``.
        convention (str): ``"exp(-iwt)"`` (the default, and waveEMFDFD's) where loss is a positive
            imaginary permittivity, or ``"exp(+iwt)"`` where it is negative.

    Returns:
        The ``(Nx, Ny, Nz)`` real density, in W/m^3 for SI inputs.

    Raises:
        ValueError: If the shapes disagree, if ``omega`` is not positive, if ``convention`` is not
            known, or if a lossy cell overlaps ``pml_mask``.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")
    if omega <= 0.0:
        raise ValueError(f"omega must be positive, got {omega}")
    field = _stack_slots(E, "E")
    eps = _stack_slots(eps_per_slot, "eps_per_slot")
    if tuple(field.shape) != tuple(eps.shape):
        raise ValueError(f"E is {tuple(field.shape)} and eps_per_slot is {tuple(eps.shape)}; they must match")

    if pml_mask is not None:
        assert_no_lossy_pml_overlap(eps, pml_mask)

    sign = 1.0 if convention == "exp(-iwt)" else -1.0
    loss = sign * eps.imag
    magnitude = abs(field) ** 2
    q = 0.5 * float(omega) * float(eps0) * (loss * magnitude).sum(axis=0)

    if pml_mask is not None:
        keep = np.asarray(pml_mask, dtype=bool)
        q = q * (~keep)
    return q


def total_absorbed_power(
    q: Any,
    cell_volume: float | Sequence[float] | Any,
) -> float:
    """The discrete integral of a density over the grid, ``(q * dV).sum()``.

    Args:
        q: The ``(Nx, Ny, Nz)`` density.
        cell_volume: One scalar for a uniform grid, or a ``(Nx, Ny, Nz)`` array of per-cell
            volumes for a rectilinear one.

    Returns:
        float: The absorbed power in the density's own unit times volume.
    """
    density = np.asarray(q, dtype=np.float64)
    volume = np.asarray(cell_volume, dtype=np.float64)
    if volume.ndim not in (0, 3) or (volume.ndim == 3 and volume.shape != density.shape):
        raise ValueError(f"cell_volume must be a scalar or {density.shape}, got shape {volume.shape}")
    return float((density * volume).sum())


def normalise_to_flux(
    q: Any,
    power_nondimensional: float,
    power_physical: float,
) -> Any:
    """Scale a nondimensional density onto a physical input power, without writing a unit conversion.

    A frequency-domain run in the ``c0 = eps0 = mu0 = 1`` system gives a density in its own units.
    Rather than tracking the conversion, the density is scaled by the ratio of the physical input
    power to the nondimensional input power the same run reports, so ``q`` lands in the unit the
    thermal solver's scene is drawn in (W/um^3 for a micrometre scene).

    Args:
        q: The nondimensional density.
        power_nondimensional (float): The run's own input power, e.g. its Poynting flux through the
            source plane.
        power_physical (float): The same power in the unit the thermal scene uses.

    Returns:
        The scaled density.

    Raises:
        ValueError: If the nondimensional power is zero.
    """
    if power_nondimensional == 0.0:
        raise ValueError("power_nondimensional is zero; there is nothing to normalise against")
    return q * (float(power_physical) / float(power_nondimensional))


# ------------------------------------------------------------------------------------------------
# a prescribed drive as a heat source
# ------------------------------------------------------------------------------------------------


def volumetric_heat_rate(power: float, volume: float) -> float:
    """Uniform volumetric heat source in W/m^3 from a drive power spread over a heater volume.

    Args:
        power (float): Electrical power dissipated in the heater, in watts.
        volume (float): Heater volume in cubic metres, including the out-of-plane length for a
            cross-section model.

    Returns:
        float: Volumetric heat rate in W/m^3.

    Raises:
        ValueError: If ``volume`` is not positive.
    """
    if volume <= 0:
        raise ValueError("the heater volume must be positive")
    return float(power) / float(volume)


def heater_power(rate: float, volume: float) -> float:
    """Drive power in watts from a volumetric heat rate; the inverse of :func:`volumetric_heat_rate`.

    Args:
        rate (float): Volumetric heat rate in W/m^3.
        volume (float): Heater volume in cubic metres.

    Returns:
        float: Power in watts.

    Raises:
        ValueError: If ``volume`` is not positive.
    """
    if volume <= 0:
        raise ValueError("the heater volume must be positive")
    return float(rate) * float(volume)
