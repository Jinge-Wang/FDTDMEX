"""What a case reads off a solve: the phase on a monitor plane, and the figures of merit on top.

The last stage of the pipeline. Two groups, kept in one place so every coupled case (thermo-optic,
Pockels, photoelastic) reports the same quantity the same way.

**The phase readout.** A perturbation of the material changes the propagation constant of a guided
mode. The cleanest way to read that change from two FDTD runs is the phase of the overlap of the
perturbed and the reference fields on a monitor plane,
``angle(sum_plane E_pert . conj(E_ref))``, taken on two planes along the waveguide: the source's
mode mismatch and any fixed phase offset are the same on both planes and cancel in the difference,
and ``delta n_eff = lambda (dphi_2 - dphi_1) / (2 pi (y_2 - y_1))``. The detector states are the
phasor detectors' dictionaries (``state["phasor"]`` of shape ``(1, n_wl, 6, nx, ny, nz)`` or without
the leading axis); the first three of the six components are E. The sign follows the phasor
convention of the detector.

**The figures of merit.** The arithmetic that turns a mode solve or a driven solve into the number a
device is judged by:

* *Phase.* An effective-index change ``delta n_eff`` over a length ``L`` is
  ``2 pi L delta n_eff / lambda`` of phase. Divided by the drive power it gives the phase per watt,
  and inverted at ``pi`` it gives ``P_pi``, the number a thermo-optic phase shifter is sold on.
* *Loss.* ``Im(n_eff)`` is an attenuation index; the field decays as
  ``exp(-2 pi Im(n_eff) z / lambda)``, which is ``20 log10(e) 2 pi Im(n_eff) / lambda`` decibels per
  unit length. This is Eq. 8 of Jokisch, Christiansen & Sigmund, JOSA B 41(2), A18 (2024).
* *The driven-solve objective.* ``Phi = log10 integral |E|^2`` over the waveguide region (their
  Eq. 9), evaluated on the field of a driven solve. Written against ``jax.numpy`` so it can be
  differentiated.

Sign conventions. ``Im(n_eff) > 0`` means loss here; the mode solver's sign follows the
``exp(+i k0 n_eff z)`` convention of the permittivity that was handed to it, so a cross-section
built with ``eps = (n^2 - kappa^2) - 2 i n kappa`` comes back with the opposite sign and must be
negated before it is passed in. :func:`loss_db_per_cm` is signed on purpose, so a sign slip shows up
as a negative loss rather than being hidden by an absolute value.
"""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import numpy as np

#: Decibels per neper of field amplitude, ``20 / ln(10)``.
NEPER_TO_DB = 20.0 / math.log(10.0)


# ------------------------------------------------------------------------------------------------
# the phase on a monitor plane
# ------------------------------------------------------------------------------------------------


def _e_phasor(state: dict[str, Any]) -> np.ndarray:
    phasor = np.asarray(state["phasor"])
    if phasor.ndim == 6:
        phasor = phasor[0]
    if phasor.ndim != 5:
        raise ValueError(f"expected a phasor array of shape (n_wl, 6, nx, ny, nz), got {phasor.shape}")
    return phasor[:, :3]


def plane_overlap_phase(
    state_perturbed: dict[str, Any], state_reference: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Perturbed-minus-reference phase per wavelength from the overlap of the two E fields on one plane.

    Returns the phase in (-pi, pi] and the reference power ``sum |E_ref|^2`` on the plane.
    """
    e_p, e_r = _e_phasor(state_perturbed), _e_phasor(state_reference)
    if e_p.shape != e_r.shape:
        raise ValueError(f"the two detector states differ in shape: {e_p.shape} vs {e_r.shape}")
    axes = tuple(range(1, e_p.ndim))
    overlap = np.sum(e_p * np.conj(e_r), axis=axes)
    power = np.sum(np.abs(e_r) ** 2, axis=axes)
    return np.angle(overlap), power


def two_plane_delta_neff(
    reference: tuple[dict[str, Any], dict[str, Any]],
    perturbed: tuple[dict[str, Any], dict[str, Any]],
    wavelengths: np.ndarray,
    distance: float,
) -> dict[str, np.ndarray]:
    """Effective-index change between two monitor planes ``distance`` apart (metres), per wavelength (metres).

    ``reference`` and ``perturbed`` hold the detector states of plane 1 and plane 2 of the two runs. The per-plane
    phases are unwrapped along the wavelength axis before the difference is taken.
    """
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    if distance <= 0:
        raise ValueError("distance between the planes must be positive")
    dphi_1, power_1 = plane_overlap_phase(perturbed[0], reference[0])
    dphi_2, power_2 = plane_overlap_phase(perturbed[1], reference[1])
    if wavelengths.shape != dphi_1.shape:
        raise ValueError(f"{wavelengths.size} wavelengths for {dphi_1.size} detector wavelengths")
    delta_phase = np.unwrap(dphi_2) - np.unwrap(dphi_1)
    delta_neff = delta_phase * wavelengths / (2.0 * np.pi * distance)
    return {
        "dphi_plane_1": dphi_1,
        "dphi_plane_2": dphi_2,
        "delta_phase": delta_phase,
        "delta_neff": delta_neff,
        "power_plane_1": power_1,
        "power_plane_2": power_2,
    }


def reference_neff_between_planes(
    state_plane_1: dict[str, Any],
    state_plane_2: dict[str, Any],
    wavelengths: np.ndarray,
    distance: float,
    guess: float,
) -> np.ndarray:
    """Effective index of one run from its phase advance between the planes, numerical dispersion included.

    The phase between planes is only known modulo 2 pi; the branch nearest ``guess`` (the expected index) is taken.
    Informational: the source's mode mismatch does not cancel here, unlike in :func:`two_plane_delta_neff`.
    """
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    e1, e2 = _e_phasor(state_plane_1), _e_phasor(state_plane_2)
    axes = tuple(range(1, e1.ndim))
    phi = np.unwrap(np.angle(np.sum(e2 * np.conj(e1), axis=axes)))
    turns = np.round((guess * 2.0 * np.pi * distance / wavelengths - np.abs(phi)) / (2.0 * np.pi))
    return (np.abs(phi) + 2.0 * np.pi * turns) * wavelengths / (2.0 * np.pi * distance)


# ------------------------------------------------------------------------------------------------
# phase
# ------------------------------------------------------------------------------------------------


def phase_shift(delta_neff: Any, length: float, wavelength: float) -> Any:
    """Phase accumulated over ``length`` by an effective-index change, in radians.

    Args:
        delta_neff (Any): Change of the real effective index (dimensionless). Scalar or array.
        length (float): Interaction length in metres.
        wavelength (float): Free-space wavelength in metres.

    Returns:
        Any: The phase shift in radians, same shape as ``delta_neff``.

    Raises:
        ValueError: If ``length`` or ``wavelength`` is not positive.
    """
    if length <= 0 or wavelength <= 0:
        raise ValueError("length and wavelength must be positive")
    return 2.0 * math.pi * length * delta_neff / wavelength


def delta_neff_from_phase(phase: Any, length: float, wavelength: float) -> Any:
    """Effective-index change that produces a given phase over ``length``.

    Args:
        phase (Any): Phase shift in radians.
        length (float): Interaction length in metres.
        wavelength (float): Free-space wavelength in metres.

    Returns:
        Any: The effective-index change.

    Raises:
        ValueError: If ``length`` or ``wavelength`` is not positive.
    """
    if length <= 0 or wavelength <= 0:
        raise ValueError("length and wavelength must be positive")
    return phase * wavelength / (2.0 * math.pi * length)


def phase_per_power(delta_neff: Any, length: float, wavelength: float, power: float) -> Any:
    """Phase over ``length`` per watt of drive power, in radians per watt.

    Args:
        delta_neff (Any): Effective-index change produced by ``power``.
        length (float): Interaction length in metres.
        wavelength (float): Free-space wavelength in metres.
        power (float): Drive power in watts that produced ``delta_neff``.

    Returns:
        Any: Radians per watt.

    Raises:
        ValueError: If ``power`` is not positive.
    """
    if power <= 0:
        raise ValueError("power must be positive")
    return phase_shift(delta_neff, length, wavelength) / power


def pi_power(delta_neff: Any, length: float, wavelength: float, power: float) -> Any:
    """Drive power needed for a phase shift of ``pi`` over ``length``, in watts.

    Assumes the index change is linear in the power, which a resistive heater's steady state is.

    Args:
        delta_neff (Any): Effective-index change produced by ``power``.
        length (float): Interaction length in metres.
        wavelength (float): Free-space wavelength in metres.
        power (float): Drive power in watts that produced ``delta_neff``.

    Returns:
        Any: ``P_pi`` in watts.
    """
    return math.pi / phase_per_power(delta_neff, length, wavelength, power)


def pi_power_length_product(delta_neff: Any, length: float, wavelength: float, power: float) -> Any:
    """``P_pi * L`` in watt-metres, the length-independent figure of merit.

    Args:
        delta_neff (Any): Effective-index change produced by ``power``.
        length (float): Interaction length in metres.
        wavelength (float): Free-space wavelength in metres.
        power (float): Drive power in watts that produced ``delta_neff``.

    Returns:
        Any: ``P_pi L`` in watt-metres.
    """
    return pi_power(delta_neff, length, wavelength, power) * length


# ------------------------------------------------------------------------------------------------
# loss
# ------------------------------------------------------------------------------------------------


def loss_db_per_cm(im_neff: Any, wavelength: float) -> Any:
    """Propagation loss in dB/cm from the attenuation index.

    ``20 log10(e) * 2 pi Im(n_eff) / lambda_cm``. Signed: a negative ``Im(n_eff)`` (the opposite
    time convention, or gain) returns a negative loss rather than being silently made positive.

    Args:
        im_neff (Any): Imaginary part of the effective index, positive for loss.
        wavelength (float): Free-space wavelength in metres.

    Returns:
        Any: Loss in decibels per centimetre.

    Raises:
        ValueError: If ``wavelength`` is not positive.
    """
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    wavelength_cm = wavelength * 100.0
    return NEPER_TO_DB * 2.0 * math.pi * im_neff / wavelength_cm


def im_neff_from_loss_db_per_cm(loss: Any, wavelength: float) -> Any:
    """Attenuation index that corresponds to a loss in dB/cm; the inverse of :func:`loss_db_per_cm`.

    Args:
        loss (Any): Loss in decibels per centimetre.
        wavelength (float): Free-space wavelength in metres.

    Returns:
        Any: ``Im(n_eff)``.

    Raises:
        ValueError: If ``wavelength`` is not positive.
    """
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    wavelength_cm = wavelength * 100.0
    return loss * wavelength_cm / (NEPER_TO_DB * 2.0 * math.pi)


# ------------------------------------------------------------------------------------------------
# the driven-solve objective
# ------------------------------------------------------------------------------------------------


def intensity_fom(
    field: Any,
    mask: Any | None = None,
    area_weights: Any | None = None,
    floor: float | None = None,
) -> jnp.ndarray:
    """``log10`` of the integrated field intensity over a region: the paper's Eq. 9 objective.

    ``Phi = log10( integral_Omega |E|^2 dr )``. Maximizing it maximizes the power the driven solve
    leaves inside the waveguide at the prescribed effective index, which is how the paper measures
    "this design guides well and absorbs little".

    Args:
        field (Any): Complex field, component axis first (``(3, ...)``), or a single component.
        mask (Any | None): Optional weight or indicator with the spatial shape of ``field``
            (without the component axis), selecting the region of integration.
        area_weights (Any | None): Optional cell areas, broadcastable the same way, so a
            non-uniform grid integrates rather than sums.
        floor (float | None): Lower clamp on the integral before the logarithm, so an all-zero
            field returns a large negative number instead of ``-inf``. ``None`` uses the smallest
            normal number of the accumulator's dtype, which is what a float32 run needs (a fixed
            ``1e-300`` underflows to zero there and the clamp does nothing).

    Returns:
        jnp.ndarray: The scalar objective.
    """
    values = jnp.asarray(field)
    intensity = jnp.abs(values) ** 2
    if values.ndim > 1 and values.shape[0] in (1, 3, 6):
        intensity = jnp.sum(intensity, axis=0)
    if mask is not None:
        intensity = intensity * jnp.asarray(mask)
    if area_weights is not None:
        intensity = intensity * jnp.asarray(area_weights)
    total = jnp.sum(intensity)
    tiny = float(jnp.finfo(total.dtype).tiny)
    minimum = tiny if floor is None else max(float(floor), tiny)
    return jnp.log10(jnp.maximum(total, minimum))
