"""Fit a :class:`~fdtdx.dispersion.DispersionModel` to measured optical constants.

Given tabulated :math:`n(\\lambda)` and :math:`k(\\lambda)` — typically a
refractiveindex.info record — :func:`fit_dispersion` returns the pole set whose
:math:`\\varepsilon(\\omega) = \\varepsilon_\\infty + \\sum_p \\chi_p(\\omega)`
best reproduces the data, ready to hand to
:class:`~fdtdx.materials.Material`.

Conventions
-----------
The engine uses ``exp(-i omega t)``, so the measured permittivity is
:math:`\\varepsilon = (n + i k)^2` and a passive material has
:math:`\\mathrm{Im}\\,\\varepsilon \\ge 0`. Every pole kind is parameterised so
its strength cannot go negative, which makes the fitted model passive by
construction; :attr:`FitResult.passive` re-checks that on a dense grid rather
than assuming it.

Method
------
Bounded nonlinear least squares (:func:`scipy.optimize.least_squares`, ``trf``)
on the stacked real and imaginary parts of :math:`\\varepsilon(\\omega)`.
Strictly positive quantities (resonance frequencies, oscillator strengths,
plasma frequencies, relaxation times) are optimised in ``log`` space so they
stay positive and so a search step is relative rather than absolute; damping
rates are optimised linearly with a lower bound of exactly zero, because a
transparent material's best fit really is :math:`\\gamma = 0`.

The optimiser is started from several data-driven guesses (resonances at the
peaks of :math:`\\mathrm{Im}\\,\\varepsilon`, a Drude term from the
low-frequency behaviour, a Debye term from a relaxation knee) across the
allowed mixes of pole kinds, plus seeded random jitter; the best start wins.

Reading the database
--------------------
:func:`read_refractiveindex_yaml` parses the CC0 refractiveindex.info YAML
record types this module needs — ``tabulated nk`` / ``n`` / ``k`` and the
dispersion formulas 1 (Sellmeier), 2 (Sellmeier-2) and 3 (polynomial).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from fdtdx.constants import c as c_light
from fdtdx.dispersion import DebyePole, DispersionModel, DrudePole, LorentzPole, Pole, SellmeierPole

#: Pole kinds :func:`fit_dispersion` can place.
#:
#: * ``"lorentz"`` — damped oscillator, 3 parameters.
#: * ``"sellmeier"`` — lossless Lorentz (``gamma = 0``), 2 parameters; the
#:   right choice inside a transparency window, where a free damping rate is
#:   unconstrained by the data.
#: * ``"drude"`` — free carriers, 2 parameters.
#: * ``"debye"`` — relaxation, 2 parameters.
POLE_KINDS: tuple[str, ...] = ("lorentz", "sellmeier", "drude", "debye")

_N_PARAMS = {"lorentz": 3, "sellmeier": 2, "drude": 2, "debye": 2}


@dataclass(frozen=True)
class FitResult:
    """Outcome of :func:`fit_dispersion`.

    Attributes:
        model: The fitted poles, as a :class:`~fdtdx.dispersion.DispersionModel`.
        eps_inf: High-frequency permittivity to pass as the
            :class:`~fdtdx.materials.Material` ``permittivity``.
        rms: Weighted root-mean-square of
            :math:`|\\varepsilon_\\text{fit} - \\varepsilon_\\text{data}|`
            over the fitted points.
        max_abs_error: Largest
            :math:`|\\varepsilon_\\text{fit} - \\varepsilon_\\text{data}|`.
        passive: Whether :math:`\\mathrm{Im}\\,\\varepsilon \\ge 0` holds over
            the fit range (checked on a dense grid).
        report: Human-readable summary: pole list with parameters, error
            metrics, passivity and — when ``dt`` was given — the time-step
            stability advisory.
    """

    model: DispersionModel
    eps_inf: float
    rms: float
    max_abs_error: float
    passive: bool
    report: str


# ---------------------------------------------------------------------------
# Parameter packing
# ---------------------------------------------------------------------------


def _bounds_for_kind(kind: str, omega: np.ndarray) -> tuple[list[float], list[float]]:
    """Lower/upper bounds for one pole's optimisation variables."""
    w_lo, w_hi = float(omega.min()), float(omega.max())
    log_de = (math.log(1e-8), math.log(1e4))
    # Resonances are allowed well outside the measured band (a transparency
    # window is shaped by UV and IR resonances it never samples) but not
    # arbitrarily far, where a pole degenerates into a constant offset that
    # eps_inf already carries.
    log_w = (math.log(0.02 * w_lo), math.log(50.0 * w_hi))
    if kind == "lorentz":
        return ([log_w[0], 0.0, log_de[0]], [log_w[1], 20.0 * w_hi, log_de[1]])
    if kind == "sellmeier":
        return ([log_w[0], log_de[0]], [log_w[1], log_de[1]])
    if kind == "drude":
        return ([log_w[0], math.log(1e-6 * w_lo)], [log_w[1], math.log(1e2 * w_hi)])
    if kind == "debye":
        return ([log_de[0], math.log(1e-3 / w_hi)], [log_de[1], math.log(1e3 / w_lo)])
    raise ValueError(f"Unknown pole kind {kind!r}; expected one of {POLE_KINDS}.")


def _pole_from_params(kind: str, params: np.ndarray) -> Pole:
    if kind == "lorentz":
        return LorentzPole(
            resonance_frequency=float(np.exp(params[0])),
            damping=float(params[1]),
            delta_epsilon=float(np.exp(params[2])),
        )
    if kind == "sellmeier":
        omega_0 = float(np.exp(params[0]))
        # C = (2 pi c / omega_0)^2, in m^2
        return SellmeierPole(B=float(np.exp(params[1])), C=(2.0 * math.pi * c_light / omega_0) ** 2)
    if kind == "drude":
        return DrudePole(plasma_frequency=float(np.exp(params[0])), damping=float(np.exp(params[1])))
    if kind == "debye":
        return DebyePole(delta_epsilon=float(np.exp(params[0])), relaxation_time=float(np.exp(params[1])))
    raise ValueError(f"Unknown pole kind {kind!r}; expected one of {POLE_KINDS}.")


def _model_from_x(
    x: np.ndarray,
    composition: Sequence[str],
    fixed_eps_inf: float | None,
) -> tuple[DispersionModel, float]:
    offset = 0
    if fixed_eps_inf is None:
        eps_inf = float(x[0])
        offset = 1
    else:
        eps_inf = float(fixed_eps_inf)
    poles = []
    for kind in composition:
        size = _N_PARAMS[kind]
        poles.append(_pole_from_params(kind, x[offset : offset + size]))
        offset += size
    return DispersionModel(poles=tuple(poles)), eps_inf


def _eps_of_model(model: DispersionModel, eps_inf: float, omega: np.ndarray) -> np.ndarray:
    chi = np.array([model.susceptibility(float(w)) for w in omega], dtype=np.complex128)
    return eps_inf + chi


def _eps_from_x(
    x: np.ndarray,
    composition: Sequence[str],
    fixed_eps_inf: float | None,
    omega: np.ndarray,
) -> np.ndarray:
    """Vectorised ``eps(omega)`` straight from the optimisation variables.

    Same values as building the poles and calling
    :meth:`DispersionModel.susceptibility`, but without constructing a pytree
    per residual evaluation — the inner loop of the fit runs thousands of
    times, so this is the difference between seconds and a minute.
    """
    offset = 0
    if fixed_eps_inf is None:
        eps = np.full(omega.shape, float(x[0]), dtype=np.complex128)
        offset = 1
    else:
        eps = np.full(omega.shape, float(fixed_eps_inf), dtype=np.complex128)
    for kind in composition:
        params = x[offset : offset + _N_PARAMS[kind]]
        offset += _N_PARAMS[kind]
        if kind in ("lorentz", "sellmeier"):
            omega_0 = np.exp(params[0])
            gamma = params[1] if kind == "lorentz" else 0.0
            delta_eps = np.exp(params[-1])
            eps += delta_eps * omega_0**2 / (omega_0**2 - omega**2 - 1j * gamma * omega)
        elif kind == "drude":
            omega_p = np.exp(params[0])
            gamma = np.exp(params[1])
            eps += -(omega_p**2) / (omega**2 + 1j * gamma * omega)
        elif kind == "debye":
            eps += np.exp(params[0]) / (1.0 - 1j * omega * np.exp(params[1]))
    return eps


# ---------------------------------------------------------------------------
# Start-point heuristics
# ---------------------------------------------------------------------------


def _lorentz_seeds(omega: np.ndarray, eps: np.ndarray, count: int) -> list[tuple[float, float, float]]:
    """``(omega_0, gamma, delta_epsilon)`` guesses from the peaks of Im eps.

    Peaks that the data actually resolves come first; the remainder are placed
    just outside the measured band, which is where a transparent material's
    resonances (UV electronic, IR phonon) actually sit.
    """
    seeds: list[tuple[float, float, float]] = []
    im = np.imag(eps)
    span = float(im.max() - im.min())
    if span > 0.0:
        peaks, props = find_peaks(im, prominence=0.05 * span, width=1)
        order = np.argsort(props["prominences"])[::-1]
        for idx in order:
            j = int(peaks[idx])
            w0 = float(omega[j])
            # width in samples -> width in rad/s (the data grid is not uniform)
            half = max(1, round(float(props["widths"][idx]) / 2.0))
            lo = max(0, j - half)
            hi = min(len(omega) - 1, j + half)
            gamma = max(float(omega[hi] - omega[lo]), 1e-3 * w0)
            delta_eps = max(float(im[j]) * gamma / w0, 1e-4)
            seeds.append((w0, gamma, delta_eps))
    w_lo, w_hi = float(omega.min()), float(omega.max())
    extra = [(1.5 * w_hi, 0.0, 1.0), (0.5 * w_lo, 0.0, 0.5), (4.0 * w_hi, 0.0, 2.0), (0.1 * w_lo, 0.0, 0.2)]
    i = 0
    while len(seeds) < count:
        seeds.append(extra[i % len(extra)])
        i += 1
    return seeds[:count]


def _drude_seed(omega: np.ndarray, eps: np.ndarray) -> tuple[float, float]:
    """``(omega_p, gamma)`` from the low-frequency limit ``eps ~ -wp^2/w^2``."""
    j = int(np.argmin(omega))
    w = float(omega[j])
    re, im = float(np.real(eps[j])), float(np.imag(eps[j]))
    omega_p = math.sqrt(max(-re, 1.0)) * w
    # Im eps = wp^2 gamma / w^3
    gamma = max(im * w**3 / omega_p**2, 1e-4 * w) if im > 0 else 0.05 * w
    return omega_p, gamma


def _debye_seed(omega: np.ndarray, eps: np.ndarray) -> tuple[float, float]:
    """``(delta_epsilon, tau)`` from the frequency where Im eps peaks."""
    im = np.imag(eps)
    j = int(np.argmax(im))
    w = float(omega[j]) if omega[j] > 0 else float(omega.min())
    return max(2.0 * float(im[j]), 1e-3), 1.0 / w


def _initial_x(
    composition: Sequence[str],
    omega: np.ndarray,
    eps: np.ndarray,
    fixed_eps_inf: float | None,
    eps_inf_guess: float,
) -> np.ndarray:
    lorentz_like = [k for k in composition if k in ("lorentz", "sellmeier")]
    seeds = _lorentz_seeds(omega, eps, len(lorentz_like))
    x: list[float] = []
    if fixed_eps_inf is None:
        x.append(eps_inf_guess)
    seed_i = 0
    for kind in composition:
        if kind == "lorentz":
            w0, gamma, de = seeds[seed_i]
            seed_i += 1
            x += [math.log(w0), gamma, math.log(de)]
        elif kind == "sellmeier":
            w0, _gamma, de = seeds[seed_i]
            seed_i += 1
            x += [math.log(w0), math.log(de)]
        elif kind == "drude":
            wp, gamma = _drude_seed(omega, eps)
            x += [math.log(wp), math.log(gamma)]
        elif kind == "debye":
            de, tau = _debye_seed(omega, eps)
            x += [math.log(de), math.log(tau)]
    return np.asarray(x, dtype=np.float64)


def _compositions(kinds: Sequence[str], num_poles: int, eps: np.ndarray) -> list[tuple[str, ...]]:
    """Every multiset of ``kinds`` of size ``num_poles``, most plausible first."""
    from itertools import combinations_with_replacement

    metallic = bool(np.min(np.real(eps)) < 0.0)
    lossless = bool(np.max(np.abs(np.imag(eps))) < 1e-9)
    combos = list(combinations_with_replacement(tuple(kinds), num_poles))

    def score(comp: tuple[str, ...]) -> tuple[float, ...]:
        has_drude = "drude" in comp
        has_lossy = any(k in ("lorentz", "drude", "debye") for k in comp)
        # lower is better
        return (
            0.0 if has_drude == metallic else 1.0,
            0.0 if (has_lossy != lossless) else 1.0,
            float(sum(_N_PARAMS[k] for k in comp)),
        )

    combos.sort(key=score)
    return combos


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_stability(model: DispersionModel, dt: float) -> tuple[bool, str]:
    """Check a fitted model against the ADE time-step bound.

    Uses the same ``omega_0 * dt < 2`` test the engine enforces in
    :func:`~fdtdx.dispersion.compute_pole_coefficients_tensor` — a pole whose
    resonance is not resolved by the time step makes the recurrence roots leave
    the unit circle. First-order (Debye) poles are unconditionally stable and
    always pass.

    Args:
        model: The dispersion model to check.
        dt: Simulation time step (seconds).

    Returns:
        tuple: ``(ok, message)``; ``message`` names the offending poles when
        ``ok`` is ``False``, and reports the tightest margin otherwise.
    """
    from fdtdx.dispersion import compute_pole_coefficients_tensor

    try:
        compute_pole_coefficients_tensor(model.poles, dt)
    except ValueError as exc:
        return False, f"unstable at dt = {dt:.4g} s: {exc}"
    worst = 0.0
    for p in model.poles:
        try:
            worst = max(worst, max(p.omega_0_axes) * dt)
        except NotImplementedError:
            continue  # first-order pole: unconditionally stable
    if worst == 0.0:
        return True, f"stable at dt = {dt:.4g} s (no second-order poles)"
    note = "" if worst < 0.2 else " — physically omega_0 * dt << 1, consider a smaller time step"
    return True, f"stable at dt = {dt:.4g} s: max omega_0 * dt = {worst:.3g} < 2{note}"


def fit_dispersion(
    wavelengths_m: Iterable[float],
    n: Iterable[float],
    k: Iterable[float],
    num_poles: int,
    kinds: Sequence[str] = ("lorentz", "drude", "debye"),
    eps_inf: float | None = None,
    weights: Iterable[float] | None = None,
    max_starts: int = 8,
    seed: int = 0,
    dt: float | None = None,
) -> FitResult:
    """Fit a pole model to measured refractive index data.

    Args:
        wavelengths_m: Vacuum wavelengths (metres), any order.
        n: Real refractive index at each wavelength.
        k: Extinction coefficient at each wavelength (zeros for a transparent
            record).
        num_poles: Number of poles to place.
        kinds: Pole kinds the fit may use, from :data:`POLE_KINDS`. Every
            multiset of this size is tried, most plausible first. Use
            ``("sellmeier",)`` inside a transparency window, where the data
            cannot constrain a damping rate.
        eps_inf: High-frequency permittivity. ``None`` (default) fits it,
            bounded below by 1 so the model stays passive.
        weights: Optional per-point weights (same length as the data).
        max_starts: Maximum number of optimiser starts. Compositions are tried
            first, then seeded random jitter around each.
        seed: Seed for the jitter, so a fit is reproducible.
        dt: Optional simulation time step (seconds). When given, the report
            carries the :func:`check_stability` advisory.

    Returns:
        FitResult: The best fit found.

    Raises:
        ValueError: If the inputs are inconsistent, ``num_poles`` is not
            positive, or ``kinds`` names an unknown pole kind.
    """
    lam = np.asarray(list(wavelengths_m), dtype=np.float64)
    n_arr = np.asarray(list(n), dtype=np.float64)
    k_arr = np.asarray(list(k), dtype=np.float64)
    if lam.shape != n_arr.shape or lam.shape != k_arr.shape:
        raise ValueError(
            f"wavelengths_m, n and k must have the same length, got {lam.shape}, {n_arr.shape}, {k_arr.shape}."
        )
    if lam.size < 2:
        raise ValueError("At least two data points are required to fit a dispersion model.")
    if np.any(lam <= 0.0):
        raise ValueError("Wavelengths must be positive (metres).")
    if num_poles < 1:
        raise ValueError(f"num_poles must be >= 1, got {num_poles}.")
    unknown = [kind for kind in kinds if kind not in POLE_KINDS]
    if unknown:
        raise ValueError(f"Unknown pole kind(s) {unknown}; expected a subset of {POLE_KINDS}.")
    if not kinds:
        raise ValueError("kinds must name at least one pole kind.")

    order = np.argsort(lam)[::-1]  # ascending angular frequency
    lam = lam[order]
    n_arr = n_arr[order]
    k_arr = k_arr[order]
    omega = 2.0 * math.pi * c_light / lam
    eps_data = (n_arr + 1j * k_arr) ** 2

    if weights is None:
        w = np.ones_like(lam)
    else:
        w = np.asarray(list(weights), dtype=np.float64)[order]
        if w.shape != lam.shape:
            raise ValueError("weights must have the same length as the data.")
        if np.any(w < 0.0):
            raise ValueError("weights must be non-negative.")
    w_norm = np.sqrt(w / w.sum())

    eps_inf_guess = float(np.clip(np.min(np.real(eps_data)), 1.0, 30.0))
    eps_inf_hi = float(max(2.0 * np.max(np.real(eps_data)), 10.0))

    def residual(x: np.ndarray, composition: Sequence[str]) -> np.ndarray:
        diff = _eps_from_x(x, composition, eps_inf, omega) - eps_data
        return np.concatenate([w_norm * diff.real, w_norm * diff.imag])

    compositions = _compositions(kinds, num_poles, eps_data)
    rng = np.random.default_rng(seed)

    best: tuple[float, np.ndarray, tuple[str, ...]] | None = None
    for start in range(max(1, max_starts)):
        composition = compositions[start % len(compositions)]
        jitter_round = start // len(compositions)
        lo_p, hi_p = [], []
        if eps_inf is None:
            lo_p.append(1.0)
            hi_p.append(eps_inf_hi)
        for kind in composition:
            lo_k, hi_k = _bounds_for_kind(kind, omega)
            lo_p += lo_k
            hi_p += hi_k
        lo = np.asarray(lo_p)
        hi = np.asarray(hi_p)

        x0 = _initial_x(composition, omega, eps_data, eps_inf, eps_inf_guess)
        if jitter_round > 0:
            # log-space variables take a multiplicative kick, linear ones a
            # relative one; both stay inside the bounds after the clip.
            x0 = x0 + rng.normal(scale=0.5 * jitter_round, size=x0.shape) * np.maximum(np.abs(x0), 1e-3) * 0.25
        x0 = np.clip(x0, lo + 1e-12, hi - 1e-12)

        try:
            sol = least_squares(
                residual,
                x0,
                bounds=(lo, hi),
                args=(composition,),
                method="trf",
                max_nfev=4000,
                xtol=1e-14,
                ftol=1e-14,
                gtol=1e-14,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        cost = float(np.sum(sol.fun**2))
        if best is None or cost < best[0]:
            best = (cost, sol.x, composition)

    if best is None:
        raise ValueError("Every optimiser start failed; check the input data and the requested pole kinds.")

    _cost, x_best, composition = best
    model, eps_inf_fit = _model_from_x(x_best, composition, eps_inf)
    eps_fit = _eps_of_model(model, eps_inf_fit, omega)
    diff = np.abs(eps_fit - eps_data)
    rms = float(np.sqrt(np.sum(w * diff**2) / w.sum()))
    max_abs_error = float(np.max(diff))

    # Passivity on a dense grid rather than only at the data points.
    dense = np.linspace(float(omega.min()), float(omega.max()), max(512, 4 * omega.size))
    im_dense = np.imag(_eps_of_model(model, eps_inf_fit, dense))
    tol = 1e-9 * max(1.0, float(np.max(np.abs(eps_data))))
    passive = bool(np.min(im_dense) >= -tol)

    lines = [
        f"fit_dispersion: {num_poles} pole(s), composition {'+'.join(composition)}",
        f"  data: {lam.size} points, {lam.min() * 1e6:.4g}-{lam.max() * 1e6:.4g} um",
        f"  eps_inf = {eps_inf_fit:.6g}" + ("" if eps_inf is None else " (fixed)"),
    ]
    for i, p in enumerate(model.poles):
        lines.append(f"  pole {i}: {_describe_pole(p)}")
    lines.append(f"  rms |d eps| = {rms:.4g}, max |d eps| = {max_abs_error:.4g}")
    lines.append(f"  passive (Im eps >= 0 over the fit range): {passive}; min Im eps = {np.min(im_dense):.4g}")
    if np.min(k_arr) < 0.0:
        lines.append("  note: the input data itself has k < 0 somewhere (non-passive measurement)")
    if dt is not None:
        _ok, msg = check_stability(model, dt)
        lines.append(f"  stability: {msg}")

    return FitResult(
        model=model,
        eps_inf=eps_inf_fit,
        rms=rms,
        max_abs_error=max_abs_error,
        passive=passive,
        report="\n".join(lines),
    )


def _describe_pole(p: Pole) -> str:
    if isinstance(p, SellmeierPole):
        lam0 = 2.0 * math.pi * c_light / p.omega_0
        return f"Sellmeier B={float(p.B):.6g}, C={float(p.C):.6g} m^2 (lambda_0={lam0 * 1e6:.5g} um)"
    if isinstance(p, LorentzPole):
        return (
            f"Lorentz omega_0={p.omega_0:.6g} rad/s, gamma={p.gamma:.6g} rad/s, delta_eps={float(p.delta_epsilon):.6g}"
        )
    if isinstance(p, DrudePole):
        return f"Drude omega_p={float(p.plasma_frequency):.6g} rad/s, gamma={p.gamma:.6g} rad/s"
    if isinstance(p, DebyePole):
        return f"Debye delta_eps={float(p.delta_epsilon):.6g}, tau={p.tau:.6g} s"
    return type(p).__name__


# ---------------------------------------------------------------------------
# refractiveindex.info YAML reader
# ---------------------------------------------------------------------------


def _formula_n(kind: int, coeffs: np.ndarray, lam_um: np.ndarray) -> np.ndarray:
    """Evaluate a refractiveindex.info dispersion formula on ``lam_um`` (um)."""
    c0 = float(coeffs[0])
    rest = coeffs[1:]
    if kind == 1:
        # n^2 - 1 = c0 + sum_i c_{2i-1} lam^2 / (lam^2 - c_{2i}^2)   (c_{2i} in um)
        n_sq = 1.0 + c0
        for i in range(0, len(rest) - 1, 2):
            n_sq = n_sq + rest[i] * lam_um**2 / (lam_um**2 - rest[i + 1] ** 2)
        return np.sqrt(n_sq)
    if kind == 2:
        # n^2 - 1 = c0 + sum_i c_{2i-1} lam^2 / (lam^2 - c_{2i})     (c_{2i} in um^2)
        n_sq = 1.0 + c0
        for i in range(0, len(rest) - 1, 2):
            n_sq = n_sq + rest[i] * lam_um**2 / (lam_um**2 - rest[i + 1])
        return np.sqrt(n_sq)
    if kind == 3:
        # n^2 = c0 + sum_i c_{2i-1} lam^{c_{2i}}
        n_sq = np.full_like(lam_um, c0)
        for i in range(0, len(rest) - 1, 2):
            n_sq = n_sq + rest[i] * lam_um ** rest[i + 1]
        return np.sqrt(n_sq)
    raise ValueError(
        f"refractiveindex.info dispersion formula {kind} is not supported; "
        "this reader handles formulas 1 (Sellmeier), 2 (Sellmeier-2) and 3 (polynomial), "
        "plus the tabulated record types."
    )


def _parse_table(text: str, columns: int) -> np.ndarray:
    rows = []
    for line in text.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != columns:
            raise ValueError(f"Expected {columns} columns in a tabulated record, got {len(parts)}: {line!r}")
        rows.append([float(v) for v in parts])
    if not rows:
        raise ValueError("Tabulated record is empty.")
    return np.asarray(rows, dtype=np.float64)


def read_refractiveindex_yaml(
    path: str | Path,
    wavelengths_m: Iterable[float] | None = None,
    num_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one refractiveindex.info YAML record.

    Handles the record types this package needs: ``tabulated nk``,
    ``tabulated n``, ``tabulated k`` and the dispersion formulas 1 (Sellmeier,
    ``C`` given as the resonance wavelength in um), 2 (Sellmeier-2, ``C`` in
    um^2) and 3 (polynomial). A record may mix one ``n`` source with a
    ``tabulated k``; the two are put on a common grid.

    Args:
        path: Path to the ``.yml`` record inside a database clone.
        wavelengths_m: Wavelengths (metres) to evaluate at. ``None`` (default)
            uses the tabulated grid, or a log-spaced grid over the record's
            ``wavelength_range`` for a formula-only record.
        num_points: Grid size when one has to be generated.

    Returns:
        tuple: ``(wavelengths_m, n, k)``, ascending in wavelength. ``k`` is
        zero where the record gives no absorption data.

    Raises:
        ValueError: If the record carries no usable ``DATA`` entry, an
            unsupported formula type, or if requested wavelengths fall outside
            the record's validity range.
    """
    import yaml  # imported lazily so the core package does not need it at import time

    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    entries = doc.get("DATA") if isinstance(doc, dict) else None
    if not entries:
        raise ValueError(f"{path} has no DATA section.")

    n_table: np.ndarray | None = None
    k_table: np.ndarray | None = None
    formula: tuple[int, np.ndarray] | None = None
    formula_range: tuple[float, float] | None = None

    for entry in entries:
        kind = str(entry.get("type", "")).strip()
        if kind == "tabulated nk":
            table = _parse_table(entry["data"], 3)
            n_table = table[:, :2]
            k_table = table[:, (0, 2)]
        elif kind == "tabulated n":
            n_table = _parse_table(entry["data"], 2)
        elif kind == "tabulated k":
            k_table = _parse_table(entry["data"], 2)
        elif kind.startswith("formula"):
            formula = (int(kind.split()[1]), np.asarray([float(v) for v in str(entry["coefficients"]).split()]))
            rng_txt = str(entry.get("wavelength_range", "")).split()
            if len(rng_txt) == 2:
                formula_range = (float(rng_txt[0]), float(rng_txt[1]))
        # other record types (e.g. "tabulated n2") are ignored

    if n_table is None and formula is None:
        raise ValueError(f"{path} has no usable n data (need a tabulated n/nk record or a dispersion formula).")

    # --- pick the output grid (micrometres, ascending) ----------------------
    if wavelengths_m is not None:
        lam_um = np.sort(np.asarray(list(wavelengths_m), dtype=np.float64) * 1e6)
    elif n_table is not None:
        lam_um = np.sort(n_table[:, 0])
    else:
        assert formula_range is not None, "a formula record without wavelength_range needs explicit wavelengths"
        lam_um = np.logspace(math.log10(formula_range[0]), math.log10(formula_range[1]), num_points)

    # --- n ------------------------------------------------------------------
    if n_table is not None:
        src = n_table[np.argsort(n_table[:, 0])]
        if wavelengths_m is not None and (lam_um.min() < src[0, 0] or lam_um.max() > src[-1, 0]):
            raise ValueError(
                f"Requested wavelengths {lam_um.min():.4g}-{lam_um.max():.4g} um fall outside the tabulated "
                f"range {src[0, 0]:.4g}-{src[-1, 0]:.4g} um of {path.name}."
            )
        n_vals = np.interp(lam_um, src[:, 0], src[:, 1])
    else:
        assert formula is not None
        if formula_range is not None and (lam_um.min() < formula_range[0] - 1e-12 or lam_um.max() > formula_range[1]):
            raise ValueError(
                f"Requested wavelengths {lam_um.min():.4g}-{lam_um.max():.4g} um fall outside the validity "
                f"range {formula_range[0]:.4g}-{formula_range[1]:.4g} um of {path.name}."
            )
        n_vals = _formula_n(formula[0], formula[1], lam_um)

    # --- k ------------------------------------------------------------------
    if k_table is None:
        k_vals = np.zeros_like(lam_um)
    else:
        src = k_table[np.argsort(k_table[:, 0])]
        k_vals = np.interp(lam_um, src[:, 0], src[:, 1], left=0.0, right=0.0)

    return lam_um * 1e-6, n_vals, k_vals
