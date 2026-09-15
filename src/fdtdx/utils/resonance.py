"""Resonance analysis of a recorded time signal — harmonic inversion, Lorentzian fits, ring-down Q.

Three tools, all pure ``numpy``/``scipy`` (no JAX, no simulation objects), so they run on anything
array-like: a :class:`~fdtdx.FieldDetector` trace, a column of a spectrum, a hand-built array.

* :func:`find_resonances` — **filter diagonalisation** (the filter-diagonalisation method, FDM) on a
  time series. Given a short ring-down it returns the complex frequencies (frequency + decay rate),
  Q, amplitude and phase of the modes inside a chosen band, at far better resolution than the
  ``1/T`` Fourier limit of the same record.
* :func:`fit_lorentzian` — least-squares Lorentzian through a transmission dip or a resonance peak
  (the microring through-port use), returning the line centre, full width at half maximum (FWHM)
  and Q.
* :func:`q_from_ringdown` — a one-line envelope-decay estimate of Q, as an independent cross-check
  on the two fits above.

Method and provenance
---------------------
The harmonic-inversion problem — recover the frequencies, decay rates, amplitudes and phases of a
finite sum of decaying sinusoids from a finite record — is solved here by filter diagonalisation,
after M. R. Wall and D. Neuhauser, *J. Chem. Phys.* **102**, 8011 (1995) and V. A. Mandelshtam and
H. S. Taylor, *J. Chem. Phys.* **107**, 6756 (1997). The signal is read as the autocorrelation of a
fictitious dynamical system, so that the frequencies become eigenvalues of that system's
time-evolution operator; restricting attention to a band ``[f_min, f_max]`` lets the matrix elements
of that operator be written as plain z-transforms of the record, and one small generalised
eigenproblem (of size ``n_basis``, not of the record length) returns every mode in the band.

The published method is the reference; MEEP's ``harminv`` is the reference *implementation* whose
conventions (band search, spectral density, error estimate, the mode ordering and the reported
quantities) this function follows, and whose output it is meant to be comparable with. No harminv
code is used — harminv is GPL and this repository references Meep-family algorithms without copying
code (see ``docs/licensing.md``). The reduction of the double Krylov sum to single z-transforms, the
regularised generalised eigensolve and the least-squares amplitude step below are written from the
papers' description.

Conventions
-----------
A mode is the complex exponential

.. code-block:: text

    s(t) = amplitude * exp[-i (2*pi*frequency*t - phase) - 2*pi*decay_rate*t]

so that, writing a complex frequency ``f_c = frequency - i*decay_rate``, the mode is simply
``amplitude * exp(i*phase) * exp(-2j*pi*f_c*t)``.

* **Frequency sign** — the ``exp(-i*omega*t)`` (physics / harminv) convention. ``frequency`` is an
  ordinary frequency in Hz, not an angular frequency. For a *complex* record, ``+f`` and ``-f`` are
  physically distinct and only the ones inside the search band are returned; a *real* record carries
  every mode as a ``±f`` pair, and searching a band of positive frequencies returns the ``+f``
  member, with the amplitude of the ``cos`` it belongs to split evenly between the pair (a real
  ``A*cos(2*pi*f*t)`` is reported with ``amplitude = A/2``, as harminv reports it).
* **Decay sign** — ``decay_rate`` is in Hz, the same units as ``frequency``, and is **positive for a
  decaying mode** (the field envelope falls as ``exp(-2*pi*decay_rate*t)``, the energy as
  ``exp(-4*pi*decay_rate*t)``). A negative value means the fit found a growing mode, which in a
  passive simulation is a sign of a spurious mode or of a record that is still being driven.
* **Q** — ``Q = frequency / (2*|decay_rate|)``. This is the usual
  ``Q = omega_0 * energy / (power lost)``, and it is the same number as the ``f0/FWHM`` returned by
  :func:`fit_lorentzian`, because a mode with envelope ``exp(-2*pi*decay_rate*t)`` has a power
  spectrum whose FWHM is ``2*decay_rate``. It equals harminv's ``pi*|f|/decay`` for harminv's decay
  constant ``2*pi*decay_rate``.
* **Basis size** — ``n_basis`` defaults to harminv's rule, a spectral density of 1.1 over the band:
  ``round(1.1 * (f_max - f_min) * dt * n_samples)``, clamped to ``[2, 300]``. That is an upper bound
  on how many modes can be found, not the resolution of the ones that are; densities far above 1
  make the matrices large and singular.

Limits
------
Filter diagonalisation assumes the record *is* a small number of decaying sinusoids plus a little
noise inside the band. It is not a general-purpose spectrum estimator: a broadband or
still-driven record, a band containing a continuum, or a band containing zero produce meaningless
modes. Feed it the ring-down — the part of the trace after the source has switched off — and keep
the band narrow enough that only a handful of modes live inside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
from scipy.optimize import curve_fit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import ArrayLike

# Spectral density of the filter basis (harminv's default): the band is initially searched with
# 1.1 basis functions per Fourier resolution element.
_DEFAULT_DENSITY = 1.1
# Hard cap on the basis size; above this the U matrices get large and badly conditioned.
_MAX_BASIS = 300
# Relative singular-value cut when regularising the (near-singular) U0 pencil.
_U0_RCOND = 1e-8
# Most modes ever carried into the least-squares amplitude fit (lowest error first).
_MAX_LSQ_MODES = 128
# Columns of the z-transform evaluated at once, to bound peak memory on long records.
_CHUNK = 4096


@dataclass
class Resonance:
    """One mode extracted from a time record by :func:`find_resonances`.

    The mode is ``amplitude * exp[-i (2*pi*frequency*t - phase) - 2*pi*decay_rate*t]``; see the
    module docstring for the sign and Q conventions.
    """

    #: Oscillation frequency in Hz (ordinary, not angular; signed, in the ``exp(-i*omega*t)`` sense).
    frequency: float
    #: Envelope decay rate in Hz: the field falls as ``exp(-2*pi*decay_rate*t)``. Positive = decaying.
    decay_rate: float
    #: Quality factor ``frequency / (2*|decay_rate|)``; ``inf`` for a non-decaying mode.
    q: float
    #: Amplitude of the mode (units of the input signal).
    amplitude: float
    #: Phase in radians, as defined above.
    phase: float
    #: harminv-style figure of merit for the complex frequency: smaller is better, ~1e-15 for a
    #: noiseless mode. Not an error bar.
    error: float


@dataclass
class LorentzianFit:
    """Result of :func:`fit_lorentzian`.

    ``f0`` and ``fwhm`` carry the units of the x axis that was passed in, so a wavelength axis
    returns a centre wavelength and a width in the same length units.
    """

    #: Line centre.
    f0: float
    #: Full width at half maximum.
    fwhm: float
    #: ``f0 / fwhm`` — the loaded Q of the line.
    q: float
    #: Signed extremum relative to the baseline: positive for a peak (height), negative for a dip
    #: (depth).
    depth_or_height: float
    #: The off-resonance level the line sits on.
    baseline: float
    #: Root-mean-square residual of the fit, in the units of the spectrum.
    rmse: float


def _as_1d(signal: "ArrayLike", name: str) -> np.ndarray:
    arr = np.asarray(signal)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional (got shape {np.asarray(signal).shape})")
    if arr.size < 8:
        raise ValueError(f"{name} is too short ({arr.size} samples)")
    return arr


def _fdm_transforms(c: np.ndarray, theta: np.ndarray, n_half: int, n_p: int) -> tuple[np.ndarray, ...]:
    """Evaluate the z-transforms that build the FDM operator matrices.

    For every basis point ``x_j = exp(i*theta_j)`` and every shift ``p`` this returns

    * ``G[p, j] = sum_{m=0}^{M} c[m+p] x_j^m``            (head transform),
    * ``H[p, j] = sum_{k=1}^{M} c[k+M+p] x_j^k``          (tail transform),
    * ``D[p, j] = sum_{m=0}^{2M} w_m c[m+p] x_j^m``       (the diagonal of U, ``w_m = min(m+1, 2M+1-m)``),

    with ``M = n_half``. One pass over the record serves every ``p``; the loop chunks the exponential
    so peak memory stays at ``n_basis x _CHUNK`` regardless of the record length.

    Args:
        c: the (complex) record.
        theta: phases of the basis points, ``theta_j = 2*pi*f_j*dt``.
        n_half: ``M``, the half-length of the Krylov sums.
        n_p: number of shifts ``p = 0 .. n_p-1`` to evaluate.

    Returns:
        The three arrays ``(G, H, D)``, each of shape ``(n_p, len(theta))``.
    """
    n_basis = theta.size
    total = 2 * n_half + 1  # m = 0 .. 2M
    m_all = np.arange(total)
    weights = np.minimum(m_all + 1, total - m_all).astype(float)

    g = np.zeros((n_p, n_basis), dtype=complex)
    h = np.zeros((n_p, n_basis), dtype=complex)
    d = np.zeros((n_p, n_basis), dtype=complex)

    for start in range(0, total, _CHUNK):
        stop = min(start + _CHUNK, total)
        expo = np.exp(1j * np.outer(theta, m_all[start:stop]))  # (n_basis, chunk)
        for p in range(n_p):
            d[p] += expo @ (weights[start:stop] * c[start + p : stop + p])
            head_stop = min(stop, n_half + 1)  # m in [0, M]
            if start < head_stop:
                g[p] += expo[:, : head_stop - start] @ c[start + p : head_stop + p]
            tail_start = max(start, 1)  # m in [1, M], coefficient c[m+M+p]
            if tail_start < head_stop:
                h[p] += (
                    expo[:, tail_start - start : head_stop - start]
                    @ c[tail_start + n_half + p : head_stop + n_half + p]
                )
    return g, h, d


def _fdm_matrices(c: np.ndarray, theta: np.ndarray, n_half: int, n_p: int = 3) -> list[np.ndarray]:
    """Build the complex-symmetric FDM matrices ``U^(p)``, ``p = 0 .. n_p-1``.

    ``U^(p)[j, j'] = sum_{n,n'=0}^{M} x_j^n x_j'^n' c[n+n'+p]`` is evaluated in closed form from the
    single z-transforms of :func:`_fdm_transforms`; the ``j == j'`` limit is the weighted sum ``D``.

    Args:
        c: the (complex) record.
        theta: phases of the basis points.
        n_half: ``M``, the half-length of the Krylov sums.
        n_p: number of matrices to build.

    Returns:
        A list of ``(n_basis, n_basis)`` complex arrays.
    """
    g, h, d = _fdm_transforms(c, theta, n_half, n_p)
    x = np.exp(1j * theta)
    x_pow = np.exp(1j * (n_half + 1) * theta)
    denom = x[None, :] - x[:, None]
    same = np.abs(denom) == 0.0
    denom = np.where(same, 1.0, denom)

    mats: list[np.ndarray] = []
    for p in range(n_p):
        xg = x * g[p]
        num = xg[None, :] - xg[:, None] + x_pow[None, :] * h[p][:, None] - x_pow[:, None] * h[p][None, :]
        mat = num / denom
        mat[same] = 0.0
        mat[np.diag_indices_from(mat)] = d[p]
        mats.append(0.5 * (mat + mat.T))  # symmetrise away round-off asymmetry
    return mats


def _solve_pencil(u0: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``U1 b = u U0 b`` on the well-conditioned subspace of ``U0``.

    ``U0`` is deliberately near-singular (its numerical rank is roughly the number of real modes in
    the band), so the pencil is projected onto the right singular vectors of ``U0`` whose singular
    values exceed ``_U0_RCOND`` times the largest before ``scipy.linalg.eig`` is called on it.

    Args:
        u0: the ``U^(0)`` matrix.
        u1: the ``U^(1)`` matrix.
        u2: the ``U^(2)`` matrix, used only for the error estimate.

    Returns:
        ``(eigenvalues, errors)``: the complex eigenvalues ``u_k = exp(-2j*pi*f_c*dt)`` and the
        harminv-style relative error of each, ``|b^T U2 b - u^2| / |u|^2`` for ``b`` normalised by
        ``b^T U0 b = 1``.
    """
    _, singular, vh = np.linalg.svd(u0)
    if singular.size == 0 or singular[0] == 0.0:
        return np.zeros(0, dtype=complex), np.zeros(0)
    rank = int(np.sum(singular > _U0_RCOND * singular[0]))
    rank = max(rank, 1)
    proj = vh[:rank].conj().T  # (n_basis, rank)

    a0 = proj.T @ u0 @ proj
    a1 = proj.T @ u1 @ proj
    a2 = proj.T @ u2 @ proj

    vals, vecs = scipy.linalg.eig(a1, a0)

    norm = np.einsum("ik,ij,jk->k", vecs, a0, vecs)
    quad2 = np.einsum("ik,ij,jk->k", vecs, a2, vecs)
    with np.errstate(divide="ignore", invalid="ignore"):
        u2_expect = quad2 / norm
        errors = np.abs(u2_expect - vals**2) / np.abs(vals) ** 2
    errors = np.where(np.isfinite(errors), errors, np.inf)
    errors = np.where(np.abs(norm) > 0.0, errors, np.inf)
    return vals, errors


def _fit_amplitudes(c: np.ndarray, u: np.ndarray, real_signal: bool) -> np.ndarray:
    """Complex amplitudes of the modes ``u`` by least squares over the whole record.

    Solves ``c[n] ~= sum_k a_k u_k^n``. For a real record the design matrix is closed under
    conjugation (the ``-f`` partner of every mode is added), which both makes the model complete and
    forces the conjugate-pair symmetry of the solution.

    Args:
        c: the record.
        u: the per-mode eigenvalues ``exp(-2j*pi*f_c*dt)``.
        real_signal: whether the record was real-valued.

    Returns:
        The complex amplitudes ``a_k``, one per entry of ``u``.
    """
    n_samples = c.size
    steps = np.arange(n_samples)
    design = np.exp(np.outer(steps, np.log(u)))
    if real_signal:
        design = np.hstack([design, design.conj()])
    coeffs, *_ = np.linalg.lstsq(design, c.astype(complex), rcond=None)
    return coeffs[: u.size]


def find_resonances(
    signal: "ArrayLike",
    dt: float,
    f_min: float,
    f_max: float,
    *,
    n_basis: int | None = None,
    error_threshold: float = 1e-3,
    amplitude_threshold: float = 1e-3,
) -> list[Resonance]:
    """Extract the resonances of a time record inside ``[f_min, f_max]`` by filter diagonalisation.

    The intended input is the *ring-down*: a field trace recorded at one point (a
    :class:`~fdtdx.FieldDetector` with ``reduce_volume=True``) after a short pulse has left the
    structure. The record is modelled as a sum of decaying sinusoids, so unlike a discrete Fourier
    transform the resolution is not tied to ``1 / (n_samples * dt)`` and a few tens of optical cycles
    are enough to separate closely-spaced modes and read their Q.

    The conventions (signs, Q, the default basis size) are stated in the module docstring, and
    follow harminv so the two can be compared directly.

    Args:
        signal: the recorded time series, real or complex, sampled uniformly. One dimension after
            squeezing.
        dt: sample spacing in seconds — ``config.time_step_duration`` when every step is recorded,
            times the detector's recording stride when it is not.
        f_min: low edge of the search band in Hz.
        f_max: high edge of the search band in Hz. Must exceed ``f_min`` and stay below the Nyquist
            frequency ``1 / (2*dt)``; a band containing zero gives meaningless modes.
        n_basis: number of filter basis functions, an upper bound on the number of modes that can be
            found. ``None`` (default) uses harminv's spectral-density-1.1 rule
            ``round(1.1 * (f_max - f_min) * dt * n_samples)``, clamped to ``[2, 300]``.
        error_threshold: drop modes whose :attr:`Resonance.error` exceeds this. The default 1e-3 is
            much stricter than harminv's 0.1 and is meant for clean simulation ring-downs; raise it
            towards 0.1 for noisy or short records.
        amplitude_threshold: drop modes whose amplitude is below this **fraction of the largest
            amplitude found**. Relative, so it is independent of the field units.

    Returns:
        The surviving modes as :class:`Resonance` records, sorted by frequency (ascending). The list
        is empty when nothing in the band passes the thresholds.

    Raises:
        ValueError: if the record is too short, or the band is empty / beyond Nyquist.
    """
    c = _as_1d(signal, "signal")
    real_signal = not np.iscomplexobj(c)
    c = c.astype(complex)
    n_samples = c.size

    if dt <= 0:
        raise ValueError(f"dt must be positive (got {dt})")
    if not f_max > f_min:
        raise ValueError(f"f_max must exceed f_min (got f_min={f_min}, f_max={f_max})")
    nyquist = 0.5 / dt
    if max(abs(f_min), abs(f_max)) >= nyquist:
        raise ValueError(f"the band [{f_min}, {f_max}] reaches the Nyquist frequency {nyquist}")

    # M: the Krylov half-length. c[m + p] is needed for m <= 2M and p <= 2, so 2M + 2 <= n - 1.
    n_half = (n_samples - 3) // 2
    if n_half < 2:
        raise ValueError(f"signal is too short for filter diagonalisation ({n_samples} samples)")

    if n_basis is None:
        n_basis = round(_DEFAULT_DENSITY * (f_max - f_min) * dt * n_samples)
    n_basis = int(max(2, min(_MAX_BASIS, n_half, n_basis)))

    # Basis points at the midpoints of n_basis equal sub-bands, so no point sits on a band edge.
    edges = np.arange(n_basis) + 0.5
    f_basis = f_min + edges * (f_max - f_min) / n_basis
    theta = 2.0 * np.pi * f_basis * dt

    u0, u1, u2 = _fdm_matrices(c, theta, n_half, n_p=3)
    vals, errors = _solve_pencil(u0, u1, u2)

    good = np.isfinite(vals) & (np.abs(vals) > 0.0)
    vals, errors = vals[good], errors[good]
    if vals.size == 0:
        return []

    # exp(-2j*pi*(f - i*gamma)*dt) = vals  ->  f = -arg/(2*pi*dt), gamma = -log|u|/(2*pi*dt).
    two_pi_dt = 2.0 * np.pi * dt
    freqs = -np.angle(vals) / two_pi_dt
    decays = -np.log(np.abs(vals)) / two_pi_dt

    keep = errors <= error_threshold
    # Modes that grow by more than e^50 over the record cannot be fitted and are never physical.
    keep &= (n_samples - 1) * np.log(np.abs(vals)) < 50.0
    vals, errors, freqs, decays = vals[keep], errors[keep], freqs[keep], decays[keep]
    if vals.size == 0:
        return []

    if vals.size > _MAX_LSQ_MODES:
        best = np.argsort(errors)[:_MAX_LSQ_MODES]
        vals, errors, freqs, decays = vals[best], errors[best], freqs[best], decays[best]

    amps = _fit_amplitudes(c, vals, real_signal)
    magnitude = np.abs(amps)
    largest = float(np.max(magnitude)) if magnitude.size else 0.0
    if largest <= 0.0:
        return []
    keep = magnitude >= amplitude_threshold * largest

    out: list[Resonance] = []
    for freq, decay, err, amp in zip(freqs[keep], decays[keep], errors[keep], amps[keep]):
        q = float("inf") if decay == 0.0 else float(freq / (2.0 * abs(decay)))
        out.append(
            Resonance(
                frequency=float(freq),
                decay_rate=float(decay),
                q=q,
                amplitude=float(np.abs(amp)),
                phase=float(np.angle(amp)),
                error=float(err),
            )
        )
    out.sort(key=lambda r: r.frequency)
    return out


def _lorentzian(x: np.ndarray, f0: float, fwhm: float, depth: float, baseline: float) -> np.ndarray:
    """Lorentzian line of signed amplitude ``depth`` on a flat ``baseline``."""
    return baseline + depth / (1.0 + ((x - f0) / (0.5 * fwhm)) ** 2)


def fit_lorentzian(
    frequencies: "ArrayLike",
    spectrum: "ArrayLike",
    f_guess: float | None = None,
) -> LorentzianFit:
    """Fit a single Lorentzian dip or peak in a spectrum.

    This is the frequency-domain counterpart of :func:`find_resonances`: it reads the line centre,
    width and Q off a transmission spectrum (a through-port dip, as in the microring example) or an
    emission/resonance peak, instead of off a time record. The model is

    .. code-block:: text

        S(x) = baseline + depth_or_height / (1 + ((x - f0) / (fwhm/2))**2)

    with ``depth_or_height`` negative for a dip and positive for a peak. The x axis is whatever was
    passed in — frequency or wavelength — and ``f0``/``fwhm`` come back in those units, so
    ``q = f0/fwhm`` is the loaded Q either way.

    Args:
        frequencies: the x axis, monotonically ordered (Hz or metres).
        spectrum: the measured line; same length as *frequencies*.
        f_guess: optional starting guess for the centre, in the units of *frequencies*. Without it
            the deepest dip / highest peak relative to the median is used, which is robust when the
            band holds one line but picks the strongest when it holds several.

    Returns:
        The :class:`LorentzianFit`.

    Raises:
        ValueError: if the inputs have different lengths, or fewer than four points (the model has
            four free parameters).
    """
    x = _as_1d(np.asarray(frequencies, dtype=float), "frequencies")
    y = _as_1d(np.asarray(spectrum, dtype=float), "spectrum")
    if x.size != y.size:
        raise ValueError(f"frequencies and spectrum must have the same length ({x.size} vs {y.size})")
    if x.size < 4:
        raise ValueError("at least four points are needed to fit a Lorentzian")

    baseline0 = float(np.median(y))
    if f_guess is None:
        # Dip or peak, whichever departs further from the median.
        i0 = int(np.argmin(y)) if (baseline0 - y.min()) >= (y.max() - baseline0) else int(np.argmax(y))
    else:
        i0 = int(np.argmin(np.abs(x - float(f_guess))))
    f0_0 = float(x[i0])
    depth0 = float(y[i0] - baseline0)
    if depth0 == 0.0:
        depth0 = float(np.sign(np.mean(y) - baseline0) or 1.0) * (float(np.std(y)) or 1.0)

    # Width from the half-depth crossings on either side of the extremum.
    half = baseline0 + 0.5 * depth0
    beyond = (y - half) * np.sign(depth0) > 0.0
    lo = i0
    while lo > 0 and beyond[lo - 1]:
        lo -= 1
    hi = i0
    while hi < x.size - 1 and beyond[hi + 1]:
        hi += 1
    span = float(abs(x[-1] - x[0]))
    fwhm0 = float(abs(x[hi] - x[lo])) or span / max(x.size - 1, 1)

    # Fit in scaled coordinates. A photonic x axis is ~1e-6 (metres) or ~1e14 (Hz) and the width is
    # orders of magnitude smaller again; the optimiser's relative step tolerances declare victory at
    # the starting point unless both axes are brought to order 1 first.
    x_ref, x_scale = float(x.mean()), span or 1.0
    y_scale = max(abs(depth0), float(np.std(y)), 1e-300)
    xs = (x - x_ref) / x_scale
    ys = (y - baseline0) / y_scale

    p0 = [(f0_0 - x_ref) / x_scale, fwhm0 / x_scale, depth0 / y_scale, 0.0]
    bounds = (
        [xs.min() - 1.0, 1e-9, -np.inf, -np.inf],
        [xs.max() + 1.0, 10.0, np.inf, np.inf],
    )
    try:
        popt, _ = curve_fit(_lorentzian, xs, ys, p0=p0, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError):
        popt = np.array(p0, dtype=float)

    f0 = float(popt[0]) * x_scale + x_ref
    fwhm = float(popt[1]) * x_scale
    depth = float(popt[2]) * y_scale
    baseline = float(popt[3]) * y_scale + baseline0
    resid = y - _lorentzian(x, f0, fwhm, depth, baseline)
    return LorentzianFit(
        f0=f0,
        fwhm=abs(fwhm),
        q=abs(f0 / fwhm) if fwhm else float("inf"),
        depth_or_height=depth,
        baseline=baseline,
        rmse=float(np.sqrt(np.mean(resid**2))),
    )


def q_from_ringdown(signal: "ArrayLike", dt: float, f0: float) -> float:
    """Q of a ring-down from the slope of its log envelope — the cheap cross-check.

    The record is demodulated at ``f0`` (multiplied by ``exp(+2j*pi*f0*t)``) and smoothed over one
    period, which isolates the mode near ``f0`` from its neighbours and from the ``-f0`` partner of a
    real record. A straight line is then fitted to the log of the resulting envelope, from its peak
    down to 1e-3 of that peak, giving the envelope decay rate ``gamma`` and
    ``Q = f0 / (2*gamma)`` — the same Q definition as :func:`find_resonances`.

    This is a one-mode estimate and is much less accurate than filter diagonalisation: it is here to
    catch gross errors (a wrong band, a mode that was never excited), not to replace it. Expect
    agreement at the ten-percent level, and treat a large disagreement as a sign that the band holds
    more than one mode or that the record is still being driven.

    Args:
        signal: the recorded time series, real or complex.
        dt: sample spacing in seconds.
        f0: the frequency to demodulate at, in Hz — typically the frequency of the mode of interest.

    Returns:
        The estimated Q. ``inf`` when the envelope does not decay over the record.

    Raises:
        ValueError: if the record is too short, ``dt`` is not positive, or ``f0`` is not positive.
    """
    c = _as_1d(signal, "signal").astype(complex)
    if dt <= 0:
        raise ValueError(f"dt must be positive (got {dt})")
    if f0 <= 0:
        raise ValueError(f"f0 must be positive (got {f0})")

    steps = np.arange(c.size)
    demod = c * np.exp(2j * np.pi * f0 * steps * dt)
    per_period = max(2, round(1.0 / (f0 * dt)))
    if per_period >= c.size // 2:
        raise ValueError("signal covers fewer than four periods of f0")
    kernel = np.ones(per_period) / per_period
    envelope = np.abs(np.convolve(demod, kernel, mode="valid"))

    peak = int(np.argmax(envelope))
    top = float(envelope[peak])
    if top <= 0.0:
        return float("inf")
    tail = envelope[peak:]
    usable = np.nonzero(tail < 1e-3 * top)[0]
    stop = int(usable[0]) if usable.size else tail.size
    if stop < 4:
        return float("inf")

    t = steps[:stop] * dt
    slope = float(np.polyfit(t, np.log(tail[:stop]), 1)[0])
    gamma = -slope / (2.0 * np.pi)
    if gamma <= 0.0:
        return float("inf")
    return float(f0 / (2.0 * gamma))


__all__ = [
    "LorentzianFit",
    "Resonance",
    "find_resonances",
    "fit_lorentzian",
    "q_from_ringdown",
]
