"""Unit tests for the resonance finder, the Lorentzian fit and the ring-down Q estimate.

Every case here is synthetic: a signal built from exactly the model the fit assumes, so the
recovered parameters can be checked against the numbers that generated it.
"""

import numpy as np
import pytest

from fdtdx.utils.resonance import LorentzianFit, Resonance, find_resonances, fit_lorentzian, q_from_ringdown

# A record a few thousand samples long at optical time steps — the size a short FDTD ring-down has.
_DT = 1e-16
_N = 4000


def _time() -> np.ndarray:
    return np.arange(_N) * _DT


def _damped(frequency: float, q: float, amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """One complex mode in the module's convention: A e^{i phi} exp(-2j pi f t - 2 pi gamma t)."""
    gamma = frequency / (2.0 * q)
    t = _time()
    return amplitude * np.exp(1j * phase) * np.exp(-2j * np.pi * frequency * t - 2.0 * np.pi * gamma * t)


def test_two_complex_modes_recovered():
    """Two damped complex exponentials plus 1e-6 noise: frequencies to 1e-6, Q to 1 %."""
    f1, q1 = 2.00e14, 500.0
    f2, q2 = 2.05e14, 1200.0
    rng = np.random.default_rng(0)
    signal = _damped(f1, q1, 1.0) + _damped(f2, q2, 0.6, phase=0.7)
    signal = signal + 1e-6 * (rng.standard_normal(_N) + 1j * rng.standard_normal(_N))

    modes = find_resonances(signal, _DT, 1.8e14, 2.2e14)

    assert len(modes) == 2
    assert all(isinstance(m, Resonance) for m in modes)
    assert modes[0].frequency < modes[1].frequency  # sorted by frequency

    for mode, (f_true, q_true, a_true) in zip(modes, [(f1, q1, 1.0), (f2, q2, 0.6)]):
        assert abs(mode.frequency - f_true) / f_true < 1e-6
        assert abs(mode.q - q_true) / q_true < 1e-2
        assert mode.decay_rate > 0.0  # decaying, in the module's sign convention
        assert abs(mode.q - mode.frequency / (2.0 * mode.decay_rate)) < 1e-6 * q_true
        assert abs(mode.amplitude - a_true) / a_true < 1e-2
        assert mode.error < 1e-3

    assert abs(modes[1].phase - 0.7) < 1e-3


def test_single_real_mode():
    """A real damped cosine: the +f member of the pair, at half the cosine's amplitude."""
    f0, q0, amp, phase = 1.93e14, 800.0, 2.0, 0.4
    gamma = f0 / (2.0 * q0)
    t = _time()
    # In the module's convention this is the +f mode with amplitude amp/2 and phase -phase.
    signal = amp * np.cos(2.0 * np.pi * f0 * t + phase) * np.exp(-2.0 * np.pi * gamma * t)

    modes = find_resonances(signal, _DT, 1.7e14, 2.2e14)

    assert len(modes) == 1
    mode = modes[0]
    assert abs(mode.frequency - f0) / f0 < 1e-6
    assert abs(mode.q - q0) / q0 < 1e-2
    assert abs(mode.amplitude - amp / 2.0) / (amp / 2.0) < 1e-2
    assert abs(mode.phase + phase) < 1e-2


def test_amplitude_threshold_drops_weak_mode():
    """A third mode 1e-5 of the strongest survives only when the amplitude threshold allows it."""
    strong = _damped(2.00e14, 1000.0, 1.0) + _damped(2.05e14, 2400.0, 0.6)
    signal = strong + _damped(2.12e14, 1000.0, 1e-5)

    default = find_resonances(signal, _DT, 1.8e14, 2.2e14)
    permissive = find_resonances(signal, _DT, 1.8e14, 2.2e14, amplitude_threshold=1e-8)

    assert [round(m.frequency, -10) for m in default] == [2.00e14, 2.05e14]
    assert len(permissive) == 3
    weak = permissive[-1]
    assert abs(weak.frequency - 2.12e14) / 2.12e14 < 1e-5
    assert weak.amplitude < 1e-4


def test_error_threshold_keeps_nothing_when_impossibly_strict():
    """The error filter is live: an unreachable threshold returns an empty list, not junk."""
    signal = _damped(2.0e14, 800.0)
    assert find_resonances(signal, _DT, 1.8e14, 2.2e14, error_threshold=1e-30) == []


def test_ringdown_q_matches_the_finder():
    """The envelope-decay cross-check lands on the same Q as filter diagonalisation."""
    f0, q0 = 2.0e14, 600.0
    signal = np.real(_damped(f0, q0))
    modes = find_resonances(signal, _DT, 1.8e14, 2.2e14)

    assert len(modes) == 1
    q_env = q_from_ringdown(signal, _DT, modes[0].frequency)
    assert abs(q_env - q0) / q0 < 0.05


def test_default_basis_size_follows_the_harminv_rule():
    """n_basis defaults to density 1.1 over the band; an explicit value is honoured."""
    signal = _damped(2.0e14, 800.0)
    # 1.1 * 0.4e14 * 1e-16 * 4000 = 17.6 basis functions by default; both sizes find the mode.
    assert len(find_resonances(signal, _DT, 1.8e14, 2.2e14)) == 1
    assert len(find_resonances(signal, _DT, 1.8e14, 2.2e14, n_basis=40)) == 1


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"dt": -1e-16, "f_min": 1e14, "f_max": 2e14}, "dt must be positive"),
        ({"dt": _DT, "f_min": 2e14, "f_max": 1e14}, "f_max must exceed f_min"),
        ({"dt": _DT, "f_min": 1e14, "f_max": 1e16}, "Nyquist"),
    ],
)
def test_bad_arguments_raise(kwargs, message):
    signal = _damped(2.0e14, 800.0)
    with pytest.raises(ValueError, match=message):
        find_resonances(signal, **kwargs)


def _dip(x, f0, fwhm, depth, baseline):
    return baseline + depth / (1.0 + ((x - f0) / (0.5 * fwhm)) ** 2)


def test_fit_lorentzian_dip():
    """A transmission dip with 1e-3 noise: centre to 1e-4 relative, width to 1 %."""
    f0, fwhm, depth, baseline = 1.3004e-6, 1.7e-9, -0.8, 1.0
    x = np.linspace(1.28e-6, 1.32e-6, 401)
    y = _dip(x, f0, fwhm, depth, baseline) + np.random.default_rng(1).normal(0.0, 1e-3, x.size)

    fit = fit_lorentzian(x, y)

    assert isinstance(fit, LorentzianFit)
    assert abs(fit.f0 - f0) / f0 < 1e-4
    assert abs(fit.fwhm - fwhm) / fwhm < 1e-2
    assert abs(fit.q - f0 / fwhm) / (f0 / fwhm) < 1e-2
    assert fit.depth_or_height < 0.0  # a dip
    assert abs(fit.depth_or_height - depth) < 1e-2
    assert abs(fit.baseline - baseline) < 1e-2
    assert fit.rmse < 2e-3


def test_fit_lorentzian_peak():
    """The same model reads an emission peak: positive height above the baseline."""
    f0, fwhm, height, baseline = 1.3050e-6, 1.2e-9, 1.5, 0.1
    x = np.linspace(1.28e-6, 1.33e-6, 601)
    y = _dip(x, f0, fwhm, height, baseline) + np.random.default_rng(2).normal(0.0, 1e-3, x.size)

    fit = fit_lorentzian(x, y)

    assert abs(fit.f0 - f0) / f0 < 1e-4
    assert abs(fit.fwhm - fwhm) / fwhm < 1e-2
    assert fit.depth_or_height > 0.0
    assert abs(fit.depth_or_height - height) / height < 1e-2
    assert abs(fit.baseline - baseline) < 1e-2


def test_f_guess_selects_which_line_is_fitted():
    """With two lines in the window the fit centres on the one f_guess points at.

    A single-Lorentzian model cannot describe two overlapping lines, so only the centre is
    meaningful here — the width and baseline absorb the other line's tail.
    """
    x = np.linspace(1.27e-6, 1.33e-6, 1201)
    y = 0.1 + _dip(x, 1.2900e-6, 2.0e-9, 0.6, 0.0) + _dip(x, 1.3100e-6, 1.2e-9, 1.5, 0.0)

    assert abs(fit_lorentzian(x, y).f0 - 1.3100e-6) < 1e-10  # strongest by default
    assert abs(fit_lorentzian(x, y, f_guess=1.2900e-6).f0 - 1.2900e-6) < 1e-10


def test_fit_lorentzian_rejects_mismatched_inputs():
    x = np.linspace(0.0, 1.0, 20)
    with pytest.raises(ValueError, match="same length"):
        fit_lorentzian(x, np.zeros(19))
