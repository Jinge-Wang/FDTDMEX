"""Two-plane phase readout on synthetic plane-wave phasors with a known propagation constant."""

import numpy as np
import pytest

from fdtdx.coupling import plane_overlap_phase, reference_neff_between_planes, two_plane_delta_neff

WLS = np.array([1.50e-6, 1.55e-6, 1.60e-6])
Y1, Y2 = 5e-6, 15e-6


def _state(neff: float, y: float, offset: float = 0.0, amplitude: float = 1.0) -> dict:
    """Detector-like dictionary: E on a (nx, 1, nz) plane with phase +beta y, a fixed offset, and a mode profile."""
    x = np.linspace(-1, 1, 7)[:, None]
    z = np.linspace(-1, 1, 5)[None, :]
    profile = amplitude * np.exp(-(x**2 + z**2))
    phasor = np.zeros((1, WLS.size, 6, 7, 1, 5), dtype=np.complex128)
    for i, wl in enumerate(WLS):
        beta = 2 * np.pi * neff / wl
        phasor[0, i, 0, :, 0, :] = profile * np.exp(1j * (beta * y + offset))
        phasor[0, i, 2, :, 0, :] = 0.3 * profile * np.exp(1j * (beta * y + offset))
    return {"phasor": phasor}


def test_two_plane_readout_recovers_the_index_change_and_cancels_offsets():
    n_ref, dn = 2.40, 3.0e-3
    ref = (_state(n_ref, Y1, offset=0.7), _state(n_ref, Y2, offset=0.7))
    pert = (_state(n_ref + dn, Y1, offset=1.9, amplitude=0.8), _state(n_ref + dn, Y2, offset=1.9, amplitude=0.8))
    out = two_plane_delta_neff(ref, pert, WLS, Y2 - Y1)
    np.testing.assert_allclose(out["delta_neff"], dn, rtol=1e-10)
    assert out["power_plane_1"].shape == (WLS.size,)
    assert np.all(out["power_plane_1"] > 0)


def test_plane_overlap_phase_is_perturbed_minus_reference():
    a, b = _state(2.4, Y1, offset=0.2), _state(2.4, Y1, offset=0.5)
    phase, _ = plane_overlap_phase(b, a)
    np.testing.assert_allclose(phase, 0.3, atol=1e-12)


def test_reference_neff_between_planes_takes_the_branch_nearest_the_guess():
    n = 2.44
    neff = reference_neff_between_planes(_state(n, Y1), _state(n, Y2), WLS, Y2 - Y1, guess=2.5)
    np.testing.assert_allclose(neff, n, rtol=1e-10)


def test_shape_checks():
    with pytest.raises(ValueError):
        two_plane_delta_neff((_state(2.4, Y1), _state(2.4, Y2)), (_state(2.4, Y1), _state(2.4, Y2)), WLS[:2], Y2 - Y1)
    with pytest.raises(ValueError):
        two_plane_delta_neff((_state(2.4, Y1), _state(2.4, Y2)), (_state(2.4, Y1), _state(2.4, Y2)), WLS, 0.0)
