"""Unit tests for the dispersion fitter and the refractiveindex.info reader.

Tests that need the CC0 refractiveindex.info database skip when no local clone
is found; point ``FDTDX_REFRACTIVEINDEX_DB`` at one to run them.
"""

import math
import os
import textwrap
from pathlib import Path

import numpy as np
import pytest

from fdtdx.constants import c as c_light
from fdtdx.dispersion import DispersionModel, DrudePole, LorentzPole, SellmeierPole
from fdtdx.dispersion_fit import check_stability, fit_dispersion, read_refractiveindex_yaml

_DB_ENV = os.environ.get("FDTDX_REFRACTIVEINDEX_DB")
_DB_CANDIDATES = [
    Path(_DB_ENV) if _DB_ENV else None,
    Path.home() / "Projects/ReferenceSolvers/refractiveindex/database",
]
_DB = next((p for p in _DB_CANDIDATES if p is not None and (p / "data").is_dir()), None)
requires_db = pytest.mark.skipif(_DB is None, reason="no refractiveindex.info database clone found")


def _eps_of(result, omega: np.ndarray) -> np.ndarray:
    return np.array([result.eps_inf + result.model.susceptibility(float(w)) for w in omega], dtype=np.complex128)


class TestFitSynthetic:
    def test_recovers_two_lorentz_poles(self):
        """Noiseless data from a known two-pole medium must come back exactly."""
        true = DispersionModel(
            poles=(
                LorentzPole(resonance_frequency=2.0e15, damping=1.5e14, delta_epsilon=2.0),
                LorentzPole(resonance_frequency=4.5e15, damping=3.0e14, delta_epsilon=1.0),
            )
        )
        eps_inf = 2.0
        omega = np.linspace(0.8e15, 6.0e15, 120)
        lam = 2.0 * math.pi * c_light / omega
        eps = np.array([eps_inf + true.susceptibility(float(w)) for w in omega])
        index = np.sqrt(eps)

        result = fit_dispersion(lam, index.real, index.imag, num_poles=2, kinds=("lorentz",), max_starts=6)

        eps_fit = _eps_of(result, omega)
        assert np.max(np.abs(eps_fit - eps) / np.abs(eps)) < 1e-3
        assert result.passive
        assert result.eps_inf == pytest.approx(eps_inf, rel=1e-3)
        got = sorted(result.model.poles, key=lambda p: p.omega_0)
        for fitted, expected in zip(got, true.poles, strict=True):
            assert fitted.omega_0 == pytest.approx(expected.omega_0, rel=1e-3)
            assert fitted.gamma == pytest.approx(expected.gamma, rel=1e-3)
            assert float(fitted.delta_epsilon) == pytest.approx(float(expected.delta_epsilon), rel=1e-3)

    def test_recovers_a_debye_medium(self):
        true = DispersionModel(poles=(DrudePole(plasma_frequency=9.0e15, damping=1.0e14),))
        eps_inf = 3.0
        omega = np.linspace(1.0e15, 5.0e15, 60)
        lam = 2.0 * math.pi * c_light / omega
        eps = np.array([eps_inf + true.susceptibility(float(w)) for w in omega])
        index = np.sqrt(eps)
        result = fit_dispersion(lam, index.real, index.imag, num_poles=1, kinds=("drude",), max_starts=4)
        assert result.rms < 1e-6
        assert result.passive

    def test_fixed_eps_inf_is_honoured(self):
        true = DispersionModel(poles=(LorentzPole(resonance_frequency=3e15, damping=2e14, delta_epsilon=1.5),))
        omega = np.linspace(1e15, 5e15, 40)
        lam = 2.0 * math.pi * c_light / omega
        eps = np.array([2.5 + true.susceptibility(float(w)) for w in omega])
        index = np.sqrt(eps)
        result = fit_dispersion(lam, index.real, index.imag, num_poles=1, kinds=("lorentz",), eps_inf=2.5)
        assert result.eps_inf == 2.5
        assert "(fixed)" in result.report
        assert result.rms < 1e-6

    def test_weights_bias_the_fit(self):
        # One pole cannot fit two resonances; weighting the low-frequency half
        # must make the fit follow that half more closely than the other.
        true = DispersionModel(
            poles=(
                LorentzPole(resonance_frequency=1.5e15, damping=1e14, delta_epsilon=2.0),
                LorentzPole(resonance_frequency=5.0e15, damping=3e14, delta_epsilon=2.0),
            )
        )
        omega = np.linspace(1.0e15, 6.0e15, 80)
        lam = 2.0 * math.pi * c_light / omega
        eps = np.array([2.0 + true.susceptibility(float(w)) for w in omega])
        index = np.sqrt(eps)
        low = omega < 3.0e15
        weights = np.where(low, 100.0, 1.0)
        plain = fit_dispersion(lam, index.real, index.imag, num_poles=1, kinds=("lorentz",), max_starts=4)
        biased = fit_dispersion(
            lam, index.real, index.imag, num_poles=1, kinds=("lorentz",), weights=weights, max_starts=4
        )
        err_plain = np.abs(_eps_of(plain, omega) - eps)[low].mean()
        err_biased = np.abs(_eps_of(biased, omega) - eps)[low].mean()
        assert err_biased < err_plain

    def test_report_contains_the_essentials(self):
        omega = np.linspace(1e15, 4e15, 20)
        lam = 2.0 * math.pi * c_light / omega
        n = np.full_like(lam, 1.5)
        k = np.zeros_like(lam)
        result = fit_dispersion(lam, n, k, num_poles=1, kinds=("sellmeier",), dt=1e-17)
        for token in ("fit_dispersion", "eps_inf", "rms", "passive", "stability"):
            assert token in result.report

    def test_invalid_inputs_raise(self):
        lam = np.array([1e-6, 2e-6])
        with pytest.raises(ValueError, match="same length"):
            fit_dispersion(lam, [1.0], [0.0], num_poles=1)
        with pytest.raises(ValueError, match="num_poles"):
            fit_dispersion(lam, [1.0, 1.0], [0.0, 0.0], num_poles=0)
        with pytest.raises(ValueError, match="Unknown pole kind"):
            fit_dispersion(lam, [1.0, 1.0], [0.0, 0.0], num_poles=1, kinds=("plasmon",))
        with pytest.raises(ValueError, match="positive"):
            fit_dispersion([1e-6, -1e-6], [1.0, 1.0], [0.0, 0.0], num_poles=1)


class TestCheckStability:
    def test_resolved_pole_is_stable(self):
        model = DispersionModel(poles=(LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0),))
        ok, msg = check_stability(model, dt=1e-17)
        assert ok and "stable" in msg

    def test_unresolved_pole_is_rejected(self):
        model = DispersionModel(poles=(LorentzPole(resonance_frequency=1e16, damping=1e13, delta_epsilon=2.0),))
        ok, msg = check_stability(model, dt=1e-15)
        assert not ok and "omega_0 * dt" in msg

    def test_debye_pole_is_always_stable(self):
        from fdtdx.dispersion import DebyePole

        model = DispersionModel(poles=(DebyePole(delta_epsilon=2.0, relaxation_time=1e-18),))
        ok, msg = check_stability(model, dt=1e-15)
        assert ok and "no second-order poles" in msg


class TestReadRefractiveIndexYaml:
    def _write(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / "record.yml"
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        return path

    def test_tabulated_nk(self, tmp_path):
        path = self._write(
            tmp_path,
            """\
            DATA:
              - type: tabulated nk
                data: |
                    0.50 1.40 0.01
                    1.00 1.45 0.02
                    2.00 1.50 0.03
            """,
        )
        lam, n, k = read_refractiveindex_yaml(path)
        assert np.allclose(lam, [0.5e-6, 1e-6, 2e-6])
        assert np.allclose(n, [1.40, 1.45, 1.50])
        assert np.allclose(k, [0.01, 0.02, 0.03])

    def test_tabulated_n_gives_zero_k(self, tmp_path):
        path = self._write(
            tmp_path,
            """\
            DATA:
              - type: tabulated n
                data: |
                    0.50 1.40
                    1.00 1.45
            """,
        )
        _lam, _n, k = read_refractiveindex_yaml(path)
        assert np.all(k == 0.0)

    def test_formula_and_tabulated_k_are_merged(self, tmp_path):
        path = self._write(
            tmp_path,
            """\
            DATA:
              - type: formula 2
                wavelength_range: 0.5 1.6
                coefficients: 0 0.75831 0.01007 0.08495 8.91377
              - type: tabulated k
                data: |
                    0.50 1.0e-9
                    1.60 1.0e-4
            """,
        )
        lam, n, k = read_refractiveindex_yaml(path, num_points=25)
        assert lam.size == 25
        assert n.min() > 1.3 and n.max() < 1.4  # water in the visible / NIR
        assert k[0] == pytest.approx(1e-9, rel=1e-6)
        assert k[-1] == pytest.approx(1e-4, rel=1e-6)

    def test_formula_1_matches_the_data_sheet(self, tmp_path):
        # Malitson fused silica, the canonical formula-1 record.
        path = self._write(
            tmp_path,
            """\
            DATA:
              - type: formula 1
                wavelength_range: 0.21 6.7
                coefficients: 0 0.6961663 0.0684043 0.4079426 0.1162414 0.8974794 9.896161
            """,
        )
        lam, n, _k = read_refractiveindex_yaml(path, wavelengths_m=np.array([1.55e-6]))
        b = (0.6961663, 0.4079426, 0.8974794)
        c = (0.0684043, 0.1162414, 9.896161)
        expected = math.sqrt(1.0 + sum(bi * 1.55**2 / (1.55**2 - ci**2) for bi, ci in zip(b, c, strict=True)))
        assert lam[0] == pytest.approx(1.55e-6)
        assert n[0] == pytest.approx(expected, rel=1e-12)
        assert n[0] == pytest.approx(1.444, abs=1e-3)

    def test_formula_3_polynomial(self, tmp_path):
        path = self._write(
            tmp_path,
            """\
            DATA:
              - type: formula 3
                wavelength_range: 0.43 1.1
                coefficients: 2.986556 0.01828907 -2 -0.01445419 2
            """,
        )
        _lam, n, _k = read_refractiveindex_yaml(path, wavelengths_m=np.array([0.5e-6]))
        expected = math.sqrt(2.986556 + 0.01828907 * 0.5**-2 - 0.01445419 * 0.5**2)
        assert n[0] == pytest.approx(expected, rel=1e-12)

    def test_formula_4(self, tmp_path):
        # DeVore TiO2 (ordinary): n^2 = 5.913 + 0.2441 / (lam^2 - 0.0803)
        path = self._write(
            tmp_path,
            """\
            DATA:
              - type: formula 4
                wavelength_range: 0.43 1.53
                coefficients: 5.913 0.2441 0 0.0803 1 0 0 0 1
            """,
        )
        _lam, n, _k = read_refractiveindex_yaml(path, wavelengths_m=np.array([0.5893e-6]))
        expected = math.sqrt(5.913 + 0.2441 / (0.5893**2 - 0.0803))
        assert n[0] == pytest.approx(expected, rel=1e-12)
        assert n[0] == pytest.approx(2.613, abs=2e-3)  # rutile ordinary ray at the sodium D line

    def test_unsupported_formula_raises(self, tmp_path):
        path = self._write(
            tmp_path,
            """\
            DATA:
              - type: formula 5
                wavelength_range: 0.4 1.0
                coefficients: 1.5 0.01 -2
            """,
        )
        with pytest.raises(ValueError, match="formula 5 is not supported"):
            read_refractiveindex_yaml(path)

    def test_out_of_range_request_raises(self, tmp_path):
        path = self._write(
            tmp_path,
            """\
            DATA:
              - type: formula 1
                wavelength_range: 0.21 6.7
                coefficients: 0 0.6961663 0.0684043 0.4079426 0.1162414 0.8974794 9.896161
            """,
        )
        with pytest.raises(ValueError, match="outside the validity"):
            read_refractiveindex_yaml(path, wavelengths_m=np.array([10e-6]))

    def test_missing_data_raises(self, tmp_path):
        path = self._write(tmp_path, "REFERENCES: nothing\n")
        with pytest.raises(ValueError, match="no DATA"):
            read_refractiveindex_yaml(path)


@requires_db
class TestFitDatabaseRecords:
    def test_fused_silica_two_lossless_poles(self):
        """Malitson fused silica, 0.4-2 um, two lossless Lorentz (Sellmeier) poles."""
        assert _DB is not None
        lam_req = np.linspace(0.4e-6, 2.0e-6, 80)
        lam, n, k = read_refractiveindex_yaml(_DB / "data/main/SiO2/nk/Malitson.yml", wavelengths_m=lam_req)
        assert np.all(k == 0.0)
        result = fit_dispersion(lam, n, k, num_poles=2, kinds=("sellmeier",), max_starts=8)

        omega = 2.0 * math.pi * c_light / lam
        n_fit = np.sqrt(_eps_of(result, omega))
        assert np.max(np.abs(np.abs(n_fit) - n)) < 1e-4
        assert result.passive
        assert all(isinstance(p, SellmeierPole) for p in result.model.poles)
        # The two poles bracket the transparency window: one UV, one IR.
        lam0 = sorted(2.0 * math.pi * c_light / p.omega_0 for p in result.model.poles)
        assert lam0[0] < 0.4e-6 < 2.0e-6 < lam0[1]

    def test_gold_johnson_christy_drude_lorentz(self):
        """Johnson & Christy gold, 0.6-1.6 um, one Drude plus one Lorentz pole."""
        assert _DB is not None
        lam_all, n_all, k_all = read_refractiveindex_yaml(_DB / "data/main/Au/nk/Johnson.yml")
        sel = (lam_all >= 0.6e-6) & (lam_all <= 1.6e-6)
        lam, n, k = lam_all[sel], n_all[sel], k_all[sel]
        assert lam.size >= 8, f"expected the J&C table to cover the NIR, got {lam.size} points"

        result = fit_dispersion(lam, n, k, num_poles=2, kinds=("drude", "lorentz"), max_starts=12, seed=1)

        # |eps| runs from roughly 15 to 120 across this band, so an RMS of 0.5
        # is well under 1 % of the permittivity being fitted.
        assert result.rms < 0.5, result.report
        assert result.max_abs_error < 1.5, result.report
        assert result.passive, result.report
        # Gold in the NIR is Drude-dominated; the fit must actually use one.
        assert any(isinstance(p, DrudePole) for p in result.model.poles), result.report
        drude = next(p for p in result.model.poles if isinstance(p, DrudePole))
        assert 1.0e16 < float(drude.plasma_frequency) < 1.6e16, result.report
