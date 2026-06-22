"""Unit tests for the SMatrixResult dataclass and its plotting/serialization front-end."""

import matplotlib
import numpy as np

matplotlib.use("Agg")

from fdtdx.utils.smatrix import SMatrixResult, plot_smatrix


def _sample() -> SMatrixResult:
    sparams = {
        ("out1", "in1"): np.array([0.97 + 0.02j]),
        ("out2", "in1"): np.array([0.03 - 0.01j]),
        ("out1", "in2"): np.array([0.04 + 0.0j]),
        ("out2", "in2"): np.array([0.95 - 0.05j]),
    }
    return SMatrixResult.from_sparams(sparams, frequencies=[1.93e14])


def test_ports_and_matrix():
    sm = _sample()
    assert sm.out_ports() == ["out1", "out2"]
    assert sm.in_ports() == ["in1", "in2"]
    S = sm.matrix(0)
    assert S.shape == (2, 2)
    assert np.isclose(S[0, 0], 0.97 + 0.02j)
    assert np.isclose(S[1, 1], 0.95 - 0.05j)


def test_json_roundtrip(tmp_path):
    sm = _sample()
    path = tmp_path / "smatrix.json"
    sm.to_json(path)
    sm2 = SMatrixResult.load_json(path)
    assert np.allclose(sm.matrix(0), sm2.matrix(0))
    assert sm2.frequencies == [1.93e14]


def test_plot_smatrix_runs(tmp_path):
    sm = _sample()
    for value in ("magnitude", "magnitude_db", "phase"):
        fig = plot_smatrix(sm, value=value, filename=tmp_path / f"{value}.png")
        assert (tmp_path / f"{value}.png").exists()
        del fig


def test_missing_entry_is_nan():
    sparams = {("out1", "in1"): np.array([1.0 + 0j])}  # out2/in absent
    sm = SMatrixResult.from_sparams(sparams)
    S = sm.matrix(0)
    assert S.shape == (1, 1)
    assert np.isclose(S[0, 0], 1.0)
