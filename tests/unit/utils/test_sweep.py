"""Unit tests for the parameter-sweep runner.

``run_fdtd`` is replaced by a stub: these tests cover the sweep's bookkeeping (point expansion,
caching, the table and its exports), not the engine. ``tests/simulation`` covers a real run.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from fdtdx.utils.sweep import SweepResult, run_sweep


@pytest.fixture
def stub_engine(monkeypatch):
    """Replace run_fdtd with a stub that echoes the set-up build() handed it."""

    def fake_run_fdtd(arrays, objects, config, show_progress=False, **kwargs):
        del objects, config, show_progress, kwargs
        return (0, arrays)

    monkeypatch.setattr("fdtdx.utils.sweep.run_fdtd", fake_run_fdtd)


class _Counter:
    """A build() that records how often it was called."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **point):
        self.calls.append(dict(point))
        return (point, "objects", "config")


def _evaluate(result, **point):
    del point
    setup = result[1]
    return {"loss": float(setup["gap"]) * 2.0}


def test_cartesian_product_and_table(stub_engine):
    """A two-parameter mapping sweeps the full product; the table keeps parameters then metrics."""
    build = _Counter()
    result = run_sweep(
        build,
        {"gap": [0.1, 0.2], "width": [0.5, 0.6, 0.7]},
        lambda res, **pt: {"loss": res[1]["gap"] + res[1]["width"]},
    )

    assert isinstance(result, SweepResult)
    assert len(result) == 6
    assert len(build.calls) == 6
    assert result.param_names == ["gap", "width"]
    assert result.metric_names == ["loss"]
    assert result.columns == ["gap", "width", "loss"]
    # Last parameter varies fastest.
    assert [row["width"] for row in result.rows[:3]] == [0.5, 0.6, 0.7]
    assert result.rows[0] == {"gap": 0.1, "width": 0.5, "loss": 0.6}
    assert result.n_runs == 6


def test_explicit_point_list(stub_engine):
    """A list of dicts sweeps exactly those points, product or not."""
    build = _Counter()
    points = [{"gap": 0.1, "width": 0.5}, {"gap": 0.3, "width": 0.9}]
    result = run_sweep(build, points, _evaluate)

    assert len(result) == 2
    assert build.calls == points
    assert [row["loss"] for row in result.rows] == [0.2, 0.6]


def test_cache_skips_the_second_run(stub_engine, tmp_path):
    """The second call over the same points reads the cache: build() is never entered again."""
    first_build = _Counter()
    first = run_sweep(first_build, {"gap": [0.1, 0.2]}, _evaluate, cache_dir=tmp_path)

    assert len(first_build.calls) == 2
    assert first.from_cache == [False, False]
    assert first.n_runs == 2
    assert len(list(tmp_path.glob("*.json"))) == 2

    second_build = _Counter()
    second = run_sweep(second_build, {"gap": [0.1, 0.2]}, _evaluate, cache_dir=tmp_path)

    assert second_build.calls == []  # no build, therefore no run
    assert second.from_cache == [True, True]
    assert second.n_runs == 0
    assert [row["loss"] for row in second.rows] == [row["loss"] for row in first.rows]


def test_cache_tag_invalidates(stub_engine, tmp_path):
    """A different tag is a different cache key, so the point is rebuilt and rerun."""
    run_sweep(_Counter(), {"gap": [0.1]}, _evaluate, cache_dir=tmp_path, tag="v1")
    build = _Counter()
    result = run_sweep(build, {"gap": [0.1]}, _evaluate, cache_dir=tmp_path, tag="v2")

    assert len(build.calls) == 1
    assert result.from_cache == [False]


def test_cache_only_covers_the_points_it_holds(stub_engine, tmp_path):
    """Extending a swept range reruns only the new points."""
    run_sweep(_Counter(), {"gap": [0.1, 0.2]}, _evaluate, cache_dir=tmp_path)
    build = _Counter()
    result = run_sweep(build, {"gap": [0.1, 0.2, 0.3]}, _evaluate, cache_dir=tmp_path)

    assert [call["gap"] for call in build.calls] == [0.3]
    assert result.from_cache == [True, True, False]


def test_threaded_evaluate_keeps_row_order(stub_engine):
    """max_workers only threads evaluate; rows still follow the sweep order."""
    result = run_sweep(_Counter(), {"gap": [0.4, 0.1, 0.3, 0.2]}, _evaluate, max_workers=3)

    assert [row["gap"] for row in result.rows] == [0.4, 0.1, 0.3, 0.2]
    assert [row["loss"] for row in result.rows] == [0.8, 0.2, 0.6, 0.4]


def test_to_numpy_and_to_csv(stub_engine, tmp_path):
    result = run_sweep(_Counter(), {"gap": [0.1, 0.2]}, _evaluate)

    table = result.to_numpy()
    assert table.shape == (2, 2)
    assert np.allclose(table, [[0.1, 0.2], [0.2, 0.4]])
    assert np.allclose(result.to_numpy(["loss"]), [[0.2], [0.4]])

    path = result.to_csv(tmp_path / "sweep.csv")
    lines = path.read_text().splitlines()
    assert lines[0] == "gap,loss"
    assert lines[1] == "0.1,0.2"


def test_plot_returns_a_figure(stub_engine):
    result = run_sweep(_Counter(), {"gap": [0.3, 0.1, 0.2]}, _evaluate)
    fig = result.plot("gap", "loss")

    line = fig.axes[0].lines[0]
    assert list(line.get_xdata()) == [0.1, 0.2, 0.3]  # sorted in x
    assert fig.axes[0].get_xlabel() == "gap"
    assert fig.axes[0].get_ylabel() == "loss"


def test_backend_is_forced_when_requested(stub_engine, monkeypatch):
    """A `backend` argument wraps every run in fdtdx.use_backend."""
    seen = []

    def fake_run_fdtd(arrays, objects, config, show_progress=False, **kwargs):
        from fdtdx.backend.context import get_backend_override

        del objects, config, show_progress, kwargs
        seen.append(get_backend_override())
        return (0, arrays)

    monkeypatch.setattr("fdtdx.utils.sweep.run_fdtd", fake_run_fdtd)
    run_sweep(_Counter(), {"gap": [0.1, 0.2]}, _evaluate, backend="jax")

    assert seen == ["jax", "jax"]


def test_bad_arguments_raise(stub_engine):
    with pytest.raises(ValueError, match="max_workers"):
        run_sweep(_Counter(), {"gap": [0.1]}, _evaluate, max_workers=0)
    with pytest.raises(ValueError, match="mapping"):
        run_sweep(_Counter(), {"gap": [0.1]}, lambda res, **pt: 1.0)
