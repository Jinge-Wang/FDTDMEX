"""Parameter sweeps — run one simulation per point, reduce each to numbers, cache, tabulate, plot.

A photonic study is nearly always the same script: build a scene from a few parameters (a gap, a
width, a wavelength), run it, reduce the detector states to a handful of numbers, and put those
numbers in a table or a curve. :func:`run_sweep` is that script with the bookkeeping removed — the
Cartesian product of the parameters, a content-addressed cache so an interrupted sweep resumes
instead of restarting, and a tidy table that goes straight to CSV, numpy or matplotlib.

The two functions you supply are the whole contract:

.. code-block:: python

    def build(**point) -> tuple[ArrayContainer, ObjectContainer, SimulationConfig]: ...
    def evaluate(result, **point) -> dict[str, float]: ...

``build`` sets the scene up completely (it is the place to call
:func:`~fdtdx.place_objects` and :func:`~fdtdx.apply_params`) and ``evaluate`` receives the
:class:`~fdtdx.SimulationState` that :func:`~fdtdx.run_fdtd` returned, so
``result[1].detector_states`` is where the numbers come from.

Limits
------
Runs are **serial by design**. ``max_workers`` parallelises only the ``evaluate`` step, on threads,
because the post-processing is usually numpy/scipy work that releases the GIL; the simulations
themselves are never overlapped, since concurrent JAX (or MLX) executions on one device are not
thread-safe and would contend for the same memory anyway. For real parallelism, run several sweeps
as separate processes over disjoint slices of the parameter space and let them share one
``cache_dir``.

The cache keys on the parameter values and a ``tag``, not on the contents of ``build``/``evaluate``:
change the physics of the set-up without changing a parameter and you must pass a new ``tag`` (or
clear the directory), or you will read back the old numbers.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fdtdx.backend.context import use_backend
from fdtdx.fdtd.wrapper import run_fdtd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure


@dataclass
class SweepResult:
    """The table produced by :func:`run_sweep`: one row per parameter point.

    Each row is a flat ``dict`` holding the point's parameters and the metrics ``evaluate``
    returned for it, in that order. Rows follow the order of the sweep's points.
    """

    #: One flat ``{parameter..., metric...}`` dict per point.
    rows: list[dict[str, Any]]
    #: Parameter names, in sweep order.
    param_names: list[str] = field(default_factory=list)
    #: Metric names, in first-seen order.
    metric_names: list[str] = field(default_factory=list)
    #: Per-row flag: ``True`` when the row was read from the cache and no simulation was run.
    from_cache: list[bool] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        """Column order of the table: parameters first, then metrics."""
        return [*self.param_names, *self.metric_names]

    @property
    def n_runs(self) -> int:
        """How many simulations this sweep actually ran (cache hits excluded)."""
        return int(sum(1 for hit in self.from_cache if not hit))

    def __len__(self) -> int:
        return len(self.rows)

    def numeric_columns(self) -> list[str]:
        """Columns whose value is a real number in every row."""
        out = []
        for name in self.columns:
            try:
                for row in self.rows:
                    float(row[name])
            except (TypeError, ValueError):
                continue
            out.append(name)
        return out

    def to_numpy(self, columns: Sequence[str] | None = None) -> np.ndarray:
        """Return the table as a 2-D float array of shape ``(n_points, n_columns)``.

        Args:
            columns: which columns to take, in order. Defaults to every numeric column
                (see :meth:`numeric_columns`).

        Returns:
            The float array. Empty with shape ``(0, 0)`` when the sweep has no rows.

        Raises:
            ValueError: if a requested column does not exist or does not hold numbers.
        """
        cols = list(columns) if columns is not None else self.numeric_columns()
        if not self.rows or not cols:
            return np.zeros((len(self.rows), len(cols)), dtype=float)
        try:
            return np.array([[float(row[c]) for c in cols] for row in self.rows], dtype=float)
        except KeyError as exc:
            raise ValueError(f"unknown column {exc.args[0]!r}; have {self.columns}") from exc
        except TypeError as exc:
            raise ValueError(f"column selection {cols} holds non-numeric values") from exc

    def to_csv(self, path: str | Path) -> Path:
        """Write the table as CSV (header = :attr:`columns`) and return the path written."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({c: row.get(c, "") for c in self.columns})
        return out

    def plot(self, x: str, y: str) -> "Figure":
        """Plot one column against another, points joined in ascending ``x`` order.

        Args:
            x: column for the horizontal axis (usually a swept parameter).
            y: column for the vertical axis (usually a metric).

        Returns:
            The matplotlib figure, so the caller can restyle or save it.
        """
        import matplotlib.pyplot as plt

        data = self.to_numpy([x, y])
        order = np.argsort(data[:, 0]) if data.size else np.zeros(0, dtype=int)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(data[order, 0], data[order, 1], "o-", linewidth=1.5, markersize=4)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig


def _json_default(obj: Any) -> Any:
    """Make numpy scalars / arrays and anything else hashable in a stable JSON form."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return repr(obj)


def _expand_points(params: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[str]]:
    """Turn the ``params`` argument into an explicit point list plus the parameter-name order."""
    if isinstance(params, Mapping):
        names = list(params.keys())
        if not names:
            return [], []
        values = [list(params[name]) for name in names]
        points = [dict(zip(names, combo)) for combo in itertools.product(*values)]
        return points, names

    points = [dict(point) for point in params]
    names = []
    for point in points:
        for name in point:
            if name not in names:
                names.append(name)
    return points, names


def _cache_path(cache_dir: Path, point: Mapping[str, Any], tag: str) -> Path:
    """Content-addressed cache file for one point: sha256 over the sorted values plus the tag."""
    payload = json.dumps(
        {"tag": tag, "params": {k: point[k] for k in sorted(point)}},
        sort_keys=True,
        default=_json_default,
    )
    return cache_dir / f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}.json"


def _read_cache(path: Path) -> dict[str, Any] | None:
    """Return the cached metric dict, or ``None`` when the file is absent or unreadable."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    metrics = payload.get("metrics")
    return dict(metrics) if isinstance(metrics, dict) else None


def run_sweep(
    build: Callable[..., tuple[Any, Any, Any]],
    params: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]],
    evaluate: Callable[..., Mapping[str, Any]],
    *,
    cache_dir: str | Path | None = None,
    backend: str | None = None,
    max_workers: int = 1,
    tag: str = "",
    show_progress: bool = False,
) -> SweepResult:
    """Run one FDTD simulation per parameter point and collect the results in a table.

    For every point the sweep calls ``build(**point)`` for a set-up, hands that to
    :func:`~fdtdx.run_fdtd`, and calls ``evaluate(result, **point)`` to reduce the run to numbers.
    With a ``cache_dir`` the metric dict of each point is stored under a hash of its parameter
    values and the ``tag``; on a later call a point whose file is present is **not built and not
    run**, so an interrupted sweep resumes and a re-analysis costs nothing.

    Args:
        build: set-up function, ``build(**point) -> (arrays, objects, config)``. It owns the whole
            scene: it may call :func:`~fdtdx.place_objects` and :func:`~fdtdx.apply_params`, read
            files, or build geometry. It is only called for points that miss the cache.
        params: either a mapping ``{name: sequence_of_values}``, swept as the full Cartesian product
            with the last name varying fastest, or an explicit sequence of ``{name: value}`` dicts
            when the points are not a product grid.
        evaluate: reduction, ``evaluate(result, **point) -> dict[str, float]`` where ``result`` is
            the :class:`~fdtdx.SimulationState` returned by :func:`~fdtdx.run_fdtd` — the pair
            ``(time_step, arrays)``, so ``result[1].detector_states`` holds the recordings. The
            returned values must be JSON-serialisable to be cacheable.
        cache_dir: directory for the per-point JSON cache. ``None`` disables caching. Created if it
            does not exist.
        backend: ``"mlx"`` or ``"jax"`` to force a backend for every run (see
            :func:`~fdtdx.use_backend`); ``None`` leaves the automatic routing alone.
        max_workers: threads used for the ``evaluate`` step only. The simulations always run one at
            a time in the calling thread: overlapping JAX/MLX executions on one device is not
            thread-safe. Values above 1 help only when ``evaluate`` is itself heavy.
        tag: an arbitrary label folded into the cache key. Change it whenever ``build`` or
            ``evaluate`` changes in a way that should invalidate stored results.
        show_progress: pass-through to :func:`~fdtdx.run_fdtd`'s per-run progress bar; off by
            default so a long sweep keeps a readable log.

    Returns:
        The :class:`SweepResult` table, one row per point in sweep order.

    Raises:
        ValueError: if ``max_workers`` is below 1, or ``evaluate`` returns something other than a
            mapping.
    """
    if max_workers < 1:
        raise ValueError(f"max_workers must be at least 1 (got {max_workers})")

    points, param_names = _expand_points(params)
    cache_root = Path(cache_dir) if cache_dir is not None else None
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)

    metrics: list[dict[str, Any] | None] = [None] * len(points)
    from_cache = [False] * len(points)
    paths = [_cache_path(cache_root, point, tag) if cache_root is not None else None for point in points]

    pool = ThreadPoolExecutor(max_workers=max_workers) if max_workers > 1 else None
    futures: dict[int, Future] = {}
    try:
        for index, point in enumerate(points):
            path = paths[index]
            if path is not None:
                hit = _read_cache(path)
                if hit is not None:
                    metrics[index] = hit
                    from_cache[index] = True
                    continue

            arrays, objects, config = build(**point)
            ctx = use_backend(backend) if backend is not None else contextlib.nullcontext()  # ty: ignore
            with ctx:
                state = run_fdtd(arrays=arrays, objects=objects, config=config, show_progress=show_progress)

            if pool is not None:
                futures[index] = pool.submit(evaluate, state, **point)
            else:
                metrics[index] = _as_metrics(evaluate(state, **point))
        for index, future in futures.items():
            metrics[index] = _as_metrics(future.result())
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    metric_names: list[str] = []
    rows: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        values = metrics[index] or {}
        for name in values:
            if name not in metric_names:
                metric_names.append(name)
        rows.append({**point, **values})
        path = paths[index]
        if path is not None and not from_cache[index]:
            _write_cache(path, point, tag, values)

    return SweepResult(rows=rows, param_names=param_names, metric_names=metric_names, from_cache=from_cache)


def _as_metrics(value: Any) -> dict[str, Any]:
    """Validate and copy the dict returned by a user's ``evaluate``."""
    if not isinstance(value, Mapping):
        raise ValueError(f"evaluate must return a mapping of metric name to value, got {type(value).__name__}")
    return dict(value)


def _write_cache(path: Path, point: Mapping[str, Any], tag: str, values: Mapping[str, Any]) -> None:
    """Store one point's metrics. A value that will not serialise leaves the point uncached."""
    payload = {"tag": tag, "params": dict(point), "metrics": dict(values)}
    try:
        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    except OSError:
        pass


__all__ = ["SweepResult", "run_sweep"]
