"""``run_simulation`` — the blocking worker that drives a per-job ``status.json`` in the **cwd**.

ag-fdtd's telemetry no longer parses ``PROGRESS`` stdout from a single adapter; it **watches a
per-job ``status.json`` file** so a run launched by any path is tracked uniformly. In the v2 model
**fdtdmex owns the job folder**: :func:`fdtdmex.io.run_simulation_from_hdf5` stages it and launches a
detached child whose **current working directory IS that job folder**. The child calls
:func:`run_simulation`, which writes the folder's contents — ``status.json`` (and an append-only
``progress.jsonl``), plus the results HDF5 — straight into the cwd, driven entirely off the existing
``progress(step, total)`` callback ``sim_run`` already streams. (This is the v1 ``run_with_status``
logic retargeted at the cwd; the orchestrator no longer owns/names the folder.)

The schema is ag-fdtd's (we write it verbatim)::

    {"run_id", "name", "solver": "fdtdmex",
     "status": "queued|running|completed|failed",
     "step", "total", "heartbeat": <epoch>,
     "started_at": <epoch>, "finished_at": <epoch|null>,
     "pid": <int>, "error": <str|null>}

``status.json`` is written **atomically** (temp file in the same dir + ``os.replace``), so a watcher
never reads a half-written file. The ``mock`` backend drives the same ticks (see
:mod:`fdtdmex.io.mock`), so ag-fdtd's offline, GPU-free path exercises this end-to-end.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from loguru import logger

from .run import sim_run

Status = Literal["queued", "running", "completed", "failed"]


class StatusWriter:
    """Owns a job's ``status.json`` (+ optional ``progress.jsonl``) and writes them atomically.

    Construction emits the initial ``"queued"`` state immediately. Drive it from a ``sim_run``
    progress callback via :meth:`tick`, then call :meth:`complete` or :meth:`fail` on the terminal
    state.
    """

    def __init__(
        self,
        status_path: str | Path,
        *,
        run_id: str,
        name: str = "",
        solver: str = "fdtdmex",
        jsonl_path: str | Path | None = None,
    ) -> None:
        self._path = Path(status_path)
        self._jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        now = time.time()
        self._state: dict = {
            "run_id": run_id,
            "name": name,
            "solver": solver,
            "status": "queued",
            "step": 0,
            "total": 0,
            "heartbeat": now,
            "started_at": now,
            "finished_at": None,
            "pid": os.getpid(),
            "error": None,
        }
        self._write()

    def _write(self) -> None:
        """Atomically replace ``status.json`` (temp file in the same dir + ``os.replace``)."""
        tmp = self._path.with_name(f".{self._path.name}.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(self._state))
        os.replace(tmp, self._path)

    def tick(self, step: int, total: int) -> None:
        """Advance to ``"running"``, refresh ``step``/``total``/``heartbeat``, and append a jsonl line."""
        self._state.update(status="running", step=int(step), total=int(total), heartbeat=time.time())
        self._write()
        if self._jsonl_path is not None:
            with self._jsonl_path.open("a") as f:
                f.write(
                    json.dumps({"step": int(step), "total": int(total), "heartbeat": self._state["heartbeat"]}) + "\n"
                )

    def complete(self) -> None:
        """Terminal success: ``"completed"``, ``step = total``, ``finished_at`` set."""
        now = time.time()
        self._state.update(status="completed", step=self._state["total"], heartbeat=now, finished_at=now)
        self._write()

    def fail(self, err: BaseException | str) -> None:
        """Terminal failure: ``"failed"``, ``error`` captured, ``finished_at`` set."""
        now = time.time()
        self._state.update(status="failed", error=str(err), heartbeat=now, finished_at=now)
        self._write()


def run_simulation(
    config_or_hdf5: str | Path,
    *,
    backend: Literal["mlx", "mock"] = "mlx",
    progress: Callable[[int, int], None] | None = None,
    run_id: str | None = None,
    name: str = "",
    results_name: str = "result.hdf5",
) -> Path:
    """Blocking worker: run a config HDF5 **in the current working directory**, writing telemetry there.

    Writes ``status.json`` + ``progress.jsonl`` at the cwd top and the results HDF5 at
    ``<cwd>/<results_name>`` (``results_name`` may name a subdir, e.g. ``"outputs/result.hdf5"``).
    The status file goes ``queued → running (step/total advancing) → completed`` (or ``failed`` on an
    exception, which is re-raised). This is the primitive a detached child executes — **not** an
    agent-facing call; the agent uses :func:`fdtdmex.io.run_simulation_from_hdf5` instead. The
    ``mock`` backend drives the same ticks GPU-free.

    Args:
        config_or_hdf5: A config HDF5 produced by :func:`fdtdmex.io.pack` / :func:`fdtdmex.io.sim_init`.
        backend: ``"mlx"`` (the real engine) or ``"mock"`` (schema-valid synthetic results, no GPU).
        progress: Optional user ``progress(step, total)`` callback, chained after the status update.
        run_id: The job id recorded in ``status.json`` (defaults to a fresh uuid4 hex).
        name: Human-readable job name (recorded in ``status.json``).
        results_name: Results HDF5 path relative to the cwd (default ``result.hdf5``).

    Returns:
        The written results path (``<cwd>/<results_name>``).
    """
    run_id = run_id or uuid.uuid4().hex
    cwd = Path.cwd()
    results_path = cwd / results_name
    results_path.parent.mkdir(parents=True, exist_ok=True)

    writer = StatusWriter(
        cwd / "status.json",
        run_id=run_id,
        name=name,
        jsonl_path=cwd / "progress.jsonl",
    )

    def _on_progress(step: int, total: int) -> None:
        writer.tick(step, total)
        if progress is not None:
            progress(step, total)

    try:
        sim_run(config_or_hdf5, results_path, backend=backend, progress=_on_progress)
        writer.complete()
    except Exception as exc:
        writer.fail(exc)
        logger.error(f"run_simulation: run {run_id!r} failed → {exc}")
        raise

    logger.info(f"run_simulation: run {run_id!r} completed (backend={backend}) → {results_path}")
    return results_path
