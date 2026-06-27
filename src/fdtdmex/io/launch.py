"""``run_simulation_from_hdf5`` — the agent-facing, non-blocking, job-folder-owning launcher.

This is what the agent calls. Given a packed config HDF5 (from :func:`fdtdmex.io.pack`) and a
``parent_folder`` (ag-fdtd forces it to the workspace ``jobs/`` dir), it:

1. Hashes the HDF5 → a **fresh unique** job id (content hash + entropy ⇒ unique per call, so the same
   packed bundle can launch many distinct runs).
2. Stages a job folder ``<parent_folder>/<simulation_name or job_id>/`` with an ``outputs/`` subdir,
   copies the HDF5 in, and drops a lightweight ``config.json`` snapshot (UI peek without cracking HDF5).
3. Writes the initial ``queued`` ``status.json`` synchronously (so it is present the instant we
   return — *a job folder with no ``status.json`` means the sim was staged but never run*).
4. **Launches the solver detached** (a ``python -m fdtdmex.io._runner`` child whose cwd is the job
   folder; it calls :func:`fdtdmex.io.run_simulation`, advancing ``status.json`` →
   ``running → completed|failed`` and writing the results into ``outputs/``).
5. **Returns immediately** with a :class:`JobHandle`. Non-blocking.

``backend="mock"`` flows the whole way (the detached child fabricates ticks + results, GPU-free).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger

from .status import StatusWriter

# Results land in this subfolder of the job dir (status.json/progress.jsonl stay at the job-dir top).
RESULTS_REL = "outputs/result.hdf5"


@dataclass(frozen=True)
class JobHandle:
    """Pointers to a launched job. Returned immediately by :func:`run_simulation_from_hdf5`.

    The detached solver updates ``status_path`` as it runs; poll it (``queued → running →
    completed|failed``) to track the job.
    """

    run_id: str
    job_dir: Path
    status_path: Path
    bundle_hdf5: Path
    results_path: Path
    pid: int

    def as_dict(self) -> dict:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


def _hash_file(path: Path) -> str:
    """SHA-256 of the file contents (first 12 hex chars are the job-id stem)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pid_alive(pid: int) -> bool:
    """True if `pid` is a live process. `os.kill(pid, 0)` raises ProcessLookupError when
    dead, PermissionError when alive-but-foreign (treat as alive)."""
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _active_run_guard(job_dir: Path, simulation_name: str | None) -> None:
    """Refuse to clobber a job folder whose run is STILL ACTIVE. A fixed `simulation_name`
    reuses the folder, so a second launch while the first is queued/running would otherwise
    delete its status.json/outputs mid-flight and double-fire the solver. If the existing
    status.json is queued/running AND its pid is alive, raise BEFORE any cleanup. A stale
    record (terminal status, or a dead pid) is fine to overwrite (the normal re-run path)."""
    import json

    status_path = job_dir / "status.json"
    if not status_path.is_file():
        return
    try:
        rec = json.loads(status_path.read_text())
    except (OSError, ValueError):
        return  # unreadable/half-written → let the normal cleanup handle it
    status = rec.get("status")
    if status in ("queued", "running", "submitted") and _pid_alive(int(rec.get("pid") or 0)):
        raise RuntimeError(
            f"simulation {simulation_name or job_dir.name!r} is already running "
            f"(status={status}, pid={rec.get('pid')}). Wait for it to finish, or launch under a "
            f"different simulation_name — refusing to overwrite an in-flight run."
        )


def _read_config_json_bytes(hdf5_path: Path) -> bytes | None:
    """Pull the embedded ``/config/json`` (lightweight editable config) from the HDF5, if present."""
    import h5py
    import numpy as np

    with h5py.File(hdf5_path, "r") as f:
        if "config" in f and "json" in f["config"]:
            return np.asarray(f["config"]["json"]).tobytes()
    return None


def run_simulation_from_hdf5(
    hdf5_path: str | Path | os.PathLike,
    parent_folder: str | Path,
    *,
    simulation_name: str | None = None,
    backend: str = "mlx",
    name: str = "",
) -> JobHandle:
    """Stage a job folder and launch the solver **detached**, returning immediately.

    Args:
        hdf5_path: A packed config HDF5 (from :func:`fdtdmex.io.pack`). A :class:`PackResult` works too.
        parent_folder: The folder to create the job folder under (ag-fdtd forces it to ``jobs/``).
        simulation_name: Optional explicit job-folder name; defaults to the unique job id.
        backend: ``"mlx"`` (the real engine) or ``"mock"`` (GPU-free, end-to-end).
        name: Human-readable job name (recorded in ``status.json``).

    Returns:
        A :class:`JobHandle` with the run id, job dir, status path, copied bundle, and child pid.
    """
    hdf5_path = Path(os.fspath(hdf5_path))
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"run_simulation_from_hdf5: no such HDF5 bundle: {hdf5_path}")

    # 1. Fresh unique job id: content hash + entropy (a second call on the same HDF5 ⇒ distinct id).
    run_id = f"{_hash_file(hdf5_path)[:12]}-{uuid.uuid4().hex[:8]}"

    # 2. Stage the job folder + a FRESH outputs/. When a fixed simulation_name reuses an
    #    existing folder (a re-run / fast-forward), CLEAR the prior run's artifacts so a
    #    premature load of outputs/result.hdf5 ERRORS (file absent until this run finishes)
    #    instead of silently returning STALE results — the natural stop signal for a
    #    fast-forward that runs past a not-yet-complete simulation.
    parent_folder = Path(parent_folder)
    job_dir = parent_folder / (simulation_name or run_id)
    # Guard FIRST: never delete an in-flight run's status/outputs (a same-name relaunch while it's
    # still active). Raises before any cleanup so the cell errors instead of double-firing the sim.
    _active_run_guard(job_dir, simulation_name)
    outputs_dir = job_dir / "outputs"
    if outputs_dir.exists():
        shutil.rmtree(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    progress_jsonl = job_dir / "progress.jsonl"
    if progress_jsonl.exists():
        progress_jsonl.unlink()  # reset the append-only tick bus for the new run
    # DELETE the prior run's status.json (don't just overwrite it): a fixed-name re-run reuses the
    # folder, so removing it first makes the new queued status a clean CREATE event the watcher
    # picks up as a fresh run (and a monitor that keys off run_id sees the change unambiguously).
    old_status = job_dir / "status.json"
    if old_status.exists():
        old_status.unlink()
    old_log = job_dir / "runner.log"
    if old_log.exists():
        old_log.unlink()

    # 3. Copy the bundle in; snapshot the lightweight config.json (UI peek without cracking HDF5).
    bundle_hdf5 = job_dir / hdf5_path.name
    shutil.copy2(hdf5_path, bundle_hdf5)
    cfg_bytes = _read_config_json_bytes(bundle_hdf5)
    if cfg_bytes is not None:
        (job_dir / "config.json").write_bytes(cfg_bytes)

    # 4. Initial queued status.json — guaranteed present the instant we return (closes the watcher race).
    status_path = job_dir / "status.json"
    StatusWriter(status_path, run_id=run_id, name=name)  # writes the "queued" state on construction

    # 5. Launch detached: a child whose cwd IS job_dir, running run_simulation on the copied bundle.
    log_path = job_dir / "runner.log"
    cmd = [
        sys.executable,
        "-m",
        "fdtdmex.io._runner",
        "--hdf5",
        bundle_hdf5.name,  # relative to cwd=job_dir
        "--backend",
        backend,
        "--run-id",
        run_id,
        "--name",
        name,
        "--results-name",
        RESULTS_REL,
    ]
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(job_dir),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach from the kernel's process group → survives the cell
            close_fds=True,
        )

    logger.info(f"run_simulation_from_hdf5: launched run {run_id!r} (backend={backend}, pid={proc.pid}) → {job_dir}")
    return JobHandle(
        run_id=run_id,
        job_dir=job_dir,
        status_path=status_path,
        bundle_hdf5=bundle_hdf5,
        results_path=job_dir / RESULTS_REL,
        pid=proc.pid,
    )
