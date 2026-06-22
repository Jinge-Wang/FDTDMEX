"""``sim_run`` — execute a config HDF5 on any FDTDMEX machine, writing a results HDF5.

The unwrap deserializes the resolved run seam straight back into an ``MLXState`` + frozen plans and
feeds :func:`fdtdx.backend.dispatch.run_forward_from_plans` — no ``ObjectContainer`` and no
re-resolution. Because the same resolved arrays drive the same time loop, the detector states are
bit-identical to an in-process ``run_fdtd`` on the same setup.

``backend="mock"`` skips the engine entirely (see :mod:`fdtdmex.io.mock`) so the agentic workspace
can run end-to-end without a Mac/GPU.

Layout of ``results.hdf5``::

    /                               attrs: schema_version, num_steps, backend
    /detector_states/<name>/<key>   the per-detector recorded arrays
    /config/json                    a copy of the provenance JSON (if present in the config file)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

from ._hdf5 import SCHEMA_VERSION, read_array_store, read_json


def _write_results(path: Path, detector_states: dict, num_steps: int, backend: str, config_json_bytes=None) -> Path:
    import h5py

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["num_steps"] = int(num_steps)
        f.attrs["backend"] = backend
        ds = f.create_group("detector_states")
        for name, bufs in (detector_states or {}).items():
            g = ds.create_group(name)
            for key, arr in bufs.items():
                g.create_dataset(key, data=np.asarray(arr))
        if config_json_bytes is not None:
            cfg = f.create_group("config")
            cfg.create_dataset("json", data=config_json_bytes)
    return path


def sim_run(
    config_path: str | Path,
    results_path: str | Path,
    *,
    backend: Literal["mlx", "mock"] = "mlx",
) -> Path:
    """Run a config HDF5 and write the results HDF5.

    Args:
        config_path: Path to a ``config.hdf5`` produced by :func:`fdtdmex.io.sim_init`.
        results_path: Destination ``results.hdf5`` path.
        backend: ``"mlx"`` (the real engine) or ``"mock"`` (schema-valid synthetic results, no GPU).

    Returns:
        The written results path.
    """
    import h5py

    config_path = Path(config_path)
    results_path = Path(results_path)

    if backend == "mock":
        from .mock import mock_run

        return mock_run(config_path, results_path)

    from fdtdx.backend.dispatch import run_forward_from_plans
    from fdtdx.mlx.serialize import deserialize

    with h5py.File(config_path, "r") as f:
        num_steps = int(f.attrs["num_steps"])
        courant = float(f.attrs["courant"])
        skeleton = read_json(f["payload"], "skeleton")
        array_store = read_array_store(f["payload"]["arrays"])
        config_json_bytes = np.asarray(f["config"]["json"]) if "config" in f else None

    payload = deserialize(skeleton, array_store)
    _, detector_states = run_forward_from_plans(
        payload["state"], payload["source_plans"], payload["detector_plans"], num_steps, courant
    )
    detector_states = {
        name: {k: np.asarray(v) for k, v in bufs.items()} for name, bufs in (detector_states or {}).items()
    }

    out = _write_results(results_path, detector_states, num_steps, "mlx", config_json_bytes)
    logger.info(f"sim_run: wrote results HDF5 → {out} (backend=mlx, {num_steps} steps)")
    return out
