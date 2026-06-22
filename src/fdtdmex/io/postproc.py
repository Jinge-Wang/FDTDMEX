"""``sim_postproc`` — reduce a results HDF5 to the small quantities an agent/user reads.

Returns a JSON-serializable dict of per-detector reductions (shapes, dtype, magnitude/energy
summaries, and a short preview of small arrays) — never the raw field dumps. This is the only thing
that flows back to the LLM in the agentic workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._hdf5 import read_array_store

_PREVIEW_MAX = 16


def _reduce_array(arr: np.ndarray) -> dict[str, Any]:
    """Summarize one recorded array into small scalars (+ a short preview when tiny)."""
    out: dict[str, Any] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if arr.size == 0:
        return out
    mag = np.abs(arr)
    out["max_abs"] = float(mag.max())
    out["mean_abs"] = float(mag.mean())
    if np.iscomplexobj(arr):
        out["sum_abs"] = float(mag.sum())
    else:
        out["sum"] = float(np.asarray(arr).sum())
    if arr.size <= _PREVIEW_MAX:
        flat = arr.ravel()
        out["preview"] = [complex(x).__repr__() if np.iscomplexobj(arr) else float(x) for x in flat]
    return out


def sim_postproc(results_path: str | Path) -> dict[str, Any]:
    """Reduce ``results.hdf5`` to a small JSON-serializable result dict.

    Args:
        results_path: Path to a ``results.hdf5`` from :func:`fdtdmex.io.sim_run`.

    Returns:
        ``{"num_steps": int, "backend": str, "detectors": {name: {key: {summary...}}}}``.
    """
    import h5py

    results_path = Path(results_path)
    result: dict[str, Any] = {"detectors": {}}

    with h5py.File(results_path, "r") as f:
        result["num_steps"] = int(f.attrs.get("num_steps", -1))
        result["backend"] = str(f.attrs.get("backend", "unknown"))
        if "detector_states" in f:
            for name in f["detector_states"].keys():
                bufs = read_array_store(f["detector_states"][name])
                result["detectors"][name] = {key: _reduce_array(arr) for key, arr in bufs.items()}

    return result
