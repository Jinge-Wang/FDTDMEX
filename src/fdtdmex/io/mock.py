"""Mock backend — fabricate a schema-valid ``results.hdf5`` from a config HDF5 without MLX.

The agentic workspace (pydantic-ai + AG-UI + FastAPI) develops against this so it can exercise the
whole ``sim_init → sim_run → sim_postproc`` contract with no Mac/GPU. It reads only the small
``/meta/results_spec`` block (per-detector buffer shapes + dtype names) written by
:func:`fdtdmex.io.sim_init`, so it never deserializes the MLX arrays and never imports ``mlx``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._hdf5 import SCHEMA_VERSION, read_json

_NP_DTYPE = {
    "float32": np.float32,
    "complex64": np.complex64,
    "float16": np.float16,
    "bool": np.bool_,
    "int32": np.int32,
}


def _fabricate(shape, dtype_name: str, rng: np.random.Generator) -> np.ndarray:
    dt = _NP_DTYPE.get(dtype_name, np.float32)
    if np.issubdtype(dt, np.complexfloating):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dt)
    if dt == np.bool_:
        return rng.integers(0, 2, size=shape).astype(dt)
    return rng.standard_normal(shape).astype(dt)


def mock_run(config_path: str | Path, results_path: str | Path, *, seed: int = 0) -> Path:
    """Write a synthetic ``results.hdf5`` matching the detector spec in ``config_path``."""
    import h5py

    config_path = Path(config_path)
    results_path = Path(results_path)
    rng = np.random.default_rng(seed)

    with h5py.File(config_path, "r") as f:
        num_steps = int(f.attrs["num_steps"])
        spec = read_json(f["meta"], "results_spec")
        config_json_bytes = np.asarray(f["config"]["json"]) if "config" in f else None

    with h5py.File(results_path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["num_steps"] = num_steps
        f.attrs["backend"] = "mock"
        ds = f.create_group("detector_states")
        for d in spec:
            g = ds.create_group(d["name"])
            for key, shape in d["shapes"].items():
                g.create_dataset(key, data=_fabricate(tuple(shape), d["dtypes"][key], rng))
        if config_json_bytes is not None:
            cfg = f.create_group("config")
            cfg.create_dataset("json", data=config_json_bytes)

    return results_path
