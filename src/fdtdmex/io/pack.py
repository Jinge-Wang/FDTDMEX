"""``sim_init`` — resolve a declarative setup and pack it into a self-contained config HDF5.

This is the *creation utility*: it does the heavy lifting on the authoring machine
(``place_objects`` + ``apply_params`` + freeze), then serializes the **resolved** run seam
(``MLXState`` + frozen ``SourcePlan`` / ``DetectorPlan`` lists — the exact inputs to
``run_forward_from_plans``) into one portable file. The bare-minimum rule holds: only the
resolved arrays + frozen plans ship, never the device ρ / CSG / optimization params upstream of them.

Layout of ``config.hdf5``::

    /                         attrs: schema_version, units, dtype, axis_order, num_steps, courant,
                                     n_sources, n_detectors
    /config/json              the editable JsonSetup JSON (provenance + round-trip), if serializable
    /grid/{x,y,z}_edges       resolved grid edges in metres (provenance / plotting / postproc)
    /meta/results_spec        per-detector buffer shapes+dtypes (lets the mock backend fabricate
                              schema-valid results without MLX)
    /payload/skeleton         JSON structure of the serialized run seam
    /payload/arrays/<id>      the array leaves (gzip-compressed)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from fdtdx.conversion.json import JsonSetup
from fdtdx.mlx.serialize import _MX_DTYPE_TO_NAME, serialize

from ._hdf5 import SCHEMA_VERSION, write_array_store, write_json


def _to_json_setup(setup: Any) -> JsonSetup:
    """Normalize a Scene / SceneModel / JsonSetup into a ``JsonSetup``."""
    if isinstance(setup, JsonSetup):
        return setup
    if hasattr(setup, "to_json_setup"):  # Scene, SceneModel
        return setup.to_json_setup()
    raise TypeError(
        f"sim_init expects a Scene, SceneModel, or JsonSetup, got {type(setup).__name__}. "
        "Build one of those (e.g. fdtdx.Scene(config).add(...))."
    )


def _results_spec(detector_plans) -> list[dict]:
    """Per-detector buffer shapes + dtype names, so the mock backend needs no MLX."""
    spec = []
    for p in detector_plans:
        spec.append(
            {
                "name": p.name,
                "kind": p.kind,
                "buffer_key": p.buffer_key,
                "shapes": {k: list(v) for k, v in p.buffer_shapes.items()},
                "dtypes": {k: _MX_DTYPE_TO_NAME[v] for k, v in p.buffer_dtypes.items()},
            }
        )
    return spec


def sim_init(setup: Any, path: str | Path, *, compression: str | None = "gzip") -> Path:
    """Resolve ``setup`` and write the self-contained config HDF5 to ``path``.

    Args:
        setup: A :class:`fdtdx.Scene`, a ``SceneModel``, or a ``JsonSetup``.
        path: Destination ``.hdf5`` path.
        compression: h5py compression for the array store (``"gzip"`` default; ``None`` to disable).

    Returns:
        The written path.
    """
    import h5py

    from fdtdx.backend.dispatch import select_backend  # noqa: F401  (ensures backend module import)
    from fdtdx.fdtd.initialization import apply_params, place_objects
    from fdtdx.fdtd.update import get_wrap_padding_axes
    from fdtdx.mlx.bridge import to_mlx_state
    from fdtdx.mlx.detector_freeze import freeze_detectors
    from fdtdx.mlx.source_freeze import freeze_sources

    js = _to_json_setup(setup)
    path = Path(path)

    # Resolve: place + apply (host/CPU on the authoring machine).
    objects, arrays, params, config, _ = place_objects(
        object_list=js.object_list, config=js.config, constraints=js.constraints
    )
    arrays, objects, _ = apply_params(arrays, objects, params)

    # The resolved run seam — exactly what _run_mlx_forward computes before the time loop.
    arrays = arrays.reset()
    periodic_axes = get_wrap_padding_axes(objects)
    state = to_mlx_state(arrays, config, periodic_axes, objects=objects)
    source_plans = freeze_sources(objects, config, arrays)
    detector_plans = freeze_detectors(objects, config)
    num_steps = int(config.time_steps_total)
    courant = float(config.courant_number)

    payload = {"state": state, "source_plans": list(source_plans), "detector_plans": list(detector_plans)}
    skeleton, array_store = serialize(payload)

    # Best-effort provenance JSON (a setup with un-serializable objects, e.g. devices, still packs).
    try:
        config_json = js.dumps()
    except Exception as exc:  # pragma: no cover - provenance is non-essential
        logger.warning(f"sim_init: could not serialize the config JSON for provenance ({exc}).")
        config_json = None

    grid = config.resolved_grid

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["units"] = "SI"
        f.attrs["dtype"] = "float32"
        f.attrs["axis_order"] = "xyz"
        f.attrs["num_steps"] = num_steps
        f.attrs["courant"] = courant
        f.attrs["n_sources"] = len(source_plans)
        f.attrs["n_detectors"] = len(detector_plans)

        if config_json is not None:
            cfg = f.create_group("config")
            cfg.create_dataset("json", data=np.frombuffer(config_json.encode("utf-8"), dtype=np.uint8))

        if grid is not None:
            gg = f.create_group("grid")
            gg.create_dataset("x_edges", data=np.asarray(grid.x_edges))
            gg.create_dataset("y_edges", data=np.asarray(grid.y_edges))
            gg.create_dataset("z_edges", data=np.asarray(grid.z_edges))

        meta = f.create_group("meta")
        write_json(meta, "results_spec", _results_spec(detector_plans))

        p = f.create_group("payload")
        write_json(p, "skeleton", skeleton)
        ag = p.create_group("arrays")
        write_array_store(ag, array_store, compression=compression)

    logger.info(f"sim_init: wrote config HDF5 → {path} ({num_steps} steps, {len(detector_plans)} detectors)")
    return path
