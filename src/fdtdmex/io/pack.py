"""``pack`` / ``sim_init`` — resolve a declarative setup and write a self-contained config HDF5.

This is the *creation utility*: it does the heavy lifting on the authoring machine
(``place_objects`` + ``apply_params`` + freeze), then serializes the **resolved** run seam
(``MLXState`` + frozen ``SourcePlan`` / ``DetectorPlan`` lists — the exact inputs to
``run_forward_from_plans``) into one portable file. The bare-minimum rule holds: only the
resolved arrays + frozen plans ship, never the device ρ / CSG / optimization params upstream of them.

Two entry points share one resolve/write core:

* :func:`pack` — the **agent-facing** form. Materializes into a *project folder* (``location``),
  names the HDF5 by content hash (or ``hdf5_name``), drops a lightweight editable config JSON
  sidecar next to it, and returns a :class:`PackResult` (hdf5 path + config path + content hash).
  The packed HDF5 is reusable: one bundle can back many runs (see ``run_simulation_from_hdf5``).
* :func:`sim_init` — the retained low-level primitive: ``sim_init(setup, path) -> Path`` writes the
  config HDF5 to an explicit *file* path (kept for direct callers / round-trip tests).

Layout of the config HDF5::

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

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from fdtdx.conversion.json import JsonSetup
from fdtdx.mlx.serialize import _MX_DTYPE_TO_NAME, serialize

from ._hdf5 import SCHEMA_VERSION, write_array_store, write_json


@dataclass(frozen=True)
class PackResult:
    """What :func:`pack` returns: the packed HDF5, the config JSON sidecar, and the content hash.

    Implements ``__fspath__`` (returning the HDF5 path), so a ``PackResult`` can be passed straight
    to :func:`fdtdmex.io.run_simulation_from_hdf5` in place of a bare path.
    """

    hdf5_path: Path
    config_path: Path | None
    config_hash: str

    def __fspath__(self) -> str:
        return os.fspath(self.hdf5_path)


@dataclass
class _Resolved:
    """The resolved run seam + provenance — everything :func:`_write_config_hdf5` needs."""

    skeleton: Any
    array_store: dict
    results_spec: list[dict]
    config_json: str | None
    num_steps: int
    courant: float
    grid: Any
    n_sources: int


def _to_json_setup(setup: Any) -> JsonSetup:
    """Normalize a Scene / SceneModel / JsonSetup into a ``JsonSetup``."""
    if isinstance(setup, JsonSetup):
        return setup
    if hasattr(setup, "to_json_setup"):  # Scene, SceneModel
        return setup.to_json_setup()
    raise TypeError(
        f"pack/sim_init expects a Scene, SceneModel, or JsonSetup, got {type(setup).__name__}. "
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


def _resolve(setup: Any) -> _Resolved:
    """Resolve ``setup`` (place + apply + freeze) into the serialized run seam + provenance.

    This is the heavy lifting — host/CPU on the authoring machine — shared by :func:`pack` and
    :func:`sim_init`. Mirrors exactly what ``_run_mlx_forward`` computes before the time loop.
    """
    from fdtdx.backend.dispatch import select_backend  # noqa: F401  (ensures backend module import)
    from fdtdx.fdtd.initialization import apply_params, place_objects
    from fdtdx.fdtd.update import get_wrap_padding_axes
    from fdtdx.mlx.bridge import to_mlx_state
    from fdtdx.mlx.detector_freeze import freeze_detectors
    from fdtdx.mlx.source_freeze import freeze_sources

    js = _to_json_setup(setup)

    objects, arrays, params, config, _ = place_objects(
        object_list=js.object_list, config=js.config, constraints=js.constraints
    )
    arrays, objects, _ = apply_params(arrays, objects, params)

    arrays = arrays.reset()
    periodic_axes = get_wrap_padding_axes(objects)
    state = to_mlx_state(arrays, config, periodic_axes, objects=objects)
    source_plans = freeze_sources(objects, config, arrays)
    detector_plans = freeze_detectors(objects, config)

    payload = {"state": state, "source_plans": list(source_plans), "detector_plans": list(detector_plans)}
    skeleton, array_store = serialize(payload)

    # Best-effort provenance JSON (a setup with un-serializable objects, e.g. devices, still packs).
    try:
        config_json = js.dumps()
    except Exception as exc:  # pragma: no cover - provenance is non-essential
        logger.warning(f"pack: could not serialize the config JSON for provenance ({exc}).")
        config_json = None

    return _Resolved(
        skeleton=skeleton,
        array_store=array_store,
        results_spec=_results_spec(detector_plans),
        config_json=config_json,
        num_steps=int(config.time_steps_total),
        courant=float(config.courant_number),
        grid=config.resolved_grid,
        n_sources=len(source_plans),
    )


def _write_config_hdf5(path: Path, r: _Resolved, *, compression: str | None) -> Path:
    """Write the resolved run seam ``r`` to a self-contained config HDF5 at ``path``."""
    import h5py

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["units"] = "SI"
        f.attrs["dtype"] = "float32"
        f.attrs["axis_order"] = "xyz"
        f.attrs["num_steps"] = r.num_steps
        f.attrs["courant"] = r.courant
        f.attrs["n_sources"] = r.n_sources
        f.attrs["n_detectors"] = len(r.results_spec)

        if r.config_json is not None:
            cfg = f.create_group("config")
            cfg.create_dataset("json", data=np.frombuffer(r.config_json.encode("utf-8"), dtype=np.uint8))

        if r.grid is not None:
            gg = f.create_group("grid")
            gg.create_dataset("x_edges", data=np.asarray(r.grid.x_edges))
            gg.create_dataset("y_edges", data=np.asarray(r.grid.y_edges))
            gg.create_dataset("z_edges", data=np.asarray(r.grid.z_edges))

        meta = f.create_group("meta")
        write_json(meta, "results_spec", r.results_spec)

        p = f.create_group("payload")
        write_json(p, "skeleton", r.skeleton)
        ag = p.create_group("arrays")
        write_array_store(ag, r.array_store, compression=compression)

    return path


def _config_hash(r: _Resolved) -> str:
    """A stable content hash for naming the bundle: the editable config JSON, or the skeleton."""
    if r.config_json is not None:
        raw = r.config_json.encode("utf-8")
    else:  # un-serializable setup: hash the serialized run-seam structure instead
        raw = json.dumps(r.skeleton, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def sim_init(setup: Any, path: str | Path, *, compression: str | None = "gzip") -> Path:
    """Resolve ``setup`` and write the self-contained config HDF5 to an explicit ``path``.

    The retained low-level primitive (direct callers / round-trip tests). For the agent-facing,
    folder-owning, content-addressed form, use :func:`pack`.

    Args:
        setup: A :class:`fdtdx.Scene`, a ``SceneModel``, or a ``JsonSetup``.
        path: Destination ``.hdf5`` *file* path.
        compression: h5py compression for the array store (``"gzip"`` default; ``None`` to disable).

    Returns:
        The written path.
    """
    path = Path(path)
    r = _resolve(setup)
    _write_config_hdf5(path, r, compression=compression)
    logger.info(f"sim_init: wrote config HDF5 → {path} ({r.num_steps} steps, {len(r.results_spec)} detectors)")
    return path


def pack(
    config: Any,
    location: str | Path,
    *,
    hdf5_name: str | None = None,
    compression: str | None = "gzip",
) -> PackResult:
    """Materialize a Scene / SceneModel / JsonSetup into a project folder as a portable bundle.

    Resolves ``config`` (the heavy lifting), then writes a self-contained packed HDF5 plus a
    lightweight editable config JSON sidecar into ``location``. The HDF5 is content-addressed by
    default (named ``<config_hash>.hdf5``), so re-packing the same config is idempotent and one
    packed bundle can back many runs.

    Args:
        config: A :class:`fdtdx.Scene`, a ``SceneModel``, or a ``JsonSetup``.
        location: Destination **folder** (ag-fdtd forces this to the project root). Created if absent.
        hdf5_name: Optional HDF5 filename; defaults to ``<config_hash>.hdf5``.
        compression: h5py compression for the array store (``"gzip"`` default; ``None`` to disable).

    Returns:
        A :class:`PackResult` ``(hdf5_path, config_path, config_hash)``. ``hdf5_path`` is also what
        ``os.fspath(result)`` yields, so the result can be passed straight to
        :func:`fdtdmex.io.run_simulation_from_hdf5`.
    """
    location = Path(location)
    location.mkdir(parents=True, exist_ok=True)

    r = _resolve(config)
    config_hash = _config_hash(r)

    hdf5_path = location / (hdf5_name or f"{config_hash}.hdf5")
    _write_config_hdf5(hdf5_path, r, compression=compression)

    config_path: Path | None = None
    if r.config_json is not None:
        config_path = location / f"{hdf5_path.stem}.json"
        config_path.write_text(r.config_json)

    logger.info(
        f"pack: wrote bundle → {hdf5_path} (hash={config_hash}, {r.num_steps} steps, {len(r.results_spec)} detectors)"
    )
    return PackResult(hdf5_path=hdf5_path, config_path=config_path, config_hash=config_hash)
