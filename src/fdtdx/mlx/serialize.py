"""Serialize the resolved MLX run seam (``MLXState`` + frozen source/detector plans) to
plain numpy + a JSON-safe skeleton, and back.

This is the serialized form of the ``to_mlx_state`` / ``freeze_sources`` / ``freeze_detectors``
boundary (see :mod:`fdtdx.mlx.bridge`) that the config HDF5 stores. It walks the dataclasses and
their nested containers (tuples/lists/dicts), MLX arrays/dtypes, numpy arrays, and ``slice`` objects
into:

* a flat ``dict[str, np.ndarray]`` of every array leaf (keyed by a running integer id), and
* a JSON-able **skeleton** that describes the structure and references each array leaf by id.

:func:`deserialize` reconstructs the original object graph from ``(skeleton, arrays)``. The split
keeps the large arrays separate so the IO layer can store them as chunked/compressed HDF5 datasets
while the skeleton stays a tiny JSON blob. Round-trips are exact (numpy preserves the fp32/complex64
bit patterns), so a packed-then-unpacked run is bit-identical to the in-process MLX path.
"""

from __future__ import annotations

import importlib
from dataclasses import fields, is_dataclass
from typing import Any

import mlx.core as mx
import numpy as np

# MLX dtype <-> portable name. Covers the dtypes that reach the run seam (detector buffer dtypes are
# float32 / complex64; the rest are field/material arrays carried as mx.array leaves, not mx.Dtype).
_MX_DTYPE_TO_NAME = {
    mx.bool_: "bool",
    mx.int32: "int32",
    mx.uint32: "uint32",
    mx.float16: "float16",
    mx.float32: "float32",
    mx.complex64: "complex64",
}
_NAME_TO_MX_DTYPE = {v: k for k, v in _MX_DTYPE_TO_NAME.items()}

# Classes the skeleton is allowed to reconstruct (defense against importing arbitrary modules).
_ALLOWED_DATACLASSES = {
    "fdtdx.mlx.state.MLXState",
    "fdtdx.mlx.source_freeze.SourcePlan",
    "fdtdx.mlx.detector_freeze.DetectorPlan",
}


def serialize(obj: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Flatten ``obj`` into ``(skeleton, arrays)``.

    ``skeleton`` is JSON-serializable; ``arrays`` maps string ids to numpy arrays.
    """
    arrays: dict[str, np.ndarray] = {}
    counter = [0]

    def _array_leaf(a: np.ndarray, tag: str) -> dict:
        i = counter[0]
        counter[0] += 1
        arrays[str(i)] = np.ascontiguousarray(a)
        return {"t": tag, "i": str(i)}

    def enc(o: Any) -> dict:
        if o is None:
            return {"t": "none"}
        if isinstance(o, bool):
            return {"t": "bool", "v": o}
        if isinstance(o, (int, np.integer)):
            return {"t": "int", "v": int(o)}
        if isinstance(o, (float, np.floating)):
            return {"t": "float", "v": float(o)}
        if isinstance(o, str):
            return {"t": "str", "v": o}
        if isinstance(o, slice):
            return {"t": "slice", "v": [o.start, o.stop, o.step]}
        if isinstance(o, mx.Dtype):
            name = _MX_DTYPE_TO_NAME.get(o)
            if name is None:  # pragma: no cover - guarded by supported buffer dtypes
                raise NotImplementedError(f"Cannot serialize MLX dtype {o!r}")
            return {"t": "mxdtype", "v": name}
        if isinstance(o, mx.array):
            return _array_leaf(np.array(o), "mxarray")
        if isinstance(o, np.ndarray):
            return _array_leaf(o, "ndarray")
        if isinstance(o, dict):
            return {"t": "dict", "v": {str(k): enc(v) for k, v in o.items()}}
        if isinstance(o, tuple):
            return {"t": "tuple", "v": [enc(x) for x in o]}
        if isinstance(o, list):
            return {"t": "list", "v": [enc(x) for x in o]}
        if is_dataclass(o) and not isinstance(o, type):
            qual = f"{type(o).__module__}.{type(o).__name__}"
            if qual not in _ALLOWED_DATACLASSES:
                raise NotImplementedError(f"Cannot serialize dataclass {qual}")
            return {
                "t": "dataclass",
                "cls": qual,
                "v": {f.name: enc(getattr(o, f.name)) for f in fields(o)},
            }
        raise NotImplementedError(f"Cannot serialize object of type {type(o)!r}")

    return enc(obj), arrays


def deserialize(skeleton: dict[str, Any], arrays: dict[str, np.ndarray]) -> Any:
    """Inverse of :func:`serialize`. Reconstruct the object graph from ``(skeleton, arrays)``."""

    def dec(node: dict) -> Any:
        t = node["t"]
        if t == "none":
            return None
        if t in ("bool", "int", "float", "str"):
            return node["v"]
        if t == "slice":
            return slice(*node["v"])
        if t == "mxdtype":
            return _NAME_TO_MX_DTYPE[node["v"]]
        if t == "mxarray":
            return mx.array(np.ascontiguousarray(arrays[node["i"]]))
        if t == "ndarray":
            return np.ascontiguousarray(arrays[node["i"]])
        if t == "dict":
            return {k: dec(v) for k, v in node["v"].items()}
        if t == "tuple":
            return tuple(dec(x) for x in node["v"])
        if t == "list":
            return [dec(x) for x in node["v"]]
        if t == "dataclass":
            qual = node["cls"]
            if qual not in _ALLOWED_DATACLASSES:
                raise NotImplementedError(f"Cannot deserialize dataclass {qual}")
            mod_name, cls_name = qual.rsplit(".", 1)
            cls = getattr(importlib.import_module(mod_name), cls_name)
            return cls(**{k: dec(v) for k, v in node["v"].items()})
        raise NotImplementedError(f"Unknown skeleton tag {t!r}")

    return dec(skeleton)
