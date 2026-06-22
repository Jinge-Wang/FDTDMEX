"""Small HDF5 helpers shared by the sim_init / sim_run / sim_postproc utilities.

JSON blobs (the editable config, the serialization skeleton, the results spec) are stored as
UTF-8 byte datasets to dodge h5py's variable-length-string quirks and the 64 KB attribute cap.
Array stores are written as (optionally gzip-compressed) datasets.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

SCHEMA_VERSION = "fdtdmex.sim.v1"


def write_json(group, name: str, obj: Any) -> None:
    """Store a JSON-serializable object as a UTF-8 byte dataset under ``group/name``."""
    data = np.frombuffer(json.dumps(obj).encode("utf-8"), dtype=np.uint8)
    group.create_dataset(name, data=data)


def read_json(group, name: str) -> Any:
    """Inverse of :func:`write_json`."""
    raw = bytes(np.asarray(group[name]).tobytes())
    return json.loads(raw.decode("utf-8"))


def write_array_store(group, arrays: dict[str, np.ndarray], compression: str | None = "gzip") -> None:
    """Write a ``{id: ndarray}`` store as datasets. Compression is skipped for empty arrays."""
    for key, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        if compression is not None and arr.size > 0 and arr.ndim > 0:
            group.create_dataset(key, data=arr, compression=compression)
        else:
            group.create_dataset(key, data=arr)


def read_array_store(group) -> dict[str, np.ndarray]:
    """Read every dataset directly under ``group`` into a ``{name: ndarray}`` store."""
    return {key: np.asarray(group[key]) for key in group.keys()}
