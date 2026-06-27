"""FDTDMEX IO layer — the agentic hand-off seam.

The portable contract between the declarative front end and any compute node. The agent-facing flow
is **assemble → pack → launch → postproc**:

* :func:`pack` ``(config, location) → PackResult`` — resolve + pack into a project folder on the
  authoring machine (content-addressed HDF5 + lightweight config JSON sidecar). One bundle, many runs.
* :func:`run_simulation_from_hdf5` ``(hdf5, parent_folder) → JobHandle`` — stage a job folder and
  launch the solver **detached**, returning immediately while ``status.json`` advances. Non-blocking.
* :func:`sim_postproc` ``(results.hdf5) → small results`` — reduce to scalars the agent reads.

Lower-level primitives (not agent-facing; used internally / by the detached child / tests):

* :func:`sim_init` ``(setup, path) → config.hdf5`` — write a config HDF5 to an explicit file path.
* :func:`run_simulation` ``(config_or_hdf5) → results.hdf5`` — the **blocking** worker; runs in the
  cwd, writing ``status.json`` / ``progress.jsonl`` / results there (this is what the child executes).
* :func:`sim_run` ``(config.hdf5, results.hdf5) → results.hdf5`` — the bare engine executor
  (``backend="mlx"``) or a GPU-free schema-valid fabrication (``backend="mock"``).

Plus :class:`SceneModel`, the pydantic facade over fdtdx's ``JsonSetup`` (the small editable config),
and :class:`StatusWriter` (the atomic ``status.json`` driver).

Requires the ``io`` extra (``h5py`` + ``pydantic``). See ``docs/mcp-and-ui.md``.
"""

from __future__ import annotations

from .launch import JobHandle, run_simulation_from_hdf5
from .pack import PackResult, pack, sim_init
from .postproc import sim_postproc
from .run import sim_run
from .schema import SceneModel
from .status import StatusWriter, run_simulation

__all__ = [
    "JobHandle",
    "PackResult",
    "SceneModel",
    "StatusWriter",
    "pack",
    "run_simulation",
    "run_simulation_from_hdf5",
    "sim_init",
    "sim_postproc",
    "sim_run",
]
