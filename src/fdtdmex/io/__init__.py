"""FDTDMEX IO layer — the agentic hand-off seam.

The portable contract between the declarative front end and any compute node:

* :func:`sim_init` ``(setup) → config.hdf5`` — resolve + pack on the authoring machine.
* :func:`sim_run` ``(config.hdf5) → results.hdf5`` — run on any FDTDMEX machine (``backend="mlx"``)
  or fabricate schema-valid results without a GPU (``backend="mock"``).
* :func:`sim_postproc` ``(results.hdf5) → small results`` — reduce to scalars the agent reads.

Plus :class:`SceneModel`, the pydantic facade over fdtdx's ``JsonSetup`` (the small editable config).

Requires the ``io`` extra (``h5py`` + ``pydantic``). See ``docs/mcp-and-ui.md``.
"""

from __future__ import annotations

from .pack import sim_init
from .postproc import sim_postproc
from .run import sim_run
from .schema import SceneModel

__all__ = ["SceneModel", "sim_init", "sim_postproc", "sim_run"]
