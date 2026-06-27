"""WS-D — FDTDMEX MCP discovery server for LLM-orchestrated simulation setup.

A small, fixed 4-tool discovery surface — ``list_solver_apis`` / ``get_api_schema`` /
``search_docs`` / ``get_doc`` — that lets an agent discover the native run API
(``pack`` / ``run_simulation_from_hdf5`` / ``sim_postproc`` / ``compute_mode``, introspected live
from ``fdtdmex.io`` + the ``SceneModel`` schema) and a BM25-indexed corpus generated from the repo's
real ``examples/``, ``docs/``, and docstrings. It speaks the same contract as ag-fdtd's mock so the
agent can't tell them apart. Runs over stdio (``fdtdmex-mcp`` / ``python -m fdtdmex_mcp``) — launched
by ag-fdtd's UI, or standalone in any MCP host (see server/README.md). Requires the ``mcp`` extra.

Discovery only — the agent runs simulations natively in its own kernel (``pack`` → non-blocking
``run_simulation_from_hdf5``), never through this server.
"""

from .server import mcp

__all__ = ["mcp"]
