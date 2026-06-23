"""WS-D — FDTDMEX MCP discovery server for LLM-orchestrated simulation setup.

A small, fixed 4-tool discovery surface — ``list_solver_apis`` / ``get_api_schema`` /
``search_docs`` / ``get_doc`` — that lets an agent discover the ``run_fdtd_fdtdmex`` run
API (introspected live from the adapter + ``fdtdmex.io`` schema) and a BM25-indexed corpus
generated from the repo's real ``examples/``, ``docs/``, and docstrings. It speaks the same
contract as ag-fdtd's mock so the agent can't tell them apart. Runs over stdio
(``fdtdmex-mcp`` / ``python -m fdtdmex_mcp``). See docs/mcp-and-ui.md; requires the ``mcp`` extra.

Discovery only — simulations run through ``agentic_adapter/real_solver.py``, not this server.
"""

from .server import mcp

__all__ = ["mcp"]
