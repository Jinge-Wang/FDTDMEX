# Orchestration: Config, MCP Server & Web UI (WS-D)

Goal: an **LLM-orchestratable** workflow — an agent plans, queries available options via tools, writes a declarative setup, runs the simulation in a notebook-like environment, and inspects results — fronted by a locally-hostable **web** UI with a Lumerical-like interactive 3D editor.

## Tidy3D-like architecture

Declarative Python front end → serialized config → compute backend → results. Layers:
- **Config schema** (`fdtdmex/io/`, pydantic): typed, validated models for Volume, Materials, Structures, Sources, Detectors, Boundaries, Grid, Run settings. Round-trips to **JSON** (config) and **HDF5** (large field results).
- **Backend**: the WS-A engine consumes a resolved config and returns results.
- This mirrors Tidy3D (and the direction FDTDX's own PyTorch refactor is taking). FDTDX already proves the pattern with a JSON round-trip (`../fdtdx/src/fdtdx/conversion/json.py`); we use pydantic for schema + validation from the start.

## MCP server (`server/fdtdmex_mcp/`)

Tools exposed to an LLM:
- **introspect** — list available material/source/detector/boundary types and their parameter schemas (derived from the pydantic models).
- **build / edit / validate** — construct or mutate a setup; return validation errors.
- **run** — execute a (resolved) setup; stream progress.
- **results** — fetch fields/flux/phasors and rendered plots.

This maps directly onto "LLM gets options → writes a setup script → calls run → reads results."

## Git-like history (app layer)

Append-only event log of `(prompt, tool_call, result)`. A **revert** forks a new branch from an earlier event; the old branch is archived (hidden), and **fast-forward** replays the recorded tool/prompt sequence. Independent of the solver; lives in the orchestration layer.

## Web UI (`web/`)

Decided: **web** front end (best for agentic workflows), **locally hostable** for local/desktop runs. Rendering options, browser-capable:
- **plotly** — native web 3D; quickest to stand up.
- **pyvista via trame** — server-side VTK streamed as VTK.js; keeps the Python scene model, so it *does* run in the browser. Good for a richer scene/editor.
- **three.js / react-three-fiber** — most control, most effort; reserve for a polished editor.

Target interactions (Lumerical-like): click objects to select; tabbed panels to switch boundary conditions, materials, source/detector properties; toggle field overlays. Start with plotly/pyvista-trame to reuse Python geometry; layer three.js later if needed.

## Sequencing
MCP/config layer can start immediately against the pydantic schema (independent of the GPU engine for schema work). The 3D editor is the largest, most open-ended piece. See [roadmap.md](roadmap.md).
