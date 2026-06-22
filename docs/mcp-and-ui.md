# Orchestration: Config, MCP Server & Web UI (WS-D)

Goal: an **LLM-orchestratable** workflow — an agent plans, queries available options via tools, writes a declarative setup, runs the simulation in a notebook-like environment, and inspects results — fronted by a locally-hostable **web** UI with a Lumerical-like interactive 3D editor.

## Roles & the data boundary (who touches what)

Three layers, with a hard rule: **the LLM/agent never reads or writes large arrays** (ε/material maps,
field dumps). It works only at the *script + config* level and reads back only small results
(scalars, fluxes, n_eff, rendered thumbnails).

The public API is a matched trio: **`sim_init` → `sim_run` → `sim_postproc`**.

- **Agentic layer (LLM + MCP).** The LLM writes a short Python script that calls **`sim_init`** with
  high-level parameters (geometry, materials, sources/detectors *by description*), triggers
  **`sim_run`**, and inspects via **`sim_postproc`**. It discovers the right function/class names + IO
  via the MCP server. It does **not** hand-author arrays and does **not** ingest large outputs.
- **fdtdmex front end — `sim_init` (the simulation creation utility).** A first-class fdtdmex API
  (`fdtdmex/io/`) that takes the declarative setup and **does the heavy lifting on the front-end
  machine**: resolve objects + design parameters → rasterize geometry → assemble the ε/µ/σ +
  dispersive distributions, freeze source/detector profiles, build the grid/boundary spec → **write a
  self-contained config HDF5**. This is where resolution lives (by design — not the agent, not the
  backend).
- **fdtdmex backend — `sim_run`.** Any machine with fdtdmex — **local or remote** — takes the config
  HDF5, unwraps it, runs the time loop, and writes a results HDF5. The config HDF5 is the portable
  artifact that ships between machines.
- **`sim_postproc`.** Reduces a results HDF5 to the **small** quantities the agent/user reads —
  scalars, fluxes, n_eff, S-parameters, rendered thumbnails — so large field data never flows through
  the LLM.

## Tidy3D-like architecture

Declarative Python front end → serialized config → compute backend → results. Layers:
- **Config schema** (`fdtdmex/io/`, pydantic): typed, validated models for Volume, Materials, Structures, Sources, Detectors, Boundaries, Grid, Run settings. Round-trips to **JSON** (config) and **HDF5** (large field results).
- **Backend**: the WS-A engine consumes a resolved config and returns results.
- This mirrors Tidy3D (and the direction FDTDX's own PyTorch refactor is taking). FDTDX already proves the pattern with a JSON round-trip (`../fdtdx/src/fdtdx/conversion/json.py`); we use pydantic for schema + validation from the start.

## HDF5 simulation payload — the wrap/unwrap contract

The hand-off between front end and backend is **one self-contained config HDF5 file** (the Tidy3D
`.hdf5` model). **`sim_init(setup) → config.hdf5`** on the front end; **`sim_run(config.hdf5) →
results.hdf5`** on any fdtdmex machine; **`sim_postproc(results.hdf5) → small results`**. This is the
single contract the agentic workspace and the solver agree on, so the workspace can be developed
against a **mocked backend** and bridged later.

**Pipeline — the creation utility resolves *before* packing:**
1. **Author** — the LLM script (or UI) describes the high-level declarative objects (Volume,
   Structures, Sources, Detectors, Boundaries, Grid, Run) — plus any *design* parameters (device
   density ρ, latent/optimization variables). The LLM passes these as parameters; it does not build
   arrays.
2. **Resolve + pack** — the **creation utility** (`place_objects` + `apply_params`, host/CPU on the
   front-end machine) compiles the object graph **down to the arrays the time loop actually consumes**
   — the ε/µ/σ + dispersive-coefficient distributions, frozen source profiles (TFSF/mode `_E`/`_H` +
   Yee time offsets), detector specs, boundary geometry (PML extents / periodic / PEC-PMC masks), grid
   spacings, and run settings — and packs that **resolved** payload into the config HDF5.

**Bare-minimum rule:** the HDF5 ships **only what the simulation needs** — the *resolved* material
distributions (final ε etc.), not the pre-simulation data that produces them (no device ρ, no
optimization parameters, no object CSG that gets rasterized into ε). The resolved `ArrayContainer` is
the canonical payload; everything upstream of it stays on the authoring side. (Rationale: the file is
the minimal, portable, reproducible thing a compute node runs — smaller, no design-tooling
dependency on the backend, and a clean trust boundary.)

**Layout:**
- a `config` group — the JSON setup (run-level: grid, time/Courant, boundary spec, source/detector
  descriptors) for provenance + round-trip;
- `arrays` datasets — the large resolved fields (ε/µ/σ, dispersive c1/c2/c3, frozen source profiles,
  detector init), chunked/compressed;
- a `meta` attr block — schema version, units, dtype, axis conventions.

`wrap`/`unwrap` live in `fdtdmex/io/`; `unwrap` feeds straight into the existing
[`to_mlx_state`](../src/fdtdx/mlx/bridge.py)/`freeze_sources`/`freeze_detectors` boundary (the MLX
bridge is already a "resolved arrays → run" seam, so the HDF5 is essentially its serialized form).
Align dataset/field names with fdtdx's emerging JSON round-trip (`../fdtdx/src/fdtdx/conversion/json.py`)
where practical, so fdtdx setups can be ingested.

## MCP server (`server/fdtdmex_mcp/`)

**Interaction model:** the LLM **writes and runs Python scripts** that call the fdtdmex creation
utility; the MCP server is how it *discovers the API* — it fetches the correct function/class names
and their inputs/outputs (from the pydantic/`autoinit` models) so the generated script is valid. The
script calls `sim_init(...)` → config HDF5, dispatches `sim_run` on a target machine (local/remote),
and reads back only the **small** `sim_postproc` outputs (scalars, fluxes, n_eff, thumbnails) — never
the large ε maps or field dumps. ("LLM gets options → `sim_init` → `sim_run` → `sim_postproc`.")

Tools exposed to an LLM:
- **introspect** — list available material/source/detector/boundary types and their parameter schemas
  (names, types, defaults, docstrings) derived from the pydantic models — i.e. the exact signatures the
  LLM needs to write a correct script.
- **build / edit / validate** — construct or mutate a setup; return validation errors.
- **sim_init** — resolve a setup → config HDF5.
- **sim_run** — execute a config HDF5 on a target machine; stream progress.
- **sim_postproc** — fetch the small reduced results (scalars/flux/n_eff/thumbnails), not raw fields.

This maps directly onto "LLM gets options → writes a script → `sim_init` → `sim_run` → `sim_postproc`."

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
