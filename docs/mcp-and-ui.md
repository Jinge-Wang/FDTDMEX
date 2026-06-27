# Orchestration: Config, MCP Server & Web UI

Goal: an **LLM-orchestratable** workflow — an agent plans, queries available options via tools, writes a declarative setup, runs the simulation in a notebook-like environment, and inspects results — fronted by a locally-hostable **web** UI with a Lumerical-like interactive 3D editor.

## Roles & the data boundary (who touches what)

Three layers, with a hard rule: **the LLM/agent never reads or writes large arrays** (ε/material maps, field dumps). It works only at the *script + config* level and reads back only small results (scalars, fluxes, n_eff, rendered thumbnails).

The public API is a matched trio: **`pack` → `run_simulation_from_hdf5` → `sim_postproc`**.

- **Agentic layer (LLM + MCP).** The LLM writes a short Python script that calls **`pack`** with a Scene assembled from high-level parameters (geometry, materials, sources/detectors *by description*), launches it **non-blocking** via **`run_simulation_from_hdf5`**, and (when the job finishes) inspects via **`sim_postproc`**. It discovers the right function/class names + IO via the MCP server. It does **not** hand-author arrays and does **not** ingest large outputs.
- **fdtdmex front end — `pack` (the simulation creation utility).** A first-class fdtdmex API (`fdtdmex/io/`) that takes the declarative setup and **does the heavy lifting on the front-end machine**: resolve objects + design parameters → rasterize geometry → assemble the ε/µ/σ + dispersive distributions, freeze source/detector profiles, build the grid/boundary spec → **write a self-contained config HDF5** (content-addressed) into a project folder, plus a lightweight editable config JSON. This is where resolution lives (by design — not the agent, not the backend). (`sim_init` is the same packer aimed at an explicit file path — the retained low-level primitive.)
- **fdtdmex launcher — `run_simulation_from_hdf5`.** **Non-blocking.** Stages a job folder under `parent_folder`, copies the bundle in, writes `status.json`, and **launches the solver detached** — returning immediately with a `JobHandle`. The detached child (running the bare `run_simulation` worker on any fdtdmex machine, local or remote) unwraps the config HDF5, runs the time loop, and writes the results HDF5 into the job folder's `outputs/`. The config HDF5 is the portable artifact that ships between machines.
- **`sim_postproc`.** Reduces a results HDF5 to the **small** quantities the agent/user reads — scalars, fluxes, n_eff, S-parameters, rendered thumbnails — so large field data never flows through the LLM.

## Tidy3D-like architecture

Declarative Python front end → serialized config → compute backend → results. Layers:
- **Config schema** (`fdtdmex/io/`, pydantic): typed, validated models for Volume, Materials, Structures, Sources, Detectors, Boundaries, Grid, Run settings. Round-trips to **JSON** (config) and **HDF5** (large field results).
- **Backend**: the forward engine consumes a resolved config and returns results.
- This mirrors Tidy3D's declarative design; we use pydantic for schema and validation from the start.

## HDF5 simulation payload — the wrap/unwrap contract

The hand-off between front end and backend is **one self-contained config HDF5 file** (the Tidy3D `.hdf5` model). **`pack(setup, location) → bundle.hdf5`** on the front end; **`run_simulation_from_hdf5(bundle, parent_folder) → JobHandle`** launches it detached on any fdtdmex machine (the child writes `outputs/result.hdf5`); **`sim_postproc(results.hdf5) → small results`**. This is the single contract the agentic workspace and the solver agree on, so the workspace can be developed against a **mocked backend** and bridged later.

**Pipeline — the creation utility resolves *before* packing:**
1. **Author** — the LLM script (or UI) describes the high-level declarative objects (Volume, Structures, Sources, Detectors, Boundaries, Grid, Run) — plus any *design* parameters (device density ρ, latent/optimization variables). The LLM passes these as parameters; it does not build arrays.
2. **Resolve + pack** — the **creation utility** (`place_objects` + `apply_params`, host/CPU on the front-end machine) compiles the object graph **down to the arrays the time loop actually consumes** — the ε/µ/σ + dispersive-coefficient distributions, frozen source profiles (TFSF/mode `_E`/`_H` + Yee time offsets), detector specs, boundary geometry (PML extents / periodic / PEC-PMC masks), grid spacings, and run settings — and packs that **resolved** payload into the config HDF5.

**Bare-minimum rule:** the HDF5 ships **only what the simulation needs** — the *resolved* material distributions (final ε etc.), not the pre-simulation data that produces them (no device ρ, no optimization parameters, no object CSG that gets rasterized into ε). The resolved `ArrayContainer` is the canonical payload; everything upstream of it stays on the authoring side. (Rationale: the file is the minimal, portable, reproducible thing a compute node runs — smaller, no design-tooling dependency on the backend, and a clean trust boundary.)

**Layout:**
- a `config` group — the JSON setup (run-level: grid, time/Courant, boundary spec, source/detector descriptors) for provenance + round-trip;
- `arrays` datasets — the large resolved fields (ε/µ/σ, dispersive c1/c2/c3, frozen source profiles, detector init), chunked/compressed;
- a `meta` attr block — schema version, units, dtype, axis conventions.

`wrap`/`unwrap` live in `fdtdmex/io/`; `unwrap` feeds straight into the existing [`to_mlx_state`](../src/fdtdx/mlx/bridge.py)/`freeze_sources`/`freeze_detectors` boundary (the MLX bridge is already a "resolved arrays → run" seam, so the HDF5 is essentially its serialized form). Align dataset/field names with fdtdx's JSON round-trip where practical, so fdtdx setups can be ingested.

## MCP server (`server/fdtdmex_mcp/`)

**Interaction model:** the LLM **writes and runs native fdtdmex Python** (`Scene` / `pack` / `run_simulation_from_hdf5` / `sim_postproc` / `compute_mode`) in its own notebook kernel — the kernel runs in *this* repo's venv. The MCP server is **discovery-only**: it teaches the agent *what to write* (the correct function/class names + signatures, and verified examples) so the generated script is valid. It does **not** run simulations — execution is native in the agent's kernel, and the launch itself is **non-blocking** (`run_simulation_from_hdf5` detaches the solver), so no run ever blocks the kernel through this server. The script calls `pack(...)` → bundle HDF5, launches `run_simulation_from_hdf5(...)` (which writes `status.json`, below), and once the job completes reads back only the **small** `sim_postproc` outputs (scalars, fluxes, n_eff, thumbnails) — never the large ε maps or field dumps.

**The 4-tool discovery contract (as built, `server/fdtdmex_mcp/`).** The agent's tool surface stays small and fixed — four read-only tools — no matter how many docs/examples exist:
- **`list_solver_apis`** — list the native pack/launch/post-proc entry points (`pack`, `run_simulation_from_hdf5`, `sim_postproc`, `compute_mode`) with one-line summaries. (The bare blocking worker `run_simulation` is deliberately *not* advertised — the agent never calls it directly.)
- **`get_api_schema(name)`** — the full signature of one entry point, introspected **live** (`inspect.signature` of the real function, so it can't drift) + the `SceneModel` payload fields.
- **`search_docs(query)`** — BM25 search over a corpus built from real on-disk sources (docs + runnable `examples/`); returns ranked refs + snippets.
- **`get_doc(ref)`** — fetch one doc/example page in full (a copyable verified setup, or guide prose).

This maps onto "LLM discovers the native API + a verified example → writes a script → `pack` → `run_simulation_from_hdf5` → `sim_postproc`."

### Run telemetry — the `status.json` contract

A run is tracked uniformly through a per-job **`status.json`** file rather than parsed stdout. In the v2 model **fdtdmex owns the job folder**: `fdtdmex.io.run_simulation_from_hdf5(hdf5, parent_folder, *, simulation_name=None, backend, name)` stages `job_dir = <parent_folder>/<simulation_name or unique-job-id>/` (copying the bundle in, snapshotting a lightweight `config.json`, creating `outputs/`), writes the initial `queued` `status.json`, and **launches the solver detached** — returning immediately with a `JobHandle`. The detached child (the bare `run_simulation` worker, cwd = `job_dir`) drives `status.json` + an append-only `progress.jsonl` off the existing `progress(step, total)` callback and writes the results to `outputs/result.hdf5`. **A job folder with no `status.json` means the sim was packed/staged but never run.**

`status.json` schema (written atomically via `os.replace`, so a watcher never reads a half-written file):

```json
{"run_id": "...", "name": "...", "solver": "fdtdmex",
 "status": "queued|running|completed|failed",
 "step": 740, "total": 2000, "heartbeat": <epoch>,
 "started_at": <epoch>, "finished_at": <epoch|null>,
 "pid": <int>, "error": <str|null>}
```

It advances `queued → running` (step/total + `heartbeat` refreshed each tick) `→ completed`, or `→ failed` (with `error`) on an exception. The `mock` backend drives the same ticks, so the offline GPU-free path exercises telemetry end-to-end.

## Git-like history (app layer)

Append-only event log of `(prompt, tool_call, result)`. A **revert** forks a new branch from an earlier event; the old branch is archived (hidden), and **fast-forward** replays the recorded tool/prompt sequence. Independent of the solver; lives in the orchestration layer.

## Web UI (`web/`)

Decided: **web** front end (best for agentic workflows), **locally hostable** for local/desktop runs. Rendering options, browser-capable:
- **plotly** — native web 3D; quickest to stand up.
- **pyvista via trame** — server-side VTK streamed as VTK.js; keeps the Python scene model, so it *does* run in the browser. Good for a richer scene/editor.
- **three.js / react-three-fiber** — most control, most effort; reserve for a polished editor.

Target interactions (Lumerical-like): click objects to select; tabbed panels to switch boundary conditions, materials, source/detector properties; toggle field overlays. Start with plotly/pyvista-trame to reuse Python geometry; layer three.js later if needed.

## Sequencing
MCP/config layer can start immediately against the pydantic schema (independent of the GPU engine for schema work). The 3D editor is the largest, most open-ended piece. See [roadmap.md](../dev-docs/roadmap.md).
