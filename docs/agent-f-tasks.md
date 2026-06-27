# Agent F tasks — FDTDMEX support for native ag-fdtd execution (v2, supersedes v1)

**Self-contained.** You are the FDTDMEX-repo agent. The sibling **ag-fdtd** agentic workspace
(`~/Projects/Kronos/ag-fdtd`) executes simulations **natively**: the LLM/user writes fdtdmex
Python in a notebook kernel that runs in *this* repo's venv, **discovered only via your MCP
server** — there are NO hidden ag-fdtd helper functions injected into the kernel. The agent knows
*only* the MCP-taught API.

**This v2 replaces v1.** v1 told you to make the run **blocking** and let ag-fdtd own the job
folder + do the detaching. That was wrong. The correct model: **a non-blocking, job-folder-owning
launch function lives HERE** (the agent calls it; it returns immediately while the solver runs
detached). v1's `run_with_status`/blocking assumptions are superseded below.

---

## Hard rules
- **Do NOT touch the core solver numerics** (`src/fdtdx` time loop, mode-solver math). The **MCP
  server**, the **io seam's non-numeric surface** (pack/launch/IO/status), the **agent adapter**,
  the **corpus**, and **docs** are all fair game — modify/tailor freely.
- **The `mock` backend must keep working, GPU-free** (`backend="mock"`): ag-fdtd's offline CI runs
  against it. The new launch path must support `backend="mock"` end-to-end (status + results).
- **MCP stays discovery-only** (4 tools: `list_solver_apis`/`get_api_schema`/`search_docs`/
  `get_doc`). It teaches *what to write*; it never runs a sim.
- **Clean up as you go** — delete the legacy adapter + stale docs/handoffs this work obsoletes
  (listed under "Cleanup"). Don't leave dead functions behind.

---

## The model (what the agent writes)

Three layers of identity, on disk in two places:

```python
# 1. ASSEMBLE — a config OBJECT in the kernel namespace; nothing on disk yet.
config = build_scene(...)            # fdtdx Scene / SceneModel, with data-file paths inside

# 2. PACK — materialize into the PROJECT FOLDER (ag-fdtd forces `location` = project root).
bundle = pack(config, location, hdf5_name=None)
#   • writes a self-contained packed HDF5 (config + resolved/ingested data) at `location`
#   • hdf5_name OPTIONAL — if omitted, name it by the config-content hash
#   • ALSO writes/hashes a lightweight config file (the small editable JSON) at `location`
#   • returns the HDF5 path (+ hashes). Reusable: one packed HDF5 can back many runs.

# 3. RUN — agent ALWAYS uses the *_from_hdf5 form. Non-blocking.
handle = run_simulation_from_hdf5(bundle_hdf5, parent_folder, simulation_name=None, backend="mlx")
#   • parent_folder: ag-fdtd forces it to the workspace `jobs/` dir
#   • hashes the HDF5 → a NEW UNIQUE job id (HDF5 hash + entropy), so the SAME packed HDF5
#     can launch MULTIPLE distinct runs
#   • job_dir = <parent_folder>/<simulation_name OR new-unique-hash>/
#   • COPIES the HDF5 into job_dir; creates job_dir/outputs/; drops a lightweight config.json
#     snapshot in job_dir (so the UI can show config without cracking the HDF5)
#   • LAUNCHES THE SOLVER DETACHED and RETURNS IMMEDIATELY with pointers (job_dir / status path)
#   • the detached solver writes + updates status.json (+ progress.jsonl) IN job_dir as it runs
```

**Status-file semantics (the cross-repo contract): a job folder with NO `status.json` ⇒ the sim
was packed/staged but never run.** Presence + contents (`queued→running→completed|failed`) are how
ag-fdtd tracks state. ag-fdtd's watcher fires on the **new `jobs/<job_name>/` folder appearing**
(created by `run_simulation_from_hdf5`) → it loads a sim object and lets the user click to view the
`config.json` there.

---

## Task 1 — `pack` (extend `sim_init`)
`sim_init`/`pack` already resolves a Scene into a self-contained `config.hdf5`. Extend it to the
signature above:
- `pack(config, location, hdf5_name=None) -> paths`. Write the packed HDF5 to `location`; name it
  `hdf5_name` if given, else `<config_hash>.hdf5`. Also write the **lightweight config JSON** to
  `location` and return both paths + the config hash. Keep the config JSON reachable as a small
  HDF5 group too (cheap peek), as today.
- Keep `sim_init`'s existing callers working (alias/clear migration if you rename).

## Task 2 — `run_simulation` (bare primitive, NOT agent-facing)
- `run_simulation(config_or_hdf5, *, backend="mlx", progress=None) -> results_path` — runs **in the
  current working directory of the process**, writing status/progress/results **into the cwd**.
  This is the BLOCKING worker (it *is* what a detached child executes). **Do not teach it via MCP**;
  the agent must never call it directly. (This is essentially today's `run_with_status` retargeted
  at cwd — fold `run_with_status` into it.)

## Task 3 — `run_simulation_from_hdf5` (NEW — the agent-facing launcher)
- `run_simulation_from_hdf5(hdf5_path, parent_folder, *, simulation_name=None, backend="mlx",
  name="") -> handle`:
  1. Hash `hdf5_path` → a fresh unique job id (HDF5 content hash + entropy ⇒ unique per call).
  2. `job_dir = <parent_folder>/<simulation_name or job_id>/`; create it + `job_dir/outputs/`.
  3. Copy the HDF5 into `job_dir`; write a lightweight `job_dir/config.json` snapshot.
  4. **Launch the solver DETACHED** (a child process that `run_simulation`s the copied HDF5 inside
     `job_dir`, driving a `StatusWriter` → `status.json` + `progress.jsonl`). The child survives the
     parent cell finishing.
  5. **Return immediately** with a handle/pointers (`job_dir`, `status.json` path). **Non-blocking.**
- `backend="mock"` must work the whole way (detached child fabricates ticks + results, GPU-free).
- The detach mechanism is yours (double-fork / `subprocess.Popen` of a small runner module /
  `multiprocessing` with `daemon=False`) — just guarantee: returns immediately, child outlives the
  caller, status written atomically into `job_dir`.
- **Done-check:** calling it returns in <1s; `status.json` appears in `job_dir` and advances
  `queued→running→completed`; a second call on the same HDF5 makes a *second distinct* `job_dir`.

## Task 4 — MCP discovery: teach the native path, drop the legacy
- `list_solver_apis` / `get_api_schema` advertise: **`pack`**, **`run_simulation_from_hdf5`**,
  `sim_postproc`, `compute_mode`. Do **not** advertise `run_simulation` (primitive) — and remove any
  advertisement of the old ag-fdtd-side `run_fdtd`/`build_simulation`/`update_param`/`SimConfig`.
- The corpus (`search_docs`/`get_doc`) examples must show the real flow: assemble Scene → `pack(...,
  location)` → `run_simulation_from_hdf5(..., parent_folder)` → (later) `sim_postproc(outputs)`.
  Remove the stale `get_doc` "Next: update_param … build_simulation" hint and the
  "sim_init→sim_run→sim_postproc inline" hint (that one blocks — it must be the `_from_hdf5` launch).
- **Done-check:** `search_docs("run a simulation") → get_doc(...)` shows the pack + `_from_hdf5`
  launch (non-blocking) pattern; `get_api_schema("run_simulation_from_hdf5")` shows live params.

## Task 5 — Cleanup (delete dead weight)
- **`agentic_adapter/real_solver.py`** — the CLI adapter ag-fdtd used to spawn. The native launch
  path replaces it. Delete it (or reduce to a thin shim) once Tasks 1–3 land; nothing in ag-fdtd
  will spawn it anymore.
- **Stale docs/handoffs:** `docs/agent-f-mcp-brief.md`, `docs/agent-f-progress-task.md` (folded in),
  and any `mcp-and-ui.md` text describing the blocking/`run_with_status`/adapter model — reconcile
  to this v2. Delete this very file's v1 assumptions (you're reading v2).

---

## Cross-repo contracts (keep these stable; ag-fdtd depends on them)
1. **4-tool MCP discovery** surface + corpus.
2. **Job-folder layout** written by `run_simulation_from_hdf5`: `<parent>/<job_name>/` containing
   `status.json` (the schema below), `progress.jsonl`, `config.json` (lightweight snapshot), the
   copied bundle HDF5, and `outputs/`.
3. **`status.json` schema** (atomic writes; absent ⇒ not run):
   `{run_id, name, solver:"fdtdmex", status:"queued|running|completed|failed", step, total,
   heartbeat, started_at, finished_at|null, pid, error|null}`.
4. **`backend="mock"`** GPU-free end-to-end.

If any of these change, flag ag-fdtd (`dev-docs/TARGET_ARCHITECTURE.md` + `ACTION_PLAN.md`) so both
update together.
