# FDTDMEX — action plan

Single entry point for a fresh agent. **The forward engine and the mode solver are done and
validated.** The next phase is the **front end + agentic workspace** so simulations are easy to
define, see, run, and orchestrate (today everything is Python-only, which is limiting). A new agent
can read this top-to-bottom without prior context.

## What this project is

FDTDMEX is a fork of [fdtdx](https://github.com/ymahlau/fdtdx) (a JAX FDTD Maxwell solver) that adds a
native **MLX/Metal forward backend** for Apple Silicon. On a Mac a supported forward `run_fdtd`
auto-routes to the MLX time loop; gradients / unsupported features / non-Apple platforms run the
unchanged JAX engine (also the **parity oracle**). Import stays `import fdtdx`; `src/fdtdmex` is a thin
brand alias. The engine is functional / out-of-place (race-free), mirrors fdtdx element-wise, and is
fp32. Goal: fast, large *forward* simulations on a single Mac; inverse design stays on JAX/CUDA.

## Status — done

- **Engine (Phases 1–3).** MLX/Metal custom kernels at the memory-bandwidth floor, default-on
  (`FDTDMEX_METAL_KERNEL=0` forces the MLX-op cores). Full anisotropy, lossy + 9-tensor conductivity,
  CPML + periodic + PEC/PMC, non-uniform grids, and Drude–Lorentz ADE dispersion (folded into the
  E-kernel). All element-wise parity-validated vs forced-JAX. Depth: [`docs/performance.md`](docs/performance.md),
  [`docs/phase2-metal-kernels.md`](docs/phase2-metal-kernels.md).
- **Mode solver (Phase 4 Track A).** Native, **Tidy3D-free** full-vectorial FD mode solver in
  [`src/fdtdx/core/physics/mode_backend/`](src/fdtdx/core/physics/mode_backend/), behind a
  `mode_backend` seam in [`modes.py`](src/fdtdx/core/physics/modes.py) (**default `"fdtdmex"`**, env
  `FDTDMEX_MODE_BACKEND`). **Tidy3D is now optional** (`tidy3d` extra). Straight waveguide, uniform +
  rectilinear grids, isotropic + diagonal anisotropy; off-diagonal tensors + bends raise and auto-route
  to Tidy3D if installed. Matches Tidy3D to ~1e-16 on a Si strip. Depth: [`docs/mode-solver.md`](docs/mode-solver.md).
- **Subpixel smoothing (WS-C core).** [`core/physics/subpixel.py`](src/fdtdx/core/physics/subpixel.py)
  Kottke/Farjadpour tensor smoothing (validated vs analytic effective medium; ~15× mode staircase-error
  cut). Standalone utility today; not yet auto-applied during placement. Depth:
  [`docs/subpixel-smoothing.md`](docs/subpixel-smoothing.md).
- **First front-end pieces.** [`utils/plot_modes.py`](src/fdtdx/utils/plot_modes.py) (`plot_mode`) and
  [`utils/smatrix.py`](src/fdtdx/utils/smatrix.py) (`SMatrixResult` + `plot_smatrix`).

## Phase 5 — front end + agentic workspace (next, the focus)

Make FDTDMEX usable like Tidy3D in a notebook: **define → see → run → inspect**, then expose the same
flow to an LLM through an MCP server. Spec: [`docs/mcp-and-ui.md`](docs/mcp-and-ui.md). Three layers,
build bottom-up; each is independently useful.

### The flow today (what we're improving)

A simulation is assembled as `config` + an `object_list` + placement `constraints`, then
`place_objects(...) → apply_params(...) → run_fdtd(...)`, and results read from
`arrays.detector_states`. See [`examples/simulate_gaussian_source.py`](examples/simulate_gaussian_source.py).
Visualization exists but is matplotlib/save-to-disk via `Logger`: `plot_setup`, `plot_material`,
`plot_field_slice`, detector `plot2d`/`video`, plus the new `plot_mode` / `plot_smatrix`. The gaps:
no single object to hold/inspect a whole setup, no inline reprs, no one-call run, no schema/serialization
for hand-off, no interactive 3D.

### 5a. Notebook front end (do first — unblocks day-to-day use)

Goal: a Tidy3D-like notebook experience reusing the existing matplotlib utilities (they already return
`Figure`s, so they render inline).
- **A `Scene`/`Simulation` facade** (thin, optional) that bundles `config + objects + constraints` and
  offers `.place()`, `.plot()` (wrap `plot_setup`/`plot_material` so the *unplaced* setup can be viewed),
  `.run()` (wrap `place_objects → apply_params → run_fdtd`, MLX auto-routed), and `.results` access. Keep
  the existing low-level API intact; the facade only removes boilerplate.
- **Inline reprs** — `_repr_html_` / `__repr__` summaries for the facade and key objects (counts,
  extents, materials, sources/detectors) so a notebook cell shows the setup at a glance.
- **A quickstart notebook / narrated example** that walks define → `plot_setup` → `plot_material` →
  `run` → `plot_field_slice` / `plot_mode` / `plot_smatrix`, end-to-end, inline. This is the concrete
  "see how a sim is defined, built, run" artifact.
- Verify each plot renders inline and the facade `.run()` matches the explicit-call result.

### 5b. Config schema + the sim_init/sim_run/sim_postproc trio (the agentic seam)

The portable contract an LLM and any compute node agree on; the **LLM never touches large arrays**.
- **Config schema** (`fdtdmex/io/`, pydantic): typed, validated models for Volume, Materials,
  Structures, Sources, Detectors, Boundaries, Grid, Run. Round-trips to **JSON**; align field names with
  fdtdx's JSON round-trip ([`conversion/json.py`](src/fdtdx/conversion/json.py)) so fdtdx setups ingest.
- **`sim_init(setup) → config.hdf5`** — front-end creation utility: resolve objects + design params →
  assemble the **resolved** ε/µ/σ + dispersion + frozen source/detector profiles + grid/boundary spec →
  write a self-contained config HDF5 (h5py). **Bare-minimum rule:** ship the resolved
  `ArrayContainer`, not the pre-sim data that produces it (no device ρ / optimization params / CSG).
- **`sim_run(config.hdf5) → results.hdf5`** — unwrap and run on any FDTDMEX machine (local/remote). The
  unwrap feeds the existing [`to_mlx_state`](src/fdtdx/mlx/bridge.py)/`freeze_*` seam (the MLX bridge is
  already a "resolved arrays → run" boundary, so the HDF5 is essentially its serialized form).
- **`sim_postproc(results.hdf5) → small results`** — reduce to scalars/fluxes/n_eff/S-params/thumbnails.
- Buildable now against a **mocked backend**; validate JSON/HDF5 round-trip and that a resolved config
  reproduces a direct `run_fdtd`.

### 5c. MCP server (`server/fdtdmex_mcp/`, stub today)

API discovery + the trio, so an LLM writes a valid script. Tools: **introspect** (type names + param
schemas from the pydantic/`autoinit` models), **build/edit/validate**, and **sim_init/sim_run/
sim_postproc**. Install via the `mcp` extra. Depends on 5b's schema.

### 5d. Web UI (`web/`, placeholder) — later

Locally-hostable, Lumerical-like 3D editor (click-select, tabbed material/source/boundary panels, field
overlays). Start with plotly or pyvista-via-trame (reuse the Python scene); three.js later. Largest,
most open-ended piece — after 5a–5c.

## Engine backlog (deferred, not blocking Phase 5)

- **Tensorial mode solver** — off-diagonal 9-tensor cross-sections (the 4N×4N complex eigenproblem;
  even Tidy3D's base package defers this to a paid extra). Today these auto-route to Tidy3D if installed.
- **Bends / PML leaky modes** in the native solver (route to Tidy3D meanwhile).
- **WS-C auto-integration** — apply `subpixel.py` during placement (host-side geometry supersampling),
  opt-in, default off to preserve parity.
- **Bloch / complex (nonzero-k) propagation** — promote the MLX forward path to complex64 end-to-end
  (curl, E/H update, CPML, sources, detectors, kernels). Parity vs the JAX complex oracle. The last
  same-port-pattern engine feature; gradients stay out of scope.

## Engine map (`src/fdtdx/mlx/` + mode backend)

| file | role |
|---|---|
| [`mlx/loop.py`](src/fdtdx/mlx/loop.py) | time-loop driver; builds E/H cores (kernel or MLX-op), host-gated source injection |
| [`mlx/kernels.py`](src/fdtdx/mlx/kernels.py) | custom Metal E/H bulk kernels (per-cell `cb`, in-kernel CPML fold, non-uniform metric, ADE dispersion) + block hybrid for full-tensor inclusions; `kernel_eligible` |
| [`mlx/curl.py`](src/fdtdx/mlx/curl.py) · [`update.py`](src/fdtdx/mlx/update.py) · [`pml.py`](src/fdtdx/mlx/pml.py) | pad-free Yee curl + slab-CPML; E/H update (iso/diag + full-tensor); CPML coeff precompute |
| [`mlx/bridge.py`](src/fdtdx/mlx/bridge.py) · [`state.py`](src/fdtdx/mlx/state.py) | ArrayContainer ↔ MLXState (the resolved-arrays seam Phase 5b serializes) |
| [`backend/dispatch.py`](src/fdtdx/backend/dispatch.py) | routing + feature gating; [`backend/context.py`](src/fdtdx/backend/context.py) `use_backend` |
| [`core/physics/mode_backend/`](src/fdtdx/core/physics/mode_backend/) | native FD mode solver (operator + eigensolver + adapter) |
| [`core/physics/modes.py`](src/fdtdx/core/physics/modes.py) | `compute_mode` + `mode_backend` seam (fdtdmex default, tidy3d optional) |
| [`core/physics/subpixel.py`](src/fdtdx/core/physics/subpixel.py) | Kottke/Farjadpour subpixel smoothing |
| [`utils/plot_*`](src/fdtdx/utils/) · [`utils/smatrix.py`](src/fdtdx/utils/smatrix.py) | matplotlib visualization + S-matrix result |

## Physics-correctness contract (every engine change)

- **Out-of-place / race-free**; **leapfrog order** `update_E → inject E → update_H → inject H →
  detectors` (never merge E and H). Source/detector gating stays host-side.
- **Element-wise parity** vs the forced-JAX oracle, rel < 1e-3. Marginal failure → raise resolution,
  never loosen tolerance. fp32 is the floor.

## Validation & commands

```bash
uv run --with pytest pytest tests/validation -q                          # parity (kernel default-on)
FDTDMEX_METAL_KERNEL=0 uv run --with pytest pytest tests/validation -q   # parity, MLX-op cores
uv run --with pytest pytest tests/validation/test_mode_solver.py tests/validation/test_subpixel.py -q
uv run python benchmarks/bench_forward.py --backends mlx,jax --sizes 96,128,192,256 --steps 500 --repeats 2
uvx ruff format src/fdtdx && uvx ruff check src/fdtdx
```
Force a backend: `with fdtdx.use_backend("mlx"|"jax")` or `FDTDMEX_BACKEND=mlx|jax`. Mode backend:
`FDTDMEX_MODE_BACKEND=fdtdmex|tidy3d` or the `mode_backend=` arg. Work on a branch off `mlx-fork`;
local commits only.
