# FDTDMEX — action plan

Single entry point for the project. The forward engine, the native mode solver, the notebook front end, the portable HDF5 hand-off, and the MCP discovery server are **done and validated**. What remains is the **web UI**, widening the Metal engine to cover the mode-source path, **staying synced with upstream fdtdx**, and a few engine-backlog items. A new contributor can read this top-to-bottom without prior context. For the capability-by-capability arc and what's coming (with rationale), see [roadmap.md](roadmap.md); for the JAX→MLX porting recipe see [porting.md](porting.md); for how we track upstream and what flows back, see [UPSTREAM_SYNC.md](UPSTREAM_SYNC.md) + [UPSTREAM_CONTRIB.md](UPSTREAM_CONTRIB.md).

## What this project is

FDTDMEX is a fork of [fdtdx](https://github.com/ymahlau/fdtdx) (a JAX FDTD Maxwell solver) that adds a native **MLX/Metal forward backend** for Apple Silicon. On a Mac a supported forward `run_fdtd` auto-routes to the MLX time loop; gradients, unsupported features, and non-Apple platforms run the unchanged JAX engine (also the **parity oracle**). Import stays `import fdtdx`; `src/fdtdmex` is a thin brand alias. The engine is functional / out-of-place (race-free), mirrors fdtdx element-wise, and is fp32. Goal: fast, large *forward* simulations on a single Mac; inverse design stays on JAX/CUDA.

## Done

- **Forward engine.** MLX/Metal custom kernels at the memory-bandwidth floor, default-on (`FDTDMEX_METAL_KERNEL=0` forces the MLX-op cores). Full anisotropy, lossy + 9-tensor conductivity, CPML + periodic + PEC/PMC, non-uniform grids, and Drude–Lorentz dispersion (folded into the E-kernel). All element-wise parity-validated vs forced-JAX. Depth: [`docs/performance.md`](../docs/performance.md) (kernel design + roofline + Apple-Silicon ceilings), [`porting.md`](porting.md) (how each piece was ported).
- **Mode solver.** Native, Tidy3D-optional full-vectorial FD mode solver ([`core/physics/mode_backend/`](../src/fdtdx/core/physics/mode_backend/)) behind a `mode_backend` seam ([`modes.py`](../src/fdtdx/core/physics/modes.py), default `"fdtdmex"`). Straight waveguides, uniform + rectilinear grids, isotropic + diagonal anisotropy; matches the analytic slab dispersion to ~2e-4. Depth: [`docs/mode-solver.md`](../docs/mode-solver.md).
- **Mode-expansion monitor.** [`utils/mode_expansion.py`](../src/fdtdx/utils/mode_expansion.py) (`compute_mode_expansion`) projects a recorded field onto a user-specified mode set → per-mode transmission + complex S-parameters, with a validated mode cache.
- **Subpixel smoothing.** [`core/physics/subpixel.py`](../src/fdtdx/core/physics/subpixel.py) Kottke tensor smoothing (validated; standalone utility, not yet auto-applied during placement).
- **Notebook front end.** `Scene` facade ([`scene.py`](../src/fdtdx/scene.py)), interactive 3D ([`utils/plot_setup_3d.py`](../src/fdtdx/utils/plot_setup_3d.py)), and the matplotlib utilities (`plot_setup`, `plot_material`, `plot_mode`). Tour: [`examples/ring_resonator_demo/`](../examples/ring_resonator_demo/).
- **Portable HDF5 hand-off.** [`src/fdtdmex/io/`](../src/fdtdmex/io/) — `SceneModel` (pydantic facade over fdtdx's JSON), and the agent-facing `pack` → `run_simulation_from_hdf5` → `sim_postproc` flow (non-blocking detached launch; fdtdmex stages + owns the job folder), over the `sim_run` engine primitive, with a GPU-free `mock` backend. The packed run reproduces a direct `run_fdtd` bit-for-bit.
- **MCP discovery server.** [`server/fdtdmex_mcp/`](../server/fdtdmex_mcp/) — a **discovery-only** stdio MCP server (4 fixed tools: `list_solver_apis` / `get_api_schema` via live `inspect.signature` / `search_docs` / `get_doc` over a BM25 corpus built from real sources) teaching an agent the native run flow (`pack` → non-blocking `run_simulation_from_hdf5` → `sim_postproc`, plus `compute_mode`). It never runs a sim. Launched by ag-fdtd's UI (`uv run fdtdmex-mcp`) or standalone in any MCP host. Deps via the `mcp` (+ `io`) extra. See [`docs/mcp-and-ui.md`](../docs/mcp-and-ui.md).
- **Run progress telemetry.** `sim_run(progress=cb)` streams the Metal loop's step counter (host-side, no GPU sync); `run_simulation_from_hdf5` (via `launch.py` + `_runner.py`) drives a per-job `status.json` so the agent's non-blocking launch is observable. fdtdmex owns the job folder end to end.

## Next — the focus

Two remaining build items, plus the ongoing upstream-sync track below.

### 1. Mode sources / detectors on Metal

Today the mode-expansion workflow (mode source + mode-overlap detector) routes the forward time loop to **JAX**. Porting mode-source injection and the mode-overlap detector into the MLX path (the freeze seam in [`mlx/source_freeze.py`](../src/fdtdx/mlx/source_freeze.py) / [`mlx/detector_freeze.py`](../src/fdtdx/mlx/detector_freeze.py)) would run the whole PIC workflow — and the showcase — on the Metal engine. **Done when** a mode-source forward run is MLX-eligible and parity-clean vs JAX.

### 2. Web UI ([`web/`](../web/), placeholder)

A locally-hostable reactive editor consuming `SceneModel` + `to_plotly_json(plot_setup_3d(...))`: click to select objects, tabbed panels for materials / sources / boundaries, field overlays, and a confirm-before-run gate that records only the confirmed setup. Start with plotly or pyvista-via-trame. The largest, most open-ended piece.

## Upstream sync & contributions

FDTDMEX is an additive fork; staying current with [ymahlau/fdtdx](https://github.com/ymahlau/fdtdx) is an ongoing track, not a one-off. The full protocol, branch model, porting rules, and contract-surface checklist live in [UPSTREAM_SYNC.md](UPSTREAM_SYNC.md); what flows *back* upstream (autodiff-safe only) is in [UPSTREAM_CONTRIB.md](UPSTREAM_CONTRIB.md).

**Sync in — fork-base now `e5351a4` (synced 2026-06-27; was `77e1281`). Queue clear.**
- [x] Merged **#372** (nonuniform PML staggered-profile fix). The Metal path inherits it free via `mlx/bridge.py`; graded-PML parity covered by `test_mlx_nonuniform.py` (PML-z stretched grid).
- [x] Merged **#363** (quasi-uniform grid + origin-at-center). Verified a no-op for the engine: object placement is index-invariant (the L/2 offset that was added explicitly under corner-origin is now the grid origin `-L/2`, so `bounds_for_center` yields identical cells); the MLX engine never reads grid origin / absolute coords. `QuasiUniformGrid` resolves to a `RectilinearGrid` and flows through transparently (no dispatch gate). No examples/`Scene`/HDF5 used corner-origin absolute coords. No performance impact (the only added work is a one-time grid-resolve at placement).
- [x] Clean merge, zero conflicts; `test_mlx_parity.py` (10) + `test_mlx_nonuniform.py` (4) green. Local commits only — not pushed.

**Contribute out (autodiff-safe; see UPSTREAM_CONTRIB.md for detail):**
- [ ] **Off-diagonal anisotropic averaging** width-weighting → JAX `fdtd/misc.py` (upstream's `/4` is 1st-order on graded grids for full-tensor media; also fixes our own JAX path). Cleanest first PR.
- [ ] **Region-restricted detector interpolation** + activity-gating → `update_detector_states` (upstream interpolates the whole domain every step; the ring example went 1478→377 s from this class of change). Differentiable, scan-safe.
- [ ] **Nyquist-aware DFT subsampling** for phasor/frequency monitors (`lax.cond` stride gate + Riemann normalization). New capability upstream lacks.
- [ ] **Tidy3D-free FD mode solver** (numpy/scipy, not in the gradient path) → optional `mode_backend`. RFC first.
- [ ] **Kottke subpixel smoothing** — discuss a JAX (differentiable) port; numpy version is forward-accuracy-only.

## Engine + front-end backlog (deferred, non-blocking)

- **`SceneModel` ↔ `ExtrudedPolygon` round-trip** — the JSON config exports GDS-derived polygons but reconstructing live `ExtrudedPolygon` objects hits their derived-shape guard; small follow-up.
- **Subpixel-smoothing auto-integration** — apply [`subpixel.py`](../src/fdtdx/core/physics/subpixel.py) during placement (host-side supersampling), opt-in, default off to preserve parity.
- **Tensorial mode solver** — off-diagonal 9-tensor cross-sections (the 4Nx4N complex eigenproblem); routes to Tidy3D today if installed.
- **Bends / leaky modes** in the native solver (route to Tidy3D meanwhile).
- **Bloch / complex (nonzero-k) propagation** — promote the MLX forward path to complex64 end-to-end; parity vs the JAX complex oracle. Gradients stay out of scope.
- **Production-resolution ring validation** — ✅ **done**: [`examples/ring_mrm_oband/`](../examples/ring_mrm_oband/) — O-band carrier-depletion MRM, a full design-verification run end to end on **Metal** at a 20 nm grid (build + mode → mesh convergence **40→20 nm** → cold `T(λ)`/Q/ER + field maps → gap sweep / coupling control → Soref–Bennett EO `Δλ(V)`). Uses a Gaussian source + phasor monitors and a **net-Poynting two-run** transmission (mode sources/detectors would force JAX/CPU here). Method, recipe, and acceptance notes live in the example's [README](../examples/ring_mrm_oband/README.md); ~5 h for the full suite at 20 nm.

## Engine map (`src/fdtdx/mlx/` + mode backend + front end)

| file | role |
|---|---|
| [`mlx/loop.py`](../src/fdtdx/mlx/loop.py) | time-loop driver; builds E/H cores (kernel or MLX-op), host-gated source injection |
| [`mlx/kernels.py`](../src/fdtdx/mlx/kernels.py) | custom Metal E/H bulk kernels (per-cell `cb`, in-kernel CPML fold, non-uniform metric, ADE dispersion) + block hybrid for full-tensor inclusions |
| [`mlx/curl.py`](../src/fdtdx/mlx/curl.py) · [`update.py`](../src/fdtdx/mlx/update.py) · [`pml.py`](../src/fdtdx/mlx/pml.py) | pad-free Yee curl + slab-CPML; E/H update; CPML coeff precompute |
| [`mlx/bridge.py`](../src/fdtdx/mlx/bridge.py) · [`state.py`](../src/fdtdx/mlx/state.py) · [`serialize.py`](../src/fdtdx/mlx/serialize.py) | ArrayContainer ↔ MLXState (the resolved-arrays seam) + numpy↔mx serialization for HDF5 |
| [`backend/dispatch.py`](../src/fdtdx/backend/dispatch.py) | routing + feature gating; `run_forward_from_plans` (the HDF5 run tail) |
| [`core/physics/mode_backend/`](../src/fdtdx/core/physics/mode_backend/) · [`modes.py`](../src/fdtdx/core/physics/modes.py) | native FD mode solver + `compute_mode` seam |
| [`utils/mode_expansion.py`](../src/fdtdx/utils/mode_expansion.py) | mode-expansion monitor + mode cache |
| [`scene.py`](../src/fdtdx/scene.py) · [`utils/plot_setup_3d.py`](../src/fdtdx/utils/plot_setup_3d.py) | `Scene` facade + interactive plotly 3D |
| [`src/fdtdmex/io/`](../src/fdtdmex/io/) | `SceneModel` schema + `pack` / `run_simulation_from_hdf5` (non-blocking launch, `launch.py` + `_runner.py`) / `run_simulation` (cwd worker) / `sim_run` (engine) / `sim_postproc` + mock backend |

## Physics-correctness contract (every engine change)

- **Out-of-place / race-free**; **leapfrog order** `update_E → inject E → update_H → inject H → detectors` (never merge E and H). Source/detector gating stays host-side.
- **Element-wise parity** vs the forced-JAX oracle, rel < 1e-3. Marginal failure → raise resolution, never loosen tolerance. fp32 is the floor.

## Validation & commands

```bash
uv run --with pytest pytest tests/validation -q                          # parity (kernel default-on)
FDTDMEX_METAL_KERNEL=0 uv run --with pytest pytest tests/validation -q   # parity, MLX-op cores
uv run --with pytest pytest tests/validation/test_mode_solver.py tests/validation/test_mode_expansion.py tests/validation/test_io_roundtrip.py -q
uv run python benchmarks/bench_forward.py --backends mlx,jax --sizes 96,128,192,256 --steps 500 --repeats 2
uvx ruff format src/fdtdx src/fdtdmex && uvx ruff check src/fdtdx src/fdtdmex
```
Force a backend: `with fdtdx.use_backend("mlx"|"jax")` or `FDTDMEX_BACKEND=mlx|jax`. Mode backend: `FDTDMEX_MODE_BACKEND=fdtdmex|tidy3d`. Work on a branch off `mlx-fork`; local commits only.
