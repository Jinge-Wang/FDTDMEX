# FDTDMEX — action plan

Single entry point for the project. The forward engine, the native mode solver, the notebook front end, and the portable HDF5 hand-off are **done and validated**. What remains is the orchestration layer (MCP server, web UI), widening the Metal engine to cover the mode-source path, and a few engine-backlog items. A new contributor can read this top-to-bottom without prior context. For the capability-by-capability arc and what's coming (with rationale), see [roadmap.md](roadmap.md); for the JAX→MLX porting recipe and worked examples, see [porting.md](porting.md).

## What this project is

FDTDMEX is a fork of [fdtdx](https://github.com/ymahlau/fdtdx) (a JAX FDTD Maxwell solver) that adds a native **MLX/Metal forward backend** for Apple Silicon. On a Mac a supported forward `run_fdtd` auto-routes to the MLX time loop; gradients, unsupported features, and non-Apple platforms run the unchanged JAX engine (also the **parity oracle**). Import stays `import fdtdx`; `src/fdtdmex` is a thin brand alias. The engine is functional / out-of-place (race-free), mirrors fdtdx element-wise, and is fp32. Goal: fast, large *forward* simulations on a single Mac; inverse design stays on JAX/CUDA.

## Done

- **Forward engine.** MLX/Metal custom kernels at the memory-bandwidth floor, default-on (`FDTDMEX_METAL_KERNEL=0` forces the MLX-op cores). Full anisotropy, lossy + 9-tensor conductivity, CPML + periodic + PEC/PMC, non-uniform grids, and Drude–Lorentz dispersion (folded into the E-kernel). All element-wise parity-validated vs forced-JAX. Depth: [`docs/performance.md`](../docs/performance.md) (kernel design + roofline + Apple-Silicon ceilings), [`porting.md`](porting.md) (how each piece was ported).
- **Mode solver.** Native, Tidy3D-optional full-vectorial FD mode solver ([`core/physics/mode_backend/`](../src/fdtdx/core/physics/mode_backend/)) behind a `mode_backend` seam ([`modes.py`](../src/fdtdx/core/physics/modes.py), default `"fdtdmex"`). Straight waveguides, uniform + rectilinear grids, isotropic + diagonal anisotropy; matches the analytic slab dispersion to ~2e-4. Depth: [`docs/mode-solver.md`](../docs/mode-solver.md).
- **Mode-expansion monitor.** [`utils/mode_expansion.py`](../src/fdtdx/utils/mode_expansion.py) (`compute_mode_expansion`) projects a recorded field onto a user-specified mode set → per-mode transmission + complex S-parameters, with a validated mode cache.
- **Subpixel smoothing.** [`core/physics/subpixel.py`](../src/fdtdx/core/physics/subpixel.py) Kottke tensor smoothing (validated; standalone utility, not yet auto-applied during placement).
- **Notebook front end.** `Scene` facade ([`scene.py`](../src/fdtdx/scene.py)), interactive 3D ([`utils/plot_setup_3d.py`](../src/fdtdx/utils/plot_setup_3d.py)), and the matplotlib utilities (`plot_setup`, `plot_material`, `plot_mode`). Tour: [`examples/ring_resonator_demo/`](../examples/ring_resonator_demo/).
- **Portable HDF5 hand-off.** [`src/fdtdmex/io/`](../src/fdtdmex/io/) — `SceneModel` (pydantic facade over fdtdx's JSON), and the agent-facing `pack` → `run_simulation_from_hdf5` → `sim_postproc` flow (non-blocking detached launch; fdtdmex stages + owns the job folder), over the `sim_run` engine primitive, with a GPU-free `mock` backend. The packed run reproduces a direct `run_fdtd` bit-for-bit.

## Next — orchestration layer (the focus)

Build bottom-up; each piece is independently useful. The schema and IO contract are done, so this layer can be built directly against them.

### 1. MCP discovery server ([`server/fdtdmex_mcp/`](../server/fdtdmex_mcp/)) — ✅ **done**

The **discovery-only** API surface so an LLM can drive a simulation *natively* in its own kernel (it
teaches *what to write*, never runs a sim). Four fixed tools — `list_solver_apis` / `get_api_schema`
(live `inspect.signature`) / `search_docs` / `get_doc` (BM25 corpus from real sources) — advertising
the native flow: assemble a `Scene` → `pack(config, location)` → non-blocking
`run_simulation_from_hdf5(bundle, parent_folder)` → `sim_postproc(results)`, plus `compute_mode`. The
server is launched by ag-fdtd's UI (`uv run fdtdmex-mcp`), and also runs standalone in any MCP host
(`server/install.sh`). Install the deps via the `mcp` (+ `io`) extra. See
[`docs/mcp-and-ui.md`](../docs/mcp-and-ui.md).

### 2. Mode sources / detectors on Metal

Today the mode-expansion workflow (mode source + mode-overlap detector) routes the forward time loop to **JAX**. Porting mode-source injection and the mode-overlap detector into the MLX path (the freeze seam in [`mlx/source_freeze.py`](../src/fdtdx/mlx/source_freeze.py) / [`mlx/detector_freeze.py`](../src/fdtdx/mlx/detector_freeze.py)) would run the whole PIC workflow — and the showcase — on the Metal engine. **Done when** a mode-source forward run is MLX-eligible and parity-clean vs JAX.

### 3. Web UI ([`web/`](../web/), placeholder)

A locally-hostable reactive editor consuming `SceneModel` + `to_plotly_json(plot_setup_3d(...))`: click to select objects, tabbed panels for materials / sources / boundaries, field overlays, and a confirm-before-run gate that records only the confirmed setup. Start with plotly or pyvista-via-trame. The largest, most open-ended piece.

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
