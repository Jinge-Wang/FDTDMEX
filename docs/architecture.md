# Architecture

## Goal & scope

FDTDMEX is a **forward** FDTD Maxwell solver on **MLX/Metal** for a single Apple-Silicon machine.
The thesis: Apple **unified memory** (up to 512 GB) is the decisive advantage for **large, full-tensor anisotropic** simulations, whose per-voxel 3×3 ε tensors (~9× the isotropic footprint) overflow or thrash a single CUDA GPU's VRAM/PCIe.

**Out of scope:** gradient-based inverse design on Metal. That needs cluster-scale parallelism and stays on JAX/CUDA (e.g. FDTDX). Consequently we do **not** port the reversible/custom-VJP gradient machinery — a huge simplification.

## Data flow

```
            ┌─────────────── host (CPU / numpy) ───────────────┐      ┌──── Metal GPU (MLX) ────┐
 config ──► │ geometry → voxelization → subpixel smoothing(WS-C)│      │  forward time loop:     │
 (pydantic) │ → material tensor arrays (1/3/9-component)         │ ───► │   curl → E/H update     │
            │ PML profiles, source temporal profiles            │ bridge│   → CPML → source inject │
            │ mode solve (WS-B, scipy sparse eig)               │ np→mx │   → detector accumulate  │
            └───────────────────────────────────────────────────┘      └──────────────────────────┘
                                                                              │
                                                            results (fields, flux, phasors) ──► viz / HDF5
```

Everything left of the bridge is host-side and largely framework-agnostic (much of it reusable from FDTDX's CPU front end). Everything right of the bridge is the MLX hot loop we own.

## Workstreams

- **WS-A — Forward MLX engine (foundational).** Curl, E/H update (isotropic/diagonal fast path + full-anisotropic 3×3 path), CPML, source injection, detector accumulation, the Python+`mx.compile` time loop. **Non-uniform grids first-class** (spacing-weighted; see [nonuniform-grid.md](nonuniform-grid.md)).
- **WS-B — Mode solver front end.** 2D-Yee full-vectorial FD eigenproblem (scipy sparse `eigs`, host) → (n_eff, transverse fields); mode overlap = spatial integral vs a field monitor; injection ported from FDTDX's TFSF. Reuses WS-C's index averaging. See [mode-solver.md](mode-solver.md).
- **WS-C — Subpixel smoothing.** Kottke/Farjadpour effective-**tensor** averaging as a host pre-step that emits the per-Yee-component material tensors WS-A consumes. See [subpixel-smoothing.md](subpixel-smoothing.md).
- **WS-D — Orchestration.** Declarative pydantic config → (JSON/HDF5) → backend; MCP server for LLM tool-use; git-like branch/revert of the prompt+tool history; locally-hosted **web** UI with a Lumerical-like 3D editor. See [mcp-and-ui.md](mcp-and-ui.md).

## Dependency graph

```
WS-A (engine) ──┬──> WS-C (smoothing) ──> WS-B (modes; reuses WS-C index averaging)
                └──> WS-D (MCP/UI; needs a runnable backend)
WS-D MCP/config layer can start independently against a declarative schema.
```

Suggested order: **WS-A (MVP → anisotropic) + WS-C → WS-B → WS-D (MCP first, then web UI)**.

## Package map (`src/fdtdmex/`)

| Module | Responsibility |
|---|---|
| `backend/` | MLX helpers: dtype/device, `mx.compile` wrappers, complex helpers, np↔mx bridge. |
| `core/` | config (pydantic), constants, grid (uniform + non-uniform, cell-size arrays), typing. |
| `fdtd/` | WS-A engine: curl, update_E/H, pml (CPML), time loop. |
| `materials/` | material model (1/3/9-tensor, dispersion ADE); `smoothing/` = WS-C. |
| `geometry/` | shapes, GDS import, voxelization, `chi1p1` continuous-ε evaluator (feeds WS-C). |
| `sources/` | plane/dipole/TFSF + mode injection. |
| `detectors/` | field/energy/Poynting/phasor + mode overlap. |
| `modes/` | WS-B FD mode solver + overlap. |
| `io/` | serialization (pydantic ↔ JSON, HDF5 results), FDTDX array-bridge. |
| `viz/` | matplotlib/plotly export; pyvista/trame for 3D/web. |

## Key design decisions
- **Forward-only, no on-device autodiff** → no reversible gradient, no `custom_vjp`, no checkpointing.
- **Functional/out-of-place updates** → race-free without ping-pong buffers (see [porting-notes.md](porting-notes.md)).
- **Spacing-weighted operators** → correct on non-uniform grids by construction.
- **Host/GPU split at a plain-array bridge** → reuse FDTDX's mature front end; own only the hot loop.
- See [decisions/0001-mlx-forward-first.md](decisions/0001-mlx-forward-first.md).
