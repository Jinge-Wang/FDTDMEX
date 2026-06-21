# Roadmap

Estimates are part-time, AI-paired. Code generation is cheap; **physics validation and Metal perf are the long pole**.

**Status:** WS-A (the forward MLX engine) is **complete and validated** through non-uniform grids. WS-B/C/D and the physics extensions are open.

## WS-A — Forward MLX engine ✅ **complete**

| Milestone | Scope | Status |
|---|---|---|
| **M1 — MVP** | array bridge; curl + E/H (isotropic/diagonal) + CPML; point-dipole source; Energy/Field detectors; Python time loop | ✅ done, element-wise vs JAX-CPU |
| **M2 — sources/detectors/loss** | Uniform/Gaussian **TFSF** plane sources (+ tilted); Poynting + Phasor detectors; electric/magnetic conductivity | ✅ done, element-wise vs JAX-CPU |
| **M3 — full anisotropy** | per-cell analytic 3×3 inverse + 9-tensor A/B update; tensor energy; periodic boundaries | ✅ done, element-wise vs JAX-CPU (uniform grid) + birefringence |
| **M4 — non-uniform grid** | metric-scaled curl + spacing-weighted detector interpolation + **spacing-weighted off-diagonal anisotropic averaging** | ✅ done; iso/diag element-wise vs JAX, off-diagonal average **2nd-order on a graded mesh** (measured slope 2.00 vs 1.00 unweighted) — see [nonuniform-grid.md](nonuniform-grid.md) |

Validation suites: `tests/validation/test_mlx_parity.py` (uniform), `tests/validation/test_mlx_nonuniform.py` (non-uniform + convergence), `tests/visualization/` (birefringence + convergence figures); fdtdx's own physics tests pass auto-routed to MLX.

**Not yet on MLX (gated → JAX):** dispersion (ADE), lossy-anisotropic, 9-tensor conductivity, Bloch/complex propagation, PEC/PMC, mode sources/detectors, gradients. The dispatcher (`src/fdtdx/backend/dispatch.py`) declines these and falls back to the unchanged JAX engine.

### Next on WS-A (performance — currently the engine is eager)
- **`mx.compile` the per-step body.** The time loop is a plain eager Python `for` with periodic `mx.eval`; wrapping the step in `mx.compile` (time-step + amplitude scalars as compiled args, host-side source/detector gating) is the main forward-perf lever and is not yet done.
- **Benchmark Metal vs JAX-CPU/CUDA** on a large full-tensor anisotropic domain (the unified-memory thesis) and profile.

## WS-C — Subpixel smoothing (parallel with WS-A)
Port the Kottke/Farjadpour kernel (`../meep/src/anisotropic_averaging.cpp`, ~150 core lines) + `chi1p1` continuous-ε evaluator; validate vs MEEP. **~2 weeks.** Requires WS-A's tensor path to consume output.

## WS-B — Mode solver (after WS-C)
2D-Yee FD eigensolver (scipy) + overlap + injection port (FDTDX TFSF). Reuses WS-C index averaging.
**~1.5–2 weeks.**

## WS-D — Orchestration (largest, open-ended)
| Piece | Estimate |
|---|---|
| Declarative config (pydantic) + JSON/HDF5 + run pipeline | part of MCP |
| MCP server (introspect options → build setup → run → fetch results) | ~2–4 wk |
| Git-like branch/revert of prompt+tool history (app layer) | ~1–2 wk |
| Web UI + Lumerical-like interactive 3D editor (plotly / pyvista-trame → three.js) | ~4–8 wk |

## Physics extensions (as needed)
- Dispersion (ADE Lorentz/Drude): impl ~3–5 d + val ~3–5 d.
- χ² nonlinear (local NL-polarization term in E-update; forward-only): impl ~3–5 d + SHG val ~1 wk.
- Near-to-far-field (port `../meep/src/near2far.cpp`): self-contained, optional.

## Suggested order
**WS-A ✅ → `mx.compile` perf pass + Metal benchmark → WS-C → WS-B → WS-D (MCP first, then web UI).** Dispersion (ADE) can slot in whenever a use case needs it.

## Strategic note
Upstream FDTDX is being rewritten in **PyTorch** ("The Big Refactor," disc. #349) with a new Tidy3D-like API and **no timeline** (realistically 12–24 mo to parity). PyTorch's MPS backend has no FFT and weak complex support, so it would *not* give good native Metal anyway. This forward MLX engine is independent of that timeline and reusable as the seed of a future MLX backend.
