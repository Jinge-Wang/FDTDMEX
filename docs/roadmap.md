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

### WS-A performance — Phase 1 complete
- **Status:** MLX/Metal now leads JAX-CPU for all N ≥ 64 (1.25–1.4×) with no plateau; default path 277 Mcs/s / 36 RT at N=192 iso (2.6× the original engine). Pad-free slice-diff curl + `mx.compile`d E/H cores + slab-CPML landed; physics exact.
- **Next:** Phase 2 — custom Metal update kernels (M1 go/no-go). See [ACTION_PLAN.md](../ACTION_PLAN.md), [performance.md](performance.md) (roofline + results), [phase2-metal-kernels.md](phase2-metal-kernels.md) (kernel spec).

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

## Widening the MLX surface — already in fdtdx, just not ported yet (demand-driven)

These features **already exist in upstream fdtdx's JAX engine**; the dispatcher (`src/fdtdx/backend/dispatch.py`) only gates them to JAX because the MLX *kernel* isn't ported. Adding each is the **same JAX→MLX port pattern as M1–M4** (translate the kernel, precompute any time-invariant coefficients on the host, add an element-wise parity test, un-gate) — **no MEEP needed**. They don't depend on or block WS-B/C/D; slot any of them in whenever a use case demands it.

| Feature | Where it lives in fdtdx | MLX port effort |
|---|---|---|
| **Drude-Lorentz dispersion (ADE)** | [`dispersion.py`](../src/fdtdx/dispersion.py) + `fdtd/update.py` ADE block | low–medium: c1/c2/c3 are time-invariant → precompute on host like CPML; carry `P_curr`/`P_prev` in `MLXState`; add one `E += inv_eps·Σ(P_curr − P_new)` term; non-dispersive cells have c3=0 so it's inert elsewhere |
| **Lossy (conductive) full-anisotropic** | the 9-tensor A/B update already takes `sigma` | low: `mlx/aniso.py` already has the A/B path; thread the σ tensor through `compute_anisotropic_update_matrices_mlx` and un-gate |
| **PEC / PMC boundaries** | `objects/boundaries/{pec,pmc}.py` | low: a per-step field-masking pass in the MLX loop + un-gate |

## Genuinely new physics (needs MEEP reference or new derivation)
- **Subpixel smoothing** — WS-C; fdtdx lacks it (Kottke/Farjadpour, `../meep/src/anisotropic_averaging.cpp`).
- **Near-to-far-field** — port `../meep/src/near2far.cpp`; self-contained, optional.
- **χ² nonlinear** (local NL-polarization term in the E-update; forward-only): impl ~3–5 d + SHG validation ~1 wk.

## Suggested order
**Active plan: [../ACTION_PLAN.md](../ACTION_PLAN.md) — Metal forward performance is the current top priority.** Phase 1 (fix the eager plateau: drop no-op elementwise → `mx.compile` → remove per-step padding → slab CPML; ~4× measured headroom) → Phase 2 (deep-GPU feasibility: coalescing / layout / hand-rolled Metal kernel) → Phase 3 (broaden: lossy-aniso, PEC/PMC, ADE — low effort).

Longer arc: **WS-A ✅ → Metal perf Phase 1 ✅ → Metal perf Phase 2 ([ACTION_PLAN.md](../ACTION_PLAN.md)) → WS-C → WS-B → WS-D (MCP first, then web UI).** The "already in fdtdx" features are cheap and demand-driven — pull any forward whenever a use case needs it.

## Exposed potential issues (future exploration, after WS-D)

Open robustness items surfaced while building the engine. **Both reproduce in pure JAX** (`fdtdx.use_backend("jax")`), so they are upstream fdtdx behavior, *not* MLX-port bugs — but they bound what the MLX engine can be trusted with and are worth a dedicated stability study later.

- **Quirk A — strongly off-diagonal (9-tensor) anisotropy is unstable.** A full-tensor permittivity with a large off-diagonal element (e.g. a uniaxial crystal with optic axis rotated 45° in the x–z plane, `ε ≈ ((4.0,0,1.755),(0,2.25,0),(1.755,0,4.0))`) **diverges to NaN even at Courant 0.3**. A small off-diagonal (≈0.3–0.5) is stable. Likely the unweighted/weighted off-diagonal averaging and/or the explicit per-cell A/B update is not unconditionally stable for strong coupling. *Investigate:* von-Neumann-style stability analysis of the anisotropic update; whether a symmetrized average or a sub-Courant factor restores stability. Forced the birefringence demo to use a *diagonal* uniaxial crystal with oblique incidence instead of optic-axis walk-off.
- **Quirk B — finite-aperture `GaussianPlaneSource` is unstable.** A Gaussian sized with a *partial* transverse aperture (`partial_real_shape` smaller than the domain) **NaNs**, while a **full-aperture** Gaussian (`same_size(vol, axes=(0,1))`, `radius` controlling the width) is stable; certain thin/periodic transverse dimensions also went unstable. *Investigate:* the TFSF profile construction / energy normalization in `objects/sources/linear_polarization.py:apply` for partial apertures and thin dimensions.

Reproductions: vary the crystal off-diagonal / source aperture in `tests/visualization/test_birefringence_visual.py` and force JAX.

## Strategic note
Upstream FDTDX is being rewritten in **PyTorch** ("The Big Refactor," disc. #349) with a new Tidy3D-like API and **no timeline** (realistically 12–24 mo to parity). PyTorch's MPS backend has no FFT and weak complex support, so it would *not* give good native Metal anyway. This forward MLX engine is independent of that timeline and reusable as the seed of a future MLX backend.
