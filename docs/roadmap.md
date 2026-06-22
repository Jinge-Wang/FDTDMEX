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

**Not yet on MLX (gated → JAX):** Bloch/complex propagation, mode sources/detectors, dispersive/randomized plane sources, gradients. The dispatcher (`src/fdtdx/backend/dispatch.py`) declines these and falls back to the unchanged JAX engine. *(Phase 3 added lossy-anisotropic + 9-tensor conductivity, PEC/PMC, and Drude–Lorentz ADE dispersion — now supported.)*

### WS-A performance — Phase 1 ✅, Phase 2 M1 ✅ + M2 ✅ + M3 ✅, Phase 3 ✅
- **Status:** custom Metal E/H kernels run the forward loop and are **default-on** (`src/fdtdx/mlx/kernels.py`; `FDTDMEX_METAL_KERNEL=0` forces the MLX-op cores). M3 folded CPML into the kernel (N=192 iso CPML-on 374 → **1826 Mcs/s / 5 RT**, at the bandwidth floor; diagonal 1711), added in-kernel non-uniform metric, and a block hybrid that keeps the kernel on the diagonal bulk around compact full-tensor inclusions (N=128 8³ inclusion 125 → 1124 Mcs/s). Physics exact (element-wise vs JAX). Ineligible cases (lossy, scattered/oversized tensor, gradients) fall back to the MLX-op cores via `kernel_eligible`.
- **Status:** Phases 1–3 are **complete**. Phase 3 broadened the supported surface — lossy full-anisotropic + 9-tensor conductivity, PEC/PMC, and **Drude–Lorentz ADE dispersion** (Drude/Lorentz only; Debye is absent upstream) — the last with the per-pole ADE recurrence folded into the Metal E-kernel, so dispersive media also ride the bandwidth floor (1216 vs 255 Mcs/s on the MLX-op cores; non-dispersive kernel byte-identical). See [ACTION_PLAN.md](../ACTION_PLAN.md), [performance.md](performance.md) (roofline + results + history), [phase2-metal-kernels.md](phase2-metal-kernels.md) (kernel spec).

## WS-C — Subpixel smoothing (parallel with WS-A)
Port the Kottke/Farjadpour kernel (`../meep/src/anisotropic_averaging.cpp`, ~150 core lines) + `chi1p1` continuous-ε evaluator; validate vs MEEP. **~2 weeks.** Requires WS-A's tensor path to consume output.

## WS-B — Mode solver (Tidy3D-free; coupled with WS-C) — **a current Phase-4 track**
2D-Yee full-vectorial FD eigensolver (scipy, Zhu & Brown 2002) + overlap, built as a **swappable
backend** that drops in behind fdtdx's mode call — fdtdx routes *all* mode work to **Tidy3D**
(`core/physics/modes.py`, the only Tidy3D coupling in the stack), so an own solver removes that
dependency. Injection/`ModePlaneSource`/`ModeOverlapDetector` kept as-is. Reuses WS-C index averaging
(Zhu–Brown's interface averaging *is* subpixel smoothing). Validate on physics (analytic/MPB), not
byte-parity. **~1.5–2 weeks.** See [mode-solver.md](mode-solver.md).

## WS-D — Orchestration / agentic workspace (largest, open-ended) — **a current Phase-4 track**
The **HDF5 wrap/unwrap contract + MCP server** start now against the resolved-arrays seam (a mocked
backend is fine), in parallel with WS-B; the web 3D editor is the long tail. See [mcp-and-ui.md](mcp-and-ui.md).

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
| **Drude-Lorentz dispersion (ADE)** | [`dispersion.py`](../src/fdtdx/dispersion.py) + `fdtd/update.py` ADE block | ✅ **done (Phase 3)**: c1/c2/c3 precomputed host-side and carried in `MLXState`; `P_curr`/`P_prev` threaded through the E-side of the loop; ADE term in both `mlx/update.py` (MLX-op) and folded into the Metal E-kernel (`mlx/kernels.py` `_ade_lines`). Iso/diagonal only (fdtdx forbids off-diagonal + dispersive). Parity in `test_mlx_dispersion.py` |
| **Lossy (conductive) full-anisotropic + 9-tensor conductivity** | the 9-tensor A/B update already takes `sigma` | ✅ **done (Phase 3)**: `mlx/aniso.py` A/B path already consumes σ; un-gated in `dispatch.py`; parity in `test_mlx_lossy_aniso.py`. Runs on the MLX-op cores (lossless kernel falls back via `kernel_eligible`) |
| **PEC / PMC boundaries** | `objects/boundaries/{pec,pmc}.py` | ✅ **done (Phase 3)**: frozen tangential keep-masks (`mlx/boundary_mask.py`) applied post-injection in `loop.py`, composing with the Metal kernel + MLX-op cores; un-gated; parity + tangential-zero in `test_mlx_pec_pmc.py` |

## Genuinely new physics (needs MEEP reference or new derivation)
- **Subpixel smoothing** — WS-C; fdtdx lacks it (Kottke/Farjadpour, `../meep/src/anisotropic_averaging.cpp`).
- **Near-to-far-field** — port `../meep/src/near2far.cpp`; self-contained, optional.
- **χ² nonlinear** (local NL-polarization term in the E-update; forward-only): impl ~3–5 d + SHG validation ~1 wk.
- **Anisotropic + dispersive media (low priority, future — needs a MEEP-style derivation).** Upstream fdtdx forbids the combination (`NotImplementedError`), so our ADE port is iso/diagonal only and there is **no fdtdx parity oracle** for the anisotropic case — it requires a new derivation validated against MEEP. **State-of-the-art reference — MEEP supports it:** each Lorentz/Drude pole carries a full **symmetric 3×3 susceptibility tensor** `σ` (`sigma_diag` + `sigma_offdiag`), and the ADE polarization update for component `c` sums the tensor-weighted contributions of all three field components, the off-diagonal ones **Yee-averaged** to `c`'s location (the same averaging our non-dispersive 9-tensor update already does). Code: [`../meep/src/susceptibility.cpp`](../../meep/src/susceptibility.cpp) `lorentzian_susceptibility::update_P` (isotropic / 2×2 / 3×3 paths via the `OFFDIAG` Yee-average macro; stability hack from MEEP PR #666); API: `Susceptibility`/`LorentzianSusceptibility` (`sigma_diag`/`sigma_offdiag`) in [`../meep/python/geom.py`](../../meep/python/geom.py); docs: MEEP *Material Dispersion* (`Materials.md#material-dispersion`). MEEP additionally has `GyrotropicLorentzianSusceptibility` (gyrotropy models `GYROTROPIC_LORENTZIAN/DRUDE/SATURATED`) for magneto-optic media with an **antisymmetric** bias-field coupling — a distinct, more specialized anisotropic-dispersive form. **Motivating case:** lithium niobate (anisotropic + dispersive + χ²) — implementing this cleanly would be a FDTDMEX differentiator; forward-only keeps it tractable (no autodiff entanglement). Approach if pursued: extend the iso/diagonal ADE recurrence so each pole's `c3` becomes a 3×3 tensor with Yee-averaged off-diagonal E reads (mirroring MEEP), validate vs MEEP rather than fdtdx.

## Suggested order
**[../ACTION_PLAN.md](../ACTION_PLAN.md) — Phases 1–3 complete.** Phase 1 (fix the eager plateau: drop no-op elementwise → `mx.compile` → remove per-step padding → slab CPML; ~4× measured headroom) ✅ → Phase 2 (deep-GPU feasibility: coalescing / layout / hand-rolled Metal kernel) ✅ → Phase 3 (broaden: lossy-aniso, PEC/PMC, ADE dispersion) ✅. The forward engine is fast and broad; the next frontier is the genuinely-new-physics work-streams below.

Longer arc: **WS-A ✅ → Metal perf Phase 1 ✅ → Phase 2 ✅ → Phase 3 (surface widening) ✅ → Phase 4 (now): two parallel tracks — [WS-B mode solver](mode-solver.md) (Tidy3D-free) + [WS-C subpixel smoothing](subpixel-smoothing.md) as one physics track, and the [WS-D agentic workspace](mcp-and-ui.md) (HDF5 + MCP) as the other → then the web UI long tail.** Bloch/complex (nonzero-k) is the one remaining same-port engine feature, demand-driven; gradients stay out of scope (forward-only).

## Exposed potential issues (future exploration, after WS-D)

Open robustness items surfaced while building the engine. **Both reproduce in pure JAX** (`fdtdx.use_backend("jax")`), so they are upstream fdtdx behavior, *not* MLX-port bugs — but they bound what the MLX engine can be trusted with and are worth a dedicated stability study later.

- **Quirk A — strongly off-diagonal (9-tensor) anisotropy is unstable.** A full-tensor permittivity with a large off-diagonal element (e.g. a uniaxial crystal with optic axis rotated 45° in the x–z plane, `ε ≈ ((4.0,0,1.755),(0,2.25,0),(1.755,0,4.0))`) **diverges to NaN even at Courant 0.3**. A small off-diagonal (≈0.3–0.5) is stable. Likely the unweighted/weighted off-diagonal averaging and/or the explicit per-cell A/B update is not unconditionally stable for strong coupling. *Investigate:* von-Neumann-style stability analysis of the anisotropic update; whether a symmetrized average or a sub-Courant factor restores stability. Forced the birefringence demo to use a *diagonal* uniaxial crystal with oblique incidence instead of optic-axis walk-off.
- **Quirk B — finite-aperture `GaussianPlaneSource` is unstable.** A Gaussian sized with a *partial* transverse aperture (`partial_real_shape` smaller than the domain) **NaNs**, while a **full-aperture** Gaussian (`same_size(vol, axes=(0,1))`, `radius` controlling the width) is stable; certain thin/periodic transverse dimensions also went unstable. *Investigate:* the TFSF profile construction / energy normalization in `objects/sources/linear_polarization.py:apply` for partial apertures and thin dimensions.

Reproductions: vary the crystal off-diagonal / source aperture in `tests/visualization/test_birefringence_visual.py` and force JAX.

## Strategic note
Upstream FDTDX is being rewritten in **PyTorch** ("The Big Refactor," disc. #349) with a new Tidy3D-like API and **no timeline** (realistically 12–24 mo to parity). PyTorch's MPS backend has no FFT and weak complex support, so it would *not* give good native Metal anyway. This forward MLX engine is independent of that timeline and reusable as the seed of a future MLX backend.
