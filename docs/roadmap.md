# Roadmap

Estimates are part-time, AI-paired. Code generation is cheap; **physics validation and Metal perf are the long pole**. Status as of repo creation: **pre-implementation scaffold**.

## WS-A — Forward MLX engine

| Milestone | Scope | Impl | Validation |
|---|---|---|---|
| **MVP** | array bridge; curl + E/H (isotropic/diagonal) + CPML; 1 source; field/Poynting detector; Python+`mx.compile` loop | ~2–3 d | slab + waveguide + PML reflection + Courant: ~3–5 d |
| **Full anisotropy** | per-cell 3×3 solve + spacing-weighted off-diagonal interpolation | ~1–2 d | birefringence/walk-off + element-wise vs FDTDX: ~3–5 d |
| **Non-uniform grid** | spacing-weighted curl & interpolation throughout | folded into above | 2nd-order convergence on graded mesh |

→ Validated full-anisotropic forward engine: **~2–3 weeks elapsed**.

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
**WS-A (MVP → anisotropic) + WS-C → WS-B → WS-D (MCP first, then web UI).**

## Strategic note
Upstream FDTDX is being rewritten in **PyTorch** ("The Big Refactor," disc. #349) with a new Tidy3D-like API and **no timeline** (realistically 12–24 mo to parity). PyTorch's MPS backend has no FFT and weak complex support, so it would *not* give good native Metal anyway. This forward MLX engine is independent of that timeline and reusable as the seed of a future MLX backend.
