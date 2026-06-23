# AGENTS.md

This file exists for agent tools that look for `AGENTS.md`. The authoritative agent guidance for this repository lives in **[CLAUDE.md](CLAUDE.md)** — read it first.

Quick orientation:
- **Project:** FDTDMEX — forward-first, Metal-native FDTD on MLX (Apple Silicon), built as a **git fork of fdtdx**. See [README.md](README.md) and [docs/architecture.md](docs/architecture.md).
- **Phase:** the forward MLX engine is **complete and fast** — WS-A physics (iso/diag/full-tensor anisotropy, conductivity, CPML + periodic + PEC/PMC, dipole + TFSF sources, all four detectors, non-uniform grids) plus performance Phases 1–3 (custom Metal kernels at the bandwidth floor, default-on) and **Drude–Lorentz ADE dispersion** — all validated element-wise vs JAX. **Next is Phase 4** (see [ACTION_PLAN.md](dev-docs/ACTION_PLAN.md), the single entry point): two parallel tracks — a Tidy3D-free mode solver + subpixel smoothing, and the agentic workspace (HDF5 + MCP) — plus Bloch/complex. See also [docs/roadmap.md](dev-docs/roadmap.md).
- **Conventions:** MLX functional/out-of-place (race-free); non-uniform grids first-class (spacing-weighted); validate against FDTDX/MEEP references. Details in [CLAUDE.md](CLAUDE.md).
- **References:** the fork's own `src/fdtdx/` tracks **upstream** `ymahlau/fdtdx` (MIT; also the sibling `../fdtdx` read-only clone); `../meep` (GPL) is consulted for smoothing/N2F. Licensing is owner-managed — porting from either is fine; see [docs/licensing.md](docs/licensing.md).
