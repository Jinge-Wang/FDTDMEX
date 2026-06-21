# AGENTS.md

This file exists for agent tools that look for `AGENTS.md`. The authoritative agent guidance for
this repository lives in **[CLAUDE.md](CLAUDE.md)** — read it first.

Quick orientation:
- **Project:** FDTDMEX — forward-first, Metal-native FDTD on MLX (Apple Silicon), built as a **git fork of fdtdx**. See [README.md](README.md) and [docs/architecture.md](docs/architecture.md).
- **Phase:** WS-A (forward MLX engine) **complete through M1–M4** — iso/diag/full-tensor anisotropy, conductivity, CPML + periodic boundaries, dipole + TFSF sources, all four detectors, and non-uniform (spacing-weighted) grids, validated element-wise vs JAX. Next: `mx.compile` perf pass, then WS-C/WS-B/WS-D. See [docs/roadmap.md](docs/roadmap.md).
- **Conventions:** MLX functional/out-of-place (race-free); non-uniform grids first-class (spacing-weighted); validate against FDTDX/MEEP references. Details in [CLAUDE.md](CLAUDE.md).
- **References:** the fork's own `src/fdtdx/` tracks **upstream** `ymahlau/fdtdx` (MIT; also the sibling `../fdtdx` read-only clone); `../meep` (GPL) is consulted for smoothing/N2F.
  Licensing is owner-managed — porting from either is fine; see [docs/licensing.md](docs/licensing.md).
