# FDTDMEX Documentation

Forward-first, Metal-native FDTD electromagnetics on Apple Silicon (MLX).

## Start here
- [Getting started](getting-started.md) — dependencies, environment setup, MLX↔JAX mental model.
- [Architecture](architecture.md) — design, data flow, the four workstreams, dependency graph.
- [Roadmap](roadmap.md) — phasing, milestones, effort estimates.

## Engine & physics
- [Physics](physics.md) — Yee grid, update equations, field normalization, Courant.
- [Non-uniform grids](nonuniform-grid.md) — spacing-weighted curl & interpolation.
- [Materials & anisotropy](materials-anisotropy.md) — full-tensor heterogeneous ε/µ, dispersion, χ².
- [Subpixel smoothing](subpixel-smoothing.md) — Kottke/Farjadpour effective-tensor averaging (WS-C).
- [Mode solver](mode-solver.md) — 2D-Yee FD eigensolver, overlap, injection (WS-B).
- [Porting notes](porting-notes.md) — FDTDX (JAX) → FDTDMEX (MLX).

## Orchestration & process
- [MCP server & UI](mcp-and-ui.md) — declarative config, MCP tools, git-like history, web 3D editor (WS-D).
- [Licensing](licensing.md) — Apache-2.0 (provisional), references, owner-managed reconciliation.
- [Decisions](decisions/) — architecture decision records (ADRs).
