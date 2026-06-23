# FDTDMEX Documentation

Forward-first, Metal-native FDTD electromagnetics on Apple Silicon (MLX).

## Start here
- [Getting started](getting-started.md) — install, environment, confirming the Metal GPU, and the MLX↔JAX mental model.
- [Architecture](architecture.md) — how the backend routes, the host/GPU split, and where the code lives.

## Engine & physics
- [Physics & conventions](physics.md) — Yee grid, update equations, field normalization, stability.
- [Non-uniform grids](nonuniform-grid.md) — spacing-weighted curl, interpolation, and anisotropic averaging.
- [Materials & anisotropy](materials-anisotropy.md) — full-tensor heterogeneous ε/µ and Drude–Lorentz dispersion.
- [Subpixel smoothing](subpixel-smoothing.md) — Kottke/Farjadpour effective-tensor averaging.
- [Mode solver](mode-solver.md) — native full-vectorial finite-difference mode solver and overlap.

## Performance & orchestration
- [Performance](performance.md) — scaling vs JAX-CPU, the roofline, and the bandwidth-floor model.
- [MCP server & web UI](mcp-and-ui.md) — declarative config and the portable HDF5 hand-off.
- [Licensing](licensing.md) — Apache-2.0 (provisional), references, owner-managed reconciliation.

## Contributing
Developer and process material — the roadmap, porting recipes, Metal-kernel internals, the action plan, and architecture decision records — lives in [`dev-docs/`](../dev-docs/).
