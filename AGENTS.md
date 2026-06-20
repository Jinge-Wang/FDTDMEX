# AGENTS.md

This file exists for agent tools that look for `AGENTS.md`. The authoritative agent guidance for
this repository lives in **[CLAUDE.md](CLAUDE.md)** — read it first.

Quick orientation:
- **Project:** FDTDMEX — forward-first, Metal-native FDTD on MLX (Apple Silicon). See [README.md](README.md) and [docs/architecture.md](docs/architecture.md).
- **Phase:** pre-implementation scaffold; modules are stubs. Build order WS-A → WS-C/WS-B → WS-D.
- **Conventions:** MLX functional/out-of-place (race-free); non-uniform grids first-class (spacing-weighted); validate against FDTDX/MEEP references. Details in [CLAUDE.md](CLAUDE.md).
- **References (not vendored):** `../fdtdx` (MIT, port source), `../meep` (GPL, smoothing/N2F).
  Licensing is owner-managed — porting from either is fine; see [docs/licensing.md](docs/licensing.md).
