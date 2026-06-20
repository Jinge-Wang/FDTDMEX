# Contributing to FDTDMEX

This is an early-stage, owner-driven project developed largely via AI pair-programming ("vibe coding"). These notes keep contributions — human or agent — consistent.

## Environment

```bash
uv sync --extra dev
uv run pytest          # run tests
uv run ruff check      # lint
uv run ruff format     # format
```

Apple Silicon + recent macOS required for the MLX/Metal path. See [docs/getting-started.md](docs/getting-started.md).

## Ground rules

- **Match the physics conventions** in [docs/physics.md](docs/physics.md) (Yee staggering, field normalization, Courant number). Do not silently change conventions.
- **MLX is functional / out-of-place.** Compute a new array and return it; never rely on in-place mutation for correctness. This is what makes the update race-free (see [docs/porting-notes.md](docs/porting-notes.md)).
- **Non-uniform grids are first-class.** Any new curl/interpolation/update must accept and use the per-axis Yee cell-size arrays — no hard-coded uniform spacing. See [docs/nonuniform-grid.md](docs/nonuniform-grid.md).
- **Validate, don't just smoke-test.** New physics needs a `validation`-marked test comparing to an analytic result or to the FDTDX (JAX) / MEEP reference. Marginal failures → raise resolution, don't loosen tolerances.
- **Porting from references is fine.** You may read and adapt `../fdtdx` (MIT) and `../meep` (GPL). Licensing reconciliation is owner-managed (see [docs/licensing.md](docs/licensing.md)); just note in the commit/PR when code is derived from a reference and from which file.

## Commit / PR

- Keep commits focused; describe what changed and why.
- Reference the workstream (WS-A…WS-D) and any reference file you ported from.
