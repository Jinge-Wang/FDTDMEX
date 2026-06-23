# tests/

Three tiers (markers in `pyproject.toml`):

- `unit/` — fast, isolated component tests (no time stepping).
- `integration/` — object placement, array bridge, multi-component wiring.
- `validation/` — full forward runs vs analytic results or the FDTDX (JAX) / MEEP oracles (`uv sync --extra validation`).

Run: `uv run pytest` (all) or `uv run pytest -m unit` (subset). See the `physics-validation` skill for the validation methodology (benchmarks, two-run normalization, convergence checks).
