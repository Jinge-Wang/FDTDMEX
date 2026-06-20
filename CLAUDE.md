# CLAUDE.md

Guidance for Claude Code / agents working in the FDTDMEX repository.

## What this project is

FDTDMEX is a **forward-first, Metal-native FDTD** Maxwell solver built on **MLX** (Apple's array framework) for Apple Silicon. The differentiating goal is fast, large *forward* simulations on a single Mac using **unified memory** — especially **full-tensor anisotropic, heterogeneous** materials whose per-voxel 3×3 tensors are too big for a single CUDA GPU. Inverse design is *not* a goal here (it stays on CUDA/JAX clusters).

Read [docs/architecture.md](docs/architecture.md) for the design and [docs/roadmap.md](docs/roadmap.md) for phasing. **New to MLX/JAX?** [docs/getting-started.md](docs/getting-started.md).

## Current phase

**Pre-implementation scaffold.** Package modules are stubs (docstrings + `NotImplementedError`).
Build order: **WS-A** (forward engine: curl → E/H → CPML → source → detector → time loop) →
**WS-C** (subpixel smoothing) + **WS-B** (mode solver) → **WS-D** (MCP server, web UI).

## Commands

```bash
uv sync --extra dev          # install with dev tooling
uv run pytest                # tests (markers: unit, integration, validation)
uv run pytest -m unit        # fast subset
uv run ruff check            # lint
uv run ruff format           # format
python -c "import fdtdmex"   # import sanity check
```

## Coding conventions (read before writing kernels)

- **MLX is functional / out-of-place.** Compute a *new* array and return it. Never depend on in-place mutation for correctness — this is exactly what makes the FDTD update **race-free** (no ping-pong buffers, no atomics). See [docs/porting-notes.md](docs/porting-notes.md).
- **Yee grid + field normalization** follow [docs/physics.md](docs/physics.md) (component layout, eta0-normalized H, Courant number). Do not change conventions silently. 
- **Non-uniform grids are first-class.** Every curl / interpolation / update **must** accept and use per-axis Yee cell-size arrays (primal/dual spacings) and do spacing-weighted interpolation — not the unweighted average FDTDX uses. Thread spacing through the API from the start. See [docs/nonuniform-grid.md](docs/nonuniform-grid.md).
- **Complex fields** (Bloch, phasor) use `complex64`; MLX supports complex arrays + complex FFT.
- **Time loop:** plain Python `for` loop over steps with the per-step body wrapped in `mx.compile`; call `mx.eval` periodically to bound the lazy graph.
- **Validate, don't just smoke-test.** New physics gets a `validation`-marked test vs. an analytic result or the FDTDX/MEEP reference. If a physics test fails marginally, raise resolution rather than loosen tolerances.

## Reference sources (sibling dirs, not vendored)

- `../fdtdx` — **MIT**. The JAX solver this project ports. Key files: `src/fdtdx/fdtd/update.py` (E/H updates incl. full-anisotropic path), `core/physics/curl.py`, `fdtd/misc.py` (`compute_anisotropic_update_matrices`, off-diagonal averaging), `objects/boundaries/perfectly_matched_layer.py` (CPML), `objects/sources/tfsf.py`+`mode.py` (mode injection), `dispersion.py` (ADE), `fdtd/initialization.py` (how material arrays are built — the array-bridge reference).
- `../meep` — **GPL v2+**. Consulted for `src/anisotropic_averaging.cpp` (subpixel smoothing, WS-C) and `src/near2far.cpp` (near-to-far field). Also `libpympb/` (MPB mode solver, WS-B cross-check).

**Licensing:** the project license (Apache-2.0) is a provisional placeholder and is **owner-managed** — you may freely read and port from both references, including translating MEEP kernels. Just note in the commit which reference file you adapted. See [docs/licensing.md](docs/licensing.md).

## Skills

`.claude/skills/` seeds the workflow: `fdtdmex` (framework/physics conventions),
`porting-from-fdtdx` (JAX→MLX recipe, what NOT to port), `physics-validation` (how to validate).
