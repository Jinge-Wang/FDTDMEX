# CLAUDE.md

Guidance for Claude Code / agents working in the FDTDMEX repository.

## What this project is

FDTDMEX is a **fork of [fdtdx](https://github.com/ymahlau/fdtdx)** (the JAX FDTD Maxwell solver)
that adds a native **MLX (Metal) forward backend** for Apple Silicon. On a Mac, a forward
`run_fdtd` automatically routes to the MLX time loop; gradients/inverse design and non-Apple
platforms run the unchanged JAX engine. You keep fdtdx's entire front end and **import it the same
way**: `import fdtdx` (the MLX backend is built in). `src/fdtdmex` is a thin brand alias.

The differentiating goal is fast, large *forward* simulations on a single Mac via **unified
memory** — especially full-tensor anisotropic, heterogeneous materials. Inverse design stays on
CUDA/JAX clusters.

## Repo is a fork

- `origin` = `Jinge-Wang/FDTDMEX`; `upstream` = `ymahlau/fdtdx` (added so `git merge upstream/main`
  stays clean and MLX features can be PR'd back). The MLX backend is **additive**: new
  `src/fdtdx/backend/` + `src/fdtdx/mlx/` packages plus a ~4-line guarded hook in
  `src/fdtdx/fdtd/wrapper.py:run_fdtd`. The rest of the tree tracks fdtdx.

## Where the MLX backend lives

- `src/fdtdx/backend/` — `platform` (Apple-Silicon/mlx probes), `dispatch`
  (`maybe_run_mlx_forward`, milestone gating, warn-once JAX fallback), `context` (`use_backend`).
- `src/fdtdx/mlx/` — the forward engine: `bridge` (ArrayContainer↔MLX, host-precomputed
  time-invariant CPML coeffs), `curl`, `update`, `pml`, `interpolate`, `metrics`, `source_freeze`
  + `inject`, `detector_freeze` + `accumulate`, `loop`, and `kernels` (Phase 2 M2: custom Metal E/H
  bulk kernels + spatial-hybrid slab-CPML, behind `FDTDMEX_METAL_KERNEL`; see `ACTION_PLAN.md`).

**Backend control:** auto on Apple Silicon for supported forward runs; force with
`with fdtdx.use_backend("jax"|"mlx")` or `FDTDMEX_BACKEND=jax|mlx`. The custom Metal kernels are an
extra opt-in within the MLX path: `FDTDMEX_METAL_KERNEL=1` (default off until parity-clean across the
surface). Forcing JAX (CPU) is how
validation gets a reference oracle (JAX-Metal is unusable).

## Current state

**M1–M4 implemented and validated element-wise vs JAX-CPU** (`tests/validation/test_mlx_parity.py`,
`tests/validation/test_mlx_nonuniform.py`):
- Engine: curl → E/H update → CPML → source → detector → time loop.
- Materials: isotropic, diagonal-anisotropic, **full-tensor (9-component) anisotropic**,
  electric/magnetic conductivity (lossy).
- Sources: `PointDipoleSource`; `UniformPlaneSource` / `GaussianPlaneSource` (TFSF), including
  **tilted (azimuth/elevation)** beams.
- Detectors: `EnergyDetector`, `FieldDetector`, `PoyntingFluxDetector`, `PhasorDetector`.
- Boundaries: CPML, and **periodic** (wrap-padding, real-valued / Bloch-k0).
- **Non-uniform (rectilinear) grids (M4):** metric-scaled curl, spacing-weighted detector
  interpolation, and **spacing-weighted off-diagonal anisotropic averaging** (2nd-order on graded
  meshes — *more correct* than fdtdx, which leaves that average unweighted). On uniform grids every
  weighted form reduces exactly to the M3 path (verified element-wise).
- fdtdx's own physics tests (plane wave, Fresnel slab, skin depth, birefringence, non-uniform grid)
  pass auto-routed to MLX.

**Deferred → falls back to JAX:** dispersive (ADE) materials; lossy + full-anisotropic together;
full-anisotropic (9-tensor) conductivity; tilted+randomized / dispersive plane sources; mode
sources/detectors; Bloch (nonzero-k) / forced-complex propagation; PEC/PMC boundaries; gradients.
The dispatcher gates all of these; widen the gate as kernels land.

## Commands

```bash
uv sync                       # core (jax + fdtdx stack; mlx auto-installs on Apple Silicon)
uv run python -c "import fdtdx, mlx.core, jax"   # import sanity
uv run --with pytest pytest tests/validation/test_mlx_parity.py -q   # MLX-vs-JAX parity (Mac)
uvx ruff format src/fdtdx/mlx src/fdtdx/backend  # format (ruff is in the dev extra; uvx is handy)
uvx ruff check  src/fdtdx/mlx src/fdtdx/backend
```

## Coding conventions (read before writing kernels)

- **MLX is functional / out-of-place.** Compute a *new* array and return it — this is what makes
  the FDTD update race-free (no ping-pong buffers, no atomics).
- **Mirror fdtdx exactly** so results cross-check element-wise. Port kernels verbatim (watch
  details: JAX clamps OOB integer indexing — isotropic `inv_eps[axis]` → component 0; the H-source
  update samples temporal at the `+0.5` half step). Reuse precomputed host quantities (CPML a/b are
  time-invariant; source `_E/_H`/offsets and detector `init_state` shapes come from the placed
  objects).
- **Yee grid + eta0-normalized H** per fdtdx; don't change conventions silently.
- **Non-uniform grids are supported (M4)** — the MLX curl is metric-scaled, detector
  interpolation and the off-diagonal anisotropic average are spacing-weighted (the latter is
  2nd-order on graded meshes, unlike fdtdx's unweighted average). On uniform grids the weights are
  scalar `1.0` / plain means, so the M3 path is recovered byte-for-byte (the parity bar).
- **Time loop:** plain Python `for` loop; bound the lazy graph with periodic `mx.eval`. It is
  currently eager; wrapping the per-step body in `mx.compile` is a future perf optimization.
- **Validate, don't just smoke-test.** New physics gets a `validation`-marked element-wise parity
  test vs forced-JAX (and/or fdtdx's physics tests auto-routed to MLX). Marginal failure → raise
  resolution, not loosen tolerance. Beware float32 numerical traps (e.g. MLX complex/real division
  underflows for tiny denominators — normalize weights before contracting).

## Reference sources (sibling dirs)

- `../fdtdx` — pristine **MIT** upstream clone (read-only reference; also reachable as the
  `upstream` remote). The fork's own `src/fdtdx/` tracks it.
- `../meep` — **GPL v2+**. Consulted for subpixel smoothing (`src/anisotropic_averaging.cpp`) and
  near-to-far field (`src/near2far.cpp`); record provenance, don't copy GPL into the tree.

**Licensing:** this fork inherits fdtdx's MIT lineage; the project's own additions are provisionally
Apache-2.0 (owner-managed). See [docs/licensing.md](docs/licensing.md).

## Skills

`.claude/skills/` seeds the workflow: `fdtdmex` (framework/physics conventions),
`porting-from-fdtdx` (JAX→MLX recipe, what NOT to port), `physics-validation` (how to validate).
Note these predate the fork pivot and still describe a separate `fdtdmex` package; the recipes
(MLX conventions, array-bridge, validation) still apply — just build inside `src/fdtdx/{backend,mlx}`.
