# Architecture

## Goal & scope

FDTDMEX is a **forward** FDTD Maxwell solver that runs on **MLX/Metal** on a single Apple-Silicon machine. It is a fork of [fdtdx](https://github.com/ymahlau/fdtdx): the entire front end — geometry, GDS import, constraints, materials, sources, detectors, boundaries — is fdtdx's, and the MLX backend is purely additive. On a Mac, a supported forward `run_fdtd` runs the time loop natively on the GPU; everything else runs the unchanged JAX engine.

The thesis is **memory, not just compute**: Apple unified memory (up to 512 GB) lets the GPU address a whole large, full-tensor anisotropic domain — whose per-voxel 3x3 ε tensors (~9x the isotropic footprint) overflow or thrash a single CUDA GPU's VRAM/PCIe.

**Out of scope:** gradient-based inverse design on Metal. That needs cluster-scale parallelism and stays on JAX/CUDA, so the MLX backend is forward-only — a large simplification (no reversible-gradient machinery to port).

## Data flow

```
            ┌─────────────── host (CPU / numpy) ────────────────┐       ┌──── Metal GPU (MLX) ─────┐
 config ──► │ geometry → voxelization → subpixel smoothing      │       │  forward time loop:      │
            │ → material tensor arrays (1/3/9-component)        │  ───► │   curl → E/H update      │
            │ PML profiles, source temporal profiles            │ bridge│   → CPML → source inject │
            │ mode solve (finite-difference eigensolver, host)  │ np→mx │   → detector accumulate  │
            └───────────────────────────────────────────────────┘       └──────────────────────────┘
                                                                              │
                                                            results (fields, flux, phasors) ──► viz / HDF5
```

Everything left of the bridge is host-side and reuses fdtdx's front end unchanged. Everything right of the bridge is the MLX time loop FDTDMEX owns. The fields and `detector_states` that come back are the same arrays fdtdx would produce, so all downstream code (detector reading, plotting, S-parameters) is identical.

## Where the code lives

The fork's `src/fdtdx/` tracks upstream fdtdx; the MLX backend is two added packages plus a small guarded hook, so `git merge upstream/main` stays clean.

| Path | Role |
|---|---|
| `src/fdtdx/backend/` | Platform/MLX probes and the routing decision (`maybe_run_mlx_forward`, feature gating, warn-once JAX fallback), and `use_backend`. |
| `src/fdtdx/mlx/` | The forward engine: array bridge, curl, E/H update, CPML, source injection, detector accumulation, the time loop, and the custom Metal E/H kernels. |
| `src/fdtdx/core/physics/mode_backend/`, `core/physics/subpixel.py` | Native full-vectorial mode solver and Kottke subpixel smoothing. |
| `src/fdtdx/fdtd/wrapper.py` | Upstream's `run_fdtd`, with the ~4-line guarded hook that routes a supported forward run to MLX. |
| `src/fdtdmex/` | Thin brand alias re-exporting `fdtdx`, plus `io/` — the `sim_init` / `sim_run` / `sim_postproc` HDF5 hand-off. |

## Key design decisions
- **Forward-only, no on-device autodiff** → no reversible gradient, `custom_vjp`, or checkpointing.
- **Functional / out-of-place updates** → race-free without ping-pong buffers; each step computes a new field from the old one and returns it.
- **Spacing-weighted operators** → 2nd-order accurate on non-uniform grids by construction.
- **Host/GPU split at a plain-array bridge** → reuse fdtdx's mature front end; own only the hot loop.

The rationale behind the forward-first split is recorded in [`dev-docs/decisions/`](../dev-docs/decisions/).
