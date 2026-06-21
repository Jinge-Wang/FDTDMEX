# ADR 0001 — MLX, forward-first, host front end

- **Status:** Accepted
- **Date:** 2026-06-20

## Context

We want native Apple-Silicon (Metal) FDTD. JAX's Metal path is unusable (no JIT on macOS). FDTDX (JAX) is the most capable open differentiable FDTD but is being rewritten in PyTorch (disc. #349, no timeline), and PyTorch's MPS backend lacks FFT and has weak complex support — so PyTorch would not give good native Metal either. The user's need is **forward** simulation on a **single Mac**; inverse design stays on CUDA/JAX clusters.

## Decision

1. **Build on MLX**, not a JAX-Metal shim or waiting for the PyTorch rewrite. MLX has a JAX-like functional core, complex64 + complex FFT, and unified memory.
2. **Forward-only on Metal.** Do not port the reversible/`custom_vjp` gradient or checkpointing. This removes the single hardest porting blocker.
3. **Host/GPU split at a plain-array bridge.** Reuse FDTDX's mature CPU front end (geometry, constraints, GDS, PML profiles, source profiles); own only the MLX hot loop (~1.5–3k lines).
4. **Functional/out-of-place updates** for race-freedom without ping-pong buffers/atomics.
5. **Non-uniform grids are first-class** (spacing-weighted operators), improving on FDTDX.
6. **Mode solver written in-house** (host scipy eig), avoiding MLX's missing complex GPU eig and the Tidy3D dependency.

## Consequences

- No gradient-based inverse design on Metal (acceptable; that's cluster work).
- Time loop is a Python loop + `mx.compile` (no traced `scan`/`while_loop`); manage the lazy graph with periodic `mx.eval`.
- The engine is reusable as the seed of a future MLX backend for the PyTorch FDTDX, but does not depend on that timeline.
- Unified memory is the strategic advantage, especially for large full-anisotropic tensor fields.

## 2026-06-20 — Addendum: adopted as a fork (supersedes "not vendored")

Decision #3 above ("reuse FDTDX's front end via a plain-array bridge") was realized by making this
repo a **fork of `ymahlau/fdtdx`** rather than a separate `fdtdmex` package consuming a sibling.
fdtdx's history is grafted in (`upstream = ymahlau/fdtdx`), the MLX backend lives **inside the
package** at `src/fdtdx/{backend,mlx}`, and `run_fdtd` carries a ~4-line guarded hook that routes
supported forward-only runs to MLX on Apple Silicon (override via `fdtdx.use_backend` /
`FDTDMEX_BACKEND`). The plain-array bridge is now in-process (`src/fdtdx/mlx/bridge.py`,
ArrayContainer↔MLX). `src/fdtdmex` is a thin alias. This keeps element-wise cross-checks against the
JAX reference and lets MLX work flow back upstream. Everything else in this ADR stands.
