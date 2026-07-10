# Task 3 — Turn on Apple-GPU support in fdtdx via jax-mps (the "canvas", no custom kernel)

**Type:** small enabling PR to **upstream fdtdx** (or a fork branch first). **Parallelizable:** yes.
**Background:** [`../research/jax-mps-eval.md`](../research/jax-mps-eval.md) (Track 2B/2C/2D),
[`../research/hybrid-and-optimization-plan.md`](../research/hybrid-and-optimization-plan.md) (WS-2).

## Why this exists

`pip install jax-mps` + `JAX_PLATFORMS=mps` already makes **unmodified fdtdx** run on the Apple GPU — forward
(~2× JAX-CPU, no fusion) **and** the reversible-FDTD adjoint (correct gradients, ~CPU-parity). That is a real
capability upstream fdtdx lacks today (on a Mac it runs CPU-only). The goal here is **not** the fork's raw
kernel speed — it is a **robust, low-effort infrastructure ("canvas")** so that:
- fdtdx users on Apple Silicon get GPU forward + **inverse design on a Mac** immediately, and
- once Task 1's kernel-via-`custom_call` lands in jax-mps, fdtdx is *already* running on `mps` and simply
  gets faster — no further fdtdx plumbing needed for the boost.

## What's missing (the one blocker) — and it's small

fdtdx assumes the JAX platform is one of `cpu`/`gpu`/`tpu`/`METAL`. Under jax-mps the platform is **`mps`**,
which fdtdx doesn't know, so config resolution fails ("Unknown backend cpu"/"gpu"). Exactly two spots:
- `src/fdtdx/config.py` `SimulationConfig.__post_init__` — the backend-resolution branch
  (`current_platform`, `self.backend in ["gpu","tpu"]`, the METAL/CPU fallback) doesn't handle `mps`.
- `src/fdtdx/core/jax/sharding.py` `create_named_sharded_matrix` — calls `jax.devices(backend=config.backend)`
  which throws for an `mps`-only runtime.

The research harness proves this is the whole gap: it works with a 2-line shim (make `jax.devices` return the
default device; stop fdtdx repinning `jax_platform_name`) — see the top of
`benchmarks/research/engine_matrix.py` and `inverse_design_adjoint.py`. **Replace that shim with proper
support.**

## Steps
1. Add `mps` as a recognized **single-device** platform: when the active JAX platform is `mps` (or the user
   sets `backend="mps"`), resolve sharding to the single `mps` device and **don't** repin the platform name.
   Keep `cpu`/`gpu`/`tpu`/`METAL` behavior unchanged.
2. Ensure `create_named_sharded_matrix` (and any other `jax.devices(backend=…)` callers) tolerate a
   single-device `mps` runtime.
3. Validate on a Mac with `pip install jax-mps`:
   - forward: `JAX_PLATFORMS=mps JAX_MPS_ASYNC_DISPATCH=1 python benchmarks/research/engine_matrix.py jax isotropic 64,128 100 1`
     (expect ~2× CPU, finite fields), and
   - inverse design: `JAX_PLATFORMS=mps … python benchmarks/research/inverse_design_adjoint.py 48 60`
     (expect correct gradient, FD rel-err ~1e-4).
4. Document the recipe in fdtdx docs: install jax-mps (jax 0.10.x, Python ≥3.13), set `JAX_PLATFORMS=mps`
   (+ `JAX_MPS_ASYNC_DISPATCH=1`); note it's op-graph speed today, kernel-accelerated later (Task 1).

## Deliverable
A small PR making fdtdx run out-of-the-box under the `mps` platform (config + sharding), validated for
forward and gradient runs, with a short docs section. No MLX/fork engine code.

## Caveats
- This is deliberately the **low-effort** path (no hand-rolled kernel). It establishes the canvas; the
  performance boost arrives via Task 1 without further fdtdx changes.
- PyTorch-refactor risk: same as Task 2 — low-regret because the change is tiny and the platform-recognition
  idea ports; and PyTorch's MPS backend lacks FFT/complex, so JAX-fdtdx + jax-mps may stay the better Metal
  path regardless.
- Coordinate the exact `mps`-recognition patch with Task 1 (both touch how fdtdx runs under `mps`).
