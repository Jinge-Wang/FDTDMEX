# Plan: hybrid integration, forward-engine optimization, and upstream contributions

Derived from the measured study in [`jax-mps-eval.md`](jax-mps-eval.md). Four workstreams, prioritized.
The guiding conclusion: **keep the fused forward kernel (it wins 5–13× where it has a kernel), add
jax-mps for Mac inverse design, and pursue a `custom_call`-registered kernel as the eventual "fast +
differentiable + one codebase" unification.** Anisotropic coverage exposed the fork's real weak spot
(full-tensor), which reorders the optimization priorities.

---

## WS-1 — Forward engine: close the anisotropic gap, then beat the floor (highest ROI)

Measured regimes (M4 Pro, N=128): iso fused **1659** Mcs/s, diagonal **1080**, **full_aniso only ~99
(~1.4× CPU)** because the fused kernel has *no in-kernel 9-tensor path* and falls back to MLX-op cores.
The spill gate is positive (~35% throughput loss N=192→512). So there are two ranked levers, both from
`performance-roadmap.md` but now empirically ordered:

1. **In-kernel full-tensor (and per-tile material) path — do first.** Biggest measured gap: full-tensor
   anisotropic gets ~1.4× CPU vs ~13× for iso. The roadmap's **per-tile material compaction** (§5.2) is
   the fix — a threadgroup tile carries one descriptor (scalar / diagonal / full tensor, same machinery),
   so a uniform-tensor domain stops paying MLX-op-core prices and rides a fused kernel. This also retires
   the block-hybrid carve-out. Target: bring uniform full_aniso from ~99 toward the diagonal band.
2. **Tiled sub-floor engine (temporal blocking + spatial tile) — do second.** Spill gate justifies the
   spatial tile at large N; temporal blocking (§5.1) is the only sub-floor lever. ~2–3× confident on the
   heterogeneous/tensor regimes, 5–6× stretch at depth 3–4. Ship as one unit with (1) per §5.
3. **Gate + parity** at each material class and depth `T` against forced-JAX-CPU (`rel<1e-3`), exactly as
   the existing kernel does. Limits found: throughput is bandwidth-bound (peak ~1400 iso / ~1300 diagonal
   at N≈192, spilling ~35% by N=512); capacity-bound at ~N≈900 iso on 52 GB unified (double-buffered).

Deliverable stays inside the fork (`src/fdtdx/mlx/kernels.py` + the tile infra). Forward-only, so **not**
autodiff-constrained; not upstreamable (Apple-specific) — this is the fork's core value.

## WS-2 — Hybrid: jax-mps inverse design on a Mac (small, near-term, high value)

jax-mps runs unmodified fdtdx forward **and** the reversible adjoint on the GPU with correct gradients
(the capability ADR 0001 dropped). Make it first-class:

1. **Teach the fork to recognize the `mps` platform.** Today `config.py` / `core/jax/sharding.py` assume
   `cpu`/`gpu`/`METAL` and fail under jax-mps (I shim `jax.devices` + `jax_platform_name` in the research
   harness). Add `mps` as a known single-device platform (small, autodiff-safe, upstreamable to fdtdx).
2. **Document the recipe**: `pip install jax-mps` (jax 0.10.x, Python ≥3.13) + `JAX_PLATFORMS=mps` +
   `JAX_MPS_ASYNC_DISPATCH=1`; gradients run on Metal at ~CPU-parity (turns favorable at larger N like the
   forward did). Keep forward on the fused kernel; route only the gradient path through jax-mps.
3. **No engine code** — it's a dependency + a platform-recognition patch. Zero fork divergence.

## WS-3 — Unification via `custom_call` (the "fast + differentiable + one codebase" endgame)

The path to keeping the fused-kernel speed *and* getting autodiff *and* collapsing the ~1548 LOC of
bridge/loop/dispatch scaffolding into JAX:

1. **Add `stablehlo.custom_call` dispatch + a kernel-registration API to jax-mps** (applejax already has a
   `CustomCallRegistry`; jax-mps does not). Contribution to jax-mps: route a custom_call target to an
   `mx.fast.metal_kernel` — reusing the fork's *existing MLX MSL* directly.
2. **Wrap the FDTD forward + reverse-time adjoint as JAX custom primitives with `custom_vjp`**, lowering to
   those custom_call targets. Result: JAX-native, differentiable, fused FDTD on Metal — one `import fdtdx`,
   forward at kernel speed, gradients included, most scaffolding retired.
3. **Alternative host — fix applejax.** applejax already has the registry (MPSGraph) but crashes on a
   `CFRelease` double-free in `MpsExecutable::Execute`; root-cause + patch would make its complex/linalg
   breadth available too. Lower priority than the jax-mps custom_call path (which reuses our MLX kernel).

This is the big, longer-horizon item; it is *architecturally validated* (custom_call + custom_vjp is the
standard JAX hand-rolled-kernel mechanism) but is a real plugin contribution, not a config change.

## WS-4 — Upstream contributions (branch off **upstream fdtdx**, not the fork)

Per the "work on a local branch with latest upstream" instruction — these are autodiff-safe and general:

1. **Monitor optimization → JAX detectors.** The 3.9× win (region-restricted interpolation + activity-
   gating + DFT auto-subsampling) currently lives only in `src/fdtdx/mlx/{detector_freeze,accumulate,
   interpolate}.py`. Re-implement it **differentiably in fdtdx's JAX detector classes**
   (`objects/detectors/`) on a fresh branch off `upstream/main`, parity-test, PR. Region-restriction +
   gating are exact; DFT subsampling exact within the oversampling margin. Strong standalone PR.
2. **`mps`-platform recognition** (from WS-2.1) — also a clean upstream PR (helps any Mac user with a
   plugin installed).
3. Already landed / queued: offdiag spacing-weight is upstream **#378**; remaining `UPSTREAM_CONTRIB.md`
   items (Nyquist DFT subsampling, region interpolation) fold into WS-4.1.

Mechanics: `git worktree` or a clone at `upstream/main`; do **not** carry MLX/fork code into these PRs.

---

## Sequencing
- **Now / near:** WS-2 (mps platform + recipe — days) ‖ WS-4.1 (monitor upstream PR — independent, on an
  upstream branch).
- **Next:** WS-1.1 (in-kernel full-tensor / per-tile material — the biggest measured perf gap).
- **Then:** WS-1.2 (tiled sub-floor engine, spill-justified).
- **Longer horizon:** WS-3 (custom_call unification) — the highest-leverage but largest item; decide after
  WS-1/WS-2 land whether to invest in the plugin contribution.

## Open verification hooks
- WS-1: re-run `benchmarks/research/engine_matrix.py` per material class + the O-band MRM reference wall
  (`examples/ring_mrm_oband/field_maps_100nm.py`), parity vs forced-JAX at each step.
- WS-2/3: `benchmarks/research/inverse_design_adjoint.py` under `JAX_PLATFORMS=mps`, gradient vs CPU + FD.
- Heterogeneous distributions still un-benchmarked (build_case fills uniformly) — add a device+background
  scene to quantify the per-tile-compaction win (WS-1.1) where the roadmap predicts it is largest.
