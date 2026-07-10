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

**This is the strategic north star**: if a `custom_call`-registered kernel matches the fork's speed *inside
JAX*, then fdtdx upstream can ship Metal acceleration behind `pip install jax-mps`, the fork is **retired**,
upstream commits stop needing hand-porting, and the speedup reaches every fdtdx user. Feasibility is now
**high** — jax-mps already has the exact machinery:

- Its own fusion passes emit `stablehlo.custom_call @mps.softmax / @mps.layer_norm / @mps.rms_norm / @mps.rope`
  and its handler (`ops/control_flow.cc`) dispatches by `call_target_name` to `mlx::core::fast::…` calls —
  **~20-line handlers**, several with **built-in VJP rules** (rms_norm/layer_norm backward). It `#include
  <mlx/fast.h>`, so **`mlx::core::fast::metal_kernel`** (the C++ twin of the fork's `mx.fast.metal_kernel`)
  is available. Registering the FDTD update is "add another handler like rms_norm, calling `fast::metal_kernel`
  with our existing MSL string." **The kernel is the durable asset; only the invocation path changes.**

Path:
1. **Contribute a general `@mps.metal_kernel` custom_call to jax-mps** (source MSL + grid/attrs → `fast::
   metal_kernel`). More reusable than an FDTD-specific target and the likeliest-accepted upstream PR — it is
   effectively "JAX FFI for Metal kernels." (applejax has a `CustomCallRegistry` too but uses MPSGraph, which
   the community warns against and which is what crashed it — prefer jax-mps/MLX.)
2. **fdtdx side:** a JAX primitive that emits that custom_call for the E/H update, reusing the fork's MSL
   (`mlx/kernels.py`). Forward-only first (fast forward in JAX, no gradient on that path).
3. **Differentiable tier (`custom_vjp`):** provide the reverse pass. Two options — (a) cheap: fall back to
   the existing op-graph `reversible_fdtd` for gradients (runs under jax-mps at ~CPU-parity — already
   validated), fast forward via the kernel; (b) full: a reverse-time Metal kernel as the custom_vjp for fast
   inverse design too. Start with (a); (b) is a later perf item.

**Effort estimate.** Moderate, front-loaded on a spike: (i) **de-risk spike** — one E-update MSL kernel behind
a jax-mps `@mps.metal_kernel` handler, prove correctness + ~5-RT speed inside JAX (days, needs a jax-mps
source build). (ii) **generalize + port** all kernel variants (iso/diagonal/full-tensor, CPML fold, metric,
ADE, periodic, PEC/PMC) to the custom_call form — this is the bulk, but it is *re-expressing the existing
1444-LOC kernel*, not new physics. (iii) **retire scaffolding** (~1548 LOC bridge/loop/dispatch/freeze).
(iv) upstream the jax-mps PR + the fdtdx Metal-backend PR. The reverse kernel (3b) is a separable follow-on.
Net (revised by the 2026-07 audit below): the *pattern* exists in jax-mps (23 hard-coded `custom_call`
handlers with VJPs, `mlx/fast.h` available), but the **generic user-kernel hook does not** and is not
reliably coming (#203 unanswered) — so step (i) now includes **implementing** `mps.metal_kernel_jit` on a
jax-mps fork/PR ourselves, plus clearing the MLX-resource race (#169) and the no-donation cost. That makes
this a **months-scale research bet, not weeks** — see the Reassessment section.

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

## Reassessment (2026-07, after a fresh jax-mps code + issue audit)

**Is `custom_call` the *only* bottleneck — i.e., without it is jax-mps hard-capped below fdtdmex?**
Yes, it is the **primary and necessary** one, but not the *only* factor for full parity:

1. **Stencil fusion (the hard cap) — needs `custom_call`.** jax-mps runs the update as one
   `mlx::core::compile()` op-graph; that fuses elementwise chains but **cannot** merge the stencil's halo
   reads / keep the working set on-chip. Measured ceiling ≈ **2× CPU / ~240 Mcs/s**, ~5× below the fork's
   fused kernel. **No jax-mps op-level tuning crosses this** — only injecting a hand-written fused kernel
   does. So without `custom_call`, jax-mps forward is permanently ~5× under fdtdmex. **Confirmed cap.**
2. **Buffer donation is unimplemented** (`PJRT_Buffer_DonateWithControlDependency = nullptr`). The JAX
   time loop carries E/H/ψ as the `while`/`scan` carry; without donation those buffers can be copied each
   step (fdtdmex avoids this with a plain loop + MLX's caching allocator). Residual traffic — partly
   absorbed by MLX's allocator, magnitude **TBD, measure**.
3. **MLX "untracked-resource" races for custom Metal kernels** (jax-mps#169). jax-mps *tried* a hand-written
   Metal kernel (eigh Jacobi) and **abandoned it** — "intermittently races under MLX's untracked-resource
   model" — falling back to CPU LAPACK. Our kernel is race-free in fdtdmex (MLX functional/out-of-place),
   but jax-mps's execution model wraps it differently; a WAR-fence workaround may be needed. **Real risk.**
4. **Control-flow loop overhead** — largely mitigated on main by the **counted-loop fast path** (#193/#194);
   our FDTD loop is counted, so it benefits. Minor.

**So:** overcoming `custom_call` is **necessary and make-or-break** — it lifts forward from ~2× to most of
the ~13× CPU band — but (2)+(3) may leave a **residual gap** vs fdtdmex's hand-tuned loop. We don't need
*exact* parity (the goal is unification + most of the speed), but the Task-1 spike must **measure** the real
custom_call-kernel throughput (and race-freedom) rather than assume parity.

**Status of the hook (#203), re-checked:** the generic metal-kernel dispatch is an **open, unanswered
proposal** — no maintainer response, no milestone/assignee, **no PR** (the ~1500-LOC prototype is unsubmitted,
awaiting guidance). It is **missing and not reliably upcoming.** We therefore **cannot wait for it**; the
realistic path is to *drive it ourselves* (implement `mps.metal_kernel_jit` on a jax-mps fork/PR, ideally
co-developing with #203's author) — a meaningful commitment, not a "turn it on."

**Consequence for the roadmap:** WS-3 (unification / retire-the-fork) is a **longer, higher-uncertainty
research bet**, not a near-term certainty. → **Do not retire the fork on its expectation.** Keep investing in
the **kernel** (WS-1) — it is the durable asset reused by *any* future path (fork bridge today, custom_call
later) — and land the low-risk upstream wins (WS-2/WS-4) now. Gate the whole unification on the Task-1 spike.

## Sequencing (revised)
- **Now / near (low-risk, independent, parallel):** WS-2 / Task 3 (mps-platform recognition + recipe — days)
  ‖ WS-4.1 / Task 2 (monitor upstream PR, on an `upstream/main` branch). Both ship value regardless of WS-3.
- **Main ongoing investment:** WS-1.1 (in-kernel full-tensor / per-tile material — the biggest *measured*
  perf gap, full_aniso only ~1.4× CPU) then WS-1.2 (tiled sub-floor engine, spill-justified). **The kernel is
  the durable asset** — it is what a future `custom_call` path would register, so this work is not lost to WS-3.
- **Gated research bet — Task 1 / WS-3:** the `custom_call` spike (build jax-mps, implement/borrow
  `mps.metal_kernel_jit`, prove one FDTD kernel runs race-free at kernel speed inside JAX). **Do this before
  any decision to retire the fork.** Its outcome — does the injected kernel hit the acceptable throughput band
  despite no donation + the MLX-race model — is the go/no-go for the whole unification.
- **Do NOT retire the fork** on the *expectation* of WS-3; only after the spike proves out.

## Open verification hooks
- WS-1: re-run `benchmarks/research/engine_matrix.py` per material class + the O-band MRM reference wall
  (`examples/ring_mrm_oband/field_maps_100nm.py`), parity vs forced-JAX at each step.
- WS-2/3: `benchmarks/research/inverse_design_adjoint.py` under `JAX_PLATFORMS=mps`, gradient vs CPU + FD.
- Heterogeneous distributions still un-benchmarked (build_case fills uniformly) — add a device+background
  scene to quantify the per-tile-compaction win (WS-1.1) where the roadmap predicts it is largest.
