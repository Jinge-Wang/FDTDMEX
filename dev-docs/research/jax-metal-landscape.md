# The JAX-on-Apple-Metal landscape (2026-07)

Deep survey of every way to run JAX on the Apple GPU, for the "fold Metal acceleration into upstream
fdtdx" decision. Sources: project repos + issue trackers, the `jax-ml/jax` discussion, independent blogs,
Swift Forums, and MLX-ecosystem coverage (links at bottom). **Bottom line up front:** the substrate war is
settled in MLX's favor; **jax-mps is the leader and the right vehicle**, and its open issue **#203 is
literally the generic-Metal-kernel hook we need**; **MetalHLO** is the ambitious wildcard to watch;
applejax and Apple's jax-metal are deprioritized.

## The substrate: MLX vs MPSGraph (this decides everything)

- **MLX** (Apple ML Research): ~27.3k★, v0.31.2, WWDC-blessed, **Ollama switched its Apple-Silicon engine
  to it (Mar 2026)**. The thriving, recommended substrate. It is the *engine underneath* jax-mps.
- **MPSGraph** (Apple's older graph API): community verdict is blunt — *"avoid MPSGraph at all costs:
  extremely buggy, poorly documented, performance unpredictable."* applejax (MPSGraph) crashed on our fdtdx
  graph (a CFRelease double-free); MetalHLO (MPSGraph-default) reports CNN-training drift + random-normal
  divergence. **Anything MLX-based inherits the healthier foundation.**

## The five players

### 1. MLX itself — not a JAX plugin, but the foundation
The array framework. You'd only use it directly if you *rewrite* in MLX (that's what this fork's engine
does). For "keep JAX code," you want a plugin that targets MLX → jax-mps.

### 2. jax-mps (tillahoffmann) — **the leader, and our vehicle**
- **What:** C++ PJRT plugin, StableHLO→**MLX**, registers as the `mps` JAX platform. jax/jaxlib 0.10.x.
- **Momentum:** **182★, 21 forks, 23 releases**, v0.10.8 (Jul 2026), active (recent work: quantization,
  async dispatch, fusion). JAX maintainer (Jake VDP) called the approach *"sound"* (no official endorsement).
- **Perf:** ~3.7× CPU on ResNet18; for *our* memory-bound FDTD stencil it's ~2× CPU (op-graph ceiling — it
  runs the whole program as one `mlx::core::compile()` graph; XLA/StableHLO does no Metal-specific fusion,
  so it can't fuse the stencil — the ~5× gap to our fused kernel).
- **🔑 Issue #203 "Generic metal kernel dispatch" (OPEN):** a contributor (porting ColabFold/AlphaFold) has a
  **~1500-line prototype** for `mps.metal_kernel_jit` (compile+dispatch a Metal kernel from source) and
  `mps.metal_kernel_lib` (load `.metallib`), which *"would enable external code to register custom fused
  Metal kernels without modifying jax-mps core."* **This is exactly the hook Task 1 needs** — awaiting the
  maintainer's design call. Today there is otherwise **no runtime FFI/registration** (custom_call targets
  are hard-coded C++: `mps.rms_norm`, etc.), so without #203 you must fork+rebuild jax-mps.
- **Known correctness watch:** #195 — nondeterministic/incorrect gradients through `fori_loop` with nested
  conditionals. Relevant to the reversible-FDTD adjoint; validate gradients carefully.
- **Verdict:** best substrate (MLX), most momentum, reuses our MSL kernel, and the extensibility hook is
  already in flight. **Primary bet.**

### 3. applejax (danielpcox) — MPSGraph fork, deprioritize
- **What:** fork of jax-mps, but swapped to **MPSGraph + Accelerate LAPACK**; jax/jaxlib **0.9.x** (behind).
- **Momentum:** **2★, 2 forks**, v0.9.7 (Mar 2026), 303 commits — actively developed but **very low-profile**
  (this is why it's hard to find). 71+ ops, 2000+ tests.
- **Adds over jax-mps:** full linalg (Cholesky/QR/SVD/eigh, real+complex), complex numbers, scatter/gather,
  a `CustomCallRegistry` (used for LAPACK targets).
- **Cons:** MPSGraph substrate (the buggy one) — **crashed on our fdtdx init** (CFRelease double-free); no
  f64, no complex sort/conv, **linalg crashes inside control flow**, no buffer donation, zero-size arrays
  unsupported; behind on jax version.
- **Verdict:** only compelling if you specifically need its complex/linalg breadth (FDTD doesn't). Riskier
  substrate, niche, trailing jax. **Deprioritized** (its perf here is untested only because it crashes — a
  fixable library bug, not a verdict, but not worth pursuing over jax-mps).

### 4. MetalHLO (pedronahum) — **the ambitious wildcard to watch**
- **What:** a **Swift** StableHLO compiler+runtime (Apache-2.0). Three backends: **MPSGraph** (default),
  **custom Metal kernels** (peak perf), and **heterogeneous GPU+ANE+CPU**. XLA-style optimizer with real
  **fusion** (attention/GELU/LayerNorm/softmax) at O0–O3. Ships Swift, C, **and PJRT** (so it's a JAX backend).
- **Capability (the most complete of any here):** full forward+backward training verified on ResNet18 /
  nanoGPT / Flax; `value_and_grad`, optax, `vmap`, `scan`, `remat`, mixed precision; ~88% StableHLO ops
  (92/105), 191/277 conformance. Its custom Metal kernels **beat MPSGraph 1.2–3.4×**, and at −O3/TF32 **beat
  MLX** on FFN (up to 4.63×) and attention (1.0–1.76×). ResNet18 8.7× over CPU (M5 Pro).
- **Cons / immaturity:** **23★, 3 forks, one developer, no releases, "experimental/alpha,"** 0 open issues
  (tiny community). MPSGraph-default (drift/`random.normal` divergence; O3 "under repair" with known bugs).
  **No public user-facing custom-kernel API** (its custom kernels are internal-only). Swift-centric ecosystem.
  `while`>1000 iters + complex bodies fall back / can crash.
- **Verdict:** philosophically the closest to our goal (**custom Metal kernels + fusion + ANE + full
  autodiff in one PJRT backend**) and technically impressive, but **too immature and single-maintainer to
  bet on now**, and MPSGraph-rooted. **Watch it; consider as a future host or collaboration** if it grows a
  public kernel API and matures. Its existence proves the "fused-kernel + differentiable + StableHLO" thesis
  is real.

### 5. jax-metal (Apple official) — dead
OpenXLA + MPSGraph, Apple's own. **Unmaintained/inactive**, Sonoma-only, chronic install/version breakage.
Apple's actual investment is **MLX**, not JAX-on-Metal. Do not use.

## Community sentiment (synthesized)

- **Substrate consensus:** MLX ≫ MPSGraph. Apple, Ollama, and WWDC all point at MLX.
- **JAX-on-Metal is a real but painful, early niche.** Independent take (V. Glazer, Apr 2026): jax-mps is
  *"a way to potentially speed up your JAX code if your ops are supported,"* modest ~4×, with a *"real risk
  it goes the way of jax-metal"*; for **general** ML on Metal he still points to **PyTorch**. (Caveat for us:
  PyTorch's MPS backend lacks FFT and has weak complex — so for *differentiable FDTD* it is **not** a better
  path; JAX+jax-mps remains the stronger Metal route for our niche.)
- **Longevity risk is real for all of them** (young, small teams). jax-mps (182★, active, MLX) is the safest;
  MetalHLO (23★, one dev) the most capable but riskiest.
- Apple is not shipping a first-party JAX-on-Metal path — this space stays community-driven.

## Implications for our plan

1. **jax-mps stays the primary vehicle** — MLX substrate, leader, reuses our kernel, and #203 is our hook.
2. **Engage issue #203 first** (support / co-develop `mps.metal_kernel_jit`) *before* forking jax-mps
   ourselves — if it lands, we register our MSL kernel **from Python, no fork/rebuild**. This materially
   lowers Task 1's cost and risk. (Fold this into [`../tasks/task-1-jaxmps-custom-kernel.md`](../tasks/task-1-jaxmps-custom-kernel.md).)
3. **Watch MetalHLO** as the wildcard — it already has custom Metal kernels + fusion + ANE + autodiff. Not a
   bet today (alpha, one dev, MPSGraph-default, no public kernel API), but the one to revisit; its numbers
   validate that "fused kernel + differentiable via StableHLO" works.
4. **Drop applejax** from active consideration (MPSGraph, trailing jax, niche; keep only as a "if we ever
   need complex/linalg breadth" note).
5. **Guard the adjoint:** jax-mps #195 (control-flow gradient correctness) means we must keep validating
   inverse-design gradients vs CPU/finite-difference, not assume correctness.

## Sources
- MLX: github.com/ml-explore/mlx · Apple ML Research (M5/MLX) · Ollama→MLX (2026-03-30)
- jax-mps: github.com/tillahoffmann/jax-mps (repo, issues #203/#195/#201/#196), PyPI
- applejax: github.com/danielpcox/applejax, PyPI
- MetalHLO: github.com/pedronahum/MetalHLO · forums.swift.org/t/…/85209
- jax-metal: developer.apple.com/metal/jax · jax-ml/jax discussion #34648, issue #8074
- Independent: vglazer.github.io/jax-mps (2026-04-12); ndalton12 "Jax Metal vs MLX"
