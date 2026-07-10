# Task 1 — Feasibility & effort study: hand-rolling custom-kernel (`custom_call`) support in jax-mps

**Type:** deep investigation → **feasibility report with a grounded effort/resource/timeline estimate**
(C++/Metal/MLIR-heavy). **Parallelizable:** yes (independent of Tasks 2/3).
**Background:** [`../research/jax-mps-eval.md`](../research/jax-mps-eval.md),
[`../research/jax-metal-landscape.md`](../research/jax-metal-landscape.md),
[`../research/hybrid-and-optimization-plan.md`](../research/hybrid-and-optimization-plan.md) (WS-3 + Reassessment).

> **⚠️ This is a scoping study first, not an implementation sprint.** The owner (project lead) is **not
> familiar with this class of work** (PJRT plugin internals, Metal kernel authoring, MLIR/StableHLO) and needs
> a **realistic estimate of how much time, skill, and risk** this carries before committing. So the **primary
> deliverable is the feasibility report + effort estimate**; a minimal proof-of-concept is secondary (do it
> only if the build + first kernel prove quick). This gates whether the fork can eventually be retired —
> **do not retire the fork on this task's expectation.** Sequencing: run Tasks 2 & 3 and the fork's kernel
> work (WS-1) first; start this deliberately.

## The question to answer

Can we ourselves add a **generic "run this Metal kernel" `custom_call`** to jax-mps (so fdtdx can emit
`stablehlo.custom_call @mps.metal_kernel(...)` and run our fused FDTD kernel *inside JAX*, with `custom_vjp`
for autodiff), and **what does that actually cost** — in person-weeks, required skills, and risk? Today this
is the *only* way to get the fork's ~5-RT kernel speed while staying in JAX; the generic hook is **not in
jax-mps** and issue #203 that proposes it is an **unanswered, unsubmitted** proposal (audited 2026-07, main
`7b1ae82`) — so we cannot wait for it, we would build it.

## Investigation area A — jax-mps infrastructure (how much would we add, and where)

Read the plugin end-to-end and map exactly what a generic-kernel `custom_call` touches:
- The **PJRT plugin surface** (`src/pjrt_plugin/pjrt_api.cc`, `mlx_executable.{cc,h}`, `mlx_client`,
  `mlx_buffer`): how StableHLO is parsed and lowered to an MLX lazy graph and run under one
  `mlx::core::compile()` + `async_eval`.
- The **op-handler + custom_call dispatch** (`ops/control_flow.cc`): the ~23 hard-coded targets
  (`mps.rms_norm`, `mps.eigh/qr/svd`, `mps.sdpa`, …), several with **built-in VJP handlers** — this is the
  *template*. What a new `@mps.metal_kernel` target needs: unpack operands/attrs (MSL source, grid,
  I/O dtypes/shapes) → build & dispatch via **`mlx::core::fast::metal_kernel`** (`#include <mlx/fast.h>` is
  already present) → return the output arrays. Compare with the **#203 prototype's ~1500 LOC** estimate.
- **Build system reality** (the first friction, quantify it): scikit-build-core + CMake + **pinned
  LLVM/StableHLO matching jaxlib 0.10.x**. Can we build jax-mps from source on this machine, and how long/how
  fragile? Document the recipe.
- **The XLA-FFI alternative**: would implementing the PJRT/XLA-FFI custom-call extension (so kernels register
  from Python with **no rebuild**) be more work but far more reusable/upstreamable than a bespoke target?
  Scope both.

## Investigation area B — how Apple Silicon actually runs these kernels (the risk surface)

Understand the Metal execution model well enough to judge whether our kernel injects **correctly and fast**:
- How MLX dispatches a `fast::metal_kernel`: MSL → `MTLComputePipelineState` → command buffer, the
  **unified-memory** buffer model, the lazy graph + `eval`, threadgroup/SIMD, and the "no device-wide barrier
  inside a kernel" constraint.
- **The race hazard (make-or-break):** jax-mps *wrote and abandoned* a hand-written Metal kernel (eigh Jacobi)
  because it **"intermittently races under MLX's untracked-resource model"** (jax-mps#169), reverting to CPU.
  Our fdtdmex kernel is race-free (MLX functional/out-of-place), but jax-mps wraps execution differently —
  **determine what MLX's resource-tracking model requires** (WAR-fences, `array` dependency edges) so an
  injected kernel is deterministic. This is the single biggest technical unknown.
- **The no-donation cost:** `PJRT_Buffer_DonateWithControlDependency = nullptr` → the `while`/`scan` carry
  (E/H/ψ) may be copied each step. Estimate the residual traffic; note the counted-loop fast path (#193/#194)
  helps. This bounds how close we can get to fdtdmex even *with* the kernel.

## Investigation area C — what optimized numerical libraries Apple already provides (leverage vs build)

Motivated by the owner's Apple background (an internal team's LAPACK-for-Apple-Silicon work). Map the
landscape so we know what to **reuse** vs **hand-write**, and to contextualize the eigh-on-CPU decision:
- **CPU — Accelerate (vecLib + BLAS + LAPACK) on the AMX coprocessor.** Apple's strong, optimized dense-linalg
  path (AMX is L2-cache-coupled, high-throughput; beats OpenBLAS at medium/large sizes). This is what jax-mps
  calls for `eigh` and why linalg lives on CPU. Confirm current scope + any GPU/Accelerate crossover APIs.
- **GPU — Metal.** Enumerate what Apple ships: **MPS** (`MPSMatrixMultiplication`, `MPSMatrixDecomposition*`
  LU/Cholesky, `MPSMatrixSolve` — BLAS/LAPACK-*style* but ML/image-oriented and incomplete), **MPSGraph**, and
  the newer **MetalPerformancePrimitives** (macOS 26, matmul). Establish the key fact for us: **there is no
  comprehensive GPU LAPACK** (no robust GPU `eigh`/`svd`; "upstream MLX has no GPU eigh either"), which is why
  dense linalg stays on CPU/AMX. **Verify whether that has changed** (MetalPerformancePrimitives, newer MPS).
- **The conclusion this drives:** the **FDTD update is a memory-bound stencil, not dense linalg** — precisely
  the regime where the **GPU beats AMX** and where **no Apple library helps**, so it is genuinely a
  hand-written **`fast::metal_kernel`** (our existing MSL). Apple's numerical libraries are *not* a shortcut
  for the FDTD kernel; they only matter for the mode solver's eig/linalg (which should stay CPU/Accelerate,
  like jax-mps's eigh). Record this explicitly so the effort estimate doesn't assume a library shortcut that
  doesn't exist.

## Deliverable — the feasibility report (the point of this task)

A written report answering, with evidence:
1. **Is it feasible** to add generic Metal-kernel `custom_call` to jax-mps and inject our kernel race-free? Y/N
   with the specific unknowns resolved (race model, build, donation).
2. **Effort & resources:** person-weeks with confidence bands, broken down (build setup; the generic handler
   ≈ #203's ~1500 LOC?; JAX-side custom primitive + `custom_vjp`; porting kernel variants iso/diagonal/
   full-tensor/CPML/ADE/boundaries; clearing the race; upstreaming). **Required skills** (C++, Metal/MSL, MLIR/
   StableHLO, PJRT) — so the owner can staff it.
3. **Expected payoff & residual gap:** if we injected one E-update kernel, where would throughput land
   (~1600 Mcs/s kernel regime vs ~240 op-graph), and what residual gap vs fdtdmex remains from donation/race.
4. **Go/no-go recommendation** + the cheapest next step (usually: build jax-mps + one trivial `a+b` custom
   kernel to de-risk the build & the race model before scoping the rest).
5. **Optional PoC** (only if quick): one FDTD E-update kernel via `@mps.metal_kernel`, element-wise-equal to
   the fork's kernel, race-checked over many repeats, with a throughput number.

## Caveats
- Target **jax-mps (MLX)**, not applejax/MetalHLO (MPSGraph — community-warned, and applejax crashes here).
- The upstream-fdtdx PyTorch-refactor risk applies to the *fdtdx-side* wiring, not to this jax-mps work
  (a Metal-kernel-via-custom_call mechanism is framework-agnostic and reusable).
- Coordinate the `mps`-platform handling with Task 3.
