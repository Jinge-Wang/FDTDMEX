# Task 1 — Make jax-mps able to run our hand-rolled Metal kernel

**Type:** investigation + de-risking spike (C++/build-heavy). **Parallelizable:** yes (independent of Tasks 2/3).
**Background:** [`../research/jax-mps-eval.md`](../research/jax-mps-eval.md), [`../research/hybrid-and-optimization-plan.md`](../research/hybrid-and-optimization-plan.md) (WS-3), [`../research/jax-metal-landscape.md`](../research/jax-metal-landscape.md).

> **⚠️ Scheduling note (may be postponed).** A fresh audit (2026-07, jax-mps @ `7b1ae82`) confirms the
> generic user-kernel hook **is not implemented and not reliably upcoming**: issue #203 is an *unanswered*
> proposal — **no maintainer response, no milestone, no PR** (the ~1500-LOC prototype is unsubmitted). So this
> task is **not "turn on a coming feature"** — it means *building* `mps.metal_kernel_jit` ourselves (fork/PR),
> and clearing two known hazards (below). It is a **months-scale research bet**. **Sequencing:** do Tasks 2 & 3
> and the fork's kernel work (WS-1) first; treat this as the **go/no-go spike that gates retiring the fork** —
> start it deliberately, and consider postponing full investment until the low-risk wins land and/or the
> maintainer signals on #203. **Do not retire the fork on this task's expectation.**

## Why this exists (the robust answer to "why can't I just use it")

jax-mps is a **PJRT plugin that executes JAX's StableHLO by mapping each op to MLX** (`mlx::core::…`) and
running the whole program as one `mlx::core::compile()` graph. That op-graph is exactly the fork's own
"MLX-op cores" ceiling (~200–240 Mcs/s); it **cannot fuse the FDTD stencil into one dispatch** — only a
hand-written `metal_kernel` does (the fork's 5-RT kernel, ~5× faster). "Inject our kernel via `custom_call`"
means: emit a `stablehlo.custom_call @<target>` from JAX in place of the op-graph update, and have jax-mps
recognize `@<target>` and run **our** Metal kernel for it. That is the *only* way to get the 5-RT speed
while staying inside JAX (and keeping autodiff via `custom_vjp`).

**What already exists in jax-mps** (so this is a moderate extension, not new infra):
- custom_call **dispatch by target name** — `src/pjrt_plugin/ops/control_flow.cc` has
  `if (callTargetName == "mps.rms_norm") { … mlx::core::fast::rms_norm(...) }`, plus `mps.layer_norm`,
  `mps.rope`, `mps.scaled_dot_product_attention`. Each handler is ~20–40 lines. **Several already ship a
  VJP/gradient rule** (rms_norm/layer_norm backward) — a template for our `custom_vjp`.
- `#include <mlx/fast.h>` is present, so **`mlx::core::fast::metal_kernel`** (the C++ twin of the Python
  `mx.fast.metal_kernel` the fork uses) is directly callable — **our existing MSL string is reusable**.
- Its own fusion passes (`passes/fuse_softmax|layer_norm|rms_norm`) already emit `custom_call @mps.*`.

**What is MISSING (why you can't use it today) — audited on main @ `7b1ae82`:**
- **No runtime / FFI custom_call registration.** No XLA-FFI or PJRT-FFI custom-call extension
  (`PJRT_Buffer_DonateWithControlDependency = nullptr`, only a *profiler* extension in `pjrt_api.cc`). jax-mps
  recognizes **only ~23 hard-coded target names** (`mps.sdpa`, `mps.rms_norm`, `mps.layer_norm[_bwd]`,
  `mps.eigh/qr/svd`, `mps.quantized_matmul`, …); any other target → error. So you **cannot register a kernel
  from Python** — you must add a handler to jax-mps's source and **rebuild**.
- **The generic hook (#203) is an unanswered proposal, NOT in flight.** `mps.metal_kernel_jit` /
  `mps.metal_kernel_lib` are proposed by a ColabFold contributor with a **~1500-LOC prototype**, but as of the
  audit: **no maintainer response, no milestone/assignee, no PR** (prototype unsubmitted). Not on main, not on
  any origin branch. **We cannot assume it lands** — the realistic path is to *implement it ourselves* on a
  jax-mps fork/PR (co-develop with #203's author if possible).
- **Two known hazards the spike must clear (precedent inside jax-mps):**
  1. **MLX "untracked-resource" races (jax-mps#169).** jax-mps *wrote and then abandoned* a hand-written Metal
     kernel (eigh Jacobi) because it *"intermittently races under MLX's untracked-resource model"* (and was
     slower), reverting to CPU LAPACK. Our fdtdmex kernel is race-free (MLX functional/out-of-place), but
     jax-mps wraps execution differently — **prove race-freedom explicitly**, expect to need a WAR-fence.
  2. **No buffer donation** → the `while`/`scan` carry (E/H/ψ) may be copied each step (fdtdmex avoids this via
     a plain loop + MLX caching allocator). **Measure** the residual cost. (The counted-loop fast path, #194,
     is on main and helps; our FDTD loop is counted.)
  See the landscape survey [`../research/jax-metal-landscape.md`](../research/jax-metal-landscape.md) and the
  Reassessment in the plan.

## Goal

Deliver a **working spike + a spec** proving that an FDTD update kernel, injected via `custom_call`, runs
inside JAX on Metal at ~kernel speed — and decide the cleanest mechanism to contribute upstream.

## Steps

1. **Build jax-mps from source** (`github.com/tillahoffmann/jax-mps`, or the clone in the session
   scratchpad). This is the first real friction: scikit-build-core + CMake + LLVM/StableHLO pinned to the
   jaxlib 0.10.x bytecode. **Document the build recipe** (deps, versions, gotchas) — it gates everything.
2. **Prototype the general mechanism (preferred over a bespoke FDTD target):** add a general
   **`custom_call @mps.metal_kernel`** handler that takes the MSL source + grid/threadgroup + I/O
   dtypes/shapes as `backend_config`/attributes and dispatches via `mlx::core::fast::metal_kernel`. Mirror
   the `mps.rms_norm` handler. *Also assess* implementing the **PJRT/XLA-FFI extension** so JAX's
   `jax.ffi.register_ffi_target` works — that would let kernels register from Python with **no rebuild**, a
   far more reusable contribution likely welcomed in jax-mps upstream. Recommend one.
3. **End-to-end trivial proof:** from JAX, emit that custom_call for a toy kernel (`out = a + b` via a
   hand MSL) and confirm correct GPU execution under `JAX_PLATFORMS=mps`.
4. **FDTD proof — the go/no-go measurement:** register a single **E-update** kernel reusing the fork's MSL
   (`src/fdtdx/mlx/kernels.py` `_field_source`/`_common`). Verify (a) **race-freedom** across many repeats
   (the #169 hazard — no intermittent wrong results), (b) **element-wise equality** to the fork's kernel, and
   (c) **throughput** in a real `run_fdtd` loop. Report where it lands: the ~5-RT / ~1600+ Mcs/s regime
   (success) vs the ~240 op-graph ceiling (failure), and **attribute any residual gap vs fdtdmex** to the
   no-donation carry copies and/or WAR-fence overhead. This number is the whole task's verdict.
5. **Spec the full port:** enumerate what it takes to cover all kernel variants (iso/diagonal/full-tensor,
   CPML fold, non-uniform metric, ADE, periodic, PEC/PMC) and the **`custom_vjp`** (start: fall back to
   the op-graph `reversible_fdtd` for gradients — already runs under jax-mps at ~CPU-parity; later: a
   reverse-time Metal kernel). Give an effort estimate.

## Deliverables
- jax-mps build recipe; the `@mps.metal_kernel` (and/or FFI) patch on a jax-mps fork/branch; the trivial +
  FDTD spikes with correctness + throughput numbers; a written recommendation (bespoke handler vs general
  metal_kernel custom_call vs full FFI) and full-port effort estimate. **Update the research log with results.**

## Caveats
- Community sentiment strongly favors **MLX over MPSGraph** ("avoid MPSGraph at all costs"), so target
  **jax-mps (MLX)**, not applejax (MPSGraph; it also crashes here — see the research log).
- The upstream-fdtdx PyTorch refactor risk applies to the *fdtdx-side* wiring, not to this jax-mps work
  (a Metal-kernel-via-custom_call mechanism in jax-mps is framework-agnostic and reusable regardless).
