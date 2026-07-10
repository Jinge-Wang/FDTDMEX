# Task 1 — Make jax-mps able to run our hand-rolled Metal kernel

**Type:** investigation + de-risking spike (C++/build-heavy). **Parallelizable:** yes (independent of Tasks 2/3).
**Background:** [`../research/jax-mps-eval.md`](../research/jax-mps-eval.md), [`../research/hybrid-and-optimization-plan.md`](../research/hybrid-and-optimization-plan.md) (WS-3).

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

**What is MISSING (why you can't use it today):**
- **No runtime / FFI custom_call registration.** There is *no* XLA-FFI or PJRT-FFI custom-call extension
  (grep confirms: only a *profiler* extension in `pjrt_api.cc`; no `XLA_FFI`/`register_custom_call`). jax-mps
  recognizes **only the target names hard-coded in its C++**; any other target → error. So you **cannot
  register a kernel from Python** — you must add a handler to jax-mps's source and **rebuild the plugin**.
- Therefore there is no FDTD handler and no general "run this MSL" hook.

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
4. **FDTD proof:** register a single **E-update** kernel reusing the fork's MSL
   (`src/fdtdx/mlx/kernels.py` `_field_source`/`_common`). Validate the field is **element-wise equal**
   to the fork's kernel output, and measure throughput (target: the ~5-RT / ~1600+ Mcs/s regime, not the
   ~240 op-graph ceiling).
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
