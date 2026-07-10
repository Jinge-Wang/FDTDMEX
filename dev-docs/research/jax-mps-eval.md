# jax-mps / applejax vs the FDTDMEX MLX engine — deep-research log

Running record. Question: is a JAX-on-Metal **plugin** (jax-mps / applejax) a better long-term base
than the hand-written MLX/Metal fork — weighing forward performance, features (esp. autodiff /
inverse design), and **interface unification** (one fdtdx, retire fdtdmex)? Libraries judged on
measured merit; immaturity is a contribution opportunity, not a disqualifier.

Plan: `~/.claude/plans/i-was-tipped-by-joyful-elephant.md`.

---

## Track 1 — upstream sync to a clean base (DONE, 2026-07-09)

Done first so the plugin comparison runs against a fork that is consistent with current upstream
(and so the user's now-upstreamed anisotropic fix is retired into upstream rather than carried).

**Delta:** upstream `main` `e5351a4 → 65e0fd4` = **15 commits** (a fresh `git fetch` — cached refs
had shown only 1). Buckets: fork-aligned (#378 offdiag spacing-weight = the user's own commit, now
merged upstream; #376 2D modes / etched / `use_etching` / `compute_integrated_power` /
`grid.subgrid`), engine contract-surface (#379 fast PML, #384 PML-constants-on-object, #382 complex
materials, #383 CCPR dispersion), and low-risk front-end (#375 field-projection detectors, #367 GDS
sidewall, #386 GDS masks, #389 mode scaling, #393 JSON serialize, #391 plot aspect, dep bumps).

**Merge:** branch `sync/upstream-65e0fd4` off `mlx-fork`; `git merge upstream/main` was **near-clean**
(fork is additive) — only 2 conflicts:
- `conversion/json.py`: adopt upstream's canonical array serialization (#393), keep the fork's legacy
  `__ndarray__` decoder for backward-compat reads.
- `tests/unit/core/physics/test_modes.py`: upstream's 4-element `transverse_coords` + the fork's
  `mode_backend="tidy3d"` seam arg.

**Behavioral re-point (the real work).** Two upstream restructurings hit the MLX bridge's contract
surface:
1. **PML (#379/#384).** CPML `a/b/inv_kappa` moved off `ArrayContainer.alpha/kappa/sigma` onto each
   `PerfectlyMatchedLayer` object (`pml_a_E/b_E/inv_kappa_E` + `_H`, slab-sized), and `psi_E/psi_H`
   became `PmlAuxField = dict[pml_name → (psi_1, psi_2)]`. Re-point (`mlx/pml.py`, `mlx/bridge.py`):
   - `build_cpml_coeffs_from_pml_objects()` re-assembles the fork's global `(6, Nx, Ny, Nz)` coeff
     view (E-side axis-k → channel k, H-side → k+3) from the per-object arrays; `detect_pml_slabs`
     and the Metal kernel are untouched.
   - ψ is zero after `arrays.reset()` (bridge-in) → allocate zero slabs. Bridge-out rebuilds the
     per-PML dict via `PML_AXIS_TO_PSI_CHANNELS` (axis 0→(3,4), 1→(5,0), 2→(1,2)), derived + verified
     against `curl_E`/`curl_H` `step_cpml` ordering (debug showed the mapped channel matching JAX ψ
     to rel 0.0000). `to_array_container` now takes `objects`.
2. **Dispersion.** ADE polarization `P` moved into `FieldState` (`arrays.fields.dispersive_P_*`);
   coefficients stayed on `ArrayContainer`. Bridge read/write re-pointed.

**New fallback gate:** CCPR (#383) adds a 4th ADE coeff `dispersive_c4` (the `b·dE/dt` term) the MLX
fold doesn't carry → gate to JAX. #382 complex materials need **no** gate (`from_complex_*` splits
to real ε + equivalent conductivity = already-supported lossy path; arrays stay real).

**Contract-surface change to remember:** `to_mlx_state` / `to_array_container` now **require the
`ObjectContainer`** (CPML coefficients are read from `objects.pml_objects`). Any caller must pass it.

**Validation:** full MLX parity gate **55 passed / 1 skip** (`tests/validation/`), covering iso /
diagonal / full-tensor anisotropy, CPML, periodic, PEC/PMC, non-uniform grids, Drude–Lorentz
dispersion — MLX-op cores and the fused Metal kernel both match JAX-CPU. 378/386 unit tests pass;
the 8 failures are `tidy3d`-optional-extra mode tests (real `import tidy3d`, extra not installed) —
not engine regressions. `io` roundtrip tests need the `pydantic`/`io` extra (not installed).
Commits on `sync/upstream-65e0fd4`: `c0410b3` (merge), `38fb4df` (re-point).

**Env note (helps Track 2B):** the fork already resolves to **jax 0.10.1** — the jaxlib 0.10.x line
jax-mps targets — so the plugin/​fork jaxlib pin conflict may be smaller than feared.

---

## Track 2A — architectural comparison of jax-mps vs applejax (DONE)

Both are C++ PJRT plugins that parse StableHLO and execute on the Apple GPU, registered as the
`mps` JAX platform (`JAX_PLATFORMS=mps`). They diverge sharply in engine, jax pin, and breadth:

| | **jax-mps** (tillahoffmann) | **applejax** (danielpcox) |
|---|---|---|
| version | 0.10.9 | 0.9.7 |
| jax/jaxlib pin | **`>=0.10,<0.11`** (matches the fork's jax 0.10.1) | **`>=0.9,<0.10`** (needs a separate env) |
| execution engine | **MLX** C++ API (`mlx_executable/mlx_client/mlx_buffer`) — same engine the fork uses | **MPSGraph + Metal + Accelerate LAPACK** (Obj-C++ `mps_*.mm`) |
| op layer | MLIR visitor in `stablehlo_parser.cc`; ML **fusion passes** (softmax, layer-norm) | **modular `ops/`** registry (binary, unary, reduction, shape, gather_scatter, fft, sort, linalg, convolution, control_flow, bitwise, tensor_creation) |
| StableHLO coverage | 95.2% of JAX's test suite (excl. float64, sub-byte dtypes, multi-device); README: "a lot of ops not yet implemented" | **comprehensive** — verified in source: add/sub/mul/div, pad, slice, dynamic_slice, **dynamic_update_slice**, **gather**, **scatter**, select_and_scatter, reduce, reduce_window, **while/if/case**, convolution, **fft**, **complex/real/imag**, cholesky, triangular_solve, sort, iota, reverse, transpose, concatenate |
| complex / linalg / autodiff | not documented; leaner (transformer/CNN focus) | complex64 + full linalg (Cholesky/QR/SVD/eigh/eig via Accelerate); control flow cond/switch/while/fori/scan/associative_scan; grad/jacobian/hessian/custom-JVP-VJP |
| LOC (C++ / Py) | ~11.5k / 12.6k | ~12.3k / 5.9k |
| tests | JAX upstream suite | 2000+, CUDA-parity doc |

**Soundness read.** applejax is the more *general-purpose, extensible* design (per-op files + registry,
LAPACK/MPSGraph for linalg, aims for CUDA parity). jax-mps is leaner and ML-throughput-oriented (MLX +
fusion passes for softmax/layer-norm — irrelevant to FDTD). For *FDTD* specifically, applejax's breadth
matters and jax-mps's fusion passes don't; but jax-mps sharing MLX + jax 0.10.x with the fork makes it
the more natural "unify" target for the forward path.

**FDTD op-need vs coverage.** Forward FDTD needs elementwise + pad + slice + dynamic_update_slice
(source injection) + reduce (detectors) + reverse/roll + select — applejax has all; jax-mps very likely
(these are core ML ops). Inverse design additionally needs `while` (scan/fori_loop lowering) + scatter/
gather (reversible-FDTD `.at[].set`/adjoint) + custom-VJP — **applejax documents all of these; jax-mps's
coverage of scatter/control-flow completeness is the open question** the experiments must probe.

**applejax caveats vs FDTD** (from `CUDA_PARITY.md`): no float64 (FDTD is f32 — fine); complex sort/conv
crash (FDTD does neither — fine); **"linalg inside control flow crashes"** — FDTD's time loop has *no*
linalg inside it (stencils only; the mode solver's eig runs on the host via scipy), so this should not
bite inverse design. Net: **applejax is architecturally capable of both FDTD forward and the reversible
adjoint**; that is the headline hypothesis for 2D.

## Track 2B — environments (DONE)
Both plugins ship **prebuilt wheels** (no source build). Isolated venvs in scratchpad:
- `env-jaxmps`: Python 3.13 + `jax==0.10.1`/`jaxlib==0.10.1` + `jax-mps==0.10.8` + the fork (editable).
  jax stayed 0.10.1; `mlx` came along (arm64), so this one env runs JAX-CPU, jax-mps, **and** both MLX
  paths. Smoke: matmul/pad/roll/scatter on `MpsDevice` OK.
- `env-applejax`: Python 3.13 + `jax==0.9.2`/`jaxlib==0.9.2` + `applejax==0.9.7` + the fork. Smoke: matmul OK.

**Fork shim needed (small, notable).** fdtdx's `config`/`sharding` assume `cpu`/`gpu`/`METAL` platforms and
call `jax.devices(backend=...)` / `jax.config.update("jax_platform_name", …)`; the plugin platform is
`mps`, so the fork's config resolution fails ("Unknown backend cpu"). A 2-line research shim
(`jax.devices` → default device; swallow `jax_platform_name` repin) fixes it. **To make a plugin a
first-class option the fork should recognize the `mps` platform** — a small, autodiff-safe change.

## Track 2C — forward benchmark matrix (DONE; M4 Pro, isotropic, 60 steps, Mcs/s)

| N | JAX-CPU | **jax-mps** (async) | MLX-op cores | **fused Metal kernel** |
|---:|---:|---:|---:|---:|
| 48 | 17.5 | 34.0 | 149.7 | 335.9 |
| 64 | 43.2 | 95.8 | 228.3 | 533.4 |
| 96 | 77.8 | 165.5 | — | 871.9 |
| 128 | 128.9 | 235.9 | — | 1038.5 |

- **jax-mps ≈ 2× JAX-CPU**, steady across N (async dispatch on; `JAX_MPS_ASYNC_DISPATCH=1` ~doubled it).
  Real but modest — an op-by-op StableHLO→MLX translation is dispatch/bandwidth-bound and **cannot fuse
  the stencil**, exactly as hypothesized.
- **Fused Metal kernel ≈ 4.4–5.6× jax-mps** (≈ 8–12× CPU), lead grows with N (absolute climbs toward the
  documented ~1400–1800 Mcs/s floor at N≥128 / 500 steps). The kernel's whole reason to exist is confirmed.
- All engines produced finite fields matching the physics.
- **applejax forward: crashes.** MPSGraph aborts *natively* during `place_objects` (array init) — after
  JIT/donation warnings, no Python traceback, exit 1. Basic FDTD-shaped ops (pad, roll, dynamic_update,
  scatter_add, stacked curl) each run fine under applejax in isolation, so it's a specific deeper op in
  fdtdx's setup graph, not a basic-op gap. Not usable out-of-the-box; root-causing is follow-up.

## Track 2D — inverse-design adjoint (DONE — the pivotal result)

`jax.value_and_grad` of a field-energy loss through `run_fdtd` with `GradientConfig(method="checkpointed")`
— exercises the full FDTD **backward pass** (scatter/gather + control flow), the capability ADR 0001
dropped from the MLX path.

| N, steps | JAX-CPU | **jax-mps** | gradient vs CPU |
|---|---|---|---|
| 32, 40 | 2.53 s | 4.87 s | dL/dscale −1.4573e-8 vs −1.4575e-8; FD rel-err 5e-4 |
| 48, 60 | 3.28 s | 3.90 s | matches; FD rel-err 2e-4 |

- **The reversible-FDTD adjoint runs end-to-end on the Apple GPU under jax-mps, with correct gradients.**
  This is the headline: **jax-mps restores gradient-based inverse design on a Mac** — the exact thing the
  fork punts to CUDA/JAX clusters — running *unmodified* fdtdx, zero fork divergence.
- Perf ≈ CPU-parity (slightly slower at tiny N — compile/dispatch-bound; gap closes by N=48, and the
  forward trend says it turns favorable at larger N). Not a speed story yet; a **capability** story.
- applejax inverse design: not reached (crashes at init, same as forward).

---

## Track 3 — decision (DONE)

**These are not competitors; they are complements.** The fused Metal kernel and jax-mps occupy the two
corners the fork's own thesis predicted:

| axis | fused Metal kernel (fork) | jax-mps (plugin) |
|---|---|---|
| forward throughput | **~4.5–5.6× jax-mps**, at the bandwidth floor | ~2× CPU (can't fuse) |
| autodiff / inverse design | **none** (dropped by design) | **full, correct, on Metal** |
| divergence / maintenance | permanent fork, per-feature parity gate | **zero** — runs unmodified fdtdx |
| jax / engine | MLX, jax 0.10.x | MLX, jax 0.10.x (same!) |

**Recommendation — Option D (hybrid), leaning on jax-mps (not applejax).**
1. **Keep the fused Metal kernel as the forward engine.** It is ~5× faster than any plugin can be on this
   memory-bound stencil; the measurements validate the fork's whole reason to exist. Do **not** retire it,
   and do **not** rebuild the forward path on a plugin.
2. **Adopt jax-mps as an optional, non-intrusive path that restores inverse design on a Mac** — the gap
   ADR 0001 left open. It runs unmodified fdtdx with correct gradients at ~CPU-parity, needs no engine
   code, and shares the fork's jax 0.10.x + MLX. Concretely: teach the fork's `config`/`sharding` to
   recognize the `mps` platform (small, autodiff-safe, upstreamable), then document
   `pip install jax-mps` + `JAX_PLATFORMS=mps` as the "small/medium inverse design on Apple Silicon" recipe.
3. **applejax: not now.** Broader on paper (complex/linalg/MPSGraph) but crashes natively on the fdtdx
   graph and sits on the older jax 0.9.x. Revisit only if a workload needs its complex/linalg breadth; a
   "contribute to improve it" investment is premature while jax-mps already covers forward + adjoint.
4. **Unify & retire the fork? No** — for the user's forward-heavy Apple-Silicon use, retiring the fused
   kernel forfeits ~5×. The realistic unification is narrower and already in hand: *one* `import fdtdx`
   that routes forward to the fused kernel and, when a gradient is requested on a Mac, can run under
   jax-mps instead of only CPU/CUDA.

**Net:** the tip was half-right — jax-mps does give "MLX for free," but ~5× below the fused kernel on
forward, so it doesn't replace the engine. Its real value is the **opposite** end: it hands the fork back
the one thing it gave up (Metal autodiff / inverse design), for free and without divergence.

---

## Deep-dive round 2 (regression, root-causes, broader coverage, the custom_call path)

### Post-merge regression check — none; a **+14%** gain
Rigorous A/B of the fused kernel, pre-merge `10d39ae` vs post-merge, N=128/192 @ 500 steps:
128: 1459 → **1659** (+13.7%); 192: 1586 → **1810** (+14.1%). The kernel MSL is byte-identical, so
this is upstream **#379 "memory-efficient PML"** flowing in — CPML coeffs now sit slab-sized on the PML
objects instead of full `(6,N³)` container arrays, lightening setup/bridge. No per-step regression.

### Why jax-mps lags — structural, **not** autodiff
`jax-mps/src/pjrt_plugin/mlx_executable.cc:868` wraps the whole StableHLO program in
**`mlx::core::compile()`** (+ `async_eval` pipelining). So jax-mps runs the *identical* computation as
the fork's **MLX-op cores** (`mx.compile`-fused MLX ops) — and lands at the same **op-graph ceiling**
(~200–240 Mcs/s, matching the fork's MLX-op path). `compile()` fuses elementwise chains but **cannot**
merge the stencil's neighbour reads or keep the working set on-chip; the fused kernel's ~5× edge is the
**traffic gap** (single-dispatch MSL, 5 RT vs ~21–36 RT), a property of `mx.fast.metal_kernel` that op-
graph compilation structurally can't reach. The forward gap appears with **no autodiff involved**, so the
user's "choked by auto-diff" hypothesis is **disproven** — the cause is fusion granularity, not gradients.

### Can a plugin "hand-roll" like we did? — **yes, via `custom_call`** (the key strategic path)
JAX supports hand-written kernels wrapped for autodiff (`jax.ffi` / a custom primitive with
`custom_vjp`). The plugin just has to route `stablehlo.custom_call` to a kernel. **applejax already has a
`CustomCallRegistry`** (`ops/control_flow_ops.mm:222`, `CustomCallRegistry::Find(target)`) — it's how it
does LAPACK QR/SVD/eigh. **jax-mps has no such hook** (no custom_call/FFI handling in source). So the
"holy grail" — **fused-kernel forward speed + autodiff + one JAX codebase** — is *architecturally
reachable*: register the FDTD update (and its reverse-time adjoint) as `custom_call` targets, wrap them in
`custom_vjp`. It is **not available today**; it is a *contribution* (add custom_call + kernel registration
to the plugin; a Metal/MSL kernel behind the target). jax-mps (MLX, matches the fork's kernel) is the more
natural host for this; applejax already has the registry but uses MPSGraph and currently crashes.

### Why applejax crashes — a fixable library bug, not a verdict
Native **SIGTRAP / EXC_BREAKPOINT** in `jax_mps::MpsExecutable::Execute → CFRelease.cold.2`
(CoreFoundation **double-release / use-after-free**), inside the Apple GPU driver `AGXMetalG16X`, on the
fdtdx init graph. Basic FDTD ops (pad/roll/dynamic_update/scatter/stacked-curl) each run fine in isolation,
so it's a ref-counting bug in applejax's execute path on a non-trivial graph — **fixable** (root-cause the
over-released buffer, rebuild). applejax's forward/inverse perf is therefore **untested**, not ruled out.

### Broader material coverage (M4 Pro, 150–200 steps, Mcs/s) — the anisotropic flip
| material | JAX-CPU | jax-mps | MLX-op cores | fused kernel |
|---|---:|---:|---:|---:|
| isotropic N=128 | 129 | 236 | 195 | **1659** |
| diagonal N=128 | ~70 | 208 | 218 | **1080** |
| **full_aniso N=128** | 70 | **46** | 99 | **99** (falls back to MLX-op) |

- **iso/diagonal:** fused kernel ~5–6× jax-mps, ~8–13× CPU — the fork's strong regime.
- **full-tensor anisotropic (uniform 9-tensor):** the fused kernel **has no in-kernel tensor path** (the
  block hybrid only accelerates *compact* inclusions), so it runs the MLX-op aniso cores → only **~1.4×
  CPU**, and **jax-mps is slower than CPU** (46 vs 70). This is the fork's weakest regime and, per
  `performance-roadmap.md §8`, the **largest untapped win** (per-tile material compaction / an in-kernel
  tensor update). The user's demand for anisotropic coverage surfaced this — it was invisible in the
  iso-only numbers.

### Large-N spill gate (the tiling decision) — **positive**
Fused-kernel throughput vs N (iso, 200 steps): 128=1176, 192=**1395 (peak)**, 256=1307, 384=979,
512=905 — a **~35% drop** from the N=192 peak to N=512. The cache spills on the strided x/y neighbour
lines exactly as `performance-roadmap.md §4` predicts, so the tiled engine's **spatial-tile** component is
load-bearing at large N (the fork's target regime), on top of temporal blocking + material compaction.

### How much cleaner would jax-mps make the fork?
Fork MLX engine + backend = **3004 LOC**: ~**1444** physics/kernel (`kernels/curl/aniso/pml/update/
interpolate` — the *keeper*, wrappable as a `custom_call`) + ~**1548** scaffolding (`bridge/loop/state/
source_freeze/detector_freeze/inject/accumulate/boundary_mask/serialize/metrics` + `backend/dispatch`).
A JAX-native `custom_call` approach retires most of the **1548 scaffolding LOC** (JAX owns tracing,
execution, the sim loop, autodiff) plus the whole element-wise parity harness, while keeping the kernel.
The forward-vs-unification trade is now quantified: unify fully on the op-graph plugin = lose ~5× forward;
unify via `custom_call`-registered kernel = keep the kernel, shed the plumbing, gain autodiff.

### Monitor optimization — where it lives, and upstreaming
The 3.9× monitor win (region-restricted interpolation + activity-gating + DFT auto-subsampling;
`docs/performance.md §Monitor recording`) lives **only in the MLX path** (`mlx/detector_freeze.py`,
`mlx/accumulate.py`, `mlx/interpolate.py`) — the JAX detector classes don't have it. Upstreaming it means
**re-implementing it in fdtdx's JAX detector path, differentiably**, on a branch off *upstream* fdtdx (not
the fork). Region-restriction + activity-gating are exact; DFT subsampling is exact within the
oversampling margin — all three are autodiff-safe and generally useful, so they are strong upstream PRs.

## Track 3 — strategic decision memo (PENDING)
Option matrix (A unify+retire fork / B keep fused kernel + upstream Metal / C improve the sounder
plugin / D hybrid), scored on forward Mcs/s × autodiff × features × unification/divergence cost.
