# ACTION PLAN — Cut the redundant per-step memory traffic, then (maybe) fuse, then broaden

> **For the agent picking this up cold.** Single entry point for the next phase. Read this
> top-to-bottom, then read the evidence it rests on — **[docs/metal-bottleneck-analysis.md](docs/metal-bottleneck-analysis.md)**
> (the measured diagnosis; it supersedes perf-baseline §1a wherever they differ) — and skim
> [CLAUDE.md](CLAUDE.md). You should not need to re-derive any benchmark result.

**Priority order (top = do first):**

1. **PHASE 1 — Cut the redundant per-step memory traffic** (TOP PRIORITY). At N=192 the engine moves
   **~99 full-array DRAM round-trips per step** at the measured **240 GB/s** roofline when only ~5–8
   are physically necessary — i.e. **only ~3% of the moved bytes are useful; the bus is ~85%
   saturated on waste**. Bring ~99 → ~23 round-trips (→ ~440 Mcell·steps/s, ~2.3× JAX-CPU) by
   removing that waste, **with the physics held byte-for-byte** (§P).
2. **PHASE 2 — One fused custom Metal kernel** (`mx.fast.metal_kernel`) to reach the ~5–8 *necessary*
   round-trips (600+ Mcs/s). Conditional on Phase 1's result. This is a *fusion* play, **not** a
   layout change (component-last measured 1.00× — see §0.6).
3. **PHASE 3 — Broaden the supported surface** (PEC/PMC, lossy full-anisotropic, ADE). Independent of
   perf; pull forward if a use case needs it. Build each new term **compile-friendly** (§P).

**Decision gate (read before investing).** The value of the MLX backend is **throughput-only** —
there is *no* memory advantage (JAX-CPU and MLX have ~equal footprint to N=384; §0.6). And the cheap
half of Phase 1 (compile + slab-CPML → ~225 Mcs/s) is only ~1.2× the zero-maintenance JAX-CPU path.
**So: commit to near-full Phase 1 (→ ~440, 2.3× CPU) or don't start** — the decisive win lives in the
*last* fixes (drop padding / ψ-stack), not the first.

---

## 0. Cold-start orientation

### 0.1 What this project is
FDTDMEX is a fork of [fdtdx](https://github.com/ymahlau/fdtdx) (JAX FDTD Maxwell solver) that adds a
native **MLX (Metal) forward backend** for Apple Silicon. A supported forward `run_fdtd` auto-routes
to the MLX time loop; gradients / unsupported features / non-Apple platforms run the unchanged JAX
engine. Import stays `import fdtdx`. Full detail: [CLAUDE.md](CLAUDE.md).

### 0.2 Where the MLX engine lives (`src/fdtdx/mlx/`)
| file | role |
|---|---|
| [`loop.py`](src/fdtdx/mlx/loop.py) | time-loop driver (`run_forward_mlx`) — eager Python `for`, `mx.eval` every 8 steps |
| [`curl.py`](src/fdtdx/mlx/curl.py) | Yee curl + CPML recurrence + field padding (`pad_fields_mlx`, `curl_E_mlx`, `curl_H_mlx`) |
| [`update.py`](src/fdtdx/mlx/update.py) | E/H update — iso/diagonal fast path + full-tensor A/B path |
| [`aniso.py`](src/fdtdx/mlx/aniso.py) | 9-tensor anisotropic helpers |
| [`pml.py`](src/fdtdx/mlx/pml.py) | host-side precompute of time-invariant CPML `a`/`b`/`1/kappa` |
| [`bridge.py`](src/fdtdx/mlx/bridge.py) | ArrayContainer ↔ MLXState (host↔device, once per `run_fdtd`); `_grid_metrics` → `1.0` on uniform grids |
| [`state.py`](src/fdtdx/mlx/state.py) | `MLXState` dataclass (fields the loop carries) |
| [`inject.py`](src/fdtdx/mlx/inject.py) / [`source_freeze.py`](src/fdtdx/mlx/source_freeze.py) | source injection + host-frozen plans |
| [`accumulate.py`](src/fdtdx/mlx/accumulate.py) / [`detector_freeze.py`](src/fdtdx/mlx/detector_freeze.py) | detector recording + frozen plans |
| dispatch: [`backend/dispatch.py`](src/fdtdx/backend/dispatch.py) | routing + milestone gating (`_run_mlx_forward` builds state→plans→loop) |
| force backend: [`backend/context.py`](src/fdtdx/backend/context.py) | `fdtdx.use_backend("mlx"\|"jax")` |

Pristine upstream reference (read-only): **`../fdtdx`** — esp. `core/physics/curl.py` (inline CPML
`a`/`b`) and `fdtd/update.py` (E/H + ADE + anisotropic update).

### 0.3 Hardware (M4 Pro, this machine)
- **Measured coalesced bandwidth ~240 GB/s** (88% of the 273 GB/s spec) — this is the roofline
  denominator, not 273. (`mx.device_info()` exposes no bandwidth field.)
- `max_recommended_working_set_size` = **40.2 GB** (GPU ceiling); `memory_size` 51.5 GB total;
  `max_buffer_length` 30.1 GB (single-array cap; matters only beyond ~N≈1000). `applegpu_g16s`, MLX 0.31.2.

### 0.4 The measured diagnosis (what Phase 1 acts on)
The full evidence is **[docs/metal-bottleneck-analysis.md](docs/metal-bottleneck-analysis.md)**. The
one model to carry in your head — the **round-trip (RT) budget** (1 RT = read+write of one `(3,N³)`
field; per step, N=192, isotropic, CPML on; reconciles across N=96/192/256):

```
  ~99 RT  eager, CPML on .......... today (~100 Mcs/s)               <- bus ~85% saturated on waste
  ~62 RT  + mx.compile ............ fuses intermediate RT (1.5x)
  ~44 RT  + slab-CPML ............. removes carried-psi RT compile CANNOT fuse (-> 2.1x stacked)
  ~23 RT  + drop pad / psi-stack .. the lean ceiling (~440 Mcs/s, ~2.3x JAX-CPU)
  ~5-8 RT  one fused custom kernel . the necessary floor (Phase 2, 600+ Mcs/s)
```
The bottleneck is **redundant memory traffic** (too many full-array kernels each doing a DRAM
round-trip), confirmed by toggling CPML in the real loop (removes exactly ~24 RT) and by the
predictive check (compiling the *real* body gives 1.5× and **stalls at ~62 RT** because the CPML ψ
and the ψ-stack are *carried state* that must round-trip — fusion can't remove them). It is **not**
bandwidth-ceiling, **not** dispatch-starvation, **not** layout/coalescing.

### 0.6 Rejected — do NOT spend time on these (measured)
- **Component-last `(N,N,N,3)` layout** (was Phase-2 Task 2.2): stencil measured **1.00×** vs
  `(3,N,N,N)`; coalesced copy already hits 88% of peak. No coalescing penalty to fix.
- **"JAX-CPU can't fit large domains / uses 3–4× memory":** false. Subprocess-isolated footprints are
  ~equal to N=384 (JAX 28.0 GB vs MLX 28.6 GB). No memory moat — the MLX case is throughput-only.
- **"Plateau won't improve with N (CPML waste grows with N)":** CPML is a *constant* ~25% of traffic
  at all N; throughput plateaus (~100), doesn't collapse. slab-CPML is a steady ~1.4×, not N-growing.
- **`roll`→slice as a speedup for the difference itself:** ~1.0× (`y - roll(y)` ≈ slice-diff). The
  win in "drop padding" is removing the `mx.pad` *copies*, not roll-vs-slice.
- **Dispatch / `eval_every` / `async_eval` tuning:** minor; `eval_every=8` is already at plateau
  (sync-every-step costs only ~20%).

---

## P. Physics-correctness contract (non-negotiable for EVERY Phase-1/2 change)

Every fix below is a **pure performance transform**: it must reduce round-trips while producing
**byte-for-byte (rel < 1e-3) the same fields** as the JAX-CPU oracle. Specifically:

1. **Functional / out-of-place — no races.** MLX updates compute *new* arrays and return them; this
   is what makes the Yee update race-free (no ping-pong buffers, no atomics). Do **not** introduce
   in-place mutation or buffer aliasing (incl. inside a custom kernel) that lets a cell's write be
   read by a neighbor's update in the same pass. **Key asymmetry (drives the kernel design):** the
   *isotropic* E-update reads only its own `E_old[i]` + `H` neighbors (no E-neighbor read → in-place
   safe), but the *anisotropic* E-update's off-diagonal terms read `E_old` at **neighboring cells**
   (the other components averaged to this component's Yee site) — so the anisotropic kernel **must
   double-buffer E** (read old from the input buffer, write new to a separate output) and load an
   E-halo. Same for H↔μ-tensor. This is *why* region/cube specialization (cheap isotropic path where
   the cell is isotropic) is both faster and simpler.
2. **Leapfrog order preserved.** The step is `update_E (reads Hⁿ⁻½) → inject E-sources → update_H
   (reads the just-updated Eⁿ⁺½, source included) → inject H-sources → record detectors`
   ([loop.py](src/fdtdx/mlx/loop.py)). When you compile, compile **two cores** (E-core, then H-core)
   with the host-side source injection *between* them — never merge E and H into one pass or compute
   both from the same time level. The H update must see the source-injected E.
3. **Source/detector gating stays host-side.** The compiled core is the pure all-cell math (curl →
   field update → CPML recurrence); per-step `float(coeff[n])`/`bool(on_steps[n])` (numpy, host-only)
   stay outside it so the compiled graph is static across steps.
4. **slab-CPML must equal full-domain ψ.** ψ ≡ 0 outside the PML slabs *by construction* (a=b=0
   there), so restricting it to slabs is mathematically identical — parity must be exact, not
   approximate. Validate a PML-on-some-faces case too.
5. **Tolerance is fixed.** Marginal parity failure → raise resolution, never loosen tolerance. Beware
   float32 traps (CLAUDE.md "Coding conventions").

---

## 0.7 Validation + measurement protocol (run after EVERY edit)

**Correctness (the bar):**
```bash
uv run --with pytest pytest tests/validation/test_mlx_parity.py -q       # uniform-grid parity
uv run --with pytest pytest tests/validation/test_mlx_nonuniform.py -q   # NON-UNIFORM parity + convergence
uvx ruff format src/fdtdx/mlx src/fdtdx/backend && uvx ruff check src/fdtdx/mlx src/fdtdx/backend
```
- `test_mlx_nonuniform.py` is the tripwire for the metric/×1 work (the metric is a real per-axis
  array there, not `1.0`); `test_mlx_parity.py` covers the uniform + periodic + CPML paths.

**Performance (auditing each fix removes its predicted RT):**
```bash
uv run python benchmarks/profile_engine.py --N 192 --steps 200   # per-step RT + 2x2 compile×CPML
uv run python benchmarks/profile_metal.py  --N 192 --iters 100   # roofline + dispatch (rarely changes)
uv run python benchmarks/bench_forward.py --backends mlx,jax \
  --materials isotropic,diagonal,full_aniso --sizes 96,128,192,256 --steps 250 --repeats 2 \
  --out benchmarks/results/<name>.jsonl                          # the tracked figure
```
- After each fix, `profile_engine.py`'s **RT/step must drop by the predicted amount** (the model is
  in §0.4). If it doesn't, stop — the fix didn't do what the diagnosis says, and the model needs
  revisiting before continuing.
- Benchmark at **≥200 steps** (short runs are dominated by the once-per-call bridge).

**Workflow.** Git fork (`origin=Jinge-Wang/FDTDMEX`, `upstream=ymahlau/fdtdx`); MLX backend is
additive. One fix = one commit, with its before/after RT and throughput in the message.

---

## PHASE 1 — Reduce the waste (ordered; each step: implement → §P parity → §0.7 RT audit → commit)

Goal: ~99 → ~23 RT/step (~440 Mcs/s, ~2.3× JAX-CPU). Order is by payoff-per-risk; the two big levers
are **compile** and **slab-CPML**.

> **Status (branch `mlx-fork`):** **Fix 1.1 + 1.3 landed together** (`curl.py` pad-free slice-diff +
> guards; `update.py` split into pure cores; `loop.py` compiles E-core/H-core with host-gated
> injection between). Measured N=192 iso: default path **105 → 211 Mcs/s (2.0×, now > JAX-CPU ~190)**,
> 99 → **47 RT/step**; all 14 validation tests green (physics held). **Remaining: Fix 1.2 slab-CPML** —
> the ~17 carried-ψ RT compile can't fuse; `profile_engine` shows CPML-off at **338 Mcs/s / 30 RT**,
> i.e. slab-CPML is the lever from 211 → ~330 (~1.8× CPU) toward the ~440 lean ceiling.

**Fix 1.1 — `mx.compile` the per-step core (~1.5×; lowest risk; do first).**
- *Targets:* ~37 RT of fusable intermediates (curl differences, the `inv_kappa` combine, the
  field-update arithmetic, and the `*1.0` metric which compile constant-folds).
- *Do:* wrap the pure math of **update_E** (curl_H → E update + CPML recurrence) and **update_H**
  (curl_E → H update + CPML recurrence) as two `mx.compile`d functions; keep source injection
  host-gated *between* them (§P.2/P.3). Inputs/outputs: E, H, ψ_E, ψ_H (+ materials & CPML `a`/`b`/`ik`
  & metric as captured constants); pass `c` as a compiled arg. Keep the signature stable so it does
  not re-trace (watch shape/key changes).
- *Tripwire:* parity (uniform+nonuniform+periodic+CPML) unchanged; `profile_engine.py` shows
  CPML-on RT ~99 → ~62.

**Fix 1.2 — slab-CPML: carry/advance ψ only on the boundary slabs (~1.5× more → ~330 Mcs/s). ⭐ NEXT**
- *Targets:* the ~17 carried-ψ RT compile can't fuse (realized: compiled CPML-on **47 RT** vs CPML-off
  **30 RT**). ψ_E/ψ_H are full `(6,N³)` but `a=b=0` **and `inv_kappa=1`** except in the ~8-cell PML
  slabs, so the whole ψ machinery is zero-valued traffic over ~77% of the domain.
- *Approach (exact — decompose, don't approximate):* outside the PML, the curl combine is just the
  plain difference `d_a − d_b` (since `inv_kappa=1`, `ψ=0`). Inside a slab it is
  `(ik·d_a + ψ_a) − (ik·d_b + ψ_b)`. So compute `curl0 = d_a − d_b` **full-domain (cheap)**, then ADD a
  **slab-localized correction** `(ik−1)·d + ψ` that is nonzero only on the slabs. Each ψ component is
  tied to one axis (`a[0]↔x, a[1]↔y, a[2]↔z`) and lives only in that axis's low+high slabs (e.g.
  `(2·p, N, N)` for x). Carry ψ as those small slab arrays in `MLXState`; advance the `b·ψ + a·d`
  recurrence and build the correction on **slab slices of `d`** only.
- *Geometry:* detect each axis's slab extent on the host at bridge time from the nonzero support of
  `b−1` / `a` (`b=1, a=0` outside PML) — handles PML-on-some-faces and asymmetric thickness exactly.
- *Risk (highest in Phase 1):* slab indexing / off-by-one. It is a mathematical no-op (§P.4) → parity
  must be **EXACT**. Add a **PML-on-some-faces** parity case (only a subset of the 6 faces have PML).
- *Tripwire:* `profile_engine.py` compiled CPML-on RT **47 → ~30**; parity exact; `bench_forward`
  iso/diag/aniso each improve ~1.5×.

**Fix 1.3 — drop per-step field padding + don't materialize the ψ-stack (~1.9× more → ~23 RT, ~440).**
- *Targets:* the remaining ~21 RT — the 6 `mx.pad` full-field copies/step
  ([pad_fields_mlx](src/fdtdx/mlx/curl.py)) and the unconditional `mx.stack([…6 ψ…])`
  ([curl.py:95,146](src/fdtdx/mlx/curl.py#L95)) that rebuilds a `(6,N³)` array even when ψ is
  untouched (and is moot once ψ is slab-sized from 1.2).
- *Do:* compute differences by slicing (`f[1:] - f[:-1]`) with an explicit single ghost cell — **zero
  for PML/PEC axes, wrap for periodic** (reproduce the exact values [pad_fields_mlx](src/fdtdx/mlx/curl.py#L30)
  / [`_wrap_pad_axis`](src/fdtdx/mlx/curl.py#L21) produce). Return/assemble ψ only when it changed.
- *Risk:* the single boundary cell is where ghost-value bugs hide. Validate the **periodic** test
  (`test_periodic_boundaries_match_jax`) and CPML parity.
- *Tripwire:* `profile_engine.py` RT → ~23; `bench_forward.py` large-N MLX > JAX-CPU for iso/diag.

**Phase 1 done when:** large-N MLX throughput exceeds JAX-CPU for iso/diag (ideally full_aniso); all
validation suites green; `docs/metal-bottleneck-analysis.md` RT table updated with realized numbers +
a fresh `bench_forward` figure; roadmap WS-A row updated.

---

## PHASE 2 — Custom Metal update kernels (the deep lever; fp32; SEPARATE E and H kernels)

**Why:** even the lean compiled step is ~23 RT — 3–4× above the ~5–8 *necessary* — because MLX
op-fusion can't keep a stencil's working set in threadgroup memory or reuse neighbour reads across the
sub-ops; each op still round-trips DRAM. Past ~440 Mcs/s the only lever is hand-written kernels via
`mx.fast.metal_kernel` (confirmed 0.31.2). Target the ~5–8 RT floor → **~400–600 Mcs/s (~3× CPU)**.
**fp32 is the floor — no mixed precision.** **Do this only after Phase 1 lands.**

**"Fused" means fusing the ~20 ops *within one field update* into one kernel — NOT fusing E and H
together** (leapfrog forbids that; H must read the completed E). So: one **E-update kernel** and one
**H-update kernel**.

- *2.5D plane-marching with threadgroup memory.* Each threadgroup owns an XY tile and marches along z,
  keeping the resident z-plane(s) it needs in threadgroup memory so each value is read from DRAM ~once
  (the classic CUDA shared-memory stencil — Metal threadgroup memory = CUDA shared memory; threadgroup
  = block). The Yee curl is **one-sided** (cell `i` needs `i−1` *or* `i+1`, not both) → a 1-cell halo
  on one side per axis (a lean tile).
- *Race / buffering (§P.1).* E-kernel reads H neighbours, writes E. **Isotropic** E reads only its own
  `E_old` → in-place safe, no E-halo. **Anisotropic** E reads `E_old` at neighbours (off-diagonal
  averaging) → **must double-buffer E + load an E-halo**. Same for H↔μ.
- *Region / cube subdivision.* Today the engine runs ONE global path — if any cell is 9-tensor the
  **whole** domain pays the anisotropic neighbour-averaging. Tag cells by material class and run the
  cheap isotropic kernel on isotropic sub-blocks, the full per-cell 3×3 A/B kernel only on anisotropic
  ones (per-block branch or separate launches over index sets). Big win for **heterogeneous** devices
  (the target use case), and it removes needless double-buffering/halo where the cell is isotropic.
- *CPML.* ψ recurrence runs only on the boundary-slab threadgroups — **slab-CPML (Fix 1.2) carries over
  directly**; its slab geometry is exactly what the kernel's boundary path needs.
- *Scope.* Start with the **isotropic-uniform interior** kernel (the common path, ~77% of cells); keep
  the MLX-ops path as fallback for diagonal/anisotropic/non-uniform/CPML (a hybrid is fine). Extend per
  measured payoff.

- *Validation:* same `test_mlx_parity.py` element-wise (add a dedicated kernel test); gate behind a
  flag, fall back to MLX-ops until parity-clean; measure vs the compiled MLX-ops step; deliver a
  go/no-go. **Not** a layout change — component-last is 1.00× (§0.6).

---

## PHASE 3 — Broaden the supported surface (independent of perf)

Follow **[docs/widening-mlx-port-plan.md](docs/widening-mlx-port-plan.md)**. Order (ascending effort),
each = translate kernel → host-precompute invariants → thread state/loop → un-gate in
[dispatch.py](src/fdtdx/backend/dispatch.py) → element-wise parity test. **Build each new per-step
term compile-friendly (§P): host-side gating, arrays carried as state** so it survives Fix 1.1.

1. **Lossy full-anisotropic + 9-tensor conductivity** — lowest effort; the A/B kernel already threads
   σ. Un-gate the two returns in `_unsupported_reason_arrays` + add a parity test (keep off-diagonals
   ≤0.5; strong off-diagonals are unstable in both backends).
2. **PEC / PMC boundaries** — per-step tangential-component masking (precompute a `(3,N³)` keep-mask,
   multiply after the E (PEC) / H (PMC) update + source injection); un-gate `pec_objects/pmc_objects`.
3. **Drude–Lorentz dispersion (ADE)** — host-precompute `c1/c2/c3`, carry `P_curr`/`P_prev` in
   `MLXState`, add `E += inv_eps·Σ(P_curr − P_new)` in the iso/diagonal branch; un-gate `dispersive_c1`.

---

## Quick reference — commands
```bash
uv sync
uv run python -c "import fdtdx, mlx.core, jax"                                  # import sanity
uv run --with pytest pytest tests/validation -q                                 # parity (uniform + non-uniform)
uv run python benchmarks/profile_engine.py --N 192 --steps 200                 # per-step RT + 2x2 (the Phase-1 audit)
uv run python benchmarks/profile_metal.py  --N 192 --iters 100                 # roofline + dispatch
uv run python benchmarks/bench_forward.py --backends mlx,jax --sizes 96,128,192,256 --steps 250 --repeats 2
uvx ruff format src/fdtdx/mlx src/fdtdx/backend && uvx ruff check src/fdtdx/mlx src/fdtdx/backend
```
Force a backend: `with fdtdx.use_backend("mlx"|"jax"):` or `FDTDMEX_BACKEND=mlx|jax`. JAX (CPU) is the
oracle; MLX is Metal GPU.
