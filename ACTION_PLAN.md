# ACTION PLAN — Metal performance first, then deep GPU, then broaden the surface

> **For the agent picking this up cold.** This is the single entry point for the next phase of work.
> Read this file top-to-bottom; it embeds the measured findings and points to every doc/file you
> need, so you should **not** need to re-derive the benchmark results or re-read prior chat history.
> Then skim [CLAUDE.md](CLAUDE.md) and the three deep docs linked in §0.4 before touching code.

**Priority order (top = do first):**

1. **PHASE 1 — Fix the eager Metal plateau** (TOP PRIORITY). The forward engine tops out at
   ~100 Mcell·steps/s on an M4 Pro while using only ~3% of memory bandwidth. ~4× is sitting on the
   floor as redundant per-step work. Re-verify each cause, then fix one-by-one with parity held.
2. **PHASE 2 — Deep GPU feasibility study** (memory coalescing, layout, a hand-rolled Metal stencil
   kernel). Investigate whether we can push past the ~4× op-fusion ceiling. Analysis + prototype +
   go/no-go, *before* committing to a big rewrite.
3. **PHASE 3 — Broaden the supported surface** (PEC/PMC, lossy full-anisotropic, ADE dispersion).
   Low-effort ports already scoped in [docs/widening-mlx-port-plan.md](docs/widening-mlx-port-plan.md).

Phases are ordered by priority but **Phase 3 is independent** — if a use case needs PEC/ADE before
the perf work lands, pull it forward (it doesn't touch the perf hot path).

---

## 0. Cold-start orientation

### 0.1 What this project is
FDTDMEX is a fork of [fdtdx](https://github.com/ymahlau/fdtdx) (JAX FDTD Maxwell solver) that adds a
native **MLX (Metal) forward backend** for Apple Silicon. On a Mac a supported forward `run_fdtd`
auto-routes to the MLX time loop; gradients / unsupported features / non-Apple platforms run the
unchanged JAX engine. Import stays `import fdtdx`. Full detail: [CLAUDE.md](CLAUDE.md).

### 0.2 Where the MLX engine lives (`src/fdtdx/mlx/`)
| file | role |
|---|---|
| [`loop.py`](src/fdtdx/mlx/loop.py) | the time-loop driver (`run_forward_mlx`) — eager Python `for`, `mx.eval` every 8 steps |
| [`curl.py`](src/fdtdx/mlx/curl.py) | Yee curl + CPML recurrence + field padding (`pad_fields_mlx`, `curl_E_mlx`, `curl_H_mlx`) |
| [`update.py`](src/fdtdx/mlx/update.py) | E/H update — iso/diagonal fast path + full-tensor A/B path |
| [`aniso.py`](src/fdtdx/mlx/aniso.py) | 9-tensor anisotropic helpers (analytic 3×3 inverse, A/B matrices, off-diagonal averaging) |
| [`pml.py`](src/fdtdx/mlx/pml.py) | host-side precompute of time-invariant CPML `a`/`b`/`1/kappa` |
| [`bridge.py`](src/fdtdx/mlx/bridge.py) | ArrayContainer ↔ MLXState (host↔device, once per `run_fdtd`) |
| [`state.py`](src/fdtdx/mlx/state.py) | `MLXState` dataclass (fields the loop carries) |
| [`inject.py`](src/fdtdx/mlx/inject.py) / [`source_freeze.py`](src/fdtdx/mlx/source_freeze.py) | source injection + host-frozen source plans |
| [`accumulate.py`](src/fdtdx/mlx/accumulate.py) / [`detector_freeze.py`](src/fdtdx/mlx/detector_freeze.py) | detector recording + frozen detector plans |
| [`interpolate.py`](src/fdtdx/mlx/interpolate.py) / [`metrics.py`](src/fdtdx/mlx/metrics.py) | Yee interpolation + energy/Poynting metrics |
| dispatch: [`src/fdtdx/backend/dispatch.py`](src/fdtdx/backend/dispatch.py) | routing + milestone gating (`select_backend`, `_unsupported_reason*`) |
| force backend: [`src/fdtdx/backend/context.py`](src/fdtdx/backend/context.py) | `fdtdx.use_backend("mlx"\|"jax")` |

Pristine upstream reference (read-only, for porting parity): **`../fdtdx`** — especially
`../fdtdx/src/fdtdx/core/physics/curl.py` (the inline CPML `a`/`b`) and
`../fdtdx/src/fdtdx/fdtd/update.py` (E/H + ADE + anisotropic update).

### 0.3 Hardware facts (M4 Pro, this machine) — from `mx.device_info()`
- Memory bandwidth ~**273 GB/s** (shared CPU+GPU unified memory).
- `memory_size` = 51.5 GB total, but `max_recommended_working_set_size` = **40.2 GB** (the practical
  GPU ceiling; `mx.metal.set_wired_limit(...)` can push it higher at your own risk).
- `max_buffer_length` = **30.1 GB** — a *single* `mx.array` cannot exceed this (matters only for
  domains beyond ~N≈1000; not a near-term constraint).
- `architecture` = `applegpu_g16s`.

### 0.4 The three docs that already hold the detail (read before coding)
- **[docs/perf-baseline.md](docs/perf-baseline.md)** — the measured baseline + **§1a "Why does MLX
  plateau"** (the root-cause analysis Phase 1/2 act on). Has the numbers; don't re-run to learn them.
- **[docs/widening-mlx-port-plan.md](docs/widening-mlx-port-plan.md)** — the full Phase 3 recipe.
- **[docs/roadmap.md](docs/roadmap.md)** — overall status; WS-A "next" mirrors this plan.
- Methodology: [docs/perf-eval-plan.md](docs/perf-eval-plan.md). Skills:
  [`.claude/skills/porting-from-fdtdx`](.claude/skills/porting-from-fdtdx),
  [`fdtdmex`](.claude/skills/fdtdmex), [`physics-validation`](.claude/skills/physics-validation).

### 0.5 The measured baseline you're improving (M4 Pro, float32, 250 steps)
Large-N throughput (Mcell·steps/s): MLX **~97** iso / **~96** diag / **~61** full_aniso; JAX-CPU
**~190** / **~193** / **~96**. So at scale **JAX-CPU is ~2× faster than eager MLX** — that is the bar
to beat. Isolated-kernel measurement: real `update_E`+`update_H` = **104 Mcs/s**; a lean+compiled
version of the same math = **443 Mcs/s** → **~4.3× headroom**, and even that uses only ~50 GB/s (so
it is *not* bandwidth-bound — see Phase 2). Full decomposition table is in perf-baseline.md §1a.

---

## 0.6 Shared protocol — VALIDATION and MEASUREMENT (read once, apply in every phase)

**Correctness bar (non-negotiable).** Every change must keep the element-wise parity vs the JAX-CPU
oracle. After *any* edit to the engine:
```bash
uv run --with pytest pytest tests/validation/test_mlx_parity.py -q      # uniform-grid parity
uv run --with pytest pytest tests/validation/test_mlx_nonuniform.py -q  # NON-UNIFORM parity + convergence
uvx ruff format src/fdtdx/mlx src/fdtdx/backend && uvx ruff check src/fdtdx/mlx src/fdtdx/backend
```
- The non-uniform suite is the **tripwire for Phase 1 fix #1** (the metric `*1.0` skip must NOT
  change non-uniform grids, where the metric is a real per-axis array, not 1.0).
- Parity tolerance is `rel < 1e-3` (float32). **Marginal failure → raise resolution, never loosen
  the tolerance.** Beware float32 traps (see CLAUDE.md "Coding conventions").
- Cross-check physics: `tests/visualization/` (birefringence, non-uniform convergence) and fdtdx's
  own physics tests auto-routed to MLX should still pass.

**Performance measurement (the only numbers that count).**
```bash
# full sweep, both backends, the figure the project tracks:
uv run python benchmarks/bench_forward.py --backends mlx,jax \
  --materials isotropic,diagonal,full_aniso --sizes 32,48,64,96,128,160,192 \
  --steps 250 --repeats 2 --out benchmarks/results/<name>.jsonl
uv run python benchmarks/plot_results.py benchmarks/results/<name>.jsonl

# fast inner-loop A/B while iterating (isolates the step, no JAX):
uv run python benchmarks/microbench_fusion.py --N 192 --iters 60
```
- Always compare against the committed baseline `benchmarks/results/matched_s250.jsonl`.
- Benchmark at **≥200 steps** (short runs are dominated by the per-call bridge; see perf-baseline.md
  §1a "What I ruled out"). Report median throughput + MLX peak memory.
- Profiling deeper: `mx.metal.start_capture(path)` / `mx.metal.stop_capture()` to capture a GPU
  trace (open in Xcode → Metal System Trace) for kernel counts / occupancy / bandwidth; Xcode
  Instruments "Metal System Trace" for timeline. `mx.get_peak_memory()` for memory.

**Workflow.** This is a git fork (`origin=Jinge-Wang/FDTDMEX`, `upstream=ymahlau/fdtdx`); the MLX
backend is additive. Work on a branch; keep each Phase-1 fix a separate commit with its before/after
throughput in the message so the wins are auditable.

---

## PHASE 1 — Fix the eager Metal plateau  ⭐ TOP PRIORITY

**Goal:** lift large-N MLX throughput from ~100 to ~300–440 Mcell·steps/s (above JAX-CPU's ~190) by
removing redundant per-step memory traffic — *no new physics, parity held throughout*.

**Step 1.0 — Re-audit before fixing (the user runs this on a high-effort model).** Do not take the
cause list on faith. Independently:
- Re-read the hot path: [`loop.py`](src/fdtdx/mlx/loop.py) → [`update.py`](src/fdtdx/mlx/update.py)
  → [`curl.py`](src/fdtdx/mlx/curl.py), counting, per time step, every `mx.*` op that allocates or
  copies a full `(C, N, N, N)` array (pad, roll, slice, stack, elementwise). Confirm the ~130
  kernels/step estimate or correct it.
- Reproduce the decomposition table (perf-baseline.md §1a) with `microbench_fusion.py` and the
  real-kernel isolation. Capture a `start_capture` GPU trace at N=192 to see the actual kernel count
  and per-kernel bandwidth — this is ground truth, the static count is an estimate.
- Confirm the "ruled out" items still hold (no per-step host sync in `inject.py`/`accumulate.py`;
  bridge is once-per-call). Look especially for any **`.item()` / `float(mx.array)` / `bool(mx.array)`
  / `np.asarray(mx.array)`** inside the per-step path (would force a device sync) — grep the loop and
  everything it calls. Add any newly-found cause to the list below.

Then fix in this order (cheapest/highest-payoff first), re-running the validation + microbench after
each so each win is isolated and parity-checked:

**Fix 1.1 — Drop no-op elementwise work on uniform / interior grids (~2×, cheapest).**
- *Cause:* in [`curl.py`](src/fdtdx/mlx/curl.py) each difference is `* my_`/`* mx_`/`* mz_`
  (lines 71–76 and 122–127). On a **uniform grid the metric is the Python scalar `1.0`** (set in
  [`bridge.py`](src/fdtdx/mlx/bridge.py) `_grid_metrics`, line ~48) → 12 multiply-by-1 full-array
  kernels/step. The curl combine (lines 97–100, 148–151) multiplies by `inv_kappa` (=1 outside PML)
  and adds ψ (=0 outside PML); and `psi_*_updated = mx.stack([...6...])` (lines 95, 146) **rebuilds a
  (6,N,N,N) array every step even when `simulate_boundaries` leaves ψ untouched.**
- *Fix:* guard the scalar case — `if isinstance(m, float) and m == 1.0: skip the multiply`. Only
  build/return the ψ stack when ψ actually changed; when `simulate_boundaries` is False (or a given
  axis has no PML) don't add ψ into the curl. Keep the non-uniform / PML paths byte-identical.
- *Tripwire:* `test_mlx_nonuniform.py` must stay green (proves the scalar guard didn't touch the
  metric-array path). `test_mlx_parity.py` proves the uniform path is unchanged.

**Fix 1.2 — `mx.compile` the per-step body (~1.4×).**
- *Cause:* [`loop.py`](src/fdtdx/mlx/loop.py) is eager; every op is its own Metal kernel, every
  intermediate streams to DRAM.
- *Fix:* wrap the pure-array core of one step (curl→update→CPML) in `mx.compile`. Per
  [perf-eval-plan.md §8](docs/perf-eval-plan.md): pass `time_step` + amplitude **scalars as compiled
  arguments**; keep source/detector **gating host-side** (skip inactive sources/detectors outside the
  compiled core, so the compiled graph is static across steps). Inputs/outputs are E, H, ψ_E, ψ_H
  (+ material arrays as captured constants). Watch: `mx.compile` re-traces if input shapes/keys
  change — keep the signature stable. Verify the compiled function's outputs match eager to 1e-3.
- *Note:* compile may subsume part of Fix 1.1 (constant-folding `*1.0` if the metric is a literal),
  but do 1.1 first — explicit guards are robust and help the eager path too.

**Fix 1.3 — Remove per-step field padding (~1.3×).**
- *Cause:* [`pad_fields_mlx`](src/fdtdx/mlx/curl.py) (lines 30–46) does 3 sequential `mx.pad` calls
  (a full-array copy each) **per field per step**; the curl then `roll`s the padded copy and slices
  it back (`[1:-1,1:-1,1:-1]`).
- *Fix:* compute differences directly with slicing (`f[1:] - f[:-1]`) and handle the single boundary
  cell explicitly (zero ghost for PML/PEC axes, wrap for periodic axes — see `_wrap_pad_axis`,
  lines 21–27, for the periodic rule). Reproduce the exact ghost-cell values the pad produced.
  Validate parity incl. the periodic test (`test_periodic_boundaries_match_jax` in
  `test_mlx_parity.py`).

**Fix 1.4 — Restrict CPML to the boundary slabs (~1.36×, and it grows with N).**
- *Cause:* ψ_E/ψ_H (12 × N³ arrays) and the `a`/`b` recurrence run over **every** cell, but
  `a=b=0` outside the ~8-cell PML slabs. At N=192 ~77% of cells do zero-valued ψ math. This is *why
  the plateau doesn't improve with N*.
- *Fix (bigger refactor, do last):* store ψ only for the 6 boundary slabs (each ~`pml_thickness ×
  N²`), update the recurrence and add ψ into the curl **only there**. Needs the per-boundary slab
  geometry (available from the boundary objects / the placed `arrays.alpha/kappa/sigma` nonzero
  support). Higher parity risk — validate carefully, including a case with PML on only some faces.
- *Alternative if slabs are too invasive:* keep full-domain ψ but skip the recurrence where
  `b==0` via the compiled kernel's masking — smaller win, lower risk.

**Phase 1 done when:** large-N MLX throughput in `bench_forward.py` exceeds JAX-CPU for iso/diag and
ideally full_aniso; all validation suites green; perf-baseline.md updated with the new numbers and a
fresh figure; roadmap WS-A row updated.

---

## PHASE 2 — Deep GPU feasibility study (memory coalescing / hand-rolled Metal)

**Why:** even the lean **compiled** step reaches only ~50 GB/s of 273 (perf-baseline.md §1a "deeper
ceiling"). Op-level fusion cannot merge a stencil's neighbor reads, and the `(3, N, N, N)`
component-leading layout makes `roll`/interior-slice access **strided / uncoalesced**. Past ~4×, the
ceiling is access-pattern and kernel-shape, not arithmetic. This phase decides whether a deeper
rewrite is worth it.

This is **study + prototype + go/no-go**, not a commitment to rewrite. Deliver a short
`docs/metal-deep-perf.md` with findings and a recommendation.

**Task 2.1 — Quantify the access-pattern ceiling.** Use `mx.metal.start_capture` GPU traces (and/or
Xcode Metal System Trace) on the compiled lean step to measure achieved bandwidth, occupancy, and
per-kernel time. Microbench a pure copy / pure `roll` / pure 7-point stencil at N=192/256 to find the
practical sustained bandwidth for *coalesced* vs *strided* access on this GPU. Establishes the real
roofline (the % of 273 GB/s actually attainable for stencils).

**Task 2.2 — Layout experiment.** Prototype a **component-last `(N, N, N, 3)`** (or fully separate
per-component arrays) layout for E/H/ψ and re-measure the stencil. Component-last can make the
3-vector contiguous per cell (better coalescing) but changes every kernel and **breaks byte-for-byte
fdtdx parity ordering** — so this is a measurement to justify (or kill) the idea, behind a flag, not
a default. Quantify the speedup vs the parity/maintenance cost.

**Task 2.3 — Hand-rolled Metal stencil kernel (the headline option).** Prototype the curl+update (and
ideally the CPML recurrence) as a **single fused custom kernel** via
**`mx.fast.metal_kernel(name, input_names, output_names, source=..., ...)`** (confirmed available in
mlx 0.31; verify the exact signature against the installed version). One kernel that reads E/H tiles
into threadgroup memory and writes the updated fields in one pass would collapse the ~130
kernels/step to ~1–2 and do a single DRAM round-trip — the path to a large multiple of the current
throughput. Scope: start with the isotropic uniform case (the common path), measure vs the compiled
MLX-ops version, then decide whether to extend to diagonal/anisotropic/non-uniform/CPML or keep the
MLX-ops path for those (a hybrid is fine).
- *Validation:* the custom kernel must pass the **same** `test_mlx_parity.py` element-wise (add a
  dedicated test). Custom Metal is the highest-risk/highest-reward item — gate it behind a flag and
  fall back to the MLX-ops path until it's parity-clean.

**Task 2.4 — Go/no-go.** Recommend one of: (a) MLX-ops + `mx.compile` is enough (Phase 1 hit the
target) — stop; (b) ship the hand-rolled isotropic kernel + MLX-ops fallback for the rest; (c) full
custom-kernel engine. Decide on measured speedup vs maintenance/parity cost.

**Cross-cutting Phase-2 ideas to also weigh:** persistent `MLXState` across steps (build device state
once, step thousands of times, bridge out once — amortizes the per-call bridge, relevant for real
runs; today `_run_mlx_forward` rebuilds state every `run_fdtd`); larger/auto `eval_every` or
`mx.async_eval` to cut CPU-side stalls; in-place buffer reuse where MLX allows.

---

## PHASE 3 — Broaden the supported surface (low-effort ports)

Fully specified in **[docs/widening-mlx-port-plan.md](docs/widening-mlx-port-plan.md)** — follow it
directly. Order (ascending effort), each = translate kernel → host-precompute invariants → thread
state/loop → un-gate in [`dispatch.py`](src/fdtdx/backend/dispatch.py) → element-wise parity test:

1. **Lossy full-anisotropic + 9-tensor conductivity** — *lowest effort: the MLX A/B kernel already
   threads σ* ([`update.py`](src/fdtdx/mlx/update.py) `_update_aniso` →
   [`aniso.py`](src/fdtdx/mlx/aniso.py) `compute_anisotropic_update_matrices_mlx`). Just **un-gate**
   the two returns in `_unsupported_reason_arrays` (dispatch.py lines ~109–114) + add a parity test.
   Caveat: strong off-diagonals are unstable in *both* backends (roadmap "Quirk A") — keep test
   off-diagonals ≤0.5.
2. **PEC / PMC boundaries** — a per-step tangential-component masking pass (precompute a `(3,N,N,N)`
   keep-mask, multiply after the E (PEC) / H (PMC) update + source injection in
   [`loop.py`](src/fdtdx/mlx/loop.py)); un-gate the `pec_objects/pmc_objects` return (dispatch.py
   line ~70). Reference: [`objects/boundaries/pec.py`](src/fdtdx/objects/boundaries/pec.py),
   [`pmc.py`](src/fdtdx/objects/boundaries/pmc.py).
3. **Drude–Lorentz dispersion (ADE)** — host-precompute `c1/c2/c3`, carry `P_curr`/`P_prev` in
   `MLXState`, add `E += inv_eps·Σ(P_curr − P_new)` in the iso/diagonal branch of `update_E_mlx`,
   un-gate the `dispersive_c1` return (dispatch.py line ~115). Reference:
   [`src/fdtdx/dispersion.py`](src/fdtdx/dispersion.py) + the ADE block in `../fdtdx`'s
   `fdtd/update.py`.

**Build Phase 3 perf-aware:** since Phase 1/2 may change the loop/curl, write each new per-step term
**compile-friendly** (host-side gating, arrays as carried state) so it survives the `mx.compile`
pass — see widening-mlx-port-plan.md §5.

---

## Quick reference — commands
```bash
uv sync
uv run python -c "import fdtdx, mlx.core, jax"                                  # import sanity
uv run --with pytest pytest tests/validation -q                                 # parity (uniform + non-uniform)
uv run python benchmarks/bench_forward.py --backends mlx,jax --sizes 32,48,64,96,128,160,192 --steps 250 --repeats 2
uv run python benchmarks/plot_results.py benchmarks/results/<file>.jsonl
uv run python benchmarks/microbench_fusion.py --N 192 --iters 60               # fast step A/B
uvx ruff format src/fdtdx/mlx src/fdtdx/backend && uvx ruff check src/fdtdx/mlx src/fdtdx/backend
```
Force a backend: `with fdtdx.use_backend("mlx"|"jax"):` or `FDTDMEX_BACKEND=mlx|jax`. JAX is the
CPU oracle (`JAX_PLATFORMS=cpu` in conftest); MLX is Metal GPU. Confirm with the device check the
benchmark prints at startup.
