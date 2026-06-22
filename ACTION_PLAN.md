# FDTDMEX — forward-engine performance plan

Single entry point. A fresh agent should be able to read this top-to-bottom and start the current
next step (**Phase 3 — broaden supported surface**, see "NEXT STEP" below; Phase 2 M1–M3 are
complete and the Metal kernel path is default-on) without prior context. Depth references:
- [`docs/performance.md`](docs/performance.md) — roofline, the round-trip (RT) model, current measured
  results + a **History** section (what was tried, the gains, what didn't work — the "why").
- [`docs/phase2-metal-kernels.md`](docs/phase2-metal-kernels.md) — custom-kernel design, region
  specialization, memory/in-place, milestones (§9), and the Apple-Silicon speedup table.

> **Milestone naming.** "Phase 2 M1/M2/M3" in *this file* are the **Metal-kernel** performance
> milestones (M1 microbench → M2 kernels-in-engine → M3 heterogeneous/CPML-fold). They are unrelated
> to the **WS-A porting** milestones M1–M4 in [`docs/roadmap.md`](docs/roadmap.md) (which track the
> physics surface: sources, anisotropy, non-uniform grids — all complete).

## What this project is

FDTDMEX is a fork of [fdtdx](https://github.com/ymahlau/fdtdx) (JAX FDTD Maxwell solver) that adds a
native **MLX/Metal forward backend** for Apple Silicon. On a Mac a supported forward `run_fdtd`
auto-routes to the MLX time loop; gradients / unsupported features / non-Apple platforms run the
unchanged JAX engine (which is also the **parity oracle**). Goal: fast, large *forward* simulations on
a single Mac. Import stays `import fdtdx`. The engine is **functional / out-of-place** (race-free),
mirrors fdtdx element-wise (the parity bar), and is fp32.

## Status

- **Phase 1 — complete.** Default MLX path = pad-free slice-diff curl + `mx.compile`d E/H cores +
  slab-CPML. **277 Mcs/s / 36 RT** at N=192 iso on M4 Pro (2.6× the original engine); MLX leads JAX-CPU
  for all N ≥ 64 across isotropic/diagonal/full-aniso (1.25–1.4×). All validation green. (`docs/performance.md`.)
- **Phase 2 M1 — complete, GO.** Standalone custom Metal kernels for the isotropic-uniform interior
  ([`benchmarks/m1_kernel.py`](benchmarks/m1_kernel.py)) reach **~3 RT / 3150 Mcs/s — 5.8× over
  compiled MLX-ops, bit-exact**. This proves a hand kernel hits the bandwidth floor the MLX op-graph
  cannot (the op path is stuck at ~18–36 RT). JAX-CPU's effective traffic is ~36 RT, so the floor
  kernel is far above it.
- **Phase 2 M2 — complete (superseded by M3).** Landed custom Metal E/H bulk kernels in the engine
  ([`src/fdtdx/mlx/kernels.py`](src/fdtdx/mlx/kernels.py)) for the iso/diagonal path, with CPML as a
  spatial hybrid (bulk kernel + an MLX-op slab correction). Reached the bulk floor with CPML off
  (2219 Mcs/s / 5 RT) but the slab correction's full-array rebuild left CPML-on at 374 Mcs/s / 27 RT.
  Details and the measured history in `docs/performance.md`.
- **Phase 2 M3 — complete; kernel path default-on.** Three independent kernel extensions, each
  parity-gated ([`tests/validation/test_mlx_kernel.py`](tests/validation/test_mlx_kernel.py):
  kernel-vs-ops rel < 1e-4, vs-JAX rel < 1e-3) and benchmarked on M4 Pro:
  1. **CPML folded into the kernel** — each PML-slab thread advances ψ and adds the κ-stretch/ψ
     correction in-kernel (compact slab ψ + per-axis `a/b/1κ` as extra in/out buffers), so the kernel
     writes the final E/H. Removes the M2 slab-correction full-array rebuild. **CPML-on 374 → 1826
     Mcs/s, 27 → 5 RT (4.9×), at the bulk floor** (N=192 iso; 1711 diag).
  2. **Non-uniform metric in-kernel** — each difference scaled by its per-axis
     `reference_spacing/cell_width` buffer; non-uniform iso/diagonal now ride the kernel at the same
     ~5 RT floor.
  3. **Heterogeneous full-tensor via a block hybrid** — kernel runs the diagonal bulk; the
     off-diagonal inclusion's bounding box gets the MLX-op aniso update over a haloed interior slice,
     spliced back. **N=128, 8³ inclusion: 125 → 1124 Mcs/s (9.0×)**.

  `FDTDMEX_METAL_KERNEL` is now default-on (`=0` forces the MLX-op cores); ineligible cases fall back
  automatically via `kernel_eligible`. Full validation suite green default-on (20 passed).
- **NEXT: Phase 3** — broaden the supported physics surface (independent of perf). See "Next step".

## Engine map (`src/fdtdx/mlx/`)

| file | role |
|---|---|
| [`loop.py`](src/fdtdx/mlx/loop.py) | time-loop driver; `_build_cores` compiles E-core/H-core (kernel cores when `use_metal_kernel` + eligible, else MLX-op cores); host-gated source injection between them |
| [`kernels.py`](src/fdtdx/mlx/kernels.py) | **M2+M3**: custom Metal E/H bulk kernels (per-cell `cb=c·diag(inv_eps)`, periodic ghost, **in-kernel CPML fold** via slab ψ + a/b/1κ buffers, **in-kernel non-uniform metric** `m{k}`) + **block hybrid** for full-tensor inclusions (`_offdiag_box`/`_set_box` + MLX-op aniso over the bbox) + `kernel_eligible`; `build_kernel_cores` returns compiled cores |
| [`curl.py`](src/fdtdx/mlx/curl.py) | pad-free slice-diff Yee curl + slab-CPML decomposition (`_cpml_curl`, `_slab_take/_slab_add`, `_AX`) |
| [`update.py`](src/fdtdx/mlx/update.py) | E/H update — iso/diagonal fast path + full-tensor A/B path; pure `_update_E/_update_H` |
| [`pml.py`](src/fdtdx/mlx/pml.py) | host CPML coeff precompute + `detect_pml_slabs` (slab geometry M2 reuses) |
| [`bridge.py`](src/fdtdx/mlx/bridge.py) | ArrayContainer ↔ MLXState; slices ψ to slabs in, reconstructs full ψ out |
| [`state.py`](src/fdtdx/mlx/state.py) | `MLXState` (slab ψ tuples + `cpml_extents`) |
| `inject.py` / `accumulate.py` (+ `*_freeze.py`) | host-gated source injection / detector recording |
| [`backend/dispatch.py`](src/fdtdx/backend/dispatch.py) | routing + milestone gating; [`backend/context.py`](src/fdtdx/backend/context.py) `use_backend` |

M1 microbench (the bit-exact standalone kernel M2 generalised, kept as the roofline reference):
[`benchmarks/m1_kernel.py`](benchmarks/m1_kernel.py).

## Decisions (measured)

- FDTD is **memory-bandwidth-bound**; the lever is reducing per-step DRAM round-trips (RT), not arithmetic.
- M4 Pro roofline is the **measured 240 GB/s** coalesced (88% of the 273 spec), not 273.
- Layout stays **`(3, N, N, N)`** (z contiguous → coalesced).
- **fp32 is the floor** (no mixed precision).
- A hand-written Metal kernel reaches the ~3–8 RT floor; MLX op-fusion does not (M1).

## Rejected (do not re-explore)

- **Component-last `(N,N,N,3)` layout** — 1.00×, no coalescing penalty.
- **A memory advantage over JAX-CPU** — footprints ~equal; the capacity advantage is vs a discrete GPU's VRAM.
- **CPML-grows-with-N as a cause** — CPML is a constant ~25% of traffic.
- **Mixed precision** — fp32 minimum.
- **Manual in-place for speed** — in-place saves footprint, not bandwidth (capacity lever only; phase2 doc §7).

## Phase 2 M3 — complete (see Status for results)

All three perf/coverage sub-tasks landed in [`src/fdtdx/mlx/kernels.py`](src/fdtdx/mlx/kernels.py),
each gated by [`tests/validation/test_mlx_kernel.py`](tests/validation/test_mlx_kernel.py)
(kernel-vs-ops rel < 1e-4, vs-JAX rel < 1e-3) and benchmarked:
- **CPML fold** — `_corr_blocks` emits the in-kernel ψ recurrence + κ-stretch correction over slab
  threads; `build_kernel_cores` passes the compact slab ψ and per-axis `a/b/1κ` as extra in/out
  buffers. Replaced M2's `_slab_correction` full-array rebuild. CPML-on 375 → 1826 Mcs/s (4.9×).
- **Non-uniform metric** — `_metric_lines`/`_metric_side` scale each difference by its `m{k}` buffer.
- **Block hybrid** — `_offdiag_box` finds the inclusion bbox; `_box_correct` runs the MLX-op
  `_update_E`/`_update_H` aniso over a haloed interior slice; `_set_box` splices it back.
- **z-march tiling** — still deferred (measurement-gated, N ≥ 384; `docs/phase2-metal-kernels.md` §3,
  §11): the thread-per-cell kernel hits the floor at N ≤ 256, so it is not needed yet.

`FDTDMEX_METAL_KERNEL` is now **default-on** ([`backend/dispatch.py`](src/fdtdx/backend/dispatch.py)
`_metal_kernel_enabled`); `=0` forces the MLX-op path.

### Deferred / falls back to the MLX-op cores (and why)

- **Per-cell in-kernel 3×3 anisotropic branch** — the block hybrid was chosen instead: it reuses the
  already-validated MLX-op aniso update and is bit-identical on the inclusion cells, so it carries far
  less parity risk. The per-cell branch (one thread doing the full 3×3 + weighted curl-averaging) is
  the general form for anisotropic cells that are scattered or ring every interface (subpixel
  smoothing); it is left for when such a distribution is the target.
- **Anisotropic inclusions that are lossy, non-uniform-grid, oversized (> ½ the domain), or overlap a
  PML slab** — the block hybrid assumes a compact, lossless, interior inclusion so its haloed slice
  needs no CPML or metric. Outside that envelope the whole run falls back to the MLX-op aniso cores
  (correct, just not accelerated). `kernel_eligible` is the gate.
- **Conductivity (lossy media) in the kernel** — the bulk kernel is lossless; any conductivity sends
  the run to the MLX-op cores. Folding it in is Phase 3 (`docs/widening-mlx-port-plan.md`).
- **z-march tiling** — measurement-gated. The thread-per-cell kernel already sits at the bandwidth
  floor for N ≤ 256; build it only if `profile_engine.py` shows RT climbing at N ≥ 384
  (`docs/phase2-metal-kernels.md` §3, §11).

### ⚠ Known performance gap — dense/whole-domain full-anisotropy (flagged for a later phase)

The M3 scaling sweep (`docs/performance.md`) shows **uniform full-tensor anisotropy at only ~1.3× over
JAX-CPU**, versus ~6.5–7× for isotropic/diagonal. This is expected, not a regression: the block hybrid
accelerates the *iso/diagonal bulk* around a compact off-diagonal inclusion, so the realized speedup
scales with **how compact the anisotropic region is and how small it is relative to the domain**. A
domain *filled* with off-diagonal tensor cells has no bulk to accelerate — the whole run is the MLX-op
aniso cores (the unchanged pre-M3 path, which already leads JAX-CPU ~1.3×). So a uniform-aniso sweep
seeing little gain is the correct behavior of the *compact-inclusion* design, not a problem with it.

Closing this gap — accelerating *dense* or whole-domain anisotropy — needs the full-tensor update to
run on the GPU directly: the per-cell in-kernel 3×3 + weighted curl-averaging branch (above), or a
dedicated full-tensor Metal kernel. Worth investigating in a later phase **if dense-anisotropic
domains become a target use case** (the stated target is local inclusions, where the block hybrid is
already at the floor). Until then this is a documented, intentional limit, not a TODO blocking Phase 3.

## NEXT STEP — Phase 3: broaden supported surface (independent of perf)

Spec: [`docs/widening-mlx-port-plan.md`](docs/widening-mlx-port-plan.md). Order (ascending effort):
lossy full-anisotropic + 9-tensor conductivity; PEC/PMC boundaries; Drude–Lorentz (ADE) dispersion.
Build each compile/kernel-friendly (host-side gating, arrays carried as state). As each lands, widen
`kernel_eligible` so the new surface rides the Metal kernels where it can (the dual-conductivity /
full-tensor cases currently fall back to the MLX-op cores).

## Physics-correctness contract (every change)

- **Out-of-place / race-free.** Isotropic update reads only its own field cell + the *other* field's
  neighbours; anisotropic reads neighbour `E_old` (off-diagonal averaging) → double-buffer + halo.
- **Leapfrog order:** `update_E (reads Hⁿ⁻½) → inject E → update_H (reads injected Eⁿ⁺½) → inject H →
  detectors`. Never merge E and H into one pass.
- **Source/detector gating stays host-side** (compiled/kernel core is pure all-cell math).
- **Element-wise parity** vs the forced-JAX oracle, rel < 1e-3. Marginal failure → raise resolution,
  never loosen tolerance.

## Validation & measurement

```bash
uv run --with pytest pytest tests/validation -q                                 # parity (kernel default-on)
FDTDMEX_METAL_KERNEL=0 uv run --with pytest pytest tests/validation -q          # parity with the MLX-op cores (kernel off)
uv run python benchmarks/m1_kernel.py --N 192 --iters 200                       # M1 kernel-vs-MLXops microbench
uv run python benchmarks/profile_engine.py --N 192 --steps 200 --kernel        # per-step RT (ops × CPML + metal-kernel)
uv run python benchmarks/profile_metal.py  --N 192 --iters 100                  # roofline
uv run python benchmarks/bench_forward.py --backends mlx,jax --sizes 96,128,192,256 --steps 500 --repeats 2
uv run python benchmarks/plot_results.py benchmarks/results/<file>.jsonl
uvx ruff format src/fdtdx/mlx src/fdtdx/backend && uvx ruff check src/fdtdx/mlx src/fdtdx/backend
```
Force a backend: `with fdtdx.use_backend("mlx"|"jax")` or `FDTDMEX_BACKEND=mlx|jax`. Work on a branch;
local commits only (no push); one fix = one commit with before/after RT + throughput.
