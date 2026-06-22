# FDTDMEX — forward-engine performance plan

Single entry point. A fresh agent should be able to read this top-to-bottom and start the current
next step (**Phase 2 M2**) without prior context. Depth references:
- [`docs/performance.md`](docs/performance.md) — roofline, the round-trip (RT) model, current measured
  results (the "why" behind every perf decision).
- [`docs/phase2-metal-kernels.md`](docs/phase2-metal-kernels.md) — custom-kernel design, region
  specialization, memory/in-place, milestones, and the Apple-Silicon speedup table.

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
- **Phase 2 M2 — complete.** The iso/diagonal forward path runs on custom Metal E/H kernels in the
  engine ([`src/fdtdx/mlx/kernels.py`](src/fdtdx/mlx/kernels.py)), CPML via the spatial hybrid (bulk
  kernel + slab-CPML MLX-op correction), behind `FDTDMEX_METAL_KERNEL` (default off). Realized on M4
  Pro at N=192 (`profile_engine.py --kernel`): iso **CPML-off 2219 Mcs/s / 5 RT (4.5× over compiled
  ops, at the M1 floor)**; iso **CPML-on 374 Mcs/s / 27 RT (1.31× over compiled ops, 1.35× over the
  Phase-1 277)**; diagonal 2036 / 346 Mcs/s. CPML-on is discounted by the slab correction
  (`_slab_add` rebuilds full arrays via concatenate, ~22 RT on top of the 5 RT bulk). Kernel cores
  are `mx.compile`d (the metal kernel composes as a graph node) — **uncompiled the slab ops dominate
  and the CPML-on step is slower than ops**. Validated element-wise vs MLX-op cores (rel < 1e-4) and
  vs the JAX oracle (rel < 1e-3) for iso/diagonal/periodic + CPML ([`tests/validation/test_mlx_kernel.py`](tests/validation/test_mlx_kernel.py)); full suite green with the flag forced on.
- **NEXT: Phase 2 M3** — heterogeneous materials (per-cell/block region specialization) + non-uniform
  metric in the kernel; and fold CPML into the kernel for slab cells to close the CPML-on gap. See
  "Next step" below.

## Engine map (`src/fdtdx/mlx/`)

| file | role |
|---|---|
| [`loop.py`](src/fdtdx/mlx/loop.py) | time-loop driver; `_build_cores` compiles E-core/H-core (kernel cores when `use_metal_kernel` + eligible, else MLX-op cores); host-gated source injection between them |
| [`kernels.py`](src/fdtdx/mlx/kernels.py) | **M2**: custom Metal E/H bulk kernels (per-cell `cb=c·inv_eps`, periodic ghost) + slab-CPML hybrid (`_slab_correction`) + `kernel_eligible`; `build_kernel_cores` returns compiled cores |
| [`curl.py`](src/fdtdx/mlx/curl.py) | pad-free slice-diff Yee curl + slab-CPML decomposition (`_cpml_curl`, `_slab_take/_slab_add`, `_AX`) |
| [`update.py`](src/fdtdx/mlx/update.py) | E/H update — iso/diagonal fast path + full-tensor A/B path; pure `_update_E/_update_H` |
| [`pml.py`](src/fdtdx/mlx/pml.py) | host CPML coeff precompute + `detect_pml_slabs` (slab geometry M2 reuses) |
| [`bridge.py`](src/fdtdx/mlx/bridge.py) | ArrayContainer ↔ MLXState; slices ψ to slabs in, reconstructs full ψ out |
| [`state.py`](src/fdtdx/mlx/state.py) | `MLXState` (slab ψ tuples + `cpml_extents`) |
| `inject.py` / `accumulate.py` (+ `*_freeze.py`) | host-gated source injection / detector recording |
| [`backend/dispatch.py`](src/fdtdx/backend/dispatch.py) | routing + milestone gating; [`backend/context.py`](src/fdtdx/backend/context.py) `use_backend` |

Working custom kernels (validated, M2's starting point): [`benchmarks/m1_kernel.py`](benchmarks/m1_kernel.py).

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

## NEXT STEP — Phase 2 M3 (heterogeneous materials + close the CPML-on gap)

M2 is done (custom Metal kernels in the engine; see Status). M3 widens the kernel path and recovers
the CPML-on throughput the slab MLX-op correction currently leaves on the table.

1. **Region specialization** for heterogeneous domains (isotropic bulk + local anisotropic/smoothed
   inclusions — the target use case): a per-cell material-class branch in the kernel (cheap diagonal
   update where iso, the 3×3 A/B + neighbour-average only where not). On unified memory + out-of-place
   this is pure compute placement — no halo arrays, no neighbour bookkeeping. `docs/phase2-metal-kernels.md` §5.
2. **Non-uniform metric in the kernel** — per-axis scale arrays (the `metric_fwd`/`metric_bwd` the
   eligibility check currently gates off). The slab path already handles array metrics (`_slab_diff`).
3. **Fold CPML into the kernel for slab cells** to close the CPML-on gap: the spatial hybrid's
   `_slab_add` rebuilds full component arrays via `concatenate` (~22 RT on top of the 5 RT bulk at
   N=192). A second small kernel that scatter-adds the κ-stretch + ψ correction over only the slab
   cells (ψ as state) would drop CPML-on toward the bulk floor. (Deferred from M2 by design.)
4. **z-march tiling** only if `profile_engine.py` shows RT climbing above the floor at large N
   (N ≥ 384) — the cache already captures the neighbour reuse at N ≤ 256, so this is measurement-gated
   (`docs/phase2-metal-kernels.md` §3, §11). The thread-per-cell kernel is a self-contained swap.

**Then** flip `FDTDMEX_METAL_KERNEL` default-on for the eligible cases once M3 lands and the surface
is broad enough that the kernel rarely falls back. `docs/phase2-metal-kernels.md` §5, §9.

## Phase 3 — broaden supported surface (independent of perf)

Spec: [`docs/widening-mlx-port-plan.md`](docs/widening-mlx-port-plan.md). Order (ascending effort):
lossy full-anisotropic + 9-tensor conductivity; PEC/PMC boundaries; Drude–Lorentz (ADE) dispersion.
Build each compile/kernel-friendly (host-side gating, arrays carried as state).

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
uv run --with pytest pytest tests/validation -q                                 # parity (uniform + non-uniform)
FDTDMEX_METAL_KERNEL=1 uv run --with pytest pytest tests/validation -q          # parity with the M2 kernel path on
uv run python benchmarks/m1_kernel.py --N 192 --iters 200                       # M1 kernel-vs-MLXops microbench
uv run python benchmarks/profile_engine.py --N 192 --steps 200 --kernel        # per-step RT (ops × CPML + metal-kernel)
uv run python benchmarks/profile_metal.py  --N 192 --iters 100                  # roofline
uv run python benchmarks/bench_forward.py --backends mlx,jax --sizes 96,128,192,256 --steps 500 --repeats 2
uv run python benchmarks/plot_results.py benchmarks/results/<file>.jsonl
uvx ruff format src/fdtdx/mlx src/fdtdx/backend && uvx ruff check src/fdtdx/mlx src/fdtdx/backend
```
Force a backend: `with fdtdx.use_backend("mlx"|"jax")` or `FDTDMEX_BACKEND=mlx|jax`. Work on a branch;
local commits only (no push); one fix = one commit with before/after RT + throughput.
