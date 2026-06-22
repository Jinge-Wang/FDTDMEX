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
- **NEXT: Phase 2 M2** — bring those kernels into the engine. See "Next step" below.

## Engine map (`src/fdtdx/mlx/`)

| file | role |
|---|---|
| [`loop.py`](src/fdtdx/mlx/loop.py) | time-loop driver; `_build_cores` compiles E-core/H-core; host-gated source injection between them. **M2 adds the kernel path + flag here.** |
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

## NEXT STEP — Phase 2 M2 (custom Metal kernels in the engine)

**Goal:** the isotropic/diagonal forward path runs on custom Metal E/H kernels (the M1 win) including
CPML, behind a flag, parity-clean — so real `run_fdtd` calls get the speedup, not just the microbench.

1. **Generalize the M1 kernels** ([`m1_kernel.py`](benchmarks/m1_kernel.py) has the working, bit-exact
   source). Replace the scalar `Cb` with a per-cell material read: `Cb = c · inv_eps[cell]`
   (isotropic 1-component first, then diagonal 3-component); uniform metric (=1) first, non-uniform
   later. Keep the leapfrog/ghost rules already in the source (backward diff + zero/wrap ghost for E,
   forward diff for H).
2. **CPML.** Interior cells use the plain-curl kernel; the PML boundary slabs need the κ-stretch + ψ
   correction. **Recommended approach: spatial hybrid** — the kernel computes the interior bulk; the
   existing slab-CPML MLX-op path (`curl.py:_cpml_curl` + slab ψ from `state.cpml_extents`, geometry
   from `pml.detect_pml_slabs`) handles the thin boundary slabs; merge the two (disjoint writes →
   race-free, out-of-place). Slab cells are a few % of the domain, so the bulk still gets the kernel
   speed. (Folding CPML into the kernel for slab cells is possible but more complex — defer.)
3. **Integrate in [`loop.py`](src/fdtdx/mlx/loop.py)** behind a flag (e.g. a `use_metal_kernel` arg on
   `run_forward_mlx`, default **off** until parity-clean), with the compiled MLX-op cores as fallback.
   Preserve leapfrog exactly: E-kernel → inject E (host) → H-kernel → inject H (host) → detectors.
4. **Validate.** Add an element-wise kernel parity test in `tests/validation/` vs forced-JAX
   (rel < 1e-3) for isotropic + diagonal, CPML on, plus a periodic case. Then `profile_engine.py`
   (RT/step) and `bench_forward.py` (scaling) for the realized gain.

**Done when:** engine isotropic/diagonal throughput jumps well above 277 Mcs/s toward the M1 floor
(discounted for CPML/sources), all validation green, flag default-on for those cases. **Then M3**
(heterogeneous materials via per-cell/block region specialization; non-uniform metric) —
`docs/phase2-metal-kernels.md` §5, §9.

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
uv run python benchmarks/m1_kernel.py --N 192 --iters 200                       # M1 kernel-vs-MLXops microbench
uv run python benchmarks/profile_engine.py --N 192 --steps 200                  # per-step RT (compile × CPML 2×2)
uv run python benchmarks/profile_metal.py  --N 192 --iters 100                  # roofline
uv run python benchmarks/bench_forward.py --backends mlx,jax --sizes 96,128,192,256 --steps 500 --repeats 2
uv run python benchmarks/plot_results.py benchmarks/results/<file>.jsonl
uvx ruff format src/fdtdx/mlx src/fdtdx/backend && uvx ruff check src/fdtdx/mlx src/fdtdx/backend
```
Force a backend: `with fdtdx.use_backend("mlx"|"jax")` or `FDTDMEX_BACKEND=mlx|jax`. Work on a branch;
local commits only (no push); one fix = one commit with before/after RT + throughput.
