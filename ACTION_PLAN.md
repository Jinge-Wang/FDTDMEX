# FDTDMEX — forward-engine performance plan

Single entry point for the MLX/Metal forward-perf work. Reference docs:
[`docs/performance.md`](docs/performance.md) (roofline, RT model, results) and
[`docs/phase2-metal-kernels.md`](docs/phase2-metal-kernels.md) (Phase 2 kernel spec).

## Status

**Phase 1 complete.** Default MLX path (compiled E/H cores + pad-free slice-diff + slab-CPML),
M4 Pro, N=192 isotropic: **277 Mcs/s, 36 RT/step — 2.6× the original engine.** MLX/Metal leads
JAX-CPU for all N ≥ 64 across isotropic / diagonal / full-anisotropic (1.25–1.4×; see
`docs/performance.md`). All validation green (physics exact).

**Next: Phase 2 M1** — isotropic-uniform custom Metal kernel (go/no-go on reaching the ~5–8 RT floor).

## Engine map (`src/fdtdx/mlx/`)

| file | role |
|---|---|
| [`loop.py`](src/fdtdx/mlx/loop.py) | time-loop driver; builds compiled E-core/H-core, host-gated source injection between them |
| [`curl.py`](src/fdtdx/mlx/curl.py) | pad-free slice-diff Yee curl + slab-CPML decomposition (`_cpml_curl`, `_slab_take/_slab_add`) |
| [`update.py`](src/fdtdx/mlx/update.py) | E/H update — iso/diagonal fast path + full-tensor A/B path; pure `_update_E/_update_H` |
| [`pml.py`](src/fdtdx/mlx/pml.py) | host CPML coeff precompute + `detect_pml_slabs` |
| [`bridge.py`](src/fdtdx/mlx/bridge.py) | ArrayContainer ↔ MLXState; slices ψ to slabs in, reconstructs full ψ out |
| [`state.py`](src/fdtdx/mlx/state.py) | `MLXState` (carries slab ψ tuples + `cpml_extents`) |
| `inject.py`/`accumulate.py` (+ `*_freeze.py`) | host-gated source injection / detector recording |
| [`backend/dispatch.py`](src/fdtdx/backend/dispatch.py) | routing + milestone gating; [`backend/context.py`](src/fdtdx/backend/context.py) `use_backend` |

## Decisions

- FDTD is **memory-bandwidth-bound**; the lever is reducing per-step DRAM round-trips (RT), not arithmetic.
- M4 Pro roofline is the **measured 240 GB/s** coalesced (88% of the 273 spec), not 273.
- Layout stays **`(3, N, N, N)`** (z contiguous).
- **fp32 is the floor** (no mixed precision).
- Backend: auto MLX on Apple Silicon for supported forward runs; **JAX (CPU) is the parity oracle + fallback**.

## Rejected (measured / decided)

- **Component-last `(N,N,N,3)` layout** — 1.00×, no coalescing penalty.
- **A memory advantage over JAX-CPU** — footprints are ~equal; the capacity advantage is vs a discrete GPU's VRAM, not vs JAX-CPU.
- **CPML-grows-with-N as the plateau cause** — CPML is a constant ~25% of traffic.
- **Mixed precision** — fp32 minimum.
- **Manual in-place to gain speed** — in-place saves footprint, not bandwidth (capacity lever only; see `phase2-metal-kernels.md` §6).

## Phase 1 — eager-path traffic reduction (DONE)

Pad-free slice-diff curl + `mx.compile`d E/H cores + slab-CPML (ψ + κ-stretch confined to the PML
boundary slabs via an exact algebraic split). 99 → 36 RT, 105 → 277 Mcs/s, physics exact. RT ladder
in `docs/performance.md`.

## Phase 2 — custom Metal update kernels (NEXT)

Separate E-update and H-update kernels via `mx.fast.metal_kernel`, fp32. Full spec, region
specialization for heterogeneous materials, memory/in-place, the Apple-Silicon speedup table, and
milestones in **[`docs/phase2-metal-kernels.md`](docs/phase2-metal-kernels.md)**. **M1 is the
go/no-go**: an isotropic-uniform interior kernel — does a hand kernel approach the ~5–8 RT floor
(≫ the compiled 21/36 RT), and what is JAX's effective traffic? Proceed only if yes.

## Phase 3 — broaden supported surface (independent of perf)

Spec: [`docs/widening-mlx-port-plan.md`](docs/widening-mlx-port-plan.md). Order (ascending effort):
lossy full-anisotropic + 9-tensor conductivity; PEC/PMC boundaries; Drude–Lorentz (ADE) dispersion.
Build each compile-friendly (host-side gating, arrays carried as state).

## Physics-correctness contract (every change)

- **Out-of-place / race-free.** Isotropic update reads only its own field cell + the other field's
  neighbours; anisotropic reads neighbour `E_old` (off-diagonal averaging) → double-buffer + halo.
- **Leapfrog order:** `update_E (reads Hⁿ⁻½) → inject E → update_H (reads injected Eⁿ⁺½) → inject H →
  detectors`. Never merge E and H into one pass.
- **Source/detector gating stays host-side** (compiled/kernel core is pure all-cell math).
- **Element-wise parity** vs the forced-JAX oracle, rel < 1e-3. Marginal failure → raise resolution,
  never loosen tolerance.

## Validation & measurement

```bash
uv run --with pytest pytest tests/validation -q                                 # parity (uniform + non-uniform)
uv run python benchmarks/profile_engine.py --N 192 --steps 200                  # per-step RT (compile × CPML 2×2)
uv run python benchmarks/profile_metal.py  --N 192 --iters 100                  # roofline
uv run python benchmarks/bench_forward.py --backends mlx,jax --sizes 96,128,192,256 --steps 500 --repeats 2
uv run python benchmarks/plot_results.py benchmarks/results/<file>.jsonl
uvx ruff format src/fdtdx/mlx src/fdtdx/backend && uvx ruff check src/fdtdx/mlx src/fdtdx/backend
```
Force a backend: `with fdtdx.use_backend("mlx"|"jax")` or `FDTDMEX_BACKEND=mlx|jax`. Work on a branch;
one fix = one local commit with before/after RT + throughput.
