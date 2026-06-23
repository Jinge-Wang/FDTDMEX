# Performance — roofline, model, and current results

Measured reference for the MLX/Metal forward engine on Apple Silicon. fp32 throughout.

## Current scaling (M4 Pro)

![Forward scaling — MLX/Metal vs JAX-CPU](../benchmarks/figures/forward_scaling.png)

Full `run_fdtd` wall-clock (warmup excluded), 500 steps, **kernel path default-on**. Regenerate: `bench_forward.py --backends mlx,jax --materials isotropic,diagonal,full_aniso --sizes 64,96,128,192,256 --steps 500 --repeats 2 --isolate` then `plot_results.py <file>.jsonl --out benchmarks/figures/forward_scaling.png`. **MLX/Metal leads JAX-CPU for every N ≥ 64 across all three materials, with no plateau:**

| material | MLX Mcs/s (N=192 / 256) | JAX-CPU | speedup | path |
|---|--:|--:|--:|---|
| isotropic | 1392 / 1300 | 197 / 201 | **~6.5–7x** | Metal kernel (CPML folded) |
| diagonal | 1359 / 1293 | 196 / 200 | **~6.5–7x** | Metal kernel (CPML folded) |
| full_aniso | 123 / 116 | 96 / 98 | ~1.3x | MLX-op cores (uniform 9-tensor → kernel fallback) |

The iso/diagonal full-`run_fdtd` throughput (~1.3–1.4k Mcs/s) sits below the ~1826 Mcs/s pure-loop profile because it includes source injection, the host↔device bridge, and per-run setup; the kernel update itself is at the bandwidth floor (RT table below). The uniform full-tensor sweep fills the whole domain with a 9-tensor, which the block hybrid does not accelerate (it targets *compact* inclusions) — so it runs on the MLX-op aniso cores, keeping the pre-existing ~1.3x edge. Panel (d) memory: MLX peak is exact; the JAX line is in-process RSS (use `benchmarks/profile_memory.py` for a clean per-cell figure).

## Roofline (M4 Pro, `benchmarks/profile_metal.py`)

- **Coalesced copy: 240 GB/s = 88% of the 273 GB/s spec** — the real ceiling (the spec is not achievable; 240 is the denominator for all roofline math).
- Component-leading `(3,N,N,N)` vs component-last `(N,N,N,3)` stencil: **1.00x** — no coalescing penalty from the layout.
- `roll`-diff vs slice-diff on the engine's `y − shift(y)` pattern: **0.89–1.13x** — `roll` is not a culprit.

## The round-trip (RT) model

FDTD is memory-bandwidth-bound. **1 RT = read+write of one `(3,N³)` field**; per-step time ≈ `RT x 170 MB / 240 GB/s` at N=192. The bottleneck is redundant traffic — too many full-array passes, not arithmetic, dispatch-starvation, or layout (confirmed by toggling CPML, which removes exactly the carried-ψ RT, and by `profile_engine.py`'s eager-vs-compiled x CPML 2x2).

| engine state (N=192 iso) | Mcs/s | RT/step |
|---|--:|--:|
| original (pad+roll, full-domain CPML) | 105 | ~99 |
| + pad-free slice-diff (eager) | 130 | 77 |
| + `mx.compile` E/H cores | 211 | 47 |
| + slab-CPML (MLX-op cores, CPML-on) | **277** | **36** |
| compiled MLX-op cores, CPML off (op-graph ceiling) | 473 | 21 |
| + Metal kernel, CPML on (spatial-hybrid slab correction) | 374 | 27 |
| **+ Metal kernel, CPML folded in, CPML on (default-on)** | **1826** | **5** |
| **+ Metal kernel, CPML off** | **2219** | **5** |
| necessary floor (read E,H + materials; write E,H) | ~3150¹ | ~3 |

¹ the standalone microbench ([`benchmarks/m1_kernel.py`](../benchmarks/m1_kernel.py)) at the bare read-once/write-once floor; the in-engine CPML-off number (2219) is that floor discounted for source injection + detector recording.

The MLX-op path is stuck near its op-graph ceiling at ~21–36 RT (op fusion can't keep a stencil's working set on-chip or merge neighbour reads). The custom Metal kernels ([`src/fdtdx/mlx/kernels.py`](../src/fdtdx/mlx/kernels.py), behind `FDTDMEX_METAL_KERNEL`) reach the ~5 RT bandwidth floor with CPML folded in — see *How the engine reached the bandwidth floor* below for the step-by-step, and *Kernel design* for the structure.

## Metal vs CPU/JAX — two factors

CPU and GPU share one DRAM, so speedup is not "GPU flops":
- **(a) bandwidth-utilization gap (~1.4x, chip-dependent).** GPU sustains ~85% of rated unified BW; a multicore CPU sustains ~55–65% and caps at a per-die ceiling (~240 GB/s). This is the measured 1.37x on M4 Pro at equal traffic; it widens only where rated BW outruns the CPU (top-bin Max, Ultra).
- **(b) traffic gap (chip-independent, the real prize).** JAX/XLA on CPU does not tile the stencil (effective traffic ~tens of RT, like the pre-optimization engine). A fused Metal kernel at the ~5–8 RT floor adds up to ~4x on top of (a) — *if* JAX stays traffic-heavy (measured directly).

## How the engine reached the bandwidth floor

The forward step is memory-bound, so every optimization targets redundant DRAM traffic (round trips, RT). The path from the first eager implementation to the current Metal kernels — what each change bought, and what was tried and didn't help (all N=192 isotropic on one M4 Pro):

- **Eager op-graph tuning (105 → 277 Mcs/s).** Dropping the per-step `mx.pad` for a pad-free slice-diff curl, wrapping the E/H cores in `mx.compile` to fuse the elementwise chain, and confining CPML to boundary slabs each cut measurable RT. A component-last `(N,N,N,3)` memory layout and `roll`-vs-slice differencing were tried and made no difference. The op graph then plateaus near 21–36 RT — op fusion cannot keep a stencil's working set on-chip or merge neighbour reads.
- **A standalone Metal kernel (≈3150 Mcs/s, 3 RT).** A hand-written kernel for the isotropic-uniform interior reaches the read-once/write-once floor the op graph cannot, confirming a custom kernel was worth building.
- **Kernels in the engine (CPML-off 2219 Mcs/s; CPML-on initially 374).** Generalised to per-cell isotropic/diagonal materials and non-cubic domains. CPML was first added as a *spatial hybrid* — the kernel did the bulk while the thin PML slabs got a separate correction — but that correction rebuilt full field arrays and cost ~22 RT on top of the 5 RT bulk, so CPML-on stayed slow. (Running the kernel cores eagerly was also tried and came out *slower* than the op path; compiling the whole core fixed it.)
- **CPML folded into the kernel (CPML-on 374 → 1826 Mcs/s, 5 RT).** Moving the per-slab ψ recurrence and κ-stretch correction inside the bulk kernel removed the array rebuild and put the common CPML-on path at the bulk floor (1826 isotropic, 1711 diagonal). The same kernel also gained the non-uniform metric (graded grids ride the same floor) and a **block hybrid** for compact full-tensor inclusions — the kernel runs the diagonal bulk while the inclusion's bounding box gets the MLX-op anisotropic update (N=128 8³ inclusion 125 → 1124 Mcs/s). The block hybrid was chosen over a per-cell in-kernel 3x3 branch because it reuses the already-validated anisotropic ops and stays bit-identical on box cells. The Metal kernels are now default-on, with the full parity suite green.

## Kernel design

The forward update is two `mx.fast.metal_kernel`s — one for E, one for H — because leapfrog needs E and H in separate passes (H reads the just-updated E). Host-gated source injection runs between them, unchanged from the MLX-op loop.

- **Thread-per-cell.** Each thread computes one cell's six metric-scaled curl differences and the `cb = c·inv_eps` (or `inv_mu`) multiply, reading neighbours from global memory with cache reuse. Layout stays `(3, N, N, N)` (z contiguous → coalesced loads). The Yee curl is a one-sided difference, so a thread needs only a 1-cell halo on one side per axis; domain edges reproduce the pad-free ghost rule (zero on PML/PEC faces, wrapped on periodic faces).
- **Out-of-place = race-free, for free.** Every kernel reads the frozen old field and writes a separate new one, so a neighbour reading an already-overwritten cell cannot happen regardless of execution order. On unified memory there is one address space, so a boundary cell reads its neighbour's old value by global index — no halo copy, no inter-device exchange. This is what makes heterogeneous material handling a placement choice rather than a distributed-stencil problem.
- **CPML folded in.** A thread inside a PML boundary slab also advances that slab cell's ψ recurrence and adds the κ-stretch + ψ correction into the curl before the `cb` multiply, so the kernel writes the final field — no post-kernel array rebuild. ψ and the per-axis `a/b/1κ` coefficients are the compact boundary-slab buffers already in `MLXState`.
- **Non-uniform metric in-kernel.** Each difference is scaled by its per-axis `reference_spacing/ cell_width` buffer; uniform axes carry a scalar `1.0` and emit no multiply, so the uniform path is byte-for-byte unchanged.
- **Block hybrid for full-tensor inclusions.** The kernel runs the diagonal bulk while a compact off-diagonal inclusion's bounding box gets the validated MLX-op anisotropic update over a haloed slice, spliced back; box cells are bit-identical to the whole-domain ops path. Eligible only for lossless, uniform-grid, compact, PML-disjoint inclusions; otherwise the whole domain falls back to the MLX-op anisotropic cores (via `kernel_eligible`).

## Monitor recording (detector overhead)

The bulk kernel runs at the floor, but on a *monitored* run the full `run_fdtd` used to sit ~7× below the pure loop — not because of the update, but because detectors were fed by **interpolating the whole `(3,N³)` field every step** (`pad_fields_mlx` + `interpolate_fields_mlx` + the `(H_prev+H)/2` time-average), then slicing each monitor's handful of cells out of the result. On the O-band MRM reference workload (4.9 M cells, ~73 k steps, [`examples/ring_mrm_oband/field_maps_100nm.py`](../examples/ring_mrm_oband/field_maps_100nm.py)) that overhead dominated the wall. Three compounding fixes removed it, all without touching the update kernel:

- **Region-restricted interpolation.** The Yee co-location stencil reaches only ±1 cell, so each detector is interpolated over just its `grid_slice` + a 1-cell halo. `interpolate_region_mlx` ([`mlx/interpolate.py`](../src/fdtdx/mlx/interpolate.py)) builds the windowed padded sub-block straight from the raw field (interior crop + the single ghost row only at a true domain edge, matching the zero/wrap pad rule) and feeds the *unchanged* `interpolate_fields_mlx`, so the region result is **element-wise identical** to slicing the old full-domain pass. A full in-plane `|E|²` slice shrinks from the whole z-extent to ~3 layers; small port boxes become ~free.
- **Activity-gating.** The whole record block is skipped on steps where no detector actually records (it used to interpolate every step whenever *any* detector existed).
- **DFT auto-subsampling.** A phasor's signal is band-limited at the free-space `c/λ`, and the FDTD `dt` is ~10–20× below that Nyquist, so phasors record only every `floor(1/(k·f_max·dt))`-th step (oversampling margin `k=12`) with the per-sample weight scaled by the stride. Default-on; `FDTDMEX_DFT_STRIDE=1` forces exact every-step recording (used by the element-wise parity tests).

Region-interpolation + gating are exact (parity-tested against the JAX oracle with `FDTDMEX_DFT_STRIDE=1`); subsampling is exact within the oversampling margin (physics-tested). **Result on the reference workload: 1478 s → 377 s (3.9×)** with no observable physics change (resonance dip 1307.1 nm both, extinction depth 0.302 both, max normalized-transmission Δ across the 18-λ spectrum 0.0000, on-resonance `|E|²` map rel-L2 1e-4). `profile_engine.py --detector phasor` measures the residual per-step recording cost.

## Apple-Silicon ceilings

The equal-traffic Metal:CPU ratio is factor (a) above — multiply by up to ~4x (factor b) for the full custom-kernel ceiling. The *ratio* over CPU is ~constant because Apple scales CPU and GPU bandwidth together, except where bandwidth outruns the CPU ceiling (M4 Max, Ultra); a bigger chip's decisive wins are **absolute throughput** (∝ BW) and **capacity** (RAM → domains a discrete GPU can't hold). Max-N is a rough isotropic estimate (~50 B/cell double-buffered, 70% working set); all numbers are model estimates anchored to one M4 Pro measurement.

| Chip | Rated BW (GB/s) | Max RAM | GPU sustained | Metal:CPU ceiling (equal-traffic) | ~max iso N |
|---|--:|--:|--:|:--:|--:|
| M1 Pro | 200 | 32 GB | ~170 | ~1.4x | ~760 |
| M2 Pro | 200 | 32 GB | ~170 | ~1.4x | ~760 |
| M3 Pro | 150 | 36 GB | ~128 | ~1.4x | ~790 |
| M4 Pro | 273 | 64 GB | ~240 (meas) | ~1.4x (measured 1.37) | ~965 |
| M1 Max | 400 | 64 GB | ~340 | ~1.4x | ~965 |
| M2 Max | 400 | 96 GB | ~340 | ~1.4x | ~1100 |
| M3 Max | 300–400 | 128 GB | ~255–340 | ~1.4x | ~1220 |
| M4 Max | 410–546 | 128 GB | ~350–464 | ~1.4–1.9x | ~1220 |
| M1/M2 Ultra | 800 | 128 / 192 GB | ~680 | ~1.4x (≤~2.7x if CPU caps) | ~1220 / 1390 |
| M3 Ultra | 800 | 512 GB | ~680 | ~1.4x (≤~2.7x if CPU caps) | ~1930 |
