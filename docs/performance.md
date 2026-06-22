# Performance — roofline, model, and current results

Measured reference for the MLX/Metal forward engine on Apple Silicon. fp32 throughout.

## Current scaling (M4 Pro)

![Forward scaling — MLX/Metal vs JAX-CPU](images/forward_scaling.png)

`benchmarks/results/scaling_s500.jsonl` (500 steps, warmup excluded → steady-state wall-clock).
**MLX/Metal leads JAX-CPU for every N ≥ 64 across all three materials, with no plateau:**

| material | MLX Mcs/s (N=256) | JAX-CPU | speedup | crossover |
|---|--:|--:|--:|--:|
| isotropic | 266.8 | 194.7 | 1.37× | N≈64 |
| diagonal | 267.6 | 196.3 | 1.36× | N≈64 |
| full_aniso | 120.9 | 96.5 | 1.25× | N≈48–64 |

Below N≈48 JAX-CPU wins (MLX kernel-launch overhead dominates tiny domains). Panel (d) memory: MLX
peak is exact; the JAX line is in-process RSS (use `benchmarks/profile_memory.py` for a clean
per-cell figure).

## Roofline (M4 Pro, `benchmarks/profile_metal.py`)

- **Coalesced copy: 240 GB/s = 88% of the 273 GB/s spec** — the real ceiling (the spec is not
  achievable; 240 is the denominator for all roofline math).
- Component-leading `(3,N,N,N)` vs component-last `(N,N,N,3)` stencil: **1.00×** — no coalescing
  penalty from the layout.
- `roll`-diff vs slice-diff on the engine's `y − shift(y)` pattern: **0.89–1.13×** — `roll` is not a
  culprit.

## The round-trip (RT) model

FDTD is memory-bandwidth-bound. **1 RT = read+write of one `(3,N³)` field**; per-step time ≈
`RT × 170 MB / 240 GB/s` at N=192. The bottleneck is redundant traffic — too many full-array passes,
not arithmetic, dispatch-starvation, or layout (confirmed by toggling CPML, which removes exactly the
carried-ψ RT, and by `profile_engine.py`'s eager-vs-compiled × CPML 2×2).

| engine state (N=192 iso, compiled CPML-on) | Mcs/s | RT/step |
|---|--:|--:|
| original (pad+roll, full-domain CPML) | 105 | ~99 |
| + pad-free slice-diff (eager) | 130 | 77 |
| + `mx.compile` E/H cores | 211 | 47 |
| + slab-CPML (current default) | **277** | **36** |
| compiled, CPML off (MLX-op ceiling) | 473 | 21 |
| necessary floor (read E,H + materials; write E,H) | ~600+ | ~5–8 |

The MLX-op path is near its ceiling at ~21–36 RT; reaching the ~5–8 RT floor needs a custom Metal
kernel (Phase 2, `phase2-metal-kernels.md`).

## Metal vs CPU/JAX — two factors

CPU and GPU share one DRAM, so speedup is not "GPU flops":
- **(a) bandwidth-utilization gap (~1.4×, chip-dependent).** GPU sustains ~85% of rated unified BW;
  a multicore CPU sustains ~55–65% and caps at a per-die ceiling (~240 GB/s). This is the measured
  1.37× on M4 Pro at equal traffic; it widens only where rated BW outruns the CPU (top-bin Max,
  Ultra).
- **(b) traffic gap (chip-independent, the real prize).** JAX/XLA on CPU does not tile the stencil
  (effective traffic ~tens of RT, like the pre-Phase-1 engine). A fused Metal kernel at the ~5–8 RT
  floor adds up to ~4× on top of (a) — *if* JAX stays traffic-heavy (measured in Phase 2 M1).

Per-chip ceilings and the Apple-Silicon table: `phase2-metal-kernels.md` §8.
