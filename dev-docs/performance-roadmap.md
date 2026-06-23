# Performance roadmap — beating the memory-bandwidth floor

The MLX/Metal bulk update already runs at the **memory-bandwidth floor**, so further speedups cannot come from moving bytes faster — they must come from **moving fewer bytes per step**. This document is the standalone plan for that: the model, how to locate bottlenecks, the ranked strategies, and the decisions already made (including why a memory-layout transpose is *not* the answer). A new contributor can start here.

**Hard constraints.** fp32 throughout (accuracy is non-negotiable — no fp16/mixed precision). Every change stays **expandable** (no scenario-specific fast paths that a different boundary/source/material would break) and **parity-validated** element-wise against the forced-JAX-CPU oracle (`rel < 1e-3`; marginal failure → raise resolution, never loosen tolerance). Background: [`performance.md`](../docs/performance.md) (roofline, kernel design, current numbers, Apple-Silicon ceilings) and [`porting.md`](porting.md) (the JAX→MLX kernel recipe).

## 1. Where the engine is now

| state (N=192 iso, M4 Pro) | Mcs/s | RT/step |
|---|--:|--:|
| compiled MLX-op cores, CPML on | 277 | 36 |
| **Metal kernel, CPML folded in (default)** | **1826** | **5** |
| Metal kernel, CPML off | 2219 | 5 |
| read-once/write-once floor | ~3150 | ~3 |
| full `run_fdtd` (incl. sources + monitors + bridge) | ~1370 | — |

Two facts set the whole agenda: (a) the bulk kernel is within ~1.7× of the hard floor and the rest of that gap is CPML ψ + material traffic, not inefficiency; (b) the **full `run_fdtd` (~1370) sits well below the pure loop (1826)** because of per-step monitor interpolation, source injection, and the host↔device bridge — that is removable fat *before* the kernel is touched.

On a *real* workload the overhead is far larger than the N=192 microbench suggests. The **reference workload** (§3.1) — the O-band MRM at 25 nm — runs at **≈250 Mcs/s end-to-end, ~7× below the bulk-kernel rate**, and almost all of that gap is per-step full-domain interpolation + monitor accumulation, not the update. That is the headroom Phases 1–2 chase.

## 2. The model: round-trips (RT)

FDTD is memory-bound. **1 RT = read+write of one `(3,N³)` field.** A single step's hard floor is ~3 RT (read E, read H, write E for the E-update; same for H), plus material reads and, on boundary slabs, ψ. The roofline is **240 GB/s** sustained (88% of the M4 Pro's 273 spec); per-step time ≈ `RT × bytes / 240 GB/s`. **Below the floor is only reachable by reducing the number of full-array passes** (temporal blocking) or the bytes per pass (precision — *off the table here*). Everything in §5 is "drive RT/step down."

## 3. How to locate bottlenecks (profiling)

### 3.1 Reference workload (the before/after metric)

Use a single, fixed, reproducible end-to-end run as the headline scoreboard, so every optimization is a direct wall-time comparison on the same machine: the **O-band MRM at the 25 nm gap-sweep grid**, [`examples/ring_mrm_oband/field_maps_100nm.py`](../examples/ring_mrm_oband/field_maps_100nm.py).

| | value |
|---|---|
| grid | 336 × 306 × 48 ≈ **4.9 M cells** |
| steps | ~73 k (3.5 ps settle) |
| monitors | through-port + input phasors (small boxes) **plus a full in-plane `|E|²` slice** (18 λ) |
| **current wall** | **≈ 1478 s (~25 min)** on the dev machine (single Metal GPU) |
| end-to-end throughput | ≈ **250 Mcs/s** — ~7× below the bulk-kernel pure-loop rate |

The ~7× gap is almost entirely **per-step full-domain field interpolation + monitor accumulation**, not the update — confirmed by the lighter-monitor gap-sweep ring run (same grid, through-port phasors only, no field slice) landing at **≈ 1381 s**: dropping the large field-slice monitor changes the wall by only ~7%, so the cost is the fixed per-step interpolation (§5-D), not the monitor size or frequency count. Re-run this exact script after each phase; **the wall time is the metric.** (The synthetic `profile_engine.py` sweep below is still used to isolate the *kernel* RT floor and the large-N spill, which the end-to-end number conflates.)

### 3.2 Instrumentation

1. **RT/step scoreboard** — [`benchmarks/profile_engine.py`](../benchmarks/profile_engine.py) reports implied RT/step at the 240 GB/s roofline. Extend it with a **monitor axis** (`--detector phasor`) so the recording overhead is measured, not just the update.
2. **Large-N spill test (the key decision gate for tiling)** — sweep **N = 256 / 384 / 512 / 768** and watch RT/step. If it stays flat at ~5, the L2 cache is absorbing the stencil's neighbour reuse and spatial tiling is a no-op; if RT/step **climbs with N**, the cache is spilling on the strided x/y neighbour lines (see §4) and tiling is justified. The project targets *large* unified-memory domains, so run this before designing any tile.
3. **Metal GPU counters** — capture a `.gputrace` (`mx.metal.start_capture`/`stop_capture`) and read **achieved DRAM bandwidth** (vs 240 GB/s), **occupancy**, and **memory-stall %** per kernel. ~85–90% of 240 GB/s confirms bandwidth-bound (don't touch ALU); lower means a latency/occupancy problem (tile/threadgroup tuning).
4. **Component attribution** — time bulk-only ([`benchmarks/m1_kernel.py`](../benchmarks/m1_kernel.py)) → +CPML → +sources → +monitors to split "floor" from "removable overhead." Validate every kernel change against forced-JAX.

## 4. The Yee stencil reuse structure

The curl is a **one-sided 3-neighbour stencil**, and each H component is reused along only **two** axes (from [`mlx/kernels.py`](../src/fdtdx/mlx/kernels.py) `_field_source`): `Hx` at own/y−1/z−1; `Hy` at own/x−1/z−1; `Hz` at own/x−1/y−1. So of the 6 neighbour reads per cell, **2 are z−1 and 4 are x/y**, each value shared by a cell and its +1 neighbour along two axes (~2× redundancy per axis).

In the current **z-contiguous** layout (`(3,NX,NY,NZ)`, z innermost; SIMD-group = 32 consecutive z-lanes), **all neighbour reads are already coalesced 128-byte lines** (32 z at z, at y−1, at x−1). The redundancy is **cross-threadgroup**: the line at (x, y−1, :) is loaded by the (x,y) group and the (x,y−1) group. At **N ≲ 256** L2 absorbs that → the kernel is at the floor. At **large N** the y/x-adjacent groups execute far apart, the line is evicted before reuse, and it re-streams from DRAM. **That large-N spill is the only thing spatial tiling fixes** — which is why §3.2 is the gate.

## 5. Strategies (ranked by ROI)

### D — monitor traffic (free, do first)
When monitors record, [`mlx/loop.py`](../src/fdtdx/mlx/loop.py) interpolates the **whole domain** every step (`pad_fields_mlx` + `interpolate_fields_mlx` + `(H_prev+H)/2` over the full `(3,N³)` E and H) and only then slices per monitor in [`mlx/accumulate.py`](../src/fdtdx/mlx/accumulate.py). This fixed per-step full-domain pass is the dominant cost of the reference workload (§3.1) — it is what makes the end-to-end run ~7× slower than the bulk kernel, independent of how many monitors or frequencies there are. Three compounding fixes, all exact for the linear (phasor/field) monitors:
- **Interpolate only the monitor regions** (the slice + a 1-cell halo), not the full domain — this alone removes the per-step full-domain pass.
- **Defer interpolation to the end:** interpolation is linear and phasor/field accumulation is linear (`buf += EH·phasor`), so `interp(Σ wₙ Fₙ) = Σ wₙ interp(Fₙ)` — accumulate the raw-grid phasor and interpolate once at the end (fold the H time-average into the phasor weights). Energy/Poynting are quadratic → stay per-step.
- **Auto-subsample the DFT:** the field's temporal frequency is the free-space `c/λ` (independent of `n_eff`), so from the monitor's `f_max` and `dt` the default stride is `floor(T_min/(k·dt))` with an oversampling margin `k≈8–12` (the FDTD `dt` is already ~10–20× below Nyquist).

**Gain: large on monitor-heavy runs — on the §3.1 reference workload it targets most of the ~7× end-to-end overhead; exact for linear monitors; low effort.**

### C — material compaction (exact; biggest on the anisotropic path)
The kernel reads `cb = c·inv_eps` per cell even when a region is uniform (it only uses the scalar-literal path when the *whole* material is a Python scalar). **Collapse homogeneity per subdivision**: a uniform tile carries a **single scalar (isotropic) or single tensor (full-anisotropic)** read once into registers and broadcast, eliminating the per-cell material array entirely. The saving scales with component count: iso-uniform eliminates ~1 RT; full-tensor-uniform eliminates the A/B (~18 floats/cell ≈ several RT). Heterogeneous tiles fall back to per-cell (or RLE). For the common "uniform background + localised device," the bulk goes to ~zero material traffic. **Do not** recompute the factor from raw `inv_eps+σ` each step (same bytes in, plus a 3×3 solve — a loss); compute it from a compact descriptor. Composes with the per-tile descriptor in A. **Gain: small for iso bulk, large for heterogeneous full-tensor.**

### A — interior temporal blocking (the main sub-floor lever)
Advance the **homogeneous interior** several steps per DRAM pass via **trapezoidal (overlapped) tiling**: a tile loads a `T`-deep halo and advances `T` steps with its valid region shrinking by one per sub-step, with **no cross-tile reads within the `T` steps** → race-free by construction, no within-block global sync. Inside a tile, E↔H is a `threadgroup_barrier`; the one unavoidable global sync (Metal has no device-wide barrier inside a kernel) becomes the **kernel boundary** — one launch per `T`-block instead of per step. Keep **boundary/CPML tiles at `T=1`** (ψ recurrence + the seam); the interior halo must cover the boundary's `T`-step reach. Halo *memory* is negligible (surface/volume); the halo *recompute* caps practical depth at `T≈2–4`. The hard part is correctness, which the trapezoid resolves. **Gain ~2–3×.** This is where a 5–6× would come from, and it **subsumes spatial tiling** (it stages the same on-chip tile, in the existing z-fast layout — see §6).

### F — spatial tiling / SIMD: fold into A, do not pursue standalone
The valuable spatial reuse (the 4 x/y neighbour reads that spill at large N) is **a free side-effect of A's tiling** in the existing z-fast layout. The only cheap standalone piece is a `simd_shuffle_up` for the **z** neighbour — but z is already the cheap, same-line read, so it is **low ROI**. The high-value xy reuse via SIMD shuffle would require an **xy-contiguous layout transpose**, which is invasive and *not* clearly faster (§6). **Decision: implement the on-chip tile inside A (z-fast, threadgroup memory + halo); add z-shuffle only as a micro-opt if profiling shows z-neighbour misses; do not transpose.** Useful SIMD elsewhere: **SIMD-group reductions** (`simd_sum`/`simd_prefix`) for volume-integrating monitors (`reduce_volume`) and the mode-overlap integral (today `mx.sum` tree-reductions); and the anisotropic off-diagonal averaging (reads neighbouring components). Interpolation is *not* a SIMD target once D defers it to the end.

### E — graph contiguity (longer-term; readability trade-off)
Source injection and PEC/PMC masks currently run **between** the two compiled cores in [`mlx/loop.py`](../src/fdtdx/mlx/loop.py), breaking the fused graph. A general, BC/source-agnostic fusion needs two abstractions: **source as an additive per-cell buffer** (every source type reduces to "add Δ to these cells this step," host-computed cheaply) and **BC as a mask** (PEC/PMC already; periodic wrap and CPML already in-kernel). Then the whole step fuses into one graph. The work is the source-buffer plumbing without regressing the sparse-source fast path; it trades the current per-substep readability, so defer until the perf case justifies it. **Gain ~1.1–1.3× (fewer launches/evals).**

## 6. Memory layout: why xy-contiguous is not necessarily faster

It is tempting to transpose the field arrays to make x/y the fast axis so the expensive x/y neighbour reuse can be served by `simd_shuffle`. It is **not** clearly a win:

- **SIMD shuffle reaches only within a 32-lane SIMD-group, and the lanes map to whatever axis is contiguous.** So the shuffle axis is *forced equal* to the contiguous axis. You can have z-contiguous **with** z-shuffle, or xy-contiguous **with** xy-shuffle — not a free mix. z-shuffle targets the cheap (same-line) neighbour; xy-shuffle targets the expensive ones but only exists in the xy-contiguous layout.
- **In the current z-contiguous layout the neighbour reads are already coalesced** (§4). xy-contiguity does **not** fix a coalescing problem — there isn't one — it only changes *which* redundancy can be served from a register/shuffle vs the cache.
- **The reuse xy-contiguity would capture is already captured by A's on-chip tile**, which works in the existing z-fast layout (a tile of contiguous z-columns + xy halo loads coalesced per column). So the transpose buys a single-step reuse that temporal blocking gets anyway.
- **The transpose is invasive and diverges from upstream.** Every field/material/ψ array, every kernel's indexing, the CPML slab geometry, and detector slicing assume fdtdx's `(3,NX,NY,NZ)` z-fast layout — the same layout the element-wise-parity bridge relies on. Transposing means re-validating the entire stack and maintaining a layout that diverges from fdtdx.

**Conclusion:** keep the **z-contiguous** layout. Capture x/y reuse through A's threadgroup-memory tile, not through a layout transpose. Reserve the xy-contiguous option only if a late profiling pass proves the x/y DRAM spill is the final bottleneck *and* temporal blocking did not capture it — an unlikely outcome that would still have to justify the parity-divergence cost.

## 7. How the balance shifts by application

| Regime | Tiling / F | Material (C) | Notes |
|---|---|---|---|
| iso / diagonal | helps only at large N | small | at N≤256 it's at the floor; lever is pure field reuse |
| full-tensor anisotropic | helps more (bigger stencil) | **large** | material dominates → C first; path is MLX-op cores today |
| uniform / homogeneous | best case (no divergence) | material → free | ideal for both C and tiling |
| compact inclusion | tile the bulk | vacuum → scalar | block-hybrid already isolates the inclusion |
| sparse / subpixel-smoothed | tile divergence; per-tile homogeneity flag | **degrades** (smoothed interfaces become tensors) | smoothed cells route to the aniso/correction path |
| heterogeneous | field tiling still helps | degrades (expected) | reuse is geometry-independent |
| CPML | interior tiles clean; boundary tiles T=1 | — | seam needs the interior halo to reach the boundary |
| PEC / PMC | trivial (post-mask) | — | composes with any tiling |
| periodic | tile halo wraps at the edge | — | edge-case in tile loading |
| Bloch / complex (future) | more valuable (2× data) | — | complex arithmetic + register pressure |

## 8. Staged plan and targets

1. **Phase 1 — free overhead (do first).** D + E + the `profile_engine.py` monitor axis. On a light-monitor run this closes the `1370 → 1826` gap (~1.4×); on the **§3.1 reference workload** (monitor-bound, currently **1478 s**) it removes the dominant per-step full-domain interpolation, so expect a **multi-× wall-time drop**. Zero physics risk.
2. **Phase 2 — material + profiling.** C (per-subdivision homogeneity collapse) + the §3.2 large-N spill sweep to decide A's tile design. **~1.2–1.3×**, exact.
3. **Phase 3 — temporal blocking.** A, interior-only, boundary tiles `T=1`, parity-gated at each depth `T`. **~2–3×** on the update — the lever for 5–6×, gated on the spill test showing the cache actually spilling at the target N.

**Targets (fp32-only, expandable):** **~2.5–3× confident** from Phases 1–2 plus A's spatial tiling; **5–6× stretch**, reachable only if temporal blocking lands at depth ~3–4 with good efficiency. A pure-time-loop 5–6× (no monitors) rests entirely on A. **The headline metric is the §3.1 reference workload**: after each phase, re-run `field_maps_100nm.py` and compare the wall against the **1478 s** baseline. Use the synthetic `profile_engine.py` sweep only to attribute *which* strategy moved the number (kernel RT vs monitor overhead vs spill).

## 9. Validation discipline

Every change carries a `validation`-marked element-wise parity test vs forced-JAX-CPU (`rel < 1e-3`, fp32 floor), a physics sanity check where cheap (Fresnel/cavity/FSR), and an RT/step regression on `profile_engine.py`. Temporal blocking and material compaction are gated like the existing block-hybrid: fall back to the current kernel/op cores for any case a new path does not yet cover, so the engine is always correct and only conditionally faster.
