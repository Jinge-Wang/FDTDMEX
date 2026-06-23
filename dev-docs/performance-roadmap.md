# Performance roadmap — beating the memory-bandwidth floor

The MLX/Metal bulk update already runs at the **memory-bandwidth floor**, so further speedups cannot come from moving bytes faster — they must come from **moving fewer bytes per step**. The per-step monitor/interpolation overhead has already been removed (region-restricted interpolation + activity-gating + DFT auto-subsampling — see [`docs/performance.md`](../docs/performance.md) §"Monitor recording"); that closed the dominant *full `run_fdtd`* overhead on monitored runs without touching the kernel. What remains is the harder, more valuable frontier: going **below the per-step floor**. This document is the standalone plan for that — the model, how to locate bottlenecks, the single next deliverable, and the decisions already made (including why a memory-layout transpose is *not* the answer). A new contributor can start here.

**Hard constraints.** fp32 throughout (accuracy is non-negotiable — no fp16/mixed precision). Every change stays **expandable** (no scenario-specific fast paths that a different boundary/source/material would break) and **parity-validated** element-wise against the forced-JAX-CPU oracle (`rel < 1e-3`; marginal failure → raise resolution, never loosen tolerance). Background: [`docs/performance.md`](../docs/performance.md) (roofline, kernel design, current numbers, Apple-Silicon ceilings) and [`porting.md`](porting.md) (the JAX→MLX kernel recipe).

## 1. Where the engine is now

| state (N=192 iso, M4 Pro) | Mcs/s | RT/step |
|---|--:|--:|
| compiled MLX-op cores, CPML on | 277 | 36 |
| **Metal kernel, CPML folded in (default)** | **1826** | **5** |
| Metal kernel, CPML off | 2219 | 5 |
| read-once/write-once floor | ~3150 | ~3 |
| full `run_fdtd` (incl. sources + bridge; monitors now cheap) | ~1370 | — |

Two facts set the agenda: (a) the bulk kernel is within ~1.7× of the hard floor and the rest of that gap is CPML ψ + material traffic, not inefficiency; (b) the per-step **monitor** overhead that used to drag the full `run_fdtd` far below the pure loop is gone, so the remaining full-run gap is source injection + the host↔device bridge. **The only lever left that beats the floor is reducing the number of full-array passes per step (temporal blocking) and the material bytes per cell (per-tile compaction)** — and those two are the same piece of infrastructure, which is what §5 builds.

## 2. The model: round-trips (RT) + material bytes

FDTD is memory-bound. The per-step cost has **two** terms, and a sub-floor engine has to attack both:

- **Field round-trips. 1 RT = read+write of one `(3,N³)` field.** A single step's hard floor is ~3 RT (read E, read H, write E for the E-update; same for H), plus, on boundary slabs, ψ. The roofline is **240 GB/s** sustained (88% of the M4 Pro's 273 spec); per-step time ≈ `RT × bytes / 240 GB/s`. **Below the floor is only reachable by reducing the number of full-array passes** (temporal blocking) or the bytes per pass (precision — *off the table here*).
- **Material bytes per cell.** The kernel also reads the per-cell coefficient (`cb = c·inv_eps`, plus conductivity / the full A/B for tensors). In the current **dense** layout this width is set *globally* by the worst-case voxel: a single 9-tensor cell makes the whole `inv_eps` a `(9,N³)` buffer, so **every** cell pays tensor-width read traffic; even an all-diagonal but heterogeneous domain pays a full `(3,N³)` per-cell read where most cells are identical background. This is a second, independent lever — collapse the material traffic so per-cell bytes track the *local* material, not the rarest one anywhere in the domain.

Everything in §5 drives both terms down at once.

## 3. How to locate bottlenecks (profiling)

### 3.1 Reference workload (the before/after metric)

Use a single, fixed, reproducible end-to-end run as the headline scoreboard: the **O-band MRM at the 25 nm gap-sweep grid**, [`examples/ring_mrm_oband/field_maps_100nm.py`](../examples/ring_mrm_oband/field_maps_100nm.py).

| | value |
|---|---|
| grid | 336 × 306 × 48 ≈ **4.9 M cells** |
| steps | ~73 k (3.5 ps settle) |
| monitors | through-port + input phasors (small boxes) **plus a full in-plane `|E|²` slice** (18 λ) |
| **current wall** | **377 s** (after the monitor-overhead removal; was ~1478 s) |

Re-run this exact script after each change; **the wall time is the metric.** To compare against a saved baseline without overwriting it, set `MRM_OUT_SUFFIX` (the optimized run writes `_optimized`-suffixed artifacts). The synthetic `profile_engine.py` sweep below is used to isolate the *kernel* RT floor and the large-N spill, which the end-to-end number conflates.

### 3.2 Instrumentation

1. **RT/step scoreboard** — [`benchmarks/profile_engine.py`](../benchmarks/profile_engine.py) reports implied RT/step at the 240 GB/s roofline, with a `--detector {none,energy,phasor}` axis so the recording overhead is measured, not just the update.
2. **Large-N spill test (the key decision gate for tiling)** — sweep **N = 256 / 384 / 512 / 768** and watch RT/step. If it stays flat at ~5, the L2 cache is absorbing the stencil's neighbour reuse and spatial tiling is a no-op; if RT/step **climbs with N**, the cache is spilling on the strided x/y neighbour lines (§4) and tiling is justified. The project targets *large* unified-memory domains, so **run this before designing the tile.**
3. **Metal GPU counters** — capture a `.gputrace` (`mx.metal.start_capture`/`stop_capture`) and read **achieved DRAM bandwidth** (vs 240 GB/s), **occupancy**, and **memory-stall %** per kernel. ~85–90% of 240 GB/s confirms bandwidth-bound (don't touch ALU); lower means a latency/occupancy problem (tile/threadgroup tuning).
4. **Component attribution** — time bulk-only ([`benchmarks/m1_kernel.py`](../benchmarks/m1_kernel.py)) → +CPML → +sources to split "floor" from "removable overhead." Validate every kernel change against forced-JAX.

## 4. The Yee stencil reuse structure

The curl is a **one-sided 3-neighbour stencil**, and each H component is reused along only **two** axes (from [`mlx/kernels.py`](../src/fdtdx/mlx/kernels.py) `_field_source`): `Hx` at own/y−1/z−1; `Hy` at own/x−1/z−1; `Hz` at own/x−1/y−1. So of the 6 neighbour reads per cell, **2 are z−1 and 4 are x/y**, each value shared by a cell and its +1 neighbour along two axes (~2× redundancy per axis).

In the current **z-contiguous** layout (`(3,NX,NY,NZ)`, z innermost; SIMD-group = 32 consecutive z-lanes), **all neighbour reads are already coalesced 128-byte lines** (32 z at z, at y−1, at x−1). The redundancy is **cross-threadgroup**: the line at (x, y−1, :) is loaded by the (x,y) group and the (x,y−1) group. At **N ≲ 256** L2 absorbs that → the kernel is at the floor. At **large N** the y/x-adjacent groups execute far apart, the line is evicted before reuse, and it re-streams from DRAM. **That large-N spill is the only thing spatial tiling fixes** — which is why §3.2 is the gate.

## 5. The next deliverable: a tiled sub-floor engine

There is one coherent next step, and it **binds three formerly-separate ideas — interior temporal blocking, per-tile material compaction, and spatial tiling — into a single tiled kernel**, because they share the same piece of infrastructure (an on-chip threadgroup tile) and only compose cleanly when built together. A standalone per-cell material change on the current flat kernel would be thrown away the moment tiling lands, and it could not retire the block hybrid without per-cell branching; the tile is what makes the material compaction divergence-free. So they ship as one unit.

### 5.1 Temporal blocking (the sub-floor lever, ~2–3×)

Advance the **homogeneous interior** several steps per DRAM pass via **trapezoidal (overlapped) tiling**: a tile loads a `T`-deep halo and advances `T` steps with its valid region shrinking by one per sub-step, with **no cross-tile reads within the `T` steps** → race-free by construction, no within-block global sync. Inside a tile, E↔H is a `threadgroup_barrier`; the one unavoidable global sync (Metal has no device-wide barrier inside a kernel) becomes the **kernel boundary** — one launch per `T`-block instead of per step. Keep **boundary/CPML tiles at `T=1`** (ψ recurrence + the seam); the interior halo must cover the boundary's `T`-step reach. Halo *memory* is negligible (surface/volume); the halo *recompute* caps practical depth at `T≈2–4`. The hard part is correctness, which the trapezoid resolves.

### 5.2 Per-tile material compaction (the second lever, large on heterogeneous / full-tensor)

The threadgroup tile is the natural unit of material homogeneity, so the material layer rides on it:

- **Uniform tile → a single descriptor**, loaded once into threadgroup memory / registers and broadcast across the tile (a scalar for isotropic, the diagonal for diagonal, or the full tensor for anisotropic — *the same machinery for all three*), eliminating the per-cell material array entirely. For the common "uniform background + localised device," the bulk goes to **~zero material traffic.**
- **Heterogeneous / anisotropic tile → tile-local material**: the tile's distinct descriptors are loaded into threadgroup memory and indexed locally, and the update path is **specialised at tile granularity** — only tiles that actually contain a tensor/lossy voxel run the tensor/lossy path. Because the specialisation is per-tile (uniform across the SIMD-group), there is **no per-cell `if/else` and no warp divergence**.
- This decouples both per-cell *traffic* and *compute* from the global worst-case material width: a lone 9-tensor voxel costs **one tile**, not the whole domain (§2). It also **retires the block hybrid** — there is no longer a compact-box / PML-disjoint gate, and scattered tensor inclusions are handled in-kernel.
- Compute the per-cell/per-tile factor from a **compact descriptor**, never by re-deriving it from raw `inv_eps + σ` each step (same bytes in, plus a 3×3 solve — a loss).

### 5.3 Spatial reuse folds in for free

The valuable spatial reuse — the 4 x/y neighbour reads that spill at large N (§4) — is a **side-effect of the on-chip tile** in the existing z-fast layout (a tile of contiguous z-columns + xy halo loads coalesced per column). The only cheap standalone extra is a `simd_shuffle_up` for the **z** neighbour, but z is already the cheap same-line read, so it is **low ROI**: add z-shuffle only if profiling shows z-neighbour misses; **do not transpose** (§6). Useful SIMD elsewhere, independent of the tile: **SIMD-group reductions** (`simd_sum`/`simd_prefix`) for volume-integrating monitors (`reduce_volume`) and the mode-overlap integral.

### 5.4 Metal / MLX expressibility

The primitives all exist — `threadgroup` memory (= CUDA `__shared__`), `threadgroup_barrier`, `simd_broadcast`/`simd_shuffle_up`. The catch is that today's `mx.fast.metal_kernel` body is **flat per-thread** (a global `idx`, no tile, no shared memory — see [`kernels.py`](../src/fdtdx/mlx/kernels.py) `_common`). Introducing the threadgroup tile *is* the core new work here; once it exists, both the temporal depth (§5.1) and the per-tile material load (§5.2) hang off it.

### 5.5 Gating, gain, and the spill gate

Gate like the existing block hybrid: fall back to the current kernel / op cores for any case the tiled path does not yet cover, so the engine is always correct and only conditionally faster, parity-gated at **each depth `T`** and **each material class**. **Gain:** ~2–3× from temporal blocking plus the material-traffic collapse (small on iso bulk, large on heterogeneous / full-tensor domains); **5–6× stretch** only if depth ~3–4 lands with good efficiency. **Gated on the §3.2 large-N spill test** showing the cache actually spilling at the target N — run that sweep before designing the tile.

## 6. Remaining smaller item

### E — graph contiguity (longer-term; readability trade-off)
Source injection and PEC/PMC masks currently run **between** the two compiled cores in [`mlx/loop.py`](../src/fdtdx/mlx/loop.py), breaking the fused graph. A general, BC/source-agnostic fusion needs two abstractions: **source as an additive per-cell buffer** (every source type reduces to "add Δ to these cells this step," host-computed cheaply) and **BC as a mask** (PEC/PMC already; periodic wrap and CPML already in-kernel). Then the whole step fuses into one graph. The work is the source-buffer plumbing without regressing the sparse-source fast path; it trades the current per-substep readability, so defer until the perf case justifies it. **Gain ~1.1–1.3× (fewer launches/evals).**

## 7. Memory layout: why xy-contiguous is not necessarily faster

It is tempting to transpose the field arrays to make x/y the fast axis so the expensive x/y neighbour reuse can be served by `simd_shuffle`. It is **not** clearly a win:

- **SIMD shuffle reaches only within a 32-lane SIMD-group, and the lanes map to whatever axis is contiguous.** So the shuffle axis is *forced equal* to the contiguous axis. You can have z-contiguous **with** z-shuffle, or xy-contiguous **with** xy-shuffle — not a free mix. z-shuffle targets the cheap (same-line) neighbour; xy-shuffle targets the expensive ones but only exists in the xy-contiguous layout.
- **In the current z-contiguous layout the neighbour reads are already coalesced** (§4). xy-contiguity does **not** fix a coalescing problem — there isn't one — it only changes *which* redundancy can be served from a register/shuffle vs the cache.
- **The reuse xy-contiguity would capture is already captured by the §5 on-chip tile**, which works in the existing z-fast layout. So the transpose buys a single-step reuse that the tile gets anyway.
- **The transpose is invasive and diverges from upstream.** Every field/material/ψ array, every kernel's indexing, the CPML slab geometry, and detector slicing assume fdtdx's `(3,NX,NY,NZ)` z-fast layout — the same layout the element-wise-parity bridge relies on. Transposing means re-validating the entire stack and maintaining a layout that diverges from fdtdx.

**Conclusion:** keep the **z-contiguous** layout. Capture x/y reuse through the §5 threadgroup-memory tile, not through a layout transpose. Reserve the xy-contiguous option only if a late profiling pass proves the x/y DRAM spill is the final bottleneck *and* the tile did not capture it — an unlikely outcome that would still have to justify the parity-divergence cost.

## 8. How the balance shifts by application

| Regime | Temporal tile | Per-tile material | Notes |
|---|---|---|---|
| iso / diagonal | helps only at large N | small | at N≤256 it's at the floor; lever is pure field reuse |
| full-tensor anisotropic | helps more (bigger stencil) | **large** | material dominates → the per-tile collapse is the main win; lone tensor voxel costs one tile |
| uniform / homogeneous | best case (no divergence) | material → ~free | ideal for both depth and the descriptor broadcast |
| compact inclusion | tile the bulk | inclusion → its own tiles | replaces the block-hybrid carve-out |
| sparse / subpixel-smoothed | tile divergence; per-tile homogeneity flag | smoothed interfaces become tensor tiles | smoothed cells route to the per-tile tensor path |
| heterogeneous | field tiling still helps | per-cell bytes track local material, not the global worst case | the §2 contamination fix |
| CPML | interior tiles clean; boundary tiles `T=1` | — | seam needs the interior halo to reach the boundary |
| PEC / PMC | trivial (post-mask) | — | composes with any tiling |
| periodic | tile halo wraps at the edge | — | edge-case in tile loading |
| Bloch / complex (future) | more valuable (2× data) | — | complex arithmetic + register pressure |

## 9. Staged plan and targets

1. **Spill gate (do first).** Run the §3.2 large-N sweep (N = 256/384/512/768). If RT/step is flat, the tile's spatial-reuse value is small and the design should lean on temporal depth + material compaction; if it climbs, the tile's xy capture is also load-bearing. This decides the tile geometry.
2. **The tiled sub-floor engine (§5).** Temporal blocking (interior, boundary `T=1`) + per-tile material compaction + the folded-in spatial tile, as one deliverable. Parity-gated at each depth `T` and each material class, falling back to the current kernel/op cores for uncovered cases. **~2–3× confident** (more on heterogeneous/full-tensor via the material collapse); **5–6× stretch** if depth ~3–4 lands efficiently.
3. **E — graph contiguity (§6).** Source-as-buffer + BC-as-mask fusion. **~1.1–1.3×**, deferred until the perf case justifies the readability trade.

**Headline metric:** after each change, re-run [`field_maps_100nm.py`](../examples/ring_mrm_oband/field_maps_100nm.py) (with `MRM_OUT_SUFFIX`) and compare the wall against the **377 s** current baseline. Use `profile_engine.py` to attribute *which* lever moved the number (kernel RT vs material bytes vs spill).

## 10. Validation discipline

Every change carries a `validation`-marked element-wise parity test vs forced-JAX-CPU (`rel < 1e-3`, fp32 floor), a physics sanity check where cheap (Fresnel/cavity/FSR), and an RT/step regression on `profile_engine.py`. Temporal blocking and per-tile material compaction are gated like the existing block hybrid: fall back to the current kernel/op cores for any case a new path does not yet cover, so the engine is always correct and only conditionally faster.
