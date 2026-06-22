# Phase 2 — custom Metal update kernels (design + staging)

> Status: **M1 ✅ + M2 ✅ + M3 ✅ done; Metal kernel path default-on.** Phase 1 default = 277 Mcs/s /
> 36 RT (MLX-op cores, CPML on). M2 landed the custom Metal E/H kernels in the engine
> ([`src/fdtdx/mlx/kernels.py`](../src/fdtdx/mlx/kernels.py)) with the CPML spatial hybrid (CPML-on
> 374 / 27 RT). **M3 folded CPML into the kernel (CPML-on 374 → 1826 Mcs/s / 5 RT, at the bulk
> floor), added in-kernel non-uniform metric, and a block hybrid for full-tensor inclusions
> (heterogeneous 125 → 1124 Mcs/s), then flipped `FDTDMEX_METAL_KERNEL` default-on** (`=0` forces the
> MLX-op path) — see [performance.md](performance.md) for the table + history. This document is the
> design/spec; §9 tracks the staging. fp32 only.

## 1. Goal

A forward step is memory-bound. The minimum traffic is ~5–8 full-array round-trips (RT) per step
(read E, H, and the material arrays; write E, H). The compiled MLX-op engine moves 21 RT (CPML off) /
36 RT (CPML on) because op-level fusion cannot keep a stencil's working set in on-chip memory or merge
neighbour reads — every op round-trips DRAM. One kernel per field that loads each value ~once and
writes once targets the ~5–8 RT floor → an estimated **~450–600 Mcs/s** (vs 277 today).

## 2. API (mlx 0.31.2, verified)

```python
kernel = mx.fast.metal_kernel(
    name, input_names=[...], output_names=[...], source=<MSL body>, header="", ensure_row_contiguous=True)
outs = kernel(inputs=[...], output_shapes=[...], output_dtypes=[...], grid=(gx,gy,gz), threadgroup=(tx,ty,tz))
```
- `source` is the **body** of the Metal function; the signature is generated from `input_names`/
  `output_names`. `T` is the element template type. Thread indices via `thread_position_in_grid`,
  `threadgroup_position_in_grid`, `thread_position_in_threadgroup`. Threadgroup memory via
  `threadgroup T tile[...]` in the body + `header` for helpers.
- The kernel is a normal node in the lazy graph (composes with the existing loop + `mx.eval`).

## 3. Update kernels — two separate kernels (E-update, H-update)

Leapfrog requires E and H in separate passes (H reads the just-updated E), so this is **one E-update
kernel and one H-update kernel**, called in sequence with host-gated source injection between them
(unchanged from the current loop):

```
E-kernel:  E_new  = E + c · inv_eps · curl_H(H)            (+ CPML correction on boundary slabs)
H-kernel:  H_new  = H − c · inv_mu  · curl_E(E_new)        (+ CPML correction on boundary slabs)
```

Per-field kernel structure:
- **Inputs:** the field being differentiated (H for E-kernel; E for H-kernel), the field being
  updated, the material array(s), and the CPML slab data + geometry. **Outputs:** the updated field
  (+ advanced ψ slabs).
- **Tiling / marching.** Partition the domain into XY tiles; each tile marches along the contiguous
  z axis, holding the z-plane(s) it currently needs in threadgroup memory so each field value is
  loaded from DRAM ~once per step. Layout stays `(3, N, N, N)` (z contiguous → coalesced loads).
- **Halo.** The Yee curl is a one-sided difference (cell `i` uses `i−1` *or* `i+1`, not both), so the
  tile needs a 1-cell halo on one side per axis only.
- **Domain edges.** Reproduce the existing ghost rule exactly: zero ghost on PML/PEC faces, wrapped
  neighbour on periodic faces (the pad-free slice-diff rule already in `curl.py`).

## 4. Race / buffering rule (from the field equations)

- **Isotropic update** reads only the cell's own old field value + the *other* field's neighbours
  (E-kernel reads its own `E[i]` and `H` neighbours). No same-field neighbour read → the updated
  field can be written in place; no halo of the field being updated.
- **Anisotropic update** mixes components: `E_new[i]` depends on the other E components averaged to
  this component's Yee site, i.e. it reads **neighbouring `E_old`**. So the anisotropic kernel must
  **double-buffer** the updated field (read old from an input buffer, write new to a separate output)
  and load a halo of that field. Same for H with a μ-tensor.

## 5. Region specialization for heterogeneous materials (single-Mac subdivision)

The current engine selects one global path: if any cell is full-tensor, the whole domain runs the
anisotropic update (with its neighbour-averaging). For heterogeneous domains (isotropic bulk + local
anisotropic inclusions — the target use case) we want the cheap isotropic update where the cell is
isotropic and the 3×3 A/B update only where it isn't.

**On a single Mac this is a compute-placement optimization, not a distributed-stencil problem.** A
distributed (multi-GPU/multi-node) implementation needs: per-partition iso/aniso cube lists, recorded
neighbours, explicit halo arrays exchanged between partitions, and compute/exchange overlap (streams)
— all driven by (a) in-place updates and (b) each device holding only its partition. **Neither
applies here:**
- **Unified memory:** one address space; every threadgroup can read the whole `E_old` directly. A
  boundary cell of an anisotropic region reads its neighbour's `E_old` by global index — no halo copy,
  no inter-device exchange. The "halo" is just a threadgroup-memory load from the shared buffer.
- **Functional / out-of-place = double-buffered:** all kernels read the frozen `E_old` and write a
  separate `E_new`. The cross-region race (a neighbour reading an already-overwritten cell) **cannot
  occur**, independent of which side is iso or aniso and of execution order. No ordering constraint,
  no interior/boundary split.

So subdivision reduces to placement, expressible three ways (increasing complexity):
1. **Per-cell branch in one kernel** — each thread reads a material-class tag; iso threads do the
   diagonal update, aniso threads the 3×3 + neighbour-average. Metal SIMD-groups are 32-wide and
   material regions are contiguous, so divergence is near-zero; a per-threadgroup "any aniso?" flag
   lets iso-only tiles skip the E-halo load. Simplest; likely sufficient.
2. **Block-tagged dispatch** — tag at tile granularity; launch the iso kernel over iso-tile indices
   and the aniso kernel over aniso-tile indices. Direct analog of the CUDA cube lists, but with **no
   neighbour bookkeeping and no halo arrays** (neighbours come from the shared `E_old`).
3. **Gather/scatter index sets** — full separation; only if (1)/(2) show real divergence cost.

Tile-size trade-off here is **occupancy / threadgroup-memory reuse** (larger tiles) vs **divergence
at iso/aniso interfaces** (tiles that don't straddle the boundary) — *not* comm-vs-compute. The
interface is a 2D surface in a 3D volume, so the straddling fraction is small and a single moderate
tile size suffices for most material distributions; no per-region adaptive sizing needed.

## 6. CPML

ψ and the κ-stretch correction are confined to the boundary slabs (Fix 1.2). In the kernel, the ψ
recurrence + correction run only on boundary-slab tiles; interior tiles compute the plain curl. The
per-axis slab extents (`pml.detect_pml_slabs`) and the slab ψ layout carry over directly.

## 7. Memory and in-place updates

In-place updates reduce **footprint, not bandwidth**: a bandwidth-bound stencil reads `E_old` (3N³),
reads `H` (3N³), and writes `E` (3N³) whether the write aliases the input or a fresh buffer —
identical traffic, identical throughput. In-place is therefore a **capacity** lever (larger max
domain), not a speed lever, and is decoupled from M1.

MLX's caching allocator reuses dead buffers, so out-of-place is throughput-equivalent (last step's
`E_old` buffer is reused for this step's `E_new` — no per-step alloc/zeroing). The only cost of
staying out-of-place is one transient extra field-array of peak memory (~2×E + H during the
E-update). **Measure `mx.get_peak_memory()` at the target N before building any manual in-place path**
— materials + the domain itself usually dominate, not that one array.

If footprint is the binding constraint (very large or interface-heavy domains), the capacity scheme:
keep one full field updated in place, plus a small compact buffer holding only the `E_old` cells some
neighbour's update reads. Race-free as **two passes with a barrier**: (1) gather the needed `E_old`
into the compact buffer; (2) in-place update — isotropic cells write in place (read own cell + `H`),
anisotropic/smoothed cells read stale neighbours from the compact buffer. The needed set is static
(the stencil support of every full-tensor cell) and includes **subpixel-smoothed interfaces**
(smoothing yields an effective ε *tensor* even between isotropic materials), so an interface-rich
device shrinks the saving. Index with a dense row-major bijection `x·Ny·Nz + y·Nz + z` (not base-10
powers — they leave gaps and overflow int32 past N≈1000); the compact buffer needs a sparse gather
list, and a global per-cell reverse map itself costs ~⅓ of a field, so keep the structure implicit
(process anisotropic regions block-wise). **Capacity optimization only — revisit after M1.**

## 8. Speedup ceiling vs CPU/JAX (roofline) + Apple-Silicon table

FDTD is memory-bound and CPU+GPU share one DRAM, so the speedup is two factors, not "GPU flops":

- **(a) Bandwidth-utilization gap (chip-dependent, ~1.4×).** GPU sustains ~0.85× rated unified BW
  (measured 0.88 on M4 Pro); a multicore CPU sustains ~0.55–0.65× of the *same* bus and caps at a
  per-die ceiling (~240 GB/s, the measured M1-Max CPU max). Back-calc from our data: JAX-CPU ≈ 170
  GB/s effective on M4 Pro → ceiling 0.85/0.63 ≈ **1.4×** at equal traffic. It widens only where rated
  BW outruns the CPU ceiling (top-bin Max, Ultra).
- **(b) Traffic gap (chip-independent, the real prize).** JAX/XLA on CPU, like the pre-Phase-1 engine,
  does not tile the stencil — it streams ~tens of full-array passes (compiled MLX ≈ 36 RT; JAX's
  effective traffic is similar). A fused kernel at the ~5–8 RT floor adds up to **~4×** on top of (a)
  *if* JAX stays traffic-heavy. **M1 must measure JAX's effective traffic** to size this.

The current 267 vs 190 Mcs/s (1.37×) is factor (a) alone (both at ~36 RT). The kernel chases (b).

Rated BW is documented; GPU-sustained ~0.85×; the ceiling column is the **equal-traffic** ratio (a) —
multiply by (b, ≤~4×) for the full custom-kernel ceiling. Max-domain N is a rough RAM estimate
(isotropic, ~50 B/cell double-buffered, 70% working set).

| Chip | Rated BW (GB/s) | Max RAM | GPU sustained | Metal:CPU ceiling (equal-traffic) | ~max iso N |
|---|--:|--:|--:|:--:|--:|
| M1 Pro | 200 | 32 GB | ~170 | ~1.4× | ~760 |
| M2 Pro | 200 | 32 GB | ~170 | ~1.4× | ~760 |
| M3 Pro | 150 | 36 GB | ~128 | ~1.4× | ~790 |
| M4 Pro | 273 | 64 GB | ~240 (meas) | ~1.4× (measured 1.37) | ~965 |
| M1 Max | 400 | 64 GB | ~340 | ~1.4× | ~965 |
| M2 Max | 400 | 96 GB | ~340 | ~1.4× | ~1100 |
| M3 Max | 300–400 | 128 GB | ~255–340 | ~1.4× | ~1220 |
| M4 Max | 410–546 | 128 GB | ~350–464 | ~1.4–1.9× | ~1220 |
| M1/M2 Ultra | 800 | 128 / 192 GB | ~680 | ~1.4× (≤~2.7× if CPU caps) | ~1220 / 1390 |
| M3 Ultra | 800 | 512 GB | ~680 | ~1.4× (≤~2.7× if CPU caps) | ~1930 |

The *ratio* over CPU is ~constant (Apple scales CPU & GPU BW together) except where BW outruns the CPU
(M4 Max, Ultra). A bigger chip's decisive wins are **absolute throughput** (∝ BW: Max/Ultra ≈ 1.5–3×
a Pro in Mcs/s) and **capacity** (RAM → domains a discrete GPU can't hold). Numbers are model
estimates anchored to one M4 Pro measurement; treat as order-of-magnitude.

## 9. Staging (M1 is the go/no-go)

1. **M1 — isotropic, uniform, interior (no CPML), go/no-go. ✅ GO (`benchmarks/m1_kernel.py`).**
   Thread-per-cell E and H `mx.fast.metal_kernel`s (curl read from global, neighbour reuse via cache),
   bit-exact vs the compiled MLX-ops same-math step (maxdiff 0.0). M4 Pro:

   | | Mcs/s | RT/step |
   |---|--:|--:|
   | compiled MLX-ops (same math) | 545–550 | ~18 |
   | **custom Metal kernels** | **3150–3190** | **~3** |

   The kernel reaches the **~3 RT bandwidth floor** (75 B/cell-step ≈ read+write E,H), a **5.8×**
   speedup over the compiled MLX-ops path, consistent at N=192 and 256 — so MLX op-fusion leaves ~6×
   on the floor for the bulk update, and factor (b) is confirmed large. (JAX-CPU's effective traffic
   is ~36 RT — its 195 Mcs/s ≈ compiled-MLX at the CPU's ~170 GB/s — so the floor kernel is far above
   it.) **Proceed.** Even discounted for CPML/sources, the headroom is decisive.
2. **M2 — + CPML, kernels in the engine. ✅ DONE ([`src/fdtdx/mlx/kernels.py`](../src/fdtdx/mlx/kernels.py)).**
   The M1 kernel generalised to per-cell materials (iso + diagonal) and non-cubic domains (per-axis
   Nx/Ny/Nz), with periodic wrap ghosts. CPML via the **spatial hybrid**, not in-kernel slab tiles:
   the kernel computes the full-domain plain-curl bulk, then the thin PML slabs get an additive MLX-op
   correction (`_slab_correction` → `curl._slab_add`/`_slab_diff`, reusing `detect_pml_slabs`
   geometry + slab ψ). Integrated in `loop.py` behind `FDTDMEX_METAL_KERNEL` (default off), MLX-op
   cores as fallback for ineligible cases (lossy / full-tensor / non-uniform metric). Validated
   element-wise vs MLX-op cores (rel < 1e-4) and vs the JAX oracle (rel < 1e-3) for iso/diagonal/
   periodic + CPML (`tests/validation/test_mlx_kernel.py`). M4 Pro, N=192 (`profile_engine.py --kernel`):

   | | Mcs/s | RT/step |
   |---|--:|--:|
   | compiled MLX-op cores, CPML on | 285 | 35 |
   | **Metal kernel, CPML on** | **374** | **27** |
   | **Metal kernel, CPML off** | **2219** | **5** |

   *Two findings:* (a) the kernel cores **must be `mx.compile`d** (the metal kernel composes as a graph
   node) — eager, the slab-correction ops dispatch one-by-one and the CPML-on step is slower than the
   op path; (b) the CPML-on gap (27 vs 5 RT) is the slab `_slab_add` rebuilding full arrays via
   `concatenate` (~22 RT) — folding CPML into the kernel for the slab cells (the in-kernel slab-tile
   approach §6 anticipated) is deferred to M3.
3. **M3 — ✅ DONE.** Three independent wins, all in [`kernels.py`](../src/fdtdx/mlx/kernels.py),
   parity-gated (`test_mlx_kernel.py`) + benchmarked, then `FDTDMEX_METAL_KERNEL` flipped default-on:
   - **Fold CPML into the kernel** (the §6 in-kernel slab approach): `_corr_blocks` runs the ψ
     recurrence + κ-stretch/ψ correction over slab threads; the compact slab ψ + per-axis `a/b/1κ`
     are passed as extra in/out buffers. Replaced the M2 `_slab_correction` `concatenate` rebuild.
     **CPML-on 374 → 1826 Mcs/s, 27 → 5 RT (4.9×, at the bulk floor)** iso / 1711 diag (N=192).
   - **Non-uniform metric in-kernel** (`_metric_lines`/`_metric_side`): each difference scaled by its
     per-axis `m{k} = reference_spacing/cell_width` buffer; uniform axes emit nothing.
   - **Heterogeneous full-tensor via the §5 *block* hybrid** (not the per-cell branch): the kernel
     does the diagonal bulk (`cb = c·diag(inv_eps)`); `_offdiag_box` finds the inclusion bbox and
     `_box_correct` runs the validated MLX-op `_update_E`/`_update_H` aniso over a haloed interior
     slice, `_set_box` splicing it back. Gated lossless + uniform + compact (<½ domain) + PML-disjoint
     interior; else the whole domain falls back to MLX-op aniso. N=128 8³ inclusion **125 → 1124
     Mcs/s (9.0×)**. Box cells are bit-identical to the whole-domain ops path (same local stencil,
     real neighbours via the 2-cell halo); the diagonal bulk matches to float32.

## 10. Validation & integration

- Element-wise parity vs the forced-JAX oracle (`tests/validation/`), rel < 1e-3; add a dedicated
  kernel parity test. Marginal failure → raise resolution, never loosen tolerance.
- Gate behind a flag; fall back to the MLX-op path until each case is parity-clean (a hybrid kernel +
  MLX-op engine is acceptable).
- Benchmark with `profile_engine.py` (RT/step) and `bench_forward.py` (scaling) against the Phase-1
  numbers.

## 11. Open questions

- **Resolved (M2): `mx.compile` composes with `mx.fast.metal_kernel`** — the kernel is a normal graph
  node, so wrapping the whole core (kernel + slab-CPML ops) in `mx.compile` works and is **required**:
  eager, the slab ops dominate and CPML-on is slower than the op path. The per-step `mx.eval` cadence
  (`eval_every`) is unchanged from the MLX-op loop.
- Threadgroup-memory budget on `applegpu_g16s` and the best tile shape — only relevant if/when the
  z-march tiled kernel is built (M3+, measurement-gated: thread-per-cell already hits the floor at
  N ≤ 256; revisit if `profile_engine.py` shows RT climbing at large N).
- M2 passes the slab ψ as a captured 6-tuple to the MLX-op slab correction (not into the kernel); the
  open question is how to pass ψ + per-axis metric into the kernel itself when CPML/metric are folded
  in (M3).
