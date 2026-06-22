# Phase 2 scoping — custom Metal update kernels

> Status: scoping (not started). Phase 1 is complete (default path 277 Mcs/s / 36 RT, CPML on;
> MLX-op ceiling 473 Mcs/s / 21 RT — see [metal-bottleneck-analysis.md](metal-bottleneck-analysis.md)).
> This document scopes the next step: replacing the per-field MLX-op chain with one hand-written
> Metal kernel per field. fp32 only.

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

## 5. Region specialization (heterogeneous materials)

The current engine selects one global path: if any cell is full-tensor, the whole domain runs the
anisotropic update (with its neighbour-averaging). For heterogeneous domains, tag cells by material
class and dispatch the cheap isotropic kernel on isotropic sub-blocks and the per-cell 3×3 A/B kernel
only on anisotropic sub-blocks (per-block branch, or separate launches over index sets). This removes
the anisotropic neighbour-averaging and double-buffering wherever the cell is isotropic.

## 6. CPML

ψ and the κ-stretch correction are confined to the boundary slabs (Fix 1.2). In the kernel, the ψ
recurrence + correction run only on boundary-slab tiles; interior tiles compute the plain curl. The
per-axis slab extents (`pml.detect_pml_slabs`) and the slab ψ layout carry over directly.

## 7. Staging

1. **Isotropic, uniform, interior (no CPML)** E-kernel and H-kernel — the common bulk. Validate
   element-wise against the MLX-op path on an interior-only region; measure RT and Mcs/s.
2. **+ CPML** on boundary-slab tiles; full-domain parity.
3. **+ diagonal / full-anisotropic** via region specialization; **+ non-uniform metric** (per-axis
   scale arrays passed in). Go/no-go on extending vs keeping the MLX-op fallback for these.

## 8. Validation & integration

- Element-wise parity vs the forced-JAX oracle (`tests/validation/`), rel < 1e-3; add a dedicated
  kernel parity test. Marginal failure → raise resolution, never loosen tolerance.
- Gate behind a flag; fall back to the MLX-op path until each case is parity-clean (a hybrid kernel +
  MLX-op engine is acceptable).
- Benchmark with `profile_engine.py` (RT/step) and `bench_forward.py` (scaling) against the Phase-1
  numbers.

## 9. Open questions

- Threadgroup-memory budget on `applegpu_g16s` and the best tile shape (measure).
- Cleanest way to pass the slab ψ (6 per-component arrays of differing shape) and per-axis metric
  arrays into the kernel signature.
- Interaction with `mx.compile` / `mx.eval` cadence when the step is one (or two) custom-kernel nodes.
- Whether the isotropic interior kernel writing E in place (no double-buffer) composes safely with the
  lazy graph, or whether to double-buffer uniformly for simplicity first, then optimize.
