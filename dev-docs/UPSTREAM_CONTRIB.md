# Upstream contributions — what flows back to fdtdx

Companion to [UPSTREAM_SYNC.md](UPSTREAM_SYNC.md). This is the plan for the parts of FDTDMEX's work
that **improve upstream fdtdx without compromising its mission**.

## The filter

FDTDMEX's MLX route exists for **forward speed on Apple Silicon**; it deliberately **gives up
autodiff**. Upstream fdtdx is a **differentiable** JAX solver for inverse design. So the filter for
"does this go upstream?" is:

> **A contribution must be (a) implementable in pure JAX, (b) differentiable or autodiff-neutral
> (not in the gradient path), and (c) reduce to identical behavior on the existing path at its
> trivial setting** — a strict improvement, never a regression.

That **excludes** the MLX backend, the Metal kernels, the dispatch/routing, `mx.compile`/eager
execution, temporal blocking, per-tile material compaction, the bridge, and the HDF5/MCP workspace.
Those are forward-only and Apple-specific; adding a non-differentiable eager path would dilute
upstream's autodiff core. What's left is genuinely portable, and it's substantial.

---

## Track 1 — Monitor / detector performance (the big one)

Upstream's `fdtd/update.py:update_detector_states` pads and runs `interpolate_fields` over the
**entire domain on every step** — unconditionally, *before* the per-detector `lax.cond` gate. For a
typical photonic run (small flux/phasor monitors in a large domain) that is the dominant non-kernel
cost. FDTDMEX removed it; on the O-band ring example the full `run_fdtd` dropped from **~1478 s →
377 s** purely from monitor handling, with no kernel change. All three pieces below are pure
slicing/arithmetic — **differentiable, scan/checkpoint-safe, and identical at the trivial setting.**

### 1a. Region-restricted interpolation ⭐ (highest value, lowest risk)
- **What:** instead of interpolating the whole domain, interpolate only each detector's `grid_slice`
  + the 1-cell stencil halo. Reference: `mlx/interpolate.py:interpolate_region_mlx` /
  `_region_padded_block` (synthesizes the boundary ghost cell — zero on PML/PEC, wrapped neighbor on
  periodic — without padding the whole field). Element-wise identical to slicing the full-domain
  result (the H time-average commutes with the windowed pad).
- **Port:** rewrite `update_detector_states` to crop per detector before `interpolate_fields`. Pure
  `jnp` slicing + concatenate → differentiable, works under `lax.scan`/checkpointing.
- **Win:** scales with (domain / monitor) volume ratio — typically 10–1000× less interpolation work
  on the exact-evaluation path. Mostly benefits the final high-accuracy eval (inverse-design
  optimization defaults `exact_interpolation=False`), but it's a clean, safe win there.
- **Effort:** medium. **Risk:** low (drop-in, identical results).

### 1b. Activity-gating the interpolation
- **What:** upstream computes `interpolated_E/H` even on steps where **no** detector records. Gate
  the whole region-interpolation behind "any detector on at this step" with `jax.lax.cond` (one
  branch executes → real compute skip under scan).
- **Port:** trivial once 1a lands. **Win:** small but free. **Risk:** low.

### 1c. Nyquist-aware DFT subsampling for frequency/phasor monitors ⭐ (novel capability)
- **What:** the running DFT in a `PhasorDetector` samples every step, but the FDTD `dt` sits ~10–20×
  below the Nyquist rate of the highest recorded frequency. Sampling every `stride` steps (stride
  from a conservative Nyquist margin — keep ~12 samples/period) and compensating the normalization
  by `stride` (the Riemann-sum weight) is accurate to well within fp32. Reference:
  `mlx/detector_freeze.py:_dft_stride` + `_phasor_plan` (the `static_scale *= stride` compensation).
- **Port:** gate the phasor accumulate with `lax.cond(step % stride == 0, accumulate, noop)`; derive
  `stride` from the detector's max frequency (override + `stride=1` exact mode for parity tests).
- **Why safe:** differentiable; reduces to exact every-step accumulation at `stride=1`. Upstream has
  **no** temporal subsampling today, so this is a new capability, not just a refactor — and spectral
  / steady-state sweeps are the common photonics workflow.
- **Effort:** medium. **Risk:** low–medium (needs the conservative Nyquist default + a parity test
  at `stride=1` and at the auto stride).

**Note for inverse design:** none of 1a–1c touch gradient correctness — they're differentiable
recording-path changes. During optimization upstream uses raw fields (`exact_interpolation=False`),
so 1a/1b mostly help final evaluation; 1c helps any spectral objective.

---

## Track 2 — Accuracy: width-weighted off-diagonal anisotropic averaging ⭐

- **What:** upstream `fdtd/misc.py:avg_anisotropic_E/H_component` colocates off-diagonal tensor terms
  with a plain `/4` 4-point mean → **1st-order on graded grids for full-tensor anisotropy** (still so
  at `e5351a4`). FDTDMEX's `mlx/aniso.py` splits it into two separable half-steps and width-weights
  the center→edge step → **2nd-order**. (Upstream's *curl* and *detector interpolation* are already
  weighted/2nd-order; this off-diagonal average is the one remaining gap.)
- **Port:** reimplement the `aniso.py` weighting in JAX `misc.py`; thread `cell_widths` through
  `fdtd/update.py` (the curl-side `_backward_edge_average` already proves upstream accepts the
  pattern). **Note:** this also fixes FDTDMEX's *own* JAX path, which still has the `/4`.
- **Why safe:** pure arithmetic (`jax.grad`-clean); byte-identical to `/4` on uniform / per-axis-
  uniform grids, so existing results and tests are unchanged. The gap only opens for full-tensor
  anisotropy on a *truly graded* `RectilinearGrid`.
- **Effort:** small. **Risk:** very low. **This is the cleanest single PR.** Pair with a graded-mesh
  full-tensor convergence test (slope 1→2).

---

## Track 3 — Tidy3D-free native mode solver

- **What:** `core/physics/mode_backend/` is a **numpy + scipy.sparse** full-vectorial FD eigensolver
  behind the `mode_backend` seam (`modes.py`, default `"fdtdmex"`). Upstream depends on Tidy3D for
  mode solving.
- **Why safe:** mode solving is a setup/analysis step, **not** in the gradient path, so a
  non-differentiable scipy solver is autodiff-neutral. It's already structured as a swappable
  backend — exactly the shape upstream would want as an *optional, dependency-free* alternative
  (Tidy3D stays the default, or offer parity).
- **Port:** lift `mode_backend/` + the `modes.py` seam; bring the analytic slab-dispersion validation
  tests. **Effort:** large — propose as an issue/RFC first. **Risk:** medium (API surface, keeping
  Tidy3D parity).

---

## Track 4 — Kottke subpixel smoothing (caveated)

- **What:** `core/physics/subpixel.py` (numpy) effective-tensor averaging.
- **Caveat:** it's numpy, so **not differentiable** — it improves *forward* accuracy but, as-is,
  can't participate in inverse-design gradients (upstream's core use case). To be a first-class
  upstream feature it wants a JAX reimplementation so smoothing sits inside the gradient path during
  placement.
- **Recommendation:** offer as a forward-accuracy utility / discuss a JAX port with maintainers
  rather than shipping numpy into the differentiable path. Lower priority.

---

## Track 5 — Tests & validation harness (low-effort goodwill)

The element-wise parity methodology and the graded-mesh / full-tensor convergence tests are
framework-agnostic. Contributing the **non-uniform / full-tensor convergence tests** (independent of
MLX) strengthens upstream coverage and substantiates Tracks 1–2.

---

## Suggested sequence

1. **Track 2** (off-diag weighting) — small, self-contained, fixes a real upstream *and* own-JAX
   accuracy gap; pair with the Track 5 convergence test. Warm-up PR that builds trust.
2. **Track 1a + 1b** (region interpolation + activity gate) — high-value, drop-in, identical results.
3. **Track 1c** (DFT subsampling) — high-value new capability; lands after 1a/1b since it composes
   with them.
4. **Track 3** (mode solver) — RFC → PR; high impact (drops a Tidy3D dependency).
5. **Track 4** (subpixel) — discussion, not an immediate PR (differentiability caveat).
</content>
