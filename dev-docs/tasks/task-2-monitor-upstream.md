# Task 2 — Upstream the monitor-recording optimization to fdtdx (JAX, differentiable)

**Type:** clean feature PR to **upstream fdtdx** (`github.com/ymahlau/fdtdx`). **Parallelizable:** yes.
**Base:** a fresh feature branch off **`upstream/main`** — do **not** carry any MLX/fork code into this PR.
**Background:** `docs/performance.md` §"Monitor recording"; reference impl in `src/fdtdx/mlx/`.

## Why this exists

On a *monitored* run, fdtdx's full `run_fdtd` used to sit ~7× below the pure loop — not because of the
field update, but because detectors were fed by **interpolating the whole `(3,N³)` field every step**, then
slicing each monitor's handful of cells out. The fork removed that (MLX path only) for a **3.9× wall-clock
win** on the O-band MRM reference (1478 s → 377 s) with **no physics change**. The techniques are generic,
autodiff-safe, and belong in upstream fdtdx's JAX detector path so every user (and inverse design) benefits.

## The three techniques to port (all in the JAX time loop / detector classes)

1. **Region-restricted interpolation.** The Yee co-location stencil reaches only ±1 cell, so interpolate
   each detector over just its `grid_slice` + a 1-cell halo (with the single ghost row only at a true domain
   edge, matching the zero/wrap pad rule), instead of the full domain. Result is **element-wise identical**
   to slicing the old full-domain interpolation. Fork reference: `src/fdtdx/mlx/interpolate.py`
   `interpolate_region_mlx` (windowed padded sub-block → the unchanged interpolation).
2. **Activity-gating.** Skip the whole record/interpolate block on steps where **no** detector actually
   records (today it interpolates every step whenever any detector exists).
3. **DFT auto-subsampling (phasor detectors).** A phasor's signal is band-limited at `c/λ` and the FDTD `dt`
   is ~10–20× below that Nyquist, so record only every `floor(1/(k·f_max·dt))`-th otherwise-active step
   (oversampling margin `k≈12`), scaling each kept sample's DFT weight by the stride (Riemann-sum weight) to
   match the every-step magnitude. Fork reference: `src/fdtdx/mlx/detector_freeze.py` `_dft_stride` + the
   stride/scale logic. Provide a knob to force exact every-step recording (parity/testing).

## Constraints
- **Differentiable / autodiff-safe** — this path is used for inverse design, so it must survive `jax.grad`
  through `run_fdtd` (region-restriction and gating are just index-restricted reads/conditionals; DFT
  subsampling is a static per-step mask + weight — all trace-friendly; avoid Python-side data-dependent
  control flow inside the scanned loop, use masks/`lax.cond` as fdtdx already does).
- **Exact** for region-restriction + gating (parity-test vs the current full-field path, `rel<1e-3`);
  **exact within the oversampling margin** for subsampling (physics-test: resonance/extinction unchanged).

## Steps
1. On the upstream JAX side, find where detectors are fed interpolated fields each step (the JAX time loop —
   `src/fdtdx/fdtd/forward.py` / the detector `update` call path; **not** the MLX `mlx/accumulate.py`).
2. Implement region-restricted interpolation in the JAX detector update (per-detector `grid_slice` + halo).
3. Add activity-gating around the record block.
4. Add opt-in phasor DFT subsampling with the stride/weight rule + an exact-mode flag.
5. Tests: element-wise parity (exact mode) + a physics test (a small resonator: dip wavelength / extinction
   unchanged vs full recording). Benchmark the monitored-run speedup.
6. Open the PR against `ymahlau/fdtdx` with the `docs/performance.md` rationale distilled.

## Deliverable
A clean upstream PR (feature branch off `upstream/main`) implementing the three techniques in fdtdx's JAX
detectors, differentiable, parity + physics tested, with a before/after monitored-run benchmark.

## Caveat
Upstream is (untimed) exploring a PyTorch rewrite. This JAX PR is still low-regret: it's a real speedup for
current fdtdx and the *algorithm* ports to any framework. Keep the change self-contained and idea-documented.
