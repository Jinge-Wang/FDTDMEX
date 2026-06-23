# Side quest — O-band microring-modulator validation

**Start a fresh session here.** Self-contained task: produce a clean, physically convincing
design-verification of an **O-band (1310 nm) carrier-depletion microring modulator (MRM)**, forward-run on
the **Metal** engine. The example already exists and is structured —
[`examples/ring_mrm_oband/`](../examples/ring_mrm_oband/) (script + `figures/` + README). This is a
**documentation + validation** task: do **not** modify engine code. When done, resume
[`ACTION_PLAN.md`](../ACTION_PLAN.md).

## Scope (bounded — do exactly this, nothing more)

The script [`examples/ring_mrm_oband/ring_mrm_oband.py`](../examples/ring_mrm_oband/ring_mrm_oband.py)
already implements all five steps. The job is to **run it at production resolution, sanity-check the
figures, and update numbers** — not to add features.

1. **Model + mode** — racetrack ring + bus; bus TE₀ mode → `n_eff`, `n_g`, `Γ`; geometry reference
   resonance `λ_ref = n_eff·L/m`.
2. **Mesh convergence 40 → 20 nm** — cold `T(λ)` at each grid; resonance λ and loaded Q vs grid.
3. **Cold run (20 nm)** — through-port `T(λ)`, resonance (dip nearest `λ_ref`), `|E|²` field maps on/off
   resonance.
4. **Coupling** — through-port `T(λ)` vs bus–ring gap and ER vs gap.
5. **Static EO** — Soref–Bennett free-carrier perturbation → resonance red-shift vs reverse bias.

## Validated recipe (already implemented — do not rediscover)

- **Stay on Metal.** Mode sources/detectors force the slow JAX/CPU path (JAX here is **CPU-only** — no
  jax-metal). So excite with a broadband **`GaussianPlaneSource`** (TE: E along width y) + **`PhasorDetector`**
  monitors; both are MLX-eligible with non-dispersive Si/oxide.
- **Transmission = two-run net Poynting flux**: `T(λ) = P_thru^ring / P_thru^bus-only`, with the per-frequency
  net flux `½·Re ∮(E×H*)·n̂` from the recorded phasors. Net power (not `|mode-overlap|²`) avoids the
  standing-wave `T>1` artifact; the bus-only reference cancels the Gaussian launch's radiative loss
  (baseline → ~1). Settle **~3–3.5 ps** (high-Q needs a long ring-down).
- **Geometry:** FDTD device is a full-etch **strip** ring (clean, affordable); the rib SOI stack is implicit
  in the mode/EO analysis. Inner ring carve material is **oxide** (background is oxide). Bus–ring `gap` is in
  **metres** but `R/WG` are in **µm** → `CY = WG + gap*1e6 + R`. Source/monitor box (W=1.2, H=0.5 µm) sits
  strictly inside the interior; bus at YBUS=0.8 µm, LZ=0.8 µm (else it pokes into the PML). A PML grid-tiling
  retry grows the volume by a cell (x **and** y) until `place_objects` resolves.
- **Resonance fit:** exclude band edges (low pulse power → spurious half-dips) and fit the dip **nearest
  `λ_ref`** so the same resonance is tracked across grids/gaps; baseline = capped max over the central band.
- **EO:** reverse bias removes carriers → index **up** → **red** shift. `Δn_eff = 0.5·Δn_bulk(ND,NA)·
  [Γ(W(V)/2) − Γ(W0/2)]` (0.5 = abrupt-junction symmetry). `Γ(half_w)` interpolates the **cumulative** modal
  energy (smooth — a hard cell mask at the 10 nm mode grid makes a staircase). O-band Soref coefficients.

## Resolution & runtime (the real constraint)

- Starting mesh `λ/(n_eff·15) ≈ 32 nm`; production **20 nm**. Convergence spans 40 → 20 nm.
- The MLX time loop is **eager**, so wall time ∝ cells × steps: **40 nm ≈ 5 min/run, 20 nm ≈ 1–1.5 h/run**.
  The full suite (convergence + cold + gap sweep + EO) is **several hours** — run deliberately (overnight /
  background). To trim: coarsen `GAP_RES` (default 25 nm), drop a convergence point, or shrink the band.
- Smoke first: `MRM_FAST=1 …` (~5 min, physics meaningless) confirms the code path before the long run.

## Run

```bash
cd FDTDMEX
MRM_FAST=1 uv run --extra viz python examples/ring_mrm_oband/ring_mrm_oband.py   # quick smoke
uv run --extra viz python examples/ring_mrm_oband/ring_mrm_oband.py             # production (hours)
uv run --with pytest pytest tests/validation -q                                 # no regressions (49 pass)
```

Generated artifacts land in [`examples/ring_mrm_oband/figures/`](../examples/ring_mrm_oband/figures/)
(`setup.png`, `mode.png`, `convergence.png`, `cold_spectrum.png`, `field_maps.png`, `gap_sweep.png`,
`eo_response.png`, `operating_point.npz`).

## Acceptance

- Convergence: resonance λ and Q **settle** toward 20 nm (positions stop moving ≫ at coarse grids). If they
  still jump at 20 nm, the device isn't converged — note it; don't fake a trend.
- Cold `T(λ)`: clean dip near `λ_ref`, baseline ~1; Q and ER quoted from the 20 nm run.
- Field maps: energy clearly **inside the ring** on resonance, **passing through** off resonance.
- Gap sweep: ER varies with gap (under-/critical-/over-coupling); operating gap = max ER.
- EO: monotonic **red** shift, plausible magnitude (tens of pm/V), explicitly optical-only.
- README / examples/README / this doc reference [`examples/ring_mrm_oband/`](../examples/ring_mrm_oband/);
  `tests/validation` still pass.

## History / pitfalls (so this isn't re-litigated)

Earlier 40 nm runs were **not converged** (resonance moved ~½ FSR between 40/48/60 nm) — hence the 40→20
convergence. Bugs already fixed in the script: mode unit bug (slab vs confined strip; `WG*1e-6`), x–y vs
**y–z** cross-section labeling, monitors poking into the PML, **air-vs-oxide** ring interior, resonance
metric grabbing band edges, **gap a no-op** (µm/m mix), placement retry missing the y-axis, EO sign (red
shift) + magnitude (0.5 junction factor) + staircase (cumulative-Γ interpolation). The convention going
forward: **each example is a self-contained folder** (script + `figures/` + README), docs link the folder.
