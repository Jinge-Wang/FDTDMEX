# ring_mrm_oband — O-band carrier-depletion microring modulator

Self-contained design-verification of an **O-band (1310 nm) silicon microring modulator (MRM)**,
forward-simulated on the **Metal** engine (MLX backend). Everything for this example — script and
artifacts — lives in this folder.

```
ring_mrm_oband/
├── ring_mrm_oband.py    # the study (percent-format; run as a script or open as a notebook)
├── figures/             # generated figures + operating_point.npz
└── README.md
```

## What it does

1. **Model + mode** — racetrack ring + bus authored in gdstk; bus **TE₀** mode → `n_eff`, `n_g`, `Γ`, and
   the geometry's reference resonance `λ_ref = n_eff·L/m`.
2. **Mesh convergence** — cold spectrum from **40 nm down to 20 nm**; resonance λ and loaded Q vs grid.
3. **Cold run** — through-port `T(λ)`, the resonance (dip nearest `λ_ref`), and `|E|²` field maps on / off
   resonance (light trapped in the ring vs passing through).
4. **Coupling** — through-port `T(λ)` vs bus–ring gap and the ER-vs-gap trend (under/critical/over-coupling).
5. **Static EO** — Soref–Bennett free-carrier perturbation → resonance red-shift vs reverse bias.

## Method (why)

Mode sources/detectors would force the slow JAX/CPU path here, so the cold run uses a broadband
**Gaussian** source + **phasor monitors** and reports the standing-wave-immune **net Poynting flux** with
a bus-only reference, `T(λ) = P_thru^ring / P_thru^bus`. The FDTD device is a full-etch **strip** (clean,
affordable); the rib SOI stack is implicit in the mode/EO analysis where the lateral PN junction lives.

## Resolution & runtime

Starting mesh guideline: `λ/(n_eff·15) ≈ 1310/(2.69·15) ≈ 32 nm`; production sign-off at **20 nm**. The
MLX time loop is eager, so wall time scales with cells × steps: a 40 nm run is ~5 min, a **20 nm run is
~1–1.5 h**. The full suite (convergence 40→20 + cold + gap sweep + EO) is **several hours** — run it
deliberately (e.g. overnight), or coarsen `GAP_RES` / drop a convergence point to trade accuracy for time.

## Run

```bash
# quick coarse smoke (physics meaningless, exercises the whole code path in ~5 min):
MRM_FAST=1 uv run --extra viz python examples/ring_mrm_oband/ring_mrm_oband.py

# production (writes figures/ + operating_point.npz; several hours at 20 nm):
uv run --extra viz python examples/ring_mrm_oband/ring_mrm_oband.py
```

Tunable knobs at the top of the script: `CONV_RES`, `PROD_RES`, `GAP_RES`, `GAPS`, `BAND`, `SETTLE`,
device geometry (`R`, `WG`, `LC`).
