# examples/

Runnable examples. Each is a **self-contained folder** — script(s) plus any generated artifacts (figures) live together; documentation links the folder.

- **[`quickstart_notebook/`](quickstart_notebook/)** — start here. A `# %%`-cell notebook (VS Code interactive window or Jupyter) walking the full flow with **inline** figures: define → `plot_setup` → `plot_material` → run → `plot_field_slice` → solve a mode (`compute_mode` + `plot_mode`) → `SMatrixResult`
  + `plot_smatrix`.
- [`simulate_gaussian_source/`](simulate_gaussian_source/) — Gaussian source, forward + backward (adjoint) run, energy-detector video; isotropic, birefringent, and full-tensor variants.
- [`dispersive_gaussian_pulse/`](dispersive_gaussian_pulse/) — Drude–Lorentz dispersion.
- [`bloch_band_structure/`](bloch_band_structure/) — periodic / Bloch (zero-k) band structure.
- [`optimize_ceviche_corner/`](optimize_ceviche_corner/) — inverse design (JAX gradient path).
- [`width_sweep_analysis/`](width_sweep_analysis/) — parameter sweep + analysis.
- [`ring_resonator_demo/`](ring_resonator_demo/) — silicon ring resonator walkthrough (gdstk → geometry → `plot_setup_3d` → mode → mode-expansion → S-parameters → HDF5 hand-off). `make_showcase_images.py` regenerates the README showcase figures into `ring_resonator_demo/figures/`.
- **[`ring_mrm_oband/`](ring_mrm_oband/)** — self-contained design-verification of an O-band (1310 nm) carrier-depletion **microring modulator**, forward-simulated on **Metal**: racetrack + bus → 2-D mode (`n_eff`, `n_g`, `Γ`) → **mesh convergence 40→20 nm** → cold `T(λ)` (Q, ER) + resonant `|E|²` field maps → bus–ring **gap sweep** (coupling control) → static **Soref–Bennett** EO `Δλ_res(V)`. See its [README](ring_mrm_oband/README.md). `MRM_FAST=1` for a quick coarse smoke.
