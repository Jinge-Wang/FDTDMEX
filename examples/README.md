# examples/

Runnable examples.

- **[`quickstart_notebook.py`](quickstart_notebook.py)** — start here. A `# %%`-cell notebook (open in
  the VS Code interactive window or Jupyter) that walks the full flow with **inline** figures: define →
  `plot_setup` → `plot_material` → run → `plot_field_slice` → solve a mode (`compute_mode` + `plot_mode`)
  → `SMatrixResult` + `plot_smatrix`.
- [`simulate_gaussian_source.py`](simulate_gaussian_source.py) — Gaussian source, forward + backward
  (adjoint) run, energy-detector video.
- [`simulate_gaussian_source_anisotropic.py`](simulate_gaussian_source_anisotropic.py) /
  [`..._fully_anisotropic.py`](simulate_gaussian_source_fully_anisotropic.py) — birefringent / full-tensor media.
- [`dispersive_gaussian_pulse.py`](dispersive_gaussian_pulse.py) — Drude–Lorentz dispersion.
- [`bloch_band_structure.py`](bloch_band_structure.py) — periodic / Bloch (zero-k) band structure.
- [`optimize_ceviche_corner.py`](optimize_ceviche_corner.py) — inverse design (JAX gradient path).
- [`width_sweep_analysis.py`](width_sweep_analysis.py) — parameter sweep + analysis.
