# ring_resonator_agentic — microring transmission via the agentic `fdtdmex.io` contract

The **agent-facing template** for a silicon microring resonator study. It reproduces the core of
`ring_mrm_oband` — bus **TE0 mode** → cold **FDTD run** → through-port **transmission `T(λ)`** and
loaded **Q** — but runs the simulation the way an LLM agent / ag-fdtd notebook must: through
`pack` → `run_simulation_from_hdf5` (detached, non-blocking) → read `result.hdf5`. Copy this when
asked to "set up and run a ring resonator."

```
ring_resonator_agentic/
├── ring_resonator_agentic.py   # percent-format; run as a script or step the # %% cells
└── README.md
```

## The hard rule (why this differs from most examples)

Most `examples/` scripts call `fdtdx.run_fdtd` / `apply_params` for fast **in-process** iteration.
In the agentic workflow that is **forbidden** — it blocks the kernel and bypasses the detached job
system. Here, `fdtdx` only **builds the scene** (geometry, materials, boundaries, source, detectors,
`Scene`) and **solves modes** (`fdtdx.compute_mode`); the simulation **job** is always:

```python
from fdtdmex.io import pack, run_simulation_from_hdf5
bundle = pack(scene, ".")                                         # → one self-contained config HDF5
job    = run_simulation_from_hdf5(bundle, "jobs", simulation_name="ring-cold", backend="mlx")
# returns immediately; the solver runs detached and writes jobs/ring-cold/outputs/result.hdf5
```

When you copy another example, **reuse its geometry but rewrite the run section** as the two lines above.

## What it does

1. **Bus TE0 mode** — `fdtdx.compute_mode` on the strip cross-section → `n_eff`, `n_g`, `|E|²` profile.
   (A small linear solve; fine to call in-process — it is not a simulation run.)
2. **Cold run** — a `gdstk` racetrack ring (outer Si + inner oxide carve) side-coupled to a bus, with
   input/through `PhasorDetector`s → `pack` → detached `run_simulation_from_hdf5`.
3. **Transmission** — `T(λ) = P_thru / P_in` (net Poynting flux) and the loaded **Q** from the
   resonance notch, read from `result.hdf5`.

## Reading results — `sim_postproc` vs the full fields

`sim_postproc(job.results_path)` returns small per-detector **scalars** (shape, `max_abs`, `mean_abs`)
— enough for magnitudes, but it drops the field arrays. A **spectrum / transmission / Poynting flux**
needs the full complex fields, so read them directly:

```python
import h5py, numpy as np
with h5py.File(job.results_path, "r") as f:
    phasor = np.asarray(f["detector_states"]["thru"]["phasor"])   # full complex (1, n_freq, 6, *plane)
```

## Run it

```bash
python ring_resonator_agentic.py            # real Metal solve (resonance notch)
RING_BACKEND=mock python ring_resonator_agentic.py   # fast GPU-free pipeline check (synthetic fields)
```

Coarse `RES` / short `SETTLE` keep the real solve feasible; raise `SETTLE` for sharper resonances.
