# Analysis utilities — resonance finder and sweep runner

Two post-processing tools that sit beside the engine rather than inside it: `fdtdx.utils.resonance`
turns a recorded time trace or a measured spectrum into resonance numbers, and `fdtdx.utils.sweep`
runs one simulation per parameter point and hands back a table. Neither touches JAX or MLX — the
resonance module is pure numpy/scipy, and the sweep module only calls `run_fdtd`.

```python
import fdtdx

modes = fdtdx.find_resonances(signal, dt, f_min, f_max)     # ring-down -> f, Q, amplitude, phase
fit = fdtdx.fit_lorentzian(wavelengths, transmission)       # spectrum  -> f0, FWHM, Q
q = fdtdx.q_from_ringdown(signal, dt, f0)                   # cross-check
table = fdtdx.run_sweep(build, {"gap": gaps}, evaluate)     # sweep     -> rows, CSV, figure
```

## Resonance finder — `find_resonances`

### Method

`find_resonances(signal, dt, f_min, f_max, *, n_basis=None, error_threshold=1e-3,
amplitude_threshold=1e-3) -> list[Resonance]` solves the **harmonic-inversion** problem: given a
finite record that is a sum of decaying sinusoids, recover their frequencies, decay rates,
amplitudes and phases. It uses **filter diagonalisation** (the filter-diagonalisation method, FDM),
after Wall and Neuhauser, *J. Chem. Phys.* **102**, 8011 (1995) and Mandelshtam and Taylor,
*J. Chem. Phys.* **107**, 6756 (1997).

The record is read as the autocorrelation of a fictitious dynamical system, which turns "find the
complex frequencies" into "find the eigenvalues of that system's time-evolution operator". Band
limiting is what makes it cheap: restricted to `[f_min, f_max]`, the matrix elements of that
operator are plain z-transforms of the record, so the work is one pass over the samples plus a
generalised eigenproblem of size `n_basis` — typically a few tens, never the record length.

```text
record c[n]                 band [f_min, f_max]
     |                              |
     +--> z-transforms G, H, D  <---+          O(n_samples * n_basis)
                |
                v
     U(0), U(1), U(2)   (complex-symmetric, n_basis x n_basis)
                |
                v
     U(1) b = u U(0) b        solved on the well-conditioned subspace of U(0)
                |
                +--> u  -> frequency, decay rate, Q
                +--> b  -> error = norm(b' U(2) b - u^2) / norm(u^2)
                +--> least squares over the record -> amplitude, phase
```

The pay-off over a discrete Fourier transform of the same record is resolution: the DFT cannot
separate lines closer than `1/(n_samples*dt)`, while FDM fits a model and is limited by noise
instead. A few tens of optical cycles of ring-down are usually enough.

**Provenance.** MEEP's `harminv` is the reference implementation and this function follows its
conventions and reports the same quantities, so the two are directly comparable. No harminv code is
used: harminv is GPL and this repository references Meep-family algorithms without copying code
(see [licensing.md](licensing.md)). The closed form of the operator matrices, the regularised
eigensolve and the least-squares amplitude step are written from the papers.

### Conventions

A mode is

```text
s(t) = amplitude * exp[-i (2*pi*frequency*t - phase) - 2*pi*decay_rate*t]
```

| Quantity | Convention |
|---|---|
| `frequency` | Ordinary frequency in Hz, not angular. `exp(-i omega t)` (physics / harminv) sign, so a decaying mode sits at `f - i*decay_rate` in the complex plane. |
| `decay_rate` | In Hz, the **same units as the frequency**. The field envelope falls as `exp(-2*pi*decay_rate*t)`, the energy as `exp(-4*pi*decay_rate*t)`. **Positive means decaying**; a negative value is a growing mode, which in a passive run means a spurious fit or a record that is still being driven. |
| `q` | `frequency / (2 * abs(decay_rate))` — the usual `omega_0 * energy / power lost`. |
| `amplitude`, `phase` | From a least-squares fit of the found modes to the whole record, not from the eigenvectors. A **real** record carries every mode as a `+f`/`-f` pair, so a real `A*cos(2*pi*f*t)` is reported with `amplitude = A/2` (as harminv reports it). |
| `error` | harminv's figure of merit for the complex frequency: the `U(2)` consistency residual of the mode's eigenvector shown above, around 1e-15 for a noiseless mode. **Not an error bar** — a small error means the mode is self-consistent, not that it is accurate to that fraction. |

The Q here and the `q` from `fit_lorentzian` are the **same number**: a mode whose envelope decays
as `exp(-2*pi*decay_rate*t)` has a power spectrum whose FWHM is `2*decay_rate`, so
`f/(2*decay_rate) = f0/FWHM`.

### Basis size and thresholds

| Knob | Default | What it does |
|---|---|---|
| `n_basis` | `round(1.1 * (f_max - f_min) * dt * n_samples)`, clamped to `[2, 300]` | harminv's spectral-density-1.1 rule. An **upper bound on how many modes can be found**, not the resolution of the ones that are. Raising it far above the rule makes the matrices large and singular; lowering it risks missing modes. |
| `error_threshold` | `1e-3` | Drops modes whose `error` exceeds it. Much stricter than harminv's `0.1`, which suits clean simulation ring-downs; raise it towards `0.1` for short or noisy records. |
| `amplitude_threshold` | `1e-3` | **Relative**: drops modes below this fraction of the largest amplitude found, so it is independent of the field units. |

The band is a *preference*, not a hard window. Like harminv, the search is biased to
`[f_min, f_max]` but can return a mode outside it — most often the negative-frequency partner of a
real record's strongest line. Filter the returned list on `frequency`, or take the largest
`amplitude`, when a hard window matters.

### Limits

- The record must **be** a small number of decaying sinusoids plus a little noise inside the band.
  A still-driven trace, a continuum, or a band straddling zero produce meaningless modes. Feed it
  the ring-down: switch the detector on after the source has died away
  (`fdtdx.OnOffSwitch(start_time=...)`).
- Keep the band narrow enough that only a handful of modes live inside it.
- `dt` is the **recording** spacing: `config.time_step_duration` when the detector records every
  step, times its stride when it does not.
- Cost is `O(n_samples * n_basis) + O(n_basis**3)`; with the 300 cap, minutes are not a risk.

### Worked check

`tests/simulation/physics/test_resonance_finder.py` runs a 1-D Fabry-Perot slab (eps = 12, 40 cells
of 25 nm, 3x3 periodic transverse cells, PML in z) with a short Gaussian pulse and reads the m = 4
longitudinal mode off the ring-down, in about five seconds on CPU. The measured line sits 0.39 %
below the analytic `f_m = m c / (2 n L)`, which is the grid's own numerical dispersion: for axial
propagation the numerical wavenumber exceeds the exact one by `(k d)^2 (1 - S^2)/24`, and with
20 cells per wavelength inside the slab and the medium Courant number `S = c dt/(n d)` that
predicts 0.40 %. Q comes out 2 % from the analytic pole Q `m pi / (2 ln(1/r))`, and
`q_from_ringdown` agrees with the finder to 0.1 %.

## Lorentzian line fit — `fit_lorentzian`

`fit_lorentzian(frequencies, spectrum, f_guess=None) -> LorentzianFit` is the frequency-domain
counterpart: it reads a line off a measured spectrum instead of a time record. The model is

```text
S(x) = baseline + depth_or_height / (1 + ((x - f0)/(fwhm/2))**2)
```

with `depth_or_height` negative for a transmission dip and positive for a peak. The x axis is
whatever is passed in — frequency in Hz or wavelength in metres — and `f0`/`fwhm` come back in those
units, so `q = f0/fwhm` is the loaded Q either way.

The initial guess is taken from the data (the strongest departure from the median, and the
half-depth crossings on either side of it), and `f_guess` selects which line is fitted when the
window holds several. Both axes are rescaled to order 1 before `scipy.optimize.curve_fit` runs:
without that, a 1e-6 m axis with a 1e-9 m width already satisfies the optimiser's relative step
tolerance at the starting point and the fit never moves. `rmse` is the residual — check it, because
a single Lorentzian cannot describe two overlapping lines and will quietly absorb the neighbour's
tail into the baseline and the width.

## Ring-down Q — `q_from_ringdown`

`q_from_ringdown(signal, dt, f0)` demodulates the record at `f0`, smooths over one period, and fits
a straight line to the log envelope from its peak down to 1e-3 of the peak. It is a one-mode
estimate and much less accurate than filter diagonalisation; it is there to catch gross errors — a
wrong band, a mode that was never excited, a record that is still being driven — not to replace the
fit. A large disagreement with `find_resonances` usually means the band holds more than one mode.

## Sweep runner — `run_sweep`

### Contract

```python
def build(**point) -> tuple[ArrayContainer, ObjectContainer, SimulationConfig]: ...
def evaluate(result, **point) -> dict[str, float]: ...

result = fdtdx.run_sweep(
    build,
    {"gap": [100e-9, 200e-9], "width": [450e-9, 500e-9]},   # Cartesian product
    evaluate,
    cache_dir="sweeps/gap_width",
    tag="v1",
)
```

`build` owns the whole set-up and is the place to call `place_objects` and `apply_params`. Each
point is then run through `run_fdtd`, and `evaluate` receives that call's `SimulationState` — the
pair `(time_step, arrays)`, so `result[1].detector_states` is where the numbers come from — plus the
point's parameters as keywords. `params` is either a mapping of name to sequence (swept as the full
product, last name varying fastest) or an explicit list of `{name: value}` dicts when the points are
not a grid. `backend="jax"` or `"mlx"` wraps every run in `fdtdx.use_backend`.

`SweepResult` is a tidy table: `rows` (one flat dict per point, parameters then metrics),
`to_numpy(columns=None)`, `to_csv(path)`, and `plot(x, y)` returning a matplotlib figure.
`from_cache` and `n_runs` say how much of the sweep actually ran.

### Cache

With a `cache_dir`, each point's metric dict is written as JSON under a sha256 of its **sorted
parameter values plus the `tag`**. On a later call a point whose file exists is neither built nor
run, so an interrupted sweep resumes where it stopped and re-plotting costs nothing.

The key is the parameters, **not** the contents of `build`/`evaluate`. Change the physics of the
set-up without changing a parameter value and you must bump `tag` (or clear the directory), or you
will read back the old numbers. Metric values must be JSON-serialisable to be cached; a point whose
metrics will not serialise simply stays uncached.

### Limits

- **Runs are serial by design.** `max_workers` threads the `evaluate` step only. Concurrent JAX (or
  MLX) executions on one device are not thread-safe and would contend for the same memory anyway.
  For real parallelism run several processes over disjoint slices of the parameter space, sharing
  one `cache_dir`.
- Every point's simulation is built from scratch; there is no warm start and no shared compilation
  across points beyond what JAX's own cache provides.
- `run_sweep` does not checkpoint field arrays — only the reduced metric dict survives a point. Put
  anything else worth keeping (figures, raw traces) in `evaluate`.

## Note on `examples/ring_mrm_oband`

The microring example keeps its own `resonance_metrics`, which locates the through-port dip by
discrete half-maximum crossings rather than by a fit. Swapping it for `fit_lorentzian` would change
the published numbers (resonance wavelength, FWHM, loaded Q, extinction ratio) and every figure that
carries them: the crossing estimate is quantised to the spectrum's wavelength grid, and the all-pass
ring line shape is not exactly Lorentzian, so a fit lands somewhere else. The example is therefore
left as it is; new analyses should use `fit_lorentzian`.
