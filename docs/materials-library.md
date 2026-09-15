# Dispersion models, the fitter, and the material library

Three layers, each usable on its own: the **pole models** the engine integrates,
a **fitter** that turns measured n/k into those poles, and a **library** of
already-fitted materials with their provenance.

```python
import fdtdx

si   = fdtdx.get_material("Si")                       # dispersive, telecom band
au   = fdtdx.get_material("gold")                     # Drude + Lorentz
glass = fdtdx.get_material("SiO2", wavelength=1.55e-6)  # frozen at one wavelength
fdtdx.list_materials()
```

## Pole models

Linear dispersion is `ε(ω) = ε∞ + Σ_p χ_p(ω)`, each pole a two-step recurrence
`P[n+1] = c1·P[n] + c2·P[n-1] + c3·E[n]` in an auxiliary polarization field
updated alongside E (see [materials-anisotropy.md](materials-anisotropy.md) for
where it sits in the update and how it is folded into the Metal kernel).

| pole | susceptibility (`exp(-iωt)`) | parameters | recurrence |
|---|---|---|---|
| `LorentzPole` | `Δε·ω₀² / (ω₀² − ω² − iγω)` | `resonance_frequency`, `damping`, `delta_epsilon` | 2nd order, needs `ω₀·dt < 2` |
| `DrudePole` | `−ω_p² / (ω² + iγω)` | `plasma_frequency`, `damping` | 2nd order (`ω₀ = 0`) |
| `SellmeierPole` | one data-sheet term `B·λ²/(λ² − C)` | `B`, `C` in **m²** (`from_micrometres` for µm² data sheets) | lossless Lorentz: `ω₀ = 2πc/√C`, `Δε = B`, `γ = 0` |
| `DebyePole` | `Δε / (1 − iωτ)` | `delta_epsilon`, `relaxation_time` | 1st order, unconditionally stable |

A pole type only has to override two hooks — `recurrence_coefficients_axes(dt)`
and `susceptibility_axes(omega)`. Everything downstream (the per-axis, tensor
and per-material coefficient builders, the analytic spectrum, the source
impedance correction, the mode solver, the MLX/Metal fold) is generic in
`(c1, c2, c3)` and needs no change for a new pole kind. Debye poles therefore
work on every path Lorentz and Drude do, including oriented (off-diagonal)
poles.

### Debye discretisation and its one caveat

Integrating `τ·ṗ + p = Δε·E` exactly across one step with `E` held constant
gives

```
c1 = exp(-dt/τ)      c2 = 0      c3 = Δε·(1 - exp(-dt/τ))
```

whose recurrence roots are `{exp(-dt/τ), 0}` — stable for every time step, with
no `ω₀·dt < 2` bound.

The engine supplies `E` at the **left endpoint** of the step, while the exact
update wants the exponentially-weighted mean of `E` over the step, which sits at
the midpoint to leading order. The realised susceptibility therefore leads the
analytic one by half a step:

```
χ_discrete = χ(ω)·(1 + i·ω·dt/2) + O((ω·dt)²)
```

about 4.5 % at 40 cells per wavelength with a Courant-limited step. A
midpoint-consistent coefficient needs `E[n+1]`, i.e. an implicit fold into the
E-update that the shared `c1/c2/c3` recurrence (and its Metal counterpart)
cannot express. The explicit alternatives that do fit the shape carry the same
leading error (forward Euler `c1 = 1 − dt/τ`, trapezoidal
`c1 = (2τ − dt)/(2τ + dt)`) or are unconditionally unstable (the leapfrog-centred
form `c1 = −2dt/τ`, `c2 = 1`). The exact exponential form is kept because it
alone reproduces the physical decay rate at every `dt`. Measured in
`tests/simulation/physics/test_dispersion.py`: a Debye half-space at `ωτ = 1`
reflects within 2.7 % of the analytic Fresnel value, and its absorption
coefficient comes out 6.7 % high — consistent with the `ω·dt/2` bias.

### How the pole order is carried

Only `c1`, `c2` and `c3` are threaded to the engine, so the paths that
reconstruct `χ(ω)` from stored coefficients alone (the broadband source
impedance correction, the mode solver's complex permittivity) read the pole
order off `c2`: a second-order pole has `c2 = −(1 − γ·dt/2)/D`, which is exactly
zero only at the unphysical `γ·dt = 2`, nudged one ulp off zero when it happens.
`c2 == 0` therefore means a first-order pole.

## The fitter

`fdtdx.fit_dispersion` turns measured `n(λ)`, `k(λ)` into a `DispersionModel`.

```python
lam, n, k = fdtdx.read_refractiveindex_yaml("main/SiO2/nk/Malitson.yml")
fit = fdtdx.fit_dispersion(lam, n, k, num_poles=2, kinds=("sellmeier",), dt=2e-17)
print(fit.report)
material = fdtdx.Material(permittivity=fit.eps_inf, dispersion=fit.model)
```

- Bounded least squares (`scipy.optimize.least_squares`, `trf`) on the real and
  imaginary parts of `ε(ω) = (n + ik)²`.
- Strictly positive quantities (resonance and plasma frequencies, oscillator
  strengths, relaxation times) are optimised in `log` space; damping rates
  linearly with a lower bound of exactly zero, so a transparent material's best
  fit really is `γ = 0`.
- Starts come from the data — resonances at the peaks of `Im ε`, a Drude term
  from the low-frequency limit, a Debye term from a relaxation knee — over every
  allowed mix of pole kinds, plus seeded jitter. `seed` makes a fit
  reproducible.
- Every pole strength is bounded non-negative, so the fit is passive by
  construction; `FitResult.passive` re-checks `Im ε ≥ 0` on a dense grid.
- Pass `dt` for the `ω₀·dt < 2` advisory, or call `fdtdx.check_stability(model, dt)`.

`kinds` accepts `"lorentz"`, `"sellmeier"` (lossless Lorentz), `"drude"` and
`"debye"`. Inside a transparency window use `"sellmeier"`: the data has no
absorption to constrain a damping rate, and the extra free parameter only buys
noise.

`fdtdx.read_refractiveindex_yaml` reads the
[refractiveindex.info](https://refractiveindex.info) record types this needs —
`tabulated nk` / `n` / `k` and dispersion formulas 1 (Sellmeier), 2
(Sellmeier-2), 3 (polynomial) and 4.

## The library

`src/fdtdx/data/materials_library.json` is checked in and regenerated by
`scripts/build_material_library.py` from a local database clone:

```bash
python scripts/build_material_library.py --database /path/to/refractiveindex/database
python scripts/build_material_library.py --only Si,SiO2 --dry-run
```

`fdtdx.MATERIALS[name]` is the full record — poles, `eps_inf`, the page and
paper the data came from, the band the fit covers, its residual, and notes.
`fdtdx.get_material(name)` returns the dispersive `Material`;
`fdtdx.get_material(name, wavelength=...)` returns a non-dispersive one frozen
at that wavelength, carrying the absorption as an equivalent conductivity
`σ = ω·ε₀·Im ε` (cheaper for a narrowband run). Names and aliases match
case-insensitively.

| name | range (µm) | poles | rms Δε | source page |
|---|---|---|---|---|
| `Ag` | 0.413–1.39 | Drude + Lorentz | 0.34 | `main/Ag/nk/Johnson.yml` |
| `Al` | 0.401–1.99 | Drude + 2×Lorentz | 0.66 | `main/Al/nk/Rakic-LD.yml` |
| `Al2O3` | 0.404–1.99 | 2×Sellmeier | 7.1e-07 | `main/Al2O3/nk/Malitson-o.yml` |
| `Au` | 0.6–1.6 | Drude + Lorentz | 0.24 | `main/Au/nk/Johnson.yml` |
| `Cu` | 0.6–1.6 | Drude + Lorentz | 0.50 | `main/Cu/nk/Johnson.yml` |
| `GaAs` | 1.02–5.9 | 3×Sellmeier | 2.3e-15 | `main/GaAs/nk/Skauli.yml` |
| `Ge` | 2–14 | 2×Sellmeier | 3.2e-04 | `main/Ge/nk/Burnett.yml` |
| `H2O` | 0.4–1.6 | 2×Sellmeier | 7.0e-04 | `main/H2O/nk/Hale.yml` |
| `H2O_farIR` | 15.5–200 | Debye + 3×Lorentz | 0.053 | `main/H2O/nk/Hale.yml` |
| `InP` | 1.01–3.95 | 2×Sellmeier | 1.7e-15 | `main/InP/nk/Pettit.yml` |
| `LiNbO3_e` | 0.505–3.96 | 3×Sellmeier | 1.3e-06 | `main/LiNbO3/nk/Zelmon-e.yml` |
| `LiNbO3_o` | 0.505–3.96 | 3×Sellmeier | 1.2e-06 | `main/LiNbO3/nk/Zelmon-o.yml` |
| `PMMA` | 0.44–1.04 | Sellmeier | 4.7e-08 | `organic/…(PMMA)/nk/Sultanova.yml` |
| `Si` | 1.36–1.7 | 2×Sellmeier | 7.0e-06 | `main/Si/nk/Salzberg.yml` |
| `Si3N4` | 0.404–1.99 | 2×Sellmeier | 1.9e-06 | `main/Si3N4/nk/Luke.yml` |
| `SiO2` | 0.41–1.97 | 2×Sellmeier | 3.0e-07 | `main/SiO2/nk/Malitson.yml` |
| `Si_visible` | 0.5–1.1 | 3×Lorentz | 0.042 | `main/Si/nk/Green-2008.yml` |
| `TiO2` | 0.454–1.5 | 2×Sellmeier | 9.1e-15 | `main/TiO2/nk/Devore-o.yml` |

`rms Δε` is the root-mean-square of `|ε_fit − ε_data|` over the entry's band.
Entries whose source page is itself a Sellmeier or polynomial formula land at
machine precision because the fit is reproducing an analytic curve, not
scattered measurements; the metals are fitted to tabulated n/k and carry a real
residual.

## Provenance and licensing

Every number comes from the refractiveindex.info database, which waives
copyright via [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). The
pole coefficients are our own fits of that public-domain data, produced by
`fdtdx.dispersion_fit`. No coefficients are taken from another simulator's
material file; in particular nothing is derived from Meep's `materials.py` or
any other GPL source. Each record stores the page and the paper behind it, so a
simulation result traces back to a measurement.

## Limits

- **A fit is valid only inside its `wavelength_range_m`.** Pole models
  extrapolate confidently and wrongly; `get_material(..., wavelength=...)` warns
  outside the band, and nothing warns when you run a broadband source past it.
- **Every entry is isotropic.** Uniaxial crystals appear as separate ordinary and
  extraordinary entries (`LiNbO3_o` / `LiNbO3_e`); building the tensor from them
  — per-axis or oriented poles — is up to you.
- **Metals have no constant-permittivity form.** `Re ε < 0` is unconditionally
  unstable in explicit FDTD, so `get_material("Au", wavelength=...)` raises and
  points at the dispersive model.
- **Transparency-window entries carry no absorption.** A lossless Sellmeier fit
  has `Im ε = 0` by construction; residual absorption in the source data (for
  example water's `k ~ 1e-4` in the near-IR) is dropped.
- **Band edges are not modelled.** `Si` covers the telecom band only and
  `Si_visible` the 0.5–1.1 µm range; the metals below ~0.6 µm (`Au`, `Cu`) omit
  the interband transitions.
- **Anisotropic + dispersive together** is still restricted to the isotropic and
  diagonal paths in the engine — see
  [materials-anisotropy.md](materials-anisotropy.md).
