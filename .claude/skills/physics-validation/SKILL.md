---
name: physics-validation
description: How to validate FDTDMEX physics — analytic benchmarks, FDTDX/MEEP cross-checks, conservation tests, two-run normalization, and steady-state extraction. Use when adding or verifying any physics kernel.
user-invocable: false
---

# Physics Validation

Generated FDTD kernels routinely pass smoke tests while being physically wrong (sign errors, off-by-one Yee staggering, bad PML/Courant coefficients). Every physics addition needs a `validation`-marked test. **If a test fails marginally, raise resolution — do not loosen tolerances.**

## Tiers (pytest markers)

- `unit` — pure component tests (shapes, coefficients), no time stepping.
- `integration` — object placement, array bridge, multi-component wiring.
- `validation` — full forward runs vs an analytic result or a reference solver.

## Reference oracles

- **FDTDX (JAX) on CPU** — `uv sync --extra validation`. Run the *same* grid/dt/source through FDTDX and FDTDMEX and compare E/H arrays **element-wise** at matched time steps; expect single-precision agreement. This is the strongest check for a faithful port.
- **MEEP** (`../meep`) — oracle for subpixel smoothing (WS-C) and mode profiles (WS-B / MPB).
- **Analytic** — see below.

## Canonical benchmarks

- **Dielectric slab** transmission/reflection vs Fresnel — checks update + impedance + PML.
- **Waveguide mode propagation** — checks the mode source/injection and dispersion.
- **Point dipole in vacuum** — radiated power / field decay.
- **Birefringence / walk-off** in a uniaxial/biaxial crystal — checks the **full-anisotropic** off-diagonal coupling and spacing-weighted interpolation.
- **PML reflection** — measure residual reflection at a boundary (should be ≪ 1).
- **Energy / Poynting conservation** in a lossless closed cavity.

## Two-run normalization pattern

Normalize out source/grid factors by taking a ratio of a "test" run to a "reference" run that shares a helper:

```python
def _run(setup): ...                       # place → bridge → forward → detector states
def _mean_flux(arrays, name): ...          # mean over last N steady-state steps
transmission = _mean_flux(_run(test), "det") / _mean_flux(_run(ref), "det")
```

## Steady-state extraction

Average over the last few optical periods:

```python
steps_per_period = round(wavelength / (c0 * dt))
steady = float(mx.mean(flux[-10 * steps_per_period:]))
```

## Non-uniform-grid checks

Verify 2nd-order convergence on a **graded** mesh (error ∝ Δ²), not just uniform — this is the point of the spacing-weighted curl/interpolation. A naive (unweighted) average will show 1st-order convergence and fail this.
