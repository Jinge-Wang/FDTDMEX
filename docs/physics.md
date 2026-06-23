# Physics & Conventions

These are the Yee-grid and update conventions the forward engine uses; they match the JAX reference exactly, so the two cross-check element-wise.

## Yee grid

Axes: 0=x, 1=y, 2=z. Field arrays are `(3, Nx, Ny, Nz)` (index 0 = component). Staggered positions (Taflove convention):

```
E_x: (i+½, j,   k  )     H_x: (i,   j+½, k+½)
E_y: (i,   j+½, k  )     H_y: (i+½, j,   k+½)
E_z: (i,   j,   k+½)     H_z: (i+½, j+½, k  )
```

Leapfrog: E at integer steps, H at half steps. **Single-step order:** update E (curl H) → update H (curl E) → inject sources → record detectors.

## Field normalization

H is **eta0-normalized** (η₀ ≈ 376.73 Ω folded into the update coefficients rather than written explicitly), which symmetrizes the E/H updates.

## Update equations (isotropic / diagonal, lossless)

```
E^(n+1) = E^n + c · curl(H) · inv_eps
H^(n+1/2) = H^(n-1/2) − c · curl(E) · inv_mu
```
with `c = courant_number = courant_factor / sqrt(3)` (3D; default factor ≈ 0.99 → c ≈ 0.571).

**With conductivity (lossy):**
```
factor_E = 1 − c·σ_E·η0·inv_eps/2
E = factor_E · E + c · curl(H) · inv_eps ;   E = E / (1 + c·σ_E·η0·inv_eps/2)
factor_H = 1 − c·σ_H/η0·inv_mu/2
H = factor_H · H − c · curl(E) · inv_mu ;    H = H / (1 + c·σ_H/η0·inv_mu/2)
```
Note the asymmetry: σ_E multiplied by η0, σ_H divided by η0.

## Full-anisotropic update (off-diagonal coupling)

Material arrays store **inverse** tensors. For the 9-component case the update is implicit per cell:
```
E^(n+1) = A · E^n + B · curl(H) ,   A = solve(M1, M2), B = c·solve(M1, inv_eps)
M1 = I + (c·η0/2)·(inv_eps·σ_E),  M2 = I − (c·η0/2)·(inv_eps·σ_E)
```
Because Ex/Ey/Ez live at different Yee locations, off-diagonal terms (e.g. ε⁻¹_xy·E_y at E_x's location) require interpolating the other components to the target location. The interpolation is spacing-weighted (see [nonuniform-grid.md](nonuniform-grid.md)), which keeps it 2nd-order accurate on graded meshes.

## Material representation

- inverse permittivity/permeability stored; sizing `1/3/9` components, global (any full-anisotropic object → all arrays 9-component).
- conductivity scaled by resolution at setup (don't pre-scale).
- dispersion: ε(ω) = ε∞ + χ(ω) via ADE poles (Lorentz/Drude); ε in `permittivity` is ε∞ when dispersive. See [materials-anisotropy.md](materials-anisotropy.md).

## Boundaries

CPML (graded σ/κ/α with ψ auxiliary fields), PEC (zero tangential E), PMC (zero tangential H), Bloch/periodic (phase-shifted; complex fields when k≠0).

## Stability

CFL: `c·dt·sqrt(Σ 1/Δ_i²) ≤ 1`. On non-uniform grids the **minimum** spacing sets the stable time step.
