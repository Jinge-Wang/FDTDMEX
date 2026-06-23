# Non-Uniform Grids (spacing-weighted operators)

**Design requirement, not an afterthought.** FDTDMEX treats graded/non-uniform grids as first-class. FDTDX's anisotropic off-diagonal averaging is an *unweighted* 4-point mean, which is only 1st-order accurate on stretched grids. We carry per-axis **Yee cell-size arrays** through the engine and use **spacing-weighted** finite differences and interpolation, keeping the curl *and* the anisotropic coupling 2nd-order on graded meshes.

> **Implemented and validated.** The MLX engine threads per-axis cell widths through the curl (metric-scaled differences), the detector interpolation, and the anisotropic off-diagonal average, with the widths precomputed once on the host. On a uniform grid every weighted form reduces *exactly* to a plain unweighted average (verified element-wise). The off-diagonal average is **2nd-order on a graded mesh** where an unweighted average is only 1st-order — measured convergence slopes **2.00 (weighted) vs 1.00 (unweighted)**:
>
> ![Convergence](../tests/visualization/figures/nonuniform_convergence_mlx.png)

## Grid representation

A rectilinear non-uniform grid is defined by **edge coordinates** per axis: `x_edges`, `y_edges`, `z_edges`. From these derive:
- **primal spacings** `Δ_i = edges[i+1] − edges[i]` (cell sizes), and
- **dual spacings** `Δ̃_i = (Δ_i + Δ_{i-1}) / 2` (distances between cell centers / Yee duals).

The E and H components, being staggered by half a cell, "see" different spacings (primal vs dual) along each direction. The grid object must expose both as 1-D arrays per axis (broadcastable into `(Nx,Ny,Nz)`), plus cell volumes and face areas for energy/flux integrals.

## Spacing-weighted curl

A derivative `∂f/∂x` across a face is `(f[i+1] − f[i]) / Δ_x` using the **local** spacing for that location (primal for one field, dual for the other), not a global constant. Implement curl as finite differences divided by the appropriate per-axis spacing array (broadcast), e.g.

```
(∂H_z/∂y − ∂H_y/∂z)  with  ∂H_z/∂y = (roll(H_z, -1, y) − H_z) / Δ̃_y[None,:,None]
```

(exact primal/dual assignment follows the Yee staggering in [physics.md](physics.md)).

## Spacing-weighted interpolation (off-diagonal anisotropy)

To place component `E_b` at the location of component `E_a`, interpolate using **distance weights** from the cell-size arrays rather than a plain mean. For a target at fractional position between two samples separated by spacings `Δ⁻, Δ⁺`, the linear weight is `w⁺ = Δ⁻/(Δ⁻+Δ⁺)` (and symmetrically), generalized to the 4-point (bilinear) stencil as a product of per-axis weighted 1-D interpolations. On a uniform grid these weights reduce to ¼ each (recovering FDTDX's average); on a graded grid they restore 2nd-order accuracy.

The same weighted interpolation is applied to all six cross-terms in both the E and H anisotropic updates.

## Conductivity & coefficients

The conductivity→coefficient scaling and any spacing-dependent normalization use the **local** cell size rather than a single global resolution.

## Validation

Convergence is measured at **2nd order on a graded mesh** (error ∝ Δ²) for both the curl on an analytic field and a birefringence/walk-off case exercising the off-diagonal interpolation; an unweighted average shows up as 1st-order on the same test.
