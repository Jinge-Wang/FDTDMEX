# Non-Uniform Grids (spacing-weighted operators)

**Design requirement, not an afterthought.** FDTDMEX treats graded/non-uniform grids as first-class. FDTDX's anisotropic off-diagonal averaging is an *unweighted* 4-point mean, which is only 1st-order accurate on stretched grids. We carry per-axis **Yee cell-size arrays** through the engine and use **spacing-weighted** finite differences and interpolation, keeping the curl *and* the anisotropic coupling 2nd-order on graded meshes.

> **Status — implemented & validated (M4).** The MLX engine threads per-axis cell widths through the curl ([`mlx/curl.py`](../src/fdtdx/mlx/curl.py) — metric-scaled differences), the detector interpolation ([`mlx/interpolate.py`](../src/fdtdx/mlx/interpolate.py)), and the anisotropic off-diagonal average ([`mlx/aniso.py`](../src/fdtdx/mlx/aniso.py)); widths are precomputed on the host in [`mlx/bridge.py`](../src/fdtdx/mlx/bridge.py). On a uniform grid every weighted form reduces *exactly* to the unweighted M3 path (verified element-wise). The off-diagonal average is **2nd-order on a graded mesh** where FDTDX's is 1st-order — measured convergence slopes **2.00 (weighted) vs 1.00 (unweighted)**:
>
> ![Convergence](images/nonuniform_convergence_mlx.png)
>
> Tests: `tests/validation/test_mlx_nonuniform.py`, `tests/visualization/test_nonuniform_convergence_visual.py`.
>
> **One subtlety the implementation gets right:** only the *center→edge* half-step is width-weighted; the *edge→center* half-step lands on the exact geometric cell-center midpoint and stays an unweighted mean. By the Yee staggering this means the **backward** roll is always the weighted one and the **forward** roll a plain mean — matching FDTDX's own `interpolate_fields`, which only weights its `_backward_edge_average` steps.

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

Implement a small reusable helper, e.g. `interp_component_to(field, comp, target_loc, spacings)`, and use it for all 6 cross-terms in both the E and H anisotropic updates.

## Conductivity & coefficients

The conductivity→coefficient scaling and any spacing-dependent normalization must use the **local** cell size, not a single global resolution. Keep a single source of truth for spacings in the grid object and pass it (or views of it) into every operator.

## Validation

Test **2nd-order convergence on a graded mesh** (error ∝ Δ²) for: (a) the curl on a known analytic field, and (b) a birefringence/walk-off case exercising the off-diagonal interpolation. A naive unweighted average will reveal itself as 1st-order and fail this test. See the `physics-validation` skill.

## API rule

Every public operator (curl, update_E/H, interpolation, energy/flux) **accepts the grid (or its spacing arrays)**. Do not bake in uniform spacing; retrofitting non-uniform support later is painful and error-prone.
