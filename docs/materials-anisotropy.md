# Materials & Anisotropy

The headline capability: **full-tensor (off-diagonal) anisotropic, spatially heterogeneous** materials — the niche where Apple unified memory beats a single CUDA GPU.

## Representation

Materials carry ε, µ, σ_E, σ_H, each as a 3x3 tensor (row-major xx,xy,xz,yx,yy,yz,zx,zy,zz). Input forms accepted: scalar (isotropic), 3-tuple (diagonal), 9-tuple or nested 3x3 (full). Stored as **inverse** tensors for the hot loop.

On the grid, arrays are sized **globally**:
- `(1, Nx,Ny,Nz)` if every object is isotropic,
- `(3, …)` if any is diagonal-anisotropic,
- `(9, …)` if any is full-anisotropic.

Different regions hold different tensors (heterogeneous): uniform objects write their inverted 3x3 into their grid slice; multi-material objects/devices map a per-voxel material index into the ordered material list.

## Update

Full-anisotropic E/H updates do a per-cell 3x3 solve with off-diagonal coupling, requiring spacing-weighted interpolation of components across Yee locations — see [physics.md](physics.md) and [nonuniform-grid.md](nonuniform-grid.md).

## Dispersion (ADE)

Linear dispersion via auxiliary differential equations: ε(ω) = ε∞ + Σ χ_p(ω), each pole a two-step recurrence `P[n+1] = c1 P[n] + c2 P[n-1] + c3 E[n]` in an auxiliary polarization field updated alongside E.
- **Lorentz**: `χ = Δε·ω₀² / (ω₀² − ω² − iγω)`.
- **Drude**: `χ = −ω_p² / (ω² + iγω)` (ω₀ = 0).
- **Sellmeier**: one data-sheet term `B λ²/(λ² − C)`, i.e. a lossless Lorentz pole (`ω₀ = 2πc/√C`, `Δε = B`, `γ = 0`).
- **Debye**: `χ = Δε / (1 − iωτ)` — first order, so it overrides the recurrence hook with the exact exponential update `c1 = exp(−dt/τ)`, `c2 = 0`, `c3 = Δε(1 − exp(−dt/τ))`. Unconditionally stable; see [materials-library.md](materials-library.md) for the half-step sampling caveat.

Coefficients computed once at setup (host). **On MLX:** `P_curr`/`P_prev` threaded through the E-side of the loop, the per-pole recurrence both in the MLX-op `_update_E` and folded into the Metal E-kernel (`mlx/kernels.py` `_ade_lines`) so dispersive media also hit the bandwidth floor. The fold is generic in `(c1, c2, c3)`, so Debye rides the same path (there is no upstream parity oracle for it — upstream has Drude + Lorentz only).

**Known FDTDX restriction (iso/diagonal dispersion only):** full-anisotropic **+** dispersive simultaneously is not supported upstream (`NotImplementedError`), so the MLX port is iso/diagonal only and there is no fdtdx oracle for the anisotropic case. Lithium niobate (anisotropic + dispersive + χ²) is the motivating case for lifting it — a documented low-priority future item with the MEEP state-of-the-art reference (per-pole 3x3 σ tensor with Yee-averaged off-diagonal coupling) in [roadmap.md](../dev-docs/roadmap.md#genuinely-new-physics-needs-meep-reference-or-new-derivation).

## Subpixel smoothing → anisotropy

Subpixel smoothing produces effective **tensors** even for isotropic geometry at tilted interfaces, so it always feeds the 9-component path. See [subpixel-smoothing.md](subpixel-smoothing.md).

## χ² nonlinearity (future)

Second-order nonlinearity (Pockels/SHG; LiNbO₃) is a **local nonlinear-polarization term** added to the E-update: `P_NL,i = ε₀ Σ_{jk} χ²_{ijk} E_j E_k`. In a *forward* time-domain solver this is a straightforward per-cell contribution. The χ² tensor is itself anisotropic (e.g. LiNbO₃); it is not part of the engine today and is a planned future addition.
