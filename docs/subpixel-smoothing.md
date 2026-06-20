# Subpixel Smoothing (WS-C)

Static geometry voxelized as binary masks staircases at interfaces, degrading FDTD accuracy.
Subpixel smoothing replaces the per-Yee-component permittivity with an **effective tensor** that captures the sub-cell interface, recovering ~2nd-order accuracy. This is a **host-side pre-time-stepping** step that emits the material tensor arrays WS-A consumes.

## The algorithm (Kottke / Farjadpour)

Key fact: smoothing even an *isotropic* material at a tilted interface yields an **anisotropic effective tensor** — so WS-C output always feeds WS-A's 9-component path.

For each Yee component location:
1. **Interface normal `n`** — estimate by sampling ε on a small quadrature sphere/cube around the point and taking the (weighted) gradient. If ε is uniform in the neighbourhood → trivial, skip.
2. **Fill averages** over the pixel volume: `meps = <ε>` (arithmetic mean) and `minveps = <1/ε>` (harmonic-side mean), via sub-sampling / adaptive cubature.
3. **Effective inverse tensor** = project: harmonic mean **normal** to the interface, arithmetic mean **tangential**:
   ```
   chi1inv = P·(minveps − 1/meps) + I·(1/meps),   P_ij = n_i n_j
   ```

References (algorithm is published / public-domain):
- A. Farjadpour et al., "Improving accuracy by subpixel smoothing in FDTD," *Opt. Lett.* 31, 2972 (2006).
- C. Kottke, A. Farjadpour, S. G. Johnson, *Phys. Rev. E* 77, 036611 (2008).

Reference implementation to adapt: `../meep/src/anisotropic_averaging.cpp` — `material_function::normal_vector()` and `eff_chi1inv_row()` are the ~150 self-contained core lines (the surrounding `set_chi1inv()` chunk loop is MEEP's data model and is replaced by a plain numpy/MLX loop over Yee points). The quadrature table is `../meep/src/sphere-quad.h`.

## What FDTDMEX needs

- **`chi1p1(point) → ε`**: a "scalar permittivity at an arbitrary continuous point" evaluator built from the geometry primitives (shapes / GDS), or by supersampling occupancy. Lives in `geometry/`.
- **smoothing kernel** in `materials/smoothing/`: normal estimation + fill averages + tensor projection, looped over Yee points (vectorizable; embarrassingly parallel).
- Output: per-Yee-component inverse-ε tensor arrays in the `(9, Nx, Ny, Nz)` layout WS-A expects.

## Implementation notes

- It's a setup-time step; correctness and clarity matter more than speed, but it vectorizes well (and unified memory hands the result to the GPU with no transfer).
- Get the Yee-component placement / half-cell offsets exactly right — mis-aligned tensors land on the wrong points and silently degrade accuracy.
- Validate against MEEP: same geometry → compare effective tensors and a transmission/scattering result (see `physics-validation` skill).

## Licensing

The algorithm is public-domain (papers above). MEEP source is GPL; if any code is adapted from it, licensing reconciliation is owner-managed — see [licensing.md](licensing.md).
