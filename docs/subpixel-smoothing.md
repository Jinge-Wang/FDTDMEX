# Subpixel smoothing

Binary-voxelized geometry staircases at interfaces and degrades FDTD and mode-solver accuracy. Subpixel smoothing replaces the per-cell permittivity with an **effective tensor** that captures the sub-cell interface, recovering ~2nd-order accuracy. Smoothing even an *isotropic* material at a tilted interface yields an **anisotropic** tensor, so the output always feeds the 9-component path.

## Algorithm (Kottke / Farjadpour)

For each cell, from a supersampled scalar permittivity raster:
- `meps = <ε>` (arithmetic mean) and `minveps = <1/ε>` (harmonic-side mean) over the sub-cells;
- interface normal `n` from the fine-grid ε gradient; uniform cells stay isotropic;
- project — harmonic mean **normal** to the interface, arithmetic mean **tangential**:
  ```
  chi1inv = P (minveps − 1/meps) + I (1/meps),   P_ij = n_i n_j
  ```

References (public-domain): Farjadpour et al., *Opt. Lett.* 31, 2972 (2006); Kottke, Farjadpour, Johnson, *Phys. Rev. E* 77, 036611 (2008). MEEP's `anisotropic_averaging.cpp` is a GPL **reference only** (algorithm reproduced, no code copied — see [licensing.md](licensing.md)).

## API and validation

- `smooth_inverse_permittivity(eps_fine, factor) → (9, Nx, Ny, Nz)` and the 2-D mode-solver wrapper `smooth_cross_section_2d(eps_fine_2d, factor) → (9, Nx, Ny)`. Both exported from `fdtdx`.
- [`../tests/validation/test_subpixel.py`](../tests/validation/test_subpixel.py): a 50/50 cell matches the analytic effective medium exactly; uniform cells stay isotropic; a 45° interface gives the expected off-diagonal tensor; and feeding the smoothed tensor to the native mode solver cuts the off-grid-slab n_eff staircase error ~15x.

## Integration

The smoothing utility takes a supersampled raster and is called explicitly. Automatic smoothing during `place_objects` (supersampling object geometry and smoothing the assembled grid) is opt-in and off by default, so standard runs stay byte-for-byte comparable to the unsmoothed path.
