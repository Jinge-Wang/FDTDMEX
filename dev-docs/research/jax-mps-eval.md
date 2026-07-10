# jax-mps / applejax vs the FDTDMEX MLX engine — deep-research log

Running record. Question: is a JAX-on-Metal **plugin** (jax-mps / applejax) a better long-term base
than the hand-written MLX/Metal fork — weighing forward performance, features (esp. autodiff /
inverse design), and **interface unification** (one fdtdx, retire fdtdmex)? Libraries judged on
measured merit; immaturity is a contribution opportunity, not a disqualifier.

Plan: `~/.claude/plans/i-was-tipped-by-joyful-elephant.md`.

---

## Track 1 — upstream sync to a clean base (DONE, 2026-07-09)

Done first so the plugin comparison runs against a fork that is consistent with current upstream
(and so the user's now-upstreamed anisotropic fix is retired into upstream rather than carried).

**Delta:** upstream `main` `e5351a4 → 65e0fd4` = **15 commits** (a fresh `git fetch` — cached refs
had shown only 1). Buckets: fork-aligned (#378 offdiag spacing-weight = the user's own commit, now
merged upstream; #376 2D modes / etched / `use_etching` / `compute_integrated_power` /
`grid.subgrid`), engine contract-surface (#379 fast PML, #384 PML-constants-on-object, #382 complex
materials, #383 CCPR dispersion), and low-risk front-end (#375 field-projection detectors, #367 GDS
sidewall, #386 GDS masks, #389 mode scaling, #393 JSON serialize, #391 plot aspect, dep bumps).

**Merge:** branch `sync/upstream-65e0fd4` off `mlx-fork`; `git merge upstream/main` was **near-clean**
(fork is additive) — only 2 conflicts:
- `conversion/json.py`: adopt upstream's canonical array serialization (#393), keep the fork's legacy
  `__ndarray__` decoder for backward-compat reads.
- `tests/unit/core/physics/test_modes.py`: upstream's 4-element `transverse_coords` + the fork's
  `mode_backend="tidy3d"` seam arg.

**Behavioral re-point (the real work).** Two upstream restructurings hit the MLX bridge's contract
surface:
1. **PML (#379/#384).** CPML `a/b/inv_kappa` moved off `ArrayContainer.alpha/kappa/sigma` onto each
   `PerfectlyMatchedLayer` object (`pml_a_E/b_E/inv_kappa_E` + `_H`, slab-sized), and `psi_E/psi_H`
   became `PmlAuxField = dict[pml_name → (psi_1, psi_2)]`. Re-point (`mlx/pml.py`, `mlx/bridge.py`):
   - `build_cpml_coeffs_from_pml_objects()` re-assembles the fork's global `(6, Nx, Ny, Nz)` coeff
     view (E-side axis-k → channel k, H-side → k+3) from the per-object arrays; `detect_pml_slabs`
     and the Metal kernel are untouched.
   - ψ is zero after `arrays.reset()` (bridge-in) → allocate zero slabs. Bridge-out rebuilds the
     per-PML dict via `PML_AXIS_TO_PSI_CHANNELS` (axis 0→(3,4), 1→(5,0), 2→(1,2)), derived + verified
     against `curl_E`/`curl_H` `step_cpml` ordering (debug showed the mapped channel matching JAX ψ
     to rel 0.0000). `to_array_container` now takes `objects`.
2. **Dispersion.** ADE polarization `P` moved into `FieldState` (`arrays.fields.dispersive_P_*`);
   coefficients stayed on `ArrayContainer`. Bridge read/write re-pointed.

**New fallback gate:** CCPR (#383) adds a 4th ADE coeff `dispersive_c4` (the `b·dE/dt` term) the MLX
fold doesn't carry → gate to JAX. #382 complex materials need **no** gate (`from_complex_*` splits
to real ε + equivalent conductivity = already-supported lossy path; arrays stay real).

**Contract-surface change to remember:** `to_mlx_state` / `to_array_container` now **require the
`ObjectContainer`** (CPML coefficients are read from `objects.pml_objects`). Any caller must pass it.

**Validation:** full MLX parity gate **55 passed / 1 skip** (`tests/validation/`), covering iso /
diagonal / full-tensor anisotropy, CPML, periodic, PEC/PMC, non-uniform grids, Drude–Lorentz
dispersion — MLX-op cores and the fused Metal kernel both match JAX-CPU. 378/386 unit tests pass;
the 8 failures are `tidy3d`-optional-extra mode tests (real `import tidy3d`, extra not installed) —
not engine regressions. `io` roundtrip tests need the `pydantic`/`io` extra (not installed).
Commits on `sync/upstream-65e0fd4`: `c0410b3` (merge), `38fb4df` (re-point).

**Env note (helps Track 2B):** the fork already resolves to **jax 0.10.1** — the jaxlib 0.10.x line
jax-mps targets — so the plugin/​fork jaxlib pin conflict may be smaller than feared.

---

## Track 2 — plugin evaluation (PENDING)
- 2A: clone + architectural comparison of jax-mps and applejax (StableHLO→MLX mapping, op coverage,
  complex/linalg/autodiff/control-flow internals).
- 2B: isolated `env-jaxmps` / `env-applejax` (Python 3.13 + pinned jax/jaxlib/plugin).
- 2C: forward benchmark matrix (JAX-CPU / jax-mps / applejax / MLX-op / fused kernel), apples-to-apple.
- 2D: inverse-design experiment (reversible-FDTD adjoint on Metal via applejax).

## Track 3 — strategic decision memo (PENDING)
Option matrix (A unify+retire fork / B keep fused kernel + upstream Metal / C improve the sounder
plugin / D hybrid), scored on forward Mcs/s × autodiff × features × unification/divergence cost.
