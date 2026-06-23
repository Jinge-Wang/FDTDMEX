# Mode solver & overlap — native, Tidy3D-free

**Implemented.** A host-side **full-vectorial** waveguide mode solver that replaces the Tidy3D call fdtdx used to make. `ModePlaneSource` / `ModeOverlapDetector` and the injection path are unchanged — only the engine underneath them swapped. Tidy3D is now an **optional** dependency.

## The swap seam

`compute_mode(...)` in [`../src/fdtdx/core/physics/modes.py`](../src/fdtdx/core/physics/modes.py) dispatches on `mode_backend`: `"fdtdmex"` (default) or `"tidy3d"` (optional, for fully tensorial media, bends, and cross-checks). Select via the `mode_backend=` argument or the `FDTDMEX_MODE_BACKEND` env var. Both backends return `ModeTupleType(neff, Ex..Hz)` per mode in Tidy3D's z-propagation convention; all rotation, η₀-scaling (`eta0` from fdtdx constants), and Poynting normalisation happen in `compute_mode` after the call, so the two are interchangeable. Off-diagonal-tensor and bend cases raise `NotImplementedError` in the native backend and auto-route to Tidy3D when it is installed.

## Method

Standard full-vectorial transverse-E finite-difference operator on a 2-D Yee mesh (Zhu & Brown 2002; Fallahkhair, Li & Murphy 2008), solved as a sparse shift-invert eigenproblem (`scipy.sparse.linalg.eigs`) near the target n_eff. Independent implementation in [`../src/fdtdx/core/physics/mode_backend/`](../src/fdtdx/core/physics/mode_backend/):
- `operator.py` — forward/backward Yee difference matrices on uniform or rectilinear grids, with Dirichlet (PEC) / Neumann (PMC) walls from `symmetry`.
- `solve.py` — assembles the transverse-E operator from the diagonal ε/µ components, solves, and recovers all six field components (H scaled by `-1j/eta0` so the downstream `* eta0` restores units).
- `__init__.py` — `fdtdmex_mode_computation_wrapper`, mirroring the Tidy3D wrapper's signature/return.

## Capability and limits

- **Supported:** straight waveguide; uniform **and** rectilinear transverse grids; isotropic and diagonally-anisotropic permittivity/permeability; `te`/`tm` filter; `mode_index`; PEC/PMC symmetry.
- **Deferred (auto-route to Tidy3D):** fully tensorial (off-diagonal) media — the 4N×4N complex tensorial solver, which even Tidy3D's base package defers to a paid extra — and bends / PML leaky modes.

## Validation (physics, not byte-parity)

[`../tests/validation/test_mode_solver.py`](../tests/validation/test_mode_solver.py): analytic symmetric-slab TE₀ n_eff on uniform (~1.5e-5) and graded (~2.3e-6) grids; diagonal anisotropy; and a Tidy3D cross-check through the full `compute_mode` pipeline matching to ~1e-16 on a Si strip. End-to-end mode-source propagation and S-parameter transmission tests pass on the native backend.

## Subpixel smoothing

The solver consumes pre-smoothed cross-section tensors when provided (the subpixel-smoothing seam, [subpixel-smoothing.md](subpixel-smoothing.md)); feeding a smoothed interface cuts the staircase n_eff error ~15×. Reference: [`reference/oe-10-17-853.pdf`](../../reference/oe-10-17-853.pdf).
