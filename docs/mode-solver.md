# Mode Solver & Overlap (WS-B) — a swappable, Tidy3D-free backend

A lightweight, host-side, **full-vectorial** waveguide mode solver + mode overlap, packaged as a
**drop-in replacement for the Tidy3D call** that fdtdx currently uses. Mode *injection* and the
`ModePlaneSource`/`ModeOverlapDetector` front-end objects are **kept as-is** — we only swap the engine
underneath them.

## Why (strategic)

fdtdx delegates all mode work to **Tidy3D** ([`src/fdtdx/core/physics/modes.py`](../src/fdtdx/core/physics/modes.py)
imports `tidy3d.components.mode.solver.compute_modes`; Tidy3D is a hard dependency, `tidy3d>=2.8.0`).
That is the **only** Tidy3D coupling in the whole stack — nothing else in the forward engine touches
it. Tidy3D is Flexcompute's commercial product; for an independent project an own solver removes the
dependency, the conflict-of-interest optics, and the licensing/direction risk. The algorithm is
standard and small, so this is low-risk.

**Parity note:** bit-for-bit agreement with Tidy3D is *not* a goal (Tidy3D and Lumerical don't agree
bit-for-bit either). The contract is "return mode profiles **shaped like** Tidy3D's so the existing
source/detector keep working," validated on **physics** (analytic n_eff + MPB), not on a byte oracle.

## The swap point — a single indirection

Today: `compute_mode(...)` → `tidy3d ... compute_modes(...)` → `(neff, E, H)`.

Target: a `mode_backend` indirection so `compute_mode` dispatches to either `"tidy3d"` (kept for
cross-checking) or `"fdtdmex"` (our module). The two must agree on **inputs and outputs** so
`ModePlaneSource`/`ModeOverlapDetector` are untouched:

- **Inputs** (what fdtdx already assembles in `modes.py`): the permittivity slice on the propagation
  plane (1/3/9-tensor, rotated to the solver's transverse convention), the transverse grid (uniform
  `resolution` *or* rectilinear `transverse_coords`), `wavelength`, `num_modes`, `target_neff`,
  `filter_pol` (`te`/`tm`/None), `symmetry` (PEC/PMC walls at the min edge), permeability, and the
  optional bend (`bend_radius`/`bend_axis`).
- **Output:** `ModeTupleType(neff, Ex, Ey, Ez, Hx, Hy, Hz)` per mode (the named tuple already in
  `modes.py`), with fields normalized by Poynting flux and **η₀-scaled to fdtdx's H convention** (the
  Tidy3D path multiplies `mode_H * tidy3d.constants.ETA_0` — reproduce that so amplitudes match).

Build the module as package code (e.g. `src/fdtdx/core/physics/mode_backend/`) and make `compute_mode`
select it via an argument / config / env. Default can stay Tidy3D until the own solver passes its
physics suite, then flip.

## Method — full-vectorial FD on a 2-D Yee mesh (Zhu & Brown 2002)

Reference: **Zhu & Brown, "Full-vectorial finite-difference analysis of microstructured optical
fibers," Opt. Express 10(17):853–864 (2002)** — local copy [`reference/oe-10-17-853.pdf`](../../reference/oe-10-17-853.pdf).
The method is exactly the lineage Tidy3D/EMpy/MPB descend from, so reproducing Tidy3D-like profiles is
expected. Open-source implementations to crib from (MIT/BSD): **EMpy** `FullVectorialModeSolver`,
**modesolverpy** (jtambasco), **wgms3d** — all the same Zhu–Brown discretization.

Formulation (paper §2):
- Yee's **2-D** transverse mesh; solve for the **transverse magnetic field** `(Hx, Hy)` as the
  eigenvector, eigenvalue `β²` (→ `n_eff = β/k₀`). Eqs. (4)–(5) discretize Maxwell on the mesh; Eq. (7)
  assembles the sparse block operator relating `H` and `E` components via the difference operators `U`.
- **Index averaging** at interfaces (Eqs. 6a–c: 2-cell averages for `εx`,`εy`; 4-cell for `εz`) — this
  is the paper's built-in **subpixel smoothing** to soften the staircase. It is the *same* idea as
  WS-C: when WS-C lands, the solver should consume WS-C's per-cell smoothed material tensors instead of
  re-deriving the averaging; until then, the Zhu–Brown 2/4-cell average is self-contained and
  sufficient. (This is the concrete mode-solver ↔ subpixel-smoothing interconnection.)
- Assemble as a **sparse** generalized eigenproblem; solve the few highest-`n_eff` modes with
  `scipy.sparse.linalg.eigs`/`eigsh` (shift-invert near `(n_max·k₀)²`). Boundary walls: Dirichlet/PEC
  and Neumann/PMC for `symmetry`; (PML for leaky/confinement loss is a later add).

## Components — `src/fdtdx/core/physics/mode_backend/`
- `operator.py` — assemble the sparse 2-D-Yee transverse operator from the ε slice + grid spacings
  (uniform or rectilinear); index averaging (Eqs. 6) or WS-C tensors.
- `solve.py` — `scipy.sparse.linalg.eigs` wrapper → sorted/filtered `(neff, Et, Ht)`; reconstruct
  `Ez`/`Hz` from the transverse fields; `te`/`tm` polarization-fraction filter + `mode_index`.
- `overlap.py` — modal overlap integral `(E_sim × H_mode* + E_mode* × H_sim)` over the plane,
  power-normalized → forward/backward amplitudes (what `ModeOverlapDetector.compute_overlap` needs).
- A thin `compute_modes_fdtdmex(...)` adapter matching the Tidy3D `compute_modes` call so `modes.py`'s
  wrapper swaps cleanly.

## Scope / staging
1. Straight waveguide, uniform grid, isotropic/diagonal ε → fundamental + low-order modes; `te/tm`
   filter; PEC/PMC walls. (Covers the common case.)
2. Rectilinear (non-uniform) transverse grid; full 9-tensor ε.
3. **Defer:** bends (`bend_radius` conformal transform) and PML/leaky-mode confinement loss — the only
   genuinely fiddly parts; keep routing those to Tidy3D until needed.

## Validation (physics, not byte-parity)
- Analytic: symmetric **dielectric slab** TE/TM `n_eff` (transcendental dispersion relation); step-index
  **fiber** LP modes (Bessel). Assert `n_eff` and field-profile shape.
- Cross-check vs **MPB** (MEEP's bundled solver, `../meep/libpympb/`) and, during development only, vs
  **Tidy3D** (the kept backend) on the same cross-section.
- End-to-end: a single-mode waveguide should show `ModeOverlapDetector` transmission ≈ 1 (lossless),
  reusing fdtdx's injection unchanged.

## Estimate
Solver ~3–5 d, overlap ~2–3 d, adapter/swap + validation ~3–5 d → ~1.5–2 wk. Independent of the
agentic-workspace track ([mcp-and-ui.md](mcp-and-ui.md)); the two can proceed in parallel.
