# Mode Solver & Overlap (WS-B)

The smallest workstream: a lightweight, host-side waveguide mode solver plus mode overlap. Mode **injection** is the hard part and is *ported* from FDTDX (don't reinvent it).

## Method

Full-vectorial finite-difference eigenmode solver on a **2D Yee cross-section** (the classic approach, e.g. Zhu & Brown, *Opt. Express* 10(17):853, 2002): discretize the transverse-field operator, assemble a **sparse** generalized eigenproblem, and solve a few modes with `scipy.sparse.linalg.eigs` on the host → effective index `n_eff` and transverse E/H fields.

The only boundary subtlety is **index averaging** at material interfaces — which is exactly WS-C's subpixel smoothing. So the solver consumes WS-C's per-cell material tensors; build WS-C first.

Cross-check option: MEEP's bundled MPB (`../meep/libpympb/`, `../meep/src/mpb.cpp`).

## Mode overlap

Given a recorded field monitor, the overlap with a solved mode is the spatial integral of the cross-product `(E_sim × H_mode* + E_mode* × H_sim)` over the plane (power-normalized). This yields forward/backward modal amplitudes → S-parameters. It is a simple host-side spatial integral once a field monitor exists.

## Injection (ported, not reinvented)

Unidirectional mode launching is TFSF (total-field/scattered-field): an equivalence-principle current sheet with the half-step E/H Yee offset and impedance/phase matching so the mode propagates one way without back-radiation. Port from `../fdtdx/src/fdtdx/objects/sources/tfsf.py` and `mode.py`, and feed it *our* solved mode profile.

## Components in `fdtdmex/modes/`
- `fd_operator` — assemble the sparse 2D-Yee transverse operator (uses grid spacings + WS-C tensors).
- `solve` — `scipy.sparse.linalg.eigs` wrapper → `(n_eff, E_t, H_t)`, sorted/filtered (TE/TM, mode index).
- `overlap` — modal overlap integral vs a field monitor.
- injection lives in `sources/` (ported TFSF), consuming a solved profile.

## Estimate
After WS-C: solver ~3–5 d, overlap ~2–3 d, injection port ~3–5 d, validation ~3–5 d → ~1.5–2 wk.
Validate `n_eff` and field profiles against MPB/analytic slab modes.
