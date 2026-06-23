# reference/

> **Update (fork pivot):** This repo is now a **fork of fdtdx** — fdtdx's source lives in-tree at `src/fdtdx/` (with the MLX backend added under `src/fdtdx/{backend,mlx}`) and upstream is the `upstream` git remote. So fdtdx is no longer "external"; the sibling `../fdtdx` below is just a pristine read-only reference. MEEP remains an external algorithmic reference. See [dev-docs/decisions/0001-mlx-forward-first.md](../dev-docs/decisions/0001-mlx-forward-first.md).

The reference projects below live as sibling directories in the development workspace:

- **`../fdtdx`** — FDTDX (MIT). Primary porting source and CPU cross-check oracle.
  - `src/fdtdx/fdtd/update.py` — E/H updates, incl. full-anisotropic (9-tensor) path.
  - `src/fdtdx/core/physics/curl.py` — Yee curl.
  - `src/fdtdx/fdtd/misc.py` — `compute_anisotropic_update_matrices`, off-diagonal averaging.
  - `src/fdtdx/objects/boundaries/perfectly_matched_layer.py` — CPML.
  - `src/fdtdx/objects/sources/{tfsf,mode}.py` — mode injection (TFSF).
  - `src/fdtdx/dispersion.py` — ADE dispersion.
  - `src/fdtdx/fdtd/initialization.py` — material-array construction (array-bridge reference).
  - `src/fdtdx/conversion/json.py` — declarative config round-trip (WS-D reference).
  - `.claude/skills/fdtdx/SKILL.md` — FDTDX conventions (authoritative for physics parity).

- **`../meep`** — MEEP (GPL v2+). Algorithmic reference only.
  - `src/anisotropic_averaging.cpp` — subpixel smoothing (WS-C); `src/sphere-quad.h` quadrature table.
  - `src/near2far.cpp` — near-to-far-field transform (optional future feature).
  - `libpympb/`, `src/mpb.cpp` — MPB mode solver (WS-B cross-check).

**Licensing:** porting from either is fine during development; reconciliation is owner-managed. See [../docs/licensing.md](../docs/licensing.md). Do not copy GPL MEEP source into the package tree without recording provenance.

If you relocate this repo away from the workspace, either re-point these paths or vendor the needed files here (and update licensing/attribution accordingly).
