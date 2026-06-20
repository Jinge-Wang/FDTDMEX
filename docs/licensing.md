# Licensing

## Current state

The project is provisionally licensed **Apache-2.0** (see [`LICENSE`](../LICENSE), [`NOTICE`](../NOTICE)). This is a **placeholder**. **Final licensing is owner-managed** — Jinge will reconcile and sanitize licensing before any public distribution.

## Working policy during development

Agents and contributors may **freely read, analyze, and port** from the reference projects, including directly translating kernels where that is fastest. There is no clean-room restriction during development. The only ask: **note in the commit/PR which reference file you adapted**, so provenance is traceable for the later reconciliation.

## References and their licenses

| Reference | License | Used for |
|---|---|---|
| `../fdtdx` | **MIT** | primary porting source (forward engine, conventions); CPU cross-check oracle. MIT is permissive — Apache-2.0-compatible with attribution. |
| `../meep` | **GPL v2+** | subpixel smoothing (`src/anisotropic_averaging.cpp`), near-to-far field (`src/near2far.cpp`), MPB mode-solver cross-check. **Copyleft** — code derived from MEEP source carries GPL obligations. |

## The GPL consideration (for the owner's later reconciliation)

The **algorithms** FDTDMEX borrows from MEEP (Kottke/Farjadpour subpixel smoothing; near-to-far transform) are **published and public-domain** (see citations in [subpixel-smoothing.md](subpixel-smoothing.md)); algorithms are not copyrightable. Only MEEP's specific **code expression** is GPL. Before distribution, the owner should decide one of:
1. Ensure MEEP-derived modules are independent reimplementations from the papers (keep Apache-2.0), or
2. Isolate any MEEP-derived code in a separately-licensed (GPL) optional component, or
3. License the whole project GPL.

Until then, treat the boundary as a development convenience, not a distribution decision. The MEEP source is **referenced externally** (`../meep`), never vendored into this repo.

## Attribution

Keep [`NOTICE`](../NOTICE) current with heritage/attribution. FDTDX is acknowledged as the numerical-design source; MEEP as the algorithmic reference for smoothing/N2F.
