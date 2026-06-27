# Upstream sync — staying current with fdtdx

How FDTDMEX tracks [ymahlau/fdtdx](https://github.com/ymahlau/fdtdx) over time. This is the
**porting statement**: the rules and the routine for absorbing upstream changes without breaking the
MLX engine, plus how to decide what flows back the other way (see
[UPSTREAM_CONTRIB.md](UPSTREAM_CONTRIB.md) for the contribution plan).

For the JAX→MLX *kernel* translation recipe (how to port a piece of physics into `src/fdtdx/mlx/`),
see [porting.md](porting.md). This document is the complementary half: how to merge upstream's own
changes *in*.

## The core premise

The fork is **additive**. The only edits to upstream-tracked files are:

- new packages `src/fdtdx/backend/` + `src/fdtdx/mlx/` (the entire MLX engine),
- new `src/fdtdx/core/physics/mode_backend/` + the `mode_backend` seam in `modes.py`,
- a ~4-line guarded hook in `src/fdtdx/fdtd/wrapper.py:run_fdtd`,
- export lines in `src/fdtdx/__init__.py`,
- `src/fdtdmex/` (the brand alias + the io/HDF5 + MCP layer — entirely new tree).

Everything else **tracks upstream verbatim**. So `git merge` is almost always clean at the text
level. **The risk is behavioral, not textual:** the MLX engine *mirrors* upstream physics
element-wise and *reads from* upstream data structures, so an upstream change can pass `git merge`
cleanly and still silently break parity. The whole protocol below exists to catch that.

## Branch model

- `main` — clean upstream mirror, **no MLX**. Fast-forwarded from `upstream/main`.
- `mlx-fork` — the MLX trunk. Phase work branches off it; local commits only.
- **Sync = `upstream/main` → `main` (fast-forward), then `main` → `mlx-fork` (merge).**
  This keeps every conflict isolated to the handful of hand-owned files above.

Remotes (already configured): `origin = Jinge-Wang/FDTDMEX`, `upstream = ymahlau/fdtdx`.

## The rules (porting statement)

1. **Upstream is the source of truth for the shared physics.** Never hand-edit an upstream-tracked
   file to "fix" something that should be fixed upstream — port the fix in, or send it upstream and
   merge it back. The fork's own changes live only in the additive trees.
2. **Correct-by-fallback first, fast second.** When upstream adds a new *forward* feature, the
   default action is to **widen the JAX-fallback gate** (`backend/dispatch.py:kernel_eligible` /
   the feature gate) so the case is *correct* on JAX, then implement MLX support as a follow-up.
   Never let an unrecognized feature fall through to the MLX path and silently produce a wrong result.
3. **A merge is not done until the parity gate is green.** Every core-physics merge re-runs the
   element-wise parity suite (below). Marginal failure → raise resolution, never loosen tolerance.
4. **Behavioral changes that parity tests can't see must be audited by hand.** Placement /
   coordinate-convention changes (e.g. origin-at-center) don't show up in field parity — they move
   objects. Audit examples, `Scene`, and the HDF5 contract explicitly.
5. **Free fixes are still reviewed.** Some upstream fixes flow into the MLX path automatically
   because the engine bridges the affected array (e.g. PML `kappa/sigma`). "Automatic" still means
   "add a regression test that proves it."
6. **Keep the fork-base pointer current.** After each sync, record the upstream commit `mlx-fork`
   was reconciled to, so the next triage is incremental.

## Per-sync protocol

1. `git fetch upstream && git fetch upstream --tags`
2. **Triage the delta** — `git log --stat <fork-base>..upstream/main`; bucket each commit:
   - **Noise** — deps / CI / docs / `uv.lock`. Merge, ignore.
   - **Core physics** — curl, update, PML, grid, materials, sources, detectors. Merge **and** run the
     parity gate + the contract-surface checklist.
   - **Front-end / placement semantics** — coordinates, constraints, placement order, GDS bounds.
     Merge on a branch, then audit examples / `Scene` / HDF5 before promoting.
3. **Merge** into `main` (ff), then `main` → `mlx-fork`. Resolve only the hand-owned files
   (`__init__.py` exports, the `wrapper.py` hook).
4. **Run the parity gate** (the definition of "done"):
   ```bash
   uv run --with pytest pytest tests/validation -q                          # kernel default-on
   FDTDMEX_METAL_KERNEL=0 uv run --with pytest pytest tests/validation -q   # MLX-op cores
   uv run --with pytest pytest tests/validation/test_mlx_nonuniform.py -q
   ```
5. **Bump the fork-base pointer.**

## Contract-surface checklist (run on every core-physics merge)

These upstream surfaces the MLX engine silently depends on. After a clean merge, diff each against
what `src/fdtdx/mlx/` assumes:

| Upstream surface | MLX consumer | Failure mode if it changes |
|---|---|---|
| `ArrayContainer` pytree field names / shapes | `mlx/bridge.py` | bridge `getattr` breaks loudly (good) or bridges the wrong array (bad) |
| grid `edges()` / `cell_widths()` API | `mlx/curl.py`, `mlx/metrics.py` | metric weights wrong → silent accuracy loss on graded grids |
| PML `kappa/sigma/alpha` profile (e.g. #372) | `mlx/pml.py` via bridged arrays | absorber profile wrong — but flows in *free* since MLX reads `arrays.kappa/sigma` |
| Yee staggering / `eta0`-normalized H | every MLX update kernel | element-wise parity break (caught by the gate) |
| source `_E/_H` precompute, detector `init_state` shapes | `mlx/source_freeze.py`, `mlx/detector_freeze.py` | shape mismatch (loud) or wrong injection/recording (silent) |
| placement / coordinate origin (#363) | examples, `Scene`, HDF5 contract | objects silently move — **not** caught by parity tests |
| forward feature set | `backend/dispatch.py:kernel_eligible` | a new upstream forward feature runs on MLX unsupported → wrong result instead of JAX fallback |

## Automation (optional, cheap)

- A scheduled job: `git fetch upstream`, attempt the `main`→`mlx-fork` merge on a throwaway branch,
  run the parity gate, report the delta + green/red. Turns "did upstream break us?" into a passive
  notification.
- A pre-merge grep over the contract surfaces (ArrayContainer fields, grid API signatures, PML
  profile) to flag a contract change *before* trusting the green checkmark.

## Known queue (as of fork-base `77e1281`, upstream tip `e5351a4`)

Upstream is 5 commits ahead; 3 are dependency/CI bumps (noise). The two that matter:

- **#372 — Fix nonuniform PML staggered profile.** *Merge now.* It corrects the JAX PML
  `kappa/sigma` profile on graded grids; since `mlx/bridge.py` bridges those arrays, the Metal path
  inherits the fix automatically. **The fork currently carries the pre-fix (buggy) code.** Add a
  graded-grid CPML parity test to lock it (rule 5).
- **#363 — Quasi-uniform grid + origin-at-center.** *Deliberate migration, not a routine merge.*
  Two things bundled: (a) a new `QuasiUniformGrid` policy (per-axis-uniform; trivially MLX-eligible
  since spacing is constant per axis — add to the gate or fall back), and (b) **coordinate origin
  moved to the domain center**, a breaking placement-semantics change. Before promoting: audit
  examples / `Scene` / HDF5 for lower-corner-origin assumptions; update placement-dependent parity
  tests; decide the `QuasiUniformGrid` dispatch.
</content>
</invoke>
