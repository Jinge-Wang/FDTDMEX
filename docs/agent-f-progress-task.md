# Agent F task — fine-grained run progress during the Metal solve

## Why

ag-fdtd shows a live telemetry bar for a detached solver run. Right now
`agentic_adapter/real_solver.py` emits `PROGRESS` at only **4 milestones** (scene-build,
before `sim_run`, after `sim_run`, write). The actual **6–7 min Metal FDTD solve happens
entirely between two of those ticks**, so the bar sits frozen at ~50% for the whole solve
then jumps. Goal: stream progress **during `sim_run`** so the bar advances smoothly across
the multi-minute run.

This is solver-side only — **do not touch the engine's numerics**, just surface progress.

## The contract (already honored on the ag-fdtd side)

ag-fdtd parses stdout lines of the form:

```
PROGRESS <i>/<N>
```

- one per line, `flush=True`, `i` monotonically non-decreasing from 1 up to `N`.
- **`N` is the denominator and ag-fdtd now ADOPTS it as the bar's total** (it no longer
  assumes `N == --steps`). So you choose `N` at whatever resolution you like — make it the
  real timestep count, or a capped resolution like 200. (ag-fdtd change:
  `backend/app/jobs/local.py::_run` reads both `i` and `N`; the mock still emits `i/steps`,
  so this is backward-compatible.)
- The **bulk of the increments must occur during `sim_run`** (the Metal solve), not just at
  the milestones. End at (or near) `N/N`; ag-fdtd's terminal `completed` frame pins 100%.

## What to implement

1. Add an **optional progress callback** to `fdtdmex.io.sim_run` (and thread it into the
   Metal time-stepping loop), e.g. `sim_run(config, results, *, backend, progress=None)`
   where `progress(step:int, total:int)` is called periodically from the loop. Default
   `None` = current behavior (no API break for other callers).
2. In `real_solver.py`, pass a callback that prints `PROGRESS {step}/{total}` (reuse the
   existing `_progress`/flush). Map the solve onto the progress budget — simplest is to
   forward the loop's `(step, total_timesteps)` directly as `N = total_timesteps`. Keep a
   couple of milestone ticks for scene-build/postproc if you like; just stay monotonic.
3. **Throttle** the cadence so you emit ~100–200 lines max over the run (e.g. every
   `max(1, total//200)` timesteps), not one per timestep — avoid flooding stdout.
4. The **`mock` backend** can emit a handful of synthetic ticks too (nice for offline UX),
   but the real target is `mlx`. Don't change the CLI contract
   (`--bundle/--out-dir/--run-id/--steps/--domain/--solver/--backend/[--fail-mode]`) or the
   3-file output contract.

## Don't break

- The argparse CLI + the 3 contract files (`{id}_result.hdf5` / `_summary.json` /
  `_preview.json`) stay exactly as they are.
- `--steps` keeps its current meaning to the adapter (spectrum sample count); progress `N`
  is independent of it now.
- ag-fdtd's guarded test must still pass:
  `FDTDMEX_PYTHON=… FDTDMEX_REPO=… uv run pytest tests/test_real_solver.py` (engine=mock).

## Quick verify

```bash
# from the FDTDMEX repo — a mock run should now stream many PROGRESS lines:
uv run python agentic_adapter/real_solver.py \
  --bundle <any sim bundle .hdf5> --out-dir /tmp/run1 --run-id r1 \
  --steps 11 --domain fdtd --solver fdtdmex --backend mock | grep -c PROGRESS
# expect ≫ 4 (was 4). Then an mlx run should advance the bar steadily over the solve.
```
