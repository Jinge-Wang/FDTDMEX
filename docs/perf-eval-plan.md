# Action Plan — Forward-engine performance evaluation (WS-A baseline)

> **For the agent picking this up in a fresh session.** Read this top-to-bottom first, then read
> [CLAUDE.md](../CLAUDE.md) and [roadmap.md](roadmap.md). This is a *planning* doc; the code it
> describes does not exist yet — your first job is to **write the benchmark harness, then stop**
> (see "Workflow constraint"). The benchmark is run by the human, not by you.

---

## 1. Background (cold-start orientation)

FDTDMEX is a **git fork of [fdtdx](https://github.com/ymahlau/fdtdx)** (a JAX FDTD Maxwell solver)
that adds a native **MLX (Metal) forward backend** for Apple Silicon. On a Mac, a supported
forward `run_fdtd` auto-routes to the MLX engine; everything else (gradients, unsupported
features, non-Apple platforms) runs the unchanged JAX engine. Import name stays `fdtdx`.

- **WS-A (the forward MLX engine) is complete and validated through M1–M4**: iso / diagonal /
  full-tensor (9-component) anisotropic materials, electric/magnetic conductivity, CPML + periodic
  boundaries, dipole + (tilted) TFSF plane sources, all four detector types, and **non-uniform
  (rectilinear) grids** with spacing-weighted operators. Every surface is checked element-wise vs
  the JAX-CPU oracle (`tests/validation/`).
- **The engine is EAGER.** The time loop ([`src/fdtdx/mlx/loop.py`](../src/fdtdx/mlx/loop.py)) is a
  plain Python `for` over steps with a periodic `mx.eval` to bound the lazy graph. **`mx.compile`
  of the per-step body is NOT done** — it is the main forward-perf lever and the next optimization
  after this baseline.
- **The thesis this phase tests.** The project's bet is *memory, not just compute*: a full-tensor
  anisotropic sim stores a 3×3 ε tensor per voxel (~9× the isotropic footprint), which saturates a
  single CUDA GPU's VRAM, whereas Apple **unified memory** (up to 512 GB) lets the GPU address the
  whole domain with no host↔device streaming. We need numbers that show (a) how MLX/Metal forward
  throughput compares to JAX-CPU, and (b) how far the domain can scale, *especially* for the
  full-tensor anisotropic case.

Key files: dispatch seam [`src/fdtdx/backend/dispatch.py`](../src/fdtdx/backend/dispatch.py);
bridge [`src/fdtdx/mlx/bridge.py`](../src/fdtdx/mlx/bridge.py); loop
[`src/fdtdx/mlx/loop.py`](../src/fdtdx/mlx/loop.py). Backend forcing: `with fdtdx.use_backend("mlx"|"jax")`
or `FDTDMEX_BACKEND=mlx|jax`. The JAX oracle runs on CPU (conftest pins `JAX_PLATFORMS=cpu`;
JAX-Metal is unusable).

## 2. Goal of this phase

Establish a **performance baseline of the current (eager) MLX forward engine vs the JAX-CPU
oracle**, across problem sizes and material types — headlined by large full-tensor anisotropic
domains. This baseline (a) quantifies the eager engine's standing and (b) is the control against
which the later `mx.compile` optimization is measured. **This is a measurement task, not an
optimization task** — do not change engine code in this phase.

## 3. Workflow constraint (IMPORTANT — read twice)

The human runs the benchmark, not the agent. Concretely:

1. **You (this session): write the benchmark harness** (`benchmarks/bench_forward.py` + any helper)
   and a short usage note. Make it self-contained and runnable. **Then STOP and hand back** — do
   **not** execute the benchmark, do not start long runs, do not `uv run` it beyond a tiny
   smoke-check of `--help`/argument parsing (≤ a few seconds, smallest possible size, if at all).
2. **The human runs it** on their Mac (they control the max domain size and total wall time) and
   saves the results file(s).
3. **You (a later session): resume and analyze** the results the human provides — see §6.

Rationale: Metal timing must run on the user's machine under their control; large sweeps can take
many minutes and shouldn't be agent-driven; the user wants to inspect/scale runs interactively.

## 4. What the harness must measure

For each (backend, material, size) cell, run a **fixed number of time steps** on an otherwise
identical case and record:

- **wall-clock** of the forward loop (seconds), median of N timed repeats after a warmup repeat;
- **throughput** in **Mcell·steps/s** = `Nx·Ny·Nz · steps / seconds`;
- **peak memory**: for MLX use `mx.get_peak_memory()` (reset with `mx.reset_peak_memory()` per
  case); for both backends also capture process RSS (e.g. `resource.getrusage(...).ru_maxrss` or
  `psutil`). Note `ru_maxrss` is bytes on macOS, KiB on Linux — record platform.

Sweep axes:

- **backend**: `mlx` (Metal) and `jax` (CPU), forced via `fdtdx.use_backend(...)`.
- **material**: `isotropic` (baseline), `diagonal` (3-tensor), `full_aniso` (9-tensor — the
  headline; ~9× material memory). Optionally `iso_conductive`.
- **size**: a geometric sweep of cubic domains (e.g. N ∈ {32, 48, 64, 96, 128, 192, 256, …})
  scaling up until the user's machine limits — the CLI must let the user cap the max N. Report
  cells = N³.

Hold fixed across cells: dtype float32, Courant factor, time-step count, a simple source (point
dipole or a plane source) + at most one cheap detector (or none — measure the pure loop). Use a
fixed seed. Keep PML thickness modest and constant.

## 5. Methodology / pitfalls (bake these into the harness)

- **MLX is lazy.** Timing is meaningless unless you force evaluation: call `mx.eval(...)` on the
  outputs (or `mx.synchronize()`) **inside the timed region's end**, and do a **warmup run** first
  (the first run includes graph build + Metal kernel compilation). `run_fdtd` already ends with an
  `mx.eval`, but confirm and, if needed, `mx.synchronize()` before stopping the timer.
- **JAX warmup** similarly — the first call traces/compiles; discard it.
- **Measure the forward loop, not setup.** `place_objects` / `apply_params` / the array bridge are
  one-time costs; exclude them from the timed region (time `run_fdtd` itself, and optionally report
  bridge time separately). Build + place the case once per cell, then time repeated `run_fdtd`
  calls (each `reset`s internally).
- **Memory hygiene**: `mx.reset_peak_memory()` before each MLX case; consider `mx.clear_cache()`
  between cases; record `mx.get_peak_memory()` and `mx.get_active_memory()`.
- **Per-step overhead vs kernel time** is the key diagnostic for "does `mx.compile` help": at small
  N the eager Python/`mx.eval` per-step overhead dominates; at large N kernel time dominates.
  Sweeping N exposes this crossover — make sure small N is included.
- **Don't let one cell hang the sweep**: wrap each case in try/except, record failures (e.g.
  out-of-memory at large N is itself a useful data point — note which backend/size OOMs), and keep
  going. Flush results to disk **after each cell** so a crash doesn't lose the run.
- **Determinism / fairness**: identical placed geometry for both backends; same step count; report
  device info (chip, RAM), mlx version, jax version, dtype, git commit.

## 6. Deliverable (this session)

- **`benchmarks/bench_forward.py`** — a CLI. Suggested flags: `--backends mlx,jax`,
  `--materials isotropic,diagonal,full_aniso`, `--sizes 32,64,96,128`, `--steps 200`, `--repeats 3`,
  `--out benchmarks/results/<timestamp>.json`. Writes one JSON (or JSONL) record per cell with all
  fields from §4 plus metadata from §5, flushed incrementally.
- Update **`benchmarks/README.md`** with the one-line run command and where results land.
- Keep it dependency-light (stdlib `time`, `json`, `argparse`, `platform`, `resource`; `psutil`
  only if already available — otherwise fall back to `ru_maxrss`). It builds cases with the public
  `fdtdx` API exactly like the validation tests in `tests/validation/test_mlx_parity.py` (reuse
  that construction pattern).

Then **stop** and tell the user how to run it.

## 7. Analysis phase (later session, once results exist)

When the human returns with a results file:

1. Load it; produce a compact table + a couple of plots (throughput vs N per backend/material;
   peak memory vs N; MLX/JAX speedup vs N). Save figures under `outputs/` and commit a render under
   `docs/images/` (follow the `tests/visualization/` pattern).
2. Answer the load-bearing questions: **MLX-vs-JAX-CPU crossover size**; whether **eager per-step
   overhead dominates at the sizes of interest** (→ how much `mx.compile` could buy); **how the
   full-tensor anisotropic case scales** vs isotropic in time *and* memory; the **largest domain
   that fit** on the machine (the unified-memory claim).
3. Write a short findings note (e.g. `docs/perf-baseline.md`) and update the roadmap's WS-A "next"
   bullet with the measured motivation for the `mx.compile` pass.

## 8. After the baseline (preview, not this phase)

- **`mx.compile` the per-step body**: pass `time_step` + amplitude scalars as compiled *arguments*;
  keep source/detector gating host-side (skip inactive ones outside the compiled core); re-run this
  same benchmark and quantify the speedup against the baseline.
- Optional **CUDA comparison** for the anisotropic case on a Linux/JAX box (the inverse-design
  target hardware), to frame the unified-memory argument quantitatively.
