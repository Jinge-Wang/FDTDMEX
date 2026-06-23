# benchmarks/

Forward-engine performance harness: **MLX (Metal GPU) vs JAX (CPU)** on one Apple-Silicon
machine. See [`docs/performance.md`](../docs/performance.md) for the roofline, the round-trip (RT)
model, current results + history, and [`docs/phase2-metal-kernels.md`](../docs/phase2-metal-kernels.md)
for the custom-Metal-kernel design and staging.

## Run

```bash
# headline scaling sweep (both backends, 3 materials, representative step count)
uv run python benchmarks/bench_forward.py \
    --backends mlx,jax --materials isotropic,diagonal,full_aniso \
    --sizes 32,48,64,96,128,160,192 --steps 250 --repeats 2 \
    --out benchmarks/results/forward.jsonl

# then plot (4-panel: throughput, wall-clock, MLX/JAX speedup, peak memory)
uv run python benchmarks/plot_results.py benchmarks/results/forward.jsonl
```

Results land in `benchmarks/results/<file>.jsonl` (one JSON record per cell, flushed after
each cell; first line is a metadata record with chip / RAM / mlx & jax versions / device
split / git commit). Figures land where `--out` points (committed copies live in `benchmarks/figures/`).

### Useful flags

- `--steps N` fixed time-step count per run (the harness pins `time = N·dt`; dt is constant for
  fixed spacing, so every cell runs the same number of steps). **Use a representative count
  (≥200)** — at very short runs the MLX per-call bridge cost is not amortized.
- `--sizes` cubic side lengths `N` (cells = N³). The CLI lets you cap the max `N` to your machine.
- `--detector none|energy` — `none` (default) times the pure update loop; `energy` adds one
  reduce-volume `EnergyDetector`.
- `--isolate` runs **each cell in a fresh subprocess**: slower (re-imports per cell) but gives a
  clean per-cell process-RSS peak for *both* backends and isolates OOM/crashes (a child dying does
  not kill the sweep). **Recommended for the memory / max-domain-size run** — in the default
  in-process mode, `ru_maxrss` is a monotonic high-water mark (coarse for JAX); MLX peak memory
  (`mx.get_peak_memory`) is exact in either mode.
- `--single backend:material:N` runs exactly one cell (used internally by `--isolate`; also handy
  for a one-off).

## What it measures

Per `(backend, material, size)` cell: median wall-clock of `run_fdtd` over `--repeats` timed runs
(after 1 warmup), throughput in **Mcell·steps/s** = `cells·steps/seconds`, MLX exact peak/active
GPU memory, and process RSS. The case is a cubic domain uniformly filled with one material
(`isotropic` / `diagonal` / `full_aniso` (9-tensor) / `iso_conductive`), CPML on all sides, a
point-dipole source, no detector by default. Backends are forced with `fdtdx.use_backend(...)`;
the harness asserts **MLX → Metal GPU** and **JAX → CPU** at startup and records both device lists.

## Profiling (where the time / memory / bandwidth goes — MLX only)

These dissect *why* the MLX engine performs as it does (see [`docs/performance.md`](../docs/performance.md)):

- **`profile_engine.py`** — times the *real* engine loop in a 2×2 of `mx.compile` × CPML (and, with
  `--kernel`, the M2 custom-Metal-kernel path), and reports the implied **DRAM round-trips per step**
  at the measured 240 GB/s roofline. This is the per-fix audit: each fix must drop RT/step by its
  predicted amount.
- **`profile_metal.py`** — measures achieved bandwidth from *known* traffic (coalesced copy vs
  strided roll; component-leading vs component-last layout) and an eager-vs-compiled / eval-frequency
  probe. Establishes the roofline denominator.
- **`profile_memory.py`** — one `(backend, N)` per fresh subprocess for a clean peak-memory
  high-water mark (loop a driver over N to find where each backend hits the memory wall).

```bash
uv run python benchmarks/profile_engine.py --N 192 --steps 200 --material isotropic
uv run python benchmarks/profile_metal.py  --N 192 --iters 100
```
