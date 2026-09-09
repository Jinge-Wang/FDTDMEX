#!/usr/bin/env python3
"""Forward-engine performance benchmark: MLX (Metal) vs JAX-CPU.

Measures the wall-clock of the forward FDTD time loop (``run_fdtd``) across
(backend x material x domain-size) cells on one Apple-Silicon machine, forcing each
backend with ``fdtdx.use_backend``. JAX runs on CPU (the only usable JAX device on a
Mac); MLX runs on the Metal GPU. See ``docs/perf-eval-plan.md`` for the methodology.

The case is a cubic domain of side ``N`` (cells = N^3), uniformly filled with one of
three materials (isotropic / diagonal / full-tensor anisotropic), CPML on all sides, a
single point-dipole source, and (by default) no detector -- so the timed region is the
pure update loop. The step count is pinned (``time = steps * dt``, and dt is constant for
fixed spacing) so every cell runs the same number of steps.

Two run modes:
  * in-process (default): sweep all cells in one process. Fast; MLX peak memory is exact
    (reset per cell) but process RSS is a monotonic high-water mark (coarse for JAX).
  * ``--isolate``: run each cell in a fresh subprocess (re-invokes ``--single``). Slower
    (re-imports per cell) but gives a clean per-cell process-RSS peak for *both* backends
    and isolates OOM/crashes (a child dying does not kill the sweep). Recommended for the
    memory / max-domain-size scaling story.

Results are written as JSONL (one record per cell), flushed after every cell.

Usage:
    uv run python benchmarks/bench_forward.py \
        --backends mlx,jax --materials isotropic,diagonal,full_aniso \
        --sizes 32,48,64,96,128 --steps 200 --repeats 3

    uv run python benchmarks/bench_forward.py --isolate --sizes 64,128,192,256 ...

    # then plot:
    uv run python benchmarks/plot_results.py benchmarks/results/<file>.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Materials: uniform fill across the whole domain. The full-tensor case keeps a
# *modest* off-diagonal (0.5) -- large off-diagonals are numerically unstable in the
# current explicit anisotropic update (see roadmap "Quirk A"), which would NaN the run
# and pollute the timing.
# ---------------------------------------------------------------------------
MATERIALS = {
    "isotropic": dict(permittivity=2.25),
    "diagonal": dict(permittivity=(2.0, 3.0, 4.0)),
    "full_aniso": dict(permittivity=((2.5, 0.5, 0.0), (0.5, 3.0, 0.0), (0.0, 0.0, 4.0))),
    "iso_conductive": dict(permittivity=2.25, electric_conductivity=0.05),
}

SENTINEL = "@@RECORD@@ "  # child (--single) prints exactly one record line with this prefix


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _sysctl(key: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def collect_metadata(args) -> dict:
    import jax
    import mlx.core as mx

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "chip": _sysctl("machdep.cpu.brand_string"),
        "ram_bytes": int(_sysctl("hw.memsize") or 0),
        "python": sys.version.split()[0],
        "mlx_version": mx.__version__,
        "jax_version": jax.__version__,
        "mlx_default_device": str(mx.default_device()),
        "jax_devices": [str(d) for d in jax.devices()],
        "ru_maxrss_unit": "bytes" if sys.platform == "darwin" else "kib",
        "dtype": "float32",
        "spacing_m": args.spacing,
        "wavelength_m": args.wavelength,
        "pml": args.pml,
        "courant_factor": args.courant,
        "steps_requested": args.steps,
        "repeats": args.repeats,
        "detector": args.detector,
    }


# ---------------------------------------------------------------------------
# Case construction (mirrors tests/validation/test_mlx_parity.py).
# ---------------------------------------------------------------------------
def _dt_for(spacing: float, courant: float):
    """Time-step duration for a domain at this spacing (independent of N / material)."""
    import jax.numpy as jnp

    import fdtdx

    cfg = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=spacing), time=1e-15, dtype=jnp.float32, courant_factor=courant
    )
    return float(cfg.time_step_duration)


def build_case(material: str, n: int, args):
    """Place a cubic, uniformly-filled domain; return (arrays, oc, config, key, info)."""
    import jax
    import jax.numpy as jnp

    import fdtdx

    dt = _dt_for(args.spacing, args.courant)
    sim_time = args.steps * dt  # pins the step count (dt is constant for fixed spacing)
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=args.spacing), time=sim_time, dtype=jnp.float32, courant_factor=args.courant
    )

    objects, constraints = [], []
    mat = fdtdx.Material(**MATERIALS[material])
    vol = fdtdx.SimulationVolume(partial_real_shape=(n * args.spacing,) * 3, material=mat)
    objects.append(vol)

    bdict, clist = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=args.pml), vol)
    constraints.extend(clist)
    objects.extend(bdict.values())

    src = fdtdx.PointDipoleSource(
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=args.wavelength),
        polarization=2,
        amplitude=1.0,
    )
    constraints.append(src.place_at_center(vol, axes=(0, 1, 2)))
    objects.append(src)

    if args.detector == "energy":
        det = fdtdx.EnergyDetector(name="energy", reduce_volume=True, plot=False)
        constraints.extend([det.same_size(vol, axes=(0, 1, 2)), det.place_at_center(vol, axes=(0, 1, 2))])
        objects.append(det)
    elif args.detector == "phasor":
        # A full in-plane (None, None, 1) phasor slice — the recording-overhead axis from the
        # performance roadmap (§3.2.1): exercises the per-step region interpolation + DFT accumulate
        # that dominate monitored runs, on the same grid as the bulk-kernel RT measurement.
        det = fdtdx.PhasorDetector(
            name="phasor",
            wave_characters=(fdtdx.WaveCharacter(wavelength=args.wavelength),),
            components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
            partial_grid_shape=(None, None, 1),
            reduce_volume=False,
            plot=False,
        )
        constraints.extend(
            [
                det.same_size(vol, axes=(0, 1)),
                det.place_at_center(vol, axes=(0, 1)),
                det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(n // 2,)),
            ]
        )
        objects.append(det)

    key = jax.random.PRNGKey(0)
    oc, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, oc, _ = fdtdx.apply_params(arrays, oc, params, key)

    info = {
        "steps_actual": int(config.time_steps_total),
        "dt_s": dt,
        "eps_components": int(getattr(arrays.inv_permittivities, "shape", [1])[0])
        if getattr(arrays.inv_permittivities, "ndim", 0) > 0
        else 1,
        "grid_shape": [int(x) for x in arrays.fields.E.shape[1:]],
    }
    return arrays, oc, config, key, info


# ---------------------------------------------------------------------------
# Timing one cell.
# ---------------------------------------------------------------------------
def _force_sync(backend: str, out):
    """Block until the forward result is fully materialized on host."""
    import numpy as np

    # Bridge-out already converts MLX->host; np.asarray blocks JAX-CPU until ready.
    np.asarray(out.fields.E)
    np.asarray(out.fields.H)
    if backend == "mlx":
        import mlx.core as mx

        mx.synchronize()


def time_cell(material: str, n: int, backend: str, args) -> dict:
    """Build, warm up, and time ``repeats`` forward runs of one cell."""
    import mlx.core as mx

    import fdtdx

    rec = {"backend": backend, "material": material, "N": n, "cells": n**3, "status": "ok"}

    arrays, oc, config, key, info = build_case(material, n, args)
    rec.update(info)
    rec["cells"] = info["grid_shape"][0] * info["grid_shape"][1] * info["grid_shape"][2]

    with fdtdx.use_backend(backend):
        # Warmup (first call traces/compiles for JAX; builds Metal kernels for MLX).
        _, out = fdtdx.run_fdtd(arrays=arrays, objects=oc, config=config, key=key, show_progress=False)
        _force_sync(backend, out)

        if backend == "mlx":
            mx.reset_peak_memory()

        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            _, out = fdtdx.run_fdtd(arrays=arrays, objects=oc, config=config, key=key, show_progress=False)
            _force_sync(backend, out)
            times.append(time.perf_counter() - t0)

        if backend == "mlx":
            rec["mlx_peak_bytes"] = int(mx.get_peak_memory())
            rec["mlx_active_bytes"] = int(mx.get_active_memory())

    med = statistics.median(times)
    rec["time_s_median"] = med
    rec["time_s_min"] = min(times)
    rec["time_s_all"] = times
    steps = info["steps_actual"]
    rec["throughput_mcellsteps_s"] = (rec["cells"] * steps / med) / 1e6 if med > 0 else 0.0
    rec["ru_maxrss_bytes"] = _ru_maxrss_bytes()

    # Check the field didn't blow up (NaN -> unstable case, throughput is meaningless).
    import numpy as np

    rec["finite"] = bool(np.isfinite(np.asarray(out.fields.E)).all())
    if not rec["finite"]:
        rec["status"] = "nonfinite"

    if backend == "mlx":
        mx.clear_cache()
    return rec


def _ru_maxrss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss) if sys.platform == "darwin" else int(rss) * 1024  # macOS: bytes; Linux: KiB


# ---------------------------------------------------------------------------
# Sweep drivers.
# ---------------------------------------------------------------------------
def run_isolated_cell(material: str, n: int, backend: str, args) -> dict:
    """Run one cell in a fresh subprocess; parse its emitted record."""
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--single",
        f"{backend}:{material}:{n}",
        "--steps",
        str(args.steps),
        "--repeats",
        str(args.repeats),
        "--pml",
        str(args.pml),
        "--spacing",
        repr(args.spacing),
        "--wavelength",
        repr(args.wavelength),
        "--courant",
        repr(args.courant),
        "--detector",
        args.detector,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.cell_timeout)
    except subprocess.TimeoutExpired:
        return {"backend": backend, "material": material, "N": n, "cells": n**3, "status": "timeout"}
    if proc.returncode != 0:
        for line in proc.stdout.splitlines():
            if line.startswith(SENTINEL):
                return json.loads(line[len(SENTINEL) :])
        return {
            "backend": backend,
            "material": material,
            "N": n,
            "cells": n**3,
            "status": "crash",
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-500:],
        }
    for line in proc.stdout.splitlines():
        if line.startswith(SENTINEL):
            return json.loads(line[len(SENTINEL) :])
    return {"backend": backend, "material": material, "N": n, "cells": n**3, "status": "no_record"}


def sweep(args):
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        f"forward_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    meta = collect_metadata(args)
    print("=" * 78)
    print(f"FDTDMEX forward benchmark   commit {meta['git_commit']}   {meta['timestamp']}")
    print(f"  chip: {meta['chip'] or '?'}   RAM: {meta['ram_bytes'] / 1e9:.0f} GB")
    print(f"  MLX  {meta['mlx_version']}  default device: {meta['mlx_default_device']}")
    print(f"  JAX  {meta['jax_version']}  devices: {meta['jax_devices']}")
    print(f"  mode: {'isolated subprocess/cell' if args.isolate else 'in-process'}   out: {out_path}")
    print("=" * 78)
    # Hard confirmation of the device split the user asked for.
    assert "gpu" in meta["mlx_default_device"].lower(), f"MLX not on Metal GPU: {meta['mlx_default_device']}"
    assert any("cpu" in d.lower() for d in meta["jax_devices"]), f"JAX not on CPU: {meta['jax_devices']}"
    print("  device check OK: MLX -> Metal GPU, JAX -> CPU\n")

    with open(out_path, "w") as f:
        f.write(json.dumps({"record_type": "metadata", **meta}) + "\n")
        f.flush()

        print(
            f"{'backend':7} {'material':14} {'N':>4} {'cells':>11} {'steps':>5} "
            f"{'med_s':>8} {'Mcell·st/s':>11} {'peakMB':>8} {'status'}"
        )
        print("-" * 86)
        for material in args.materials:
            for n in args.sizes:
                for backend in args.backends:
                    if args.isolate:
                        rec = run_isolated_cell(material, n, backend, args)
                    else:
                        try:
                            rec = time_cell(material, n, backend, args)
                        except Exception as e:  # OOM, NaN-in-construction, etc.
                            rec = {
                                "backend": backend,
                                "material": material,
                                "N": n,
                                "cells": n**3,
                                "status": "error",
                                "error": f"{type(e).__name__}: {e}",
                            }
                    rec["record_type"] = "cell"
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    peak_mb = rec.get("mlx_peak_bytes") or rec.get("ru_maxrss_bytes") or 0
                    print(
                        f"{rec['backend']:7} {rec['material']:14} {rec['N']:>4} {rec.get('cells', 0):>11} "
                        f"{rec.get('steps_actual', 0):>5} {rec.get('time_s_median', 0):>8.4f} "
                        f"{rec.get('throughput_mcellsteps_s', 0):>11.2f} {peak_mb / 1e6:>8.1f} {rec['status']}"
                    )
    print(f"\nDone. Results: {out_path}")
    print(f"Plot:  uv run python benchmarks/plot_results.py {out_path}")
    return out_path


def single(args):
    """Run exactly one cell (used by --isolate subprocesses, also usable directly)."""
    backend, material, n = args.single.split(":")
    n = int(n)
    try:
        rec = time_cell(material, n, backend, args)
    except Exception as e:
        rec = {
            "backend": backend,
            "material": material,
            "N": n,
            "cells": n**3,
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
        }
    rec["ru_maxrss_bytes"] = _ru_maxrss_bytes()
    print(SENTINEL + json.dumps(rec))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backends", default="mlx,jax", help="comma list: mlx,jax")
    p.add_argument("--materials", default="isotropic,diagonal,full_aniso", help=f"comma list from {list(MATERIALS)}")
    p.add_argument("--sizes", default="32,48,64,96,128", help="comma list of cubic side lengths N (cells = N^3)")
    p.add_argument("--steps", type=int, default=200, help="fixed number of time steps per run")
    p.add_argument("--repeats", type=int, default=3, help="timed repeats (median reported), after 1 warmup")
    p.add_argument("--pml", type=int, default=8, help="CPML thickness (cells, all sides)")
    p.add_argument("--spacing", type=float, default=50e-9, help="uniform grid spacing (m)")
    p.add_argument("--wavelength", type=float, default=1e-6, help="source wavelength (m)")
    p.add_argument("--courant", type=float, default=0.99, help="Courant factor")
    p.add_argument(
        "--detector",
        choices=["none", "energy"],
        default="none",
        help="'none' times the pure loop; 'energy' adds one reduce-volume EnergyDetector",
    )
    p.add_argument("--isolate", action="store_true", help="run each cell in a fresh subprocess (clean per-cell RSS)")
    p.add_argument("--cell-timeout", type=float, default=1800.0, help="per-cell subprocess timeout (s), --isolate only")
    p.add_argument("--out", default=None, help="output JSONL path (default benchmarks/results/forward_<ts>.jsonl)")
    p.add_argument("--single", default=None, help=argparse.SUPPRESS)  # internal: 'backend:material:N'
    args = p.parse_args()
    args.backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    args.materials = [m.strip() for m in args.materials.split(",") if m.strip()]
    if args.single is None:
        args.sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    for m in args.materials:
        if m not in MATERIALS:
            p.error(f"unknown material {m!r}; choose from {list(MATERIALS)}")
    return args


def main():
    args = parse_args()
    if args.single is not None:
        single(args)
    else:
        sweep(args)


if __name__ == "__main__":
    main()
