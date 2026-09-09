#!/usr/bin/env python3
"""Profile the *real* MLX engine loop — eager vs compiled x CPML on/off, with traffic reconciliation.

Replicates dispatch._run_mlx_forward's setup, then times run_forward_mlx with ``compile_step`` and
``simulate_boundaries`` toggled. Reports per-step time, throughput, and the *implied full-array
round-trips per step* at the measured ~240 GB/s coalesced roofline — if that number is large
(~tens), the bus is saturated on redundant traffic (H2), not idle.

    uv run python benchmarks/profile_engine.py --N 192 --steps 200 --material isotropic
"""

from __future__ import annotations

import argparse
import statistics
import time
from types import SimpleNamespace

import mlx.core as mx

ROOFLINE_GBS = 240.0  # measured coalesced copy bandwidth (profile_metal.py), not the 273 spec


def build_state(material, N, steps, detector="none"):
    """Build an MLXState + frozen plans exactly as dispatch._run_mlx_forward does."""
    import jax.numpy as jnp  # noqa: F401  (imported for side effect / parity with dispatch)
    from bench_forward import build_case

    from fdtdx.fdtd.update import get_wrap_padding_axes
    from fdtdx.mlx.bridge import to_mlx_state
    from fdtdx.mlx.detector_freeze import allocate_buffers, freeze_detectors
    from fdtdx.mlx.source_freeze import freeze_sources

    args = SimpleNamespace(spacing=50e-9, courant=0.99, pml=8, wavelength=1e-6, detector=detector, steps=steps)
    arrays, oc, config, _key, info = build_case(material, N, args)
    arrays = arrays.reset()
    periodic_axes = get_wrap_padding_axes(oc)
    state = to_mlx_state(arrays, config, periodic_axes)
    source_plans = freeze_sources(oc, config, arrays)
    detector_plans = freeze_detectors(oc, config)
    detector_buffers = allocate_buffers(detector_plans)
    c = float(config.courant_number)
    num_steps = int(config.time_steps_total)
    return state, source_plans, detector_plans, detector_buffers, c, num_steps, info


def time_loop(
    material, N, steps, simulate_boundaries, compile_step, repeats=3, use_metal_kernel=False, detector="none"
):
    from fdtdx.mlx.loop import run_forward_mlx

    state, sp, dp, db, c, num_steps, info = build_state(material, N, steps, detector)
    cells = info["grid_shape"][0] * info["grid_shape"][1] * info["grid_shape"][2]

    # warmup (build Metal kernels / trace the compiled graph)
    run_forward_mlx(
        state,
        sp,
        dp,
        db,
        num_steps,
        c,
        simulate_boundaries=simulate_boundaries,
        compile_step=compile_step,
        use_metal_kernel=use_metal_kernel,
    )
    mx.synchronize()
    mx.reset_peak_memory()

    times = []
    for _ in range(repeats):
        st, sp2, dp2, db2, c2, ns2, _ = build_state(material, N, steps, detector)
        mx.synchronize()
        t0 = time.perf_counter()
        run_forward_mlx(
            st,
            sp2,
            dp2,
            db2,
            ns2,
            c2,
            simulate_boundaries=simulate_boundaries,
            compile_step=compile_step,
            use_metal_kernel=use_metal_kernel,
        )
        mx.synchronize()
        times.append(time.perf_counter() - t0)

    peak = int(mx.get_peak_memory())
    med = statistics.median(times)
    per_step_ms = med / num_steps * 1000
    mcs = cells * num_steps / med / 1e6
    implied_rt = (per_step_ms / 1000) * ROOFLINE_GBS * 1e9 / (2 * 3 * cells * 4)
    return {"per_step_ms": per_step_ms, "mcs": mcs, "rt": implied_rt, "peak_gb": peak / 1e9}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=192)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--material", default="isotropic")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--kernel", action="store_true", help="also time the custom Metal-kernel path (M2)")
    p.add_argument(
        "--detector",
        default="none",
        choices=("none", "energy", "phasor"),
        help="monitor axis: time the per-step recording overhead (roadmap §3.2.1)",
    )
    args = p.parse_args()

    print(
        f"device: {mx.default_device()}   material={args.material}  N={args.N}  steps={args.steps}  detector={args.detector}"
    )
    print(f"roofline (measured coalesced copy): {ROOFLINE_GBS} GB/s\n")
    print(f"  {'config':26} {'per-step':>10} {'throughput':>13} {'RT/step':>9} {'peak':>9}")
    for comp in (False, True):
        for sb in (True, False):
            r = time_loop(args.material, args.N, args.steps, sb, comp, args.repeats, detector=args.detector)
            tag = f"{'compiled' if comp else 'eager':8} CPML {'on' if sb else 'off'}"
            print(f"  {tag:26} {r['per_step_ms']:8.2f}ms {r['mcs']:10.1f} Mcs/s {r['rt']:8.0f}  {r['peak_gb']:7.2f}GB")
    if args.kernel:
        for sb in (True, False):
            r = time_loop(
                args.material, args.N, args.steps, sb, True, args.repeats, use_metal_kernel=True, detector=args.detector
            )
            tag = f"metal-kernel CPML {'on' if sb else 'off'}"
            print(f"  {tag:26} {r['per_step_ms']:8.2f}ms {r['mcs']:10.1f} Mcs/s {r['rt']:8.0f}  {r['peak_gb']:7.2f}GB")
    print("\nRT/step = full (3,N^3) DRAM round-trips the per-step time equals at the 240 GB/s roofline.")
    print("(eager CPML-on is the pre-Phase-1 engine; compiled CPML-on is now the default path.)")


if __name__ == "__main__":
    main()
