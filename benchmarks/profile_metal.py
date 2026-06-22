#!/usr/bin/env python3
"""Metal GPU bottleneck profiler — *measure* why the MLX FDTD step misses the roofline.

The perf-baseline §1a story ("~3% of 273 GB/s", "~4.3× headroom") is inferred from a
wall-clock decomposition; nobody measured the GPU. This script measures the things that
were assumed:

  * roofline   — the *achieved* sustained DRAM bandwidth for coalesced (copy) vs strided
                 (roll along the big-stride axis) access, and the component-leading
                 (3,N,N,N) vs component-last (N,N,N,3) stencil — the denominator the docs
                 lack (scores H3 = uncoalesced-bound).
  * dispatch   — eager vs mx.compile on the same math, and an eval-frequency sweep, to see
                 whether the CPU-side kernel encode / sync is on the critical path (H1).
  * capture    — optional .gputrace (MTL_CAPTURE_ENABLED=1) for offline Xcode inspection.

Bandwidth is computed from *known* traffic: a copy/roll of an M-byte array moves 2M bytes
(read+write), so GB/s = 2*M*iters/time is a real measurement, not a per-cell bytes guess.

    uv run python benchmarks/profile_metal.py --N 192 --iters 100
    MTL_CAPTURE_ENABLED=1 uv run python benchmarks/profile_metal.py --capture
"""

from __future__ import annotations

import argparse
import os
import time

import mlx.core as mx

F32 = mx.float32
BYTES = 4
PEAK_GBS = 273.0  # published M4 Pro unified-memory bandwidth (device_info has no BW field)


def _sync():
    mx.eval(mx.array(0))  # cheap barrier
    mx.synchronize()


def _time_chain(make, op, iters, warmup=5):
    """Time `op` applied as a data-dependent chain `x = op(x)` so it cannot be elided/fused away."""
    x = make()
    for _ in range(warmup):
        x = op(x)
    mx.eval(x)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        x = op(x)
    mx.eval(x)
    mx.synchronize()
    return time.perf_counter() - t0


def device_report():
    info = mx.device_info()
    print("=== device ===")
    for k in ("device_name", "architecture", "memory_size", "max_recommended_working_set_size", "max_buffer_length"):
        print(f"  {k:36} {info.get(k)}")
    print(f"  assumed peak bandwidth (spec)          {PEAK_GBS} GB/s")
    print()


def roofline(N, iters):
    """Achieved sustained bandwidth for known-traffic ops. Scores H3 (coalescing/layout)."""
    print(f"=== roofline @ N={N} (cells={N**3:,}), iters={iters} ===")
    arr_bytes = 3 * N**3 * BYTES  # a (3,N,N,N) float32 field
    two_way = 2 * arr_bytes * iters  # read + write

    def field():
        return mx.random.normal((3, N, N, N)).astype(F32)

    probes = {
        "copy (x+1)                 [coalesced ref]": lambda x: x + 1.0,
        "roll axis=3 (z, stride 1)  [coalesced]": lambda x: mx.roll(x, 1, axis=3),
        "roll axis=2 (y, stride N)  [mid]": lambda x: mx.roll(x, 1, axis=2),
        "roll axis=1 (x, stride N^2)[strided]": lambda x: mx.roll(x, 1, axis=1),
    }
    print(f"  {'op':44} {'time':>8} {'GB/s':>9}  {'%peak':>6}")
    copy_gbs = None
    for name, op in probes.items():
        dt = _time_chain(field, op, iters)
        gbs = two_way / dt / 1e9
        if copy_gbs is None:
            copy_gbs = gbs
        print(f"  {name:44} {dt:7.3f}s {gbs:8.1f} {100 * gbs / PEAK_GBS:5.0f}%")
    print(f"  -> coalesced copy achieves {copy_gbs:.0f} GB/s = {100 * copy_gbs / PEAK_GBS:.0f}% of the {PEAK_GBS} spec")
    print()

    # Layout experiment: a 6-neighbor stencil sum in component-leading vs component-last.
    print(f"  --- stencil6 layout (3,N,N,N) vs (N,N,N,3) @ N={N} ---")

    def stencil_cl(x):  # component-leading: spatial axes are 1,2,3
        return (
            mx.roll(x, 1, axis=1)
            + mx.roll(x, -1, axis=1)
            + mx.roll(x, 1, axis=2)
            + mx.roll(x, -1, axis=2)
            + mx.roll(x, 1, axis=3)
            + mx.roll(x, -1, axis=3)
        )

    def stencil_clast(x):  # component-last: spatial axes are 0,1,2
        return (
            mx.roll(x, 1, axis=0)
            + mx.roll(x, -1, axis=0)
            + mx.roll(x, 1, axis=1)
            + mx.roll(x, -1, axis=1)
            + mx.roll(x, 1, axis=2)
            + mx.roll(x, -1, axis=2)
        )

    dt_cl = _time_chain(lambda: mx.random.normal((3, N, N, N)).astype(F32), stencil_cl, iters)
    dt_clast = _time_chain(lambda: mx.random.normal((N, N, N, 3)).astype(F32), stencil_clast, iters)
    mcs_cl = N**3 * iters / dt_cl / 1e6
    mcs_clast = N**3 * iters / dt_clast / 1e6
    print(f"  component-leading (3,N,N,N): {dt_cl:7.3f}s  {mcs_cl:8.1f} Mcell/s")
    print(f"  component-last    (N,N,N,3): {dt_clast:7.3f}s  {mcs_clast:8.1f} Mcell/s")
    print(f"  -> layout speedup (clast/cl): {dt_cl / dt_clast:.2f}x  (>1 means component-last is faster)")
    print()


def dispatch_probe(N, iters):
    """Eager vs compiled, and an eval-frequency sweep. Scores H1 (dispatch/encode-bound)."""
    print(f"=== dispatch / fusion probe @ N={N}, iters={iters} ===")

    def stencil(x):
        return (
            mx.roll(x, 1, axis=1)
            + mx.roll(x, -1, axis=1)
            + mx.roll(x, 1, axis=2)
            + mx.roll(x, -1, axis=2)
            + mx.roll(x, 1, axis=3)
            + mx.roll(x, -1, axis=3)
        ) * 0.5

    # eager vs compiled (same math; compiled collapses ~7 kernels -> ~1)
    dt_e = _time_chain(lambda: mx.random.normal((3, N, N, N)).astype(F32), stencil, iters)
    cstencil = mx.compile(stencil)
    dt_c = _time_chain(lambda: mx.random.normal((3, N, N, N)).astype(F32), cstencil, iters)
    print(f"  eager stencil6:    {dt_e:7.3f}s  {N**3 * iters / dt_e / 1e6:8.1f} Mcell/s")
    print(f"  compiled stencil6: {dt_c:7.3f}s  {N**3 * iters / dt_c / 1e6:8.1f} Mcell/s")
    print(f"  -> fusion speedup: {dt_e / dt_c:.2f}x  (large => dispatch/encode-bound; ~1 => GPU-work-bound)")

    # eval-frequency sweep: how much does forcing a CPU<->GPU sync every k steps cost?
    print("  eval-frequency sweep (copy chain):")
    for every in (1, 4, 16, iters):
        x = mx.random.normal((3, N, N, N)).astype(F32)
        for _ in range(5):
            x = x + 1.0
        mx.eval(x)
        mx.synchronize()
        t0 = time.perf_counter()
        for i in range(iters):
            x = x + 1.0
            if (i + 1) % every == 0:
                mx.eval(x)
        mx.eval(x)
        mx.synchronize()
        dt = time.perf_counter() - t0
        tag = "once" if every == iters else f"every {every}"
        print(f"    eval {tag:9}: {dt:7.3f}s  {N**3 * iters / dt / 1e6:8.1f} Mcell/s")
    print()


def maybe_capture(N):
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        print("(set MTL_CAPTURE_ENABLED=1 to also emit a .gputrace for Xcode)\n")
        return
    path = f"benchmarks/results/stencil_N{N}.gputrace"

    def stencil(x):
        return mx.roll(x, 1, axis=1) + mx.roll(x, -1, axis=3)

    x = mx.random.normal((3, N, N, N)).astype(F32)
    x = stencil(x)
    mx.eval(x)
    mx.synchronize()
    try:
        mx.metal.start_capture(path)
        for _ in range(10):
            x = stencil(x)
        mx.eval(x)
        mx.synchronize()
        mx.metal.stop_capture()
        print(f"wrote GPU capture -> {path} (open in Xcode: File > Open)\n")
    except Exception as e:  # capture needs a Metal capture scope / Xcode tooling
        print(f"capture failed (expected without full Xcode): {e!r}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=192)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--capture", action="store_true", help="set MTL_CAPTURE_ENABLED=1 and emit a .gputrace")
    args = p.parse_args()
    if args.capture:
        os.environ["MTL_CAPTURE_ENABLED"] = "1"

    device_report()
    roofline(args.N, args.iters)
    dispatch_probe(args.N, args.iters)
    maybe_capture(args.N)


if __name__ == "__main__":
    main()
