#!/usr/bin/env python3
"""Micro-benchmark isolating *why* the eager MLX FDTD step plateaus far below memory roofline.

Mirrors the real per-step traffic pattern (roll-based Yee curl + E/H update) at a fixed N and
compares four variants on the Metal GPU:

  1. eager, pad+roll       — what the engine does today (pad each field, 6 rolls, elementwise chain)
  2. eager, slice-diff     — same math, but finite differences via slicing (no full-array pad copy)
  3. mx.compile(pad+roll)  — variant 1 wrapped in mx.compile (kernel fusion)
  4. mx.compile(slice-diff)— variant 2 wrapped in mx.compile

Reports throughput (Mcell-steps/s) and effective DRAM bandwidth assuming a *minimal* fused step
would move ~8 arrays/cell-step. The gap between (1) and (3)/(4) is the fusion headroom; the gap
between (1) and (2) is the per-step padding cost.

    uv run python benchmarks/microbench_fusion.py --N 192 --iters 60
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx


def step_pad_roll(E, H, inv_eps, c):
    """One E then H update using pad + roll (mirrors curl.py / update.py, CPML omitted)."""
    Hp = mx.pad(H, [(0, 0), (1, 1), (1, 1), (1, 1)])
    dyHz = (Hp[2] - mx.roll(Hp[2], 1, axis=1))[1:-1, 1:-1, 1:-1]
    dzHy = (Hp[1] - mx.roll(Hp[1], 1, axis=2))[1:-1, 1:-1, 1:-1]
    dzHx = (Hp[0] - mx.roll(Hp[0], 1, axis=2))[1:-1, 1:-1, 1:-1]
    dxHz = (Hp[2] - mx.roll(Hp[2], 1, axis=0))[1:-1, 1:-1, 1:-1]
    dxHy = (Hp[1] - mx.roll(Hp[1], 1, axis=0))[1:-1, 1:-1, 1:-1]
    dyHx = (Hp[0] - mx.roll(Hp[0], 1, axis=1))[1:-1, 1:-1, 1:-1]
    curl = mx.stack([dyHz - dzHy, dzHx - dxHz, dxHy - dyHx], axis=0)
    E = E + c * curl * inv_eps

    Ep = mx.pad(E, [(0, 0), (1, 1), (1, 1), (1, 1)])
    dyEz = (mx.roll(Ep[2], -1, axis=1) - Ep[2])[1:-1, 1:-1, 1:-1]
    dzEy = (mx.roll(Ep[1], -1, axis=2) - Ep[1])[1:-1, 1:-1, 1:-1]
    dzEx = (mx.roll(Ep[0], -1, axis=2) - Ep[0])[1:-1, 1:-1, 1:-1]
    dxEz = (mx.roll(Ep[2], -1, axis=0) - Ep[2])[1:-1, 1:-1, 1:-1]
    dxEy = (mx.roll(Ep[1], -1, axis=0) - Ep[1])[1:-1, 1:-1, 1:-1]
    dyEx = (mx.roll(Ep[0], -1, axis=1) - Ep[0])[1:-1, 1:-1, 1:-1]
    curl2 = mx.stack([dyEz - dzEy, dzEx - dxEz, dxEy - dyEx], axis=0)
    H = H - c * curl2
    return E, H


def _bdiff(f, axis):  # backward difference with zero ghost (matches curl_H)
    sl = [slice(None)] * f.ndim
    sl[axis] = slice(0, -1)
    lo = [slice(None)] * f.ndim
    lo[axis] = slice(1, None)
    d = mx.zeros_like(f)
    d[tuple(lo)] = f[tuple(lo)] - f[tuple(sl)]
    d[tuple([slice(0, 1) if i == axis else slice(None) for i in range(f.ndim)])] = f[
        tuple([slice(0, 1) if i == axis else slice(None) for i in range(f.ndim)])
    ]
    return d


def step_slice(E, H, inv_eps, c):
    """Same update via slicing-based differences (no full-array pad copy)."""
    dyHz = _bdiff(H[2], 1)
    dzHy = _bdiff(H[1], 2)
    dzHx = _bdiff(H[0], 2)
    dxHz = _bdiff(H[2], 0)
    dxHy = _bdiff(H[1], 0)
    dyHx = _bdiff(H[0], 1)
    curl = mx.stack([dyHz - dzHy, dzHx - dxHz, dxHy - dyHx], axis=0)
    E = E + c * curl * inv_eps

    def fdiff(f, axis):
        lo = [slice(None)] * f.ndim
        lo[axis] = slice(0, -1)
        hi = [slice(None)] * f.ndim
        hi[axis] = slice(1, None)
        d = mx.zeros_like(f)
        d[tuple(lo)] = f[tuple(hi)] - f[tuple(lo)]
        return d

    dyEz = fdiff(E[2], 1)
    dzEy = fdiff(E[1], 2)
    dzEx = fdiff(E[0], 2)
    dxEz = fdiff(E[2], 0)
    dxEy = fdiff(E[1], 0)
    dyEx = fdiff(E[0], 1)
    curl2 = mx.stack([dyEz - dzEy, dzEx - dxEz, dxEy - dyEx], axis=0)
    H = H - c * curl2
    return E, H


def time_variant(name, step_fn, N, iters, compile_it):
    E = mx.random.normal((3, N, N, N)).astype(mx.float32)
    H = mx.random.normal((3, N, N, N)).astype(mx.float32)
    inv_eps = mx.array(0.44, dtype=mx.float32)
    c = 0.5
    fn = mx.compile(step_fn) if compile_it else step_fn

    # warmup (compile + kernel build)
    for _ in range(3):
        E, H = fn(E, H, inv_eps, c)
    mx.eval(E, H)

    t0 = time.perf_counter()
    for _ in range(iters):
        E, H = fn(E, H, inv_eps, c)
    mx.eval(E, H)
    dt = time.perf_counter() - t0

    cells = N**3
    mcs = cells * iters / dt / 1e6
    # minimal fused step traffic estimate: read+write E,H (each 3 comp) + read inv_eps ~ 8*4 bytes/cell
    min_bytes = cells * iters * 8 * 4
    eff_gbs = min_bytes / dt / 1e9
    print(f"  {name:28} {dt:7.3f}s  {mcs:8.1f} Mcell·steps/s   (min-useful ~{eff_gbs:6.1f} GB/s)")
    return mcs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=192)
    p.add_argument("--iters", type=int, default=60)
    args = p.parse_args()

    print(f"device: {mx.default_device()}   N={args.N} (cells={args.N**3:,})   iters={args.iters}\n")
    print("variant                        time      throughput")
    base = time_variant("eager  pad+roll", step_pad_roll, args.N, args.iters, compile_it=False)
    time_variant("eager  slice-diff", step_slice, args.N, args.iters, compile_it=False)
    c1 = time_variant("compiled pad+roll", step_pad_roll, args.N, args.iters, compile_it=True)
    c2 = time_variant("compiled slice-diff", step_slice, args.N, args.iters, compile_it=True)
    print(f"\nfusion speedup (compiled pad+roll / eager pad+roll):  {c1 / base:.2f}x")
    print(f"best variant vs eager pad+roll:                       {max(c1, c2) / base:.2f}x")
    print("\nNote: this strips CPML/sources, so absolute Mcs/s is higher than the full engine;")
    print("the *ratios* show the headroom fusion + dropping per-step pad would unlock.")


if __name__ == "__main__":
    main()
