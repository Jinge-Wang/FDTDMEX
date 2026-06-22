#!/usr/bin/env python3
"""Phase 2 M1 go/no-go: isotropic-uniform interior update as custom Metal kernels.

Implements one FDTD step (E-update then H-update) for the isotropic, uniform, no-CPML interior as
two `mx.fast.metal_kernel`s (thread-per-cell; curl read from global, neighbour reuse via cache), and
compares against the compiled MLX-ops equivalent (the same math via slice-diff + elementwise).

Decision: if the kernel reaches far fewer DRAM round-trips (toward the ~5-8 RT floor) than the
compiled MLX-ops path (~21 RT for the CPML-off iso step), a hand kernel is worth pursuing; if it only
matches, MLX-op fusion is already near the roofline and Phase 2 stops.

    uv run python benchmarks/m1_kernel.py --N 192 --iters 200
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

ROOFLINE_GBS = 240.0  # measured coalesced copy bandwidth on M4 Pro

CB = 0.5  # c * inv_eps (E update); uniform isotropic test constant
CBH = 0.5  # c * inv_mu (H update)


def _build_kernels(N: int):
    """Two thread-per-cell kernels: E += Cb*curl_H(H) (backward diff), H -= Cbh*curl_E(E) (forward)."""
    common = f"""
        uint k = thread_position_in_grid.x;
        uint j = thread_position_in_grid.y;
        uint i = thread_position_in_grid.z;
        const uint N = {N}u;
        if (i >= N || j >= N || k >= N) return;
        const uint N2 = N*N;
        const uint N3 = N*N*N;
        uint idx = i*N2 + j*N + k;
    """
    src_E = (
        common
        + f"""
        const float Cb = {CB}f;
        float Hx = H[idx];        float Hy = H[N3+idx];        float Hz = H[2u*N3+idx];
        float Hx_jm = (j>0u) ? H[idx-N]        : 0.0f;
        float Hx_km = (k>0u) ? H[idx-1u]       : 0.0f;
        float Hy_im = (i>0u) ? H[N3+idx-N2]    : 0.0f;
        float Hy_km = (k>0u) ? H[N3+idx-1u]    : 0.0f;
        float Hz_im = (i>0u) ? H[2u*N3+idx-N2] : 0.0f;
        float Hz_jm = (j>0u) ? H[2u*N3+idx-N]  : 0.0f;
        float cx = (Hz - Hz_jm) - (Hy - Hy_km);
        float cy = (Hx - Hx_km) - (Hz - Hz_im);
        float cz = (Hy - Hy_im) - (Hx - Hx_jm);
        out[idx]       = E[idx]       + Cb*cx;
        out[N3+idx]    = E[N3+idx]    + Cb*cy;
        out[2u*N3+idx] = E[2u*N3+idx] + Cb*cz;
    """
    )
    src_H = (
        common
        + f"""
        const float Cb = {CBH}f;
        float Ex = E[idx];        float Ey = E[N3+idx];        float Ez = E[2u*N3+idx];
        float Ez_jp = (j+1u<N) ? E[2u*N3+idx+N]  : 0.0f;
        float Ey_kp = (k+1u<N) ? E[N3+idx+1u]    : 0.0f;
        float Ex_kp = (k+1u<N) ? E[idx+1u]       : 0.0f;
        float Ez_ip = (i+1u<N) ? E[2u*N3+idx+N2] : 0.0f;
        float Ey_ip = (i+1u<N) ? E[N3+idx+N2]    : 0.0f;
        float Ex_jp = (j+1u<N) ? E[idx+N]        : 0.0f;
        float cx = (Ez_jp - Ez) - (Ey_kp - Ey);
        float cy = (Ex_kp - Ex) - (Ez_ip - Ez);
        float cz = (Ey_ip - Ey) - (Ex_jp - Ex);
        out[idx]       = H[idx]       - Cb*cx;
        out[N3+idx]    = H[N3+idx]    - Cb*cy;
        out[2u*N3+idx] = H[2u*N3+idx] - Cb*cz;
    """
    )
    kE = mx.fast.metal_kernel(name="iso_E", input_names=["E", "H"], output_names=["out"], source=src_E)
    kH = mx.fast.metal_kernel(name="iso_H", input_names=["E", "H"], output_names=["out"], source=src_H)
    grid = (N, N, N)
    tg = (32, 4, 4)

    def step_kernel(E, H):
        (E2,) = kE(inputs=[E, H], output_shapes=[E.shape], output_dtypes=[E.dtype], grid=grid, threadgroup=tg)
        (H2,) = kH(inputs=[E2, H], output_shapes=[H.shape], output_dtypes=[H.dtype], grid=grid, threadgroup=tg)
        return E2, H2

    return step_kernel


def _shift(f, axis, n):
    """Zero-ghost shift by n along axis (n=+1 -> f[i-1] padded low; n=-1 -> f[i+1] padded high)."""
    pads = [(0, 0)] * f.ndim
    if n > 0:
        pads[axis] = (n, 0)
        return mx.pad(f[tuple(slice(0, -n) if a == axis else slice(None) for a in range(f.ndim))], pads)
    pads[axis] = (0, -n)
    return mx.pad(f[tuple(slice(-n, None) if a == axis else slice(None) for a in range(f.ndim))], pads)


def _step_mlxops(E, H):
    """Same iso/uniform/no-CPML step via MLX ops (slice-diff). Spatial axes of (3,N,N,N) are 1,2,3."""
    # backward diff: f - shift_low(f); component arrays H[0]=Hx,H[1]=Hy,H[2]=Hz, axes x=0,y=1,z=2 -> array axes 1,2,3
    def bwd(c, ax):
        return H[c] - _shift(H[c], ax, 1)

    cx = bwd(2, 1) - bwd(1, 2)  # dyHz - dzHy
    cy = bwd(0, 2) - bwd(2, 0)  # dzHx - dxHz
    cz = bwd(1, 0) - bwd(0, 1)  # dxHy - dyHx
    E2 = E + CB * mx.stack([cx, cy, cz], axis=0)

    def fwd(c, ax):
        return _shift(E2[c], ax, -1) - E2[c]

    dx = fwd(2, 1) - fwd(1, 2)
    dy = fwd(0, 2) - fwd(2, 0)
    dz = fwd(1, 0) - fwd(0, 1)
    H2 = H - CBH * mx.stack([dx, dy, dz], axis=0)
    return E2, H2


def _time(fn, E0, H0, iters, warmup=5):
    E, H = E0, H0
    for _ in range(warmup):
        E, H = fn(E, H)
    mx.eval(E, H)
    mx.synchronize()
    E, H = E0, H0
    t0 = time.perf_counter()
    for _ in range(iters):
        E, H = fn(E, H)
    mx.eval(E, H)
    mx.synchronize()
    return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=192)
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()
    N = args.N
    print(f"device: {mx.default_device()}   N={N} (cells={N**3:,})   iters={args.iters}\n")

    mx.random.seed(0)
    E0 = mx.random.normal((3, N, N, N)).astype(mx.float32)
    H0 = mx.random.normal((3, N, N, N)).astype(mx.float32)

    step_kernel = _build_kernels(N)
    step_ops = mx.compile(_step_mlxops)

    # correctness: one step, kernel vs compiled MLX-ops
    Ek, Hk = step_kernel(E0, H0)
    Eo, Ho = step_ops(E0, H0)
    mx.eval(Ek, Hk, Eo, Ho)
    dE = float(mx.max(mx.abs(Ek - Eo)))
    dH = float(mx.max(mx.abs(Hk - Ho)))
    print(f"correctness vs compiled MLX-ops:  maxdiff E={dE:.2e}  H={dH:.2e}  -> {'OK' if max(dE, dH) < 1e-4 else 'MISMATCH'}\n")

    cells = N**3
    rt_per = lambda dt: (dt / args.iters) * ROOFLINE_GBS * 1e9 / (2 * 3 * cells * 4)
    for name, fn in [("compiled MLX-ops", step_ops), ("custom Metal kernels", step_kernel)]:
        dt = _time(fn, E0, H0, args.iters)
        mcs = cells * args.iters / dt / 1e6
        print(f"  {name:22} {dt / args.iters * 1e3:7.3f} ms/step  {mcs:8.1f} Mcs/s   RT/step ~{rt_per(dt):4.0f}")


if __name__ == "__main__":
    main()
