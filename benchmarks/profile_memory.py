#!/usr/bin/env python3
"""Subprocess-isolated peak-memory probe (the 'value proof': where JAX-CPU hits the wall).

bench_forward's in-process ru_maxrss is polluted across cells; this runs ONE (backend, N) per
process so the high-water mark is clean. Memory is set by allocation, not step count, so a short
run suffices. Prints one JSON line; a driver loops N in fresh subprocesses for each backend.

    # one cell:
    uv run python benchmarks/profile_memory.py --backend mlx --N 192 --steps 20
    # sweep (fresh process per point, so an OOM kills only that point):
    for b in mlx jax; do for N in 96 192 256 320 384; do
      uv run python benchmarks/profile_memory.py --backend $b --N $N --steps 20; done; done
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
from types import SimpleNamespace


def _rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (rss if sys.platform == "darwin" else rss * 1024) / 1e9  # macOS: bytes; Linux: KiB


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=["mlx", "jax"])
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--material", default="isotropic")
    args = p.parse_args()

    rec = {"backend": args.backend, "N": args.N, "cells_M": round(args.N**3 / 1e6, 1), "status": "ok"}
    try:
        import numpy as np
        from bench_forward import build_case

        import fdtdx

        case_args = SimpleNamespace(
            spacing=50e-9, courant=0.99, pml=8, wavelength=1e-6, detector="none", steps=args.steps
        )
        arrays, oc, config, key, info = build_case(args.material, args.N, case_args)
        with fdtdx.use_backend(args.backend):
            _, out = fdtdx.run_fdtd(arrays=arrays, objects=oc, config=config, key=key, show_progress=False)
            np.asarray(out.fields.E)  # block until materialized
        if args.backend == "mlx":
            import mlx.core as mx

            mx.synchronize()
            rec["mlx_peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
        rec["proc_rss_gb"] = round(_rss_gb(), 2)
    except Exception as e:  # OOM / allocation failure / instability
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        rec["proc_rss_gb"] = round(_rss_gb(), 2)

    print(json.dumps(rec))


if __name__ == "__main__":
    main()
