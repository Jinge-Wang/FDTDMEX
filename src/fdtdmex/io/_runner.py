"""Detached-child entry point for :func:`fdtdmex.io.run_simulation_from_hdf5`.

``run_simulation_from_hdf5`` launches ``python -m fdtdmex.io._runner`` with the job folder as the
child's cwd. This module just parses the arguments and calls :func:`fdtdmex.io.run_simulation`, which
writes ``status.json`` / ``progress.jsonl`` at the cwd top and the results HDF5 into ``outputs/``.
It is never imported on the agent's path — only spawned as a subprocess.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fdtdmex.io._runner", description="detached fdtdmex run worker")
    ap.add_argument("--hdf5", required=True, help="config HDF5 path (relative to the cwd job folder)")
    ap.add_argument("--backend", default="mlx")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--results-name", default="result.hdf5")
    args = ap.parse_args(argv)

    from .status import run_simulation

    # run_simulation already captures failures into status.json (status="failed") before re-raising;
    # we still exit non-zero so the runner.log + child exit code reflect the failure.
    run_simulation(
        args.hdf5,
        backend=args.backend,
        run_id=args.run_id,
        name=args.name,
        results_name=args.results_name,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
