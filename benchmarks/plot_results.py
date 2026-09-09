#!/usr/bin/env python3
"""Plot a forward-benchmark JSONL: MLX (Metal) vs JAX-CPU scaling, by material.

Produces a 4-panel figure:
  (a) throughput (Mcell-steps/s) vs cells, log-log -- the headline scaling test;
  (b) wall-clock per run vs cells, log-log;
  (c) MLX/JAX speedup vs cells;
  (d) peak memory vs cells (MLX exact GPU peak; process RSS otherwise).

Usage:
    uv run python benchmarks/plot_results.py benchmarks/results/<file>.jsonl
    uv run python benchmarks/plot_results.py <file>.jsonl --out benchmarks/figures/forward_scaling.png
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAT_COLOR = {"isotropic": "#1f77b4", "diagonal": "#2ca02c", "full_aniso": "#d62728", "iso_conductive": "#9467bd"}
BACKEND_STYLE = {"mlx": dict(marker="o", linestyle="-"), "jax": dict(marker="s", linestyle="--")}
BACKEND_LABEL = {"mlx": "MLX / Metal", "jax": "JAX / CPU"}


def load(path: str):
    meta, cells = {}, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("record_type") == "metadata":
                meta = rec
            elif rec.get("record_type") == "cell":
                cells.append(rec)
    return meta, cells


def _ok(c):
    return c.get("status") == "ok" and c.get("finite", True)


def _series(cells, backend, material, xkey, ykey):
    pts = [
        (c[xkey], c[ykey])
        for c in cells
        if c["backend"] == backend and c["material"] == material and _ok(c) and c.get(ykey) is not None
    ]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def _peak_bytes(c):
    return c.get("mlx_peak_bytes") or c.get("ru_maxrss_bytes")


def plot(meta, cells, out_path):
    backends = sorted({c["backend"] for c in cells if _ok(c)}, reverse=True)  # mlx first
    materials = [
        m for m in ("isotropic", "diagonal", "full_aniso", "iso_conductive") if any(c["material"] == m for c in cells)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax_tp, ax_wall, ax_spd, ax_mem = axes.ravel()

    for material in materials:
        col = MAT_COLOR.get(material, "gray")
        for backend in backends:
            x, y = _series(cells, backend, material, "cells", "throughput_mcellsteps_s")
            if x:
                ax_tp.plot(x, y, color=col, label=f"{BACKEND_LABEL[backend]} · {material}", **BACKEND_STYLE[backend])
            x, y = _series(cells, backend, material, "cells", "time_s_median")
            if x:
                ax_wall.plot(x, y, color=col, **BACKEND_STYLE[backend])

        # speedup MLX/JAX (matched cells)
        mx = {
            c["cells"]: c["time_s_median"]
            for c in cells
            if c["backend"] == "mlx" and c["material"] == material and _ok(c)
        }
        jx = {
            c["cells"]: c["time_s_median"]
            for c in cells
            if c["backend"] == "jax" and c["material"] == material and _ok(c)
        }
        common = sorted(set(mx) & set(jx))
        if common:
            ax_spd.plot(common, [jx[c] / mx[c] for c in common], color=col, marker="o", label=material)

        # memory
        for backend in backends:
            pts = sorted(
                (c["cells"], _peak_bytes(c))
                for c in cells
                if c["backend"] == backend and c["material"] == material and _ok(c) and _peak_bytes(c)
            )
            if pts:
                ax_mem.plot([p[0] for p in pts], [p[1] / 1e9 for p in pts], color=col, **BACKEND_STYLE[backend])

    for ax in (ax_tp, ax_wall, ax_mem):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("domain size (cells)")
        ax.grid(True, which="both", alpha=0.25)
    ax_spd.set_xscale("log")
    ax_spd.set_xlabel("domain size (cells)")
    ax_spd.grid(True, which="both", alpha=0.25)

    ax_tp.set_ylabel("throughput (Mcell·steps / s)")
    ax_tp.set_title("(a) Forward throughput — higher is better")
    ax_tp.legend(fontsize=7, loc="best")

    ax_wall.set_ylabel("wall-clock per run (s)")
    ax_wall.set_title("(b) Wall-clock per forward run")

    ax_spd.axhline(1.0, color="k", lw=0.8, ls=":")
    ax_spd.set_ylabel("speedup  (JAX-CPU time / MLX time)")
    ax_spd.set_title("(c) MLX speedup over JAX-CPU  (>1 = MLX faster)")
    ax_spd.legend(fontsize=8, loc="best")

    ax_mem.set_ylabel("peak memory (GB)")
    ax_mem.set_title("(d) Peak memory  (MLX: exact GPU; JAX: process RSS)")

    chip = meta.get("chip", "?")
    steps = meta.get("steps_requested", "?")
    fig.suptitle(
        f"FDTDMEX forward engine — MLX/Metal vs JAX/CPU  ·  {chip}  ·  "
        f"{steps} steps, float32, commit {meta.get('git_commit', '?')}",
        fontsize=12,
    )
    # solid = MLX, dashed = JAX legend hint
    fig.text(
        0.5,
        0.005,
        "solid line + ○ = MLX/Metal     dashed line + □ = JAX/CPU     color = material",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    return fig


def print_table(cells):
    print(
        f"\n{'backend':7} {'material':14} {'N':>4} {'cells':>11} {'steps':>5} "
        f"{'med_s':>9} {'Mcell·st/s':>11} {'peakMB':>9} {'status'}"
    )
    print("-" * 90)
    for c in sorted(cells, key=lambda c: (c["material"], c["N"], c["backend"])):
        pk = _peak_bytes(c) or 0
        print(
            f"{c['backend']:7} {c['material']:14} {c['N']:>4} {c.get('cells', 0):>11} "
            f"{c.get('steps_actual', 0):>5} {c.get('time_s_median', 0):>9.4f} "
            f"{c.get('throughput_mcellsteps_s', 0):>11.2f} {pk / 1e6:>9.1f} {c['status']}"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", help="path to a forward_*.jsonl results file")
    p.add_argument("--out", default=None, help="output PNG (default outputs/<stem>_scaling.png)")
    args = p.parse_args()

    meta, cells = load(args.results)
    if not cells:
        raise SystemExit("no cell records found in " + args.results)
    print_table(cells)
    out = args.out or os.path.join("outputs", os.path.splitext(os.path.basename(args.results))[0] + "_scaling.png")
    plot(meta, cells, out)


if __name__ == "__main__":
    main()
