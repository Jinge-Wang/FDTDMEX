"""Visualization + physics test: spacing-weighted anisotropic averaging is 2nd-order (M4).

The genuinely-new M4 physics is the spacing-weighted off-diagonal anisotropic average
(:func:`fdtdx.mlx.aniso.avg_anisotropic_E_component_mlx`). On a non-uniform grid fdtdx leaves
this average *unweighted* (the target edge is treated as the midpoint of the two cell centers,
which is only true on a uniform grid), making it 1st-order on a graded mesh. The MLX port weights
the center->edge half-step by the neighbouring cell widths, which is the exact linear interpolant
and therefore 2nd-order.

This measures the interpolation error of both forms on a strongly graded (alternating-width)
mesh under refinement, asserts the convergence orders (~2 weighted, ~1 unweighted), and saves a
log-log convergence figure to the output dir (``$FDTDMEX_VIZ_DIR`` or ``tests/visualization/figures/``).

Runs on MLX (the average is an MLX kernel); skipped off Apple Silicon / without mlx.
"""

import os
import pathlib

import numpy as np
import pytest

from fdtdx.backend.platform import is_apple_silicon, mlx_available

pytestmark = [
    pytest.mark.validation,
    pytest.mark.skipif(
        not (is_apple_silicon() and mlx_available()),
        reason="MLX (Metal) backend requires Apple Silicon + mlx",
    ),
]

_LEVELS = [8, 16, 32, 64, 128, 256]  # number of width-pairs; the mesh has 2x as many cells
_LENGTH = 1.0


def _alternating_edges(n_pairs, ratio=2.0):
    """Strongly graded 1-D mesh: widths alternate w, ratio*w (O(1) adjacent ratio at every scale)."""
    w = _LENGTH / (n_pairs * (1.0 + ratio))
    widths = np.tile([w, ratio * w], n_pairs).astype(np.float64)
    edges = np.concatenate([[0.0], np.cumsum(widths)])
    return edges, widths


def _interp_error(n_pairs, weighted):
    """Max error of ``avg_anisotropic_E_component`` (Ex -> Ey location) reconstructing sin(kx) at
    the backward x-edges, on an alternating-width mesh. Returns ``(mean_cell_width, max_error)``.

    The field is constant along the (unweighted) location axis, so only the center->edge step
    along the component (x) axis -- the part M4 weights -- is exercised.
    """
    import mlx.core as mx

    from fdtdx.mlx.aniso import avg_anisotropic_E_component_mlx

    edges, widths = _alternating_edges(n_pairs)
    nx = widths.shape[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    k = 2.0 * np.pi / _LENGTH
    f_center = np.sin(k * centers)  # Ex samples (x at cell centers)
    f_edge = np.sin(k * edges)  # analytic value at the x-edges

    ny = nz = 3
    field = np.zeros((3, nx, ny, nz), dtype=np.float32)
    field[0] = np.broadcast_to(f_center[:, None, None], (nx, ny, nz))
    field_pad = mx.array(np.pad(field, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="edge"))

    if weighted:
        wpad = np.concatenate([widths[:1], widths, widths[-1:]]).astype(np.float32)
        aniso_widths = (
            mx.array(wpad.reshape(nx + 2, 1, 1)),
            mx.array(np.ones((1, ny + 2, 1), np.float32)),
            mx.array(np.ones((1, 1, nz + 2), np.float32)),
        )
    else:
        aniso_widths = None

    out = np.asarray(avg_anisotropic_E_component_mlx(field_pad, 0, 1, aniso_widths))
    got = out[:, ny // 2, nz // 2]  # got[i] ~ value at x_edges[i] (backward edge of cell i)
    err = float(np.abs(got[1:nx] - f_edge[1:nx]).max())  # interior edges only
    return _LENGTH / nx, err


def _output_dir() -> pathlib.Path:
    default = pathlib.Path(__file__).resolve().parent / "figures"
    d = pathlib.Path(os.environ.get("FDTDMEX_VIZ_DIR", default))
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_weighted_aniso_average_second_order_convergence_figure():
    h, err_w, err_u = [], [], []
    for n in _LEVELS:
        hi, ew = _interp_error(n, weighted=True)
        _, eu = _interp_error(n, weighted=False)
        h.append(hi)
        err_w.append(ew)
        err_u.append(eu)
    h = np.asarray(h)
    err_w = np.asarray(err_w)
    err_u = np.asarray(err_u)

    # Convergence order = slope of log(error) vs log(h).
    order_w = float(np.polyfit(np.log(h), np.log(err_w), 1)[0])
    order_u = float(np.polyfit(np.log(h), np.log(err_u), 1)[0])
    assert order_w > 1.8, f"spacing-weighted average not 2nd-order: order={order_w:.3f}, errors={err_w}"
    assert order_u < 1.3, f"unweighted average should be ~1st-order: order={order_u:.3f}, errors={err_u}"
    assert err_w[-1] < 0.2 * err_u[-1], (err_w[-1], err_u[-1])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.loglog(h, err_u, "s-", color="#c1121f", lw=2, ms=7, label=f"unweighted (fdtdx)  ~ O(h$^{{{order_u:.2f}}}$)")
    ax.loglog(
        h, err_w, "o-", color="#0353a4", lw=2, ms=7, label=f"spacing-weighted (MLX, M4)  ~ O(h$^{{{order_w:.2f}}}$)"
    )

    # Reference slope guides (O(h) and O(h^2)) anchored to each curve's coarsest point.
    ax.loglog(h, err_u[0] * (h / h[0]) ** 1.0, "--", color="#c1121f", alpha=0.5, lw=1.3, label="O(h)  reference")
    ax.loglog(h, err_w[0] * (h / h[0]) ** 2.0, "--", color="#0353a4", alpha=0.5, lw=1.3, label="O(h$^2$)  reference")

    ax.set_xlabel("mean cell width  h  (domain units)")
    ax.set_ylabel("max interpolation error of the off-diagonal average")
    ax.set_title(
        "Spacing-weighted anisotropic averaging is 2nd-order on a graded mesh (MLX / Metal)\n"
        "alternating-width mesh (w, 2w);  Ex co-located to the Ey Yee point"
    )
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()

    out = _output_dir() / "nonuniform_convergence_mlx.png"
    fig.savefig(out, dpi=115)
    plt.close(fig)
    assert out.exists()
