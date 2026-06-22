"""Visualisation for solved waveguide modes and their material cross-section (Phase 4 Track A).

Renders a solved mode profile next to the (optionally subpixel-smoothed) permittivity
cross-section it was computed on. Reuses the existing field-component renderer
(:func:`fdtdx.utils.plot_field_slice.plot_field_slice_component`) and energy metric
(:func:`fdtdx.core.physics.metrics.compute_energy`) so the look matches the rest of fdtdx.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from fdtdx.core.misc import expand_to_3x3
from fdtdx.core.physics.metrics import compute_energy
from fdtdx.utils.plot_field_slice import plot_field_slice_component

if TYPE_CHECKING:
    from fdtdx.objects.sources.mode import ModePlaneSource


def _squeeze_to_2d(field: jnp.ndarray) -> jnp.ndarray:
    """Squeeze a ``(Nx, Ny, Nz)`` field (one singleton) to its 2-D transverse slice."""
    arr = jnp.asarray(field)
    if arr.ndim == 2:
        return arr
    sq = tuple(i for i, d in enumerate(arr.shape) if d == 1)
    return jnp.squeeze(arr, axis=sq)


def _index_cross_section(inv_permittivity: jnp.ndarray) -> np.ndarray:
    """Return a real refractive-index map ``sqrt(mean diag(eps))`` from inverse permittivity.

    Works for isotropic (1), diagonal (3), and full-tensor (9) ``inv_permittivity`` layouts — the
    diagonal of the inverse tensor is averaged, inverted, and square-rooted to give a single
    scalar-index view of the (possibly smoothed) cross-section.
    """
    tensor = np.asarray(expand_to_3x3(inv_permittivity))  # (3, 3, *spatial)
    inv_diag = np.real(np.stack([tensor[k, k] for k in range(3)], axis=0))  # (3, *spatial)
    eps_mean = 1.0 / np.mean(inv_diag, axis=0)
    n_map = np.sqrt(np.clip(eps_mean, 0.0, None))
    return np.asarray(_squeeze_to_2d(jnp.asarray(n_map)))


def plot_mode(
    E: jnp.ndarray,
    H: jnp.ndarray,
    *,
    inv_permittivity: jnp.ndarray | None = None,
    inv_permeability: jnp.ndarray | float | None = None,
    component: str = "real",
    filename: str | Path | None = None,
) -> Figure:
    """Plot a solved mode: index cross-section, energy density, and the six field components.

    Args:
        E: complex electric field, shape ``(3, Nx, Ny, Nz)`` (one singleton) or ``(3, w, h)``.
        H: complex magnetic field, same shape as ``E``.
        inv_permittivity: optional inverse-permittivity cross-section (1/3/9 components) used to draw
            the refractive-index map (the *smoothed* tensor if subpixel smoothing was applied) and the
            energy density. When ``None`` the index panel is omitted and energy uses unit material.
        inv_permeability: optional inverse permeability for the energy density (defaults to 1).
        component: ``"real"``, ``"imag"``, or ``"abs"`` — which part of the complex field to render.
        filename: if given, the figure is saved (300 dpi) and closed.

    Returns:
        The matplotlib :class:`~matplotlib.figure.Figure`.
    """
    part = {"real": np.real, "imag": np.imag, "abs": np.abs}[component]
    E2 = np.stack([part(np.asarray(_squeeze_to_2d(E[k]))) for k in range(3)], axis=0)
    H2 = np.stack([part(np.asarray(_squeeze_to_2d(H[k]))) for k in range(3)], axis=0)

    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.0])

    # Top-left: refractive-index cross-section (the smoothed material, if provided).
    ax_eps = fig.add_subplot(gs[0, 0])
    if inv_permittivity is not None:
        n_map = _index_cross_section(inv_permittivity)
        im = ax_eps.imshow(n_map.T, origin="lower", aspect="equal", cmap="viridis")
        ax_eps.set_title("refractive index $n$")
        plt.colorbar(im, ax=ax_eps)
    else:
        ax_eps.axis("off")

    # Top-middle: energy density (mode confinement).
    ax_en = fig.add_subplot(gs[0, 1])
    inv_eps = inv_permittivity if inv_permittivity is not None else jnp.ones_like(jnp.asarray(E[0]).real)
    inv_mu = inv_permeability if inv_permeability is not None else 1.0
    energy = np.asarray(_squeeze_to_2d(compute_energy(E=E, H=H, inv_permittivity=inv_eps, inv_permeability=inv_mu)))
    im = ax_en.imshow(energy.T, origin="lower", aspect="equal", cmap="inferno")
    ax_en.set_title("energy density")
    plt.colorbar(im, ax=ax_en)

    fig.add_subplot(gs[0, 2]).axis("off")

    # Rows 2-3: the six field components (chosen part).
    names = [["Ex", "Ey", "Ez"], ["Hx", "Hy", "Hz"]]
    comps = [E2, H2]
    for r in range(2):
        for col in range(3):
            ax = fig.add_subplot(gs[r + 1, col])
            plot_field_slice_component(
                field=jnp.asarray(comps[r][col]),
                component_name=f"{component}({names[r][col]})",
                ax=ax,
                plot_legend=True,
            )

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close(fig)
    return fig


def plot_mode_from_source(source: "ModePlaneSource", filename: str | Path | None = None, **kwargs) -> Figure:
    """Plot the mode stored on an applied :class:`ModePlaneSource` (after ``apply``/``apply_params``)."""
    if getattr(source, "_E", None) is None or getattr(source, "_H", None) is None:
        raise Exception("Cannot plot mode before init to grid + apply params (source._E/_H are unset).")
    return plot_mode(
        source._E,
        source._H,
        inv_permittivity=source._inv_permittivity,
        inv_permeability=source._inv_permeability,
        filename=filename,
        **kwargs,
    )
