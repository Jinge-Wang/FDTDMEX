"""Interactive plotly 3D view of a simulation setup — the draggable scene for notebooks and the
web (AG-UI) front end.

Consumes the same inputs as the matplotlib :func:`fdtdx.plot_setup` (a ``SimulationConfig`` + an
``ObjectContainer``) and reuses its index→physical coordinate math (``_axis_edges_um``). Each object
is drawn as a translucent cuboid spanning its ``grid_slice_tuple``; the simulation volume is a
wireframe for context. The legend is grouped (Lumerical-like) into **Boundaries / Geometry / Sources /
Monitors** — all boundary slabs collapse into one toggleable entry, and clicking a group header
toggles the whole category (``groupclick="togglegroup"``). Pass ``show_material=True`` with the
resolved arrays to overlay a permittivity isosurface, which reveals the *true* geometry (e.g. a ring)
since the objects themselves draw only as bounding boxes.

One function serves manual scripting, general visualization, and the agentic flow: ``confirm=False``
(default) returns a plain, non-blocking figure; ``confirm=True`` tags the figure (``layout.meta``)
with the confirm-required affordance the orchestration layer reads — the actual confirmation gate
lives there, not here. :func:`to_plotly_json` produces a JSON-safe payload the browser can render.
"""

from __future__ import annotations

import json
from typing import Any

from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ObjectContainer
from fdtdx.objects.boundaries.bloch import BlochBoundary
from fdtdx.objects.boundaries.pec import PerfectElectricConductor
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
from fdtdx.objects.boundaries.pmc import PerfectMagneticConductor
from fdtdx.objects.detectors.detector import Detector
from fdtdx.objects.sources.source import Source
from fdtdx.utils.plot_setup import _axis_edges_um

# Cuboid triangulation (8 vertices → 12 triangles) shared by every box mesh.
_BOX_I = [0, 0, 0, 0, 4, 4, 6, 6, 1, 1, 2, 3]
_BOX_J = [1, 2, 3, 4, 5, 6, 5, 2, 5, 6, 6, 7]
_BOX_K = [2, 3, 7, 7, 6, 7, 1, 1, 6, 2, 7, 4]

_BOUNDARY_TYPES = (PerfectlyMatchedLayer, BlochBoundary, PerfectElectricConductor, PerfectMagneticConductor)

# Legend categories (Lumerical-like): grouped headers the user can toggle.
_CAT_BOUNDARIES = "Boundaries"
_CAT_GEOMETRY = "Geometry"
_CAT_SOURCES = "Sources"
_CAT_MONITORS = "Monitors"


def _category(obj) -> str:
    if isinstance(obj, _BOUNDARY_TYPES):
        return _CAT_BOUNDARIES
    if isinstance(obj, Source):
        return _CAT_SOURCES
    if isinstance(obj, Detector):
        return _CAT_MONITORS
    return _CAT_GEOMETRY


def _color_str(obj) -> str:
    color = getattr(obj, "color", None)
    if color is None:
        return "rgb(150,150,150)"
    r, g, b = color.to_rgb_255()
    return f"rgb({r},{g},{b})"


def _bounds_um(config: SimulationConfig, obj, plane_size: tuple[int, int, int]):
    """Return ((x0,x1),(y0,y1),(z0,z1)) physical bounds in micrometres for an object."""
    sl = obj.grid_slice_tuple
    return tuple(_axis_edges_um(config, axis, sl[axis], plane_size[axis]) for axis in range(3))


def _box_mesh_arrays(boxes: list):
    """Concatenate several ((x0,x1),(y0,y1),(z0,z1)) boxes into one Mesh3d's x/y/z + i/j/k arrays."""
    xs, ys, zs, ii, jj, kk = [], [], [], [], [], []
    for n, ((x0, x1), (y0, y1), (z0, z1)) in enumerate(boxes):
        xs += [x0, x1, x1, x0, x0, x1, x1, x0]
        ys += [y0, y0, y1, y1, y0, y0, y1, y1]
        zs += [z0, z0, z0, z0, z1, z1, z1, z1]
        off = 8 * n
        ii += [off + v for v in _BOX_I]
        jj += [off + v for v in _BOX_J]
        kk += [off + v for v in _BOX_K]
    return xs, ys, zs, ii, jj, kk


def plot_setup_3d(
    config: SimulationConfig,
    objects: ObjectContainer,
    *,
    material_arrays: Any | None = None,
    show_material: bool = False,
    exclude_object_list: list | None = None,
    opacity: float = 0.45,
    confirm: bool = False,
):
    """Build an interactive plotly 3D figure of the simulation setup.

    Args:
        config: The simulation config (provides grid coordinates).
        objects: The placed ``ObjectContainer``.
        material_arrays: Optional resolved ``ArrayContainer`` for a permittivity isosurface overlay.
        show_material: Draw the permittivity isosurface (requires ``material_arrays``).
        exclude_object_list: Objects to omit from the drawing.
        opacity: Cuboid opacity for filled objects.
        confirm: Tag the figure for the agentic confirm-required affordance (non-blocking here).

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError("plot_setup_3d requires plotly. Install the 'viz' extra: uv sync --extra viz") from exc

    exclude = set(id(o) for o in (exclude_object_list or []))
    volume = objects.volume
    plane_size = tuple(volume.grid_shape)  # (Nx, Ny, Nz)

    traces = []
    boundary_boxes: list = []  # all boundary slabs → one merged legend entry
    for obj in objects.objects:
        if id(obj) in exclude or obj is volume:
            continue
        if getattr(obj, "color", None) is None:
            continue
        cat = _category(obj)
        box = _bounds_um(config, obj, plane_size)
        if cat == _CAT_BOUNDARIES:
            boundary_boxes.append(box)
            continue
        (x0, x1), (y0, y1), (z0, z1) = box
        label = type(obj).__name__ if obj.name.startswith("Object") else obj.name
        traces.append(
            go.Mesh3d(
                x=[x0, x1, x1, x0, x0, x1, x1, x0],
                y=[y0, y0, y1, y1, y0, y0, y1, y1],
                z=[z0, z0, z0, z0, z1, z1, z1, z1],
                i=_BOX_I,
                j=_BOX_J,
                k=_BOX_K,
                color=_color_str(obj),
                opacity=0.25 if cat == _CAT_GEOMETRY else 0.55,
                flatshading=True,
                name=label,
                hovertext=f"{label} ({type(obj).__name__})",
                hoverinfo="text",
                showlegend=True,
                legendgroup=cat,
                legendgrouptitle=dict(text=cat),
            )
        )

    # All boundary slabs collapsed into one toggleable "Boundaries" entry.
    if boundary_boxes:
        bx, by, bz, bi, bj, bk = _box_mesh_arrays(boundary_boxes)
        traces.insert(
            0,
            go.Mesh3d(
                x=bx,
                y=by,
                z=bz,
                i=bi,
                j=bj,
                k=bk,
                color="rgb(120,140,170)",
                opacity=0.12,
                flatshading=True,
                name="PML",
                hoverinfo="name",
                showlegend=True,
                legendgroup=_CAT_BOUNDARIES,
                legendgrouptitle=dict(text=_CAT_BOUNDARIES),
            ),
        )

    # Simulation volume as a wireframe box for context (under the Geometry group).
    (vx0, vx1), (vy0, vy1), (vz0, vz1) = _bounds_um(config, volume, plane_size)
    edges_x, edges_y, edges_z = _wireframe_box(vx0, vx1, vy0, vy1, vz0, vz1)
    traces.append(
        go.Scatter3d(
            x=edges_x,
            y=edges_y,
            z=edges_z,
            mode="lines",
            line=dict(color="rgb(80,80,80)", width=2),
            name="domain",
            hoverinfo="name",
            showlegend=True,
            legendgroup=_CAT_GEOMETRY,
            legendgrouptitle=dict(text=_CAT_GEOMETRY),
        )
    )

    if show_material and material_arrays is not None:
        iso = _material_isosurface(go, config, material_arrays, plane_size)
        if iso is not None:
            traces.append(iso)

    fig = go.Figure(data=traces)
    title = "Simulation setup" + (" — confirm required" if confirm else "")
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x (µm)",
            yaxis_title="y (µm)",
            zaxis_title="z (µm)",
            aspectmode="data",
            camera=dict(projection=dict(type="orthographic")),
        ),
        # Grouped legend (Boundaries / Geometry / Sources / Monitors). Click a row to toggle it;
        # click a group title to toggle the whole group. Anchored left so it never overlaps the
        # permittivity colorbar (which sits on the right).
        legend=dict(
            x=0.0,
            y=1.0,
            xanchor="left",
            yanchor="top",
            groupclick="togglegroup",
            itemsizing="constant",
            tracegroupgap=12,
            bgcolor="rgba(255,255,255,0.65)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        meta={"fdtdmex_confirm": bool(confirm)},
    )
    return fig


def _wireframe_box(x0, x1, y0, y1, z0, z1):
    """Edge polyline (with None breaks) tracing the 12 edges of a box."""
    corners = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [corners[a][0], corners[b][0], None]
        ys += [corners[a][1], corners[b][1], None]
        zs += [corners[a][2], corners[b][2], None]
    return xs, ys, zs


def _material_isosurface(go, config, arrays, plane_size):
    """Optional permittivity isosurface from the resolved inverse-permittivity (component 0)."""
    import numpy as np

    inv_eps = np.asarray(arrays.inv_permittivities)
    eps = 1.0 / np.clip(inv_eps[0], 1e-12, None)  # (Nx, Ny, Nz)
    nx, ny, nz = eps.shape
    xc = np.array([0.5 * sum(_axis_edges_um(config, 0, (i, i + 1), plane_size[0])) for i in range(nx)])
    yc = np.array([0.5 * sum(_axis_edges_um(config, 1, (j, j + 1), plane_size[1])) for j in range(ny)])
    zc = np.array([0.5 * sum(_axis_edges_um(config, 2, (k, k + 1), plane_size[2])) for k in range(nz)])
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
    lo, hi = float(eps.min()), float(eps.max())
    if hi - lo < 1e-6:
        return None
    return go.Isosurface(
        x=X.ravel(),
        y=Y.ravel(),
        z=Z.ravel(),
        value=eps.ravel(),
        isomin=lo + 0.5 * (hi - lo),
        isomax=hi,
        surface_count=2,
        opacity=0.5,
        colorscale="Viridis",
        showscale=True,
        # A compact colorbar pinned to the right edge, clearly labelled — the material isosurface is
        # what reveals the *true* geometry (e.g. the ring), since objects themselves draw as boxes.
        colorbar=dict(title=dict(text="εᵣ", side="right"), len=0.55, thickness=14, x=1.0, xpad=0),
        name="ε (material)",
        showlegend=True,
        legendgroup=_CAT_GEOMETRY,
        legendgrouptitle=dict(text=_CAT_GEOMETRY),
        caps=dict(x_show=False, y_show=False, z_show=False),
    )


def to_plotly_json(fig) -> dict:
    """Return a JSON-safe ``dict`` (figure data + layout) for the web/AG-UI front end."""
    return json.loads(fig.to_json())
