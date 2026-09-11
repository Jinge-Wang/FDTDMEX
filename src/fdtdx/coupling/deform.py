"""The shape channel: a deformation moves the interfaces, not only the material inside them.

The material contribution of a mechanical deformation — a strained crystal's changed permittivity —
goes through :mod:`fdtdx.coupling.perturb`. The *geometric* contribution is that the interfaces
themselves move, and for an extruded polygon that is a change of its vertex array and nothing else:
``contains``, ``normal_at``, ``box_fill_fraction`` and therefore the sub-pixel blend are all derived
from those vertices.

One detail of :class:`~fdtdx.objects.static_material.polygon.ExtrudedPolygon` makes a naive vertex
update wrong. The class derives its cross-sectional size from the vertex bounding box and expects
the vertices centred on that box, so a displacement that moves the polygon bodily has nowhere to go
and would be silently swallowed. It is returned here instead, as a translation the caller puts into
the placement constraint.

The other cost is not visible in the arrays and has to be stated: re-rasterising a displaced polygon
changes which cells the interface cuts, so the staircase error no longer cancels between a displaced
run and its reference the way it does for a pure material perturbation. A two-run difference built
on displaced geometry is only as good as its sub-pixel smoothing.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def displace_polygon(
    obj: Any,
    vertex_map: np.ndarray | Callable[[np.ndarray], np.ndarray],
) -> tuple[Any, np.ndarray]:
    """A new extruded polygon with moved vertices, plus the net translation it could not absorb.

    Args:
        obj (ExtrudedPolygon): The polygon to move. Its ``vertices`` are ``(N, 2)`` in metres, in
            the object's own frame (the bounding-box centre is the origin), ordered
            ``(horizontal_axis, vertical_axis)``.
        vertex_map (np.ndarray | Callable): The **new vertex positions**, either as an ``(N, 2)``
            array in that same frame or as a callable taking the current vertices and returning
            them. A sampled displacement field is applied by passing ``obj.vertices + u``; a shape
            parameterisation passes its own vertices, with no finite-element solve in between.

    Returns:
        tuple: ``(polygon, translation)``. ``polygon`` is a new object with the moved, re-centred
        vertices and a cross-sectional size taken from their bounding box; ``translation`` is a
        ``(3,)`` array in metres on the grid axes, zero on the extrusion axis, holding the motion of
        the bounding-box centre. Add it to the object's placement constraint — the polygon itself
        cannot carry it.

    Raises:
        TypeError: If ``obj`` is not an :class:`ExtrudedPolygon`.
        ValueError: If the new vertices do not have the shape of the old ones, are not finite, or
            collapse the polygon onto a line.
    """
    from fdtdx.objects.static_material.polygon import ExtrudedPolygon

    if not isinstance(obj, ExtrudedPolygon):
        raise TypeError(f"displace_polygon needs an ExtrudedPolygon, got {type(obj).__name__}")
    old = np.asarray(obj.vertices, dtype=np.float64).reshape(-1, 2)
    moved = vertex_map if isinstance(vertex_map, np.ndarray) else vertex_map(old)
    new = np.asarray(moved, dtype=np.float64)
    if new.shape != old.shape:
        raise ValueError(f"the vertex map returned shape {new.shape}, the polygon has {old.shape}")
    if not np.all(np.isfinite(new)):
        raise ValueError("the displaced vertices contain non-finite values")

    old_center = 0.5 * (old.min(axis=0) + old.max(axis=0))
    new_center = 0.5 * (new.min(axis=0) + new.max(axis=0))
    size = new.max(axis=0) - new.min(axis=0)
    if np.any(size <= 0.0):
        raise ValueError(f"the displaced polygon has no extent along an axis (bounding box {size})")

    real_shape = list(obj.partial_real_shape)
    real_shape[obj.horizontal_axis] = float(size[0])
    real_shape[obj.vertical_axis] = float(size[1])
    polygon = obj.aset("partial_real_shape", tuple(real_shape)).aset("vertices", new - new_center)

    translation = np.zeros(3, dtype=np.float64)
    shift = new_center - old_center
    translation[obj.horizontal_axis] = float(shift[0])
    translation[obj.vertical_axis] = float(shift[1])
    return polygon, translation
