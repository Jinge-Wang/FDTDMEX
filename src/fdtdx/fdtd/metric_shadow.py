"""Metric (continuous) shadow of the object placement solve.

The constraint solver in :mod:`fdtdx.fdtd.initialization` resolves every object onto whole grid
cells: a requested size is rounded with ``length_to_cell_count`` and the resulting box is snapped
with ``bounds_for_center`` / ``bounds_for_anchor``. That is what the simulation runs on, and this
module does not change it.

What this module adds is a second, float-valued reading of the *same* solve: for every object and
every axis, the extent and position the user actually asked for, in metres, before any rounding.
This "metric shadow" is what the Yee-point material sampler draws the continuous shapes from
(``SimulationConfig.material_sampling="yee"``), and it is what the placement report compares the
realised integer box against.

Resolution rules, one per source of placement information:

============================  ====================================================================
requested through             metric value
============================  ====================================================================
``partial_real_shape``        the size in metres, exactly as given
``partial_grid_shape``        no metric intent — falls back to the placed box
``partial_real_position``     centre = position + domain centre, exactly
``RealCoordinateConstraint``  the named side sits exactly on the given coordinate
``GridCoordinateConstraint``  the named side sits on ``grid.edges(axis)[coordinate]`` (exact)
``PositionConstraint``        anchor arithmetic in metres, no ``bounds_for_anchor`` search
``SizeConstraint``            ``other_metric_length * proportion + offset``, unrounded
``SizeExtensionConstraint``   the other object's metric anchor, or the domain edge
nothing of the above          the placed box: ``(edges[b0], edges[b1])``
============================  ====================================================================

An axis whose size is known but whose position is not is centred on the placed box's centre, so the
continuous shape sits where the integer solver put it and only its extent is restored.

Mirror symmetry (``config.symmetry``) clips the integer boxes after the solve, which the metric
arithmetic above does not model. With symmetry active every object therefore falls back to its
placed (clipped) box; ``material_sampling="yee"`` rejects that combination outright.
"""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from fdtdx.config import SimulationConfig
from fdtdx.objects.object import (
    GridCoordinateConstraint,
    PositionConstraint,
    RealCoordinateConstraint,
    SimulationObject,
    SizeConstraint,
    SizeExtensionConstraint,
)
from fdtdx.objects.static_material.static import SimulationVolume

MetricBounds3D = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

_MAX_ITER = 20


@dataclass
class _AxisShadow:
    """Per-object, per-axis state of the metric solve."""

    lo: float | None = None
    hi: float | None = None
    size: float | None = None
    size_source: str = "box_fallback"
    position_source: str = "box_fallback"

    def set_size(self, value: float, source: str) -> bool:
        if self.size is not None:
            return False
        self.size = float(value)
        self.size_source = source
        return True

    def set_bound(self, side: int, value: float, source: str) -> bool:
        current = self.lo if side == 0 else self.hi
        if current is not None:
            return False
        if side == 0:
            self.lo = float(value)
        else:
            self.hi = float(value)
        if self.position_source == "box_fallback":
            self.position_source = source
        return True


def _anchor(lo: float, hi: float, position: float) -> float:
    """Metric form of ``RectilinearGrid.anchor_coordinate`` (no grid search)."""
    return lo + 0.5 * (position + 1.0) * (hi - lo)


def _grid_margin_length(config: SimulationConfig, margin: int) -> float:
    """Convert an index-space margin into metres using the local (uniform) cell width.

    Non-zero index-space margins are rejected by the integer solver on non-uniform grids, so the
    single uniform spacing is the only case that can reach here.
    """
    if not margin:
        return 0.0
    return float(margin) * config.uniform_spacing()


def resolve_metric_shadow(
    object_list: Sequence[SimulationObject],
    constraints: Sequence[Any],
    config: SimulationConfig,
    resolved_slices: dict[str, Any],
) -> dict[str, MetricBounds3D]:
    """Resolve the continuous metric extent of every object alongside its placed integer box.

    Args:
        object_list (Sequence[SimulationObject]): The objects handed to ``place_objects``.
        constraints (Sequence[Any]): The constraint list handed to ``place_objects``.
        config (SimulationConfig): Configuration whose grid is already resolved.
        resolved_slices (dict[str, Any]): The integer solve's ``{name: ((b0, b1), ...)}`` result.

    Returns:
        dict[str, MetricBounds3D]: Per object name, the per-axis ``(lower, upper)`` metric bounds.
    """
    shadows = resolve_metric_shadow_detailed(object_list, constraints, config, resolved_slices)
    return {name: _to_bounds(axes) for name, axes in shadows.items()}


def _to_bounds(axes: list[_AxisShadow]) -> MetricBounds3D:
    out = []
    for sh in axes:
        assert sh.lo is not None and sh.hi is not None
        out.append((sh.lo, sh.hi))
    return (out[0], out[1], out[2])


def resolve_metric_shadow_detailed(
    object_list: Sequence[SimulationObject],
    constraints: Sequence[Any],
    config: SimulationConfig,
    resolved_slices: dict[str, Any],
) -> dict[str, list[_AxisShadow]]:
    """Same solve as :func:`resolve_metric_shadow`, keeping the per-axis provenance."""
    grid = config.resolved_grid
    if grid is None:
        raise ValueError("resolve_metric_shadow requires a resolved RectilinearGrid.")
    edges = [np.asarray(grid.edges(axis), dtype=float) for axis in range(3)]
    domain_center = [0.5 * (float(e[0]) + float(e[-1])) for e in edges]

    placed = {name: obj for name, obj in ((o.name, o) for o in object_list) if name in resolved_slices}
    shadows: dict[str, list[_AxisShadow]] = {name: [_AxisShadow() for _ in range(3)] for name in placed}

    if config.has_symmetry:
        # Symmetry clips the boxes after the solve; the metric arithmetic below does not model that.
        _apply_box_fallback(shadows, placed, resolved_slices, edges)
        return shadows

    for name, obj in placed.items():
        if isinstance(obj, SimulationVolume):
            for axis in range(3):
                sh = shadows[name][axis]
                sh.lo = float(edges[axis][0])
                sh.hi = float(edges[axis][-1])
                sh.size = sh.hi - sh.lo
                sh.size_source = "domain"
                sh.position_source = "domain"
            continue
        for axis in range(3):
            sh = shadows[name][axis]
            if obj.partial_grid_shape[axis] is not None:
                sh.size_source = "grid_shape"
            elif obj.partial_real_shape[axis] is not None:
                sh.set_size(float(obj.partial_real_shape[axis]), "partial_real_shape")

    for _ in range(_MAX_ITER):
        changed = False
        changed |= _apply_static_positions(shadows, placed, domain_center)
        changed |= _link_size_and_bounds(shadows)
        for constraint in constraints:
            if constraint.object not in shadows:
                continue
            if isinstance(constraint, GridCoordinateConstraint):
                changed |= _apply_grid_coordinate(shadows, constraint, edges)
            elif isinstance(constraint, RealCoordinateConstraint):
                changed |= _apply_real_coordinate(shadows, constraint)
            elif isinstance(constraint, PositionConstraint):
                changed |= _apply_position(shadows, constraint, config)
            elif isinstance(constraint, SizeConstraint):
                changed |= _apply_size(shadows, constraint, config)
            elif isinstance(constraint, SizeExtensionConstraint):
                changed |= _apply_size_extension(shadows, constraint, config, edges)
        changed |= _link_size_and_bounds(shadows)
        if not changed:
            break

    _apply_box_fallback(shadows, placed, resolved_slices, edges)
    return shadows


def _apply_static_positions(
    shadows: dict[str, list[_AxisShadow]],
    placed: dict[str, SimulationObject],
    domain_center: list[float],
) -> bool:
    changed = False
    for name, obj in placed.items():
        positions = getattr(obj, "partial_real_position", None)
        if positions is None:
            continue
        for axis in range(3):
            if positions[axis] is None:
                continue
            sh = shadows[name][axis]
            if sh.size is None or (sh.lo is not None and sh.hi is not None):
                continue
            center = float(positions[axis]) + domain_center[axis]
            changed |= sh.set_bound(0, center - 0.5 * sh.size, "partial_real_position")
            changed |= sh.set_bound(1, center + 0.5 * sh.size, "partial_real_position")
    return changed


def _link_size_and_bounds(shadows: dict[str, list[_AxisShadow]]) -> bool:
    changed = False
    for axes in shadows.values():
        for sh in axes:
            if sh.size is not None and sh.lo is not None and sh.hi is None:
                sh.hi = sh.lo + sh.size
                changed = True
            elif sh.size is not None and sh.hi is not None and sh.lo is None:
                sh.lo = sh.hi - sh.size
                changed = True
            elif sh.size is None and sh.lo is not None and sh.hi is not None:
                sh.size = sh.hi - sh.lo
                sh.size_source = "bounds"
                changed = True
    return changed


def _apply_grid_coordinate(
    shadows: dict[str, list[_AxisShadow]],
    constraint: GridCoordinateConstraint,
    edges: list[np.ndarray],
) -> bool:
    changed = False
    for axis_idx, axis in enumerate(constraint.axes):
        side = 0 if constraint.sides[axis_idx] == "-" else 1
        coord = float(edges[axis][constraint.coordinates[axis_idx]])
        changed |= shadows[constraint.object][axis].set_bound(side, coord, "grid_coordinate")
    return changed


def _apply_real_coordinate(
    shadows: dict[str, list[_AxisShadow]],
    constraint: RealCoordinateConstraint,
) -> bool:
    changed = False
    for axis_idx, axis in enumerate(constraint.axes):
        side = 0 if constraint.sides[axis_idx] == "-" else 1
        coord = float(constraint.coordinates[axis_idx])
        changed |= shadows[constraint.object][axis].set_bound(side, coord, "real_coordinate")
    return changed


def _apply_position(
    shadows: dict[str, list[_AxisShadow]],
    constraint: PositionConstraint,
    config: SimulationConfig,
) -> bool:
    if constraint.other_object not in shadows:
        return False
    changed = False
    for axis_idx, axis in enumerate(constraint.axes):
        other = shadows[constraint.other_object][axis]
        own = shadows[constraint.object][axis]
        if other.lo is None or other.hi is None or own.size is None:
            continue
        if own.lo is not None and own.hi is not None:
            continue
        anchor = _anchor(other.lo, other.hi, constraint.other_object_positions[axis_idx])
        margin = constraint.margins[axis_idx]
        if margin is not None:
            anchor += float(margin)
        anchor += _grid_margin_length(config, constraint.grid_margins[axis_idx])
        lo = anchor - 0.5 * (constraint.object_positions[axis_idx] + 1.0) * own.size
        changed |= own.set_bound(0, lo, "position_constraint")
        changed |= own.set_bound(1, lo + own.size, "position_constraint")
    return changed


def _apply_size(
    shadows: dict[str, list[_AxisShadow]],
    constraint: SizeConstraint,
    config: SimulationConfig,
) -> bool:
    if constraint.other_object not in shadows:
        return False
    changed = False
    for axis_idx, axis in enumerate(constraint.axes):
        other = shadows[constraint.other_object][constraint.other_axes[axis_idx]]
        if other.lo is None or other.hi is None:
            continue
        target = (other.hi - other.lo) * constraint.proportions[axis_idx]
        if constraint.offsets[axis_idx] is not None:
            target += float(constraint.offsets[axis_idx])
        target += _grid_margin_length(config, constraint.grid_offsets[axis_idx])
        changed |= shadows[constraint.object][axis].set_size(target, "size_constraint")
    return changed


def _apply_size_extension(
    shadows: dict[str, list[_AxisShadow]],
    constraint: SizeExtensionConstraint,
    config: SimulationConfig,
    edges: list[np.ndarray],
) -> bool:
    side = 0 if constraint.direction == "-" else 1
    own = shadows[constraint.object][constraint.axis]
    if constraint.other_object is None:
        coord = float(edges[constraint.axis][0] if side == 0 else edges[constraint.axis][-1])
        return own.set_bound(side, coord, "size_extension")
    if constraint.other_object not in shadows:
        return False
    other = shadows[constraint.other_object][constraint.axis]
    if other.lo is None or other.hi is None:
        return False
    coord = _anchor(other.lo, other.hi, constraint.other_position)
    if constraint.offset is not None:
        coord += float(constraint.offset)
    coord += _grid_margin_length(config, constraint.grid_offset)
    return own.set_bound(side, coord, "size_extension")


def _apply_box_fallback(
    shadows: dict[str, list[_AxisShadow]],
    placed: dict[str, SimulationObject],
    resolved_slices: dict[str, Any],
    edges: list[np.ndarray],
) -> None:
    """Fill every still-unresolved axis from the placed integer box."""
    for name in placed:
        box = resolved_slices[name]
        for axis in range(3):
            sh = shadows[name][axis]
            b0, b1 = box[axis]
            box_lo = float(edges[axis][b0])
            box_hi = float(edges[axis][b1])
            if sh.lo is not None and sh.hi is not None:
                continue
            if sh.size is not None:
                # Size known but no metric position: keep the box's centre, restore the extent.
                center = 0.5 * (box_lo + box_hi)
                sh.lo = center - 0.5 * sh.size
                sh.hi = center + 0.5 * sh.size
                sh.position_source = "box_center"
            else:
                sh.lo = box_lo
                sh.hi = box_hi
                sh.size = box_hi - box_lo
                sh.position_source = "box_fallback"
                sh.size_source = "box_fallback"


def build_placement_report(
    object_list: Sequence[SimulationObject],
    resolved_slices: dict[str, Any],
    shadows: dict[str, list[_AxisShadow]],
    config: SimulationConfig,
) -> list[dict[str, Any]]:
    """Build one report row per object per axis, comparing the request against the placed box.

    Args:
        object_list (Sequence[SimulationObject]): Objects handed to ``place_objects``.
        resolved_slices (dict[str, Any]): The integer solve's slices.
        shadows (dict[str, list[_AxisShadow]]): Output of :func:`resolve_metric_shadow_detailed`.
        config (SimulationConfig): Configuration with a resolved grid.

    Returns:
        list[dict[str, Any]]: Rows with the requested and realised size/centre per axis.
    """
    grid = config.resolved_grid
    if grid is None:
        raise ValueError("build_placement_report requires a resolved RectilinearGrid.")
    edges = [np.asarray(grid.edges(axis), dtype=float) for axis in range(3)]
    rows: list[dict[str, Any]] = []
    for obj in object_list:
        if obj.name not in shadows:
            continue
        box = resolved_slices[obj.name]
        for axis in range(3):
            sh = shadows[obj.name][axis]
            b0, b1 = box[axis]
            box_lo = float(edges[axis][b0])
            box_hi = float(edges[axis][b1])
            assert sh.lo is not None and sh.hi is not None
            rows.append(
                {
                    "name": obj.name,
                    "type": type(obj).__name__,
                    "axis": axis,
                    "requested_size": sh.hi - sh.lo,
                    "realised_box_size": box_hi - box_lo,
                    "requested_center": 0.5 * (sh.lo + sh.hi),
                    "realised_box_center": 0.5 * (box_lo + box_hi),
                    "size_source": sh.size_source,
                    "position_source": sh.position_source,
                    "cells": b1 - b0,
                }
            )
    return rows


def format_placement_report(rows: Sequence[dict[str, Any]], tolerance: float = 1e-12) -> str:
    """Format the rows whose requested extent differs from the realised box, as one text table.

    Args:
        rows (Sequence[dict[str, Any]]): Rows from :func:`build_placement_report`.
        tolerance (float): Size difference below which a row is considered exact and skipped.

    Returns:
        str: A table with one line per mismatching row, or an empty string when all rows match.
    """
    interesting = [
        r
        for r in rows
        if abs(r["requested_size"] - r["realised_box_size"]) > tolerance
        or abs(r["requested_center"] - r["realised_box_center"]) > tolerance
    ]
    if not interesting:
        return ""
    header = f"{'object':<28}{'ax':>3}  {'requested nm':>14}{'placed nm':>12}{'d_center nm':>14}  {'cells':>6}  source"
    lines = [header, "-" * len(header)]
    for r in interesting:
        lines.append(
            f"{r['name'][:28]:<28}{r['axis']:>3}  "
            f"{r['requested_size'] * 1e9:>14.3f}{r['realised_box_size'] * 1e9:>12.3f}"
            f"{(r['requested_center'] - r['realised_box_center']) * 1e9:>14.3f}  "
            f"{r['cells']:>6}  {r['size_source']}/{r['position_source']}"
        )
    return "\n".join(lines)
