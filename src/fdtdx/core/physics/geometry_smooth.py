"""Kottke/Farjadpour sub-pixel smoothing on the Yee pixels (``material_sampling="yee_smooth"``).

Stage A (:mod:`fdtdx.core.physics.geometry_raster`) puts the exact continuous geometry on the grid,
but still samples it at a single point per field component, so an interface is staircased and the
error is first order in the cell size. This module removes that first-order term.

**The pixel.** Every E component owns a box centred on its own sample point: the primal cell on the
axes where the component sits at a cell centre, and the dual cell on the axes where it sits at an
edge. That is not a free choice — the dual width ``0.5*(w[i-1] + w[i])`` is the metric the backward
difference already divides by when it produces a quantity at ``e_a[i]``
(:mod:`fdtdx.core.physics.curl`), so the pixel is the control volume the update integrates over.
On a uniform grid every pixel is a cube of side ``h`` centred on the sample point.

**Which pixels are touched.** The material is probed at the pixel centre and at its eight corners.
One material: the pixel is uniform and keeps its point sample, bit for bit. Two materials: the pixel
straddles one interface and is smoothed. Three or more: there is no single planar interface for the
blend to describe, so the point sample is kept — the feature is under-resolved and the loader counts
those pixels. The eight corners of every pixel of one component come from a single lattice, so the
whole probe costs one extra raster pass per component rather than nine point tests per pixel.

**The blend.** With fill fraction ``f`` of the front material in the pixel,

    <eps> = f*eps_hi + (1-f)*eps_lo        <1/eps> = f/eps_hi + (1-f)/eps_lo

and unit interface normal ``n``, the effective *inverse* permittivity tensor is

    (1/eps)_eff = n n^T <1/eps> + (I - n n^T) / <eps>

i.e. the harmonic mean along the normal and the arithmetic mean in the interface plane. The default
diagonal tier writes entry ``(c, c)`` of that tensor at component ``c``'s own pixel, which is the
entry the elementwise update applies to ``E_c``; the optional full-tensor tier writes the whole row
``c`` into the 9-component layout. Conductivity and dispersion stay point-sampled.

**Why the normals are analytic.** The tensor depends on ``n`` linearly through ``n n^T``, so an
angular error ``delta`` leaves an ``O(h*delta)`` field error. A normal recovered from a fixed-size
fine raster has ``delta = O(1/S)`` independent of ``h`` and would hold the observed convergence order
near one; an analytic per-shape normal has no such floor. The fine-raster gradient inside the pixel
is kept only as a fallback for shapes that cannot answer analytically, and the loader reports how
many pixels used it.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from fdtdx.core.grid import RectilinearGrid
from fdtdx.core.physics.geometry_raster import (
    _TIE_NUDGE_FRACTION,
    E_OFFSETS,
    H_OFFSETS,
    Scene,
    _material_signature,
    front_indices,
)
from fdtdx.materials import compute_allowed_permeabilities, compute_allowed_permittivities
from fdtdx.objects.static_material.static import SimulationVolume

#: A fill fraction this close to 0 or 1 means the interface misses the pixel; keep the point sample.
_FILL_EPS = 1e-12

#: Relative tolerance on "this material is isotropic" before the pixel is left point-sampled.
_ISOTROPY_TOL = 1e-12


@dataclass
class SmoothingStats:
    """Counters the loader reports for one smoothing pass."""

    num_pixels: int = 0
    num_candidates: int = 0
    num_smoothed: int = 0
    num_three_material_fallbacks: int = 0
    num_gradient_normal_fallbacks: int = 0
    num_zero_normal_fallbacks: int = 0
    num_degenerate_fill_fallbacks: int = 0
    num_metal_skips: int = 0
    num_anisotropic_skips: int = 0
    num_supersampled_pixels: int = 0
    per_component_candidates: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Plain dictionary form, for ``info["yee_sampling_difference"]``."""
        out = {k: v for k, v in self.__dict__.items()}
        out["candidate_fraction"] = (self.num_candidates / self.num_pixels) if self.num_pixels else 0.0
        return out


# ---------------------------------------------------------------------------
# Pixel geometry
# ---------------------------------------------------------------------------


def invariant_axes(grid: RectilinearGrid) -> tuple[int, ...]:
    """Axes the simulation is invariant along, i.e. resolved by a single cell.

    fdtdx's 2-D convention is one cell with periodic boundaries on the third axis. A pixel must not
    resolve such an axis: doing so would put its corners outside the one-cell-thick objects and
    report a spurious second material everywhere.

    Half of the ``ignore_axes`` rule stated in full at
    :meth:`fdtdx.objects.static_material.static.StaticMultiMaterialObject.normal_at` (the other half
    is :func:`_spanning_axes`). This half is exact, not a heuristic.
    """
    return tuple(axis for axis in range(3) if grid.shape[axis] <= 1)


def pixel_axis_bounds(
    grid: RectilinearGrid,
    field_name: str,
    component: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-axis ``(lower, upper)`` arrays of the pixel boxes of one Yee component.

    An axis the simulation is invariant along returns a degenerate interval (``lower == upper`` at
    the cell centre), which every downstream overlap treats as a membership test.

    Args:
        grid (RectilinearGrid): The resolved simulation grid.
        field_name (str): ``"E"`` or ``"H"``.
        component (int): Component index 0, 1 or 2.

    Returns:
        list: Three ``(lower, upper)`` pairs of arrays of length ``grid.shape[axis]``.
    """
    offsets = _offsets(field_name)[component]
    ignore = invariant_axes(grid)
    bounds = []
    for axis in range(3):
        edges = np.asarray(grid.edges(axis), dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        if axis in ignore:
            bounds.append((centers.copy(), centers.copy()))
        elif offsets[axis] == 0.5:
            bounds.append((edges[:-1].copy(), edges[1:].copy()))
        else:
            widths = np.diff(edges)
            previous = np.concatenate([widths[:1], widths[:-1]])
            # Deliberate deviation from the M2 spec at i = 0. The spec asks for the *mirrored* dual
            # pixel [e[0] - w[0]/2, e[0] + w[0]/2], matching curl.py:31, which prepends widths[:1] so
            # the backward difference at the min edge divides by the full width w[0]. Here the box is
            # clipped to the domain instead, so the boundary pixel is only w[0]/2 wide -- half the
            # control volume the update actually integrates over. pixel_corner_coordinates clips its
            # probe corners the same way, so candidate detection and fill fraction stay consistent
            # with each other; the cost is that a material interface falling inside the first cell of
            # an axis would be smoothed over half the correct box. Every case in cases/ puts PML and a
            # spatially uniform background there, so no interface is ever that close to the boundary.
            # A scene with a real interface one cell from a non-PML (e.g. Bloch) boundary would need
            # the mirrored box and a background fill outside the domain.
            lower = np.clip(edges[:-1] - 0.5 * previous, edges[0], None)
            upper = edges[:-1] + 0.5 * widths
            bounds.append((lower, upper))
    return bounds


def pixel_corner_coordinates(
    grid: RectilinearGrid,
    field_name: str,
    component: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corner lattice of one component's pixels: one array per axis, ``N_a + 1`` long.

    Corner ``(dx, dy, dz)`` of the pixel of cell ``(i, j, k)`` is ``(q_x[i+dx], q_y[j+dy],
    q_z[k+dz])``, so a single :func:`fdtdx.core.physics.geometry_raster.front_indices` pass over this
    lattice yields all eight corners of all pixels as eight array slices. An invariant axis returns a
    length-1 array (the cell centre), which broadcasts.

    The top corner is pulled just inside the domain so that the resolver's tie nudge cannot push it
    past the upper face of an object that reaches the boundary.
    """
    bounds = pixel_axis_bounds(grid, field_name, component)
    ignore = invariant_axes(grid)
    coords = []
    for axis in range(3):
        lower, upper = bounds[axis]
        if axis in ignore:
            coords.append(lower[:1].copy())
            continue
        edges = np.asarray(grid.edges(axis), dtype=float)
        margin = 2.0 * _TIE_NUDGE_FRACTION * float(np.min(np.diff(edges)))
        corner = np.concatenate([lower, upper[-1:]])
        coords.append(np.clip(corner, edges[0], edges[-1] - margin))
    return coords[0], coords[1], coords[2]


def _offsets(field_name: str) -> tuple[tuple[float, float, float], ...]:
    if field_name == "E":
        return E_OFFSETS
    if field_name == "H":
        return H_OFFSETS
    raise ValueError(f"field must be 'E' or 'H', got {field_name!r}")


# ---------------------------------------------------------------------------
# Exact planar overlaps
# ---------------------------------------------------------------------------


def _quadrant_disk_area(a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    """Area of ``[0, a] x [0, b]`` inside a disk of ``radius`` centred at the origin, ``a, b >= 0``."""
    a = np.minimum(a, radius)
    b = np.minimum(b, radius)
    x_flat = np.sqrt(np.clip(radius**2 - b**2, 0.0, None))
    t0 = np.minimum(a, x_flat)

    def antiderivative(x):
        x = np.clip(x, 0.0, radius)
        return 0.5 * (x * np.sqrt(np.clip(radius**2 - x**2, 0.0, None)) + radius**2 * np.arcsin(x / radius))

    arc = np.where(a > t0, antiderivative(a) - antiderivative(t0), 0.0)
    return b * t0 + arc


def circle_rectangle_area(
    x0: np.ndarray,
    x1: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Exact area of the intersection of a disk at the origin with axis-aligned rectangles.

    Args:
        x0 (np.ndarray): Rectangle lower x bounds, relative to the disk centre.
        x1 (np.ndarray): Rectangle upper x bounds.
        y0 (np.ndarray): Rectangle lower y bounds.
        y1 (np.ndarray): Rectangle upper y bounds.
        radius (float): Disk radius.

    Returns:
        np.ndarray: Intersection areas, same shape as the inputs.
    """

    def corner(x, y):
        return np.sign(x) * np.sign(y) * _quadrant_disk_area(np.abs(x), np.abs(y), radius)

    return corner(x1, y1) - corner(x0, y1) - corner(x1, y0) + corner(x0, y0)


def _clip_halfplane(polygon: np.ndarray, axis: int, bound: float, keep_greater: bool) -> np.ndarray:
    """Sutherland-Hodgman clip of one polygon against one axis-aligned half-plane."""
    if polygon.shape[0] < 3:
        return polygon[:0]
    signed = polygon[:, axis] - bound
    if not keep_greater:
        signed = -signed
    inside = signed >= 0.0
    if inside.all():
        return polygon
    if not inside.any():
        return polygon[:0]
    next_vertex = np.roll(polygon, -1, axis=0)
    next_signed = np.roll(signed, -1)
    crosses = inside != np.roll(inside, -1)
    denom = signed - next_signed
    t = np.where(crosses & (denom != 0.0), signed / np.where(denom != 0.0, denom, 1.0), 0.0)
    intersections = polygon + t[:, None] * (next_vertex - polygon)
    stacked = np.empty((2 * polygon.shape[0], 2), dtype=float)
    stacked[0::2] = polygon
    stacked[1::2] = intersections
    keep = np.empty(2 * polygon.shape[0], dtype=bool)
    keep[0::2] = inside
    keep[1::2] = crosses
    return stacked[keep]


def _shoelace(polygon: np.ndarray) -> float:
    if polygon.shape[0] < 3:
        return 0.0
    following = np.roll(polygon, -1, axis=0)
    return 0.5 * abs(float(np.sum(polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1])))


def polygons_rectangle_area(
    polygons: Sequence[np.ndarray],
    x0: np.ndarray,
    x1: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
) -> np.ndarray:
    """Exact area of the union of polygons inside each axis-aligned rectangle.

    The polygons are assumed not to overlap each other (a GDS layer's polygon set), so the union area
    is the sum of the individual clipped areas. Each polygon is clipped against the rectangle's four
    half-planes and its area taken by the shoelace formula, which is exact for straight edges at any
    angle.

    Args:
        polygons (Sequence[np.ndarray]): ``(N, 2)`` vertex arrays.
        x0 (np.ndarray): Rectangle lower x bounds.
        x1 (np.ndarray): Rectangle upper x bounds.
        y0 (np.ndarray): Rectangle lower y bounds.
        y1 (np.ndarray): Rectangle upper y bounds.

    Returns:
        np.ndarray: Areas, same shape as the inputs.
    """
    x0 = np.asarray(x0, dtype=float)
    shape = x0.shape
    x0, x1 = x0.ravel(), np.asarray(x1, dtype=float).ravel()
    y0, y1 = np.asarray(y0, dtype=float).ravel(), np.asarray(y1, dtype=float).ravel()
    area = np.zeros(x0.shape[0], dtype=float)
    for raw in polygons:
        polygon = np.asarray(raw, dtype=float)
        if polygon.shape[0] > 1 and np.allclose(polygon[0], polygon[-1]):
            polygon = polygon[:-1]
        if polygon.shape[0] < 3:
            continue
        px0, px1 = polygon[:, 0].min(), polygon[:, 0].max()
        py0, py1 = polygon[:, 1].min(), polygon[:, 1].max()
        relevant = np.flatnonzero((x1 > px0) & (x0 < px1) & (y1 > py0) & (y0 < py1))
        for index in relevant:
            clipped = polygon
            for axis, bound, keep_greater in (
                (0, x0[index], True),
                (0, x1[index], False),
                (1, y0[index], True),
                (1, y1[index], False),
            ):
                clipped = _clip_halfplane(clipped, axis, bound, keep_greater)
                if clipped.shape[0] < 3:
                    break
            area[index] += _shoelace(clipped)
    return area.reshape(shape)


# ---------------------------------------------------------------------------
# The Kottke tensor
# ---------------------------------------------------------------------------


def kottke_inverse_permittivity(
    normal: np.ndarray,
    arithmetic: np.ndarray,
    harmonic: np.ndarray,
    component: int,
    full_tensor: bool,
) -> np.ndarray:
    """Row (or diagonal entry) ``component`` of the effective inverse permittivity tensor.

    Args:
        normal (np.ndarray): ``(M, 3)`` unit interface normals.
        arithmetic (np.ndarray): ``(M,)`` fill-weighted arithmetic mean ``<eps>``.
        harmonic (np.ndarray): ``(M,)`` fill-weighted mean of the inverse, ``<1/eps>``.
        component (int): Which row of the tensor to return.
        full_tensor (bool): Return the whole row ``(M, 3)`` instead of the single entry ``(M,)``.

    Returns:
        np.ndarray: ``(M,)`` when ``full_tensor`` is false, ``(M, 3)`` otherwise.
    """
    if not full_tensor:
        projection = normal[:, component] ** 2
        return projection * harmonic + (1.0 - projection) / arithmetic
    row = np.zeros((normal.shape[0], 3), dtype=float)
    for j in range(3):
        projection = normal[:, component] * normal[:, j]
        delta = 1.0 if j == component else 0.0
        row[:, j] = projection * harmonic + (delta - projection) / arithmetic
    return row


# ---------------------------------------------------------------------------
# Fill fractions
# ---------------------------------------------------------------------------


def _supersampled_fill(
    obj,
    lower: np.ndarray,
    upper: np.ndarray,
    supersample: int,
    degenerate: tuple[bool, bool, bool],
) -> np.ndarray:
    """Midpoint super-sampled occupancy of one object inside each box.

    Used where no analytic overlap is available (a sphere, a tapered sidewall). Midpoints sit at
    ``lo + (s + 0.5)/S * (hi - lo)``, the same convention the box path's fill fraction uses. An
    invariant axis contributes a single sample at the degenerate coordinate.
    """
    steps = []
    for axis in range(3):
        if degenerate[axis]:
            steps.append(np.zeros(1))
        else:
            steps.append((np.arange(supersample) + 0.5) / supersample)
    grids = np.meshgrid(*steps, indexing="ij")
    total = np.zeros(lower.shape[0], dtype=float)
    count = grids[0].size
    flat = [g.ravel() for g in grids]
    for sample in range(count):
        point = np.empty_like(lower)
        for axis in range(3):
            point[:, axis] = lower[:, axis] + flat[axis][sample] * (upper[:, axis] - lower[:, axis])
        total += np.asarray(obj.contains(point), dtype=float)
    return total / count


def _owner_fill_fraction(
    obj,
    lower: np.ndarray,
    upper: np.ndarray,
    supersample: int,
    degenerate: tuple[bool, bool, bool],
    stats: SmoothingStats,
) -> np.ndarray:
    """Fraction of each pixel covered by the owning object, analytic where the shape allows it."""
    analytic = obj.box_fill_fraction(lower, upper)
    if analytic is not None:
        return np.clip(np.asarray(analytic, dtype=float), 0.0, 1.0)
    stats.num_supersampled_pixels += int(lower.shape[0])
    return _supersampled_fill(obj, lower, upper, supersample, degenerate)


def _gradient_normal_from_fill(
    obj,
    lower: np.ndarray,
    upper: np.ndarray,
    supersample: int,
    degenerate: tuple[bool, bool, bool],
) -> np.ndarray:
    """Fine-raster normal inside a single pixel: the fallback when no analytic normal is available.

    The occupancy block is differenced across the pixel on each axis and normalised. The angular
    error of this normal is set by ``supersample`` alone and does not shrink with the cell size, so
    it caps the achievable convergence order at one — it exists so an exotic shape still runs, not as
    a substitute for the analytic normals.
    """
    normal = np.zeros((lower.shape[0], 3), dtype=float)
    for axis in range(3):
        if degenerate[axis]:
            continue
        half = 0.5 * (lower[:, axis] + upper[:, axis])
        low_lower, low_upper = lower.copy(), upper.copy()
        low_upper[:, axis] = half
        high_lower, high_upper = lower.copy(), upper.copy()
        high_lower[:, axis] = half
        low = _supersampled_fill(obj, low_lower, low_upper, supersample, degenerate)
        high = _supersampled_fill(obj, high_lower, high_upper, supersample, degenerate)
        normal[:, axis] = low - high
    length = np.linalg.norm(normal, axis=-1)
    safe = length > 0.0
    return np.where(safe[:, None], normal / np.where(safe, length, 1.0)[:, None], 0.0)


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


def _material_value_classes(scene: Scene) -> np.ndarray:
    """Map every global material index onto the first index with the same material definition.

    Two objects can carry the same material under two names (a carve-out drawn in the background
    material, say). Those are one material physically, and counting them as two would report a
    spurious interface. This is the reimplementation of Meep's ``material_type_equal`` test.
    """
    from fdtdx.materials import compute_ordered_names

    names = compute_ordered_names(scene.materials)
    signatures = [_material_signature(scene.materials[name]) for name in names]
    classes = np.arange(len(names), dtype=np.int32)
    seen: dict[tuple, int] = {}
    for index, signature in enumerate(signatures):
        classes[index] = seen.setdefault(signature, index)
    return classes


def _property_tensors(scene: Scene, property_kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scalar value, isotropy flag and full 3x3 tensor of one property, per global material.

    Args:
        scene (Scene): The scene whose global material list is read.
        property_kind (str): ``"permittivity"`` or ``"permeability"``.

    Returns:
        tuple: ``(scalar, isotropic, tensor)`` of shapes ``(M,)``, ``(M,)`` and ``(M, 3, 3)``. The
        scalar is the ``xx`` entry; it is what the isotropic fast path blends and it is only
        meaningful where ``isotropic`` is true.

    Raises:
        ValueError: If ``property_kind`` is not a smoothed property.
    """
    if property_kind == "permittivity":
        raw = compute_allowed_permittivities(scene.materials)
    elif property_kind == "permeability":
        raw = compute_allowed_permeabilities(scene.materials)
    else:
        raise ValueError(f"property_kind must be 'permittivity' or 'permeability', got {property_kind!r}")
    table = np.asarray(raw, dtype=float)
    diagonal = table[:, (0, 4, 8)]
    off_diagonal = table[:, (1, 2, 3, 5, 6, 7)]
    scale = np.maximum(np.max(np.abs(diagonal), axis=1), 1.0)
    isotropic = (np.ptp(diagonal, axis=1) <= _ISOTROPY_TOL * scale) & (
        np.max(np.abs(off_diagonal), axis=1) <= _ISOTROPY_TOL * scale
    )
    return diagonal[:, 0], isotropic, table.reshape(table.shape[0], 3, 3)


def _spanning_axes(entry, grid: RectilinearGrid) -> tuple[int, ...]:
    """Axes on which an entry covers the whole domain, so its "caps" are the domain boundary.

    Half of the ``ignore_axes`` rule stated in full at
    :meth:`fdtdx.objects.static_material.static.StaticMultiMaterialObject.normal_at` (the other half
    is :func:`invariant_axes`). This half is a heuristic: an object that genuinely ends at the
    domain boundary and also has a real face there loses that face.
    """
    edges = [np.asarray(grid.edges(axis), dtype=float) for axis in range(3)]
    spanning = []
    for axis in range(3):
        tolerance = 1e-9 * max(float(edges[axis][-1] - edges[axis][0]), 1e-30)
        lower, upper = entry.bounds[axis]
        if lower <= edges[axis][0] + tolerance and upper >= edges[axis][-1] - tolerance:
            spanning.append(axis)
    return tuple(spanning)


def _write_component_entries(
    target: np.ndarray,
    cells: tuple[np.ndarray, ...],
    blend: np.ndarray,
    component: int,
    full_tensor: bool,
) -> None:
    """Write one component's smoothed entries into the assembled inverse-property array.

    The single place a smoothed value reaches the material array. Meep evaluates the two
    off-diagonal entries of row ``c`` on a half-cell-shifted control volume
    (``gv.dV(here - shift1, ...)``) rather than on component ``c``'s own pixel; moving them there is
    a separate change that replaces this function and nothing else.

    Args:
        target (np.ndarray): ``(3 or 9, Nx, Ny, Nz)`` array, modified in place.
        cells (tuple): The three index arrays of the pixels being written.
        blend (np.ndarray): ``(K,)`` diagonal entries or ``(K, 3)`` rows.
        component (int): Which component's lattice is being written.
        full_tensor (bool): Whether ``target`` carries 9 components.
    """
    if full_tensor:
        for j in range(3):
            target[(3 * component + j, *cells)] = blend[:, j]
    else:
        target[(component, *cells)] = blend


def _smooth_component_lattice(
    scene: Scene,
    grid: RectilinearGrid,
    field: str,
    component: int,
    front_material: np.ndarray,
    front_owner: np.ndarray,
    classes: np.ndarray,
    scalar: np.ndarray,
    isotropic: np.ndarray,
    target: np.ndarray,
    full_tensor: bool,
    supersample: int,
    stats: SmoothingStats,
    warned_anisotropic: list[bool],
) -> None:
    """Smooth one component's lattice in place: probe, classify, blend, write.

    Args:
        scene (Scene): The scene from :func:`fdtdx.core.physics.geometry_raster.build_scene`.
        grid (RectilinearGrid): The resolved simulation grid.
        field (str): ``"E"`` or ``"H"`` — which Yee lattice this component sits on.
        component (int): Component index 0, 1 or 2.
        front_material (np.ndarray): ``(Nx, Ny, Nz)`` point-sampled global material index.
        front_owner (np.ndarray): ``(Nx, Ny, Nz)`` point-sampled winning entry index.
        classes (np.ndarray): Material index to value-class map from :func:`_material_value_classes`.
        scalar (np.ndarray): Per-material scalar property value.
        isotropic (np.ndarray): Per-material isotropy flag.
        target (np.ndarray): The inverse-property array, modified in place.
        full_tensor (bool): Write whole rows instead of the diagonal entry.
        supersample (int): Samples per axis where no analytic overlap or normal is available.
        stats (SmoothingStats): Counters, accumulated across components.
        warned_anisotropic (list): One-element mutable flag, so the warning fires once per pass.
    """
    ignore_global = invariant_axes(grid)
    degenerate_axis = tuple(axis in ignore_global for axis in range(3))

    bounds = pixel_axis_bounds(grid, field, component)
    corners = pixel_corner_coordinates(grid, field, component)
    corner_material, corner_owner = front_indices(scene, corners)

    shape = front_material.shape
    probe_material = [classes[front_material]]
    probe_owner = [front_owner]
    for dx in range(2):
        for dy in range(2):
            for dz in range(2):
                index = tuple(
                    slice(0, 1) if degenerate_axis[axis] else slice(d, d + shape[axis])
                    for axis, d in enumerate((dx, dy, dz))
                )
                probe_material.append(np.broadcast_to(classes[corner_material[index]], shape))
                probe_owner.append(np.broadcast_to(corner_owner[index], shape))
    stacked = np.stack(probe_material, axis=0)
    ordered = np.sort(stacked, axis=0)
    distinct = 1 + np.count_nonzero(np.diff(ordered, axis=0), axis=0)

    stats.num_pixels += int(np.prod(shape))
    candidate = distinct == 2
    stats.num_three_material_fallbacks += int(np.count_nonzero(distinct >= 3))
    count = int(np.count_nonzero(candidate))
    stats.num_candidates += count
    stats.per_component_candidates.append(count)
    if count == 0:
        return

    cells = np.nonzero(candidate)
    probe_classes = stacked[:, candidate]
    owners = np.stack(probe_owner, axis=0)[:, candidate]
    low_class = ordered[0][candidate]
    high_class = ordered[-1][candidate]

    # The owner is the highest-priority object among the nine probes: it is the shape whose
    # surface bounds the front material inside the pixel, so it supplies both the fill fraction
    # and the normal. This is fdtdx's write order read as Meep reads its object ids.
    winner = np.argmax(owners, axis=0)
    owner_entry = owners[winner, np.arange(count)]
    owner_class = probe_classes[winner, np.arange(count)]
    other_class = np.where(owner_class == low_class, high_class, low_class)

    lower = np.stack([bounds[axis][0][cells[axis]] for axis in range(3)], axis=-1)
    upper = np.stack([bounds[axis][1][cells[axis]] for axis in range(3)], axis=-1)
    center = 0.5 * (lower + upper)

    fill = np.zeros(count, dtype=float)
    normal = np.zeros((count, 3), dtype=float)
    for entry_index in np.unique(owner_entry):
        if entry_index < 0:
            continue
        selected = owner_entry == entry_index
        entry = scene.entries[int(entry_index)]
        if isinstance(entry.obj, SimulationVolume):
            continue
        fill[selected] = _owner_fill_fraction(
            entry.obj, lower[selected], upper[selected], supersample, degenerate_axis, stats
        )
        ignore = tuple(sorted(set(ignore_global) | set(_spanning_axes(entry, grid))))
        local = np.asarray(entry.obj.normal_at(center[selected], ignore_axes=ignore), dtype=float)
        missing = np.linalg.norm(local, axis=-1) <= 0.0
        if missing.any():
            stats.num_gradient_normal_fallbacks += int(np.count_nonzero(missing))
            subset = np.nonzero(selected)[0][missing]
            local[missing] = _gradient_normal_from_fill(
                entry.obj, lower[subset], upper[subset], supersample, degenerate_axis
            )
        for axis in ignore_global:
            local[:, axis] = 0.0
        length = np.linalg.norm(local, axis=-1)
        safe = length > 0.0
        normal[selected] = np.where(safe[:, None], local / np.where(safe, length, 1.0)[:, None], 0.0)

    value_hi = scalar[owner_class]
    value_lo = scalar[other_class]

    usable = np.ones(count, dtype=bool)
    both_isotropic = isotropic[owner_class] & isotropic[other_class]
    stats.num_anisotropic_skips += int(np.count_nonzero(~both_isotropic))
    if not both_isotropic.all() and not warned_anisotropic[0]:
        warnings.warn(
            "material_sampling='yee_smooth' met an anisotropic material at an interface pixel. "
            "The Kottke blend implemented here assumes both sides are locally isotropic, so those "
            "pixels keep their point sample.",
            UserWarning,
            stacklevel=2,
        )
        warned_anisotropic[0] = True
    usable &= both_isotropic

    positive = (value_hi > 0.0) & (value_lo > 0.0)
    stats.num_metal_skips += int(np.count_nonzero(~positive))
    usable &= positive

    degenerate_fill = (fill <= _FILL_EPS) | (fill >= 1.0 - _FILL_EPS)
    stats.num_degenerate_fill_fallbacks += int(np.count_nonzero(degenerate_fill))
    usable &= ~degenerate_fill

    zero_normal = np.linalg.norm(normal, axis=-1) <= 0.0
    stats.num_zero_normal_fallbacks += int(np.count_nonzero(zero_normal))
    usable &= ~zero_normal

    if not usable.any():
        return
    written = tuple(axis_cells[usable] for axis_cells in cells)
    arithmetic = fill[usable] * value_hi[usable] + (1.0 - fill[usable]) * value_lo[usable]
    harmonic = fill[usable] / value_hi[usable] + (1.0 - fill[usable]) / value_lo[usable]
    blend = kottke_inverse_permittivity(normal[usable], arithmetic, harmonic, component, full_tensor)
    _write_component_entries(target, written, blend, component, full_tensor)
    stats.num_smoothed += int(np.count_nonzero(usable))


def smooth_property_on_yee_pixels(
    scene: Scene,
    grid: RectilinearGrid,
    field: str,
    property_kind: str,
    front_material: np.ndarray,
    front_owner: np.ndarray,
    inverse_property: np.ndarray,
    supersample: int,
    full_tensor: bool,
) -> tuple[np.ndarray, SmoothingStats]:
    """Overwrite the point-sampled inverse property at every two-material Yee pixel with its blend.

    One pass over the three lattices of one field. The pixel rule, the nine-probe candidate test,
    the fill fraction, the normal and the blend are identical for ``"E"``/permittivity and
    ``"H"``/permeability; only the lattice the pixels sit on and the material table differ.

    Args:
        scene (Scene): The scene from :func:`fdtdx.core.physics.geometry_raster.build_scene`.
        grid (RectilinearGrid): The resolved simulation grid.
        field (str): ``"E"`` or ``"H"``.
        property_kind (str): ``"permittivity"`` or ``"permeability"``.
        front_material (np.ndarray): ``(3, Nx, Ny, Nz)`` point-sampled global material index.
        front_owner (np.ndarray): ``(3, Nx, Ny, Nz)`` point-sampled winning entry index.
        inverse_property (np.ndarray): ``(3 or 9, Nx, Ny, Nz)`` point-sampled inverse property,
            modified in place and returned.
        supersample (int): Samples per axis where no analytic overlap or normal is available.
        full_tensor (bool): Write the full Kottke row instead of the diagonal entry. Must match the
            array's own tier: a 9-component array is always written as rows, because entry ``(c, c)``
            of a row-major 3x3 sits at index ``4*c``, not at ``c``.

    Returns:
        tuple: ``(inverse_property, stats)``.

    Raises:
        ValueError: If ``full_tensor`` disagrees with the array's component count.
    """
    expected = 9 if full_tensor else 3
    if inverse_property.shape[0] != expected:
        raise ValueError(
            f"full_tensor={full_tensor} needs a {expected}-component inverse {property_kind} array, "
            f"got {inverse_property.shape[0]}."
        )
    stats = SmoothingStats()
    classes = _material_value_classes(scene)
    scalar, isotropic, _ = _property_tensors(scene, property_kind)
    warned_anisotropic = [False]

    for component in range(3):
        _smooth_component_lattice(
            scene=scene,
            grid=grid,
            field=field,
            component=component,
            front_material=front_material[component],
            front_owner=front_owner[component],
            classes=classes,
            scalar=scalar,
            isotropic=isotropic,
            target=inverse_property,
            full_tensor=full_tensor,
            supersample=supersample,
            stats=stats,
            warned_anisotropic=warned_anisotropic,
        )

    return inverse_property, stats


def smooth_inverse_permittivity_on_yee_pixels(
    scene: Scene,
    grid: RectilinearGrid,
    front_material: np.ndarray,
    front_owner: np.ndarray,
    inv_permittivities: np.ndarray,
    supersample: int,
    full_tensor: bool,
) -> tuple[np.ndarray, SmoothingStats]:
    """Smooth the permittivity on the three E lattices; see :func:`smooth_property_on_yee_pixels`."""
    return smooth_property_on_yee_pixels(
        scene=scene,
        grid=grid,
        field="E",
        property_kind="permittivity",
        front_material=front_material,
        front_owner=front_owner,
        inverse_property=inv_permittivities,
        supersample=supersample,
        full_tensor=full_tensor,
    )
