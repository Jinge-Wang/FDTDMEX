import math
from itertools import pairwise
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.path import Path

from fdtdx import constants
from fdtdx.core.axis import get_transverse_axes
from fdtdx.core.jax.pytrees import TreeClass, autoinit, field, frozen_field, frozen_private_field, private_field
from fdtdx.core.misc import validate_symmetric_axis_cells


@autoinit
class UniformGrid(TreeClass):
    """Unresolved policy for a uniform rectilinear grid.

    ``UniformGrid`` is user intent, not the solver mesh itself.  It records the
    physical cell spacing while the final simulation shape may still be unknown.
    Object placement resolves this policy to a concrete ``RectilinearGrid`` once
    the volume shape is known.

    Keeping uniform spacing here avoids a second scalar discretization source on
    ``SimulationConfig``.  Uniform grids and explicitly non-uniform grids both
    enter the solver through the same realized ``RectilinearGrid`` structure.

    The grid origin is at the **center** of the simulation domain.  Edge arrays
    therefore span ``[-N/2 * spacing, +N/2 * spacing]`` along each axis, giving
    the domain symmetric negative and positive coordinates.  ``center`` shifts
    this physical center away from the geometric origin when non-zero.
    """

    #: Physical cell spacing in metres. Must be positive.
    spacing: float = frozen_field()
    #: Physical coordinate of the domain center in metres. Defaults to (0, 0, 0).
    center: tuple[float, float, float] = frozen_field(default=(0, 0, 0))

    def __post_init__(self):
        if self.spacing <= 0:
            raise ValueError(f"Uniform grid spacing must be positive, got {self.spacing}.")

    def resolve(self, shape: tuple[int, int, int]) -> "RectilinearGrid":
        """Return a concrete solver grid for ``shape``."""
        origin = (
            self.center[0] - shape[0] * self.spacing / 2.0,
            self.center[1] - shape[1] * self.spacing / 2.0,
            self.center[2] - shape[2] * self.spacing / 2.0,
        )
        return RectilinearGrid.uniform(shape=shape, spacing=self.spacing, origin=origin)

    @property
    def is_uniform(self) -> bool:
        """Uniform policies always represent equal cell widths."""
        return True

    @property
    def min_spacing(self) -> float:
        """Smallest cell width implied by this policy."""
        return self.spacing

    @property
    def uniform_spacing(self) -> float:
        """Scalar cell width in metres."""
        return self.spacing

    def axis_extent(self, axis: int, bounds: tuple[int, int]) -> float:
        """Physical length covered by an index interval on one axis."""
        del axis
        lower, upper = bounds
        return (upper - lower) * self.spacing

    def slice_extent(
        self, slice_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ) -> tuple[float, float, float]:
        """Physical side lengths covered by a 3D grid slice."""
        return (
            self.axis_extent(0, slice_tuple[0]),
            self.axis_extent(1, slice_tuple[1]),
            self.axis_extent(2, slice_tuple[2]),
        )

    def coord_to_index(self, axis: int, coord: float, snap: str = "nearest") -> int:
        """Map a physical coordinate to a uniform-grid edge index.

        Because unresolved policies do not yet know ``shape``, this helper uses
        a center-relative basis: ``coord`` is interpreted relative to
        ``self.center[axis]`` and the returned index is a center-relative edge
        offset. Use :meth:`RectilinearGrid.coord_to_index` on the resolved grid
        when you need absolute edge indices.
        """
        origin_offset = coord - self.center[axis]
        scaled = origin_offset / self.spacing
        if snap == "nearest":
            return round(scaled)
        if snap == "lower":
            return int(np.floor(scaled))
        if snap == "upper":
            return int(np.ceil(scaled))
        raise ValueError(f"Unknown snapping rule: {snap}")

    def length_to_cell_count(self, axis: int, length: float, snap: str = "nearest") -> int:
        """Convert a physical length to a uniform-grid cell count."""
        return self.coord_to_index(axis, self.center[axis] + length, snap=snap)

    def bounds_for_center(self, axis: int, center: float, size: int) -> tuple[int, int]:
        """Convert a physical center and grid size to edge bounds."""
        grid_center = self.coord_to_index(axis, center, snap="nearest")
        lower = round(grid_center - size / 2)
        return lower, lower + size

    def anchor_coordinate(self, axis: int, bounds: tuple[int, int], position: float) -> float:
        """Return a physical anchor coordinate inside a uniform interval."""
        lower, upper = bounds
        lower_coord = self.center[axis] + lower * self.spacing
        upper_coord = self.center[axis] + upper * self.spacing
        return lower_coord + 0.5 * (position + 1.0) * (upper_coord - lower_coord)

    def bounds_for_anchor(self, axis: int, size: int, anchor: float, position: float) -> tuple[int, int]:
        """Choose a uniform-grid interval from an object anchor."""
        anchor_cell = self.coord_to_index(axis, anchor, snap="nearest")
        offset = round(0.5 * (position + 1.0) * size)
        lower = anchor_cell - offset
        return lower, lower + size

    def cell_volume(self, slice_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> jax.Array:
        """Return per-cell volumes for a slice on this uniform policy."""
        shape = tuple(upper - lower for lower, upper in slice_tuple)
        return jnp.ones(shape) * self.spacing**3

    def face_area(self, axis: int, slice_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> jax.Array:
        """Return per-face areas for a slice on this uniform policy."""
        shape = tuple(upper - lower for lower, upper in slice_tuple)
        area_shape = tuple(shape[i] for i in range(3) if i != axis)
        return jnp.ones(area_shape) * self.spacing**2


@autoinit
class QuasiUniformGrid(TreeClass):
    """Unresolved policy for a rectilinear grid with independent per-axis spacings.

    ``QuasiUniformGrid`` generalises ``UniformGrid`` to allow different cell
    widths along x, y, and z while keeping each axis internally uniform.  This
    is sometimes called a *quasi-uniform* or *anisotropic-uniform* mesh: the
    grid is rectilinear and axis-aligned, but the aspect ratio is not 1 : 1 : 1.

    Like ``UniformGrid``, this is user intent rather than the solver mesh.
    Calling :meth:`resolve` converts the policy to a concrete
    ``RectilinearGrid`` once the simulation shape is known.  The resulting grid
    is centered at ``center`` so that coordinates span symmetrically into both
    negative and positive values along every axis.

    Example::

        grid = QuasiUniformGrid(dx=10e-9, dy=10e-9, dz=20e-9)
        resolved = grid.resolve(shape=(100, 100, 50))
        # x, y edges span [-500 nm, +500 nm]; z edges span [-500 nm, +500 nm]
    """

    #: Cell width along x in metres. Must be positive.
    dx: float = frozen_field()
    #: Cell width along y in metres. Must be positive.
    dy: float = frozen_field()
    #: Cell width along z in metres. Must be positive.
    dz: float = frozen_field()
    #: Physical coordinate of the domain center in metres. Defaults to (0, 0, 0).
    center: tuple[float, float, float] = frozen_field(default=(0, 0, 0))

    def __post_init__(self):
        for name, val in (("dx", self.dx), ("dy", self.dy), ("dz", self.dz)):
            if val <= 0:
                raise ValueError(f"QuasiUniformGrid spacing {name} must be positive, got {val}.")

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, shape: tuple[int, int, int]) -> "RectilinearGrid":
        """Return a concrete ``RectilinearGrid`` for ``shape``.

        Edge arrays are built independently for each axis using the per-axis
        spacing and the requested number of cells.  The domain is centered at
        ``self.center``.

        Args:
            shape: Number of cells in ``(x, y, z)``.

        Returns:
            A ``RectilinearGrid`` whose edge arrays are piecewise-uniform (one
            constant spacing per axis) and span symmetrically around
            ``self.center``.

        Raises:
            ValueError: If any axis has an odd cell count.  The center-origin
                convention requires even cell counts on every axis so the domain
                center always lands on a cell edge.  An odd count silently shifts
                which Yee component sits at object boundaries, changing the
                effective simulated length by one cell.
        """
        for axis, n in enumerate(shape):
            if n % 2 != 0:
                raise ValueError(
                    f"QuasiUniformGrid requires an even cell count on every axis (center-origin "
                    f"convention). Axis {axis} has {n} cells (odd). Adjust the simulation "
                    f"volume size so every axis has an even number of cells."
                )
        spacings = (self.dx, self.dy, self.dz)
        edge_arrays = []
        for a in range(3):
            s = spacings[a]
            n = shape[a]
            lower = self.center[a] - n * s / 2.0
            edge_arrays.append(lower + s * jnp.arange(n + 1))
        return RectilinearGrid(
            x_edges=edge_arrays[0],
            y_edges=edge_arrays[1],
            z_edges=edge_arrays[2],
        )

    # ------------------------------------------------------------------
    # Convenience properties (mirror UniformGrid's interface)
    # ------------------------------------------------------------------

    @property
    def is_uniform(self) -> bool:
        """True only when all three spacings are equal."""
        return self.dx == self.dy == self.dz

    @property
    def min_spacing(self) -> float:
        """Smallest cell width across all three axes in metres."""
        return min(self.dx, self.dy, self.dz)

    def axis_spacing(self, axis: int) -> float:
        """Return the cell width for a single axis."""
        return (self.dx, self.dy, self.dz)[axis]

    def axis_extent(self, axis: int, bounds: tuple[int, int]) -> float:
        """Physical length covered by an index interval on one axis."""
        lower, upper = bounds
        return (upper - lower) * self.axis_spacing(axis)

    def slice_extent(
        self, slice_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ) -> tuple[float, float, float]:
        """Physical side lengths covered by a 3D grid slice."""
        return (
            self.axis_extent(0, slice_tuple[0]),
            self.axis_extent(1, slice_tuple[1]),
            self.axis_extent(2, slice_tuple[2]),
        )

    def coord_to_index(self, axis: int, coord: float, snap: str = "nearest") -> int:
        """Map a physical coordinate to a grid edge index along ``axis``.

        Coordinates are measured from ``self.center[axis]``.
        """
        s = self.axis_spacing(axis)
        scaled = (coord - self.center[axis]) / s
        if snap == "nearest":
            return round(scaled)
        if snap == "lower":
            return int(np.floor(scaled))
        if snap == "upper":
            return int(np.ceil(scaled))
        raise ValueError(f"Unknown snapping rule: {snap}")

    def length_to_cell_count(self, axis: int, length: float, snap: str = "nearest") -> int:
        """Convert a physical length to a cell count along ``axis``."""
        return self.coord_to_index(axis, self.center[axis] + length, snap=snap)

    def cell_volume(self, slice_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> jax.Array:
        """Return per-cell volume weights broadcast to a 3D slice shape."""
        shape = tuple(upper - lower for lower, upper in slice_tuple)
        return jnp.ones(shape) * (self.dx * self.dy * self.dz)

    def face_area(self, axis: int, slice_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> jax.Array:
        """Return per-face area weights for a detector plane normal to ``axis``."""
        spacings = (self.dx, self.dy, self.dz)
        transverse = [a for a in range(3) if a != axis]
        shape = tuple(slice_tuple[a][1] - slice_tuple[a][0] for a in range(3))
        area_shape = tuple(shape[a] for a in transverse)
        return jnp.ones(area_shape) * spacings[transverse[0]] * spacings[transverse[1]]


#: Hard cap on the number of cells a single graded axis may generate, so a mistyped spacing raises
#: instead of allocating an unusable mesh.
MAX_CELLS_PER_AXIS = 10_000_000


def _as_axis_tuple(value, name: str) -> tuple[float, float, float]:
    """Return a per-axis triple of positive, finite floats from a scalar or a length-3 sequence."""
    if isinstance(value, (int, float)):
        values = (float(value), float(value), float(value))
    else:
        try:
            sequence = tuple(value)
        except TypeError:
            raise ValueError(f"{name} must be a number or a length-3 sequence, got {value!r}.") from None
        if len(sequence) != 3:
            raise ValueError(f"{name} must be a number or a length-3 sequence, got {value!r}.")
        values = (float(sequence[0]), float(sequence[1]), float(sequence[2]))
    for axis, v in enumerate(values):
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be positive and finite on every axis, got {value!r} (axis {axis}).")
    return values


def _format_length(value: float) -> str:
    """Format a length in the largest unit that keeps the number readable."""
    for scale, unit in ((1e-9, "nm"), (1e-6, "um"), (1e-3, "mm")):
        if abs(value) < 1000.0 * scale:
            return f"{value / scale:.4g} {unit}"
    return f"{value:.4g} m"


def _ceil_with_tolerance(value: float, rel_tol: float = 1e-9) -> int:
    """Ceiling that ignores float noise, so ``20.0000000001`` stays 20 cells."""
    return math.ceil(value - rel_tol * max(1.0, abs(value)))


def _axis_segments(
    lower: float,
    upper: float,
    intervals: list[tuple[float, float, float]],
    background: float,
) -> list[tuple[float, float, float]]:
    """Split ``[lower, upper]`` at every interval boundary and label each piece with its target width.

    Overlapping intervals are resolved by taking the finest target on the overlap.  Neighbouring
    pieces that end up with the same target are merged, so a boundary only survives where the
    requested cell width actually changes.
    """
    tol = 1e-12 * (upper - lower)
    points = [lower, upper]
    for lo, hi, _target in intervals:
        points.extend((lo, hi))
    ordered: list[float] = []
    for point in sorted(min(max(p, lower), upper) for p in points):
        if not ordered or point - ordered[-1] > tol:
            ordered.append(point)
    ordered[0] = lower
    ordered[-1] = upper
    segments: list[tuple[float, float, float]] = []
    for x0, x1 in pairwise(ordered):
        mid = 0.5 * (x0 + x1)
        target = background
        for lo, hi, interval_target in intervals:
            if lo - tol <= mid <= hi + tol:
                target = min(target, interval_target)
        if segments and abs(segments[-1][2] - target) <= 1e-12 * background:
            segments[-1] = (segments[-1][0], x1, segments[-1][2])
        else:
            segments.append((x0, x1, target))
    return segments


def _lower_envelope(
    lower: float,
    upper: float,
    lines: list[tuple[float, float]],
) -> list[tuple[float, float, float, float]]:
    """Return ``(x0, x1, alpha, beta)`` pieces on which ``alpha + beta * x`` is the smallest line."""
    breakpoints = {lower, upper}
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            alpha_i, beta_i = lines[i]
            alpha_j, beta_j = lines[j]
            if beta_i == beta_j:
                continue
            crossing = (alpha_j - alpha_i) / (beta_i - beta_j)
            if lower < crossing < upper:
                breakpoints.add(crossing)
    ordered = sorted(breakpoints)
    pieces: list[tuple[float, float, float, float]] = []
    for x0, x1 in pairwise(ordered):
        if x1 <= x0:
            continue
        mid = 0.5 * (x0 + x1)
        alpha, beta = min(lines, key=lambda line: line[0] + line[1] * mid)
        pieces.append((x0, x1, alpha, beta))
    return pieces


def _piece_cell_measure(piece: tuple[float, float, float, float]) -> float:
    """Return the cell count ``integral dx / s(x)`` carried by one linear piece of the width field."""
    x0, x1, alpha, beta = piece
    if beta == 0.0:
        return (x1 - x0) / alpha
    return math.log((alpha + beta * x1) / (alpha + beta * x0)) / beta


def _piece_invert(piece: tuple[float, float, float, float], measure: float) -> float:
    """Return the coordinate reached after ``measure`` cells from the start of one linear piece."""
    x0, _x1, alpha, beta = piece
    if beta == 0.0:
        return x0 + alpha * measure
    return ((alpha + beta * x0) * math.exp(beta * measure) - alpha) / beta


def _enforce_ratio_bound(
    segment_widths: list[np.ndarray],
    segment_lengths: list[float],
    segment_targets: list[float],
    max_ratio: float,
    max_iterations: int = 100,
) -> list[np.ndarray]:
    """Grow the cells that are too small for their neighbours, keeping every segment length exact.

    The width field is continuous across a segment boundary, but each segment holds a whole number
    of cells, so a segment that needs, say, 1.35 field cells is filled with two cells narrower than
    the field.  That shortfall can leave the cell at the boundary too small next to its neighbour in
    the adjacent segment.  The repair enlarges only the cells that are too small — never the ones
    already at their target — and then rescales each segment back to its exact length, which can
    only shrink cells.  Widths therefore never rise above a region's target or above the background,
    and the boundary mismatch falls by the rescale factor on every pass, so a few passes suffice.

    Both sweeps are the cumulative maximum of the log widths tilted by ``k * ln(max_ratio)``, which
    is the vectorized form of ``w[k] = max(w[k], w[k-1] / max_ratio)`` and its mirror image.  A cell
    is never widened past its own segment's target, so a layout that would need that is left with
    its ratio violation for the caller to report rather than silently under-resolving a region.
    """
    widths = [np.asarray(w, dtype=np.float64) for w in segment_widths]
    caps = np.concatenate([np.full(w.shape[0], target, dtype=np.float64) for w, target in zip(widths, segment_targets)])
    log_ratio = math.log(max_ratio)
    for _ in range(max_iterations):
        flat = np.concatenate(widths)
        if flat.size < 2:
            break
        ratios = np.maximum(flat[1:] / flat[:-1], flat[:-1] / flat[1:])
        if float(ratios.max()) <= max_ratio * (1.0 + 1e-12):
            break
        forward = np.arange(flat.size, dtype=np.float64)
        backward = forward[::-1]
        log_widths = np.log(flat)
        log_widths = np.maximum.accumulate(log_widths + forward * log_ratio) - forward * log_ratio
        log_widths = np.maximum.accumulate((log_widths + backward * log_ratio)[::-1])[::-1] - backward * log_ratio
        flat = np.minimum(np.exp(log_widths), caps)
        offset = 0
        for index, segment in enumerate(widths):
            size = segment.shape[0]
            updated = flat[offset : offset + size]
            offset += size
            widths[index] = updated * (segment_lengths[index] / float(updated.sum()))
    return widths


def _graded_axis_edges(
    lower: float,
    upper: float,
    background: float,
    intervals: list[tuple[float, float, float]],
    max_ratio: float,
    axis: int,
) -> np.ndarray:
    """Build the cell edges of one axis from a background width and a list of refinement intervals.

    The generator works on a continuous *cell width field* ``s(x)``:

    * inside a refinement interval, ``s`` is the interval's realized width — the target width
      reduced just enough that a whole number of cells fills the interval exactly;
    * outside, ``s`` grows away from every interval at the rate ``ln(max_ratio)`` per metre of
      distance and is capped at the background width.

    The field is the pointwise minimum of those contributions, so it is the widest field that
    honours every target and whose induced cell sequence grows by at most ``max_ratio`` between
    neighbours (a width field with Lipschitz constant ``ln(max_ratio)`` induces a neighbour ratio of
    at most ``max_ratio``; this is the continuous form of geometric grading).

    The axis is then cut at every breakpoint where the requested width changes, and each segment is
    filled with cells of equal *cell measure* ``integral dx / s``, so segment boundaries — and
    therefore interval boundaries — always land on a cell edge and the total extent is exact.  A
    segment holds a whole number of cells, so its measure has to be rounded, and the leftover length
    is absorbed by every cell of that segment through one common factor rather than by a single last
    cell.  The rounding rule is: take the nearest whole number of cells, and round up instead when
    that would make a cell wider than the segment's own target.  Rounding to nearest keeps the
    common factor near one on both sides of a boundary, which is what keeps the ratio bound intact
    across it; rounding up is the fallback that never lets a cell exceed its target.

    Where a rounding still leaves a cell too small next to the neighbouring segment,
    :func:`_enforce_ratio_bound` widens it again at the expense of the rest of its own segment, never
    past that segment's target.  A layout that cannot meet the ratio bound at all — a stretch
    shorter than one cell between a region and the domain edge, or between two regions — is reported
    by the caller rather than silently coarsened.
    """
    segments = _axis_segments(lower, upper, intervals, background)
    sources: list[tuple[float, float, float]] = []
    for seg_lower, seg_upper, target in segments:
        if target < background * (1.0 - 1e-12):
            cells = max(1, _ceil_with_tolerance((seg_upper - seg_lower) / target))
            sources.append((seg_lower, seg_upper, (seg_upper - seg_lower) / cells))
        else:
            sources.append((seg_lower, seg_upper, background))

    growth = math.log(max_ratio)
    segment_widths: list[np.ndarray] = []
    for index, (seg_lower, seg_upper, width) in enumerate(sources):
        lines: list[tuple[float, float]] = [(width, 0.0)]
        left_alpha: float | None = None
        right_alpha: float | None = None
        for other_index, (other_lower, other_upper, other_width) in enumerate(sources):
            if other_index == index:
                continue
            if other_upper <= seg_lower:
                alpha = other_width - growth * other_upper
                left_alpha = alpha if left_alpha is None else min(left_alpha, alpha)
            else:
                alpha = other_width + growth * other_lower
                right_alpha = alpha if right_alpha is None else min(right_alpha, alpha)
        if left_alpha is not None:
            lines.append((left_alpha, growth))
        if right_alpha is not None:
            lines.append((right_alpha, -growth))

        pieces = _lower_envelope(seg_lower, seg_upper, lines)
        measures = [_piece_cell_measure(piece) for piece in pieces]
        total_measure = sum(measures)
        target = segments[index][2]
        cells = max(1, math.floor(total_measure + 0.5))
        if cells > MAX_CELLS_PER_AXIS:
            raise ValueError(
                f"Graded grid axis {axis} would need {cells:,} cells between "
                f"{_format_length(seg_lower)} and {_format_length(seg_upper)}, which exceeds the "
                f"limit of {MAX_CELLS_PER_AXIS:,}. Coarsen the refinement spacing or shrink the region."
            )
        widths = _fill_segment(pieces, measures, total_measure, cells, seg_lower, seg_upper)
        if float(widths.max()) > target * (1.0 + 1e-12):
            rounded_up = max(cells, _ceil_with_tolerance(total_measure))
            if rounded_up != cells:
                widths = _fill_segment(pieces, measures, total_measure, rounded_up, seg_lower, seg_upper)
        segment_widths.append(widths)

    segment_widths = _enforce_ratio_bound(
        segment_widths,
        [source[1] - source[0] for source in sources],
        [segment[2] for segment in segments],
        max_ratio,
    )

    edges: list[float] = [lower]
    for (_seg_lower, seg_upper, _width), widths in zip(sources, segment_widths):
        running = edges[-1]
        for cell_width in widths[:-1]:
            running += float(cell_width)
            edges.append(running)
        edges.append(seg_upper)
    return np.asarray(edges, dtype=np.float64)


def _fill_segment(
    pieces: list[tuple[float, float, float, float]],
    measures: list[float],
    total_measure: float,
    cells: int,
    seg_lower: float,
    seg_upper: float,
) -> np.ndarray:
    """Return the widths of ``cells`` cells filling one segment with equal cell measure."""
    step = total_measure / cells
    piece_index = 0
    consumed = 0.0
    coordinates: list[float] = [seg_lower]
    for cell in range(1, cells):
        target_measure = cell * step
        while piece_index < len(pieces) - 1 and consumed + measures[piece_index] < target_measure:
            consumed += measures[piece_index]
            piece_index += 1
        coordinate = _piece_invert(pieces[piece_index], target_measure - consumed)
        coordinates.append(min(max(coordinate, seg_lower), seg_upper))
    coordinates.append(seg_upper)
    return np.diff(np.asarray(coordinates, dtype=np.float64))


@autoinit
class RefinementRegion(TreeClass):
    """A physical-coordinate box that asks for a cell width inside it.

    The box is given in metres in the same absolute frame as the rest of placement: the domain is
    centred on the grid policy's ``center``, so ``0`` is the centre of the simulation volume unless
    that centre was moved.  Each axis is either an ``(lower, upper)`` pair or ``None`` for "the whole
    axis".

    The refinement of a rectilinear mesh is a per-axis statement: a region refines the x axis over
    its own x interval regardless of its y and z extent, so a compact box produces refined *slabs*
    along all three axes, not only refined cells inside the box.  This is a property of rectilinear
    grids, not of this implementation.

    Example::

        # 12.5 nm cells in a 200 nm slab around z = 0, on every x and y
        RefinementRegion(spacing=12.5e-9, z=(-100e-9, 100e-9))
    """

    #: Target cell width in metres inside the region: a scalar, or one width per axis.
    spacing: float | tuple[float, float, float] = frozen_field()
    #: ``(lower, upper)`` x bounds in metres, or ``None`` for the whole x axis.
    x: tuple[float, float] | None = frozen_field(default=None)
    #: ``(lower, upper)`` y bounds in metres, or ``None`` for the whole y axis.
    y: tuple[float, float] | None = frozen_field(default=None)
    #: ``(lower, upper)`` z bounds in metres, or ``None`` for the whole z axis.
    z: tuple[float, float] | None = frozen_field(default=None)

    def __post_init__(self):
        _as_axis_tuple(self.spacing, "RefinementRegion.spacing")
        for name, bounds in (("x", self.x), ("y", self.y), ("z", self.z)):
            if bounds is None:
                continue
            try:
                sequence = tuple(bounds)
            except TypeError:
                raise ValueError(
                    f"RefinementRegion.{name} must be an (lower, upper) pair in metres or None, got {bounds!r}."
                ) from None
            if len(sequence) != 2:
                raise ValueError(
                    f"RefinementRegion.{name} must be an (lower, upper) pair in metres or None, got {bounds!r}."
                )
            lo, hi = float(sequence[0]), float(sequence[1])
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise ValueError(f"RefinementRegion.{name} bounds must be finite, got {bounds!r}.")
            if hi <= lo:
                raise ValueError(f"RefinementRegion.{name} bounds must be increasing (lower < upper), got {bounds!r}.")

    def axis_spacing(self, axis: int) -> float:
        """Target cell width in metres along one axis."""
        return _as_axis_tuple(self.spacing, "RefinementRegion.spacing")[axis]

    def axis_bounds(self, axis: int) -> tuple[float, float] | None:
        """Physical ``(lower, upper)`` bounds along one axis, or ``None`` for the whole axis."""
        bounds = (self.x, self.y, self.z)[axis]
        if bounds is None:
            return None
        return (float(bounds[0]), float(bounds[1]))


@autoinit
class GradedGrid(TreeClass):
    """Unresolved policy for a graded rectilinear grid with mesh override regions.

    Like ``UniformGrid`` and ``QuasiUniformGrid`` this is user intent, not the solver mesh.  It
    records a background cell width, a list of :class:`RefinementRegion` boxes in physical
    coordinates, and how fast the mesh may coarsen between them.  Placement turns it into a concrete
    ``RectilinearGrid`` before anything else runs.

    Unlike the two uniform policies, the *physical extent* is the primary quantity: a cell count does
    not determine the extent once the widths vary.  The policy is therefore resolved through
    :meth:`resolve_extent`, which ``fdtdx.place_objects`` calls with the simulation volume's
    ``partial_real_shape``; :meth:`resolve` raises to keep the ambiguity visible.

    The resolved grid honours four properties on every axis:

    1. every cell inside a region is at most that region's target width,
    2. neighbouring cell widths differ by at most ``max_ratio``,
    3. every boundary where the requested width changes lands exactly on a cell edge,
    4. the edges span the requested extent exactly.

    With no regions the policy reproduces ``UniformGrid`` (or a per-axis uniform grid when
    ``spacing`` is a triple) cell for cell.

    Example::

        grid = GradedGrid(
            spacing=50e-9,
            regions=(RefinementRegion(spacing=12.5e-9, z=(-100e-9, 100e-9)),),
            max_ratio=1.4,
        )
        resolved = grid.resolve_extent((1e-6, 1e-6, 4e-6))
    """

    #: Background cell width in metres away from every region: a scalar, or one width per axis.
    spacing: float | tuple[float, float, float] = frozen_field()
    #: Mesh override regions, finest target wins where two regions overlap.
    regions: tuple[RefinementRegion, ...] = frozen_field(default=())
    #: Largest allowed ratio between neighbouring cell widths. Must be greater than one.
    max_ratio: float = frozen_field(default=1.4)
    #: Physical coordinate of the domain center in metres. Defaults to (0, 0, 0).
    center: tuple[float, float, float] = frozen_field(default=(0.0, 0.0, 0.0))

    def __post_init__(self):
        _as_axis_tuple(self.spacing, "GradedGrid.spacing")
        if not math.isfinite(self.max_ratio) or self.max_ratio <= 1.0:
            raise ValueError(f"GradedGrid.max_ratio must be greater than one, got {self.max_ratio}.")
        for region in self.regions:
            if not isinstance(region, RefinementRegion):
                raise ValueError(f"GradedGrid.regions must contain RefinementRegion instances, got {region!r}.")

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve_extent(self, real_shape: tuple[float, float, float]) -> "RectilinearGrid":
        """Return a concrete ``RectilinearGrid`` spanning ``real_shape`` metres.

        Args:
            real_shape: Physical side lengths ``(Lx, Ly, Lz)`` in metres.

        Returns:
            A ``RectilinearGrid`` centered on :attr:`center` whose edges span exactly ``real_shape``.

        Raises:
            ValueError: If the requested regions cannot be graded within :attr:`max_ratio` — a region
                narrower than about two of its own cells, or two regions a single cell apart, leave
                no room for the transition.  Widen the gap, make the region an integer number of its
                own cells wide, or allow a larger ``max_ratio``.
        """
        lengths = _as_axis_tuple(real_shape, "GradedGrid extent")
        edge_arrays = []
        for axis in range(3):
            background = self.axis_spacing(axis)
            length = lengths[axis]
            lower = self.center[axis] - length / 2.0
            upper = lower + length
            intervals: list[tuple[float, float, float]] = []
            for region in self.regions:
                bounds = region.axis_bounds(axis)
                region_lower, region_upper = (lower, upper) if bounds is None else bounds
                region_lower = max(region_lower, lower)
                region_upper = min(region_upper, upper)
                if region_upper - region_lower <= 1e-12 * length:
                    continue
                intervals.append((region_lower, region_upper, min(region.axis_spacing(axis), background)))
            edges = _graded_axis_edges(lower, upper, background, intervals, self.max_ratio, axis)
            widths = np.diff(edges)
            if np.allclose(widths, widths[0], rtol=1e-12, atol=0.0):
                # Reproduce the uniform policy's edge expression bit for bit when the axis came out
                # uniform, so a region-free GradedGrid is indistinguishable from UniformGrid.
                edge_arrays.append(lower + float(widths[0]) * jnp.arange(widths.shape[0] + 1))
            else:
                edge_arrays.append(jnp.asarray(edges))
            self._check_axis(np.asarray(edge_arrays[-1], dtype=np.float32), axis)
        return RectilinearGrid(x_edges=edge_arrays[0], y_edges=edge_arrays[1], z_edges=edge_arrays[2])

    def resolve(self, shape: tuple[int, int, int]) -> "RectilinearGrid":
        """Reject cell-count resolution: a graded grid is defined by its physical extent.

        Raises:
            ValueError: Always. Call :meth:`resolve_extent` with the physical side lengths, or let
                ``fdtdx.place_objects`` do it from the simulation volume's ``partial_real_shape``.
        """
        raise ValueError(
            f"GradedGrid cannot be resolved from a cell count (got shape {shape}): the number of "
            "cells is an output of the grading, not an input. Call resolve_extent((Lx, Ly, Lz)) "
            "with the physical side lengths in metres, or give the SimulationVolume a "
            "partial_real_shape and let place_objects resolve the grid."
        )

    def _check_axis(self, edges: np.ndarray, axis: int) -> None:
        """Verify the ratio bound on the edges actually handed to the solver."""
        widths = np.diff(edges)
        if np.any(widths <= 0):
            index = int(np.argmin(widths))
            raise ValueError(
                f"Graded grid axis {axis} produced a non-positive cell width at "
                f"{_format_length(float(edges[index]))}. The refinement spacing is too small for the "
                "floating point resolution of the domain."
            )
        ratios = np.maximum(widths[1:] / widths[:-1], widths[:-1] / widths[1:])
        if ratios.size and float(ratios.max()) > self.max_ratio * (1.0 + 1e-3):
            index = int(np.argmax(ratios))
            raise ValueError(
                f"Graded grid axis {axis} could not honour max_ratio={self.max_ratio}: neighbouring "
                f"cells at {_format_length(float(edges[index + 1]))} differ by a factor of "
                f"{float(ratios.max()):.3f}. There is no room for the transition — widen the gap "
                "between the regions, make the region an integer number of its own cells wide, or "
                "raise max_ratio."
            )

    # ------------------------------------------------------------------
    # Convenience properties (mirror the other policies' interface)
    # ------------------------------------------------------------------

    def axis_spacing(self, axis: int) -> float:
        """Background cell width in metres along one axis."""
        return _as_axis_tuple(self.spacing, "GradedGrid.spacing")[axis]

    @property
    def is_uniform(self) -> bool:
        """True only without regions and with one background width on all three axes."""
        background = _as_axis_tuple(self.spacing, "GradedGrid.spacing")
        return len(self.regions) == 0 and background[0] == background[1] == background[2]

    @property
    def min_spacing(self) -> float:
        """Finest requested cell width in metres.

        This is the pre-resolution estimate used for a CFL bound before placement.  The realized grid
        can be slightly finer, because a region whose width is not a whole number of target cells is
        filled with slightly smaller cells; the solver always takes its time step from the resolved
        ``RectilinearGrid``, which sees the true finest cell.
        """
        background = _as_axis_tuple(self.spacing, "GradedGrid.spacing")
        finest = min(background)
        for region in self.regions:
            for axis in range(3):
                finest = min(finest, region.axis_spacing(axis))
        return finest

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary_stats(self, real_shape: tuple[float, float, float]) -> dict[str, Any]:
        """Return the per-axis numbers of the grid this policy would build for ``real_shape``."""
        grid = self.resolve_extent(real_shape)
        cells: list[int] = []
        min_widths: list[float] = []
        max_widths: list[float] = []
        ratios: list[float] = []
        for axis in range(3):
            widths = np.asarray(grid.cell_widths(axis), dtype=np.float64)
            cells.append(int(widths.shape[0]))
            min_widths.append(float(widths.min()))
            max_widths.append(float(widths.max()))
            if widths.shape[0] > 1:
                pairwise = np.maximum(widths[1:] / widths[:-1], widths[:-1] / widths[1:])
                ratios.append(float(pairwise.max()))
            else:
                ratios.append(1.0)
        return {
            "cells": tuple(cells),
            "total_cells": int(cells[0] * cells[1] * cells[2]),
            "min_width": tuple(min_widths),
            "max_width": tuple(max_widths),
            "max_ratio_observed": tuple(ratios),
            "max_ratio": float(self.max_ratio),
            "ratio_ok": all(ratio <= self.max_ratio * (1.0 + 1e-3) for ratio in ratios),
            "num_regions": len(self.regions),
        }

    def summary(self, real_shape: tuple[float, float, float]) -> str:
        """Return a short human-readable description of the grid for ``real_shape``.

        Args:
            real_shape: Physical side lengths ``(Lx, Ly, Lz)`` in metres.

        Returns:
            One header line plus one line per axis, giving the cell count, the smallest and largest
            cell width, and the largest neighbouring-width ratio that was realized.
        """
        stats = self.summary_stats(real_shape)
        background = _as_axis_tuple(self.spacing, "GradedGrid.spacing")
        background_text = (
            _format_length(background[0])
            if background[0] == background[1] == background[2]
            else ", ".join(_format_length(value) for value in background)
        )
        lines = [
            f"GradedGrid: background {background_text}, {stats['num_regions']} refinement region(s), "
            f"max ratio {self.max_ratio:g}, {stats['total_cells']:,} cells total"
        ]
        for axis, name in enumerate("xyz"):
            lines.append(
                f"  {name}: {stats['cells'][axis]:,} cells, width "
                f"{_format_length(stats['min_width'][axis])} .. {_format_length(stats['max_width'][axis])}, "
                f"largest neighbour ratio {stats['max_ratio_observed'][axis]:.3f}"
            )
        lines.append("  ratio bound honoured" if stats["ratio_ok"] else "  RATIO BOUND VIOLATED")
        return "\n".join(lines)


@autoinit
class RectilinearGrid(TreeClass):
    """Realized rectilinear simulation grid described by physical cell edges.

    This is the canonical solver-facing grid representation used by fdtdx
    internals.  A uniform grid is represented by equally spaced edge arrays, not
    by a separate scalar code path.  Keeping one realized representation is
    important for the non-uniform grid migration: placement, PML profiles,
    mode-solver coordinates, detector weights, and Yee update metrics should all
    ask the grid for physical distances instead of deriving them from a global
    ``resolution`` value.

    The arrays store cell *edges* in metres.  For a grid with ``nx`` cells along
    x, ``x_edges`` has shape ``(nx + 1,)`` and must be strictly increasing.  Cell
    widths, centers, face areas, and volumes are derived from these arrays.

    Notes:
        This class intentionally does not encode automatic mesh generation
        policy.  Future policy objects such as ``AutoGrid`` or
        ``QuasiUniformGrid`` should resolve to ``RectilinearGrid`` before the
        solver runs.
    """

    #: Physical edge coordinates along x in metres, shape ``(nx + 1,)``.
    x_edges: jax.Array = field()
    #: Physical edge coordinates along y in metres, shape ``(ny + 1,)``.
    y_edges: jax.Array = field()
    #: Physical edge coordinates along z in metres, shape ``(nz + 1,)``.
    z_edges: jax.Array = field()
    _min_spacings: tuple[float, float, float] = frozen_private_field()
    _is_uniform: bool = frozen_private_field()
    _uniform_spacing: float | None = frozen_private_field()
    _cell_widths: tuple[jax.Array, jax.Array, jax.Array] = private_field(repr=False)

    def __post_init__(self):
        object.__setattr__(self, "x_edges", jnp.asarray(self.x_edges))
        object.__setattr__(self, "y_edges", jnp.asarray(self.y_edges))
        object.__setattr__(self, "z_edges", jnp.asarray(self.z_edges))
        for axis, edges in enumerate((self.x_edges, self.y_edges, self.z_edges)):
            if edges.ndim != 1:
                raise ValueError(f"Grid edge coordinates for axis {axis} must be one-dimensional.")
            if edges.shape[0] < 2:
                raise ValueError(f"Grid edge coordinates for axis {axis} must contain at least two entries.")
            if bool(jnp.any(jnp.diff(edges) <= 0)):
                raise ValueError(f"Grid edge coordinates for axis {axis} must be strictly increasing.")
        edge_arrays_np = tuple(np.asarray(edges) for edges in (self.x_edges, self.y_edges, self.z_edges))
        width_arrays = tuple(np.diff(edges) for edges in edge_arrays_np)
        object.__setattr__(self, "_cell_widths", tuple(jnp.asarray(widths) for widths in width_arrays))
        min_spacings = tuple(float(np.min(widths)) for widths in width_arrays)
        spacing = float(width_arrays[0][0])
        # Uniformity is a *relative* property. The absolute width jitter of a perfectly uniform
        # float grid grows with the edge-coordinate magnitude (~eps * max|edge|), so a fixed
        # absolute tolerance (np.allclose's default atol=1e-8) is wrong at physical scales: it
        # labels nanometre-spaced grids "uniform" even with tens-of-percent variation, and could
        # label very large coarse grids "non-uniform". Compare widths to the grid spacing with a
        # small relative tolerance plus a roundoff floor that tracks the float dtype and domain
        # size, so genuine (>~0.01%) non-uniformity is detected at any scale while a truly uniform
        # grid stays uniform regardless of cell count.
        is_uniform = True
        for edges_np, widths in zip(edge_arrays_np, width_arrays):
            eps = float(np.finfo(edges_np.dtype).eps) if np.issubdtype(edges_np.dtype, np.floating) else 0.0
            roundoff = 8.0 * eps * float(np.max(np.abs(edges_np)))
            if float(np.max(np.abs(widths - spacing))) > 1e-4 * abs(spacing) + roundoff:
                is_uniform = False
                break
        object.__setattr__(self, "_min_spacings", min_spacings)
        object.__setattr__(self, "_is_uniform", is_uniform)
        object.__setattr__(self, "_uniform_spacing", float(np.round(spacing, decimals=14)) if is_uniform else None)

    @classmethod
    def uniform(
        cls,
        shape: tuple[int, int, int],
        spacing: float,
        origin: tuple[float, float, float] | None = None,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """Create a realized rectilinear grid for a uniform grid.

        The grid is centered at ``center`` by default, so edge arrays span
        ``[center[a] - shape[a] * spacing / 2, center[a] + shape[a] * spacing / 2]``
        along each axis.  Negative and positive coordinates are used symmetrically
        around the center of the simulation domain.

        Passing an explicit ``origin`` (lower-corner coordinate) overrides
        ``center`` and restores the legacy lower-corner behaviour.  This is used
        internally by ``UniformGrid.resolve`` after it has already computed the
        lower corner from the center.

        Args:
            shape: Number of cells in ``(x, y, z)``.
            spacing: Uniform cell width in metres.
            center: Physical coordinate of the domain center. Defaults to
                ``(0, 0, 0)`` so the domain spans equally into negative and
                positive coordinates.
            origin: Physical coordinate of the **lower** domain corner.  When
                provided this takes priority over ``center``.

        Returns:
            A grid whose edge arrays are equally spaced and centered on
            ``center`` (or anchored at ``origin`` when given).
        """
        if spacing <= 0:
            raise ValueError(f"Uniform grid spacing must be positive, got {spacing}.")
        if any(n <= 0 for n in shape):
            raise ValueError(f"Uniform grid shape entries must be positive, got {shape}.")
        if origin is not None:
            lower_corner = origin
        else:
            lower_corner = tuple(center[a] - shape[a] * spacing / 2.0 for a in range(3))
        edge_arrays = tuple(lower_corner[axis] + spacing * jnp.arange(shape[axis] + 1) for axis in range(3))
        return cls(x_edges=edge_arrays[0], y_edges=edge_arrays[1], z_edges=edge_arrays[2])

    @classmethod
    def custom(
        cls,
        x_edges: jax.Array,
        y_edges: jax.Array,
        z_edges: jax.Array,
    ):
        """Create a realized rectilinear grid from explicit edge arrays.

        This constructor is equivalent to calling ``RectilinearGrid(...)``
        directly, but it makes the user-facing intent explicit: the caller is
        supplying the final grid coordinates, not an automatic meshing policy.
        """
        return cls(x_edges=x_edges, y_edges=y_edges, z_edges=z_edges)

    def reduce_symmetric(self, symmetry: tuple[int, int, int]) -> "RectilinearGrid":
        """Return the grid reduced onto the kept (upper) half along each symmetric axis.

        Used by ``place_objects`` when ``config.symmetry`` is set on a non-uniform grid: the
        simulation runs on the reduced (half/quarter/octant) domain and the result is unfolded
        afterwards. For every axis with ``symmetry[a] != 0`` this keeps the upper-half edges
        ``edges(a)[n // 2:]`` (absolute coordinates preserved — the FDTD metrics depend only on
        cell widths, which are translation-invariant). Non-symmetric axes are returned unchanged.

        Two conditions must hold on each symmetric axis so that mirroring the kept half exactly
        reconstructs the full domain:

        * an even cell count, so the split lands on a cell edge, and
        * mirror-symmetric cell widths about the center (``dx[i] == dx[n - 1 - i]``), so the
          discarded lower half is the exact mirror of the kept half.

        Args:
            symmetry (tuple[int, int, int]): Per-axis symmetry condition ``(x, y, z)``; ``0`` means
                no reduction on that axis (any nonzero value reduces it).

        Returns:
            RectilinearGrid: The reduced grid (a new instance; the original is unchanged).

        Raises:
            ValueError: If a symmetric axis has an odd (or < 2) cell count, or cell widths that are
                not mirror-symmetric about the center.
        """
        axis_names = ("x", "y", "z")
        new_edges = []
        for a in range(3):
            edges = self.edges(a)
            if symmetry[a] == 0:
                new_edges.append(edges)
                continue
            n = self.shape[a]
            validate_symmetric_axis_cells(n, axis_names[a], subject="grid")
            widths = self.cell_widths(a)
            # rtol is loose enough to tolerate the float32 cumsum/diff roundoff of a grid that is
            # mathematically symmetric, but far tighter than any genuinely asymmetric profile.
            if not bool(jnp.allclose(widths, widths[::-1], rtol=1e-4, atol=0.0)):
                raise ValueError(
                    f"Cannot apply symmetry on axis {axis_names[a]}: the cell widths must be "
                    f"mirror-symmetric about the center (dx[i] == dx[n-1-i]) so the discarded half is "
                    f"the exact mirror of the kept half. Provide a grid whose spacing is symmetric "
                    f"about the {axis_names[a]} center plane, or drop symmetry on this axis."
                )
            new_edges.append(edges[n // 2 :])
        return RectilinearGrid.custom(x_edges=new_edges[0], y_edges=new_edges[1], z_edges=new_edges[2])

    @property
    def shape(self) -> tuple[int, int, int]:
        """Number of cells along each axis."""
        return (self.x_edges.shape[0] - 1, self.y_edges.shape[0] - 1, self.z_edges.shape[0] - 1)

    @property
    def dx(self) -> jax.Array:
        """Cell widths along x in metres."""
        return self._cell_widths[0]

    @property
    def dy(self) -> jax.Array:
        """Cell widths along y in metres."""
        return self._cell_widths[1]

    @property
    def dz(self) -> jax.Array:
        """Cell widths along z in metres."""
        return self._cell_widths[2]

    @property
    def min_spacing(self) -> float:
        """Smallest cell width in the grid.

        This value is the conservative spacing used for staged CFL migration.
        The full non-uniform update should eventually use explicit local metric
        arrays, but stability remains controlled by the smallest cell.
        """
        return min(self._min_spacings)

    @property
    def min_spacings(self) -> tuple[float, float, float]:
        """Smallest cell width along each axis in metres."""
        return self._min_spacings

    def cfl_time_step(self, courant_factor: float) -> float:
        """Return the CFL-limited time step for a rectilinear 3D grid.

        The stability limit for an orthogonal FDTD grid is controlled by the
        smallest spacing on each axis:

        ``dt <= courant_factor / (c * sqrt(1/dx_min^2 + 1/dy_min^2 + 1/dz_min^2))``.

        For uniform grids this is exactly the existing ``courant_factor/sqrt(3)``
        behavior.  For anisotropic or stretched grids it avoids using one global
        spacing for all three axes.
        """
        if self._is_uniform and self._uniform_spacing is not None:
            return (courant_factor / float(np.sqrt(3.0))) * self._uniform_spacing / constants.c
        dx_min, dy_min, dz_min = self.min_spacings
        inv_metric = (1 / dx_min**2) + (1 / dy_min**2) + (1 / dz_min**2)
        return courant_factor / (constants.c * float(np.sqrt(inv_metric)))

    @property
    def is_uniform(self) -> bool:
        """Whether all cell widths match a single spacing within numerical tolerance."""
        return self._is_uniform

    @property
    def uniform_spacing(self) -> float:
        """Return the scalar spacing for a uniform grid or raise for non-uniform grids.

        This compatibility escape hatch should only be used by code that has not
        yet been migrated to metric-aware helpers.  It deliberately raises for
        non-uniform grids so unsupported paths fail loudly.
        """
        if self._uniform_spacing is None:
            raise ValueError("This operation still requires a uniform grid.")
        return self._uniform_spacing

    def edges(self, axis: int) -> jax.Array:
        """Return edge coordinates for ``axis``."""
        return (self.x_edges, self.y_edges, self.z_edges)[axis]

    def cell_widths(self, axis: int) -> jax.Array:
        """Return cell widths for ``axis``."""
        return (self.dx, self.dy, self.dz)[axis]

    def centers(self, axis: int) -> jax.Array:
        """Return cell-center coordinates for ``axis``."""
        edges = self.edges(axis)
        return 0.5 * (edges[:-1] + edges[1:])

    def axis_extent(self, axis: int, bounds: tuple[int, int]) -> float:
        """Physical length covered by an index interval on one axis."""
        lower, upper = bounds
        edges = self.edges(axis)
        return float(edges[upper] - edges[lower])

    def slice_extent(
        self, slice_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ) -> tuple[float, float, float]:
        """Physical side lengths covered by a 3D grid slice."""
        return (
            self.axis_extent(0, slice_tuple[0]),
            self.axis_extent(1, slice_tuple[1]),
            self.axis_extent(2, slice_tuple[2]),
        )

    def subgrid(
        self,
        grid_slice: tuple[slice, slice, slice],
    ):
        """Convenience wrapper to get the sub-grid of a placed fdtdx.SimulationObject given its grid_slice"""
        subgrid = self.custom(
            self.x_edges[slice(grid_slice[0].start, grid_slice[0].stop + 1)],
            self.y_edges[slice(grid_slice[1].start, grid_slice[1].stop + 1)],
            self.z_edges[slice(grid_slice[2].start, grid_slice[2].stop + 1)],
        )
        return subgrid

    def coord_to_index(self, axis: int, coord: float, snap: str = "nearest") -> int:
        """Map a physical coordinate to a grid edge index.

        Args:
            axis: Grid axis.
            coord: Coordinate in metres.
            snap: Snapping rule. ``"nearest"`` chooses the closest edge,
                ``"lower"`` chooses the previous edge, and ``"upper"`` chooses
                the next edge.

        Returns:
            Edge index after applying the requested snapping rule.
        """
        edges = np.asarray(self.edges(axis))
        if snap == "nearest":
            return int(np.argmin(np.abs(edges - coord)))
        if snap == "lower":
            return int(np.searchsorted(edges, coord, side="right") - 1)
        if snap == "upper":
            return int(np.searchsorted(edges, coord, side="left"))
        raise ValueError(f"Unknown snapping rule: {snap}")

    def length_to_cell_count(self, axis: int, length: float, snap: str = "nearest") -> int:
        """Convert a physical length to a number of cells from the lower domain edge.

        This helper preserves the old uniform-grid behavior when ``snap`` is
        ``"nearest"``.  For non-uniform placement, ``"upper"`` is usually the
        safer rule because it chooses enough cells to cover the requested metric
        size from the lower domain edge.
        """
        if length < 0:
            raise ValueError(f"Length must be non-negative, got {length}.")
        return self.coord_to_index(axis, float(self.edges(axis)[0]) + length, snap=snap)

    def bounds_for_center(self, axis: int, center: float, size: int) -> tuple[int, int]:
        """Choose a cell interval whose physical center is closest to ``center``.

        Args:
            axis: Grid axis.
            center: Desired physical center coordinate in metres.
            size: Number of cells in the interval.

        Returns:
            ``(lower, upper)`` edge indices with ``upper - lower == size``.

        Notes:
            This operation is used by object placement when a physical center
            position and an already-resolved grid-cell size are known.  On a
            non-uniform grid there is no exact analogue of ``round(x / dx)``;
            selecting the closest physical interval center gives deterministic
            snapping while preserving the requested grid-cell size.
        """
        if size <= 0:
            raise ValueError(f"Interval size must be positive, got {size}.")
        edges = np.asarray(self.edges(axis))
        max_lower = edges.shape[0] - size - 1
        if max_lower < 0:
            raise ValueError(f"Interval of size {size} does not fit on axis {axis} with shape {self.shape[axis]}.")
        lower_candidates = np.arange(max_lower + 1)
        interval_centers = 0.5 * (edges[lower_candidates] + edges[lower_candidates + size])
        lower = int(lower_candidates[np.argmin(np.abs(interval_centers - center))])
        return lower, lower + size

    def anchor_coordinate(self, axis: int, bounds: tuple[int, int], position: float) -> float:
        """Return a physical anchor coordinate inside an interval.

        ``position`` follows fdtdx object-anchor convention: ``-1`` is the lower
        side, ``0`` is the center, and ``+1`` is the upper side.
        """
        lower, upper = bounds
        edges = np.asarray(self.edges(axis))
        lower_coord = edges[lower]
        upper_coord = edges[upper]
        return float(lower_coord + 0.5 * (position + 1.0) * (upper_coord - lower_coord))

    def bounds_for_anchor(self, axis: int, size: int, anchor: float, position: float) -> tuple[int, int]:
        """Choose a cell interval whose object anchor is closest to ``anchor``.

        Args:
            axis: Grid axis.
            size: Number of cells in the interval.
            anchor: Desired physical anchor coordinate in metres.
            position: Object-relative anchor position, where ``-1`` is lower
                side, ``0`` is center, and ``+1`` is upper side.

        Returns:
            ``(lower, upper)`` edge indices with ``upper - lower == size``.
        """
        if size <= 0:
            raise ValueError(f"Interval size must be positive, got {size}.")
        edges = np.asarray(self.edges(axis))
        max_lower = edges.shape[0] - size - 1
        if max_lower < 0:
            raise ValueError(f"Interval of size {size} does not fit on axis {axis} with shape {self.shape[axis]}.")
        lower_candidates = np.arange(max_lower + 1)
        lower_edges = edges[lower_candidates]
        upper_edges = edges[lower_candidates + size]
        anchors = lower_edges + 0.5 * (position + 1.0) * (upper_edges - lower_edges)
        lower = int(lower_candidates[np.argmin(np.abs(anchors - anchor))])
        return lower, lower + size

    def face_area(self, axis: int, slice_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> jax.Array:
        """Return per-cell face-area weights for a detector plane.

        Args:
            axis: Normal axis of the face.
            slice_tuple: Grid slice containing the detector volume.  The normal
                axis is expected to have width one for a plane detector.

        Returns:
            Area weights broadcast to the detector's 3D slice shape.
        """
        transverse_axes = get_transverse_axes(axis)
        widths = []
        for transverse_axis in transverse_axes:
            lower, upper = slice_tuple[transverse_axis]
            widths.append(self.cell_widths(transverse_axis)[lower:upper])
        area_2d = widths[0][:, None] * widths[1][None, :]
        shape = [1, 1, 1]
        shape[transverse_axes[0]] = area_2d.shape[0]
        shape[transverse_axes[1]] = area_2d.shape[1]
        return area_2d.reshape(shape)

    def cell_volume(self, slice_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> jax.Array:
        """Return per-cell volume weights broadcast to a 3D slice shape."""
        x0, x1 = slice_tuple[0]
        y0, y1 = slice_tuple[1]
        z0, z1 = slice_tuple[2]
        return self.dx[x0:x1, None, None] * self.dy[None, y0:y1, None] * self.dz[None, None, z0:z1]


def calculate_spatial_offsets_yee() -> tuple[jax.Array, jax.Array]:
    offset_E = jnp.stack(
        [
            jnp.asarray([0.5, 0, 0])[None, None, None, :],
            jnp.asarray([0, 0.5, 0])[None, None, None, :],
            jnp.asarray([0, 0, 0.5])[None, None, None, :],
        ]
    )
    offset_H = jnp.stack(
        [
            jnp.asarray([0, 0.5, 0.5])[None, None, None, :],
            jnp.asarray([0.5, 0, 0.5])[None, None, None, :],
            jnp.asarray([0.5, 0.5, 0])[None, None, None, :],
        ]
    )
    return offset_E, offset_H


def _raise_if_anisotropic_property(prop: jax.Array | float, name: str) -> None:
    """Reject a material array that is not isotropic, as the legacy source path requires.

    A 3-component array must have all three diagonal entries equal; a 9-component array must be a
    scalar times the identity. Anything else has no single phase velocity to broadcast.
    """
    if not isinstance(prop, jax.Array) or prop.ndim != 4:
        return
    if prop.shape[0] == 3:
        is_isotropic = jnp.allclose(prop[0], prop[1]) & jnp.allclose(prop[1], prop[2])
    elif prop.shape[0] == 9:
        is_isotropic = jnp.allclose(prop[0], prop[4]) & jnp.allclose(prop[4], prop[8])
        for idx in (1, 2, 3, 5, 6, 7):
            is_isotropic = is_isotropic & jnp.allclose(prop[idx], 0.0)
    else:
        return

    def _raise_if_anisotropic(is_iso):
        if not is_iso:
            raise NotImplementedError(
                "Gaussian or planewave sources within anisotropic materials are not supported yet."
            )

    jax.debug.callback(_raise_if_anisotropic, is_isotropic)


def _diagonal_components(
    prop: jax.Array | float,
    name: str,
) -> list[jax.Array | float]:
    """Return the three diagonal entries of a material property array, one per field component.

    Accepts the scalar, legacy ``(Nx, Ny, Nz)``, and ``(1|3|9, Nx, Ny, Nz)`` forms. A 9-component
    tensor whose off-diagonal entries are not negligible has no per-component scalar phase velocity,
    so it is still rejected.

    Args:
        prop (jax.Array | float): The inverse permittivity or permeability.
        name (str): Property name, for the error message.

    Returns:
        list: Three arrays (or three copies of the scalar), one per component.
    """
    if not isinstance(prop, jax.Array) or prop.ndim == 0:
        return [prop, prop, prop]
    if prop.ndim == 3:
        return [prop, prop, prop]
    if prop.shape[0] == 1:
        return [prop[0], prop[0], prop[0]]
    if prop.shape[0] == 3:
        return [prop[0], prop[1], prop[2]]
    if prop.shape[0] == 9:
        off_diagonal_is_zero = jnp.asarray(True)
        for idx in (1, 2, 3, 5, 6, 7):
            off_diagonal_is_zero = off_diagonal_is_zero & jnp.allclose(prop[idx], 0.0)

        def _raise_if_off_diagonal(is_zero):
            if not is_zero:
                raise NotImplementedError(
                    f"Gaussian or planewave sources within materials with off-diagonal {name} are not supported yet."
                )

        jax.debug.callback(_raise_if_off_diagonal, off_diagonal_is_zero)
        return [prop[0], prop[4], prop[8]]
    raise Exception(f"Invalid {name} shape: {prop.shape}")


def calculate_time_offset_yee(
    center: jax.Array,
    wave_vector: jax.Array,
    inv_permittivities: jax.Array,
    inv_permeabilities: jax.Array | float,
    resolution: float,
    time_step_duration: float,
    effective_index: jax.Array | float | None = None,
    e_polarization: jax.Array | None = None,
    h_polarization: jax.Array | None = None,
    coordinate_edges: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    center_physical: jax.Array | None = None,
    allow_anisotropic: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Per-component launch-time phase-front delay of a source plane.

    ``allow_anisotropic`` selects how a per-component material is handled. ``False`` (default) keeps
    the legacy behaviour: a diagonal or full-tensor array that is not isotropic raises, because the
    delay was computed from one component and broadcast. ``True`` computes the delay per component
    from that component's own material, which is what per-Yee-point material sampling
    (``SimulationConfig.material_sampling='yee'``) produces — every waveguide a source plane cuts
    then has ``eps_xx != eps_yy != eps_zz`` in its interface cells. For an isotropic input both
    settings give the same numbers.
    """
    if inv_permittivities.ndim == 4:
        # Extract spatial shape from (1, Nx, Ny, Nz) or (3, Nx, Ny, Nz) or (9, Nx, Ny, Nz)
        spatial_shape = inv_permittivities.shape[1:]
    elif inv_permittivities.ndim == 3:
        # Legacy shape (Nx, Ny, Nz)
        spatial_shape = inv_permittivities.shape
    else:
        raise Exception(f"Invalid permittivity shape: {inv_permittivities.shape=}")

    if 1 not in spatial_shape:
        raise Exception(f"Expected one spatial axis to be one, but got {spatial_shape}")

    propagation_axis = spatial_shape.index(1)

    if coordinate_edges is None:
        # Build uniform coordinate edges from scalar resolution so the rest of
        # the function has one physical-space code path.
        e = [jnp.arange(spatial_shape[ax] + 1, dtype=jnp.float32) * resolution for ax in range(3)]
        resolved_edges: tuple[jax.Array, jax.Array, jax.Array] = (e[0], e[1], e[2])
        c0 = jnp.asarray(center[0], dtype=wave_vector.dtype) * resolution
        c1 = jnp.asarray(center[1], dtype=wave_vector.dtype) * resolution
        center_parts: list[jax.Array] = [c0, c1]
        center_parts.insert(propagation_axis, jnp.zeros((), dtype=wave_vector.dtype))
        center_physical = jnp.stack(center_parts)
    else:
        if center_physical is None:
            raise ValueError("center_physical must be provided with coordinate_edges")
        resolved_edges = coordinate_edges

    def component_positions(axis: int, offset: float) -> jax.Array:
        edges = resolved_edges[axis]
        centers = 0.5 * (edges[:-1] + edges[1:])
        if offset == 0:
            return edges[:-1]
        if offset == 0.5:
            return centers
        raise ValueError(f"Unsupported Yee offset: {offset}")

    def xyz_for_offsets(offsets: tuple[float, float, float]) -> jax.Array:
        coords = [component_positions(ax, offsets[ax]) for ax in range(3)]
        x, y, z = jnp.meshgrid(coords[0], coords[1], coords[2], indexing="ij")
        return jnp.stack([x, y, z], axis=-1) - center_physical[None, None, None, :]

    xyz_E = jnp.stack(
        [
            xyz_for_offsets((0.5, 0, 0)),
            xyz_for_offsets((0, 0.5, 0)),
            xyz_for_offsets((0, 0, 0.5)),
        ]
    )
    xyz_H = jnp.stack(
        [
            xyz_for_offsets((0, 0.5, 0.5)),
            xyz_for_offsets((0.5, 0, 0.5)),
            xyz_for_offsets((0.5, 0.5, 0)),
        ]
    )
    distance_scale = 1.0

    travel_offset_E = -jnp.dot(xyz_E, wave_vector)
    travel_offset_H = -jnp.dot(xyz_H, wave_vector)

    if effective_index is not None:
        refractive_idx_E = jnp.broadcast_to(effective_index * jnp.ones(spatial_shape), (3, *spatial_shape))
        refractive_idx_H = refractive_idx_E
    elif not allow_anisotropic:
        _raise_if_anisotropic_property(inv_permittivities, "permittivity")
        _raise_if_anisotropic_property(inv_permeabilities, "permeability")
        inv_perm_eff = inv_permittivities[0] if inv_permittivities.ndim == 4 else inv_permittivities
        if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim == 4:
            inv_perm_eff_perm = inv_permeabilities[0]
        else:
            inv_perm_eff_perm = inv_permeabilities
        refractive_idx = 1 / jnp.sqrt(inv_perm_eff * inv_perm_eff_perm)
        refractive_idx_E = jnp.broadcast_to(refractive_idx, (3, *spatial_shape))
        refractive_idx_H = refractive_idx_E
    else:
        # Per-component phase-front delay. travel_offset_E/H already carry one entry per field
        # component at that component's own Yee position, so the material must be read per component
        # too: E_c sees inv_permittivities[c] and H_c sees inv_permeabilities[c]. For an isotropic
        # input all three components are equal and this reproduces the previous scalar result
        # exactly; for a per-component (yee-sampled) input it is the physically correct delay
        # instead of an exception.
        eps_diag = _diagonal_components(inv_permittivities, "permittivity")
        mu_diag = _diagonal_components(inv_permeabilities, "permeability")
        # The permittivity is not available at the H positions, so H uses the cell's arithmetic mean
        # of the three diagonal inverse permittivities. This only sets the launch-time phase front of
        # a tilted plane wave, and it is exact wherever the three components agree, which is
        # everywhere except the interface cells. The isotropic branch is kept bit-identical.
        eps_is_isotropic = jnp.allclose(eps_diag[0], eps_diag[1]) & jnp.allclose(eps_diag[1], eps_diag[2])
        eps_for_H = jnp.where(eps_is_isotropic, eps_diag[0], (eps_diag[0] + eps_diag[1] + eps_diag[2]) / 3.0)
        refractive_idx_E = jnp.stack([1 / jnp.sqrt(eps_diag[c] * mu_diag[c]) for c in range(3)])
        refractive_idx_H = jnp.stack([1 / jnp.sqrt(eps_for_H * mu_diag[c]) for c in range(3)])

    velocity_E = constants.c / refractive_idx_E
    velocity_H = constants.c / refractive_idx_H
    time_offset_E = travel_offset_E * distance_scale / (velocity_E * time_step_duration)
    time_offset_H = travel_offset_H * distance_scale / (velocity_H * time_step_duration)
    return time_offset_E, time_offset_H


def polygon_to_mask(
    boundary: tuple[float, float, float, float],
    resolution: float,
    polygon_vertices: np.ndarray,
) -> np.ndarray:
    """
    Generate a 2D binary mask from a polygon.

    Args:
        boundary (tuple[float, float, float, float]): tuple of (min_x, min_y, max_x, max_y)
            Rectangular boundary in metrical units (meter).
        resolution (float): float
            Grid resolution (spacing between grid points) in metrical units
        polygon_vertices (np.ndarray): list of (x, y) tuples
            Vertices of the polygon in metrical units. Last point should equal first point.
            Must have shape (N, 2).
    Returns:
        np.ndarray: 2D binary mask where 1 indicates inside polygon, 0 indicates outside
    """
    assert polygon_vertices.ndim == 2
    assert polygon_vertices.shape[1] == 2
    min_x, min_y, max_x, max_y = boundary

    x_coords = np.arange(min_x, max_x + 0.5 * resolution, resolution)
    y_coords = np.arange(min_y, max_y + 0.5 * resolution, resolution)
    return polygon_to_mask_at_points(x_coords, y_coords, polygon_vertices)


def polygon_to_mask_at_points(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    polygon_vertices: np.ndarray,
) -> np.ndarray:
    """Generate a 2D polygon mask at explicit sample coordinates.

    This is the rectilinear-grid counterpart to ``polygon_to_mask``.  The caller
    supplies the physical cell-center coordinates directly, so non-uniform grids
    do not need to be resampled onto a synthetic uniform lattice.
    """
    assert polygon_vertices.ndim == 2
    assert polygon_vertices.shape[1] == 2
    x_coords = np.asarray(x_coords)
    y_coords = np.asarray(y_coords)
    X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
    points = np.column_stack((X.ravel(), Y.ravel()))
    polygon_path = Path(polygon_vertices)
    inside_polygon = polygon_path.contains_points(points)
    return inside_polygon.reshape(X.shape).astype(bool)


def multi_polygons_to_mask(
    boundary: tuple[float, float, float, float],
    resolution: float,
    polygon_list: list[np.ndarray],
) -> np.ndarray:
    """
    Generate a 2D binary mask from a list of polygons.

    Args:
        boundary (tuple[float, float, float, float]): tuple of (min_x, min_y, max_x, max_y)
            Rectangular boundary in metrical units (meter).
        resolution (float): float
            Grid resolution (spacing between grid points) in metrical units.
        polygon_list (list[np.ndarray]): list of polygon vertex arrays.
            Each array must have shape (N, 2) with (x, y) vertices in metrical units.
    Returns:
        np.ndarray: 2D boolean mask where True indicates inside at least one polygon.
    """
    if len(polygon_list) == 0:
        min_x, min_y, max_x, max_y = boundary
        x_coords = np.arange(min_x, max_x + 0.5 * resolution, resolution)
        y_coords = np.arange(min_y, max_y + 0.5 * resolution, resolution)
        return np.zeros((len(x_coords), len(y_coords)), dtype=bool)
    result = polygon_to_mask(boundary, resolution, polygon_list[0])
    for poly in polygon_list[1:]:
        result |= polygon_to_mask(boundary, resolution, poly)
    return result
