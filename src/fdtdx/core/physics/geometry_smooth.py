"""Kottke/Farjadpour sub-pixel smoothing on the Yee pixels (``material_sampling="yee_smooth"``).

Stage A (:mod:`fdtdx.core.physics.geometry_raster`) puts the exact continuous geometry on the grid,
but still samples it at a single point per field component, so an interface is staircased and the
error is first order in the cell size. This module removes that first-order term.

**The pixel.** Every component owns a box centred on its own sample point: the primal cell on the
axes where the component sits at a cell centre, and the dual cell on the axes where it sits at an
edge. For E that is not a free choice — the dual width ``0.5*(w[i-1] + w[i])`` is the metric the
backward difference already divides by when it produces a quantity at ``e_a[i]``
(:mod:`fdtdx.core.physics.curl`), so the pixel is the control volume the update integrates over.
For H the argument is different but lands on the same box: ``H_c`` sits at an **edge** on its own
axis, so the dual cell is the only box centred on the sample point, which is also Meep's rule (one
cell across, centred on the component's own point). On a uniform grid every pixel is a cube of side
``h`` centred on the sample point.

**Which properties.** The permittivity on the three E lattices, and the permeability on the three H
lattices whenever that array exists at all — which is exactly when some material is magnetic, the
fork's ``all_objects_non_magnetic`` being Meep's ``has_mu``. Conductivity and dispersion stay point
samples, as they do in Meep. One consequence is worth stating: at a magnetic interface pixel the
permeability is the blend while the magnetic conductivity is the point sample of whichever material
won at the H point, so the two can disagree about which side of the interface the pixel is on. The
electric side has been in that state since sub-pixel smoothing was introduced.

**Which pixels are touched.** The material is probed at the pixel centre and at its eight corners.
One material: the pixel is uniform and keeps its point sample, bit for bit. Two materials: the pixel
straddles one interface and is smoothed. Three or more: there is no single planar interface for the
blend to describe, so the point sample is kept — the feature is under-resolved and the loader counts
those pixels. The eight corners of every pixel of one component come from a single lattice, so the
whole probe costs one extra raster pass per component rather than nine point tests per pixel.

**The blend, two isotropic materials.** With fill fraction ``f`` of the front material in the pixel,

    <eps> = f*eps_hi + (1-f)*eps_lo        <1/eps> = f/eps_hi + (1-f)/eps_lo

and unit interface normal ``n``, the effective *inverse* permittivity tensor is

    (1/eps)_eff = n n^T <1/eps> + (I - n n^T) / <eps>

i.e. the harmonic mean along the normal and the arithmetic mean in the interface plane. The default
diagonal tier writes entry ``(c, c)`` of that tensor at component ``c``'s own pixel, which is the
entry the elementwise update applies to ``E_c``; the optional full-tensor tier writes the whole row
``c`` into the 9-component layout, and the loader counts the pixels where the diagonal tier had to
drop a non-zero off-diagonal term.

**The blend, anisotropic materials.** Averaging a tensor entrywise is wrong: the quantity that is
continuous across the interface is not ``E`` or ``D`` but the mixed vector made of the normal
component of ``D`` and the two tangential components of ``E``. Kottke's change of variables ``tau``
is the map onto that vector's conjugate, so ``tau`` of each side, averaged with ``f`` as the weight,
is the correct mean. Both tensors are rotated so that index 0 is the normal, transformed, averaged
entrywise, transformed back, inverted and rotated to the lab frame. Six entries each way, no
iteration; for two multiples of the identity it reduces exactly to the formula above, which is why
an isotropic scene keeps taking the scalar path and its numbers do not move. The transform divides
by ``n^T eps n``, so a side that is not positive definite is refused before the transform runs and
counted with the metals. Meep only checks that in a debug build, so this is stricter than shipped
Meep rather than parity with it.

**Periodic images.** On an axis carrying a periodic or Bloch boundary an object is also evaluated
one lattice vector each way, so a shape crossing the face reappears on the other side. The shape is
never moved: its bounding interval is shifted to window the lattice and the query points are shifted
back into its own frame, which is what libctl and Meep both do and what fdtdx's absolute shape bounds
force anyway. A pixel's fill fraction sums over the images that reach it — they are translates by
whole periods, so they are disjoint and the sum is exact — while the normal comes from the single
image that won the pixel. Where two images both reach one pixel the object shows two faces with
opposite normals there, no single-normal blend describes it, and the pixel is counted and left at
its point sample. The domain-edge pixel of a periodic axis becomes the full mirrored dual box, since
the material below the first edge is now defined; on a terminated axis it stays clipped, which is a
knowing deviation from Meep (Meep pads and uses the full box everywhere) taken because extending the
box past the first edge would evaluate the scene where no simulation volume exists. An axis the
simulation is invariant along never gets images: fdtdx's 2-D convention is a single periodic cell
there, and replicating along it would report a spurious second material at every pixel.

**Why the normals are analytic.** The tensor depends on ``n`` linearly through ``n n^T``, so an
angular error ``delta`` leaves an ``O(h*delta)`` field error. A normal recovered from a fixed-size
fine raster has ``delta = O(1/S)`` independent of ``h`` and would hold the observed convergence order
near one; an analytic per-shape normal has no such floor. The fine-raster gradient inside the pixel
is kept only as a fallback for shapes that cannot answer analytically, and the loader reports how
many pixels used it.
"""

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from fdtdx.core.grid import RectilinearGrid
from fdtdx.core.physics.geometry_raster import (
    _TIE_NUDGE_FRACTION,
    E_OFFSETS,
    H_OFFSETS,
    SHIFT_IDENTITY,
    V_OFFSETS,
    Scene,
    entry_shifts,
    front_indices,
    grid_periods,
    unpack_shift_codes,
)
from fdtdx.materials import compute_allowed_permeabilities, compute_allowed_permittivities
from fdtdx.objects.static_material.static import SimulationVolume

#: A fill fraction this close to 0 or 1 means the interface misses the pixel; keep the point sample.
_FILL_EPS = 1e-12

#: Relative tolerance on "this material is isotropic" before the pixel takes the scalar fast path.
_ISOTROPY_TOL = 1e-12

#: Meep's threshold for "the normal is close enough to z that (n_y, -n_x, 0) is a bad tangent".
_MEEP_TANGENT_THRESHOLD = 1e-2

#: Relative size an off-diagonal entry must reach before the diagonal tier reports it as dropped.
_OFFDIAGONAL_TOL = 1e-12


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
    #: Kept at zero. It used to count pixels refused because a side was anisotropic; the tensor
    #: blend handles those now, and the only remaining refusal (a non-positive-definite tensor) is
    #: counted under ``num_metal_skips``. The field stays so a recorded run's JSON keeps its keys.
    num_anisotropic_skips: int = 0
    #: Pixels where at least one side was anisotropic, so the tensor blend ran instead of the scalar one.
    num_anisotropic_pixels: int = 0
    #: Pixels whose blend produced a non-zero off-diagonal entry that a 3-component array cannot hold.
    num_offdiagonal_dropped: int = 0
    #: Pixels where an input tensor was not symmetric and was symmetrised before the transform.
    num_asymmetric_tensor_pixels: int = 0
    #: Pixels two periodic images of one object both reach. Counted, not smoothed: the object shows
    #: two faces with opposite normals inside one pixel, and no single-normal blend describes that.
    num_multi_shift_pixels: int = 0
    num_supersampled_pixels: int = 0
    #: Vertex pass only. Vertices where at least one of the two materials carries electric
    #: conductivity, so the off-diagonal entries are written as zero: the diagonal branch's lossy
    #: factor ``1 / (1 + c*sigma*eta0*inv_eps/2)`` is a per-component scalar and has no consistent
    #: off-diagonal form without a D-field formulation of the update.
    num_lossy_offdiag_skips: int = 0
    #: Vertex pass only. Vertices where one of the two materials is a genuinely anisotropic *bulk*
    #: tensor (non-zero off-diagonal entries of its own), which the vertex array cannot represent —
    #: its entries would then no longer be purely smoothing-induced. Written as zero and counted;
    #: the run-level gate normally keeps such a scene on the dense pixel path, so this is a
    #: defence-in-depth counter that should read zero.
    num_bulk_tensor_offdiag_skips: int = 0
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
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-axis ``(lower, upper)`` arrays of the pixel boxes of one Yee component.

    An axis the simulation is invariant along returns a degenerate interval (``lower == upper`` at
    the cell centre), which every downstream overlap treats as a membership test.

    Args:
        grid (RectilinearGrid): The resolved simulation grid.
        field_name (str): ``"E"`` or ``"H"``.
        component (int): Component index 0, 1 or 2.
        periodic_axes (tuple): Axes whose domain-edge pixel is the full mirrored dual box rather
            than the clipped half box, because the material below the first edge is defined there:
            it is the periodic image of the material below the last edge.

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
            # On a *periodic* axis the clip is dropped: the material below e[0] is the periodic
            # image of the material below e[N], which front_indices now supplies, so the pixel can
            # be the full dual box the update integrates over. The mirrored half keeps the width
            # w[0] rather than the wrapped neighbour w[N-1], because the pixel must be the control
            # volume the update integrates over and curl.py prepends widths[:1] whatever the
            # boundary; the two differ only on a graded periodic axis, and the real defect is then
            # in the curl. On a terminated axis the clip stays, which is a knowing deviation from
            # Meep -- Meep pads and uses the full box on every axis. Extending the box past e[0]
            # there would evaluate the scene where no SimulationVolume exists, so the loader would
            # invent a background material for a region that is not part of the simulation.
            lower = edges[:-1] - 0.5 * previous
            if not periodic_axes[axis]:
                lower = np.clip(lower, edges[0], None)
            upper = edges[:-1] + 0.5 * widths
            bounds.append((lower, upper))
    return bounds


def pixel_corner_coordinates(
    grid: RectilinearGrid,
    field_name: str,
    component: int,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corner lattice of one component's pixels: one array per axis, ``N_a + 1`` long.

    Corner ``(dx, dy, dz)`` of the pixel of cell ``(i, j, k)`` is ``(q_x[i+dx], q_y[j+dy],
    q_z[k+dz])``, so a single :func:`fdtdx.core.physics.geometry_raster.front_indices` pass over this
    lattice yields all eight corners of all pixels as eight array slices. An invariant axis returns a
    length-1 array (the cell centre), which broadcasts.

    The top corner is pulled just inside the domain so that the resolver's tie nudge cannot push it
    past the upper face of an object that reaches the boundary. The bottom corner follows
    :func:`pixel_axis_bounds`: clipped to the domain on a terminated axis, extended half a cell
    below the first edge on a periodic one, so candidate detection and fill fraction keep agreeing.
    """
    bounds = pixel_axis_bounds(grid, field_name, component, periodic_axes)
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
        floor = None if periodic_axes[axis] else edges[0]
        coords.append(np.clip(corner, floor, edges[-1] - margin))
    return coords[0], coords[1], coords[2]


def _offsets(field_name: str) -> tuple[tuple[float, float, float], ...]:
    if field_name == "E":
        return E_OFFSETS
    if field_name == "H":
        return H_OFFSETS
    if field_name == "V":
        return V_OFFSETS
    raise ValueError(f"field must be 'E', 'H' or 'V', got {field_name!r}")


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


def _interface_frame(normal: np.ndarray) -> np.ndarray:
    """Rotation with the interface normal as row 0, built exactly the way Meep builds its columns.

    Meep puts the normal in column 0, takes column 2 as ``(n_y, -n_x, 0)`` unless the normal is
    within ``1e-2`` of the z axis (then ``(0, -n_z, n_y)``), normalises it, and takes column 1 as
    column 2 cross column 0. Transposed into rows, that is what this returns. The effective tensor
    does not depend on which tangential pair is chosen — the tau map is equivariant under a rotation
    inside the interface plane — but taking the same choice removes one source of last-digit
    disagreement when the two codes are compared on one scene.

    Args:
        normal (np.ndarray): ``(M, 3)`` unit interface normals.

    Returns:
        np.ndarray: ``(M, 3, 3)`` rotations; ``R @ eps @ R.T`` is the tensor in the interface frame.
    """
    nx, ny, nz = normal[:, 0], normal[:, 1], normal[:, 2]
    zero = np.zeros_like(nx)
    off_axis = (np.abs(nx) > _MEEP_TANGENT_THRESHOLD) | (np.abs(ny) > _MEEP_TANGENT_THRESHOLD)
    tangent = np.where(
        off_axis[:, None],
        np.stack([ny, -nx, zero], axis=-1),
        np.stack([zero, -nz, ny], axis=-1),
    )
    length = np.linalg.norm(tangent, axis=-1)
    tangent = tangent / np.where(length > 0.0, length, 1.0)[:, None]
    return np.stack([normal, np.cross(tangent, normal), tangent], axis=1)


def _tau(tensor: np.ndarray) -> np.ndarray:
    """Kottke's change of variables on a symmetric tensor whose index 0 is the interface normal.

    ``tau`` maps a tensor onto the one whose *arithmetic* average across the interface is the
    correct one, because it acts on the field vector that is continuous there: the normal component
    of ``D`` and the two tangential components of ``E``. Entry ``00`` is the inversion along the
    normal; the ``ij`` block is the Schur complement of the normal entry, i.e. what is left of the
    tangential block once the normal direction has been eliminated. Six entries, no iteration.

    Args:
        tensor (np.ndarray): ``(M, 3, 3)`` symmetric tensors with a non-zero ``00`` entry.

    Returns:
        np.ndarray: ``(M, 3, 3)`` transformed tensors, symmetric.
    """
    out = np.zeros_like(tensor)
    m00 = tensor[:, 0, 0]
    m01, m02, m12 = tensor[:, 0, 1], tensor[:, 0, 2], tensor[:, 1, 2]
    out[:, 0, 0] = -1.0 / m00
    out[:, 0, 1] = out[:, 1, 0] = m01 / m00
    out[:, 0, 2] = out[:, 2, 0] = m02 / m00
    out[:, 1, 1] = tensor[:, 1, 1] - m01 * m01 / m00
    out[:, 2, 2] = tensor[:, 2, 2] - m02 * m02 / m00
    out[:, 1, 2] = out[:, 2, 1] = m12 - m01 * m02 / m00
    return out


def _tau_inverse(tensor: np.ndarray) -> np.ndarray:
    """Undo :func:`_tau`. Differs from it only by the sign of the two ``0j`` entries."""
    out = np.zeros_like(tensor)
    d00 = tensor[:, 0, 0]
    d01, d02, d12 = tensor[:, 0, 1], tensor[:, 0, 2], tensor[:, 1, 2]
    out[:, 0, 0] = -1.0 / d00
    out[:, 0, 1] = out[:, 1, 0] = -d01 / d00
    out[:, 0, 2] = out[:, 2, 0] = -d02 / d00
    out[:, 1, 1] = tensor[:, 1, 1] - d01 * d01 / d00
    out[:, 2, 2] = tensor[:, 2, 2] - d02 * d02 / d00
    out[:, 1, 2] = out[:, 2, 1] = d12 - d01 * d02 / d00
    return out


def _symmetric_inverse(tensor: np.ndarray) -> np.ndarray:
    """Inverse of a stack of symmetric 3x3 tensors, by the adjugate.

    ``np.linalg.inv`` raises for the *whole* stack as soon as one member is singular, which would
    make a single degenerate pixel abort the entire load. The closed form cannot raise; the callers
    guarantee a positive-definite input, and a zero determinant would only produce infinities at
    that one pixel. This is also the routine Meep uses (``sym_matrix_invert``).

    Args:
        tensor (np.ndarray): ``(M, 3, 3)`` symmetric tensors.

    Returns:
        np.ndarray: ``(M, 3, 3)`` inverses, symmetric.
    """
    m00, m11, m22 = tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 2, 2]
    m01, m02, m12 = tensor[:, 0, 1], tensor[:, 0, 2], tensor[:, 1, 2]
    c00 = m11 * m22 - m12 * m12
    c01 = m02 * m12 - m01 * m22
    c02 = m01 * m12 - m02 * m11
    determinant = m00 * c00 + m01 * c01 + m02 * c02
    safe = np.where(determinant != 0.0, determinant, 1.0)
    out = np.zeros_like(tensor)
    out[:, 0, 0] = c00 / safe
    out[:, 0, 1] = out[:, 1, 0] = c01 / safe
    out[:, 0, 2] = out[:, 2, 0] = c02 / safe
    out[:, 1, 1] = (m00 * m22 - m02 * m02) / safe
    out[:, 1, 2] = out[:, 2, 1] = (m01 * m02 - m00 * m12) / safe
    out[:, 2, 2] = (m00 * m11 - m01 * m01) / safe
    return out


def kottke_tensor(
    normal: np.ndarray,
    tensor_hi: np.ndarray,
    tensor_lo: np.ndarray,
    fill: np.ndarray,
) -> np.ndarray:
    """Effective *inverse* property tensor of a pixel straddling two anisotropic materials.

    Rotate both tensors into the interface frame, tau-transform each, average the two entrywise
    with the fill fraction as the weight, undo the transform, invert, and rotate back. The result is
    the effective inverse permittivity (or permeability) in the lab frame. For two isotropic
    materials it reduces exactly to ``n n^T <1/eps> + (I - n n^T) / <eps>``, which is what
    :func:`kottke_inverse_permittivity` computes directly.

    Args:
        normal (np.ndarray): ``(M, 3)`` unit interface normals.
        tensor_hi (np.ndarray): ``(M, 3, 3)`` symmetric positive-definite front-material tensors.
        tensor_lo (np.ndarray): ``(M, 3, 3)`` the same for the material behind.
        fill (np.ndarray): ``(M,)`` fraction of the pixel occupied by the front material.

    Returns:
        np.ndarray: ``(M, 3, 3)`` effective inverse property tensors in the lab frame.
    """
    rotation = _interface_frame(normal)
    transposed = np.swapaxes(rotation, -1, -2)
    hi = rotation @ tensor_hi @ transposed
    lo = rotation @ tensor_lo @ transposed
    averaged = fill[:, None, None] * _tau(hi) + (1.0 - fill)[:, None, None] * _tau(lo)
    effective = _tau_inverse(averaged)
    return transposed @ _symmetric_inverse(effective) @ rotation


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


#: The three independent off-diagonal entries of a symmetric 3x3, as ``(row, column)`` index pairs,
#: in the order the vertex array stores them.
OFFDIAGONAL_ENTRIES: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))


def _isotropic_offdiagonal_entries(
    normal: np.ndarray,
    arithmetic: np.ndarray,
    harmonic: np.ndarray,
) -> np.ndarray:
    """The ``(xy, xz, yz)`` entries of the effective inverse permittivity, two isotropic materials.

    Reading entry ``(i, j)`` of ``n n^T <1/eps> + (I - n n^T)/<eps>`` and separating the identity
    part leaves, for ``i != j``, exactly ``n_i n_j (<1/eps> - 1/<eps>)``. The bracket is the gap
    between the mean of the inverse and the inverse of the mean, so by the arithmetic-harmonic mean
    inequality it is strictly positive at a genuine two-material pixel and the entry vanishes if and
    only if ``n_i n_j`` does — that is, at every axis-aligned interface and on every axis the
    simulation is invariant along.

    Args:
        normal (np.ndarray): ``(M, 3)`` unit interface normals.
        arithmetic (np.ndarray): ``(M,)`` fill-weighted arithmetic mean ``<eps>``.
        harmonic (np.ndarray): ``(M,)`` fill-weighted mean of the inverse, ``<1/eps>``.

    Returns:
        np.ndarray: ``(M, 3)`` entries in the order ``(xy, xz, yz)``.
    """
    gap = harmonic - 1.0 / arithmetic
    out = np.zeros((normal.shape[0], 3), dtype=float)
    for entry, (i, j) in enumerate(OFFDIAGONAL_ENTRIES):
        out[:, entry] = normal[:, i] * normal[:, j] * gap
    return out


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


def _material_value_classes(scene: Scene, property_kind: str) -> np.ndarray:
    """Map every global material index onto the first index carrying the same value of one property.

    Two objects can carry the same material under two names (a carve-out drawn in the background
    material, say). Those are one material physically, and counting them as two would report a
    spurious interface. This is the reimplementation of Meep's ``material_type_equal`` test,
    narrowed to the property this pass is smoothing: two materials that differ only in conductivity
    or dispersion present no permittivity step, and on the H lattices a purely dielectric interface
    is not a permeability interface at all — without the narrowing every dielectric face in the
    domain would be probed, filled and normal-solved on the H pass to write back the identity.

    Args:
        scene (Scene): The scene whose global material list is read.
        property_kind (str): ``"permittivity"`` or ``"permeability"``.

    Returns:
        np.ndarray: ``int32`` class index per global material.

    Raises:
        ValueError: If ``property_kind`` is not a smoothed property.
    """
    from fdtdx.materials import compute_ordered_names

    names = compute_ordered_names(scene.materials)
    if property_kind == "permittivity":
        signatures = [tuple(scene.materials[name].permittivity) for name in names]
    elif property_kind == "permeability":
        signatures = [tuple(scene.materials[name].permeability) for name in names]
    else:
        raise ValueError(f"property_kind must be 'permittivity' or 'permeability', got {property_kind!r}")
    classes = np.arange(len(names), dtype=np.int32)
    seen: dict[tuple, int] = {}
    for index, signature in enumerate(signatures):
        classes[index] = seen.setdefault(signature, index)
    return classes


def _property_tensors(scene: Scene, property_kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scalar value, isotropy flag, symmetrised 3x3 tensor and asymmetry flag, per global material.

    fdtdx stores a material property as a general 9-tuple with no symmetry check, while the Kottke
    construction is defined for a symmetric tensor. An asymmetric input is symmetrised as
    ``0.5 * (T + T^T)`` and flagged rather than rejected, so a scene still loads and the loader can
    report how many pixels the symmetrisation touched.

    Args:
        scene (Scene): The scene whose global material list is read.
        property_kind (str): ``"permittivity"`` or ``"permeability"``.

    Returns:
        tuple: ``(scalar, isotropic, tensor, asymmetric)`` of shapes ``(M,)``, ``(M,)``,
        ``(M, 3, 3)`` and ``(M,)``. The scalar is the ``xx`` entry; it is what the isotropic fast
        path blends and it is only meaningful where ``isotropic`` is true.

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
    tensor = table.reshape(table.shape[0], 3, 3)
    magnitude = np.maximum(np.max(np.abs(tensor), axis=(1, 2)), 1.0)
    asymmetric = np.max(np.abs(tensor - np.swapaxes(tensor, -1, -2)), axis=(1, 2)) > 1e-12 * magnitude
    return diagonal[:, 0], isotropic, 0.5 * (tensor + np.swapaxes(tensor, -1, -2)), asymmetric


def _image_reaches(
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    image: np.ndarray,
    box_lower: np.ndarray,
    box_upper: np.ndarray,
) -> bool:
    """Whether one periodic image of an object can touch any of the pixel boxes being filled.

    A bounding-box test only, so it can keep an image that turns out to contribute nothing; it can
    never drop one that does. Its job is to keep an object far from a periodic face on exactly one
    fill-fraction call, so the supersampling count of a scene without images does not move.
    """
    for axis in range(3):
        if bounds[axis][1] + image[axis] <= float(np.min(box_lower[:, axis])):
            return False
        if bounds[axis][0] + image[axis] >= float(np.max(box_upper[:, axis])):
            return False
    return True


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
    write_mode: str = "row",
) -> None:
    """Write one lattice's smoothed entries into the assembled inverse-property array.

    The single place a smoothed value reaches the material array, in all three modes.

    ``write_mode="row"`` is the pixel placement: entry ``(c, c)`` of the blend at component ``c``'s
    own pixel for a 3-component array, the whole row ``c`` for a 9-component one. Meep instead
    evaluates the two off-diagonal entries of row ``c`` on a half-cell-shifted control volume
    (``gv.dV(here - shift1, ...)``), which for all three rows is the same primary-grid vertex; that
    is ``write_mode="offdiag"``, which writes the three independent off-diagonal entries
    ``(xy, xz, yz)`` of one symmetric tensor at that vertex. The two modes write different arrays
    and never both run on the same one.

    Args:
        target (np.ndarray): ``(3 or 9, Nx, Ny, Nz)`` array, modified in place.
        cells (tuple): The three index arrays of the pixels (or vertices) being written.
        blend (np.ndarray): ``(K,)`` diagonal entries, ``(K, 3)`` rows, or ``(K, 3)`` off-diagonal
            entries in the order ``(xy, xz, yz)``.
        component (int): Which component's lattice is being written. Ignored for ``"offdiag"``.
        full_tensor (bool): Whether ``target`` carries 9 components. Ignored for ``"offdiag"``.
        write_mode (str): ``"row"`` (pixel placement) or ``"offdiag"`` (vertex placement).

    Raises:
        ValueError: If ``write_mode`` is not one of the two.
    """
    if write_mode == "offdiag":
        for entry in range(3):
            target[(entry, *cells)] = blend[:, entry]
        return
    if write_mode != "row":
        raise ValueError(f"write_mode must be 'row' or 'offdiag', got {write_mode!r}")
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
    front_shift: np.ndarray,
    periodic_axes: tuple[bool, bool, bool],
    periods: tuple[float, float, float],
    classes: np.ndarray,
    scalar: np.ndarray,
    isotropic: np.ndarray,
    tensors: np.ndarray,
    asymmetric: np.ndarray,
    target: np.ndarray,
    full_tensor: bool,
    supersample: int,
    stats: SmoothingStats,
    write_mode: str = "row",
    lossy: np.ndarray | None = None,
    bulk_tensor: np.ndarray | None = None,
) -> None:
    """Smooth one component's lattice in place: probe, classify, blend, write.

    Args:
        scene (Scene): The scene from :func:`fdtdx.core.physics.geometry_raster.build_scene`.
        grid (RectilinearGrid): The resolved simulation grid.
        field (str): ``"E"`` or ``"H"`` — which Yee lattice this component sits on.
        component (int): Component index 0, 1 or 2.
        front_material (np.ndarray): ``(Nx, Ny, Nz)`` point-sampled global material index.
        front_owner (np.ndarray): ``(Nx, Ny, Nz)`` point-sampled winning entry index.
        front_shift (np.ndarray): ``(Nx, Ny, Nz)`` packed code of the winning periodic image.
        periodic_axes (tuple): Axes that carry periodic images, invariant axes already excluded.
        periods (tuple): Metric period per axis.
        classes (np.ndarray): Material index to value-class map from :func:`_material_value_classes`.
        scalar (np.ndarray): Per-material scalar property value.
        isotropic (np.ndarray): Per-material isotropy flag.
        tensors (np.ndarray): Per-material symmetrised ``(M, 3, 3)`` property tensor.
        asymmetric (np.ndarray): Per-material flag: the stored tensor was not symmetric.
        target (np.ndarray): The inverse-property array, modified in place.
        full_tensor (bool): Write whole rows instead of the diagonal entry. Ignored in
            ``write_mode="offdiag"``.
        supersample (int): Samples per axis where no analytic overlap or normal is available.
        stats (SmoothingStats): Counters, accumulated across components.
        write_mode (str): ``"row"`` for the pixel placement (this is the E and H pass), or
            ``"offdiag"`` for the vertex pass, which writes the three off-diagonal entries
            ``(xy, xz, yz)`` of the symmetric blend into a 3-component array.
        lossy (np.ndarray | None): Per global material flag "carries electric conductivity".
            Required in ``"offdiag"`` mode, where such a vertex is written as zero and counted.
        bulk_tensor (np.ndarray | None): Per global material flag "the stored tensor has a non-zero
            off-diagonal entry of its own". Required in ``"offdiag"`` mode, where such a vertex is
            written as zero and counted.
    """
    vertex_mode = write_mode == "offdiag"
    ignore_global = invariant_axes(grid)
    degenerate_axis = tuple(axis in ignore_global for axis in range(3))

    bounds = pixel_axis_bounds(grid, field, component, periodic_axes)
    corners = pixel_corner_coordinates(grid, field, component, periodic_axes)
    corner_material, corner_owner, corner_shift = front_indices(scene, corners, periodic_axes, periods)

    shape = front_material.shape
    # The pixel centre is probe 0, the eight corners follow. np.argmax below returns the first
    # maximum, so among probes of equal owner priority the centre wins and the corners are ranked
    # in (dx, dy, dz) order. That is the tie rule: the image that owns the pixel centre supplies the
    # normal whenever it owns the pixel at all. A pixel two images genuinely share is not smoothed.
    probe_material = [classes[front_material]]
    probe_owner = [front_owner]
    probe_shift = [front_shift]
    for dx in range(2):
        for dy in range(2):
            for dz in range(2):
                index = tuple(
                    slice(0, 1) if degenerate_axis[axis] else slice(d, d + shape[axis])
                    for axis, d in enumerate((dx, dy, dz))
                )
                probe_material.append(np.broadcast_to(classes[corner_material[index]], shape))
                probe_owner.append(np.broadcast_to(corner_owner[index], shape))
                probe_shift.append(np.broadcast_to(corner_shift[index], shape))
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
    probe_shifts = np.stack(probe_shift, axis=0)[:, candidate]
    low_class = ordered[0][candidate]
    high_class = ordered[-1][candidate]

    # The owner is the highest-priority object among the nine probes: it is the shape whose
    # surface bounds the front material inside the pixel, so it supplies both the fill fraction
    # and the normal. This is fdtdx's write order read as Meep reads its object ids.
    winner = np.argmax(owners, axis=0)
    owner_entry = owners[winner, np.arange(count)]
    owner_class = probe_classes[winner, np.arange(count)]
    other_class = np.where(owner_class == low_class, high_class, low_class)
    owner_image = unpack_shift_codes(probe_shifts[winner, np.arange(count)], periods)

    lower = np.stack([bounds[axis][0][cells[axis]] for axis in range(3)], axis=-1)
    upper = np.stack([bounds[axis][1][cells[axis]] for axis in range(3)], axis=-1)
    center = 0.5 * (lower + upper)

    fill = np.zeros(count, dtype=float)
    normal = np.zeros((count, 3), dtype=float)
    multi_shift = np.zeros(count, dtype=bool)
    for entry_index in np.unique(owner_entry):
        if entry_index < 0:
            continue
        selected = owner_entry == entry_index
        entry = scene.entries[int(entry_index)]
        if isinstance(entry.obj, SimulationVolume):
            continue
        box_lower, box_upper = lower[selected], upper[selected]
        # The fill fraction sums over the images that reach this pixel. They are translates of one
        # shape by whole periods, so they are disjoint and the sum is exact; with no image in range
        # it collapses to the single term the non-periodic path computes. The candidate set is
        # pruned against the boxes actually being filled, so an object nowhere near a periodic face
        # pays nothing and its supersampling count is unchanged.
        contributions = 0
        total = np.zeros(box_lower.shape[0], dtype=float)
        for image in entry_shifts(entry.bounds, periodic_axes, periods):
            if np.any(image != 0.0) and not _image_reaches(entry.bounds, image, box_lower, box_upper):
                continue
            moved = bool(np.any(image != 0.0))
            part = _owner_fill_fraction(
                entry.obj,
                box_lower - image if moved else box_lower,
                box_upper - image if moved else box_upper,
                supersample,
                degenerate_axis,
                stats,
            )
            total += part
            contributions = contributions + (part > _FILL_EPS).astype(np.int32)
        fill[selected] = np.clip(total, 0.0, 1.0)
        multi_shift[selected] = np.asarray(contributions) >= 2
        # The normal belongs to one surface, and the right one is the surface bounding the front
        # material at this pixel: the image the winning probe came from. Meep takes it the same way,
        # in the frame of the object get_front_object returned.
        image_shift = owner_image[selected]
        shifted = bool(np.any(image_shift != 0.0))
        probe_center = center[selected] - image_shift if shifted else center[selected]
        ignore = tuple(sorted(set(ignore_global) | set(_spanning_axes(entry, grid))))
        local = np.asarray(entry.obj.normal_at(probe_center, ignore_axes=ignore), dtype=float)
        missing = np.linalg.norm(local, axis=-1) <= 0.0
        if missing.any():
            stats.num_gradient_normal_fallbacks += int(np.count_nonzero(missing))
            subset = np.nonzero(selected)[0][missing]
            fallback_shift = owner_image[subset]
            local[missing] = _gradient_normal_from_fill(
                entry.obj,
                lower[subset] - fallback_shift,
                upper[subset] - fallback_shift,
                supersample,
                degenerate_axis,
            )
        for axis in ignore_global:
            local[:, axis] = 0.0
        length = np.linalg.norm(local, axis=-1)
        safe = length > 0.0
        normal[selected] = np.where(safe[:, None], local / np.where(safe, length, 1.0)[:, None], 0.0)

    value_hi = scalar[owner_class]
    value_lo = scalar[other_class]
    both_isotropic = isotropic[owner_class] & isotropic[other_class]

    usable = np.ones(count, dtype=bool)

    # Both sides must be invertible in every direction before the tau transform divides by
    # n^T eps n. A metal (eps <= 0) or an indefinite tensor keeps the point sample. The isotropic
    # pair keeps the scalar test it has always used, so this counter cannot move on a scene that
    # carries no tensor material; the eigenvalue test is paid only where one is present. The guard
    # runs before the blend, not inside it, so no pixel ever reaches a division by zero.
    positive = (value_hi > 0.0) & (value_lo > 0.0)
    if not both_isotropic.all():
        anisotropic_cells = ~both_isotropic
        smallest = np.minimum(
            np.linalg.eigvalsh(tensors[owner_class[anisotropic_cells]])[:, 0],
            np.linalg.eigvalsh(tensors[other_class[anisotropic_cells]])[:, 0],
        )
        positive[anisotropic_cells] = smallest > 0.0
    stats.num_metal_skips += int(np.count_nonzero(~positive))
    usable &= positive

    # Two images of one object inside one pixel means two surfaces with opposite normals there.
    # That is the three-material pathology in a two-material pixel: count it, keep the point sample.
    # It needs the object's extent plus the pixel width to exceed the period, so it is confined to a
    # shape that very nearly fills the domain on a periodic axis; a shape that merely crosses the
    # seam presents one face per pixel and is blended like any other.
    stats.num_multi_shift_pixels += int(np.count_nonzero(multi_shift))
    usable &= ~multi_shift

    degenerate_fill = (fill <= _FILL_EPS) | (fill >= 1.0 - _FILL_EPS)
    stats.num_degenerate_fill_fallbacks += int(np.count_nonzero(degenerate_fill))
    usable &= ~degenerate_fill

    zero_normal = np.linalg.norm(normal, axis=-1) <= 0.0
    stats.num_zero_normal_fallbacks += int(np.count_nonzero(zero_normal))
    usable &= ~zero_normal

    if vertex_mode:
        assert lossy is not None and bulk_tensor is not None
        # The diagonal branch multiplies the lossy interface by a per-component scalar
        # 1 / (1 + c*sigma*eta0*inv_eps/2). There is no off-diagonal form of that factor short of
        # writing the update on D, so a vertex whose materials are not both lossless keeps a zero
        # off-diagonal entry rather than a term the lossy factor cannot scale consistently.
        conductive = lossy[owner_class] | lossy[other_class]
        stats.num_lossy_offdiag_skips += int(np.count_nonzero(conductive))
        usable &= ~conductive
        # The vertex array carries the *smoothing-induced* off-diagonals of isotropic and diagonal
        # materials. A material with off-diagonal entries of its own has a bulk term that belongs at
        # its own cells, not on a shared vertex; the run-level gate keeps such a scene on the dense
        # pixel path, and this is the same gate applied per vertex.
        genuinely_anisotropic = bulk_tensor[owner_class] | bulk_tensor[other_class]
        stats.num_bulk_tensor_offdiag_skips += int(np.count_nonzero(genuinely_anisotropic))
        usable &= ~genuinely_anisotropic

    if not usable.any():
        return
    written = tuple(axis_cells[usable] for axis_cells in cells)
    normal_used = normal[usable]
    fill_used = fill[usable]
    isotropic_pair = both_isotropic[usable]
    used = int(np.count_nonzero(usable))
    row = np.zeros((used, 3), dtype=float)
    diagonal_entry = np.zeros(used, dtype=float)

    if isotropic_pair.any():
        value_hi_used = value_hi[usable][isotropic_pair]
        value_lo_used = value_lo[usable][isotropic_pair]
        scalar_fill = fill_used[isotropic_pair]
        arithmetic = scalar_fill * value_hi_used + (1.0 - scalar_fill) * value_lo_used
        harmonic = scalar_fill / value_hi_used + (1.0 - scalar_fill) / value_lo_used
        scalar_normal = normal_used[isotropic_pair]
        if vertex_mode:
            row[isotropic_pair] = _isotropic_offdiagonal_entries(scalar_normal, arithmetic, harmonic)
        else:
            row[isotropic_pair] = kottke_inverse_permittivity(scalar_normal, arithmetic, harmonic, component, True)
            if not full_tensor:
                diagonal_entry[isotropic_pair] = kottke_inverse_permittivity(
                    scalar_normal, arithmetic, harmonic, component, False
                )
    if not isotropic_pair.all():
        tensor_cells = ~isotropic_pair
        effective = kottke_tensor(
            normal_used[tensor_cells],
            tensors[owner_class[usable][tensor_cells]],
            tensors[other_class[usable][tensor_cells]],
            fill_used[tensor_cells],
        )
        if vertex_mode:
            row[tensor_cells] = np.stack([effective[:, 0, 1], effective[:, 0, 2], effective[:, 1, 2]], axis=-1)
        else:
            row[tensor_cells] = effective[:, component, :]
            diagonal_entry[tensor_cells] = effective[:, component, component]

    stats.num_anisotropic_pixels += int(np.count_nonzero(~isotropic_pair))
    stats.num_asymmetric_tensor_pixels += int(
        np.count_nonzero(asymmetric[owner_class[usable]] | asymmetric[other_class[usable]])
    )
    if not full_tensor and not vertex_mode:
        # The 3-component tier holds entry (c, c) only. At a tilted interface the blend genuinely
        # produces off-diagonal terms -- for two isotropic materials as much as for two tensor ones
        # -- and they are dropped here. Report the loss instead of leaving it invisible.
        others = [j for j in range(3) if j != component]
        reference = np.maximum(np.abs(diagonal_entry), 1.0)
        stats.num_offdiagonal_dropped += int(
            np.count_nonzero(np.max(np.abs(row[:, others]), axis=1) > _OFFDIAGONAL_TOL * reference)
        )

    _write_component_entries(
        target,
        written,
        row if (full_tensor or vertex_mode) else diagonal_entry,
        component,
        full_tensor,
        write_mode,
    )
    stats.num_smoothed += int(np.count_nonzero(usable))


def smooth_property_on_yee_pixels(
    scene: Scene,
    grid: RectilinearGrid,
    field: str,
    property_kind: str,
    front_material: np.ndarray,
    front_owner: np.ndarray,
    front_shift: np.ndarray,
    inverse_property: np.ndarray,
    supersample: int,
    full_tensor: bool,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
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
        front_shift (np.ndarray): ``(3, Nx, Ny, Nz)`` packed code of the winning periodic image.
        inverse_property (np.ndarray): ``(3 or 9, Nx, Ny, Nz)`` point-sampled inverse property,
            modified in place and returned.
        supersample (int): Samples per axis where no analytic overlap or normal is available.
        full_tensor (bool): Write the full Kottke row instead of the diagonal entry. Must match the
            array's own tier: a 9-component array is always written as rows, because entry ``(c, c)``
            of a row-major 3x3 sits at index ``4*c``, not at ``c``.
        periodic_axes (tuple): Axes carrying periodic images, invariant axes already excluded.

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
    classes = _material_value_classes(scene, property_kind)
    scalar, isotropic, tensors, asymmetric = _property_tensors(scene, property_kind)
    periods = grid_periods(grid)

    for component in range(3):
        _smooth_component_lattice(
            scene=scene,
            grid=grid,
            field=field,
            component=component,
            front_material=front_material[component],
            front_owner=front_owner[component],
            front_shift=front_shift[component],
            periodic_axes=periodic_axes,
            periods=periods,
            classes=classes,
            scalar=scalar,
            isotropic=isotropic,
            tensors=tensors,
            asymmetric=asymmetric,
            target=inverse_property,
            full_tensor=full_tensor,
            supersample=supersample,
            stats=stats,
        )

    return inverse_property, stats


def smooth_offdiagonal_on_vertex_lattice(
    scene: Scene,
    grid: RectilinearGrid,
    supersample: int,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[np.ndarray, SmoothingStats]:
    """The three off-diagonal Kottke entries of the inverse permittivity, on the cell vertices.

    One extra smoothing pass, on one extra lattice. The vertex sits at ``(e_x[i], e_y[j], e_z[k])``
    and its pixel is the dual box on all three axes, centred on it — the same box
    :func:`pixel_axis_bounds` builds for any axis a component sits on an edge of, so the periodic
    mirroring and the non-periodic clip are inherited unchanged. The probe rule (centre plus eight
    corners), the fill fraction, the analytic normal and the anisotropic ``tau`` path are the ones
    the E and H passes already use.

    The output is the array Meep stores per voxel and applies with its ``OFFDIAG`` macro: because
    the off-diagonal entry of row ``c`` is evaluated half a cell back along ``c``'s own axis, and
    that point is the same vertex for all three rows, both coupled rows read one shared number and
    the assembled D-to-E map is exactly symmetric. The diagonal entries do not move: they stay at
    the component pixels, written by the ordinary diagonal-tier pass.

    Args:
        scene (Scene): The scene from :func:`fdtdx.core.physics.geometry_raster.build_scene`.
        grid (RectilinearGrid): The resolved simulation grid.
        supersample (int): Samples per axis where no analytic overlap or normal is available.
        periodic_axes (tuple): Axes carrying periodic images, invariant axes already excluded.

    Returns:
        tuple: ``(inv_permittivity_offdiag, stats)`` with the array of shape ``(3, Nx, Ny, Nz)``
        holding ``(xy, xz, yz)`` and zero wherever no interface was blended.
    """
    from fdtdx.core.physics.geometry_raster import yee_lattice_coordinates

    periods = grid_periods(grid)
    coords = yee_lattice_coordinates(grid, "V", 0)
    front_material, front_owner, front_shift = front_indices(scene, coords, periodic_axes, periods)

    classes = _material_value_classes(scene, "permittivity")
    scalar, isotropic, tensors, asymmetric = _property_tensors(scene, "permittivity")
    lossy = _electrically_conductive(scene)
    off_diagonal = tensors[:, (0, 0, 1), (1, 2, 2)]
    magnitude = np.maximum(np.max(np.abs(tensors), axis=(1, 2)), 1.0)
    bulk_tensor = np.max(np.abs(off_diagonal), axis=1) > _OFFDIAGONAL_TOL * magnitude

    target = np.zeros((3, *front_material.shape), dtype=np.float64)
    stats = SmoothingStats()
    _smooth_component_lattice(
        scene=scene,
        grid=grid,
        field="V",
        component=0,
        front_material=front_material,
        front_owner=front_owner,
        front_shift=front_shift,
        periodic_axes=periodic_axes,
        periods=periods,
        classes=classes,
        scalar=scalar,
        isotropic=isotropic,
        tensors=tensors,
        asymmetric=asymmetric,
        target=target,
        full_tensor=False,
        supersample=supersample,
        stats=stats,
        write_mode="offdiag",
        lossy=lossy,
        bulk_tensor=bulk_tensor,
    )
    return target, stats


def apply_dtoe_map(
    inv_permittivities: np.ndarray,
    inv_permittivity_offdiag: np.ndarray | None,
    vector: np.ndarray,
    periodic_axes: tuple[bool, bool, bool],
) -> np.ndarray:
    """Apply the assembled D-to-E map to a ``(3, Nx, Ny, Nz)`` vector, on the host.

    The same arithmetic the JAX update does — the elementwise diagonal multiply plus the
    vertex-placed off-diagonal stencil of :func:`fdtdx.fdtd.misc.add_offdiag_correction` — written in
    NumPy so the build-time definiteness check can run it as a ``LinearOperator`` without a JAX
    trace. Boundaries follow the field padding: wrap on a periodic axis, zero elsewhere.

    Args:
        inv_permittivities (np.ndarray): ``(3, Nx, Ny, Nz)`` diagonal entries at the component
            pixels.
        inv_permittivity_offdiag (np.ndarray | None): ``(3, Nx, Ny, Nz)`` vertex entries
            ``(xy, xz, yz)``, or None for the diagonal map alone.
        vector (np.ndarray): ``(3, Nx, Ny, Nz)`` input.
        periodic_axes (tuple): Which axes wrap.

    Returns:
        np.ndarray: ``(3, Nx, Ny, Nz)`` image of ``vector``.
    """
    from fdtdx.fdtd.misc import OFFDIAG_ROW_PARTNERS

    vector = np.asarray(vector, dtype=float)
    out = np.asarray(inv_permittivities, dtype=float) * vector
    if inv_permittivity_offdiag is None:
        return out
    pad = [(0, 0)] + [(1, 1)] * 3
    field_pad = np.zeros((3, *[n + 2 for n in vector.shape[1:]]), dtype=float)
    field_pad[:, 1:-1, 1:-1, 1:-1] = vector
    entry_pad = np.pad(np.asarray(inv_permittivity_offdiag, dtype=float), pad, mode="edge")
    for axis, wrap in enumerate(periodic_axes):
        if not wrap:
            continue
        single = [(0, 0)] * 4
        single[axis + 1] = (1, 1)
        field_pad = np.pad(field_pad[_strip(axis)], single, mode="wrap")
        entry_pad = np.pad(entry_pad[_strip(axis)], single, mode="wrap")
    shape = vector.shape[1:]
    for component in range(3):
        for partner, entry in OFFDIAG_ROW_PARTNERS[component]:
            for near in (0, 1):
                offsets = [0, 0, 0]
                offsets[component] = near
                vertex = _numpy_window(entry_pad[entry], tuple(offsets), shape)
                upper = _numpy_window(field_pad[partner], tuple(offsets), shape)
                lower_offsets = list(offsets)
                lower_offsets[partner] -= 1
                lower = _numpy_window(field_pad[partner], tuple(lower_offsets), shape)
                out[component] += 0.5 * vertex * 0.5 * (upper + lower)
    return out


def _strip(axis: int) -> tuple[slice, ...]:
    """Drop the one-cell halo on one spatial axis of a ``(C, Nx+2, Ny+2, Nz+2)`` array."""
    index: list[slice] = [slice(None)] * 4
    index[axis + 1] = slice(1, -1)
    return tuple(index)


def _numpy_window(array: np.ndarray, offsets: tuple[int, int, int], shape: tuple[int, ...]) -> np.ndarray:
    """The interior of a halo-padded 3-D array, shifted by whole cells on each axis."""
    index = tuple(slice(1 + offsets[axis], 1 + offsets[axis] + shape[axis]) for axis in range(3))
    return array[index]


def min_eigenvalue_of_symmetric_part(
    inv_permittivities: np.ndarray,
    inv_permittivity_offdiag: np.ndarray,
    periodic_axes: tuple[bool, bool, bool],
    num_probes: int = 4,
    seed: int = 0,
) -> dict[str, float]:
    """Smallest eigenvalue of the symmetric part of the assembled D-to-E map, by sparse Lanczos.

    With ``M`` the D-to-E map and ``C`` the discrete curl, the semi-discrete system is
    ``d2E/dt2 = -M C^T C E``, so the squared frequencies are the eigenvalues of ``M K`` with
    ``K = C^T C`` symmetric positive semi-definite. If ``M`` is symmetric **positive definite**,
    ``M K`` is similar to ``M^(1/2) K M^(1/2)`` and every squared frequency is real and
    non-negative — no mode can grow. Symmetry alone is not enough: the vertex placement is symmetric
    to machine zero at every contrast, and yet above a permittivity contrast of roughly 30 on a
    curved rim the symmetric part loses definiteness and a real negative squared frequency appears.
    That is what this check catches, at a cost of a few dozen mat-vecs at build time and nothing per
    step.

    The measured asymmetry is reported alongside, because ``eigsh`` assumes a symmetric operator and
    a non-zero value would invalidate its answer rather than merely be interesting.

    Args:
        inv_permittivities (np.ndarray): ``(3, Nx, Ny, Nz)`` diagonal entries.
        inv_permittivity_offdiag (np.ndarray): ``(3, Nx, Ny, Nz)`` vertex entries.
        periodic_axes (tuple): Which axes wrap.
        num_probes (int): Random vector pairs used for the asymmetry estimate.
        seed (int): Seed of those probes, so the reported number is reproducible.

    Returns:
        dict: ``min_eig_sym_dtoe``, ``max_eig_sym_dtoe``, ``asym_rel_dtoe`` and
        ``min_eig_sym_dtoe_positive``.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    shape = inv_permittivities.shape
    size = int(np.prod(shape))

    def matvec(flat: np.ndarray) -> np.ndarray:
        return apply_dtoe_map(
            inv_permittivities, inv_permittivity_offdiag, flat.reshape(shape).astype(float), periodic_axes
        ).reshape(-1)

    operator = LinearOperator((size, size), matvec=matvec, rmatvec=matvec, dtype=float)

    # <u, M v> - <M u, v> over random probes: exactly zero for a symmetric map, and the number to
    # look at first if the eigenvalues ever look wrong.
    rng = np.random.default_rng(seed)
    asymmetry = 0.0
    scale = 0.0
    for _ in range(num_probes):
        u = rng.standard_normal(size)
        v = rng.standard_normal(size)
        mu, mv = matvec(u), matvec(v)
        asymmetry = max(asymmetry, abs(float(u @ mv - mu @ v)))
        scale = max(scale, abs(float(u @ mv)), abs(float(mu @ v)))

    smallest = float(eigsh(operator, k=1, which="SA", return_eigenvectors=False, tol=1e-10, maxiter=20000)[0])
    largest = float(eigsh(operator, k=1, which="LA", return_eigenvectors=False, tol=1e-10, maxiter=20000)[0])
    return {
        "min_eig_sym_dtoe": smallest,
        "max_eig_sym_dtoe": largest,
        "asym_rel_dtoe": asymmetry / scale if scale > 0.0 else 0.0,
        "min_eig_sym_dtoe_positive": bool(smallest > 0.0),
    }


def _electrically_conductive(scene: Scene) -> np.ndarray:
    """Per global material: does it carry any electric conductivity at all?"""
    from fdtdx.materials import compute_allowed_electric_conductivities

    table = np.asarray(compute_allowed_electric_conductivities(scene.materials), dtype=float)
    return np.max(np.abs(table), axis=1) > 0.0


def smooth_inverse_permittivity_on_yee_pixels(
    scene: Scene,
    grid: RectilinearGrid,
    front_material: np.ndarray,
    front_owner: np.ndarray,
    inv_permittivities: np.ndarray,
    supersample: int,
    full_tensor: bool,
    front_shift: np.ndarray | None = None,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[np.ndarray, SmoothingStats]:
    """Smooth the permittivity on the three E lattices; see :func:`smooth_property_on_yee_pixels`."""
    if front_shift is None:
        front_shift = np.full(front_owner.shape, SHIFT_IDENTITY, dtype=np.int8)
    return smooth_property_on_yee_pixels(
        scene=scene,
        grid=grid,
        field="E",
        property_kind="permittivity",
        front_material=front_material,
        front_owner=front_owner,
        front_shift=front_shift,
        inverse_property=inv_permittivities,
        supersample=supersample,
        full_tensor=full_tensor,
        periodic_axes=periodic_axes,
    )
