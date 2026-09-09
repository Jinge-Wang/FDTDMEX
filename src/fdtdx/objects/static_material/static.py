from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.colors import XKCD_LIGHT_GREY, Color
from fdtdx.core.jax.pytrees import autoinit, field, frozen_field
from fdtdx.materials import Material, compute_ordered_names
from fdtdx.objects.object import OrderableObject


def points_in_metric_slab(coordinate: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Half-open ``[lower, upper)`` membership test for one axis of a metric point set.

    Half-open on purpose: two objects that share a face (a substrate top and a waveguide bottom)
    must not both claim the sample sitting exactly on that face.

    Args:
        coordinate (np.ndarray): Coordinates along one axis, in metres.
        lower (float): Lower bound in metres.
        upper (float): Upper bound in metres.

    Returns:
        np.ndarray: Boolean array of the same shape as ``coordinate``.
    """
    return (coordinate >= lower) & (coordinate < upper)


def interval_overlap_fraction(
    lower: np.ndarray,
    upper: np.ndarray,
    obj_lower: float,
    obj_upper: float,
) -> np.ndarray:
    """Fraction of each interval ``[lower, upper]`` covered by ``[obj_lower, obj_upper)``.

    A degenerate interval (``upper <= lower``) stands for a sample point rather than a box side —
    an axis the simulation is invariant along — and returns the half-open membership of ``lower``.

    Args:
        lower (np.ndarray): Lower interval bounds in metres.
        upper (np.ndarray): Upper interval bounds in metres, same shape as ``lower``.
        obj_lower (float): Object's lower bound on this axis, in metres.
        obj_upper (float): Object's upper bound on this axis, in metres.

    Returns:
        np.ndarray: Fractions in ``[0, 1]``, same shape as ``lower``.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    overlap = np.clip(np.minimum(upper, obj_upper) - np.maximum(lower, obj_lower), 0.0, None)
    point = ((lower >= obj_lower) & (lower < obj_upper)).astype(float)
    return np.where(width > 0.0, overlap / np.where(width > 0.0, width, 1.0), point)


def _oriented_polygon_edges(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Edge start points, edge vectors and outward unit normals of one closed 2-D polygon.

    The outward direction is fixed from the polygon's own signed area, so a clockwise and a
    counter-clockwise listing of the same shape give the same outward normals.

    Args:
        vertices (np.ndarray): ``(N, 2)`` vertex array; the polygon is closed implicitly.

    Returns:
        tuple: ``(starts, edges, normals)``, each ``(N, 2)``.
    """
    verts = np.asarray(vertices, dtype=float)
    if verts.shape[0] > 1 and np.allclose(verts[0], verts[-1]):
        verts = verts[:-1]
    starts = verts
    ends = np.roll(verts, -1, axis=0)
    edges = ends - starts
    signed_area = 0.5 * float(np.sum(starts[:, 0] * ends[:, 1] - ends[:, 0] * starts[:, 1]))
    # For a counter-clockwise listing (positive area) the outward normal of edge (dx, dy) is
    # (dy, -dx); a clockwise listing flips it.
    normals = np.stack((edges[:, 1], -edges[:, 0]), axis=-1)
    if signed_area < 0.0:
        normals = -normals
    length = np.linalg.norm(normals, axis=-1)
    safe = length > 0.0
    normals = normals / np.where(safe, length, 1.0)[:, None]
    return starts, edges, normals


def nearest_polygon_edge_normal(
    polygons: list[np.ndarray],
    query_h: np.ndarray,
    query_v: np.ndarray,
    chunk: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Outward unit normal of the nearest polygon edge, and the distance to it.

    This is the in-plane half of libctl's nearest-surface rule for a prism: every side is a
    candidate surface and the closest one wins. The returned normal is signed outward; only
    ``n n^T`` is used downstream, so the sign is a convenience rather than a requirement.

    Args:
        polygons (list[np.ndarray]): One or more ``(N, 2)`` vertex arrays.
        query_h (np.ndarray): Horizontal query coordinates, any shape.
        query_v (np.ndarray): Vertical query coordinates, same shape as ``query_h``.
        chunk (int): Maximum number of point-edge pairs evaluated at once.

    Returns:
        tuple: ``(normals, distances)`` with ``normals`` of shape ``query_h.shape + (2,)``.
    """
    shape = np.asarray(query_h).shape
    points = np.stack((np.asarray(query_h, dtype=float).ravel(), np.asarray(query_v, dtype=float).ravel()), axis=-1)
    best_dist = np.full(points.shape[0], np.inf)
    best_normal = np.zeros((points.shape[0], 2), dtype=float)
    for polygon in polygons:
        starts, edges, normals = _oriented_polygon_edges(polygon)
        if starts.shape[0] == 0:
            continue
        length_sq = np.sum(edges**2, axis=-1)
        length_sq = np.where(length_sq > 0.0, length_sq, 1.0)
        step = max(1, chunk // max(1, starts.shape[0]))
        for begin in range(0, points.shape[0], step):
            block = points[begin : begin + step]
            rel = block[:, None, :] - starts[None, :, :]
            t = np.clip(np.sum(rel * edges[None, :, :], axis=-1) / length_sq[None, :], 0.0, 1.0)
            closest = rel - t[..., None] * edges[None, :, :]
            dist = np.linalg.norm(closest, axis=-1)
            idx = np.argmin(dist, axis=-1)
            local = dist[np.arange(block.shape[0]), idx]
            better = local < best_dist[begin : begin + step]
            best_dist[begin : begin + step] = np.where(better, local, best_dist[begin : begin + step])
            best_normal[begin : begin + step] = np.where(
                better[:, None], normals[idx], best_normal[begin : begin + step]
            )
    return best_normal.reshape(*shape, 2), best_dist.reshape(shape)


def _box_face_normal(
    points: np.ndarray,
    center: tuple[float, float, float],
    half: tuple[float, float, float],
    ignore_axes: tuple[int, ...],
) -> np.ndarray:
    """Outward normal of the nearest face of an axis-aligned box, libctl's block rule."""
    pts = np.asarray(points, dtype=float)
    distances = []
    for axis in range(3):
        d = np.abs(np.abs(pts[..., axis] - center[axis]) - half[axis])
        distances.append(np.full(pts.shape[:-1], np.inf) if axis in ignore_axes else d)
    stacked = np.stack(distances, axis=-1)
    normal = np.zeros(pts.shape, dtype=float)
    if not np.isfinite(stacked).any():
        return normal
    winner = np.argmin(stacked, axis=-1)
    for axis in range(3):
        if axis in ignore_axes:
            continue
        sign = np.sign(pts[..., axis] - center[axis])
        sign = np.where(sign == 0.0, 1.0, sign)
        normal[..., axis] = np.where(winner == axis, sign, 0.0)
    return normal


@autoinit
class UniformMaterialObject(OrderableObject):
    #: the material object
    material: Material = field()

    #: the color object
    color: Color | None = frozen_field(default=XKCD_LIGHT_GREY)

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Test which metric points lie inside this object's continuous box.

        The box is the object's :attr:`~fdtdx.objects.object.SimulationObject.metric_bounds`, i.e.
        the extent it was placed with before rounding to whole cells, not the rounded box.

        Args:
            points (np.ndarray): Array of shape ``(..., 3)`` with coordinates in metres.

        Returns:
            np.ndarray: Boolean array of shape ``points.shape[:-1]``.
        """
        pts = np.asarray(points, dtype=float)
        bounds = self.metric_bounds
        inside = np.ones(pts.shape[:-1], dtype=bool)
        for axis in range(3):
            inside &= points_in_metric_slab(pts[..., axis], bounds[axis][0], bounds[axis][1])
        return inside

    def normal_at(self, points: np.ndarray, ignore_axes: tuple[int, ...] = ()) -> np.ndarray:
        """Outward unit normal of the nearest face of the object's continuous box.

        Args:
            points (np.ndarray): Array of shape ``(..., 3)`` with coordinates in metres.
            ignore_axes (tuple[int, ...]): Axes whose faces are not physical surfaces (an axis the
                simulation is invariant along, or one the object spans the whole domain on). See
                :meth:`StaticMultiMaterialObject.normal_at` for the rule that decides which.

        Returns:
            np.ndarray: Array of shape ``(..., 3)``; zero where no surface is available.
        """
        bounds = self.metric_bounds
        center = self.metric_center
        half = (
            0.5 * (bounds[0][1] - bounds[0][0]),
            0.5 * (bounds[1][1] - bounds[1][0]),
            0.5 * (bounds[2][1] - bounds[2][0]),
        )
        return _box_face_normal(points, center, half, ignore_axes)

    def box_fill_fraction(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray | None:
        """Exact fraction of each axis-aligned box ``[lower, upper]`` covered by this object.

        Args:
            lower (np.ndarray): ``(..., 3)`` lower box corners in metres.
            upper (np.ndarray): ``(..., 3)`` upper box corners in metres.

        Returns:
            np.ndarray | None: Fractions of shape ``lower.shape[:-1]``.
        """
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        bounds = self.metric_bounds
        fraction = np.ones(lower.shape[:-1], dtype=float)
        for axis in range(3):
            fraction = fraction * interval_overlap_fraction(
                lower[..., axis], upper[..., axis], bounds[axis][0], bounds[axis][1]
            )
        return fraction


@autoinit
class StaticMultiMaterialObject(OrderableObject, ABC):
    #: the static material
    materials: dict[str, Material] = field()

    #: the color of the material
    color: Color | None = frozen_field(default=XKCD_LIGHT_GREY)

    #: Enable sub-pixel (sub-cell) dielectric smoothing for this object. When ``True`` the assembler
    #: replaces the binary voxel occupancy with an analytic fill-fraction and builds a smoothed,
    #: anisotropic (full 3x3 tensor) effective permittivity at interface cells following Farjadpour et
    #: al. (Meep): arithmetic mean of ``eps`` for the field components tangential to the interface and
    #: harmonic mean of ``eps`` for the component normal to it. This removes the first-order staircasing
    #: error of the Yee grid at strong dielectric jumps (2nd-order accuracy). Forces the whole
    #: simulation to allocate an anisotropic permittivity tensor (3-component diagonal by default, or a
    #: full 9-component tensor when ``subpixel_full_tensor`` is set). Requires the object to provide a
    #: fractional ``get_fill_fraction_for_shape`` (the default falls back to the binary mask, which
    #: still yields a valid but only cell-wide normal). See issue #373.
    subpixel_smoothing: bool = frozen_field(default=False)

    #: Selects the smoothing tensor representation when ``subpixel_smoothing`` is on. ``False`` (default)
    #: keeps only the DIAGONAL of the Farjadpour tensor (``eps_ii = eps_bar - (eps_bar - eps_h)*n_i**2``),
    #: allocating a cheap 3-component array that runs on the elementwise Yee update. This is EXACT for
    #: axis-aligned interfaces (their normal lies on one axis, so the off-diagonal terms vanish) and is the
    #: recommended production path for Manhattan geometries. ``True`` allocates the full 9-component tensor
    #: (keeps the off-diagonal ``-(eps_bar - eps_h)*n_i*n_j`` terms), which is more accurate for tilted
    #: interfaces (slanted sidewalls, diagonal edges) but ~3x heavier per step and forces the anisotropic
    #: update kernel. Ignored when ``subpixel_smoothing`` is False.
    subpixel_full_tensor: bool = frozen_field(default=False)

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Test which metric points lie inside this object's continuous shape.

        This is the continuous counterpart of :meth:`get_voxel_mask_for_shape`: it answers the same
        question — is this location material? — but at an arbitrary point in metres rather than at a
        cell centre of the object's rounded box. It is what ``material_sampling="yee"`` queries at
        every Yee component position.

        Args:
            points (np.ndarray): Array of shape ``(..., 3)`` with coordinates in metres, on the
                simulation grid's own axes.

        Returns:
            np.ndarray: Boolean array of shape ``points.shape[:-1]``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement contains(); it cannot be used with material_sampling='yee'."
        )

    def normal_at(self, points: np.ndarray, ignore_axes: tuple[int, ...] = ()) -> np.ndarray:
        """Outward unit normal of the nearest surface element of this object's continuous shape.

        This is the analytic counterpart of a fill-fraction gradient: the surface normal is read off
        the shape itself, not off a raster, so it carries no cell-size-dependent angular error. That
        matters for ``material_sampling="yee_smooth"``, where the effective inverse permittivity
        depends on ``n n^T`` linearly: an angular error that does not shrink with the cell size caps
        the achievable convergence order at one.

        The rule is nearest-surface, the same one libctl uses for its analytic primitives: at a point
        near a cap-and-wall junction the returned normal is the closer of the two adjoining faces.
        Where nothing can be decided (a degenerate shape, a point equidistant from two surfaces) the
        zero vector is returned and the caller keeps the point sample.

        **The ignore_axes rule** (defined here; this is its only normative statement, every
        implementation below follows it). An axis is passed in ``ignore_axes`` when a face
        perpendicular to it is not a physical interface the blend should see. The loader
        (:func:`fdtdx.core.physics.geometry_smooth.smooth_inverse_permittivity_on_yee_pixels`) marks
        an axis for exactly two reasons:

        1. **Invariant axis** — the simulation resolves it with a single cell
           (:func:`fdtdx.core.physics.geometry_smooth.invariant_axes`). fdtdx's 2-D convention is one
           cell with periodic boundaries on the third axis, so the object's extent along it is a
           modelling artifact, not a surface. Without the rule every pixel of a 2-D scene reports the
           out-of-plane cap as its nearest face and the whole cross-section gets a z normal.
        2. **Domain-spanning axis** — the object covers the full domain on that axis
           (:func:`fdtdx.core.physics.geometry_smooth._spanning_axes`), so its "caps" coincide with
           the domain boundary and are the simulation's edge rather than a material interface. A
           strip waveguide drawn the full length of the domain otherwise returns an x normal at
           every pixel of its cross-section.

        Rule 1 is exact. Rule 2 is a heuristic, and this is where it can be wrong: an object that
        genuinely ends at the domain boundary *and* has a real face there loses that face, and its
        interface pixels fall back to the next-nearest surface or to the point sample. Neither Meep
        nor libctl has this rule — it is fdtdx's own, forced by fdtdx's 2-D convention. No case in
        ``cases/`` places a real material face on the domain boundary.

        Both rules are read in the object's **own** frame, which is what makes them right on a
        periodic axis too. An object that covers the whole domain there genuinely has no cap, so
        rule 2 removes one that does not exist. An object that merely *crosses* the periodic seam
        does not span the domain in its own frame — its bounds stick out of one end — so it keeps
        its caps, and the loader asks for the normal at the query point translated back into that
        frame. Nothing here needs to know that the object was replicated.

        Args:
            points (np.ndarray): Array of shape ``(..., 3)`` with coordinates in metres, on the
                simulation grid's own axes.
            ignore_axes (tuple[int, ...]): Axes whose surfaces are not physical interfaces, per the
                rule above. An implementation must return no normal component along such an axis and
                must not let a face perpendicular to it win the nearest-surface competition.

        Returns:
            np.ndarray: Array of shape ``(..., 3)`` with unit normals, zero where undefined.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement normal_at(); it cannot be used with "
            "material_sampling='yee_smooth'."
        )

    def box_fill_fraction(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray | None:
        """Exact fraction of each axis-aligned box ``[lower, upper]`` covered by this object.

        Returning ``None`` means "no analytic overlap available"; the caller then super-samples
        :meth:`contains` inside the box instead. A degenerate axis (``upper == lower``) stands for an
        axis the simulation is invariant along and contributes a membership test rather than a
        length ratio.

        Args:
            lower (np.ndarray): ``(..., 3)`` lower box corners in metres.
            upper (np.ndarray): ``(..., 3)`` upper box corners in metres.

        Returns:
            np.ndarray | None: Fractions in ``[0, 1]`` of shape ``lower.shape[:-1]``, or ``None``.
        """
        return None

    def material_at(self, points: np.ndarray) -> np.ndarray:
        """Local material index (into ``compute_ordered_names(self.materials)``) at each point.

        Every concrete object in the tree today carries one material over its whole shape, so the
        default returns that constant. Objects whose material varies in space override this.

        Args:
            points (np.ndarray): Array of shape ``(..., 3)`` with coordinates in metres.

        Returns:
            np.ndarray: Integer array of shape ``points.shape[:-1]``.
        """
        pts = np.asarray(points, dtype=float)
        return np.full(pts.shape[:-1], self.constant_material_index(), dtype=np.int32)

    def constant_material_index(self) -> int:
        """Index of this object's single material in its own ordered material list.

        Returns:
            int: Position of ``self.material_name`` in ``compute_ordered_names(self.materials)``.

        Raises:
            NotImplementedError: If the object has no single ``material_name``.
        """
        name = getattr(self, "material_name", None)
        if name is None:
            raise NotImplementedError(f"{type(self).__name__} has no single material_name; override material_at().")
        return compute_ordered_names(self.materials).index(name)

    @abstractmethod
    def get_voxel_mask_for_shape(self) -> jax.Array:
        """Get a binary mask of the objects shape. Everything voxel not in the mask, will not be updated by
        this object. For example, can be used to approximate a round shape.
        The mask is calculated in device voxel size, not in simulation voxels.

        Returns:
            jax.Array: Binary mask representing the voxels occupied by the object
        """
        raise NotImplementedError()

    @abstractmethod
    def get_material_mapping(
        self,
    ) -> jax.Array:
        """Returns an array, which represents the material index at every voxel. Specifically, it returns the
        index of the ordered material list.

        Returns:
            jax.Array: Index array
        """
        raise NotImplementedError()

    def get_fill_fraction_for_shape(self) -> jax.Array:
        """Return the per-cell fill fraction of the object's material, in ``[0, 1]``.

        This is the sub-pixel generalisation of :meth:`get_voxel_mask_for_shape`: interior cells return
        ``1.0``, exterior cells ``0.0`` and interface cells the fraction of the cell volume covered by
        the object. The default implementation falls back to the binary mask cast to float, so a subclass
        that does not compute a genuine fill fraction still behaves correctly (albeit without the
        sub-pixel accuracy gain). Subclasses that can rasterise fractionally should override this.

        Returns:
            jax.Array: Float array of shape ``self.grid_shape`` with values in ``[0, 1]``.
        """
        return self.get_voxel_mask_for_shape().astype(float)

    def get_interface_normal_for_shape(self) -> jax.Array:
        """Return a per-cell unit interface normal derived from the fill-fraction gradient.

        The normal is ``n = -grad(fill) / |grad(fill)|`` (the sign is irrelevant downstream because only
        the symmetric outer product ``n ⊗ n`` is used). The gradient is taken with the object's physical
        cell pitch on each axis, so the direction is geometrically correct on anisotropic grids. Cells
        away from an interface (``|grad(fill)| ~ 0``) get a zero normal, which makes the smoothed tensor
        collapse back to the isotropic bulk value. Computed in NumPy at initialisation (static geometry,
        not a traced quantity).

        Returns:
            jax.Array: Float array of shape ``(3, *self.grid_shape)`` with the per-cell unit normal.
        """
        frac = np.asarray(self.get_fill_fraction_for_shape(), dtype=float)
        grid_shape = self.grid_shape
        real_shape = self.real_shape
        # Average physical pitch per axis (exact on uniform grids, a good local approximation on
        # quasi-uniform grids; only the gradient *direction* matters after normalisation).
        pitch = [(real_shape[i] / grid_shape[i]) if grid_shape[i] > 0 else 1.0 for i in range(3)]
        grad = np.zeros((3, *frac.shape), dtype=float)
        for ax in range(3):
            if frac.shape[ax] > 1 and pitch[ax] > 0:
                grad[ax] = np.gradient(frac, pitch[ax], axis=ax)
        # n points from high fill (material) to low fill (background); sign is immaterial for n⊗n.
        grad = -grad
        norm = np.sqrt(np.sum(grad**2, axis=0))
        safe = norm > 1e-12
        normal = np.zeros_like(grad)
        for ax in range(3):
            normal[ax] = np.where(safe, grad[ax] / np.where(safe, norm, 1.0), 0.0)
        return jnp.asarray(normal)


@autoinit
class SimulationVolume(UniformMaterialObject):
    """Background material for the entire simulation volume.

    Defines the default material properties for the simulation background.
    Usually represents air/vacuum with εᵣ=1.0 and μᵣ=1.0.
    """

    #: an integer values of the placement order
    placement_order: int = frozen_field(default=-1000)

    #: the static material
    material: Material = field(
        default=Material(
            permittivity=(1.0, 1.0, 1.0),
            permeability=(1.0, 1.0, 1.0),
        ),
    )
