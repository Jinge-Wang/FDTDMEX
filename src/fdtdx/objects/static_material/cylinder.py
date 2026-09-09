import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.core.axis import get_transverse_axes
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.materials import compute_ordered_names
from fdtdx.objects.static_material.static import StaticMultiMaterialObject, points_in_metric_slab


@autoinit
class Cylinder(StaticMultiMaterialObject):
    """A cylindrical optical fiber with configurable properties.

    This class represents a cylindrical fiber with customizable radius, material,
    and orientation. The fiber can be positioned along any of the three principal axes.

    The cross-section size (diameter = 2 * radius) is automatically inferred for the
    two axes perpendicular to ``axis``, so ``partial_real_shape`` does not need to be
    specified for those axes.  The extrusion axis size must still be determined by a
    constraint or an explicit ``partial_real_shape`` entry.
    """

    #: The radius of the fiber in meter.
    radius: float = frozen_field()

    #: The principal axis along which the fiber extends (0=x, 1=y, 2=z).
    axis: int = frozen_field()

    #: Name of the material in the materials dictionary to be used for the object.
    material_name: str = frozen_field()

    def __post_init__(self):
        diameter = 2.0 * self.radius
        real_shape = list(self.partial_real_shape)
        grid_shape = list(self.partial_grid_shape)
        for ax in (self.horizontal_axis, self.vertical_axis):
            if real_shape[ax] is not None:
                raise Exception(
                    f"Cylinder {self.name}: partial_real_shape for axis {ax} is derived from the radius "
                    f"({diameter:.3e} m). Do not specify it explicitly."
                )
            if grid_shape[ax] is not None:
                raise Exception(
                    f"Cylinder {self.name}: partial_grid_shape for axis {ax} is derived from the radius. "
                    f"Do not specify it explicitly."
                )
            real_shape[ax] = diameter
        object.__setattr__(self, "partial_real_shape", tuple(real_shape))

    @property
    def horizontal_axis(self) -> int:
        """Gets the horizontal axis perpendicular to the fiber axis."""
        return get_transverse_axes(self.axis)[0]

    @property
    def vertical_axis(self) -> int:
        """Gets the vertical axis perpendicular to the fiber axis."""
        return get_transverse_axes(self.axis)[1]

    def get_voxel_mask_for_shape(self) -> jax.Array:
        def local_centers(axis: int) -> jax.Array:
            """Return physical cell centers relative to this object's lower edge."""
            lower, upper = self.grid_slice_tuple[axis]
            grid = self._config.resolved_grid
            if grid is None:
                spacing = self._config.uniform_spacing()
                return (jnp.arange(self.grid_shape[axis]) + 0.5) * spacing
            edges = grid.edges(axis)
            return 0.5 * (edges[lower:upper] + edges[lower + 1 : upper + 1]) - edges[lower]

        horizontal = local_centers(self.horizontal_axis)
        vertical = local_centers(self.vertical_axis)
        horizontal_grid, vertical_grid = jnp.meshgrid(horizontal, vertical, indexing="ij")
        center_h = 0.5 * self.real_shape[self.horizontal_axis]
        center_v = 0.5 * self.real_shape[self.vertical_axis]
        grid = jnp.stack((horizontal_grid - center_h, vertical_grid - center_v), axis=-1) / self.radius

        mask = (grid**2).sum(axis=-1) < 1
        mask = jnp.expand_dims(mask, axis=self.axis)
        return mask

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Continuous point-in-shape test: a disk of ``self.radius`` extruded over the metric extent.

        Args:
            points (np.ndarray): Array of shape ``(..., 3)`` with coordinates in metres.

        Returns:
            np.ndarray: Boolean array of shape ``points.shape[:-1]``.
        """
        pts = np.asarray(points, dtype=float)
        bounds = self.metric_bounds
        center = self.metric_center
        inside = points_in_metric_slab(pts[..., self.axis], bounds[self.axis][0], bounds[self.axis][1])
        radial = (pts[..., self.horizontal_axis] - center[self.horizontal_axis]) ** 2 + (
            pts[..., self.vertical_axis] - center[self.vertical_axis]
        ) ** 2
        return inside & (radial < self.radius**2)

    def normal_at(self, points: np.ndarray, ignore_axes: tuple[int, ...] = ()) -> np.ndarray:
        """Outward normal of the nearest cylinder surface: the barrel or one of the two caps.

        Args:
            points (np.ndarray): Array of shape ``(..., 3)`` with coordinates in metres.
            ignore_axes (tuple[int, ...]): Axes whose surfaces are not physical interfaces. With the
                extrusion axis listed the caps are dropped and the barrel always wins, which is what
                a 2-D (single-cell) simulation needs. See :meth:`~fdtdx.objects.static_material.static.StaticMultiMaterialObject.normal_at` for the rule that decides which axes are listed.

        Returns:
            np.ndarray: Array of shape ``(..., 3)`` with unit normals, zero where undefined.
        """
        pts = np.asarray(points, dtype=float)
        center = self.metric_center
        extent = self.metric_extent
        h_axis, v_axis = self.horizontal_axis, self.vertical_axis
        rad_h = pts[..., h_axis] - center[h_axis]
        rad_v = pts[..., v_axis] - center[v_axis]
        prad = np.sqrt(rad_h**2 + rad_v**2)
        proj = pts[..., self.axis] - center[self.axis]
        half = 0.5 * extent[self.axis]

        normal = np.zeros(pts.shape, dtype=float)
        radial_ok = prad > 0.0
        radial_dist = np.abs(prad - self.radius)
        if self.axis in ignore_axes:
            axial = np.zeros(prad.shape, dtype=bool)
        else:
            cap_dist = np.abs(np.abs(proj) - half)
            axial = (np.abs(proj) > half) | (cap_dist < radial_dist) | ~radial_ok
        sign = np.where(np.sign(proj) == 0.0, 1.0, np.sign(proj))
        normal[..., self.axis] = np.where(axial, sign, 0.0)
        safe = np.where(radial_ok, prad, 1.0)
        normal[..., h_axis] = np.where(axial | ~radial_ok, 0.0, rad_h / safe)
        normal[..., v_axis] = np.where(axial | ~radial_ok, 0.0, rad_v / safe)
        return normal

    def box_fill_fraction(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray | None:
        """Exact circle-rectangle overlap in plane times the extrusion overlap along ``axis``."""
        from fdtdx.core.physics.geometry_smooth import circle_rectangle_area
        from fdtdx.objects.static_material.static import interval_overlap_fraction

        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        h_axis, v_axis = self.horizontal_axis, self.vertical_axis
        if np.any(upper[..., h_axis] <= lower[..., h_axis]) or np.any(upper[..., v_axis] <= lower[..., v_axis]):
            return None
        center = self.metric_center
        bounds = self.metric_bounds
        area = circle_rectangle_area(
            lower[..., h_axis] - center[h_axis],
            upper[..., h_axis] - center[h_axis],
            lower[..., v_axis] - center[v_axis],
            upper[..., v_axis] - center[v_axis],
            self.radius,
        )
        rect = (upper[..., h_axis] - lower[..., h_axis]) * (upper[..., v_axis] - lower[..., v_axis])
        in_plane = area / rect
        along = interval_overlap_fraction(
            lower[..., self.axis], upper[..., self.axis], bounds[self.axis][0], bounds[self.axis][1]
        )
        return np.clip(in_plane * along, 0.0, 1.0)

    def get_material_mapping(
        self,
    ) -> jax.Array:
        all_names = compute_ordered_names(self.materials)
        idx = all_names.index(self.material_name)
        arr = jnp.ones(self.grid_shape, dtype=jnp.int32) * idx
        return arr
