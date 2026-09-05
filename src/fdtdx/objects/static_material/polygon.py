import pathlib

import gdstk
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.path import Path

from fdtdx.core.axis import get_transverse_axes
from fdtdx.core.grid import polygon_to_mask, polygon_to_mask_at_points
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.materials import compute_ordered_names
from fdtdx.objects.static_material.static import (
    StaticMultiMaterialObject,
    nearest_polygon_edge_normal,
    points_in_metric_slab,
)


@autoinit
class ExtrudedPolygon(StaticMultiMaterialObject):
    """A polygon object specified by a list of vertices.

    The vertices must be given in a coordinate system centered at the origin, i.e. (0, 0)
    corresponds to the center of the object's bounding box. The polygon is placed so that
    its center coincides with the center of the grid region allocated to this object.

    The cross-section size is automatically inferred from the vertex bounding box for the
    two axes perpendicular to ``axis``, so ``partial_real_shape`` does not need to be
    specified for those axes.  The extrusion axis size must still be determined by a
    constraint or an explicit ``partial_real_shape`` entry.
    """

    #: Name of the material in the materials dictionary to be used for the object
    material_name: str = frozen_field()

    #: The extrusion axis.
    axis: int = frozen_field()

    #: numpy array of shape (N, 2) with vertices in metrical units (meter), centered at origin.
    vertices: np.ndarray = frozen_field()

    def __post_init__(self):
        w = float(self.vertices[:, 0].max() - self.vertices[:, 0].min())
        h = float(self.vertices[:, 1].max() - self.vertices[:, 1].min())
        real_shape = list(self.partial_real_shape)
        grid_shape = list(self.partial_grid_shape)
        for ax, size in ((self.horizontal_axis, w), (self.vertical_axis, h)):
            if real_shape[ax] is not None and real_shape[ax] != size:
                raise Exception(
                    f"ExtrudedPolygon {self.name}: partial_real_shape for axis {ax} is derived from the "
                    f"vertex bounding box ({size:.3e} m). Do not specify it explicitly."
                )
            if grid_shape[ax] is not None:
                raise Exception(
                    f"ExtrudedPolygon {self.name}: partial_grid_shape for axis {ax} is derived from the "
                    f"vertex bounding box. Do not specify it explicitly."
                )
            real_shape[ax] = size
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
        n_horizontal = self.grid_shape[self.horizontal_axis]
        n_vertical = self.grid_shape[self.vertical_axis]

        # Shift vertices from object-center coords to local grid coords.
        center_h = 0.5 * self.real_shape[self.horizontal_axis]
        center_v = 0.5 * self.real_shape[self.vertical_axis]
        grid_vertices = self.vertices + np.array([center_h, center_v])

        grid = self._config.resolved_grid
        if grid is None:
            spacing = self._config.uniform_spacing()
            half_res = 0.5 * spacing
            max_horizontal = (n_horizontal - 0.5) * spacing
            max_vertical = (n_vertical - 0.5) * spacing

            mask_2d = polygon_to_mask(
                boundary=(half_res, half_res, max_horizontal, max_vertical),
                resolution=spacing,
                polygon_vertices=grid_vertices,
            )
        else:
            h_lower, h_upper = self.grid_slice_tuple[self.horizontal_axis]
            v_lower, v_upper = self.grid_slice_tuple[self.vertical_axis]
            h_edges = np.asarray(grid.edges(self.horizontal_axis))
            v_edges = np.asarray(grid.edges(self.vertical_axis))
            h_centers = 0.5 * (h_edges[h_lower:h_upper] + h_edges[h_lower + 1 : h_upper + 1]) - h_edges[h_lower]
            v_centers = 0.5 * (v_edges[v_lower:v_upper] + v_edges[v_lower + 1 : v_upper + 1]) - v_edges[v_lower]
            mask_2d = polygon_to_mask_at_points(
                x_coords=h_centers,
                y_coords=v_centers,
                polygon_vertices=grid_vertices,
            )
        extrusion_height = self.grid_shape[self.axis]
        mask = jnp.repeat(
            jnp.expand_dims(jnp.asarray(mask_2d, dtype=jnp.bool), axis=self.axis),
            repeats=extrusion_height,
            axis=self.axis,
        )

        return mask

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Continuous point-in-shape test: the polygon cross-section, extruded over the metric extent.

        The vertices are already centred on the object's bounding-box centre, so they are compared
        against the point offsets from :attr:`metric_center`, and the extrusion runs over the
        object's metric extent along ``axis``.

        Args:
            points (np.ndarray): Array of shape ``(..., 3)`` with coordinates in metres.

        Returns:
            np.ndarray: Boolean array of shape ``points.shape[:-1]``.
        """
        pts = np.asarray(points, dtype=float)
        bounds = self.metric_bounds
        center = self.metric_center
        inside = points_in_metric_slab(pts[..., self.axis], bounds[self.axis][0], bounds[self.axis][1])
        if not inside.any():
            return inside
        local_h = pts[..., self.horizontal_axis] - center[self.horizontal_axis]
        local_v = pts[..., self.vertical_axis] - center[self.vertical_axis]
        query = np.column_stack((local_h.ravel(), local_v.ravel()))
        in_plane = Path(np.asarray(self.vertices)).contains_points(query).reshape(local_h.shape)
        return inside & in_plane

    def normal_at(self, points: np.ndarray, ignore_axes: tuple[int, ...] = ()) -> np.ndarray:
        """Outward normal of the nearest surface: a polygon side wall or one of the extrusion caps.

        Args:
            points (np.ndarray): Array of shape ``(..., 3)`` with coordinates in metres.
            ignore_axes (tuple[int, ...]): Axes whose surfaces are not physical interfaces. With the
                extrusion axis listed the caps are dropped and a side wall always wins.

        Returns:
            np.ndarray: Array of shape ``(..., 3)`` with unit normals, zero where undefined.
        """
        pts = np.asarray(points, dtype=float)
        center = self.metric_center
        extent = self.metric_extent
        h_axis, v_axis = self.horizontal_axis, self.vertical_axis
        edge_normal, edge_distance = nearest_polygon_edge_normal(
            [np.asarray(self.vertices, dtype=float)],
            pts[..., h_axis] - center[h_axis],
            pts[..., v_axis] - center[v_axis],
        )
        proj = pts[..., self.axis] - center[self.axis]
        cap_distance = np.abs(np.abs(proj) - 0.5 * extent[self.axis])
        if self.axis in ignore_axes:
            cap_wins = np.zeros(cap_distance.shape, dtype=bool)
        else:
            cap_wins = cap_distance < edge_distance
        sign = np.where(np.sign(proj) == 0.0, 1.0, np.sign(proj))
        normal = np.zeros(pts.shape, dtype=float)
        normal[..., self.axis] = np.where(cap_wins, sign, 0.0)
        normal[..., h_axis] = np.where(cap_wins, 0.0, edge_normal[..., 0])
        normal[..., v_axis] = np.where(cap_wins, 0.0, edge_normal[..., 1])
        return normal

    def box_fill_fraction(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray | None:
        """Exact polygon-rectangle overlap in plane times the extrusion overlap along ``axis``."""
        from fdtdx.core.physics.geometry_smooth import polygons_rectangle_area
        from fdtdx.objects.static_material.static import interval_overlap_fraction

        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        h_axis, v_axis = self.horizontal_axis, self.vertical_axis
        if np.any(upper[..., h_axis] <= lower[..., h_axis]) or np.any(upper[..., v_axis] <= lower[..., v_axis]):
            return None
        center = self.metric_center
        bounds = self.metric_bounds
        area = polygons_rectangle_area(
            [np.asarray(self.vertices, dtype=float)],
            lower[..., h_axis] - center[h_axis],
            upper[..., h_axis] - center[h_axis],
            lower[..., v_axis] - center[v_axis],
            upper[..., v_axis] - center[v_axis],
        )
        rect = (upper[..., h_axis] - lower[..., h_axis]) * (upper[..., v_axis] - lower[..., v_axis])
        along = interval_overlap_fraction(
            lower[..., self.axis], upper[..., self.axis], bounds[self.axis][0], bounds[self.axis][1]
        )
        return np.clip(area / rect * along, 0.0, 1.0)

    def get_material_mapping(
        self,
    ) -> jax.Array:
        all_names = compute_ordered_names(self.materials)
        idx = all_names.index(self.material_name)
        arr = jnp.ones(self.grid_shape, dtype=jnp.int32) * idx
        return arr


def extruded_polygon_from_gds(
    lib: gdstk.Library,
    cell_name: str,
    layer: int,
    datatype: int = 0,
    polygon_index: int = 0,
    **kwargs,
) -> ExtrudedPolygon:
    """Create an ExtrudedPolygon from a polygon in an already-loaded gdstk Library.

    Args:
        lib: An already-loaded gdstk Library.
        cell_name: Name of the GDS cell containing the polygon.
        layer: GDS layer number to read.
        datatype: GDS datatype (default 0).
        polygon_index: Which polygon to use when multiple exist on the layer (default 0).
        **kwargs: Forwarded to ExtrudedPolygon (axis, material_name, materials, …).

    Returns:
        ExtrudedPolygon with vertices centered around the origin in metres.

    Raises:
        ValueError: If the cell or layer/datatype combination is not found.
        IndexError: If polygon_index is out of range.
    """
    cell = next((c for c in lib.cells if isinstance(c, gdstk.Cell) and c.name == cell_name), None)
    if cell is None:
        raise ValueError(f"Cell '{cell_name}' not found in library")

    matching = [p for p in cell.polygons if p.layer == layer and p.datatype == datatype]
    if not matching:
        raise ValueError(f"No polygons on layer={layer}, datatype={datatype} in cell '{cell_name}'")
    if polygon_index >= len(matching):
        raise IndexError(
            f"polygon_index={polygon_index} out of range; found {len(matching)} polygon(s) on layer={layer}"
        )

    polygon = matching[polygon_index]
    vertices_m = np.array(polygon.points) * lib.unit  # library units → metres

    # centre vertices around origin (ExtrudedPolygon convention)
    centre = 0.5 * (vertices_m.min(axis=0) + vertices_m.max(axis=0))
    centred = vertices_m - centre

    return ExtrudedPolygon(vertices=centred, **kwargs)


def extruded_polygon_from_gds_path(
    gds_file: str | pathlib.Path,
    cell_name: str,
    layer: int,
    datatype: int = 0,
    polygon_index: int = 0,
    **kwargs,
) -> ExtrudedPolygon:
    """Create an ExtrudedPolygon from a polygon in a GDS file.

    Args:
        gds_file: Path to the .gds file.
        cell_name: Name of the GDS cell containing the polygon.
        layer: GDS layer number to read.
        datatype: GDS datatype (default 0).
        polygon_index: Which polygon to use when multiple exist on the layer (default 0).
        **kwargs: Forwarded to ExtrudedPolygon (axis, material_name, materials, …).

    Returns:
        ExtrudedPolygon with vertices centered around the origin in metres.

    Raises:
        ValueError: If the cell or layer/datatype combination is not found.
        IndexError: If polygon_index is out of range.
    """
    lib = gdstk.read_gds(str(gds_file))
    return extruded_polygon_from_gds(lib, cell_name, layer, datatype, polygon_index, **kwargs)
