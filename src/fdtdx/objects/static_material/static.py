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
