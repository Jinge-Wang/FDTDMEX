"""The coordinate frames between somebody else's mesh and the loader's grid, stated explicitly.

fdtdx places its grid with the origin at the centre of the simulation volume; a thermal or a
mechanical scene is drawn in whatever frame and length unit its author chose, and a
two-dimensional mesh lives in one plane. A transform here says what that difference is, rather than
leaving it to be guessed at the call site.

A transform is applied **twice**, and that is the whole reason this is a stage of its own:

* to the *positions*, before the field is evaluated on the other engine's mesh;
* to the sampled *components*, afterwards, for a vector or a second-rank tensor. A value solved on
  somebody else's mesh carries its components in that mesh's frame, and the same map that moved the
  point has to turn the value. Doing only the first half is a silent error — nothing downstream
  flags a strain whose shear entries sit on the wrong axes — so both maps come from the one
  ``permute`` stored on the object and cannot drift apart.

:class:`PointTransform` is the affine case (offset, scale, collapsed axes, axis permutation), whose
component map is a signed permutation and therefore the same at every point.
:class:`RadialPlaneTransform` is the axisymmetric case, where the radial direction of the mesh turns
with the azimuth, so the component map is a frame *per point*.
:func:`inverse_point_transform` turns an affine transform round for the return direction
(:mod:`fdtdx.coupling.transfer`), where the data flows from the grid back to the mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_square(values: np.ndarray) -> np.ndarray:
    """``(..., m, m)`` from a square array or from its flattened ``(..., m * m)`` form, ``m`` 2 or 3.

    ``FemField.evaluate`` returns a tensor-valued function flattened (``value_size`` components per
    point), so both shapes reach a consumer; both are accepted and neither is ambiguous for the
    sizes in play.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim >= 2 and v.shape[-1] == v.shape[-2] and v.shape[-1] in (2, 3):
        return v
    if v.ndim >= 1 and v.shape[-1] in (4, 9):
        size = 2 if v.shape[-1] == 4 else 3
        return v.reshape((*v.shape[:-1], size, size))
    raise ValueError(f"a rank-2 value needs shape (..., m, m) or (..., m * m) with m 2 or 3, got {v.shape}")


@dataclass(frozen=True)
class PointTransform:
    """Map loader (Yee) coordinates onto the FEM mesh's coordinate frame.

    The transform is applied to the Yee points before evaluation; the samples keep the Yee
    lattice's own indexing.

    Attributes:
        offset (tuple[float, float, float]): Added to the Yee coordinates, in metres.
        scale (float): Multiplies the Yee coordinates before the offset (a scene drawn in
            micrometres gets ``1e6``).
        collapse_axes (tuple[int, ...]): Axes whose coordinate is replaced by ``collapse_value``
            before the offset, so a 3-D Yee grid with one cell along ``z`` samples a 2-D mesh in
            the ``z = collapse_value`` plane.
        collapse_value (float): The coordinate written on the collapsed axes.
        permute (tuple[int, int, int] | None): Reorder the axes after the steps above, so mesh
            coordinate ``i`` is taken from Yee axis ``permute[i]``. A cross-section mesh drawn in
            ``(x, z)`` sampled by a grid whose propagation axis is ``y`` uses ``(0, 2, 1)``.
    """

    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0
    collapse_axes: tuple[int, ...] = ()
    collapse_value: float = 0.0
    permute: tuple[int, int, int] | None = None

    def apply(self, points: np.ndarray) -> np.ndarray:
        out = np.array(points, dtype=np.float64, copy=True) * float(self.scale)
        for axis in self.collapse_axes:
            out[:, axis] = float(self.collapse_value)
        out += np.asarray(self.offset, dtype=np.float64)[None, :]
        if self.permute is not None:
            if sorted(self.permute) != [0, 1, 2]:
                raise ValueError(f"permute must reorder (0, 1, 2), got {self.permute}")
            out = out[:, list(self.permute)]
        return out

    def component_matrix(self) -> np.ndarray:
        """The ``(3, 3)`` signed permutation taking mesh-frame components to Yee-frame components.

        :meth:`apply` maps a Yee point onto the mesh frame; its linear part is ``scale`` times a
        permutation, so mesh axis ``i`` *is* Yee axis ``permute[i]``. The component map is the same
        permutation read the other way: a value's component along mesh axis ``i`` belongs on Yee
        axis ``permute[i]``. The magnitude of ``scale`` is a change of length unit and does not
        touch a component; its sign does, because a negative scale mirrors all three axes.

        Returns:
            np.ndarray: ``R`` with ``R[permute[i], i] = sign(scale)``, so ``v_yee = R @ v_mesh``.
        """
        perm = (0, 1, 2) if self.permute is None else tuple(int(a) for a in self.permute)
        if sorted(perm) != [0, 1, 2]:
            raise ValueError(f"permute must reorder (0, 1, 2), got {self.permute}")
        sign = -1.0 if float(self.scale) < 0.0 else 1.0
        out = np.zeros((3, 3), dtype=np.float64)
        for mesh_axis, yee_axis in enumerate(perm):
            out[yee_axis, mesh_axis] = sign
        return out

    def apply_values(
        self,
        values: np.ndarray,
        rank: int = 1,
        points: np.ndarray | None = None,
        pseudo: bool = False,
    ) -> np.ndarray:
        """Express sampled vector or tensor values in the Yee frame, the way the positions were.

        :meth:`apply` moves the *positions*; a scalar sample needs nothing more, but a vector or a
        tensor sampled on the mesh carries its components in the mesh frame and must be turned by
        the same permutation, or the coupling is silently wrong. Both maps come from the one
        ``permute`` stored here, so they cannot drift apart.

        A mesh of geometric dimension 2 hands over 2 components (or a ``(2, 2)`` tensor). They are
        lifted to 3: mesh axis ``i`` lands on Yee axis ``permute[i]`` and the remaining Yee axis,
        ``permute[2]`` — the collapsed one for a cross-section mesh — gets zero. A field with an
        out-of-plane value (generalized plane strain) must therefore arrive as a 3-component or
        ``(3, 3)`` mesh-frame value, which is what
        :meth:`fdtdx.coupling.fem.FemField.symmetric_gradient_of` produces when given
        ``out_of_plane``.

        A pure scale and offset (``permute=None``, positive ``scale``) leave every component alone.

        Args:
            values (np.ndarray): ``(..., m)`` for ``rank=1``, ``(..., m, m)`` or its flattened
                ``(..., m * m)`` form for ``rank=2``, with ``m`` 2 or 3; anything for ``rank=0``.
            rank (int): 0 (scalar), 1 (vector) or 2 (second-rank tensor).
            points (np.ndarray | None): Unused; accepted so a caller can treat this and
                :class:`RadialPlaneTransform` alike.
            pseudo (bool): Multiply a rank-1 value by ``det(R)``, for an axial vector (a magnetic
                field, a curl). A rank-2 request with ``pseudo`` is refused rather than guessed.

        Returns:
            np.ndarray: ``(..., 3)`` for ``rank=1``, ``(..., 3, 3)`` for ``rank=2``, the input for
            ``rank=0``.
        """
        del points
        v = np.asarray(values, dtype=np.float64)
        if rank == 0:
            if pseudo:
                raise ValueError("a rank-0 value has no orientation; pseudo=True is meaningless")
            return v
        matrix = self.component_matrix()
        if rank == 1:
            size = v.shape[-1] if v.ndim else 0
            if size not in (2, 3):
                raise ValueError(f"a rank-1 value needs 2 or 3 components on its last axis, got shape {v.shape}")
            out = v @ matrix[:, :size].T
            if pseudo:
                out = out * float(np.linalg.det(matrix))
            return out
        if rank != 2:
            raise ValueError(f"rank must be 0, 1 or 2, got {rank}")
        if pseudo:
            raise NotImplementedError("pseudo=True is defined here for rank 1 only")
        t = _as_square(v)
        size = t.shape[-1]
        r = matrix[:, :size]
        return np.einsum("ia,...ab,jb->...ij", r, t, r)

    def as_dict(self) -> dict[str, Any]:
        return {
            "offset": [float(v) for v in self.offset],
            "scale": float(self.scale),
            "collapse_axes": [int(a) for a in self.collapse_axes],
            "collapse_value": float(self.collapse_value),
            "permute": None if self.permute is None else [int(a) for a in self.permute],
        }


@dataclass(frozen=True)
class RadialPlaneTransform:
    """Map Yee points of a top-view (x, y) grid onto an axisymmetric ``(r, z)`` thermal mesh.

    A ring and a concentric ring heater are axisymmetric, so their thermal problem is solved once
    in the ``(r, z)`` half-plane (thermalFEM's ``axisymmetric_scalar`` physics, with ``x = r`` and
    ``y = z`` in the mesh frame). A top-view electromagnetic grid at one height then reads the
    field at ``(r, z_plane)`` with ``r`` the distance of the Yee point from the ring axis. Objects
    that break the symmetry (a straight bus, a contact pad) are sampled at their radius; the
    error that makes is the author's to state.

    Attributes:
        center (tuple[float, float]): The ring axis ``(x, y)`` in Yee coordinates, metres.
        z_plane (float): The mesh ``z`` (its second coordinate) the grid sits at, in mesh units.
        scale (float): Multiplies the radius before it is written (a mesh drawn in micrometres
            gets ``1e6``).
    """

    center: tuple[float, float] = (0.0, 0.0)
    z_plane: float = 0.0
    scale: float = 1.0

    def apply(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        out = np.zeros((p.shape[0], 3), dtype=np.float64)
        out[:, 0] = np.hypot(p[:, 0] - self.center[0], p[:, 1] - self.center[1]) * float(self.scale)
        out[:, 1] = float(self.z_plane)
        return out

    def component_matrix(self, points: np.ndarray) -> np.ndarray:
        """``(N, 3, 3)`` frames taking mesh ``(r, z, azimuth)`` components to Yee ``(x, y, z)``.

        Unlike :meth:`PointTransform.component_matrix` this depends on the point: the radial
        direction of an axisymmetric mesh turns with the azimuth. Columns are the images of the
        mesh axes in the order :meth:`apply` writes them — radial, axial, azimuthal — so
        ``R[:, 0] = (cos t, sin t, 0)``, ``R[:, 1] = (0, 0, 1)``, ``R[:, 2] = (-sin t, cos t, 0)``.
        On the axis the azimuth is undefined and ``t = 0`` is used.

        Args:
            points (np.ndarray): ``(N, 3)`` Yee coordinates, metres — the same points
                :meth:`apply` was given.

        Returns:
            np.ndarray: ``(N, 3, 3)`` orthonormal, right-handed frames.
        """
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        dx = p[:, 0] - float(self.center[0])
        dy = p[:, 1] - float(self.center[1])
        rho = np.hypot(dx, dy)
        on_axis = rho == 0.0
        safe = np.where(on_axis, 1.0, rho)
        cos_t = np.where(on_axis, 1.0, dx / safe)
        sin_t = np.where(on_axis, 0.0, dy / safe)
        out = np.zeros((p.shape[0], 3, 3), dtype=np.float64)
        out[:, 0, 0] = cos_t
        out[:, 1, 0] = sin_t
        out[:, 2, 1] = 1.0
        out[:, 0, 2] = -sin_t
        out[:, 1, 2] = cos_t
        return out

    def apply_values(
        self,
        values: np.ndarray,
        rank: int = 1,
        points: np.ndarray | None = None,
        pseudo: bool = False,
    ) -> np.ndarray:
        """Express axisymmetric mesh-frame components in the Yee frame at the sampled points.

        A scalar needs nothing (``rank=0``). A vector solved in the ``(r, z)`` half-plane has its
        radial component along the local outward radius, which is a different Yee direction at
        every point, so the points must be given. Two components are lifted to three with the
        azimuthal one zero, which is the right reading of an axisymmetric solve with no swirl.

        Args:
            values (np.ndarray): ``(N, m)`` for ``rank=1``, ``(N, m, m)`` or ``(N, m * m)`` for
                ``rank=2``, with ``m`` 2 (radial, axial) or 3 (radial, axial, azimuthal).
            rank (int): 0, 1 or 2.
            points (np.ndarray): ``(N, 3)`` Yee coordinates in metres; required for rank 1 and 2.
            pseudo (bool): Not supported here.

        Returns:
            np.ndarray: ``(N, 3)`` or ``(N, 3, 3)`` in the Yee frame.
        """
        v = np.asarray(values, dtype=np.float64)
        if rank == 0:
            return v
        if pseudo:
            raise NotImplementedError("pseudo=True is not supported for a radial transform")
        if points is None:
            raise ValueError("a radial transform needs the Yee points to orient a vector or tensor value")
        frames = self.component_matrix(points)
        if rank == 1:
            size = v.shape[-1]
            if size not in (2, 3):
                raise ValueError(f"a rank-1 value needs 2 or 3 components on its last axis, got shape {v.shape}")
            return np.einsum("nia,na->ni", frames[:, :, :size], v.reshape(-1, size))
        if rank != 2:
            raise ValueError(f"rank must be 0, 1 or 2, got {rank}")
        t = _as_square(v)
        size = t.shape[-1]
        r = frames[:, :, :size]
        return np.einsum("nia,nab,njb->nij", r, t.reshape(-1, size, size), r)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "radial_plane",
            "center": [float(v) for v in self.center],
            "z_plane": float(self.z_plane),
            "scale": float(self.scale),
        }


def inverse_point_transform(transform: PointTransform) -> PointTransform:
    """Turn a Yee-to-mesh :class:`PointTransform` round into a mesh-to-Yee one.

    :class:`PointTransform` applies ``scale``, then the collapse, then ``offset``, then ``permute``.
    A pure scale, offset and permutation inverts exactly; a collapse throws a coordinate away and
    cannot be inverted, so it is refused rather than guessed at.

    Args:
        transform (PointTransform): The transform that maps Yee points onto the mesh frame.

    Returns:
        PointTransform: The transform that maps mesh points back onto the Yee frame.

    Raises:
        ValueError: If the transform collapses an axis or its scale is zero.
    """
    if transform.collapse_axes:
        raise ValueError(
            "a transform that collapses an axis is not invertible; build the mesh-to-grid transform explicitly instead"
        )
    scale = float(transform.scale)
    if scale == 0.0:
        raise ValueError("a transform with scale 0 is not invertible")
    offset = np.asarray(transform.offset, dtype=np.float64)
    permute = transform.permute
    if permute is not None:
        order = np.argsort(np.asarray(permute))
        inverse_permute: tuple[int, int, int] | None = (int(order[0]), int(order[1]), int(order[2]))
        offset = offset[list(permute)]
    else:
        inverse_permute = None
    shifted = -offset / scale
    return PointTransform(
        offset=(float(shifted[0]), float(shifted[1]), float(shifted[2])),
        scale=1.0 / scale,
        permute=inverse_permute,
    )
