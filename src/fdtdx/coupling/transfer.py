"""Grid arrays back to a FEM source, and the exact transpose of that map.

The return direction of the coupling: a Cartesian array the electromagnetic side computed — an
absorbed-power density, a body force, a charge density — handed to a Kronos finite-element solver,
which takes a volumetric source as a function of position. thermalFEM's
``thSim.set_source("heat_source_field", th.heat_source_field(fn))`` calls ``fn(coords)`` with
``coords`` of shape ``(3, N)`` and expects ``(N,)`` values, which the assembler interpolates into
the source space and adds as ``inner(q, v) dx``. So the seam is one multilinear interpolation.

The interpolation is written as explicit weights rather than hidden inside a
``scipy.interpolate.RegularGridInterpolator`` closure, for one reason: an inverse-design chain needs
the *transpose* of this map (a scatter-add of a cotangent from the mesh points back onto the design
grid), and a transpose is only exact if it uses the same weights the forward map used. So the
weights are the primitive (:func:`multilinear_weights`), the transfer object
(:class:`MultilinearTransfer`) holds them and exposes ``forward`` and ``transpose``, and the two
convenience functions :func:`cartesian_to_fem_source` and :func:`fem_to_cartesian_design` are thin
wrappers over it.

Conventions, all explicit because both sides of the seam have their own:

* **Where the values sit.** ``edges`` is either the grid's cell edges (length ``n + 1`` on an axis)
  or the sample points themselves (length ``n``). With cell edges, ``lattice`` says which lattice
  the values live on: ``"cell"`` (the cell centres, the default and the right one for a per-cell
  density such as absorbed power) or any of
  :data:`fdtdx.coupling.lattice.LATTICE_NAMES` for a component that sits on a Yee lattice.
* **Frames.** ``transform.apply(points)`` maps the FEM mesh's coordinates onto the Cartesian
  grid's frame — the *opposite* direction of the transform
  :func:`fdtdx.coupling.fem.sample_on_yee_lattices` takes, because the data flows the other way. A
  transform written for that function is turned round with
  :func:`fdtdx.coupling.frames.inverse_point_transform`.
* **Thin axes.** An axis with a single sample point (a quasi-1-D or 2-D scene run on a 3-D grid
  with one cell) is constant: weight one, and a point's coordinate on that axis never puts it
  outside the grid.
* **Outside the grid.** ``outside="zero"`` (the default) gives a point beyond the sample box the
  value zero, which is what a source term means there; ``"error"`` raises; ``"nearest"`` clamps to
  the boundary sample. Nothing is ever extrapolated linearly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from fdtdx.coupling.frames import PointTransform
from fdtdx.coupling.lattice import sample_axes

#: Accepted out-of-grid policies.
OUTSIDE_POLICIES: tuple[str, str, str] = ("zero", "error", "nearest")

#: Tolerance, relative to an axis' own extent, for calling a point "on the boundary" rather than out.
_EDGE_TOL = 1e-9


def _axes_for(
    edges: Sequence[np.ndarray],
    lattice: str,
    shape: tuple[int, ...] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample axes, accepting either cell edges or the coordinates themselves, per axis."""
    given = [np.asarray(e, dtype=np.float64).ravel() for e in edges]
    if shape is not None and all(g.size == n for g, n in zip(given, shape)):
        # the coordinates were handed over directly
        for axis, g in enumerate(given):
            if g.size > 1 and not np.all(np.diff(g) > 0.0):
                raise ValueError(f"edges[{axis}] must be strictly increasing")
        return given[0], given[1], given[2]
    axes = sample_axes(given, lattice)
    if shape is not None and tuple(a.size for a in axes) != tuple(shape):
        raise ValueError(
            f"the values are {tuple(shape)} but the {lattice!r} lattice of these edges is "
            f"{tuple(int(a.size) for a in axes)}"
        )
    return axes


def multilinear_weights(
    points: np.ndarray,
    edges: Sequence[np.ndarray],
    lattice: str = "cell",
    outside: str = "zero",
    shape: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The eight corner indices and weights of the multilinear interpolation at each point.

    This is the primitive the whole module is built on: the forward map is
    ``out[p] = sum_c weights[p, c] * values.ravel()[indices[p, c]]`` and the transpose is the
    scatter-add of the same weights.

    Args:
        points (np.ndarray): ``(N, 3)`` coordinates, already in the Cartesian grid's frame.
        edges (Sequence[np.ndarray]): Cell edges (length ``n + 1``) or sample coordinates
            (length ``n``) per axis.
        lattice (str): Which lattice the values sit on when ``edges`` are cell edges; see
            :func:`sample_axes`.
        outside (str): One of :data:`OUTSIDE_POLICIES`.
        shape (tuple | None): The value array's shape, when it is known; it disambiguates cell
            edges from sample coordinates and is checked against the lattice.

    Returns:
        tuple: ``(indices, weights)``, both ``(N, 8)``. ``indices`` are flat indices into an array
        of the sample shape in C order; ``weights`` rows sum to one for a point inside the grid and
        to zero for a point outside it under ``outside="zero"``.

    Raises:
        ValueError: If ``points`` is not ``(N, 3)``, if ``outside`` is not a known policy, or if a
            point lies outside the grid under ``outside="error"``.
    """
    if outside not in OUTSIDE_POLICIES:
        raise ValueError(f"outside must be one of {OUTSIDE_POLICIES}, got {outside!r}")
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    axes = _axes_for(edges, lattice, shape)
    sizes = tuple(int(a.size) for a in axes)

    lower = np.empty((pts.shape[0], 3), dtype=np.int64)
    frac = np.empty((pts.shape[0], 3), dtype=np.float64)
    inside = np.ones(pts.shape[0], dtype=bool)
    for axis in range(3):
        a = axes[axis]
        coordinate = pts[:, axis]
        if a.size == 1:
            # a thin axis is constant: every point sits on it
            lower[:, axis] = 0
            frac[:, axis] = 0.0
            continue
        span = float(a[-1] - a[0])
        tol = _EDGE_TOL * span
        inside &= (coordinate >= a[0] - tol) & (coordinate <= a[-1] + tol)
        index = np.clip(np.searchsorted(a, coordinate, side="right") - 1, 0, a.size - 2)
        width = a[index + 1] - a[index]
        t = (coordinate - a[index]) / width
        lower[:, axis] = index
        frac[:, axis] = np.clip(t, 0.0, 1.0)

    if not inside.all():
        if outside == "error":
            bad = int(np.count_nonzero(~inside))
            first = pts[~inside][0]
            raise ValueError(
                f"{bad} of {pts.shape[0]} points lie outside the Cartesian grid, the first at "
                f"{tuple(float(v) for v in first)}; pass outside='zero' or 'nearest' to allow it"
            )
        # "nearest" is the clamp that already happened above; "zero" zeroes the weights below

    strides = np.array([sizes[1] * sizes[2], sizes[2], 1], dtype=np.int64)
    indices = np.empty((pts.shape[0], 8), dtype=np.int64)
    weights = np.empty((pts.shape[0], 8), dtype=np.float64)
    for corner in range(8):
        step = ((corner >> 2) & 1, (corner >> 1) & 1, corner & 1)
        flat = np.zeros(pts.shape[0], dtype=np.int64)
        w = np.ones(pts.shape[0], dtype=np.float64)
        for axis in range(3):
            if sizes[axis] == 1:
                if step[axis] == 1:
                    w = np.zeros_like(w)  # the duplicate corner of a thin axis carries no weight
                continue
            flat += (lower[:, axis] + step[axis]) * strides[axis]
            w = w * (frac[:, axis] if step[axis] else 1.0 - frac[:, axis])
        indices[:, corner] = flat
        weights[:, corner] = w
    if outside == "zero":
        weights[~inside] = 0.0
    return indices, weights


@dataclass(frozen=True)
class MultilinearTransfer:
    """The multilinear map between a Cartesian grid and a fixed set of points, and its transpose.

    It keeps the sparse weights rather than only the values they produced, so ``forward`` and
    ``transpose`` are provably adjoint.

    Attributes:
        indices (np.ndarray): ``(N, 8)`` flat indices into an array of shape :attr:`shape`.
        weights (np.ndarray): ``(N, 8)`` interpolation weights.
        shape (tuple[int, int, int]): The Cartesian sample shape the indices refer to.
    """

    indices: np.ndarray
    weights: np.ndarray
    shape: tuple[int, int, int]

    @classmethod
    def build(
        cls,
        points: np.ndarray,
        edges: Sequence[np.ndarray],
        lattice: str = "cell",
        outside: str = "zero",
        transform: PointTransform | Any | None = None,
        shape: tuple[int, int, int] | None = None,
    ) -> MultilinearTransfer:
        """Weights for ``points`` (``(N, 3)``, in the FEM frame unless ``transform`` is ``None``)."""
        pts = np.asarray(points, dtype=np.float64)
        if transform is not None:
            pts = transform.apply(pts)
        axes = _axes_for(edges, lattice, shape)
        resolved = (int(axes[0].size), int(axes[1].size), int(axes[2].size))
        indices, weights = multilinear_weights(pts, axes, lattice=lattice, outside=outside, shape=resolved)
        return cls(indices=indices, weights=weights, shape=resolved)

    @property
    def num_points(self) -> int:
        return int(self.indices.shape[0])

    def forward(self, values: np.ndarray) -> np.ndarray:
        """Interpolate a Cartesian array at the points: ``(Nx, Ny, Nz) -> (N,)``."""
        array = np.asarray(values)
        if tuple(array.shape) != self.shape:
            raise ValueError(f"values must have shape {self.shape}, got {tuple(array.shape)}")
        flat = array.reshape(-1)
        return np.einsum("pc,pc->p", self.weights.astype(flat.dtype, copy=False), flat[self.indices])

    def transpose(self, cotangent: np.ndarray) -> np.ndarray:
        """Scatter a per-point vector back onto the Cartesian grid: ``(N,) -> (Nx, Ny, Nz)``.

        The exact adjoint of :meth:`forward`: for any ``g`` on the grid and any ``y`` at the points,
        ``y . forward(g) == transpose(y) . g``.
        """
        y = np.asarray(cotangent)
        if y.shape != (self.num_points,):
            raise ValueError(f"cotangent must have shape {(self.num_points,)}, got {y.shape}")
        out = np.zeros(int(np.prod(self.shape)), dtype=np.result_type(y.dtype, np.float64))
        np.add.at(out, self.indices.reshape(-1), (self.weights * y[:, None]).reshape(-1))
        return out.reshape(self.shape)


def cartesian_to_fem_source(
    values: np.ndarray,
    edges: Sequence[np.ndarray],
    transform: PointTransform | Any | None = None,
    outside: str = "zero",
    lattice: str = "cell",
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a Cartesian array as the ``(3, N) -> (N,)`` source callable a Kronos FEM solver takes.

    Args:
        values (np.ndarray): ``(Nx, Ny, Nz)`` on the Cartesian grid, in the FEM solver's own source
            unit (thermalFEM wants W/m^3, or W/um^3 in a micrometre scene).
        edges (Sequence[np.ndarray]): Cell edges or sample coordinates per axis; see
            :func:`sample_axes`.
        transform: Maps the FEM mesh's coordinates onto the Cartesian grid's frame. ``None`` is the
            identity. Note the direction: this is the inverse of what
            :func:`fdtdx.coupling.fem.sample_on_yee_lattices` takes, and
            :func:`fdtdx.coupling.frames.inverse_point_transform` converts one into the other.
        outside (str): One of :data:`OUTSIDE_POLICIES`; ``"zero"`` by default, so a mesh that
            reaches beyond the electromagnetic grid simply has no source out there.
        lattice (str): Which lattice ``values`` sits on; ``"cell"`` (cell centres) by default.

    Returns:
        Callable: ``fn(coords)`` taking ``(3, N)`` and returning ``(N,)`` float64. The weights are
        rebuilt on each call, so the same callable serves a mesh that is remeshed between outer
        iterations.

    Raises:
        ValueError: If ``values`` is not three-dimensional or does not match the grid.
    """
    if outside not in OUTSIDE_POLICIES:
        raise ValueError(f"outside must be one of {OUTSIDE_POLICIES}, got {outside!r}")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"values must be a 3-D Cartesian array, got shape {array.shape}")
    shape = (int(array.shape[0]), int(array.shape[1]), int(array.shape[2]))
    axes = _axes_for(edges, lattice, shape)

    def source(coords: np.ndarray) -> np.ndarray:
        pts = np.asarray(coords, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] != 3:
            raise ValueError(f"the FEM source callable takes (3, N) coordinates, got {pts.shape}")
        transfer = MultilinearTransfer.build(
            pts.T, axes, lattice=lattice, outside=outside, transform=transform, shape=shape
        )
        return transfer.forward(array)

    return source


def fem_to_cartesian_design(
    values: np.ndarray,
    points: np.ndarray,
    edges: Sequence[np.ndarray],
    transform: PointTransform | Any | None = None,
    outside: str = "zero",
    lattice: str = "cell",
    shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """The transpose of :func:`cartesian_to_fem_source`: point values scattered onto the grid.

    The contract an adjoint chain needs: the grid-to-FEM interpolation must keep its weights so
    the reverse pass can apply exactly ``W^T``. Given a vector defined at the mesh points — a
    cotangent of an objective with respect to the source values there, or any per-point quantity
    that is to be accumulated — this returns the design-grid array ``W^T y``.

    It is an accumulation, not an average: a grid cell that several mesh points lean on collects
    the sum of their weighted contributions. Dividing by ``fem_to_cartesian_design(ones, ...)``
    turns it into the weighted average, which is a different (and non-adjoint) operation; do that
    in the caller if that is what is wanted, so the adjoint stays available.

    Args:
        values (np.ndarray): ``(N,)`` at the mesh points.
        points (np.ndarray): ``(N, 3)`` or ``(3, N)`` mesh-point coordinates, the same points and
            the same order the forward map was applied to.
        edges (Sequence[np.ndarray]): Cell edges or sample coordinates per axis.
        transform: Mesh-to-grid frame map, as in :func:`cartesian_to_fem_source`.
        outside (str): One of :data:`OUTSIDE_POLICIES`; must match the forward map's policy for the
            two to be adjoint.
        lattice (str): Which lattice the grid array sits on.
        shape (tuple | None): The grid shape, when ``edges`` are ambiguous.

    Returns:
        np.ndarray: ``(Nx, Ny, Nz)``.

    Raises:
        ValueError: If the shapes disagree.
    """
    y = np.asarray(values, dtype=np.float64).ravel()
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError(f"points must be (N, 3) or (3, N), got shape {pts.shape}")
    if pts.shape[0] == 3 and pts.shape[1] != 3:
        pts = pts.T
    if pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3) or (3, N), got shape {np.asarray(points).shape}")
    if pts.shape[0] != y.size:
        raise ValueError(f"{y.size} values but {pts.shape[0]} points")
    transfer = MultilinearTransfer.build(pts, edges, lattice=lattice, outside=outside, transform=transform, shape=shape)
    return transfer.transpose(y)
