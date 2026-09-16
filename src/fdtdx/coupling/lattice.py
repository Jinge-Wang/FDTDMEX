"""Where a coupled field is read: the Yee lattices, and the container a sample set travels in.

Every stage of the coupling meets the loader at a point of a Yee lattice, so the lattice definition
and the sample container are the bottom of the package and depend on nothing else in it.

Two things live here:

* the lattice geometry. :func:`lattice_axes` writes the same rule as
  :func:`fdtdx.core.physics.geometry_raster.yee_lattice_coordinates` on plain edge arrays, so a
  sampler needs the three edge arrays and not a resolved grid object. The three E component
  lattices carry the diagonal permittivity, the three H lattices a permeability perturbation, and
  the cell-vertex lattice ``V`` the off-diagonal Kottke entries under the node placement.
* the containers. :class:`PointSamples` is values plus a coverage flag at arbitrary points;
  :class:`YeeLatticeSamples` is the same on a whole grid, one value array and one coverage mask per
  lattice, and it is also the artefact that crosses a process boundary when the two solvers do not
  share an interpreter: it round-trips through a NumPy ``.npz`` file and carries the grid edges it
  was taken on, so a file taken on a different discretisation is refused rather than reshaped.

A point with no value is ``NaN`` and flagged *not covered*, never a silent zero: the consumer
decides what an uncovered point means (an error, or "no perturbation there") and can count them.

Nothing here imports DOLFINx or JAX, so a sample file can be loaded and consumed on a machine that
never had either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from fdtdx.core.physics.geometry_raster import E_OFFSETS, H_OFFSETS, V_OFFSETS

#: The lattices a consumer can ask for: the three E component lattices, the three H component
#: lattices and the cell-vertex lattice the off-diagonal Kottke entries live on.
LATTICE_NAMES: tuple[str, ...] = ("E0", "E1", "E2", "H0", "H1", "H2", "V")


def _lattice_offsets(lattice: str) -> tuple[float, float, float]:
    if lattice not in LATTICE_NAMES:
        raise ValueError(f"lattice must be one of {LATTICE_NAMES}, got {lattice!r}")
    if lattice == "V":
        return V_OFFSETS[0]
    table = E_OFFSETS if lattice[0] == "E" else H_OFFSETS
    return table[int(lattice[1])]


def lattice_axes(
    edges: Sequence[np.ndarray],
    lattice: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three 1-D coordinate arrays of one Yee lattice, from the grid's cell edges.

    The same rule as :func:`fdtdx.core.physics.geometry_raster.yee_lattice_coordinates`, written
    on plain edge arrays so a sampler needs only the three edge arrays, not a resolved grid object.

    Args:
        edges (Sequence[np.ndarray]): The cell-edge coordinates per axis, each of length ``N + 1``.
        lattice (str): One of :data:`LATTICE_NAMES`.

    Returns:
        tuple: ``(x, y, z)`` arrays of lengths ``(Nx, Ny, Nz)``.
    """
    offsets = _lattice_offsets(lattice)
    axes = []
    for axis in range(3):
        e = np.asarray(edges[axis], dtype=np.float64)
        axes.append(e[:-1] if offsets[axis] == 0.0 else 0.5 * (e[:-1] + e[1:]))
    return axes[0], axes[1], axes[2]


def lattice_points(edges: Sequence[np.ndarray], lattice: str) -> tuple[np.ndarray, tuple[int, int, int]]:
    """All points of one lattice as an ``(N, 3)`` array in ``ij`` order, plus the lattice shape."""
    x, y, z = lattice_axes(edges, lattice)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1), (x.size, y.size, z.size)


def lattice_coordinates(
    edges: Sequence[np.ndarray],
    lattice: str,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """The points of an exported lattice, in the exported values' own ``ij`` order, and its shape.

    The name :mod:`fdtdx.coupling.export`'s consumers reach for: an external solver that is handed
    the assembled material never has to reconstruct the half-cell offsets by hand, and the export
    and the field sampler are provably reading the one lattice definition. Identical to
    :func:`lattice_points`.
    """
    return lattice_points(edges, lattice)


def sample_axes(
    edges: Sequence[np.ndarray],
    lattice: str = "cell",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The 1-D coordinate array per axis that a Cartesian value array is sampled on.

    The lattice question asked the other way round, for a value array leaving the grid
    (:mod:`fdtdx.coupling.transfer`): a per-cell density such as an absorbed power sits at the cell
    centres, a Yee component sits on its own lattice.

    Args:
        edges (Sequence[np.ndarray]): Three arrays. On each axis, length ``n + 1`` is read as the
            grid's cell edges and length ``n`` as the sample coordinates themselves.
        lattice (str): ``"cell"`` for the cell centres, or a name from :data:`LATTICE_NAMES` for a
            Yee lattice. Ignored on an axis whose coordinates were given directly.

    Returns:
        tuple: The three coordinate arrays, strictly increasing.

    Raises:
        ValueError: If ``edges`` is not three arrays, if an axis is empty or not increasing.
    """
    if len(edges) != 3:
        raise ValueError(f"edges must be three arrays, got {len(edges)}")
    given = [np.asarray(e, dtype=np.float64).ravel() for e in edges]
    for axis, e in enumerate(given):
        if e.size == 0:
            raise ValueError(f"edges[{axis}] is empty")
        if e.size > 1 and not np.all(np.diff(e) > 0.0):
            raise ValueError(f"edges[{axis}] must be strictly increasing")
    if lattice == "cell":
        centres = [e if e.size == 1 else 0.5 * (e[:-1] + e[1:]) for e in given]
        return centres[0], centres[1], centres[2]
    return lattice_axes(given, lattice)


def grid_edges(grid: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three edge arrays of a resolved ``RectilinearGrid`` (or a triple of edge arrays)."""
    if hasattr(grid, "edges"):
        source = [grid.edges(axis) for axis in range(3)]
    else:
        source = list(grid)
    if len(source) != 3:
        raise ValueError("grid must be a RectilinearGrid or three edge arrays")
    return (
        np.asarray(source[0], dtype=np.float64),
        np.asarray(source[1], dtype=np.float64),
        np.asarray(source[2], dtype=np.float64),
    )


@dataclass
class PointSamples:
    """Values of a field at arbitrary points, with a coverage flag per point.

    Attributes:
        values (np.ndarray): ``(N,)`` float64 for a scalar field, ``(N, n)`` for a field with ``n``
            components; ``NaN`` wherever ``covered`` is false.
        covered (np.ndarray): ``(N,)`` bool; true when the point lies inside a mesh cell.
        cells (np.ndarray): ``(N,)`` int64 index of the cell that contains the point, ``-1``
            where not covered. When several cells contain a point (it sits on a shared facet) the
            first one DOLFINx lists is used; a continuous Lagrange field gives the same value in
            all of them.
    """

    values: np.ndarray
    covered: np.ndarray
    cells: np.ndarray

    @property
    def num_points(self) -> int:
        return int(self.values.shape[0])

    @property
    def num_uncovered(self) -> int:
        return int(np.count_nonzero(~self.covered))


@dataclass
class YeeLatticeSamples:
    """A field sampled on Yee lattices: one value array and one coverage mask per lattice.

    This is what the material loader consumes and what crosses a process boundary. Every array has
    the lattice's own shape ``(Nx, Ny, Nz)``; ``values`` is ``NaN`` where ``covered`` is false.

    Attributes:
        edges (tuple[np.ndarray, np.ndarray, np.ndarray]): The grid's cell edges per axis, so a
            consumer can verify it is looking at the grid it built.
        values (dict[str, np.ndarray]): Lattice name to ``(Nx, Ny, Nz)`` float64 values, or
            ``(Nx, Ny, Nz, n)`` for an ``n``-component field.
        covered (dict[str, np.ndarray]): Lattice name to ``(Nx, Ny, Nz)`` bool coverage.
        name (str): Field name (``"T"``).
        unit (str | None): Unit label (``"K"``), carried through :meth:`save` and :meth:`load`.
            ``None`` states that the quantity is dimensionless — a strain, an occupancy — which is
            a different claim from ``""`` ("nobody said"). A consumer checks it at the response
            boundary, so a field solved in volts per micrometre cannot be fed to a response that
            expects volts per metre.
        transform (dict[str, Any]): The coordinate transform that mapped Yee points onto the mesh
            frame (:mod:`fdtdx.coupling.frames`), as a dictionary.
        provenance (dict[str, Any]): Free-form record of where the field came from (solver, mesh
            size, element order, commit), written by the producer.
    """

    edges: tuple[np.ndarray, np.ndarray, np.ndarray]
    values: dict[str, np.ndarray]
    covered: dict[str, np.ndarray]
    name: str = "T"
    unit: str | None = "K"
    transform: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def lattices(self) -> tuple[str, ...]:
        return tuple(self.values.keys())

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.edges[0].size) - 1, int(self.edges[1].size) - 1, int(self.edges[2].size) - 1)

    def coverage_report(self) -> dict[str, Any]:
        """Per lattice: point count, uncovered count, and the value range over covered points."""
        report: dict[str, Any] = {}
        for lattice in self.lattices:
            mask = self.covered[lattice]
            vals = self.values[lattice][mask]
            if vals.ndim > 1:
                vals = np.linalg.norm(vals, axis=-1)
            report[lattice] = {
                "num_points": int(mask.size),
                "num_uncovered": int(np.count_nonzero(~mask)),
                "min": float(vals.min()) if vals.size else None,
                "max": float(vals.max()) if vals.size else None,
            }
        return report

    def matches_edges(self, edges: Sequence[np.ndarray], rtol: float = 1e-9, atol: float = 1e-15) -> bool:
        """Whether these samples were taken on the given grid edges."""
        if len(edges) != 3:
            return False
        for mine, theirs in zip(self.edges, edges):
            theirs = np.asarray(theirs, dtype=np.float64)
            if mine.shape != theirs.shape or not np.allclose(mine, theirs, rtol=rtol, atol=atol):
                return False
        return True

    def save(self, path: str | Path) -> Path:
        """Write the samples to a NumPy ``.npz`` file (the cross-process artefact)."""
        import json

        path = Path(path)
        payload: dict[str, Any] = {
            "edges_x": self.edges[0],
            "edges_y": self.edges[1],
            "edges_z": self.edges[2],
            "meta": np.array(
                json.dumps(
                    {
                        "name": self.name,
                        "unit": self.unit,
                        "lattices": list(self.lattices),
                        "transform": self.transform,
                        "provenance": self.provenance,
                        "format": "fdtdx.coupling.YeeLatticeSamples.v1",
                    }
                )
            ),
        }
        for lattice in self.lattices:
            payload[f"values_{lattice}"] = self.values[lattice]
            payload[f"covered_{lattice}"] = self.covered[lattice]
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> YeeLatticeSamples:
        """Read a file written by :meth:`save`."""
        import json

        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            edges = (data["edges_x"], data["edges_y"], data["edges_z"])
            values = {name: data[f"values_{name}"] for name in meta["lattices"]}
            covered = {name: data[f"covered_{name}"] for name in meta["lattices"]}
        return cls(
            edges=edges,
            values=values,
            covered=covered,
            name=meta["name"],
            unit=meta["unit"],
            transform=meta.get("transform", {}),
            provenance=meta.get("provenance", {}),
        )


def uniform_samples(
    grid: Any,
    value: float,
    lattices: Sequence[str] = ("E0", "E1", "E2", "V"),
    name: str = "T",
    unit: str = "K",
) -> YeeLatticeSamples:
    """A constant field on the given lattices, fully covered: the control for a coupling test."""
    edges = grid_edges(grid)
    values: dict[str, np.ndarray] = {}
    covered: dict[str, np.ndarray] = {}
    for lattice in lattices:
        _, shape = lattice_points(edges, lattice)
        values[lattice] = np.full(shape, float(value), dtype=np.float64)
        covered[lattice] = np.ones(shape, dtype=bool)
    return YeeLatticeSamples(edges=edges, values=values, covered=covered, name=name, unit=unit)


def samples_from_callable(
    grid: Any,
    fn: Any,
    lattices: Sequence[str] = ("E0", "E1", "E2", "V"),
    name: str = "T",
    unit: str = "K",
) -> YeeLatticeSamples:
    """Sample an analytic function ``fn(points (N, 3)) -> (N,)`` on the lattices, fully covered."""
    edges = grid_edges(grid)
    values: dict[str, np.ndarray] = {}
    covered: dict[str, np.ndarray] = {}
    for lattice in lattices:
        pts, shape = lattice_points(edges, lattice)
        out = np.asarray(fn(pts), dtype=np.float64)
        values[lattice] = out.reshape(shape) if out.ndim == 1 else out.reshape((*shape, out.shape[-1]))
        covered[lattice] = np.ones(shape, dtype=bool)
    return YeeLatticeSamples(edges=edges, values=values, covered=covered, name=name, unit=unit)
