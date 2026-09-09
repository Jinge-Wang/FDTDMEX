"""Point evaluation of a finite-element scalar field on the Yee lattices, with coverage flags.

The seam between a DOLFINx/FEniCSx solver (Kronos' ``thermalFEM``) and the material loader is a
function evaluated at points. Nothing here writes or reads a mesh file: the FEM side hands over its
``dolfinx.fem.Function`` (or the function space plus the degree-of-freedom vector it solved for),
and the operator answers, for any set of points, the value at each point and whether the point lies
inside the mesh at all. A point outside the mesh is reported as *not covered* and its value is
``NaN``, never a silent zero: the consumer decides what an uncovered point means (an error, or "no
perturbation there"), and it can count them.

The evaluation is vectorised over points: one bounding-box tree per mesh, one collision query, one
cell-location query and one basis evaluation per call. No Python loop over points.

The output for a Yee grid, :class:`YeeLatticeSamples`, is also the artefact that crosses a process
boundary when the two solvers do not share an interpreter: it is the field sampled where the loader
needs it, one array plus one coverage mask per lattice, with the grid edges it was sampled on, and
it round-trips through a NumPy ``.npz`` file.

DOLFINx is imported lazily: the module can be imported without it, and a :class:`YeeLatticeSamples`
file can be loaded and consumed on a machine that never had DOLFINx.
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

#: Maximum number of points sent to one collision query, to bound peak memory.
_CHUNK_POINTS = 2_000_000


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


@dataclass(frozen=True)
class PointTransform:
    """Map loader (Yee) coordinates onto the FEM mesh's coordinate frame.

    fdtdx places its grid with the origin at the centre of the simulation volume; a thermal scene
    is drawn in whatever frame its author chose, and a 2-D thermal mesh lives in one plane. Both
    differences are stated here explicitly rather than guessed. The transform is applied to the
    Yee points before evaluation; the samples keep the Yee lattice's own indexing.

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

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "radial_plane",
            "center": [float(v) for v in self.center],
            "z_plane": float(self.z_plane),
            "scale": float(self.scale),
        }


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


class FemField:
    """A DOLFINx function (scalar or vector valued) with vectorised point evaluation and coverage.

    Args:
        function: A ``dolfinx.fem.Function`` on a Lagrange or discontinuous Lagrange space, scalar
            or blocked vector valued, any degree.
        name (str): Field name, carried into the sampled artefact.
        unit (str): Unit label, carried into the sampled artefact.
    """

    def __init__(self, function: Any, name: str = "f", unit: str = ""):
        self.function = function
        self.name = str(name)
        self.unit = str(unit)
        self.mesh = function.function_space.mesh
        shape = tuple(int(v) for v in getattr(function.function_space, "value_shape", ()))
        self.value_size = int(np.prod(shape)) if shape else 1
        self._trees: dict[float, Any] = {}

    @classmethod
    def gradient_of(cls, scalar: "FemField", scale: float = 1.0, name: str = "grad", unit: str = "") -> "FemField":
        """``scale * grad(f)`` of a scalar field as a discontinuous vector field of one degree less.

        Exact for the finite-element function: the gradient of a degree-``p`` Lagrange function is a
        degree-``p - 1`` polynomial per cell, discontinuous across cells, and a discontinuous
        Lagrange vector space of that degree holds it without projection error. With
        ``scale=-1`` this is the electrostatic field ``E = -grad(V)`` of a solved potential.

        Args:
            scalar (FemField): The scalar field.
            scale (float): Multiplier.
            name (str): Name of the new field.
            unit (str): Unit label of the new field.

        Returns:
            FemField: The vector field on the same mesh.
        """
        import ufl
        from dolfinx import fem

        V = scalar.function.function_space
        degree = int(V.element.basix_element.degree)
        gdim = int(scalar.mesh.geometry.dim)
        W = fem.functionspace(scalar.mesh, ("DG", max(degree - 1, 0), (gdim,)))
        out = fem.Function(W, name=name)
        expr = fem.Expression(float(scale) * ufl.grad(scalar.function), W.element.interpolation_points)
        out.interpolate(expr)
        return cls(out, name=name, unit=unit)

    # -- construction -----------------------------------------------------------------------

    @classmethod
    def from_dofs(cls, function_space: Any, dofs: np.ndarray, name: str = "T", unit: str = "K") -> FemField:
        """Wrap a degree-of-freedom vector on a given function space.

        This is the thermalFEM seam: ``thSim`` keeps the space as ``sim._V`` and the solved vector
        as ``sim.T_dofs`` (also ``sim.get_solution()["T_dofs"]``); neither is written to disk.

        Args:
            function_space: A ``dolfinx.fem.FunctionSpace`` (scalar Lagrange).
            dofs (np.ndarray): The local degree-of-freedom vector, length ``V.dofmap.index_map.size_local * bs``
                plus ghosts, exactly as ``Function.x.array`` stores it. A shorter vector (owned
                degrees of freedom only) is accepted and the ghost entries are left at zero, then
                scattered forward.
            name (str): Field name.
            unit (str): Unit label.

        Returns:
            FemScalarField: The wrapped field.

        Raises:
            ValueError: If the vector is longer than the space's array or contains non-finite values.
        """
        from dolfinx import fem

        function = fem.Function(function_space, name=name)
        target = function.x.array
        source = np.real(np.asarray(dofs, dtype=np.float64)).reshape(-1)
        if source.shape[0] > target.shape[0]:
            raise ValueError(
                f"dof vector has {source.shape[0]} entries, the function space array holds {target.shape[0]}"
            )
        if not np.all(np.isfinite(source)):
            raise ValueError("dof vector contains non-finite values")
        target[: source.shape[0]] = source
        function.x.scatter_forward()
        return cls(function, name=name, unit=unit)

    @classmethod
    def from_thermal_sim(cls, sim: Any, dofs: np.ndarray | None = None, unit: str = "K") -> FemField:
        """Wrap the temperature a Kronos ``thermalFEM.thSim`` has solved for, in process.

        Reads the private ``sim._V`` (the Lagrange space ``thAssembly.make_function_space`` built)
        and ``sim.T_dofs``; nothing in thermalFEM is modified and nothing is written to disk.
        ``dofs`` overrides the vector, for a transient output row (``result["T_out"][i]``).

        Args:
            sim: A ``thSim`` after ``solve()``.
            dofs (np.ndarray | None): Optional vector to use instead of ``sim.T_dofs``.
            unit (str): Unit label.

        Returns:
            FemScalarField: The wrapped temperature field.

        Raises:
            RuntimeError: If the simulator has no DOLFINx function space (no backend or no mesh) or
                has not been solved.
        """
        space = getattr(sim, "_V", None)
        if space is None:
            raise RuntimeError("thermalFEM simulator has no DOLFINx function space (no mesh built, or no backend)")
        vector = sim.T_dofs if dofs is None else dofs
        status = getattr(sim, "status", None)
        # thermalFEM reports "converged" after a solve, "failed" after a solver error (with a
        # zero-filled vector) and "not_run" before any solve; the last two carry no field.
        if dofs is None and status in ("failed", "not_run"):
            raise RuntimeError(f"thermalFEM simulator has not solved (status={status!r})")
        return cls.from_dofs(space, np.asarray(vector), name="T", unit=unit)

    # -- geometry ---------------------------------------------------------------------------

    @property
    def tdim(self) -> int:
        return int(self.mesh.topology.dim)

    @property
    def gdim(self) -> int:
        return int(self.mesh.geometry.dim)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounds of the mesh geometry, ``(lower, upper)`` each of shape ``(3,)``."""
        x = np.asarray(self.mesh.geometry.x, dtype=np.float64)
        lower = np.zeros(3)
        upper = np.zeros(3)
        lower[: x.shape[1]] = x.min(axis=0)
        upper[: x.shape[1]] = x.max(axis=0)
        return lower, upper

    def _tree(self, padding: float) -> Any:
        from dolfinx import geometry

        key = float(padding)
        if key not in self._trees:
            self._trees[key] = geometry.bb_tree(self.mesh, self.tdim, padding=key)
        return self._trees[key]

    # -- evaluation -------------------------------------------------------------------------

    def evaluate(self, points: np.ndarray, padding: float = 0.0) -> PointSamples:
        """Evaluate the field at ``points``, flagging the ones outside the mesh.

        Args:
            points (np.ndarray): ``(N, 3)`` coordinates in the mesh's frame. ``(N, 2)`` is accepted
                for a planar mesh and padded with a zero third coordinate.
            padding (float): Bounding-box padding passed to the tree, in metres. A point on the
                mesh boundary can miss every box by round-off; a padding of a small fraction of a
                cell keeps it. The exact point-in-cell test still decides coverage.

        Returns:
            PointSamples: Values, coverage flags and containing cells.
        """
        from dolfinx import geometry

        P = np.asarray(points, dtype=np.float64)
        if P.ndim != 2 or P.shape[1] not in (2, 3):
            raise ValueError(f"points must have shape (N, 3) or (N, 2), got {P.shape}")
        if P.shape[1] == 2:
            P = np.concatenate([P, np.zeros((P.shape[0], 1))], axis=1)
        P = np.ascontiguousarray(P)
        n = P.shape[0]
        values = np.full((n,) if self.value_size == 1 else (n, self.value_size), np.nan, dtype=np.float64)
        covered = np.zeros(n, dtype=bool)
        cells = np.full(n, -1, dtype=np.int64)
        if n == 0:
            return PointSamples(values, covered, cells)

        tree = self._tree(padding)
        for start in range(0, n, _CHUNK_POINTS):
            stop = min(start + _CHUNK_POINTS, n)
            chunk = P[start:stop]
            candidates = geometry.compute_collisions_points(tree, chunk)
            colliding = geometry.compute_colliding_cells(self.mesh, candidates, chunk)
            offsets = np.asarray(colliding.offsets, dtype=np.int64)
            links = np.asarray(colliding.array, dtype=np.int64)
            counts = np.diff(offsets)
            hit = counts > 0
            covered[start:stop] = hit
            first = np.full(chunk.shape[0], -1, dtype=np.int64)
            first[hit] = links[offsets[:-1][hit]]
            cells[start:stop] = first
            if hit.any():
                evaluated = np.asarray(self.function.eval(chunk[hit], first[hit].astype(np.int32)))
                if np.iscomplexobj(evaluated):
                    evaluated = np.real(evaluated)
                if self.value_size == 1:
                    values[start:stop][hit] = evaluated.reshape(-1)
                else:
                    values[start:stop][hit] = evaluated.reshape(-1, self.value_size)
        return PointSamples(values, covered, cells)


class FemScalarField(FemField):
    """A scalar DOLFINx function; see :class:`FemField`. Kept as the name the thermal path uses."""

    def __init__(self, function: Any, name: str = "T", unit: str = "K"):
        super().__init__(function, name=name, unit=unit)
        if self.value_size != 1:
            raise ValueError(f"FemScalarField needs a scalar function, this one has {self.value_size} components")


@dataclass
class YeeLatticeSamples:
    """A scalar field sampled on Yee lattices: one value array and one coverage mask per lattice.

    This is what the material loader consumes and what crosses a process boundary. Every array has
    the lattice's own shape ``(Nx, Ny, Nz)``; ``values`` is ``NaN`` where ``covered`` is false.

    Attributes:
        edges (tuple[np.ndarray, np.ndarray, np.ndarray]): The grid's cell edges per axis, so a
            consumer can verify it is looking at the grid it built.
        values (dict[str, np.ndarray]): Lattice name to ``(Nx, Ny, Nz)`` float64 values, or
            ``(Nx, Ny, Nz, n)`` for an ``n``-component field.
        covered (dict[str, np.ndarray]): Lattice name to ``(Nx, Ny, Nz)`` bool coverage.
        name (str): Field name (``"T"``).
        unit (str): Unit label (``"K"``).
        transform (dict[str, Any]): The :class:`PointTransform` that mapped Yee points onto the
            mesh frame, as a dictionary.
        provenance (dict[str, Any]): Free-form record of where the field came from (solver, mesh
            size, element order, commit), written by the producer.
    """

    edges: tuple[np.ndarray, np.ndarray, np.ndarray]
    values: dict[str, np.ndarray]
    covered: dict[str, np.ndarray]
    name: str = "T"
    unit: str = "K"
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


def sample_on_yee_lattices(
    fem_field: FemField,
    grid: Any,
    lattices: Sequence[str] = ("E0", "E1", "E2", "V"),
    transform: PointTransform | RadialPlaneTransform | None = None,
    padding: float = 0.0,
    provenance: dict[str, Any] | None = None,
) -> YeeLatticeSamples:
    """Evaluate a FEM scalar field at the points of the requested Yee lattices.

    All requested lattices are concatenated into one point set and evaluated together, then split
    back into per-lattice arrays. The permittivity lives on the three E lattices and, under the
    vertex placement, its off-diagonal entries on the vertex lattice, which is the default request;
    a permeability perturbation would add the H lattices.

    Args:
        fem_field (FemField): The field to sample (scalar or vector valued).
        grid: The resolved ``RectilinearGrid`` or a triple of edge arrays, in metres.
        lattices (Sequence[str]): Names from :data:`LATTICE_NAMES`.
        transform (PointTransform | RadialPlaneTransform | None): Yee-to-mesh coordinate map;
            identity when ``None``.
        padding (float): Bounding-box padding for the collision query, in mesh units.
        provenance (dict | None): Recorded in the result.

    Returns:
        YeeLatticeSamples: Values and coverage per lattice.
    """
    edges = grid_edges(grid)
    transform = transform or PointTransform()
    blocks: list[tuple[str, tuple[int, int, int]]] = []
    points: list[np.ndarray] = []
    for lattice in lattices:
        pts, shape = lattice_points(edges, lattice)
        blocks.append((lattice, shape))
        points.append(pts)
    stacked = transform.apply(np.concatenate(points, axis=0))
    samples = fem_field.evaluate(stacked, padding=padding)
    values: dict[str, np.ndarray] = {}
    covered: dict[str, np.ndarray] = {}
    start = 0
    for lattice, shape in blocks:
        count = int(np.prod(shape))
        block = samples.values[start : start + count]
        values[lattice] = block.reshape(shape) if block.ndim == 1 else block.reshape((*shape, block.shape[1]))
        covered[lattice] = samples.covered[start : start + count].reshape(shape)
        start += count
    return YeeLatticeSamples(
        edges=edges,
        values=values,
        covered=covered,
        name=fem_field.name,
        unit=fem_field.unit,
        transform=transform.as_dict(),
        provenance=dict(provenance or {}),
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
