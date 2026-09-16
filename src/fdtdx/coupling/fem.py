"""A finite-element field coming in: point evaluation on the Yee lattices, with coverage flags.

The seam between a DOLFINx/FEniCSx solver and the material loader is a function evaluated at
points. Nothing here writes or reads a mesh file, and nothing here knows any engine: the FEM side
hands over a ``dolfinx.fem.Function`` (or the function space plus the degree-of-freedom vector it
solved for), and :class:`FemField` answers, for any set of points, the value at each point and
whether the point lies inside the mesh at all. Which attribute of which Kronos simulator holds that
space and that vector is the one thing :mod:`fdtdx.coupling.kronos` knows and this module does not,
so this file survives a change of FEM engine unchanged.

A point outside the mesh is reported as *not covered* and its value is ``NaN``, never a silent
zero: the consumer decides what an uncovered point means (an error, or "no perturbation there"),
and it can count them. The evaluation is vectorised over points: one bounding-box tree per mesh,
one collision query, one cell-location query and one basis evaluation per call. No Python loop over
points.

Two derived fields are exact rather than projected, because the gradient of a degree-``p`` Lagrange
function is a degree-``p - 1`` polynomial per cell and the discontinuous space of that degree holds
it with no projection error: :meth:`FemField.gradient_of` (with ``scale=-1``, the electrostatic
field of a solved potential) and :meth:`FemField.symmetric_gradient_of` (the strain of a
displacement, the rate of strain of a velocity).

**Where the seam is fragile, and how to see it.** A field that is exact per cell is *discontinuous*
at every material interface of the source mesh, and a lattice point that lands on one of those
interfaces takes whichever cell the evaluator listed first. On a three-layer capacitor whose layer
boundary sat on a grid edge, the two lattices on that edge reported the oxide field (0.480 V/um)
and the lattice half a cell away reported the lithium-niobate field (0.067 V/um): a factor of
seven, from a tie-break, with nothing in the run saying so. :func:`facet_coincidence_report` says
so. It walks the loader's own smoothing record — which holds, per blended pixel, the fill fraction,
the interface normal and the two materials — and lists the sample points the material facet passes
**through**.

*Why the fill fraction is the exact test.* Each blended pixel is a box centred on its component's
sample point, and the fill fraction is the part of that box on the front material's side of a
planar interface. A box is centrally symmetric, so a plane cuts it into two equal halves if and
only if it passes through its centre: ``fill == 0.5`` means the facet passes through the sample
point itself, whatever the normal's direction. Away from one half the fill moves monotonically, and
for an axis-aligned normal it moves linearly, so ``(0.5 - fill)`` times the pixel width along that
axis is the signed distance from the sample point to the facet.

The report is geometry, not a verdict: a point on a facet is only a problem when the field being
sampled is discontinuous there. When samples are supplied the report carries the value each flagged
point took and the values of its two neighbours across the facet, so the size of the jump the
tie-break chose between is visible in the same table.

DOLFINx is imported lazily: the module can be imported without it, and a
:class:`~fdtdx.coupling.lattice.YeeLatticeSamples` file can be loaded and consumed on a machine
that never had DOLFINx.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from fdtdx.core.physics.geometry_smooth import SmoothingRecord
from fdtdx.coupling.frames import PointTransform, RadialPlaneTransform
from fdtdx.coupling.lattice import (
    PointSamples,
    YeeLatticeSamples,
    _lattice_offsets,
    grid_edges,
    lattice_axes,
    lattice_points,
)

#: Maximum number of points sent to one collision query, to bound peak memory.
_CHUNK_POINTS = 2_000_000

#: Default facet-coincidence tolerance, as a fraction of the smallest cell: a point closer to the
#: facet than a millionth of a cell is on it, and the slack absorbs the rasteriser's own round-off.
DEFAULT_TOL_CELLS = 1e-6


class FemField:
    """A DOLFINx function (scalar or vector valued) with vectorised point evaluation and coverage.

    Args:
        function: A ``dolfinx.fem.Function`` on a Lagrange or discontinuous Lagrange space, scalar
            or blocked vector valued, any degree.
        name (str): Field name, carried into the sampled artefact.
        unit (str | None): Unit label, carried into the sampled artefact. ``None`` states that the
            quantity is dimensionless (a strain), which is not the same claim as ``""``.
    """

    def __init__(self, function: Any, name: str = "f", unit: str | None = ""):
        self.function = function
        self.name = str(name)
        self.unit = None if unit is None else str(unit)
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

    @classmethod
    def symmetric_gradient_of(
        cls,
        vector: "FemField",
        scale: float = 1.0,
        out_of_plane: float | None = None,
        name: str = "S",
        unit: str | None = None,
    ) -> "FemField":
        """``scale * sym(grad(u))`` of a vector field as a discontinuous tensor field of one degree less.

        The strain of a displacement, the rate of strain of a velocity, and any other symmetric
        gradient of a primary finite-element unknown. Exact for the finite-element function, for the
        same reason :meth:`gradient_of` is: the gradient of a degree-``p`` Lagrange function is a
        degree-``p - 1`` polynomial per cell, and the discontinuous space of that degree holds it
        with no projection error. ``FemField.gradient_of`` cannot be used on a vector field — it
        builds a ``(gdim,)`` space, and the gradient of a vector needs ``(gdim, gdim)``.

        On a two-dimensional mesh the symmetric gradient has no out-of-plane entry, because the
        out-of-plane derivative is not a degree of freedom of the solve. Which value belongs there
        is a modelling choice the caller has to state:

        * ``out_of_plane=None`` returns the in-plane ``(2, 2)`` tensor and leaves the choice to the
          consumer (:meth:`fdtdx.coupling.frames.PointTransform.apply_values` then writes zero on
          the collapsed axis, which is plane strain);
        * ``out_of_plane=0.0`` returns a ``(3, 3)`` tensor with an explicit zero — plane strain,
          said out loud;
        * ``out_of_plane=eps0`` returns a ``(3, 3)`` tensor with that uniform value — generalized
          plane strain, where the out-of-plane strain is a constant fixed by an out-of-plane force
          balance rather than by the in-plane solve.

        Args:
            vector (FemField): A vector-valued field, ``gdim`` components.
            scale (float): Multiplier.
            out_of_plane (float | None): The ``[2, 2]`` entry on a 2-D mesh; see above. Refused on
                a 3-D mesh, where the entry comes from the solve.
            name (str): Name of the new field.
            unit (str | None): Unit label; ``None`` (dimensionless) is right for a strain.

        Returns:
            FemField: The tensor field on the same mesh, ``(gdim, gdim)`` or ``(3, 3)``.

        Raises:
            ValueError: If the input is not vector valued with one component per geometric
                dimension, or if ``out_of_plane`` is given for a 3-D mesh.
        """
        import ufl
        from dolfinx import fem

        gdim = int(vector.mesh.geometry.dim)
        if vector.value_size != gdim:
            raise ValueError(
                f"symmetric_gradient_of needs a vector field with {gdim} components on this mesh, "
                f"got {vector.value_size}"
            )
        V = vector.function.function_space
        degree = int(V.element.basix_element.degree)
        strain = float(scale) * ufl.sym(ufl.grad(vector.function))
        if out_of_plane is None:
            shape = (gdim, gdim)
            expression = strain
        else:
            if gdim != 2:
                raise ValueError(
                    "out_of_plane is a two-dimensional modelling choice; on a 3-D mesh the "
                    "out-of-plane entry comes from the solve"
                )
            shape = (3, 3)
            zero = ufl.constantvalue.zero()
            expression = ufl.as_matrix(
                [
                    [strain[0, 0], strain[0, 1], zero],
                    [strain[1, 0], strain[1, 1], zero],
                    [zero, zero, ufl.constantvalue.as_ufl(float(out_of_plane))],
                ]
            )
        W = fem.functionspace(vector.mesh, ("DG", max(degree - 1, 0), shape))
        out = fem.Function(W, name=name)
        out.interpolate(fem.Expression(expression, W.element.interpolation_points))
        return cls(out, name=name, unit=unit)

    # -- construction -----------------------------------------------------------------------

    @classmethod
    def from_dofs(cls, function_space: Any, dofs: np.ndarray, name: str = "T", unit: str = "K") -> FemField:
        """Wrap a degree-of-freedom vector on a given function space.

        The DOLFINx-level constructor every engine adapter in :mod:`fdtdx.coupling.kronos` ends in:
        a function space plus the vector a solve produced, neither of which is written to disk.

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
        """The temperature a Kronos thermal simulator solved for.

        Kept as an alias so the recorded cases run unchanged; the adapter itself, and the knowledge
        of which attributes a thermal simulator exposes, live in
        :func:`fdtdx.coupling.kronos.thermal_temperature`, which new code calls instead. The result
        is rewrapped in the class the alias was reached through, so
        ``FemScalarField.from_thermal_sim`` still returns a :class:`FemScalarField` and still runs
        its scalar check.
        """
        from fdtdx.coupling.kronos import thermal_temperature

        field = thermal_temperature(sim, dofs=dofs, unit=unit)
        if type(field) is cls:
            return field
        return cls(field.function, name=field.name, unit=field.unit)

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


def sample_on_yee_lattices(
    fem_field: FemField,
    grid: Any,
    lattices: Sequence[str] = ("E0", "E1", "E2", "V"),
    transform: PointTransform | RadialPlaneTransform | None = None,
    padding: float = 0.0,
    provenance: dict[str, Any] | None = None,
) -> YeeLatticeSamples:
    """Evaluate a FEM field at the points of the requested Yee lattices.

    All requested lattices are concatenated into one point set and evaluated together, then split
    back into per-lattice arrays. The permittivity lives on the three E lattices and, under the
    vertex placement, its off-diagonal entries on the vertex lattice, which is the default request;
    a permeability perturbation would add the H lattices.

    Args:
        fem_field (FemField): The field to sample (scalar, vector or tensor valued).
        grid: The resolved ``RectilinearGrid`` or a triple of edge arrays, in metres.
        lattices (Sequence[str]): Names from :data:`fdtdx.coupling.lattice.LATTICE_NAMES`.
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


# ---------------------------------------------------------------------------------------------
# The seam diagnostic: sample points a material facet passes through
# ---------------------------------------------------------------------------------------------
@dataclass
class FacetCoincidence:
    """One sample point the loader's own geometry puts on a material facet.

    Attributes:
        lattice (str): ``"E0"``..``"E2"`` or ``"V"``.
        index (tuple[int, int, int]): The point's index on that lattice.
        point (tuple[float, float, float]): Its coordinates in the grid frame, in metres.
        distance (float): Distance from the point to the facet plane, in metres. Zero when the
            facet passes exactly through the point.
        fill (float): The pixel's fill fraction of the front material; ``0.5`` is coincidence.
        normal (tuple[float, float, float]): The facet's unit normal.
        axis (int): The axis the normal leans on most, i.e. the direction the jump is across.
        material_front (str): The material on the side the loader called front.
        material_back (str): The material behind the facet.
        material_taken (str | None): The material the loader's own point sample took at this point
            (``front_E``); ``None`` on the vertex lattice, which has no per-point material.
        value (float | None): The sampled value at the point (its norm for a vector or tensor
            field), when samples were supplied.
        neighbour_values (tuple[float | None, float | None]): The same, one lattice step each way
            along ``axis``.
        neighbour_jump (float | None): ``|high - low|`` of those two: the field's change across the
            facet, measured two lattice steps apart rather than at the facet itself.
        relative_jump (float | None): ``neighbour_jump`` over the larger of the two magnitudes.
    """

    lattice: str
    index: tuple[int, int, int]
    point: tuple[float, float, float]
    distance: float
    fill: float
    normal: tuple[float, float, float]
    axis: int
    material_front: str
    material_back: str
    material_taken: str | None = None
    value: float | None = None
    neighbour_values: tuple[float | None, float | None] = (None, None)
    neighbour_jump: float | None = None
    relative_jump: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "lattice": self.lattice,
            "index": list(self.index),
            "point": list(self.point),
            "distance": float(self.distance),
            "fill": float(self.fill),
            "normal": list(self.normal),
            "axis": int(self.axis),
            "material_front": self.material_front,
            "material_back": self.material_back,
            "material_taken": self.material_taken,
            "value": self.value,
            "neighbour_values": list(self.neighbour_values),
            "neighbour_jump": self.neighbour_jump,
            "relative_jump": self.relative_jump,
        }


@dataclass
class FacetCoincidenceReport:
    """What :func:`facet_coincidence_report` found, per lattice and as a listing.

    Attributes:
        tol (float): The coincidence tolerance used, in metres.
        num_examined (dict[str, int]): Blended pixels the loader recorded on each lattice.
        num_coincident (dict[str, int]): How many of them put the facet on the sample point.
        max_relative_jump (dict[str, float | None]): The largest relative jump seen among the
            coincident points of each lattice, when samples were supplied.
        points (list[FacetCoincidence]): The worst ``max_listed`` coincident points, largest
            relative jump first (and, with no samples, smallest distance first).
    """

    tol: float
    num_examined: dict[str, int] = field(default_factory=dict)
    num_coincident: dict[str, int] = field(default_factory=dict)
    max_relative_jump: dict[str, float | None] = field(default_factory=dict)
    points: list[FacetCoincidence] = field(default_factory=list)

    @property
    def total_coincident(self) -> int:
        return int(sum(self.num_coincident.values()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "tol": float(self.tol),
            "num_examined": dict(self.num_examined),
            "num_coincident": dict(self.num_coincident),
            "total_coincident": self.total_coincident,
            "max_relative_jump": dict(self.max_relative_jump),
            "points": [p.as_dict() for p in self.points],
        }


def _pixel_widths(edges: Sequence[np.ndarray], lattice: str) -> list[np.ndarray]:
    """The smoothing box's extent per axis, per index, for one lattice.

    The primal cell on the axes the component sits at the centre of, the dual cell on the axes it
    sits on an edge of -- the boxes :mod:`fdtdx.core.physics.geometry_smooth` builds. At index 0 of
    a dual axis the loader clips the box against the domain edge on a terminated axis, so the
    distance reported for a facet in that first half cell is the un-clipped one.
    """
    offsets = _lattice_offsets(lattice)
    widths: list[np.ndarray] = []
    for axis in range(3):
        w = np.diff(np.asarray(edges[axis], dtype=np.float64))
        if offsets[axis] != 0.0:
            widths.append(w)
        else:
            dual = np.empty_like(w)
            dual[0] = w[0]
            if w.size > 1:
                dual[1:] = 0.5 * (w[:-1] + w[1:])
            widths.append(dual)
    return widths


def _magnitudes(values: np.ndarray) -> np.ndarray:
    """``(Nx, Ny, Nz)`` magnitudes of a scalar, vector or Voigt-vector sample array."""
    if values.ndim == 3:
        return np.abs(values)
    return np.linalg.norm(values.reshape(*values.shape[:3], -1), axis=-1)


def facet_coincidence_report(
    samples: YeeLatticeSamples | Mapping[str, YeeLatticeSamples],
    material_map: Mapping[str, Any],
    tol: float | None = None,
    lattices: Sequence[str] | None = None,
    max_listed: int = 32,
) -> FacetCoincidenceReport:
    """List the sample points a material facet passes through, and what they took.

    Args:
        samples (YeeLatticeSamples | Mapping[str, YeeLatticeSamples]): The samples whose points
            are being checked; they also carry the grid the points live on. A mapping (the form the
            perturbation takes) is accepted and every field in it must be on the same grid.
        material_map (Mapping[str, Any]): ``info["yee_material_map"]`` from ``place_objects``. Its
            smoothing record is what locates the facets, so the scene must have been placed with
            ``material_sampling="yee_smooth"``.
        tol (float | None): Coincidence tolerance in metres. ``None`` uses
            :data:`DEFAULT_TOL_CELLS` times the smallest cell.
        lattices (Sequence[str] | None): Which lattices to check; every recorded one when ``None``.
        max_listed (int): How many points to keep in :attr:`FacetCoincidenceReport.points`.

    Returns:
        FacetCoincidenceReport: Counts per lattice and the worst points.

    Raises:
        ValueError: If the scene carries no smoothing record, or the samples were taken on a
            different grid from the one the scene was placed on.
    """
    record: SmoothingRecord | None = material_map.get("smoothing_record")
    if record is None:
        raise ValueError(
            "the facet-coincidence diagnostic needs the loader's smoothing record, which only "
            "material_sampling='yee_smooth' writes; this scene carries none"
        )
    names = tuple(material_map.get("material_names", ()))
    front_E = np.asarray(material_map["front_E"])

    per_field: dict[str, YeeLatticeSamples] = (
        {samples.name: samples} if isinstance(samples, YeeLatticeSamples) else dict(samples)
    )
    if not per_field:
        raise ValueError("no samples given: the diagnostic reads the grid from them")
    first = next(iter(per_field.values()))
    edges = tuple(np.asarray(e, dtype=np.float64) for e in first.edges)
    for name, s in per_field.items():
        if not s.matches_edges(edges):
            raise ValueError(f"samples {name!r} were taken on a different grid from the others")

    smallest = min(float(np.min(np.diff(e))) for e in edges)
    tolerance = float(DEFAULT_TOL_CELLS * smallest if tol is None else tol)
    report = FacetCoincidenceReport(tol=tolerance)

    def _name(index: int) -> str:
        return names[index] if 0 <= index < len(names) else f"material[{index}]"

    found: list[FacetCoincidence] = []
    for entry in record.passes:
        if entry.field == "E":
            lattice = f"E{entry.component}"
        elif entry.field == "V":
            lattice = "V"
        else:
            continue
        if lattices is not None and lattice not in lattices:
            continue
        report.num_examined[lattice] = report.num_examined.get(lattice, 0) + entry.num_pixels
        if entry.num_pixels == 0:
            continue
        cells = np.asarray(entry.cells)
        fill = np.asarray(entry.fill, dtype=np.float64)
        normal = np.asarray(entry.normal, dtype=np.float64)
        widths = _pixel_widths(edges, lattice)
        # The facet leans on the axis where |n_a| * w_a is largest; that axis carries the linear
        # part of the fill-to-distance map, so it sets both the distance and the direction the
        # jump is across.
        spans = np.stack([np.abs(normal[:, a]) * widths[a][cells[:, a]] for a in range(3)], axis=1)
        axis = np.argmax(spans, axis=1)
        span = spans[np.arange(spans.shape[0]), axis]
        distance = np.abs(0.5 - fill) * span
        hit = distance <= tolerance
        report.num_coincident[lattice] = report.num_coincident.get(lattice, 0) + int(np.count_nonzero(hit))
        if not hit.any():
            continue

        axes = lattice_axes(edges, lattice)
        shape = tuple(a.size for a in axes)
        magnitudes = {name: _magnitudes(s.values[lattice]) for name, s in per_field.items() if lattice in s.values}
        worst_relative: float | None = report.max_relative_jump.get(lattice)
        for row in np.nonzero(hit)[0]:
            i, j, k = (int(cells[row, 0]), int(cells[row, 1]), int(cells[row, 2]))
            a = int(axis[row])
            idx = (i, j, k)
            low_idx = list(idx)
            high_idx = list(idx)
            low_idx[a] = idx[a] - 1
            high_idx[a] = idx[a] + 1
            value = neighbour_low = neighbour_high = None
            jump = relative = None
            # One field is enough to show the jump; the others share the geometry that caused it.
            mag = next(iter(magnitudes.values()), None)
            if mag is not None:
                value = float(mag[idx])
                if low_idx[a] >= 0 and high_idx[a] < shape[a]:
                    neighbour_low = float(mag[tuple(low_idx)])
                    neighbour_high = float(mag[tuple(high_idx)])
                    jump = abs(neighbour_high - neighbour_low)
                    scale = max(abs(neighbour_high), abs(neighbour_low))
                    relative = jump / scale if scale > 0.0 else None
            if relative is not None:
                worst_relative = relative if worst_relative is None else max(worst_relative, relative)
            found.append(
                FacetCoincidence(
                    lattice=lattice,
                    index=idx,
                    point=(float(axes[0][i]), float(axes[1][j]), float(axes[2][k])),
                    distance=float(distance[row]),
                    fill=float(fill[row]),
                    normal=(float(normal[row, 0]), float(normal[row, 1]), float(normal[row, 2])),
                    axis=a,
                    material_front=_name(int(entry.material_hi[row])),
                    material_back=_name(int(entry.material_lo[row])),
                    material_taken=(None if lattice == "V" else _name(int(front_E[entry.component][idx]))),
                    value=value,
                    neighbour_values=(neighbour_low, neighbour_high),
                    neighbour_jump=jump,
                    relative_jump=relative,
                )
            )
        report.max_relative_jump[lattice] = worst_relative

    found.sort(key=lambda p: (-(p.relative_jump or 0.0), p.distance))
    report.points = found[: max(0, int(max_listed))]
    return report
