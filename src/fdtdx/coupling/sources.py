"""Field adapters: anything that can put a field on the Yee lattices of a grid.

A coupled case should be able to swap a solved finite-element field for a constant, an analytic
stand-in or a saved artefact without touching the coupling, because that swap is how a case builds
its own control: the same call with a uniform field must leave the loader's arrays bit for bit.
:class:`FieldSource` is the one method that makes the swap possible, and the four adapters below
implement it:

* :class:`FemFieldSource` — a field another solver produced, evaluated point by point with a
  coverage flag (:mod:`fdtdx.coupling.fem`);
* :class:`UniformFieldSource` — one constant everywhere, fully covered: the control run;
* :class:`CallableFieldSource` — an analytic ``fn(points) -> values``, called with the *transformed*
  points so a stand-in and a mesh field are given the same coordinates;
* :class:`SamplesFieldSource` — samples that already exist, with the grid checked, so a file taken
  on a different discretisation is refused rather than reshaped.

:func:`as_field_source` coerces whichever of those a case hands over.

:data:`FRAME_KEY` is written into a sample's provenance once its components have been turned into
the Yee frame, so an artefact that round-trips through a file is not rotated twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from fdtdx.coupling.fem import FemField, sample_on_yee_lattices
from fdtdx.coupling.frames import PointTransform, RadialPlaneTransform
from fdtdx.coupling.lattice import YeeLatticeSamples, grid_edges, lattice_points

#: Written into a sample's provenance once its components have been turned into the Yee frame, so
#: an artefact that round-trips through a file is not rotated twice.
FRAME_KEY = "component_frame"


@runtime_checkable
class FieldSource(Protocol):
    """Anything that can put a field on the Yee lattices of a grid.

    One method, so a case can hand a coupling a solved finite-element field, a constant, an
    analytic function or a saved artefact without changing anything else.
    """

    def sample_on(
        self,
        grid: Any,
        lattices: Sequence[str],
        transform: PointTransform | RadialPlaneTransform | None = None,
        *,
        name: str = "f",
        unit: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> YeeLatticeSamples:
        """Values and coverage on ``lattices`` of ``grid``, positions mapped by ``transform``."""
        ...


@dataclass(frozen=True)
class FemFieldSource:
    """A finite-element field from another solver, evaluated point by point with coverage flags.

    Attributes:
        field (FemField): The scalar, vector or tensor field (a DOLFINx function behind
            :class:`~fdtdx.coupling.fem.FemField`).
        padding (float): Bounding-box padding for the collision query, in mesh units.
    """

    field: FemField
    padding: float = 0.0

    def sample_on(
        self,
        grid: Any,
        lattices: Sequence[str],
        transform: PointTransform | RadialPlaneTransform | None = None,
        *,
        name: str = "f",
        unit: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> YeeLatticeSamples:
        del name, unit  # the field's own label is the claim; a coupling may not relabel it
        return sample_on_yee_lattices(
            self.field,
            grid,
            lattices=lattices,
            transform=transform,
            padding=self.padding,
            provenance=dict(provenance or {}),
        )


@dataclass(frozen=True)
class UniformFieldSource:
    """One constant value everywhere, fully covered: the control run of every coupled case.

    Attributes:
        value (float | Sequence[float]): The scalar, the ``n`` components of a vector, or the
            entries of a tensor, written at every point of every lattice.
    """

    value: Any = 0.0

    def sample_on(
        self,
        grid: Any,
        lattices: Sequence[str],
        transform: PointTransform | RadialPlaneTransform | None = None,
        *,
        name: str = "f",
        unit: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> YeeLatticeSamples:
        del transform  # a constant is the same in every frame
        edges = grid_edges(grid)
        block = np.asarray(self.value, dtype=np.float64)
        values: dict[str, np.ndarray] = {}
        covered: dict[str, np.ndarray] = {}
        for lattice in lattices:
            _, shape = lattice_points(edges, lattice)
            if block.ndim == 0:
                values[lattice] = np.full(shape, float(block), dtype=np.float64)
            else:
                values[lattice] = np.broadcast_to(block, (*shape, *block.shape)).copy()
            covered[lattice] = np.ones(shape, dtype=bool)
        return YeeLatticeSamples(
            edges=edges,
            values=values,
            covered=covered,
            name=name,
            unit=unit,
            provenance={**dict(provenance or {}), "source": "uniform", FRAME_KEY: "yee"},
        )


@dataclass(frozen=True)
class CallableFieldSource:
    """An analytic field ``fn(points (N, 3)) -> (N,)`` or ``(N, ...)``, fully covered.

    Attributes:
        fn (Any): Called with the *transformed* points, i.e. in the frame the transform maps to,
            so an analytic stand-in and a mesh field are given the same coordinates.
        frame (str): ``"mesh"`` (the default) states that the returned components are in the frame
            the transform maps to and must be turned into the Yee frame like a mesh field's;
            ``"yee"`` states they are already in the Yee frame and are left alone.
    """

    fn: Any
    frame: str = "mesh"

    def sample_on(
        self,
        grid: Any,
        lattices: Sequence[str],
        transform: PointTransform | RadialPlaneTransform | None = None,
        *,
        name: str = "f",
        unit: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> YeeLatticeSamples:
        edges = grid_edges(grid)
        values: dict[str, np.ndarray] = {}
        covered: dict[str, np.ndarray] = {}
        for lattice in lattices:
            pts, shape = lattice_points(edges, lattice)
            out = np.asarray(self.fn(pts if transform is None else transform.apply(pts)), dtype=np.float64)
            values[lattice] = out.reshape(shape) if out.ndim == 1 else out.reshape((*shape, *out.shape[1:]))
            covered[lattice] = np.ones(shape, dtype=bool)
        prov: dict[str, Any] = {**dict(provenance or {}), "source": "callable"}
        if self.frame == "yee":
            prov[FRAME_KEY] = "yee"
        return YeeLatticeSamples(edges=edges, values=values, covered=covered, name=name, unit=unit, provenance=prov)


@dataclass(frozen=True)
class SamplesFieldSource:
    """Samples that already exist -- read back from an ``.npz`` artefact, or made by hand.

    The grid is checked, so a file taken on a different discretisation is refused rather than
    reshaped.
    """

    samples: YeeLatticeSamples

    def sample_on(
        self,
        grid: Any,
        lattices: Sequence[str],
        transform: PointTransform | RadialPlaneTransform | None = None,
        *,
        name: str = "f",
        unit: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> YeeLatticeSamples:
        del transform, name, unit, provenance
        if not self.samples.matches_edges(grid_edges(grid)):
            raise ValueError("the given samples were taken on a different grid than the one placed")
        missing = [lattice for lattice in lattices if lattice not in self.samples.values]
        if missing:
            raise ValueError(f"the given samples carry no lattice(s) {missing}; have {list(self.samples.lattices)}")
        return self.samples


def as_field_source(source: Any) -> FieldSource:
    """Coerce a field, a number, a callable or existing samples into a :class:`FieldSource`.

    Args:
        source: A :class:`FieldSource`, a :class:`~fdtdx.coupling.fem.FemField`, a
            :class:`~fdtdx.coupling.lattice.YeeLatticeSamples`, a number or array (constant), or
            a callable ``fn(points) -> values``.

    Returns:
        FieldSource: The source itself, or the adapter that wraps it.
    """
    if isinstance(source, YeeLatticeSamples):
        return SamplesFieldSource(source)
    if isinstance(source, FemField):
        return FemFieldSource(source)
    if hasattr(source, "sample_on"):
        return source
    if callable(source):
        return CallableFieldSource(source)
    if np.ndim(source) == 0 or isinstance(source, (list, tuple, np.ndarray)):
        return UniformFieldSource(source)
    raise TypeError(
        f"cannot read a field source from {type(source).__name__}; pass a FemField, a constant, "
        "a callable, YeeLatticeSamples, or an object with a sample_on() method"
    )
