"""The API a case declares: one object per coupled effect, and the stack that puts several on a scene.

A :class:`Coupling` declares one physical effect once -- the field it reads, the unit that field
must be in, whether it is a scalar, a vector or a second-rank tensor, and the per-material response
-- and then does the three steps a case needs:

``sample``
    evaluate a field source (:mod:`fdtdx.coupling.sources`) on the Yee lattices, with the
    coordinate transform applied to the *positions* and, for a vector or tensor field, the same
    transform applied to the *components*
    (:meth:`~fdtdx.coupling.frames.PointTransform.apply_values`). Forgetting the second half is the
    silent error this step exists to prevent.

``perturb``
    hand the samples and the response table to
    :func:`~fdtdx.coupling.perturb.perturb_arrays_with_model`.

``apply``
    both, in one call.

Concrete couplings are thin declarations::

    ThermoOptic(dn_dT={"si": 1.86e-4, "sio2": 1e-5}, reference_temperature=300.0)
    Pockels(r={"ln": r_lithium_niobate}, field_scale=1e6)      # a field solved in V/um
    Photoelastic(p={"si": p_silicon})
    PlasmaDispersion(index={"si": 3.4757 + 3.0836e-5j}, wavelength=1.55e-6)

:class:`PlasmaDispersion` is the one coupling whose response also moves the loader's electric
conductivity: it reads one two-component ``(N, P)`` carrier field in cm^-3 and drives phase and
loss through one complex index. It is declared with each responding material's *complex* index at
the operating wavelength because the loss half cannot be read back from the permittivity alone, and
:meth:`PlasmaDispersion.check_materials` refuses a scene whose material is not that index.

and :class:`MultiCoupling` puts several of them on one scene.

Why :class:`MultiCoupling` is not a loop over ``apply``
------------------------------------------------------

The engine writes each perturbed tensor from the *material table*, not from the array it is
handed: a bulk point takes the inverse of ``response.tensor(base_material, value)``, and a blended
pixel is re-blended from both materials' base tensors. Calling ``apply`` twice would therefore not
compose the two effects, it would let the second one overwrite the first. :class:`MultiCoupling`
composes at the response level instead -- one pass, one report, with a
:class:`~fdtdx.coupling.responses.CompositeResponse` for every material that answers to more than
one field -- so a point that is both hot and strained sees both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence, cast

import numpy as np

from fdtdx.coupling.frames import PointTransform, RadialPlaneTransform
from fdtdx.coupling.lattice import YeeLatticeSamples, lattice_points
from fdtdx.coupling.perturb import (
    PerturbationModel,
    PerturbationReport,
    TensorConstraints,
    check_sample_units,
    perturb_arrays_with_model,
)
from fdtdx.coupling.responses import (
    _ISOTROPY_TOL,
    _UNIT_FACTORS,
    SOREF_BENNETT_1550,
    CompositeResponse,
    MaterialResponse,
    PhotoelasticResponse,
    PlasmaDispersionResponse,
    PockelsResponse,
    SorefBennett,
    ThermoOpticResponse,
)
from fdtdx.coupling.sources import FRAME_KEY, as_field_source
from fdtdx.coupling.tensors import voigt_samples_from_tensor
from fdtdx.materials import Material

#: The lattices a permittivity perturbation can touch: the three E lattices carry the diagonal
#: entries, the cell-vertex lattice ``V`` the off-diagonal ones under the node placement.
PERMITTIVITY_LATTICES: tuple[str, ...] = ("E0", "E1", "E2", "V")

#: Relative tolerance of :meth:`PlasmaDispersion.check_materials`, the "is this scene's material the
#: one the coupling was declared with" check.
_INDEX_TOL = 1e-9


def _unit_label(expected: str | None, scale: float) -> str | None:
    """The label samples must carry for a response that expects ``expected`` after ``field_scale``.

    The inverse of the check in :func:`~fdtdx.coupling.perturb.check_sample_units`: a response
    written in volts per metre with ``field_scale=1e6`` reads samples labelled ``"V/um"``. Returns
    ``None`` (unlabelled, and therefore unchecked) when no known label converts by that factor.
    """
    if expected is None:
        return None
    table = _UNIT_FACTORS.get(expected)
    if table is None:
        return None
    for label, factor in table.items():
        if factor == float(scale):
            return label
    return None


# ---------------------------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------------------------
@dataclass
class CouplingReport:
    """What one coupling (or a stack of them) declared and what the perturbation then touched.

    Attributes:
        coupling (str): The class name of the coupling that ran.
        fields (tuple[str, ...]): The sampled fields it read.
        units (dict[str, str | None]): The unit each field was accepted in.
        perturbation (PerturbationReport | None): The engine's own record of the one pass;
            ``None`` on the per-coupling entries of a :class:`MultiCoupling` report, which share
            the parent's pass.
        extras (dict[str, Any]): The coupling's own block (its coefficients and the largest
            excursion it saw), merged into the top level of :meth:`as_dict`.
        parts (tuple[CouplingReport, ...]): One entry per coupling of a :class:`MultiCoupling`.
    """

    coupling: str
    fields: tuple[str, ...] = ()
    units: dict[str, str | None] = field(default_factory=dict)
    perturbation: PerturbationReport | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    parts: tuple["CouplingReport", ...] = ()

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {} if self.perturbation is None else dict(self.perturbation.as_dict())
        out.update(self.extras)
        out["coupling"] = self.coupling
        out["fields"] = list(self.fields)
        out["units"] = dict(self.units)
        if self.parts:
            out["parts"] = [part.as_dict() for part in self.parts]
        return out

    # The three counters a case prints, forwarded so it need not reach through `perturbation`.
    @property
    def num_bulk_points(self) -> dict[str, int]:
        """Per E lattice, points rewritten with the bulk formula."""
        return dict(self.perturbation.num_bulk_points) if self.perturbation else {}

    @property
    def num_reblended(self) -> dict[str, int]:
        """Per lattice, recorded interface entries re-evaluated at their own field value."""
        return dict(self.perturbation.num_reblended) if self.perturbation else {}

    @property
    def num_uncovered(self) -> dict[str, int]:
        """Per lattice, points that needed a field value and had none."""
        return dict(self.perturbation.num_uncovered) if self.perturbation else {}


# ---------------------------------------------------------------------------------------------
# The base class
# ---------------------------------------------------------------------------------------------
class Coupling:
    """One coupled effect: the field it reads, how each material answers, and how to apply it.

    A subclass states four things and inherits the rest:

    * :attr:`field_name` -- the key the samples are passed under, and the name the response reads;
    * :attr:`field_rank` -- 0 scalar, 1 vector, 2 second-rank tensor. The rank decides whether the
      sampled *components* are turned into the Yee frame along with the positions;
    * :attr:`expected_unit` -- the unit the response's arithmetic is written in, checked against
      the samples' own label at the boundary;
    * :meth:`responses` -- the per-material :class:`~fdtdx.coupling.responses.MaterialResponse`
      table.

    Everything a case calls (:meth:`sample`, :meth:`perturb`, :meth:`apply`) is here.
    """

    #: The samples key and the response's field name.
    field_name: ClassVar[str] = "f"
    #: 0 (scalar), 1 (vector) or 2 (second-rank tensor) as the field leaves the source's mesh.
    field_rank: ClassVar[int] = 0
    #: The unit the response's formulas assume, after the coupling's own ``field_scale``.
    expected_unit: ClassVar[str | None] = None
    #: Whether a sampled vector is axial (a magnetic field, a curl) and picks up ``det(R)``.
    pseudo_vector: ClassVar[bool] = False

    # -- declaration -------------------------------------------------------------------------
    def responses(self) -> dict[str, MaterialResponse]:
        """Which of the user's materials respond, and with what response object."""
        raise NotImplementedError

    def model(self) -> PerturbationModel:
        """The declaration as the engine wants it."""
        return PerturbationModel(responses=self.responses())

    def check_materials(self, materials: Mapping[str, Material]) -> None:
        """Refuse a material this coupling cannot describe, before anything is sampled."""
        unknown = sorted(set(self.material_names()) - set(materials))
        if unknown:
            raise KeyError(f"{type(self).__name__} names materials absent from the scene: {unknown}")

    def material_names(self) -> tuple[str, ...]:
        """Every material name the coupling was declared with, responding or not."""
        return tuple(self.responses())

    @property
    def sample_unit(self) -> str | None:
        """The label samples of this coupling's field must carry (``None``: unlabelled)."""
        return _unit_label(self.expected_unit, self.field_scale_of(None))

    def field_scale_of(self, material: str | None) -> float:
        """The ``field_scale`` the response for ``material`` applies (the common one by default)."""
        del material
        return 1.0

    # -- step 1: sampling --------------------------------------------------------------------
    def sample(
        self,
        field_source: Any,
        grid: Any,
        lattices: Sequence[str] = PERMITTIVITY_LATTICES,
        transform: PointTransform | RadialPlaneTransform | None = None,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> YeeLatticeSamples:
        """Put a field on the Yee lattices, positions *and* components in the loader's frame.

        Args:
            field_source: A :class:`~fdtdx.coupling.sources.FieldSource`, a :class:`~fdtdx.coupling.fem.FemField`, a
                constant, a callable, or ready-made samples.
            grid: The resolved ``RectilinearGrid`` or three edge arrays, in metres.
            lattices (Sequence[str]): Which lattices to evaluate.
            transform: Yee-to-mesh coordinate map; identity when ``None``.
            provenance (Mapping | None): Recorded in the result.

        Returns:
            YeeLatticeSamples: In the form :meth:`perturb` reads.
        """
        source = as_field_source(field_source)
        samples = source.sample_on(
            grid,
            lattices,
            transform,
            name=self.field_name,
            unit=self.sample_unit,
            provenance=provenance,
        )
        return self.prepare(samples, transform)

    def prepare(
        self,
        samples: YeeLatticeSamples,
        transform: PointTransform | RadialPlaneTransform | None = None,
    ) -> YeeLatticeSamples:
        """Express the sampled values in the Yee frame and in the shape the response reads.

        A scalar needs nothing. A vector or a tensor sampled on somebody else's mesh carries its
        components in that mesh's frame; the same transform that moved the positions turns them
        (:meth:`~fdtdx.coupling.frames.PointTransform.apply_values`). Samples that already
        record ``component_frame="yee"`` in their provenance are left alone, so an artefact that
        round-trips through a file is not rotated twice.
        """
        if self.field_rank == 0 or samples.provenance.get(FRAME_KEY) == "yee":
            return samples
        transform = transform or PointTransform()
        values: dict[str, np.ndarray] = {}
        for lattice, block in samples.values.items():
            points, shape = lattice_points(samples.edges, lattice)
            flat = np.asarray(block, dtype=np.float64).reshape((int(np.prod(shape)), -1))
            if self.field_rank == 1:
                turned = transform.apply_values(flat, rank=1, points=points, pseudo=self.pseudo_vector)
                values[lattice] = turned.reshape((*shape, 3))
            else:
                turned = transform.apply_values(flat, rank=2, points=points)
                values[lattice] = turned.reshape((*shape, 3, 3))
        provenance = {**dict(samples.provenance), FRAME_KEY: "yee"}
        return YeeLatticeSamples(
            edges=samples.edges,
            values=values,
            covered={lattice: np.array(mask, copy=True) for lattice, mask in samples.covered.items()},
            name=samples.name,
            unit=samples.unit,
            transform=dict(samples.transform),
            provenance=provenance,
        )

    # -- step 2: perturbation ----------------------------------------------------------------
    def perturb(
        self,
        arrays: Any,
        info: Mapping[str, Any],
        materials: Mapping[str, Material],
        samples: YeeLatticeSamples | Mapping[str, YeeLatticeSamples],
        *,
        uncovered: str = "error",
        constraints: TensorConstraints | None = None,
        offdiag_bulk: str = "error",
    ) -> tuple[Any, CouplingReport]:
        """Rewrite a placed ``ArrayContainer``'s inverse permittivities with these samples.

        Args:
            arrays: The container from ``place_objects`` (after ``extend_material_to_pml`` and
                ``apply_params``).
            info (Mapping): The ``info`` dictionary ``place_objects`` returned.
            materials (Mapping[str, Material]): The scene's material dictionary.
            samples: This coupling's samples, or a mapping from field name to samples.
            uncovered (str): ``"error"`` or ``"unperturbed"``.
            constraints (TensorConstraints | None): Per-voxel physics validation; the lossless
                dielectric set when ``None``.
            offdiag_bulk (str): ``"error"``, ``"project"`` or ``"tensor"``.

        Returns:
            tuple: ``(arrays, report)``.
        """
        self.check_materials(materials)
        table = self.samples_mapping(samples)
        model = self.model()
        out, perturbation = perturb_arrays_with_model(
            arrays,
            info,
            materials,
            table,
            model,
            uncovered=uncovered,
            constraints=constraints,
            offdiag_bulk=offdiag_bulk,
        )
        return out, self.report(perturbation, table, info, materials, model)

    def samples_mapping(
        self, samples: YeeLatticeSamples | Mapping[str, YeeLatticeSamples]
    ) -> dict[str, YeeLatticeSamples]:
        """One :class:`~fdtdx.coupling.lattice.YeeLatticeSamples` keyed by this coupling's field."""
        if isinstance(samples, YeeLatticeSamples):
            return {self.field_name: samples}
        missing = [name for name in self.model().fields if name not in samples]
        if missing:
            raise KeyError(f"{type(self).__name__} needs samples of {missing}; have {sorted(samples)}")
        return dict(samples)

    # -- step 3: both ------------------------------------------------------------------------
    def apply(
        self,
        field_source: Any,
        arrays: Any,
        info: Mapping[str, Any],
        materials: Mapping[str, Material],
        grid: Any = None,
        *,
        lattices: Sequence[str] | None = None,
        transform: PointTransform | RadialPlaneTransform | None = None,
        provenance: Mapping[str, Any] | None = None,
        uncovered: str = "error",
        constraints: TensorConstraints | None = None,
        offdiag_bulk: str = "error",
    ) -> tuple[Any, CouplingReport]:
        """Sample the field and perturb the arrays in one call.

        ``lattices`` defaults to the three E lattices, plus the vertex lattice when the scene
        carries off-diagonal entries. ``grid`` may be omitted when ``field_source`` is already
        :class:`~fdtdx.coupling.lattice.YeeLatticeSamples`.

        Returns:
            tuple: ``(arrays, report)``.
        """
        if lattices is None:
            lattices = (
                PERMITTIVITY_LATTICES
                if getattr(arrays, "inv_permittivity_offdiag", None) is not None
                else PERMITTIVITY_LATTICES[:3]
            )
        if grid is None:
            if isinstance(field_source, YeeLatticeSamples):
                grid = field_source.edges
            else:
                raise ValueError("apply() needs the placed grid (config.resolved_grid) to sample the field on")
        samples = self.sample(field_source, grid, lattices, transform, provenance=provenance)
        return self.perturb(
            arrays,
            info,
            materials,
            samples,
            uncovered=uncovered,
            constraints=constraints,
            offdiag_bulk=offdiag_bulk,
        )

    # -- reporting ---------------------------------------------------------------------------
    def null_value(self) -> float:
        """The field value at which this coupling is exactly the identity."""
        return 0.0

    def excursion(self, values: np.ndarray) -> np.ndarray:
        """``|value - null|`` per point; the norm over the component axes for a vector or tensor."""
        v = np.asarray(values, dtype=np.float64) - self.null_value()
        if self.field_rank == 0 and v.ndim <= 3:
            return np.abs(v)
        return np.sqrt(np.sum(v.reshape((*v.shape[:3], -1)) ** 2, axis=-1))

    def max_excursion(
        self,
        samples: Mapping[str, YeeLatticeSamples],
        info: Mapping[str, Any],
        materials: Mapping[str, Material],
        model: PerturbationModel,
    ) -> tuple[float, dict[str, float]]:
        """The largest excursion over the perturbed points, in total and per material name."""
        material_map = info.get("yee_material_map") or {}
        front_E = np.asarray(material_map.get("front_E"))
        matched = model.table(materials, tuple(material_map.get("material_table", ())))
        s = samples.get(self.field_name)
        largest = 0.0
        per_material: dict[str, float] = {}
        if s is None or front_E.ndim != 4:
            return largest, per_material
        for c in range(3):
            lattice = f"E{c}"
            if lattice not in s.values:
                continue
            for index, (name, _) in matched.items():
                sel = (front_E[c] == index) & s.covered[lattice]
                if not sel.any():
                    continue
                here = float(self.excursion(s.values[lattice])[sel].max())
                per_material[name] = max(per_material.get(name, 0.0), here)
                largest = max(largest, here)
        return largest, per_material

    def report_extras(
        self,
        samples: Mapping[str, YeeLatticeSamples],
        info: Mapping[str, Any],
        materials: Mapping[str, Material],
        model: PerturbationModel,
    ) -> dict[str, Any]:
        """The coupling's own block of the report; subclasses add their coefficients."""
        largest, _ = self.max_excursion(samples, info, materials, model)
        return {"max_field_excursion": largest}

    def report(
        self,
        perturbation: PerturbationReport | None,
        samples: Mapping[str, YeeLatticeSamples],
        info: Mapping[str, Any],
        materials: Mapping[str, Material],
        model: PerturbationModel | None = None,
    ) -> CouplingReport:
        """Wrap the engine's record together with what this coupling declared and saw."""
        model = model or self.model()
        units = {name: (samples[name].unit if name in samples else None) for name in model.fields}
        return CouplingReport(
            coupling=type(self).__name__,
            fields=tuple(model.fields),
            units=units,
            perturbation=perturbation,
            extras=self.report_extras(samples, info, materials, model),
        )


# ---------------------------------------------------------------------------------------------
# Concrete couplings
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ThermoOptic(Coupling):
    """``n(T) = n + (dn/dT) (T - T_ref)`` per material: a temperature moves every principal index.

    The scalar coupling the fork's thermal cases run on. The response is isotropic, so an isotropic
    material stays isotropic and every recorded interface pixel is re-blended with the loader's own
    scalar Kottke formulas, bit for bit.

    Attributes:
        dn_dT (Mapping[str, float]): Thermo-optic coefficient in 1/K per material *name*. A
            material not listed, or listed with zero, is not perturbed and needs no coverage.
        reference_temperature (float): The temperature at which the scene's permittivities hold.
    """

    dn_dT: Mapping[str, float] = field(default_factory=dict)
    reference_temperature: float = 293.15

    field_name: ClassVar[str] = "T"
    field_rank: ClassVar[int] = 0
    expected_unit: ClassVar[str | None] = "K"

    def responses(self) -> dict[str, MaterialResponse]:
        return {
            name: ThermoOpticResponse(dn_dT=float(value), reference_temperature=float(self.reference_temperature))
            for name, value in self.dn_dT.items()
            if float(value) != 0.0
        }

    def material_names(self) -> tuple[str, ...]:
        return tuple(self.dn_dT)

    def null_value(self) -> float:
        return float(self.reference_temperature)

    def perturbed_index(self, material: str, index: Any, temperature: Any) -> np.ndarray:
        """``n + (dn/dT) (T - T_ref)`` for one material, on a grid the loader never assembled.

        The same index law the response applies inside the loader's arrays, exposed for a case that
        builds its own permittivity map -- a mode solver's cross-section, say -- so the coefficients
        are declared once for both paths instead of being repeated in the case file.

        Args:
            material (str): The material name the coefficient was declared under.
            index (Any): The unperturbed refractive index (a scalar or an array).
            temperature (Any): The temperature, in the coupling's own unit.

        Returns:
            np.ndarray: The perturbed index, elementwise.
        """
        if material not in self.dn_dT:
            raise KeyError(
                f"no thermo-optic coefficient declared for material {material!r}; have {self.material_names()}"
            )
        return index + float(self.dn_dT[material]) * (
            np.asarray(temperature, dtype=np.float64) - self.reference_temperature
        )

    def perturbed_permittivity(self, material: str, index: Any, temperature: Any) -> np.ndarray:
        """:meth:`perturbed_index` squared: the relative permittivity a case writes into its grid."""
        return self.perturbed_index(material, index, temperature) ** 2

    def check_materials(self, materials: Mapping[str, Material]) -> None:
        """The historical refusal of the thermo-optic front end: isotropic materials only.

        The engine could carry a diagonal anisotropic base through the same formulas, but an index
        model written as one ``dn/dT`` per material has nothing to say about a birefringent one, so
        it is refused here rather than quietly applied to all three axes.
        """
        super().check_materials(materials)
        for name in self.responses():
            eps = np.asarray(materials[name].permittivity, dtype=np.float64).reshape(3, 3)
            diag = np.diag(eps)
            off = eps - np.diag(diag)
            scale = max(float(np.max(np.abs(diag))), 1.0)
            if np.ptp(diag) > _ISOTROPY_TOL * scale or np.max(np.abs(off)) > _ISOTROPY_TOL:
                raise NotImplementedError(
                    f"material {name!r} is anisotropic; the thermo-optic perturbation handles isotropic materials only"
                )
            if diag[0] <= 0.0:
                raise ValueError(f"material {name!r} has non-positive permittivity {diag[0]}; no index to perturb")

    def report_extras(
        self,
        samples: Mapping[str, YeeLatticeSamples],
        info: Mapping[str, Any],
        materials: Mapping[str, Material],
        model: PerturbationModel,
    ) -> dict[str, Any]:
        largest, per_material = self.max_excursion(samples, info, materials, model)
        material_map = info.get("yee_material_map") or {}
        matched = model.table(materials, tuple(material_map.get("material_table", ())))
        coefficients = {name: float(self.dn_dT[name]) for name, _ in matched.values() if float(self.dn_dT[name]) != 0.0}
        max_dn = max(
            (abs(float(self.dn_dT[name])) * value for name, value in per_material.items()),
            default=0.0,
        )
        return {
            "max_field_excursion": largest,
            "max_delta_T": largest,
            "max_delta_n": float(max_dn),
            "reference_temperature": float(self.reference_temperature),
            "coefficients": coefficients,
        }


@dataclass(frozen=True)
class Pockels(Coupling):
    """Linear electro-optic effect: an electric field tilts and stretches the index ellipsoid.

    ``d(1/eps)_I = sum_k r_Ik E_k`` with ``r`` the contracted ``(6, 3)`` electro-optic matrix in
    metres per volt, rows in Voigt order ``(xx, yy, zz, yz, xz, xy)``. The field is a vector, so
    its components are turned into the Yee frame by the same transform that moved the positions.
    The response makes off-diagonal entries in the bulk, which is what ``offdiag_bulk`` decides.

    Attributes:
        r (Mapping[str, Any]): Per material name, the ``(6, 3)`` matrix in its own frame, which
            must coincide with the grid axes (use ``voigt_permute`` for another crystal cut).
        field_scale (float | Mapping[str, float]): Multiplies the sampled field before use, so a
            field solved in volts per micrometre becomes volts per metre with ``1e6``.
    """

    r: Mapping[str, Any] = field(default_factory=dict)
    field_scale: float | Mapping[str, float] = 1.0

    field_name: ClassVar[str] = "E"
    field_rank: ClassVar[int] = 1
    expected_unit: ClassVar[str | None] = "V/m"

    def field_scale_of(self, material: str | None) -> float:
        if isinstance(self.field_scale, Mapping):
            scales = cast("Mapping[str, float]", self.field_scale)
            if material is None:
                values = {float(v) for v in scales.values()} or {1.0}
                if len(values) > 1:
                    raise ValueError(
                        "a per-material field_scale cannot label one sampled field; give the samples "
                        "one unit and one scale, or split the materials into two couplings"
                    )
                return values.pop()
            return float(scales.get(material, 1.0))
        return float(self.field_scale)

    def responses(self) -> dict[str, MaterialResponse]:
        return {
            name: PockelsResponse(r=matrix, field_scale=self.field_scale_of(name)) for name, matrix in self.r.items()
        }


@dataclass(frozen=True)
class Photoelastic(Coupling):
    """Strain changes the impermeability: ``d(1/eps)_I = sum_J p_IJ S_J`` in Voigt form.

    The sampled field is the symmetric strain tensor
    (:meth:`~fdtdx.coupling.fem.FemField.symmetric_gradient_of` of a displacement). Its
    components are turned into the Yee frame and only then contracted into the 6-vector with
    engineering shears the response reads -- contracting first would put the shear entries on the
    wrong axes. Samples that already carry six components are taken as Voigt and left alone.

    Attributes:
        p (Mapping[str, Any]): Per material name, the ``(6, 6)`` photoelastic matrix in its own
            frame, which must coincide with the grid axes.
        field_scale (float | Mapping[str, float]): Multiplies the sampled strain (dimensionless).
    """

    p: Mapping[str, Any] = field(default_factory=dict)
    field_scale: float | Mapping[str, float] = 1.0

    field_name: ClassVar[str] = "S"
    field_rank: ClassVar[int] = 2
    expected_unit: ClassVar[str | None] = "1"

    def field_scale_of(self, material: str | None) -> float:
        if isinstance(self.field_scale, Mapping):
            scales = cast("Mapping[str, float]", self.field_scale)
            if material is None:
                values = {float(v) for v in scales.values()} or {1.0}
                if len(values) > 1:
                    raise ValueError(
                        "a per-material field_scale cannot label one sampled field; give the samples "
                        "one unit and one scale, or split the materials into two couplings"
                    )
                return values.pop()
            return float(scales.get(material, 1.0))
        return float(self.field_scale)

    def responses(self) -> dict[str, MaterialResponse]:
        return {
            name: PhotoelasticResponse(p=matrix, field_scale=self.field_scale_of(name))
            for name, matrix in self.p.items()
        }

    def prepare(
        self,
        samples: YeeLatticeSamples,
        transform: PointTransform | RadialPlaneTransform | None = None,
    ) -> YeeLatticeSamples:
        first = next(iter(samples.values.values()))
        if first.ndim == 4 and first.shape[-1] == 6:
            return samples  # already the contracted 6-vector, in whatever frame its author says
        # Rotating and contracting are two different steps: samples that are already in the Yee
        # frame (a constant, or an artefact this class wrote) still have to be contracted.
        turn = None if samples.provenance.get(FRAME_KEY) == "yee" else transform
        out = voigt_samples_from_tensor(samples, turn, name=self.field_name, unit=samples.unit)
        out.provenance[FRAME_KEY] = "yee"
        return out


# ---------------------------------------------------------------------------------------------
# Several couplings on one scene
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MultiCoupling(Coupling):
    """Several couplings on one scene: one pass, composed responses, one merged report.

    The couplings are applied in the order given -- temperature first, then strain, say -- at every
    point of every responding material. A material that answers to only one of them keeps that
    coupling's own response object, so a one-element stack is bit for bit the coupling alone.

    Attributes:
        couplings (tuple[Coupling, ...]): In application order.
    """

    couplings: tuple[Coupling, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "couplings", tuple(self.couplings))
        if not self.couplings:
            raise ValueError("MultiCoupling needs at least one coupling")
        seen: dict[str, str] = {}
        for coupling in self.couplings:
            for name in coupling.model().fields:
                owner = seen.get(name)
                if owner is not None:
                    raise ValueError(
                        f"two couplings read the field {name!r} ({owner} and {type(coupling).__name__}); "
                        "one field is sampled once, so give one of them its own field name"
                    )
                seen[name] = type(coupling).__name__

    # -- declaration -------------------------------------------------------------------------
    def responses(self) -> dict[str, MaterialResponse]:
        collected: dict[str, list[MaterialResponse]] = {}
        for coupling in self.couplings:
            for name, response in coupling.responses().items():
                collected.setdefault(name, []).append(response)
        return {
            name: (parts[0] if len(parts) == 1 else CompositeResponse(parts=tuple(parts)))
            for name, parts in collected.items()
        }

    def material_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for coupling in self.couplings:
            for name in coupling.material_names():
                if name not in names:
                    names.append(name)
        return tuple(names)

    def check_materials(self, materials: Mapping[str, Material]) -> None:
        for coupling in self.couplings:
            coupling.check_materials(materials)

    # -- sampling ----------------------------------------------------------------------------
    def sample(
        self,
        field_source: Any,
        grid: Any,
        lattices: Sequence[str] = PERMITTIVITY_LATTICES,
        transform: PointTransform | RadialPlaneTransform | None = None,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> Any:
        """Sample every coupling's field.

        Args:
            field_source: A mapping from field name to source, or a sequence of sources in the
                couplings' order.
            grid: The placed grid.
            lattices (Sequence[str]): Which lattices to evaluate.
            transform: One transform for all fields, or a mapping from field name to transform.
            provenance (Mapping | None): Recorded in every result.

        Returns:
            dict[str, YeeLatticeSamples]: Keyed by field name.
        """
        sources = self._sources(field_source)
        out: dict[str, YeeLatticeSamples] = {}
        for coupling in self.couplings:
            name = coupling.field_name
            own = (
                cast("PointTransform | RadialPlaneTransform | None", transform.get(name))
                if isinstance(transform, Mapping)
                else transform
            )
            out[name] = coupling.sample(sources[name], grid, lattices, own, provenance=provenance)
        return out

    def _sources(self, field_source: Any) -> dict[str, Any]:
        if isinstance(field_source, Mapping):
            missing = [c.field_name for c in self.couplings if c.field_name not in field_source]
            if missing:
                raise KeyError(f"no field source given for {missing}; have {sorted(field_source)}")
            return dict(field_source)
        if isinstance(field_source, (list, tuple)):
            if len(field_source) != len(self.couplings):
                raise ValueError(
                    f"{len(field_source)} field sources for {len(self.couplings)} couplings; pass one each, "
                    "or a mapping keyed by field name"
                )
            return {c.field_name: s for c, s in zip(self.couplings, field_source)}
        raise TypeError("MultiCoupling takes a mapping from field name to source, or one source per coupling")

    def prepare(
        self,
        samples: YeeLatticeSamples,
        transform: PointTransform | RadialPlaneTransform | None = None,
    ) -> YeeLatticeSamples:
        raise NotImplementedError("MultiCoupling prepares each coupling's field through that coupling")

    def samples_mapping(
        self, samples: YeeLatticeSamples | Mapping[str, YeeLatticeSamples]
    ) -> dict[str, YeeLatticeSamples]:
        if isinstance(samples, YeeLatticeSamples):
            if len(self.couplings) != 1:
                raise KeyError(
                    f"MultiCoupling needs samples of {[c.field_name for c in self.couplings]}; got one array"
                )
            return {self.couplings[0].field_name: samples}
        return super().samples_mapping(samples)

    # -- reporting ---------------------------------------------------------------------------
    def report(
        self,
        perturbation: PerturbationReport | None,
        samples: Mapping[str, YeeLatticeSamples],
        info: Mapping[str, Any],
        materials: Mapping[str, Material],
        model: PerturbationModel | None = None,
    ) -> CouplingReport:
        model = model or self.model()
        units = {name: (samples[name].unit if name in samples else None) for name in model.fields}
        parts = tuple(
            CouplingReport(
                coupling=type(coupling).__name__,
                fields=tuple(coupling.model().fields),
                units={n: units.get(n) for n in coupling.model().fields},
                perturbation=None,
                extras=coupling.report_extras(samples, info, materials, coupling.model()),
            )
            for coupling in self.couplings
        )
        return CouplingReport(
            coupling=type(self).__name__,
            fields=tuple(model.fields),
            units=units,
            perturbation=perturbation,
            extras={"order": [type(c).__name__ for c in self.couplings]},
            parts=parts,
        )


def check_coupling_units(coupling: Coupling, samples: Mapping[str, YeeLatticeSamples]) -> dict[str, str]:
    """The unit assertion for a coupling's own model; :meth:`Coupling.perturb` runs it anyway."""
    return check_sample_units(coupling.model(), samples)


# ---------------------------------------------------------------------------------------------
# A concrete coupling that also moves the conductivity
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PlasmaDispersion(Coupling):
    """Free carriers change silicon's complex index: phase from ``dn``, loss from ``dalpha``.

    The coupling is declared with each responding material's unperturbed complex index at the
    operating wavelength, because the loss channel needs both halves: the real part is checked
    against the scene's permittivity and the imaginary part is the ``kappa_0`` the response adds
    its ``dk`` to. :meth:`check_materials` compares the declaration against the scene's own
    ``Material`` and refuses a mismatch, so a case cannot quietly hand the response a different
    silicon from the one it drew.

    It is a :class:`~fdtdx.coupling.effects.Coupling` rather than a bare
    :class:`~fdtdx.coupling.responses.PlasmaDispersionResponse` passed through ``PerturbationModel`` for one reason: the
    coupling owns :meth:`check_materials`, and the index check is the whole safety of the channel.
    The generic route still works for a case that wants it -- ``PerturbationModel`` carries the
    response fine -- it just does not check the material.

    Attributes:
        index (Mapping[str, complex]): Per material name, ``n0 + 1j*kappa0`` at ``wavelength``.
        wavelength (float): Free-space wavelength in metres.
        coefficients (SorefBennett): The power laws; the registry's 1.55 um set by default.
        field_scale (float): Multiplies the sampled ``(N, P)`` block, so carriers given in m^-3
            become cm^-3 with ``1e-6``.
        allow_gain (bool): Permit a negative perturbed conductivity, i.e. an amplifying medium.
    """

    index: Mapping[str, complex] = field(default_factory=dict)
    wavelength: float = 1.55e-6
    coefficients: SorefBennett = SOREF_BENNETT_1550
    field_scale: float = 1.0
    allow_gain: bool = False

    field_name: ClassVar[str] = "C"
    field_rank: ClassVar[int] = 0
    expected_unit: ClassVar[str | None] = "1/cm^3"

    def field_scale_of(self, material: str | None) -> float:
        del material
        return float(self.field_scale)

    def responses(self) -> dict[str, MaterialResponse]:
        return {
            name: PlasmaDispersionResponse(
                extinction=float(complex(value).imag),
                wavelength=float(self.wavelength),
                coefficients=self.coefficients,
                field_scale=float(self.field_scale),
                allow_gain=bool(self.allow_gain),
            )
            for name, value in self.index.items()
        }

    def material_names(self) -> tuple[str, ...]:
        return tuple(self.index)

    def check_materials(self, materials: Mapping[str, Material]) -> None:
        """Refuse a scene whose material is not the complex index this coupling was declared with.

        The comparison is made in the fork's own terms: the declared ``n + i kappa`` is turned into
        a ``Material`` by :meth:`~fdtdx.materials.Material.from_refractive_index` at the same
        wavelength, and its permittivity and conductivity are compared with the scene's.
        """
        super().check_materials(materials)
        for name, value in self.index.items():
            declared = Material.from_refractive_index(complex(value), wavelength=float(self.wavelength))
            scene = materials[name]
            eps_declared = np.asarray(declared.permittivity, dtype=np.float64)
            eps_scene = np.asarray(scene.permittivity, dtype=np.float64)
            sigma_declared = np.asarray(declared.electric_conductivity, dtype=np.float64)
            sigma_scene = np.asarray(scene.electric_conductivity, dtype=np.float64)
            eps_ok = np.allclose(eps_declared, eps_scene, rtol=_INDEX_TOL, atol=0.0)
            sigma_ok = np.allclose(sigma_declared, sigma_scene, rtol=_INDEX_TOL, atol=1e-300)
            if not (eps_ok and sigma_ok):
                raise ValueError(
                    f"material {name!r} in the scene is not the complex index {complex(value)} this "
                    f"PlasmaDispersion was declared with at {self.wavelength * 1e9:.1f} nm: the declaration "
                    f"gives permittivity {eps_declared[0]:.12g} and conductivity {sigma_declared[0]:.12g} S/m, "
                    f"the scene carries {eps_scene[0]:.12g} and {sigma_scene[0]:.12g} S/m. Build the scene's "
                    "material with Material.from_refractive_index(n + 1j*kappa, wavelength=...)."
                )

    def report_extras(
        self,
        samples: Mapping[str, YeeLatticeSamples],
        info: Mapping[str, Any],
        materials: Mapping[str, Material],
        model: PerturbationModel,
    ) -> dict[str, Any]:
        largest, per_material = self.max_excursion(samples, info, materials, model)
        return {
            "max_field_excursion": largest,
            "max_carrier_change_per_material_cm3": {k: float(v) for k, v in per_material.items()},
            "wavelength_m": float(self.wavelength),
            "index": {name: [float(complex(v).real), float(complex(v).imag)] for name, v in self.index.items()},
            "coefficients": self.coefficients.as_dict(),
        }
