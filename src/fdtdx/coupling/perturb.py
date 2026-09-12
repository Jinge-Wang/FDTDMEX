"""The engine: it rewrites the loader's assembled arrays, after the interface blend.

A sampled field (temperature, electric field, strain, ...) changes each responding material's
permittivity tensor at a point, and the change is written into the loader's inverse-permittivity
arrays in two kinds of place:

* a bulk point (one material) gets the inverse of the material's perturbed tensor;
* a pixel or vertex the loader blended is re-blended with the same Kottke construction the loader
  used, from the recorded fill fraction, unit normal and material pair, with both materials'
  tensors taken at that point's field value. Two isotropic tensors go through the scalar formulas
  (bit for bit what the loader wrote); anything else goes through the tensor form
  (:func:`fdtdx.core.physics.geometry_smooth.kottke_tensor`), which reduces to the scalar one.

The physics is declared elsewhere (:mod:`fdtdx.coupling.responses`); this module only knows that a
response, given the material's own tensor and the sampled values at ``K`` points, returns
``(K, 3, 3)`` perturbed tensors and says which points are unchanged (so the identity is exact
there).

The 3-component permittivity tier holds diagonal tensors only, so what happens to a response that
produces off-diagonal entries at a bulk point is an explicit choice, ``offdiag_bulk``:

* ``"error"`` (the default) refuses the run and names the material;
* ``"project"`` keeps the diagonal of the perturbed tensor and reports, per material and lattice,
  the largest dropped entry, the ratio of that entry to the tensor's diagonal spread, and the
  mixing angle the drop discards. The eigenvalue error of dropping an off-diagonal entry ``c`` from
  a block whose diagonal entries differ by ``d`` is ``c^2/d``: second order for a strongly
  birefringent base, first order as the block becomes degenerate, which is why the ratio is
  reported per run rather than assumed once;
* ``"tensor"`` writes the loader's 9-component tier instead, so no entry is dropped: a bulk point
  takes row ``c`` of the perturbed tensor's inverse, and a recorded interface pixel takes row ``c``
  of the Kottke blend, exactly as :mod:`fdtdx.core.physics.geometry_raster` writes them.

The vertex lattice's off-diagonal entries come from the blend, not from the material, and are
always written.

The second array: electric conductivity, which the loader does not blend
------------------------------------------------------------------------
The inverse-permittivity arrays hold a *real* tensor, so a response that changes how much a medium
absorbs has nowhere to write there. The loader keeps that in ``electric_conductivity`` in siemens
per metre and the update reads it separately (Schneider ch. 3.12; see
:func:`fdtdx.fdtd.update.update_E`), so :func:`apply_conductivity_perturbation` is a second write,
run after the permittivity pass by :func:`perturb_arrays_with_model` and a no-op for every response
that is not a :class:`~fdtdx.coupling.responses.LossyResponse`.

That write follows a different rule from the one above, because the loader does.
``load_scene_on_yee_lattices`` assembles the conductivity with
``_assemble_property(front_E, table, ncomp, invert=False) * conductivity_spacing``
(:mod:`fdtdx.core.physics.geometry_raster`): every Yee point takes the conductivity of the single
material ``front_E`` records there, including the pixels whose *permittivity* the sub-pixel
smoothing blended, and the whole array is scaled by ``conductivity_spacing``, which is the grid
resolution in metres (``c dt / courant``). The perturbation reproduces that exactly -- one bulk
lookup per Yee point of a responding material, no interface blend, the same scale factor. Matching
the loader is the point: a blend invented here would make a perturbed scene differ from a scene
drawn with the perturbed material, which is the identity the tests gate on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from fdtdx.core.physics.geometry_raster import _material_signature
from fdtdx.core.physics.geometry_smooth import (
    OFFDIAGONAL_ENTRIES,
    SmoothingRecord,
    _isotropic_offdiagonal_entries,
    kottke_inverse_permittivity,
    kottke_tensor,
)
from fdtdx.coupling.lattice import YeeLatticeSamples
from fdtdx.coupling.responses import (
    _ISOTROPY_TOL,
    _UNIT_FACTORS,
    _UNLABELLED,
    LossyResponse,
    MaterialResponse,
    TensorConstraints,
    _normalise_unit,
)
from fdtdx.materials import Material

_OFFDIAG_TOL = 1e-12

#: The off-diagonal bulk policies, in the order they give up on the diagonal tier.
OFFDIAG_BULK_POLICIES: tuple[str, ...] = ("error", "project", "tensor")


@dataclass(frozen=True)
class PerturbationModel:
    """Which of the user's materials respond, and how."""

    responses: Mapping[str, MaterialResponse]

    @property
    def fields(self) -> tuple[str, ...]:
        names: list[str] = []
        for response in self.responses.values():
            for name in response.fields:
                if name not in names:
                    names.append(name)
        return tuple(names)

    def table(
        self, materials: Mapping[str, Material], material_table: Sequence[Material]
    ) -> dict[int, tuple[str, MaterialResponse]]:
        """Table index to (user name, response), matched by material value; refuses conflicts.

        Raises:
            KeyError: If a response names a material absent from ``materials``.
            ValueError: If two names of one material value carry different responses.
        """
        unknown = set(self.responses) - set(materials)
        if unknown:
            raise KeyError(f"perturbation responses name materials absent from the scene: {sorted(unknown)}")
        by_signature: dict[tuple, tuple[str, MaterialResponse]] = {}
        for name, response in self.responses.items():
            signature = _material_signature(materials[name])
            previous = by_signature.get(signature)
            if previous is not None and previous[1] != response:
                raise ValueError(
                    f"materials {previous[0]!r} and {name!r} have the same value but different responses; "
                    "the loader treats them as one material"
                )
            by_signature[signature] = (name, response)
        matched: dict[int, tuple[str, MaterialResponse]] = {}
        for index, material in enumerate(material_table):
            hit = by_signature.get(_material_signature(material))
            if hit is not None:
                matched[index] = hit
        return matched


def check_sample_units(
    model: PerturbationModel,
    samples: Mapping[str, YeeLatticeSamples],
) -> dict[str, str]:
    """Refuse samples whose recorded unit disagrees with what the response's arithmetic assumes.

    Each response states the unit its formulas are written in (:attr:`MaterialResponse.expects_unit`)
    and reaches it by multiplying the samples by its own ``field_scale``. This checks that the two
    agree: samples labelled ``"V/um"`` are consistent with ``field_scale=1e6`` and with nothing else.
    A sample object that carries no ``unit`` attribute, or carries an empty one, is not checked --
    an unlabelled field is not the same claim as a wrong label.

    Args:
        model (PerturbationModel): The responses to check.
        samples (Mapping[str, YeeLatticeSamples]): Samples per field name, as passed to the
            perturbation.

    Returns:
        dict[str, str]: Per field name, the unit that was accepted (``""`` when unlabelled).

    Raises:
        ValueError: If a unit is not a unit of what the response expects, or if it is one but the
            response's ``field_scale`` does not convert it.
    """
    accepted: dict[str, str] = {}
    for material_name, response in model.responses.items():
        for field_name, expected, scale in response.unit_requirements():
            if expected is None:
                continue
            table = _UNIT_FACTORS.get(expected)
            if table is None:
                raise ValueError(
                    f"{type(response).__name__} declares expects_unit={expected!r}, which is not one of "
                    f"{sorted(_UNIT_FACTORS)}; add it to the unit table before using it"
                )
            s = samples.get(field_name)
            if s is None:
                continue  # the perturbation reports the missing field itself
            unit = getattr(s, "unit", None)
            if unit is None:
                continue
            unit = _normalise_unit(unit)
            if unit.lower() in _UNLABELLED:
                accepted[field_name] = ""
                continue
            factor = table.get(unit)
            if factor is None:
                raise ValueError(
                    f"{type(response).__name__} for material {material_name!r} reads its {field_name!r} samples "
                    f"in {expected!r}, but the samples are labelled {unit!r}, which is not a unit of {expected!r}. "
                    f"Known units: {sorted(table)}. Relabel the samples or convert the field."
                )
            if not math.isclose(factor, scale, rel_tol=1e-12, abs_tol=0.0):
                raise ValueError(
                    f"{type(response).__name__} for material {material_name!r} has field_scale={scale:g}, but its "
                    f"{field_name!r} samples are labelled {unit!r}, which is {factor:g} {expected}. "
                    f"Set field_scale={factor:g} or give the samples in {expected!r}."
                )
            accepted[field_name] = unit
    return accepted


@dataclass
class OffdiagRecord:
    """What one material's off-diagonal bulk entries look like on one lattice.

    The numbers a run needs to defend (or refuse) the ``"project"`` policy, all in permittivity
    units and taken over the bulk points of one material on one lattice.

    Attributes:
        num_points (int): Bulk points of this material on this lattice that carry an off-diagonal
            entry above the tier's tolerance.
        max_offdiag (float): Largest ``|eps_ij|``, ``i != j`` -- the entry ``"project"`` drops.
        max_diagonal_spread (float): Largest ``max(diag) - min(diag)`` of the perturbed tensor: the
            splitting the dropped entry has to be compared against (F's criterion).
        max_diagonal_change (float): Largest ``|eps_ii - eps_ii(base)|``: the size of the
            perturbation itself, so the dropped entry can be read as a fraction of the signal.
        ratio (float | None): ``max_offdiag / max_diagonal_spread``, the ratio the two-way thermal
            track's export gates on. ``None`` when the spread vanishes.
        max_pair_ratio (float | None): The worst ``|eps_ij| / |eps_ii - eps_jj|`` over the three
            coupled pairs and every point -- the same ratio, but with each entry weighed against the
            splitting of the two axes *it* mixes, which is the quantity that sets the mixing angle.
            ``inf`` when a pair the entry couples is exactly degenerate: there is then no small
            parameter at all, dropping the entry is first order, and ``"project"`` must not be used.
            ``None`` when no entry couples a split pair.
        ratio_to_change (float | None): ``max_offdiag / max_diagonal_change``: how large the dropped
            entry is as a fraction of the perturbation itself.
        rotation_deg (float | None): ``0.5 * atan(2 * max_pair_ratio)`` in degrees -- the rotation
            of the principal axes that ``"project"`` discards. 45 degrees in the degenerate case,
            where the dropped entry alone sets the axes.
    """

    num_points: int = 0
    max_offdiag: float = 0.0
    max_diagonal_spread: float = 0.0
    max_diagonal_change: float = 0.0
    ratio: float | None = None
    max_pair_ratio: float | None = None
    ratio_to_change: float | None = None
    rotation_deg: float | None = None

    @property
    def degenerate(self) -> bool:
        """Whether an entry couples a pair of axes the perturbed tensor leaves equal.

        There is then no small parameter: the entry sets the principal axes instead of perturbing
        them, and ``"project"`` is not an approximation but a different scene.
        """
        return self.max_pair_ratio is not None and math.isinf(self.max_pair_ratio)

    def as_dict(self) -> dict[str, Any]:
        # ``max_pair_ratio`` is infinite in the degenerate case, which is not valid JSON; the flag
        # carries that meaning instead, so a case's results file stays strict.
        return {
            "num_points": int(self.num_points),
            "max_offdiag": float(self.max_offdiag),
            "max_diagonal_spread": float(self.max_diagonal_spread),
            "max_diagonal_change": float(self.max_diagonal_change),
            "ratio": None if self.ratio is None else float(self.ratio),
            "max_pair_ratio": None if (self.max_pair_ratio is None or self.degenerate) else float(self.max_pair_ratio),
            "degenerate": self.degenerate,
            "ratio_to_change": None if self.ratio_to_change is None else float(self.ratio_to_change),
            "rotation_deg": None if self.rotation_deg is None else float(self.rotation_deg),
        }


def _offdiag_record(tensors: np.ndarray, base: np.ndarray) -> OffdiagRecord:
    """Summarise the off-diagonal content of ``(K, 3, 3)`` perturbed tensors against their base."""
    t = np.asarray(tensors, dtype=np.float64).reshape(-1, 3, 3)
    off = np.abs(t - t * np.eye(3)[None])
    diag = np.stack([t[:, 0, 0], t[:, 1, 1], t[:, 2, 2]], axis=1)
    base_diag = np.asarray(base, dtype=np.float64).reshape(3, 3).diagonal()
    max_off = float(np.max(off))
    spread = float(np.max(np.ptp(diag, axis=1)))
    change = float(np.max(np.abs(diag - base_diag[None, :])))
    # "Degenerate" has to be a relative test: two entries a response leaves equal differ by
    # round-off after one matrix inverse, and calling that a finite splitting turns a 45-degree
    # rotation into a ratio of 1e12 rather than the infinity it is.
    scale = np.maximum(np.max(np.abs(t), axis=(1, 2)), 1.0)
    pair_ratio: float | None = None
    for i, j in OFFDIAGONAL_ENTRIES:
        delta = np.abs(t[:, i, j])
        split = np.abs(diag[:, i] - diag[:, j])
        live = delta > _OFFDIAG_TOL * scale
        if not live.any():
            continue
        split_ok = split > _OFFDIAG_TOL * scale
        if (live & ~split_ok).any():
            worst = math.inf
        else:
            worst = float(np.max(delta[live] / split[live]))
        pair_ratio = worst if pair_ratio is None else max(pair_ratio, worst)
    return OffdiagRecord(
        num_points=int(t.shape[0]),
        max_offdiag=max_off,
        max_diagonal_spread=spread,
        max_diagonal_change=change,
        ratio=(max_off / spread if spread > 0.0 else None),
        max_pair_ratio=pair_ratio,
        ratio_to_change=(max_off / change if change > 0.0 else None),
        rotation_deg=(
            None
            if pair_ratio is None
            else (45.0 if math.isinf(pair_ratio) else math.degrees(0.5 * math.atan(2.0 * pair_ratio)))
        ),
    )


@dataclass
class PerturbationReport:
    """What the perturbation touched."""

    num_bulk_points: dict[str, int] = field(default_factory=dict)
    num_reblended: dict[str, int] = field(default_factory=dict)
    num_tensor_reblended: dict[str, int] = field(default_factory=dict)
    num_uncovered: dict[str, int] = field(default_factory=dict)
    uncovered_policy: str = "error"
    max_delta_eps: float = 0.0
    responding_materials: dict[str, str] = field(default_factory=dict)
    #: Which off-diagonal bulk policy ran.
    offdiag_bulk: str = "error"
    #: Per lattice, per responding material: :class:`OffdiagRecord`. Written whenever a bulk tensor
    #: carries an off-diagonal entry, under ``"project"`` (where it is dropped) and under
    #: ``"tensor"`` (where it is kept), so a run always records how large the term was.
    offdiag: dict[str, dict[str, OffdiagRecord]] = field(default_factory=dict)
    #: Per-voxel physics validation of every perturbed tensor: counts of violations and the smallest
    #: eigenvalue seen (see :class:`fdtdx.coupling.responses.TensorConstraints`). A violation
    #: raises; the counts are kept so a run's record shows what was checked.
    validation: dict[str, Any] = field(default_factory=dict)
    #: The :class:`ConductivityReport` of the second write, or ``None`` when no response in the
    #: model changes the electric conductivity -- which is every response but a
    #: :class:`~fdtdx.coupling.responses.LossyResponse`. Kept out of :meth:`as_dict` when ``None``
    #: so a lossless run's record is exactly what it was before the second write existed.
    conductivity: Any = None

    def as_dict(self) -> dict[str, Any]:
        out = {
            "num_bulk_points": dict(self.num_bulk_points),
            "num_reblended": dict(self.num_reblended),
            "num_tensor_reblended": dict(self.num_tensor_reblended),
            "num_uncovered": dict(self.num_uncovered),
            "uncovered_policy": self.uncovered_policy,
            "max_delta_eps": float(self.max_delta_eps),
            "responding_materials": dict(self.responding_materials),
            "validation": dict(self.validation),
            "offdiag_bulk": self.offdiag_bulk,
            "offdiag": {
                lattice: {name: rec.as_dict() for name, rec in per_material.items()}
                for lattice, per_material in self.offdiag.items()
            },
        }
        if self.conductivity is not None:
            out["conductivity"] = self.conductivity.as_dict()
        return out

    def max_offdiag_ratio(self) -> float | None:
        """The worst :attr:`OffdiagRecord.max_pair_ratio` over every material and lattice.

        The one number a case can gate on: dropping an off-diagonal entry ``c`` from a block whose
        diagonal entries differ by ``d`` moves the eigenvalue by ``c^2/d``, so a ratio of 0.1 is a
        1 % error on the splitting and a degenerate block (``inf``) has no small parameter at all.
        ``None`` when nothing off-diagonal was seen.
        """
        seen = [
            rec.max_pair_ratio
            for per_material in self.offdiag.values()
            for rec in per_material.values()
            if rec.max_pair_ratio is not None
        ]
        return max(seen) if seen else None


def _is_isotropic(tensor: np.ndarray) -> np.ndarray:
    """``(K,)`` bool for ``(K, 3, 3)`` tensors."""
    t = np.asarray(tensor, dtype=np.float64).reshape(-1, 3, 3)
    diag = np.stack([t[:, 0, 0], t[:, 1, 1], t[:, 2, 2]], axis=1)
    scale = np.maximum(np.max(np.abs(diag), axis=1), 1.0)
    off = np.max(np.abs(t - t * np.eye(3)[None]), axis=(1, 2))
    return (np.ptp(diag, axis=1) <= _ISOTROPY_TOL * scale) & (off <= _ISOTROPY_TOL * scale)


def _is_diagonal(tensor: np.ndarray) -> np.ndarray:
    t = np.asarray(tensor, dtype=np.float64).reshape(-1, 3, 3)
    scale = np.maximum(np.max(np.abs(t), axis=(1, 2)), 1.0)
    off = np.max(np.abs(t - t * np.eye(3)[None]), axis=(1, 2))
    return off <= _OFFDIAG_TOL * scale


def _base_tensors(material_table: Sequence[Material]) -> np.ndarray:
    out = np.zeros((len(material_table), 3, 3), dtype=np.float64)
    for index, material in enumerate(material_table):
        t = np.asarray(material.permittivity, dtype=np.float64).reshape(3, 3)
        out[index] = 0.5 * (t + t.T)
    return out


def _values_at(
    samples: Mapping[str, YeeLatticeSamples], names: Sequence[str], lattice: str, index: tuple
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Sampled values of every needed field at ``index`` on ``lattice``, and the joint coverage."""
    values: dict[str, np.ndarray] = {}
    covered = None
    for name in names:
        if name not in samples:
            raise KeyError(f"no samples given for field {name!r}; have {sorted(samples)}")
        s = samples[name]
        if lattice not in s.values:
            raise KeyError(f"samples of field {name!r} carry no lattice {lattice!r}; have {s.lattices}")
        values[name] = s.values[lattice][index]
        cov = s.covered[lattice][index]
        covered = cov if covered is None else (covered & cov)
    assert covered is not None
    return values, covered


def apply_permittivity_perturbation(
    inv_permittivities: np.ndarray,
    inv_permittivity_offdiag: np.ndarray | None,
    material_map: Mapping[str, Any],
    materials: Mapping[str, Material],
    samples: Mapping[str, YeeLatticeSamples],
    model: PerturbationModel,
    uncovered: str = "error",
    constraints: TensorConstraints | None = None,
    offdiag_bulk: str = "error",
) -> tuple[np.ndarray, np.ndarray | None, PerturbationReport]:
    """Perturb the loader's inverse-permittivity arrays with sampled fields, after the blend.

    Args:
        inv_permittivities (np.ndarray): ``(3, Nx, Ny, Nz)`` diagonal-tier inverse permittivity, or
            ``(9, Nx, Ny, Nz)`` under ``offdiag_bulk="tensor"``.
        inv_permittivity_offdiag (np.ndarray | None): ``(3, Nx, Ny, Nz)`` vertex entries or ``None``.
        material_map (Mapping): ``info["yee_material_map"]`` from ``place_objects``.
        materials (Mapping[str, Material]): The user's material dictionary.
        samples (Mapping[str, YeeLatticeSamples]): Per field name, samples on the placed grid
            carrying ``E0``..``E2`` and, when off-diagonal entries exist, ``V``.
        model (PerturbationModel): Which materials respond and how.
        uncovered (str): ``"error"`` or ``"unperturbed"``.
        constraints (TensorConstraints | None): Per-voxel physics validation of every perturbed
            tensor before it is written or blended; the lossless dielectric set (real, symmetric,
            positive definite) when ``None``. A violation raises ``ValueError`` naming the material
            and lattice.
        offdiag_bulk (str): What to do when a response produces an off-diagonal entry at a bulk
            point. ``"error"`` refuses (the default, and what the 3-component tier can represent);
            ``"project"`` keeps the diagonal of the perturbed tensor, writing ``1/eps_cc`` at
            component ``c``, and records the dropped entry in :attr:`PerturbationReport.offdiag`;
            ``"tensor"`` writes the 9-component tier instead and drops nothing, which requires the
            scene to have been placed on that tier.

    Returns:
        tuple: ``(inv_permittivities, inv_permittivity_offdiag, report)`` as new float64 arrays.

    Raises:
        NotImplementedError: For a permittivity tier the policy cannot write, a non-vertex
            placement, a dispersive responding material, or an off-diagonal bulk tensor under
            ``offdiag_bulk="error"``.
        ValueError: For shape mismatches, unknown materials, conflicting responses or an uncovered
            point under the ``"error"`` policy.
    """
    if uncovered not in ("error", "unperturbed"):
        raise ValueError(f"uncovered must be 'error' or 'unperturbed', got {uncovered!r}")
    if offdiag_bulk not in OFFDIAG_BULK_POLICIES:
        raise ValueError(f"offdiag_bulk must be one of {OFFDIAG_BULK_POLICIES}, got {offdiag_bulk!r}")
    num_perm_components = int(material_map.get("num_perm_components", 3))
    tensor_tier = offdiag_bulk == "tensor"
    if tensor_tier:
        if num_perm_components != 9:
            raise NotImplementedError(
                "offdiag_bulk='tensor' writes the loader's 9-component permittivity tier, but this scene was "
                f"placed on the {num_perm_components}-component tier. Place it on the 9-component tier "
                "(material_sampling='yee_smooth' with yee_smooth_full_tensor=True and "
                "yee_smooth_offdiag_placement='pixel', or a material carrying an off-diagonal permittivity "
                "entry of its own) and sample the field on that grid."
            )
    elif num_perm_components != 3:
        raise NotImplementedError(
            "the perturbation supports the 3-component permittivity tier only; pass offdiag_bulk='tensor' "
            "to write the 9-component tier instead"
        )
    placement = material_map.get("offdiag_placement")
    if inv_permittivity_offdiag is not None and placement not in (None, "node"):
        raise NotImplementedError(f"off-diagonal placement {placement!r} is not supported; use 'node' or none")
    front_E = np.asarray(material_map["front_E"])
    material_table = tuple(material_map["material_table"])
    record: SmoothingRecord | None = material_map.get("smoothing_record")
    inv_eps = np.array(inv_permittivities, dtype=np.float64, copy=True)
    if inv_eps.shape[0] != num_perm_components or inv_eps.shape[1:] != front_E.shape[1:]:
        raise ValueError(
            f"inv_permittivities {inv_eps.shape} does not match the {num_perm_components}-component tier "
            f"on front_E {front_E.shape}"
        )
    offdiag = (
        None if inv_permittivity_offdiag is None else np.array(inv_permittivity_offdiag, dtype=np.float64, copy=True)
    )
    needed = model.fields
    for name in needed:
        s = samples.get(name)
        if s is None:
            raise KeyError(f"no samples given for field {name!r}")
        for lattice in ("E0", "E1", "E2"):
            if lattice not in s.values or s.values[lattice].shape[:3] != front_E.shape[1:]:
                raise ValueError(f"samples of {name!r} lack lattice {lattice!r} on the placed grid {front_E.shape[1:]}")

    matched = model.table(materials, material_table)
    for index, (name, _) in matched.items():
        if material_table[index].is_dispersive:
            raise NotImplementedError(f"material {name!r} is dispersive; only the static permittivity is perturbed")
    active = np.zeros(len(material_table), dtype=bool)
    for index in matched:
        active[index] = True
    base = _base_tensors(material_table)
    constraints = constraints or TensorConstraints()
    report = PerturbationReport(
        uncovered_policy=uncovered,
        responding_materials={name: type(r).__name__ for name, r in matched.values()},
        offdiag_bulk=offdiag_bulk,
    )
    max_delta = 0.0
    validation: dict[str, Any] = {
        "constraints": {
            "real": constraints.real,
            "symmetric": constraints.symmetric,
            "positive_definite": constraints.positive_definite,
        },
        "num_checked": 0,
        "min_eigenvalue": None,
    }

    def _validate(tensors: np.ndarray, where: str) -> None:
        counts = constraints.check(tensors)
        validation["num_checked"] += counts["num"]
        smallest = counts.get("min_eigenvalue")
        if smallest is not None:
            validation["min_eigenvalue"] = (
                smallest if validation["min_eigenvalue"] is None else min(validation["min_eigenvalue"], smallest)
            )
        constraints.raise_on(counts, where)

    def _uncovered(lattice: str, count: int) -> None:
        report.num_uncovered[lattice] = report.num_uncovered.get(lattice, 0) + int(count)
        if count and uncovered == "error":
            raise ValueError(
                f"{count} point(s) on lattice {lattice} carry a responding material but lie outside the "
                "sampled field; extend the field's domain or pass uncovered='unperturbed'"
            )

    def _perturbed(index: int, values: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Perturbed ``(K, 3, 3)`` tensors of table entry ``index`` and the unchanged mask."""
        hit = matched.get(index)
        K = next(iter(values.values())).shape[0]
        if hit is None:
            return np.broadcast_to(base[index], (K, 3, 3)).copy(), np.ones(K, dtype=bool)
        response = hit[1]
        sub = {name: values[name] for name in response.fields}
        return response.tensor(base[index], sub), response.unchanged(sub)

    # 1. Bulk points.
    for c in range(3):
        lattice = f"E{c}"
        material = front_E[c]
        needs = active[material]
        idx_all = np.nonzero(needs)
        if idx_all[0].size == 0:
            report.num_bulk_points[lattice] = 0
            continue
        values, cov = _values_at(samples, needed, lattice, idx_all)
        _uncovered(lattice, int(np.count_nonzero(~cov)))
        keep = cov.copy()
        m_all = material[idx_all]
        new_inv = np.full((m_all.shape[0], 3 if tensor_tier else 1), np.nan, dtype=np.float64)
        for index in np.unique(m_all[keep]):
            sel = keep & (m_all == index)
            sub = {name: values[name][sel] for name in needed}
            tensors, unchanged = _perturbed(int(index), sub)
            sel_idx = np.nonzero(sel)[0]
            write = ~unchanged
            if not write.any():
                keep[sel_idx] = False
                continue
            t = tensors[write]
            name = matched[int(index)][0]
            _validate(t, f"material {name!r}, lattice {lattice}")
            skew = ~_is_diagonal(t)
            if skew.any():
                if offdiag_bulk == "error":
                    worst = float(np.max(np.abs(t - t * np.eye(3)[None])))
                    raise NotImplementedError(
                        f"material {name!r} gets an off-diagonal permittivity entry ({worst:.3g}) at a bulk point "
                        f"on lattice {lattice}; the 3-component tier holds diagonal tensors only. Pass "
                        "offdiag_bulk='project' to keep the diagonal and report the dropped entry, or "
                        "offdiag_bulk='tensor' to write the 9-component tier."
                    )
                report.offdiag.setdefault(lattice, {})[name] = _offdiag_record(t[skew], base[index])
            if tensor_tier:
                new_inv[sel_idx[write]] = np.linalg.inv(t)[:, c, :]
                max_delta = max(max_delta, float(np.max(np.abs(t - base[index][None]))))
            else:
                new_inv[sel_idx[write], 0] = 1.0 / t[:, c, c]
                max_delta = max(max_delta, float(np.max(np.abs(t[:, c, c] - base[index][c, c]))))
            keep[sel_idx[~write]] = False
        target = tuple(a[keep] for a in idx_all)
        if tensor_tier:
            for j in range(3):
                inv_eps[3 * c + j][target] = new_inv[keep, j]
        else:
            inv_eps[c][target] = new_inv[keep, 0]
        report.num_bulk_points[lattice] = int(np.count_nonzero(keep))

    # 2. Recorded interface pixels and vertices.
    if record is not None:
        for entry in record.passes:
            if entry.field == "E":
                lattice = f"E{entry.component}"
            elif entry.field == "V":
                lattice = "V"
            else:
                continue
            hi, lo = entry.material_hi, entry.material_lo
            involved = active[hi] | active[lo]
            if not involved.any():
                continue
            if entry.write_mode == "row" and entry.full_tensor and not tensor_tier:
                raise NotImplementedError("re-blending the 9-component row tier is not supported")
            if entry.write_mode not in ("row", "offdiag"):
                raise NotImplementedError(f"re-blending write mode {entry.write_mode!r} is not supported")
            cells = entry.cells[involved]
            index = (cells[:, 0], cells[:, 1], cells[:, 2])
            values, cov = _values_at(samples, needed, lattice, index)
            _uncovered(lattice, int(np.count_nonzero(~cov)))
            if not cov.any():
                continue
            fill = entry.fill[involved]
            normal = entry.normal[involved]
            m_hi = hi[involved]
            m_lo = lo[involved]
            t_hi = np.zeros((cells.shape[0], 3, 3))
            t_lo = np.zeros((cells.shape[0], 3, 3))
            unchanged = np.ones(cells.shape[0], dtype=bool)
            for side, m_side, t_side in (("hi", m_hi, t_hi), ("lo", m_lo, t_lo)):
                for mat in np.unique(m_side):
                    sel = m_side == mat
                    sub = {name: values[name][sel] for name in needed}
                    tensors, same = _perturbed(int(mat), sub)
                    t_side[sel] = tensors
                    unchanged[sel] &= same
            keep = cov & ~unchanged
            if not keep.any():
                continue
            k = np.nonzero(keep)[0]
            _validate(t_hi[k], f"lattice {lattice} (front side of blended pixels)")
            _validate(t_lo[k], f"lattice {lattice} (back side of blended pixels)")
            cells_k = cells[k]
            idx_k = (cells_k[:, 0], cells_k[:, 1], cells_k[:, 2])
            iso = _is_isotropic(t_hi[k]) & _is_isotropic(t_lo[k])
            row_width = 3 if (entry.write_mode == "offdiag" or entry.full_tensor) else 1
            out = np.zeros((k.size, row_width), dtype=np.float64)
            if iso.any():
                e_hi = t_hi[k][iso][:, 0, 0]
                e_lo = t_lo[k][iso][:, 0, 0]
                f = fill[k][iso]
                arithmetic = f * e_hi + (1.0 - f) * e_lo
                harmonic = f / e_hi + (1.0 - f) / e_lo
                n_iso = normal[k][iso]
                if entry.write_mode == "row":
                    out[iso] = kottke_inverse_permittivity(
                        n_iso, arithmetic, harmonic, entry.component, entry.full_tensor
                    ).reshape(-1, row_width)
                else:
                    out[iso] = _isotropic_offdiagonal_entries(n_iso, arithmetic, harmonic)
            if not iso.all():
                ten = ~iso
                effective = kottke_tensor(normal[k][ten], t_hi[k][ten], t_lo[k][ten], fill[k][ten])
                if entry.write_mode == "row":
                    out[ten] = (
                        effective[:, entry.component, :]
                        if entry.full_tensor
                        else effective[:, entry.component, entry.component].reshape(-1, 1)
                    )
                else:
                    out[ten] = np.stack([effective[:, i, j] for i, j in OFFDIAGONAL_ENTRIES], axis=-1)
                report.num_tensor_reblended[lattice] = report.num_tensor_reblended.get(lattice, 0) + int(
                    np.count_nonzero(ten)
                )
            if entry.write_mode == "row":
                if entry.full_tensor:
                    for j in range(3):
                        inv_eps[3 * entry.component + j][idx_k] = out[:, j]
                else:
                    inv_eps[entry.component][idx_k] = out[:, 0]
            else:
                if offdiag is None:
                    raise ValueError("the record holds vertex entries but no off-diagonal array was given")
                for q in range(3):
                    offdiag[q][idx_k] = out[:, q]
            report.num_reblended[lattice] = report.num_reblended.get(lattice, 0) + int(k.size)

    report.max_delta_eps = max_delta
    report.validation = validation
    return inv_eps, offdiag, report


def perturb_arrays_with_model(
    arrays: Any,
    info: Mapping[str, Any],
    materials: Mapping[str, Material],
    samples: Mapping[str, YeeLatticeSamples],
    model: PerturbationModel,
    uncovered: str = "error",
    constraints: TensorConstraints | None = None,
    offdiag_bulk: str = "error",
) -> tuple[Any, PerturbationReport]:
    """Apply :func:`apply_permittivity_perturbation` to a placed ``ArrayContainer``.

    Also checks every sample's recorded unit against the unit its response's arithmetic assumes
    (:func:`check_sample_units`), which is the boundary where a field solved in volts per micrometre
    meets a response written in volts per metre.
    """
    import jax.numpy as jnp

    material_map = info.get("yee_material_map")
    if material_map is None:
        raise ValueError(
            "place_objects info carries no 'yee_material_map': the perturbation needs material_sampling='yee' or 'yee_smooth'"
        )
    check_sample_units(model, samples)
    inv_eps = np.asarray(arrays.inv_permittivities)
    offdiag = None if arrays.inv_permittivity_offdiag is None else np.asarray(arrays.inv_permittivity_offdiag)
    new_inv, new_off, report = apply_permittivity_perturbation(
        inv_eps,
        offdiag,
        material_map,
        materials,
        samples,
        model,
        uncovered,
        constraints=constraints,
        offdiag_bulk=offdiag_bulk,
    )
    out = arrays.aset("inv_permittivities", jnp.asarray(new_inv, dtype=arrays.inv_permittivities.dtype))
    if new_off is not None:
        out = out.aset("inv_permittivity_offdiag", jnp.asarray(new_off, dtype=arrays.inv_permittivity_offdiag.dtype))
    # The second write: a response that also changes how much the medium absorbs puts the loader's
    # electric_conductivity here, by the loader's own rule (bulk lookup, no interface blend). A
    # no-op for every response that is not a LossyResponse.
    out, report.conductivity = perturb_conductivity_with_model(out, info, materials, samples, model, uncovered)
    return out, report


# ------------------------------------------------------------------------------------------------
# the conductivity write
# ------------------------------------------------------------------------------------------------
@dataclass
class ConductivityReport:
    """What the conductivity pass touched."""

    #: Per E lattice, points whose conductivity was rewritten.
    num_points: dict[str, int] = field(default_factory=dict)
    #: Per lattice, points that needed a field value and had none.
    num_uncovered: dict[str, int] = field(default_factory=dict)
    uncovered_policy: str = "error"
    #: Largest ``|sigma - sigma_base|`` in S/m over the points written.
    max_delta_sigma: float = 0.0
    #: Smallest conductivity written, in S/m. Negative means gain.
    min_sigma: float | None = None
    #: Largest conductivity written, in S/m.
    max_sigma: float | None = None
    #: Points whose conductivity came out negative (gain), which is refused unless allowed.
    num_gain_points: int = 0
    #: The responses that wrote, per material name.
    responding_materials: dict[str, str] = field(default_factory=dict)
    #: The loader's scale factor between the physical conductivity and the stored array, in metres.
    conductivity_spacing: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_points": dict(self.num_points),
            "num_uncovered": dict(self.num_uncovered),
            "uncovered_policy": self.uncovered_policy,
            "max_delta_sigma": float(self.max_delta_sigma),
            "min_sigma": None if self.min_sigma is None else float(self.min_sigma),
            "max_sigma": None if self.max_sigma is None else float(self.max_sigma),
            "num_gain_points": int(self.num_gain_points),
            "responding_materials": dict(self.responding_materials),
            "conductivity_spacing": (None if self.conductivity_spacing is None else float(self.conductivity_spacing)),
        }


def _base_conductivities(material_table: Sequence[Material]) -> np.ndarray:
    """``(M, 3)`` diagonal electric conductivity per material, in S/m."""
    out = np.zeros((len(material_table), 3), dtype=np.float64)
    for index, material in enumerate(material_table):
        sigma = np.asarray(material.electric_conductivity, dtype=np.float64).reshape(3, 3)
        out[index] = np.diag(sigma)
    return out


def _lossy_table(
    model: PerturbationModel, materials: Mapping[str, Material], material_table: Sequence[Material]
) -> dict[int, tuple[str, LossyResponse]]:
    """The subset of the model's table whose responses write a conductivity."""
    return {
        index: (name, response)
        for index, (name, response) in model.table(materials, material_table).items()
        if isinstance(response, LossyResponse)
    }


def apply_conductivity_perturbation(
    electric_conductivity: np.ndarray | None,
    material_map: Mapping[str, Any],
    materials: Mapping[str, Material],
    samples: Mapping[str, YeeLatticeSamples],
    model: PerturbationModel,
    uncovered: str = "error",
) -> tuple[np.ndarray | None, ConductivityReport]:
    """Rewrite the loader's ``electric_conductivity`` from sampled fields, the loader's own way.

    One bulk lookup per Yee point of a responding material and no interface blend, because that is
    what ``load_scene_on_yee_lattices`` does for this array: the permittivity is smoothed at an
    interface pixel, the conductivity is not. The values the response returns are physical (S/m)
    and are multiplied by ``conductivity_spacing`` on the way in, which is the same scaling the
    loader applied.

    Args:
        electric_conductivity (np.ndarray | None): ``(1, 3 or 9, Nx, Ny, Nz)`` as the loader left
            it, i.e. already scaled by ``conductivity_spacing``.
        material_map (Mapping): ``info["yee_material_map"]`` from ``place_objects``. Must carry
            ``conductivity_spacing``; a scene placed by an engine older than the loss channel does
            not, and is refused with that message.
        materials (Mapping[str, Material]): The user's material dictionary.
        samples (Mapping[str, YeeLatticeSamples]): Samples per field name.
        model (PerturbationModel): Which materials respond. Responses that are not
            :class:`~fdtdx.coupling.responses.LossyResponse` are skipped here; they moved the permittivity only.
        uncovered (str): ``"error"`` or ``"unperturbed"``.

    Returns:
        tuple: ``(electric_conductivity, report)``; the array is a new float64 array, or ``None``
        when no response writes a conductivity and none was given.

    Raises:
        NotImplementedError: For the 1- and 9-component conductivity tiers. The 9-component tier is
            a conductivity *tensor*, which no response here produces; the 1-component tier holds
            one row for all three E lattices, so a per-lattice write into it is not defined. Yee
            sampling never builds either for a diagonal material.
        ValueError: If the scene carries no conductivity array to write into, if the shapes
            disagree, if a point is uncovered under the ``"error"`` policy, or if a conductivity
            comes out negative and the response does not allow gain.
    """
    if uncovered not in ("error", "unperturbed"):
        raise ValueError(f"uncovered must be 'error' or 'unperturbed', got {uncovered!r}")
    front_E = np.asarray(material_map["front_E"])
    material_table = tuple(material_map["material_table"])
    matched = _lossy_table(model, materials, material_table)
    report = ConductivityReport(
        uncovered_policy=uncovered,
        responding_materials={name: type(r).__name__ for name, r in matched.values()},
    )
    if not matched:
        return (None if electric_conductivity is None else np.array(electric_conductivity, dtype=np.float64)), report

    if electric_conductivity is None:
        names = sorted(name for name, _ in matched.values())
        raise ValueError(
            f"the response(s) for {names} write an electric conductivity, but this scene has no conductivity "
            "array: the loader allocates one only when some material in the scene is electrically conductive. "
            "Give the responding material its own unperturbed loss (Material.from_refractive_index(n + 1j*kappa, "
            "wavelength=...)), which is the physically honest way to say the medium absorbs at all."
        )
    spacing = material_map.get("conductivity_spacing")
    if spacing is None:
        raise ValueError(
            "info['yee_material_map'] carries no 'conductivity_spacing'; the loader scales the stored "
            "conductivity by it (c dt / courant, the grid resolution in metres) and the perturbation cannot "
            "write the array without it. Place the scene with an engine that records it."
        )
    spacing = float(spacing)

    sigma_arr = np.array(electric_conductivity, dtype=np.float64, copy=True)
    if sigma_arr.ndim != 4 or sigma_arr.shape[1:] != front_E.shape[1:]:
        raise ValueError(f"electric_conductivity {sigma_arr.shape} does not sit on front_E {front_E.shape}")
    if sigma_arr.shape[0] != 3:
        raise NotImplementedError(
            f"the loss channel writes the 3-component conductivity tier; this scene carries the "
            f"{sigma_arr.shape[0]}-component tier. The 9-component tier is a conductivity tensor, which no "
            "response here produces; the 1-component tier holds one row for all three E lattices, so a "
            "per-lattice conductivity cannot be written into it"
        )
    report.conductivity_spacing = spacing

    base_eps = np.zeros((len(material_table), 3, 3), dtype=np.float64)
    for index, material in enumerate(material_table):
        tensor = np.asarray(material.permittivity, dtype=np.float64).reshape(3, 3)
        base_eps[index] = 0.5 * (tensor + tensor.T)
    base_sigma = _base_conductivities(material_table)

    active = np.zeros(len(material_table), dtype=bool)
    for index in matched:
        active[index] = True
    needed = model.fields
    max_delta = 0.0
    seen_min: float | None = None
    seen_max: float | None = None

    for c in range(3):
        lattice = f"E{c}"
        material = front_E[c]
        needs = active[material]
        idx_all = np.nonzero(needs)
        if idx_all[0].size == 0:
            report.num_points[lattice] = 0
            continue
        values, cov = _values_at(samples, needed, lattice, idx_all)
        missing = int(np.count_nonzero(~cov))
        report.num_uncovered[lattice] = report.num_uncovered.get(lattice, 0) + missing
        if missing and uncovered == "error":
            raise ValueError(
                f"{missing} point(s) on lattice {lattice} carry a lossy responding material but lie outside "
                "the sampled field; extend the field's domain or pass uncovered='unperturbed'"
            )
        keep = cov.copy()
        m_all = material[idx_all]
        new_sigma = np.full(m_all.shape[0], np.nan, dtype=np.float64)
        for index in np.unique(m_all[keep]):
            sel = keep & (m_all == index)
            sub = {name: values[name][sel] for name in needed}
            name, response = matched[int(index)]
            sigma_new = np.asarray(response.conductivity(base_eps[index], base_sigma[index], sub), dtype=np.float64)
            if sigma_new.shape != (int(sel.sum()), 3):
                raise ValueError(
                    f"{type(response).__name__}.conductivity for material {name!r} returned "
                    f"{sigma_new.shape}, expected {(int(sel.sum()), 3)}"
                )
            write = ~response.unchanged(sub)
            sel_idx = np.nonzero(sel)[0]
            if not write.any():
                keep[sel_idx] = False
                continue
            written = sigma_new[write, c]
            negative = int(np.count_nonzero(written < 0.0))
            if negative:
                report.num_gain_points += negative
                if not bool(getattr(response, "allow_gain", False)):
                    raise ValueError(
                        f"material {name!r} gets a negative electric conductivity "
                        f"({float(written.min()):.6g} S/m) at {negative} point(s) on lattice {lattice}: the "
                        "perturbed medium amplifies rather than absorbs. Check the sign of the loss model, or "
                        "set allow_gain=True on the response if an amplifying medium is meant."
                    )
            new_sigma[sel_idx[write]] = written
            max_delta = max(max_delta, float(np.max(np.abs(written - base_sigma[index][c]))))
            seen_min = float(written.min()) if seen_min is None else min(seen_min, float(written.min()))
            seen_max = float(written.max()) if seen_max is None else max(seen_max, float(written.max()))
            keep[sel_idx[~write]] = False
        target = tuple(a[keep] for a in idx_all)
        sigma_arr[c][target] = new_sigma[keep] * spacing
        report.num_points[lattice] = int(np.count_nonzero(keep))

    report.max_delta_sigma = max_delta
    report.min_sigma = seen_min
    report.max_sigma = seen_max
    return sigma_arr, report


def perturb_conductivity_with_model(
    arrays: Any,
    info: Mapping[str, Any],
    materials: Mapping[str, Material],
    samples: Mapping[str, YeeLatticeSamples],
    model: PerturbationModel,
    uncovered: str = "error",
) -> tuple[Any, ConductivityReport | None]:
    """Apply :func:`apply_conductivity_perturbation` to a placed ``ArrayContainer``.

    A no-op, returning the container unchanged and a ``None`` report, when no response in the model
    writes a conductivity -- which is every response outside this module, so a lossless run's record
    is exactly what it was. This is what
    :func:`perturb_arrays_with_model` calls after its own pass, so a
    case that goes through a :class:`~fdtdx.coupling.effects.Coupling` gets both writes from one
    call and never has to know the loss channel exists.
    """
    material_map = info.get("yee_material_map")
    if material_map is None:
        raise ValueError(
            "place_objects info carries no 'yee_material_map': the perturbation needs "
            "material_sampling='yee' or 'yee_smooth'"
        )
    if not _lossy_table(model, materials, tuple(material_map["material_table"])):
        return arrays, None
    import jax.numpy as jnp

    current = getattr(arrays, "electric_conductivity", None)
    if current is None:
        # Raises with the message that says why a lossless scene has no array to write into.
        apply_conductivity_perturbation(None, material_map, materials, samples, model, uncovered)
        raise AssertionError("unreachable: a lossy response with no conductivity array must raise")
    new_sigma, report = apply_conductivity_perturbation(
        np.asarray(current), material_map, materials, samples, model, uncovered
    )
    assert new_sigma is not None
    return arrays.aset("electric_conductivity", jnp.asarray(new_sigma, dtype=current.dtype)), report
