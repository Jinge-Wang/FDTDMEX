"""Field-driven permittivity perturbation of the loader's arrays, after the interface blend.

The general form of what :mod:`fdtdx.coupling.thermo_optic` does for one scalar field: a sampled
field (temperature, electric field, strain, ...) changes each responding material's permittivity
tensor at a point, and the change is written into the loader's inverse-permittivity arrays in two
kinds of place:

* a bulk point (one material) gets the inverse of the material's perturbed tensor;
* a pixel or vertex the loader blended is re-blended with the same Kottke construction the loader
  used, from the recorded fill fraction, unit normal and material pair, with both materials'
  tensors taken at that point's field value. Two isotropic tensors go through the scalar formulas
  (bit for bit what the loader wrote); anything else goes through the tensor form
  (:func:`fdtdx.core.physics.geometry_smooth.kottke_tensor`), which reduces to the scalar one.

Every material response is a :class:`MaterialResponse`: given the material's own tensor and the
sampled values at ``K`` points it returns ``(K, 3, 3)`` perturbed tensors, and says which points
are unchanged (so the identity is exact there). The 3-component permittivity tier holds diagonal
tensors only, so a response that produces off-diagonal entries at a bulk point is refused with the
material named; the vertex lattice's off-diagonal entries come from the blend, not from the
material, and are always written.
"""

from __future__ import annotations

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
from fdtdx.coupling.fem_field import YeeLatticeSamples
from fdtdx.materials import Material

_ISOTROPY_TOL = 1e-12
_OFFDIAG_TOL = 1e-12


class MaterialResponse:
    """How one material's permittivity tensor depends on the sampled field(s).

    Subclasses set ``fields`` (the sample names they read, e.g. ``("T",)`` or ``("E",)``) and
    implement :meth:`tensor` and :meth:`unchanged`.
    """

    fields: tuple[str, ...] = ()

    def tensor(self, base: np.ndarray, values: Mapping[str, np.ndarray]) -> np.ndarray:
        """``(K, 3, 3)`` perturbed tensors from the material's ``(3, 3)`` base and the sampled values.

        Args:
            base (np.ndarray): The material's unperturbed permittivity tensor.
            values (Mapping[str, np.ndarray]): Per field name, ``(K,)`` or ``(K, n)`` samples.

        Returns:
            np.ndarray: ``(K, 3, 3)``.
        """
        raise NotImplementedError

    def unchanged(self, values: Mapping[str, np.ndarray]) -> np.ndarray:
        """``(K,)`` bool: points where the response is exactly the identity (left bit for bit)."""
        raise NotImplementedError

    def is_isotropic_response(self) -> bool:
        """Whether an isotropic base stays isotropic under this response (enables the scalar path)."""
        return False


@dataclass(frozen=True)
class ThermoOpticResponse(MaterialResponse):
    """``n(T) = n + dn_dT (T - T_ref)`` applied to every principal index of the base tensor."""

    dn_dT: float
    reference_temperature: float = 293.15
    fields: tuple[str, ...] = ("T",)

    def tensor(self, base: np.ndarray, values: Mapping[str, np.ndarray]) -> np.ndarray:
        dT = np.asarray(values["T"], dtype=np.float64).reshape(-1) - self.reference_temperature
        base = np.asarray(base, dtype=np.float64).reshape(3, 3)
        diag = np.diag(base)
        n = np.sqrt(diag)[None, :] + self.dn_dT * dT[:, None]
        out = np.broadcast_to(base, (dT.shape[0], 3, 3)).copy()
        for c in range(3):
            out[:, c, c] = n[:, c] ** 2
        return out

    def unchanged(self, values: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.asarray(values["T"], dtype=np.float64).reshape(-1) == self.reference_temperature

    def is_isotropic_response(self) -> bool:
        return True


@dataclass(frozen=True)
class PockelsResponse(MaterialResponse):
    """Linear electro-optic effect ``d(1/eps)_ij = sum_k r_ijk E_k`` on the base tensor.

    Attributes:
        r (Sequence[Sequence[float]]): The contracted ``(6, 3)`` electro-optic matrix ``r_{Ik}`` in
            metres per volt, rows in Voigt order ``(xx, yy, zz, yz, xz, xy)``, columns the field
            components in the material's own frame, which must coincide with the grid axes.
        field_scale (float): Multiplies the sampled field before use, so a field solved in volts per
            micrometre becomes volts per metre with ``1e6``.
    """

    r: Sequence[Sequence[float]]
    field_scale: float = 1.0
    fields: tuple[str, ...] = ("E",)

    def _matrix(self) -> np.ndarray:
        m = np.asarray(self.r, dtype=np.float64)
        if m.shape != (6, 3):
            raise ValueError(f"the contracted electro-optic matrix must be (6, 3), got {m.shape}")
        return m

    def tensor(self, base: np.ndarray, values: Mapping[str, np.ndarray]) -> np.ndarray:
        E = np.asarray(values["E"], dtype=np.float64).reshape(-1, 3) * float(self.field_scale)
        delta_voigt = E @ self._matrix().T  # (K, 6): (xx, yy, zz, yz, xz, xy)
        base = np.asarray(base, dtype=np.float64).reshape(3, 3)
        inv_base = np.linalg.inv(base)
        delta = np.zeros((E.shape[0], 3, 3), dtype=np.float64)
        delta[:, 0, 0] = delta_voigt[:, 0]
        delta[:, 1, 1] = delta_voigt[:, 1]
        delta[:, 2, 2] = delta_voigt[:, 2]
        delta[:, 1, 2] = delta[:, 2, 1] = delta_voigt[:, 3]
        delta[:, 0, 2] = delta[:, 2, 0] = delta_voigt[:, 4]
        delta[:, 0, 1] = delta[:, 1, 0] = delta_voigt[:, 5]
        return np.linalg.inv(inv_base[None, :, :] + delta)

    def unchanged(self, values: Mapping[str, np.ndarray]) -> np.ndarray:
        E = np.asarray(values["E"], dtype=np.float64).reshape(-1, 3)
        return np.all(E == 0.0, axis=1)


@dataclass(frozen=True)
class PhotoelasticResponse(MaterialResponse):
    """Photoelastic effect ``d(1/eps)_I = sum_J p_IJ S_J`` (Voigt, engineering shear strains).

    Attributes:
        p (Sequence[Sequence[float]]): The ``(6, 6)`` contracted photoelastic matrix, rows and
            columns in Voigt order ``(xx, yy, zz, yz, xz, xy)``, in the material's frame, which
            must coincide with the grid axes.
        field_scale (float): Multiplies the sampled strain (unitless by default).
    """

    p: Sequence[Sequence[float]]
    field_scale: float = 1.0
    fields: tuple[str, ...] = ("S",)

    def _matrix(self) -> np.ndarray:
        m = np.asarray(self.p, dtype=np.float64)
        if m.shape != (6, 6):
            raise ValueError(f"the contracted photoelastic matrix must be (6, 6), got {m.shape}")
        return m

    def tensor(self, base: np.ndarray, values: Mapping[str, np.ndarray]) -> np.ndarray:
        S = np.asarray(values["S"], dtype=np.float64).reshape(-1, 6) * float(self.field_scale)
        delta_voigt = S @ self._matrix().T
        base = np.asarray(base, dtype=np.float64).reshape(3, 3)
        inv_base = np.linalg.inv(base)
        delta = np.zeros((S.shape[0], 3, 3), dtype=np.float64)
        delta[:, 0, 0] = delta_voigt[:, 0]
        delta[:, 1, 1] = delta_voigt[:, 1]
        delta[:, 2, 2] = delta_voigt[:, 2]
        delta[:, 1, 2] = delta[:, 2, 1] = delta_voigt[:, 3]
        delta[:, 0, 2] = delta[:, 2, 0] = delta_voigt[:, 4]
        delta[:, 0, 1] = delta[:, 1, 0] = delta_voigt[:, 5]
        return np.linalg.inv(inv_base[None, :, :] + delta)

    def unchanged(self, values: Mapping[str, np.ndarray]) -> np.ndarray:
        S = np.asarray(values["S"], dtype=np.float64).reshape(-1, 6)
        return np.all(S == 0.0, axis=1)


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
    #: Per-voxel physics validation of every perturbed tensor: counts of violations and the smallest
    #: eigenvalue seen (see :class:`TensorConstraints`). A violation raises; the counts are kept so a
    #: run's record shows what was checked.
    validation: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_bulk_points": dict(self.num_bulk_points),
            "num_reblended": dict(self.num_reblended),
            "num_tensor_reblended": dict(self.num_tensor_reblended),
            "num_uncovered": dict(self.num_uncovered),
            "uncovered_policy": self.uncovered_policy,
            "max_delta_eps": float(self.max_delta_eps),
            "responding_materials": dict(self.responding_materials),
            "validation": dict(self.validation),
        }


@dataclass(frozen=True)
class TensorConstraints:
    """What a perturbed permittivity tensor must satisfy at every voxel, by the physics it models.

    A lossless dielectric response (thermo-optic, Pockels, photoelastic) keeps the permittivity
    real, symmetric and positive definite: real because the medium is lossless, symmetric because
    it is reciprocal, positive definite because the stored energy ``E . eps . E / 2`` is positive
    [general knowledge; Landau-Lifshitz ECM ch. 11, Yariv-Yeh ch. 4]. A lossy reciprocal medium is
    complex symmetric with a positive-semidefinite imaginary part (passivity, ``e^{-i omega t}``);
    a lossless gyrotropic (magneto-optic) medium is Hermitian with an antisymmetric imaginary part
    (the fork maps that to an antisymmetric real conductivity). The static loader carries the real
    part only, so the default here is the lossless dielectric set; the flags exist so a future
    response can relax them explicitly rather than silently.

    Attributes:
        real (bool): Imaginary parts must vanish (``|Im| <= tol * scale``).
        symmetric (bool): ``|T - T^T| <= tol * scale``.
        positive_definite (bool): Every eigenvalue of the symmetric part exceeds ``tol * scale``.
        tol (float): Relative tolerance, against the largest entry of each tensor (at least 1).
    """

    real: bool = True
    symmetric: bool = True
    positive_definite: bool = True
    tol: float = 1e-10

    def check(self, tensors: np.ndarray) -> dict[str, Any]:
        """Counts of violations over ``(K, 3, 3)`` tensors, plus the smallest eigenvalue seen."""
        t = np.asarray(tensors).reshape(-1, 3, 3)
        scale = np.maximum(np.max(np.abs(t), axis=(1, 2)), 1.0)
        out: dict[str, Any] = {"num": int(t.shape[0])}
        if self.real:
            imag = np.max(np.abs(np.imag(t)), axis=(1, 2)) if np.iscomplexobj(t) else np.zeros(t.shape[0])
            out["num_complex"] = int(np.count_nonzero(imag > self.tol * scale))
        real_part = np.real(t)
        if self.symmetric:
            asym = np.max(np.abs(real_part - np.swapaxes(real_part, -1, -2)), axis=(1, 2))
            out["num_asymmetric"] = int(np.count_nonzero(asym > self.tol * scale))
        if self.positive_definite:
            sym = 0.5 * (real_part + np.swapaxes(real_part, -1, -2))
            smallest = np.linalg.eigvalsh(sym)[:, 0]
            out["num_not_positive_definite"] = int(np.count_nonzero(smallest <= self.tol * scale))
            out["min_eigenvalue"] = float(smallest.min()) if smallest.size else None
        return out

    def raise_on(self, counts: Mapping[str, Any], where: str) -> None:
        bad = {k: v for k, v in counts.items() if k.startswith("num_") and k != "num" and v}
        if bad:
            raise ValueError(f"perturbed permittivity violates its physical constraints at {where}: {bad}")


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
) -> tuple[np.ndarray, np.ndarray | None, PerturbationReport]:
    """Perturb the loader's inverse-permittivity arrays with sampled fields, after the blend.

    Args:
        inv_permittivities (np.ndarray): ``(3, Nx, Ny, Nz)`` diagonal-tier inverse permittivity.
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

    Returns:
        tuple: ``(inv_permittivities, inv_permittivity_offdiag, report)`` as new float64 arrays.

    Raises:
        NotImplementedError: For the 9-component tier, a non-vertex placement, a dispersive
            responding material, or an off-diagonal bulk tensor on the 3-component tier.
        ValueError: For shape mismatches, unknown materials, conflicting responses or an uncovered
            point under the ``"error"`` policy.
    """
    if uncovered not in ("error", "unperturbed"):
        raise ValueError(f"uncovered must be 'error' or 'unperturbed', got {uncovered!r}")
    if int(material_map.get("num_perm_components", 3)) != 3:
        raise NotImplementedError("the perturbation supports the 3-component permittivity tier only")
    placement = material_map.get("offdiag_placement")
    if inv_permittivity_offdiag is not None and placement not in (None, "node"):
        raise NotImplementedError(f"off-diagonal placement {placement!r} is not supported; use 'node' or none")
    front_E = np.asarray(material_map["front_E"])
    material_table = tuple(material_map["material_table"])
    record: SmoothingRecord | None = material_map.get("smoothing_record")
    inv_eps = np.array(inv_permittivities, dtype=np.float64, copy=True)
    if inv_eps.shape[0] != 3 or inv_eps.shape[1:] != front_E.shape[1:]:
        raise ValueError(f"inv_permittivities {inv_eps.shape} does not match front_E {front_E.shape}")
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
        uncovered_policy=uncovered, responding_materials={name: type(r).__name__ for name, r in matched.values()}
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
        new_inv = np.full(m_all.shape[0], np.nan, dtype=np.float64)
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
            _validate(t, f"material {matched[int(index)][0]!r}, lattice {lattice}")
            if not _is_diagonal(t).all():
                name = matched[int(index)][0]
                worst = float(np.max(np.abs(t - t * np.eye(3)[None])))
                raise NotImplementedError(
                    f"material {name!r} gets an off-diagonal permittivity entry ({worst:.3g}) at a bulk point on "
                    f"lattice {lattice}; the 3-component tier holds diagonal tensors only"
                )
            new_inv[sel_idx[write]] = 1.0 / t[:, c, c]
            keep[sel_idx[~write]] = False
            max_delta = max(max_delta, float(np.max(np.abs(t[:, c, c] - base[index][c, c]))))
        target = tuple(a[keep] for a in idx_all)
        inv_eps[c][target] = new_inv[keep]
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
            if entry.write_mode == "row" and entry.full_tensor:
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
            out = np.zeros((k.size, 3 if entry.write_mode == "offdiag" else 1), dtype=np.float64)
            if iso.any():
                e_hi = t_hi[k][iso][:, 0, 0]
                e_lo = t_lo[k][iso][:, 0, 0]
                f = fill[k][iso]
                arithmetic = f * e_hi + (1.0 - f) * e_lo
                harmonic = f / e_hi + (1.0 - f) / e_lo
                n_iso = normal[k][iso]
                if entry.write_mode == "row":
                    out[iso, 0] = kottke_inverse_permittivity(n_iso, arithmetic, harmonic, entry.component, False)
                else:
                    out[iso] = _isotropic_offdiagonal_entries(n_iso, arithmetic, harmonic)
            if not iso.all():
                ten = ~iso
                effective = kottke_tensor(normal[k][ten], t_hi[k][ten], t_lo[k][ten], fill[k][ten])
                if entry.write_mode == "row":
                    out[ten, 0] = effective[:, entry.component, entry.component]
                else:
                    out[ten] = np.stack([effective[:, i, j] for i, j in OFFDIAGONAL_ENTRIES], axis=-1)
                report.num_tensor_reblended[lattice] = report.num_tensor_reblended.get(lattice, 0) + int(
                    np.count_nonzero(ten)
                )
            if entry.write_mode == "row":
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
) -> tuple[Any, PerturbationReport]:
    """Apply :func:`apply_permittivity_perturbation` to a placed ``ArrayContainer``."""
    import jax.numpy as jnp

    material_map = info.get("yee_material_map")
    if material_map is None:
        raise ValueError(
            "place_objects info carries no 'yee_material_map': the perturbation needs material_sampling='yee' or 'yee_smooth'"
        )
    inv_eps = np.asarray(arrays.inv_permittivities)
    offdiag = None if arrays.inv_permittivity_offdiag is None else np.asarray(arrays.inv_permittivity_offdiag)
    new_inv, new_off, report = apply_permittivity_perturbation(
        inv_eps, offdiag, material_map, materials, samples, model, uncovered
    )
    out = arrays.aset("inv_permittivities", jnp.asarray(new_inv, dtype=arrays.inv_permittivities.dtype))
    if new_off is not None:
        out = out.aset("inv_permittivity_offdiag", jnp.asarray(new_off, dtype=arrays.inv_permittivity_offdiag.dtype))
    return out, report
