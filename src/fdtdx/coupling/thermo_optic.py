"""Thermo-optic perturbation of the assembled permittivity arrays, applied after the interface blend.

A temperature field changes each material's refractive index, ``n(T) = n + (dn/dT) (T - T_ref)``,
and the loader has already turned the scene into per-Yee-point inverse permittivities, with every
two-material pixel replaced by its Kottke blend. The perturbation is applied *to those arrays*, in
two kinds of place:

* A bulk point (one material) gets the closed form ``1 / n(T)^2`` of the material it sampled.
* A blended pixel or vertex is **re-blended**: the loader recorded its fill fraction, its unit
  normal and the two materials (:class:`fdtdx.core.physics.geometry_smooth.SmoothingPass`); the
  same Kottke formulas are evaluated with both materials' permittivities taken at the pixel's own
  temperature. Geometry is untouched, so the interface keeps its sub-pixel treatment and the
  smooth temperature field is sampled exactly once, at the pixel. Off-diagonal vertex entries get
  the same treatment with the off-diagonal formula.

This is the ordering Tidy3D's ``perturbed_mediums_copy`` arrives at from the other side: there the
perturbation makes a spatially varying medium per structure and the solver's sub-pixel averaging
runs afterwards on the perturbed media (``subpixel=True`` on the derived custom medium). Applying
the perturbation to the recorded blend is the same operation once the temperature is smooth over a
pixel, and it costs one pass over the interface set rather than a second geometry pass.

Coverage is explicit. Every temperature sample carries a flag; a point that needs a temperature
(its material has a non-zero coefficient) and has none is an error by default, or is left
unperturbed and counted under ``uncovered="unperturbed"``. Nothing is ever silently zero.

Scope of this first version, refused with an error otherwise: isotropic, non-dispersive materials
on the 3-component permittivity tier, with the off-diagonal entries either absent or on the vertex
lattice (``yee_smooth_offdiag_placement="node"``). Conductivity, permeability and dispersive poles
are not perturbed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from fdtdx.core.physics.geometry_raster import _material_signature
from fdtdx.core.physics.geometry_smooth import (
    SmoothingRecord,
    _isotropic_offdiagonal_entries,
    kottke_inverse_permittivity,
)
from fdtdx.coupling.fem_field import YeeLatticeSamples
from fdtdx.materials import Material

#: Relative tolerance under which a material counts as isotropic.
_ISOTROPY_TOL = 1e-12


@dataclass(frozen=True)
class ThermoOpticCoefficients:
    """``dn/dT`` per material name, and the temperature the scene's indices refer to.

    Attributes:
        dn_dT (Mapping[str, float]): Thermo-optic coefficient in 1/K per material *name* (the key
            of the scene's material dictionary). A material not listed is not perturbed.
        reference_temperature (float): The temperature, in the field's unit, at which the scene's
            permittivities hold. The perturbation uses ``T - reference_temperature``.
    """

    dn_dT: Mapping[str, float]
    reference_temperature: float = 293.15

    def table(
        self, materials: Mapping[str, Material], material_table: Sequence[Material]
    ) -> tuple[np.ndarray, dict[int, str]]:
        """``(M,)`` coefficient per scene material-table entry, zero for unmatched entries.

        The loader's table names a uniform object's material after the object, not after the key
        the user gave it, so entries are matched to the user's dictionary by material *value*
        (permittivity, permeability, conductivities, dispersion), never by name.

        Args:
            materials (Mapping[str, Material]): The user's material dictionary.
            material_table (Sequence[Material]): The scene's materials in table order.

        Returns:
            tuple: ``(coefficients, matched)`` with ``matched`` mapping table index to user name.

        Raises:
            KeyError: If a coefficient names a material absent from ``materials``.
            ValueError: If two user names with one material value carry different coefficients.
        """
        unknown = set(self.dn_dT) - set(materials)
        if unknown:
            raise KeyError(f"thermo-optic coefficients name materials absent from the scene: {sorted(unknown)}")
        by_signature: dict[tuple, tuple[str, float]] = {}
        for name, value in self.dn_dT.items():
            signature = _material_signature(materials[name])
            previous = by_signature.get(signature)
            if previous is not None and previous[1] != float(value):
                raise ValueError(
                    f"materials {previous[0]!r} and {name!r} have the same value but different thermo-optic "
                    f"coefficients ({previous[1]:g} and {float(value):g}); the loader treats them as one material"
                )
            by_signature[signature] = (name, float(value))
        table = np.zeros(len(material_table), dtype=np.float64)
        matched: dict[int, str] = {}
        for index, material in enumerate(material_table):
            hit = by_signature.get(_material_signature(material))
            if hit is not None:
                matched[index] = hit[0]
                table[index] = hit[1]
        return table, matched


@dataclass
class ThermoOpticReport:
    """What the perturbation touched, for a run's JSON record.

    Attributes:
        num_bulk_points (dict[str, int]): Per E lattice, points rewritten with the bulk formula.
        num_reblended (dict[str, int]): Per lattice (``E0``..``E2``, ``V``), blended entries
            re-evaluated at their temperature.
        num_uncovered (dict[str, int]): Per lattice, points that needed a temperature and had none.
        uncovered_policy (str): ``"error"`` or ``"unperturbed"``.
        max_delta_T (float): Largest ``|T - T_ref|`` over the points that were perturbed.
        max_delta_n (float): Largest ``|dn/dT (T - T_ref)|`` applied.
        reference_temperature (float): The coefficients' reference temperature.
        coefficients (dict[str, float]): The coefficients used, per material name.
    """

    num_bulk_points: dict[str, int] = field(default_factory=dict)
    num_reblended: dict[str, int] = field(default_factory=dict)
    num_uncovered: dict[str, int] = field(default_factory=dict)
    uncovered_policy: str = "error"
    max_delta_T: float = 0.0
    max_delta_n: float = 0.0
    reference_temperature: float = 0.0
    coefficients: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_bulk_points": dict(self.num_bulk_points),
            "num_reblended": dict(self.num_reblended),
            "num_uncovered": dict(self.num_uncovered),
            "uncovered_policy": self.uncovered_policy,
            "max_delta_T": float(self.max_delta_T),
            "max_delta_n": float(self.max_delta_n),
            "reference_temperature": float(self.reference_temperature),
            "coefficients": dict(self.coefficients),
        }


def perturbed_permittivity(permittivity: np.ndarray, dn_dT: np.ndarray, delta_T: np.ndarray) -> np.ndarray:
    """``(sqrt(eps) + dn/dT * dT)^2``, elementwise.

    Args:
        permittivity (np.ndarray): Unperturbed relative permittivity (positive).
        dn_dT (np.ndarray): Coefficient, 1/K.
        delta_T (np.ndarray): ``T - T_ref``, K.

    Returns:
        np.ndarray: The perturbed relative permittivity.
    """
    return (np.sqrt(permittivity) + dn_dT * delta_T) ** 2


def _scalar_permittivities(material_table: Sequence[Material], names: Sequence[str], active: np.ndarray) -> np.ndarray:
    """``(M,)`` isotropic permittivity per material; refuses anisotropic or dispersive active materials."""
    table = np.zeros(len(material_table), dtype=np.float64)
    for index, material in enumerate(material_table):
        name = names[index]
        eps = np.asarray(material.permittivity, dtype=np.float64).reshape(3, 3)
        diag = np.diag(eps)
        off = eps - np.diag(diag)
        isotropic = (
            np.ptp(diag) <= _ISOTROPY_TOL * max(np.max(np.abs(diag)), 1.0) and np.max(np.abs(off)) <= _ISOTROPY_TOL
        )
        if active[index]:
            if not isotropic:
                raise NotImplementedError(
                    f"material {name!r} is anisotropic; the thermo-optic perturbation handles isotropic materials only"
                )
            if material.is_dispersive:
                raise NotImplementedError(
                    f"material {name!r} is dispersive; the thermo-optic perturbation handles the static permittivity only"
                )
            if diag[0] <= 0.0:
                raise ValueError(f"material {name!r} has non-positive permittivity {diag[0]}; no index to perturb")
        table[index] = diag[0]
    return table


def _check_class_consistency(record: SmoothingRecord, dn: np.ndarray, names: Sequence[str]) -> None:
    """Materials the loader merged into one value class must share one coefficient."""
    classes = record.classes.get("permittivity")
    if classes is None:
        return
    for representative in np.unique(classes):
        members = np.nonzero(classes == representative)[0]
        coefficients = dn[members]
        if np.ptp(coefficients) > 0.0:
            listing = ", ".join(f"{names[m]}: {dn[m]:g}" for m in members)
            raise ValueError(
                "materials with identical permittivity were blended as one class but carry different "
                f"thermo-optic coefficients ({listing}); give them one coefficient or distinct permittivities"
            )


def _sample_at(samples: YeeLatticeSamples, lattice: str, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Temperature and coverage at listed ``(K, 3)`` cell indices of one lattice."""
    if lattice not in samples.values:
        raise KeyError(f"temperature samples carry no lattice {lattice!r}; have {samples.lattices}")
    index = (cells[:, 0], cells[:, 1], cells[:, 2])
    return samples.values[lattice][index], samples.covered[lattice][index]


def apply_thermo_optic_perturbation(
    inv_permittivities: np.ndarray,
    inv_permittivity_offdiag: np.ndarray | None,
    material_map: Mapping[str, Any],
    materials: Mapping[str, Material],
    temperature: YeeLatticeSamples,
    coefficients: ThermoOpticCoefficients,
    uncovered: str = "error",
) -> tuple[np.ndarray, np.ndarray | None, ThermoOpticReport]:
    """Perturb the loader's inverse-permittivity arrays with a sampled temperature field.

    Args:
        inv_permittivities (np.ndarray): ``(3, Nx, Ny, Nz)`` diagonal-tier inverse permittivity on
            the E lattices, as the loader (and the absorber extension) left it.
        inv_permittivity_offdiag (np.ndarray | None): ``(3, Nx, Ny, Nz)`` vertex off-diagonal
            entries ``(xy, xz, yz)``, or ``None``.
        material_map (Mapping): ``info["yee_material_map"]`` from ``place_objects``: ``front_E``,
            ``material_names``, ``smoothing_record``, ``num_perm_components``, ``offdiag_placement``.
        materials (Mapping[str, Material]): The scene's material dictionary, keyed by name.
        temperature (YeeLatticeSamples): Sampled on the same grid; must carry ``E0``..``E2`` and,
            when off-diagonal entries exist, ``V``.
        coefficients (ThermoOpticCoefficients): ``dn/dT`` per material name and ``T_ref``.
        uncovered (str): ``"error"`` (a point needing a temperature without one raises) or
            ``"unperturbed"`` (it keeps its unperturbed value and is counted).

    Returns:
        tuple: ``(inv_permittivities, inv_permittivity_offdiag, report)`` -- new arrays (float64),
        the input arrays are not modified.

    Raises:
        NotImplementedError: For the 9-component tier, a non-vertex off-diagonal placement, an
            anisotropic or dispersive perturbed material, or a tensor blend involving one.
        ValueError: For a grid mismatch, an unknown material, an inconsistent value class, or an
            uncovered point under the ``"error"`` policy.
    """
    if uncovered not in ("error", "unperturbed"):
        raise ValueError(f"uncovered must be 'error' or 'unperturbed', got {uncovered!r}")
    if int(material_map.get("num_perm_components", 3)) != 3:
        raise NotImplementedError("the thermo-optic perturbation supports the 3-component permittivity tier only")
    placement = material_map.get("offdiag_placement")
    if inv_permittivity_offdiag is not None and placement not in (None, "node"):
        raise NotImplementedError(
            f"off-diagonal placement {placement!r} is not supported; use 'node' (the vertex lattice) or none"
        )
    front_E = np.asarray(material_map["front_E"])
    names = tuple(material_map["material_names"])
    material_table = tuple(material_map["material_table"])
    record: SmoothingRecord | None = material_map.get("smoothing_record")
    inv_eps = np.array(inv_permittivities, dtype=np.float64, copy=True)
    if inv_eps.shape[0] != 3 or inv_eps.shape[1:] != front_E.shape[1:]:
        raise ValueError(f"inv_permittivities {inv_eps.shape} does not match front_E {front_E.shape}")
    offdiag = (
        None if inv_permittivity_offdiag is None else np.array(inv_permittivity_offdiag, dtype=np.float64, copy=True)
    )
    for lattice in ("E0", "E1", "E2"):
        if lattice not in temperature.values:
            raise ValueError(f"temperature samples lack lattice {lattice!r}")
        if temperature.values[lattice].shape != front_E.shape[1:]:
            raise ValueError(
                f"temperature lattice {lattice} has shape {temperature.values[lattice].shape}, "
                f"the material arrays have {front_E.shape[1:]}"
            )

    dn, matched = coefficients.table(materials, material_table)
    active = dn != 0.0
    eps_table = _scalar_permittivities(material_table, names, active)
    if record is not None:
        _check_class_consistency(record, dn, names)
    T_ref = float(coefficients.reference_temperature)

    report = ThermoOpticReport(
        uncovered_policy=uncovered,
        reference_temperature=T_ref,
        coefficients={user_name: float(dn[i]) for i, user_name in matched.items() if dn[i] != 0.0},
    )
    max_dT = 0.0
    max_dn = 0.0

    def _uncovered(lattice: str, count: int) -> None:
        report.num_uncovered[lattice] = report.num_uncovered.get(lattice, 0) + int(count)
        if count and uncovered == "error":
            raise ValueError(
                f"{count} point(s) on lattice {lattice} carry a material with a thermo-optic coefficient "
                "but lie outside the temperature mesh; extend the thermal domain or pass uncovered='unperturbed'"
            )

    # 1. Bulk points: every point whose sampled material is perturbed, with the closed form. This
    #    also overwrites the blended pixels, which step 2 puts right.
    for c in range(3):
        lattice = f"E{c}"
        material = front_E[c]
        needs = active[material]
        cov = temperature.covered[lattice]
        missing = needs & ~cov
        _uncovered(lattice, int(np.count_nonzero(missing)))
        # A point whose temperature equals the reference is left bit-for-bit alone: a zero
        # perturbation is the identity, not a round trip through sqrt and square.
        write = needs & cov & (temperature.values[lattice] != T_ref)
        count = int(np.count_nonzero(write))
        report.num_bulk_points[lattice] = count
        if count == 0:
            continue
        m = material[write]
        dT = temperature.values[lattice][write] - T_ref
        delta_n = dn[m] * dT
        inv_eps[c][write] = 1.0 / perturbed_permittivity(eps_table[m], dn[m], dT)
        max_dT = max(max_dT, float(np.max(np.abs(dT))))
        max_dn = max(max_dn, float(np.max(np.abs(delta_n))))

    # 2. Re-blend the recorded interface pixels and vertices at their own temperature.
    if record is not None:
        for entry in record.passes:
            if entry.field == "E":
                target_lattice = f"E{entry.component}"
            elif entry.field == "V":
                target_lattice = "V"
            else:
                continue  # H lattices: permeability is not perturbed
            hi, lo = entry.material_hi, entry.material_lo
            involved = active[hi] | active[lo]
            if not involved.any():
                continue
            if entry.write_mode == "row" and entry.full_tensor:
                raise NotImplementedError("re-blending the 9-component row tier is not supported")
            if entry.write_mode not in ("row", "offdiag"):
                raise NotImplementedError(f"re-blending write mode {entry.write_mode!r} is not supported")
            if not entry.isotropic_pair[involved].all():
                raise NotImplementedError(
                    f"lattice {target_lattice}: a tensor (anisotropic) blend involves a perturbed material"
                )
            cells = entry.cells[involved]
            T, cov = _sample_at(temperature, target_lattice, cells)
            _uncovered(target_lattice, int(np.count_nonzero(~cov)))
            keep = cov & (T != T_ref)
            if not keep.any():
                continue
            cells = cells[keep]
            dT = T[keep] - T_ref
            fill = entry.fill[involved][keep]
            normal = entry.normal[involved][keep]
            m_hi = hi[involved][keep]
            m_lo = lo[involved][keep]
            eps_hi = perturbed_permittivity(eps_table[m_hi], dn[m_hi], dT)
            eps_lo = perturbed_permittivity(eps_table[m_lo], dn[m_lo], dT)
            arithmetic = fill * eps_hi + (1.0 - fill) * eps_lo
            harmonic = fill / eps_hi + (1.0 - fill) / eps_lo
            index = (cells[:, 0], cells[:, 1], cells[:, 2])
            if entry.write_mode == "row":
                inv_eps[entry.component][index] = kottke_inverse_permittivity(
                    normal, arithmetic, harmonic, entry.component, False
                )
            else:
                if offdiag is None:
                    raise ValueError("the record holds vertex entries but no off-diagonal array was given")
                entries = _isotropic_offdiagonal_entries(normal, arithmetic, harmonic)
                for q in range(3):
                    offdiag[q][index] = entries[:, q]
            report.num_reblended[target_lattice] = report.num_reblended.get(target_lattice, 0) + int(cells.shape[0])
            max_dT = max(max_dT, float(np.max(np.abs(dT))))
            max_dn = max(max_dn, float(np.max(np.abs(np.maximum(np.abs(dn[m_hi]), np.abs(dn[m_lo])) * dT))))

    report.max_delta_T = max_dT
    report.max_delta_n = max_dn
    return inv_eps, offdiag, report


def perturb_arrays(
    arrays: Any,
    info: Mapping[str, Any],
    materials: Mapping[str, Material],
    temperature: YeeLatticeSamples,
    coefficients: ThermoOpticCoefficients,
    uncovered: str = "error",
) -> tuple[Any, ThermoOpticReport]:
    """Apply :func:`apply_thermo_optic_perturbation` to a placed ``ArrayContainer``.

    Reads ``arrays.inv_permittivities`` and ``arrays.inv_permittivity_offdiag`` (host copies), and
    returns a container with both replaced, in the container's own dtype. Call it after
    ``place_objects``, ``extend_material_to_pml`` and ``apply_params``: the absorber cells then
    carry the extended unperturbed material unless the temperature covers them too.

    Args:
        arrays: The ``ArrayContainer`` from placement.
        info (Mapping): The ``info`` dictionary ``place_objects`` returned (needs ``yee_material_map``).
        materials (Mapping[str, Material]): The scene's material dictionary.
        temperature (YeeLatticeSamples): Sampled on the placed grid.
        coefficients (ThermoOpticCoefficients): The perturbation model.
        uncovered (str): See :func:`apply_thermo_optic_perturbation`.

    Returns:
        tuple: ``(arrays, report)``.

    Raises:
        ValueError: If the placement did not use per-Yee-point sampling (no material map).
    """
    import jax.numpy as jnp

    material_map = info.get("yee_material_map")
    if material_map is None:
        raise ValueError(
            "place_objects info carries no 'yee_material_map': the thermo-optic perturbation needs "
            "material_sampling='yee' or 'yee_smooth'"
        )
    inv_eps = np.asarray(arrays.inv_permittivities)
    offdiag = None if arrays.inv_permittivity_offdiag is None else np.asarray(arrays.inv_permittivity_offdiag)
    new_inv, new_off, report = apply_thermo_optic_perturbation(
        inv_eps, offdiag, material_map, materials, temperature, coefficients, uncovered=uncovered
    )
    out = arrays.aset("inv_permittivities", jnp.asarray(new_inv, dtype=arrays.inv_permittivities.dtype))
    if new_off is not None:
        out = out.aset("inv_permittivity_offdiag", jnp.asarray(new_off, dtype=arrays.inv_permittivity_offdiag.dtype))
    return out, report
