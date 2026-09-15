"""A library of ready-to-use dispersive materials.

Each entry is a pole model fitted by :mod:`fdtdx.dispersion_fit` to a public
record from the `refractiveindex.info <https://refractiveindex.info>`_
database, whose data is dedicated to the public domain under CC0 1.0. The fits
are ours: nothing is copied from another simulator's material file. Every entry
records which page it came from, the paper behind that page, the wavelength
range it was fitted over and the residual of the fit, so a result can be traced
back to a measurement.

Usage::

    import fdtdx

    si = fdtdx.get_material("Si")                      # dispersive
    si_1550 = fdtdx.get_material("Si", wavelength=1.55e-6)  # constant at 1.55 um
    fdtdx.list_materials()

The data lives in ``src/fdtdx/data/materials_library.json`` and is regenerated
by ``scripts/build_material_library.py``; this module only loads it.

Limits
------
A fit is valid **only inside its ``wavelength_range_m``** — a pole model
extrapolates confidently and wrongly outside the band it was fitted over.
:func:`get_material` warns when a requested wavelength falls outside. Every
entry is isotropic; uniaxial crystals appear as separate ordinary and
extraordinary entries (combine them yourself with a per-axis or oriented pole
if you need the tensor).
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from fdtdx.constants import c as c_light
from fdtdx.constants import eps0
from fdtdx.dispersion import DebyePole, DispersionModel, DrudePole, LorentzPole, Pole, SellmeierPole
from fdtdx.materials import Material

#: Location of the checked-in library data.
DATA_PATH = Path(__file__).parent / "data" / "materials_library.json"


def _pole_to_json(p: Pole) -> dict[str, Any]:
    """Serialize one pole. Kept next to :func:`_pole_from_json` — they must agree."""
    if isinstance(p, SellmeierPole):
        return {"kind": "sellmeier", "B": float(p.B), "C": float(p.C)}
    if isinstance(p, LorentzPole):
        return {
            "kind": "lorentz",
            "resonance_frequency": float(p.resonance_frequency),
            "damping": float(p.damping),
            "delta_epsilon": float(p.delta_epsilon),
        }
    if isinstance(p, DrudePole):
        return {"kind": "drude", "plasma_frequency": float(p.plasma_frequency), "damping": float(p.damping)}
    if isinstance(p, DebyePole):
        return {
            "kind": "debye",
            "delta_epsilon": float(p.delta_epsilon),
            "relaxation_time": float(p.relaxation_time),
        }
    raise TypeError(f"Cannot serialize pole of type {type(p).__name__} into the material library.")


def _pole_from_json(entry: dict[str, Any]) -> Pole:
    kind = entry["kind"]
    if kind == "sellmeier":
        return SellmeierPole(B=entry["B"], C=entry["C"])
    if kind == "lorentz":
        return LorentzPole(
            resonance_frequency=entry["resonance_frequency"],
            damping=entry["damping"],
            delta_epsilon=entry["delta_epsilon"],
        )
    if kind == "drude":
        return DrudePole(plasma_frequency=entry["plasma_frequency"], damping=entry["damping"])
    if kind == "debye":
        return DebyePole(delta_epsilon=entry["delta_epsilon"], relaxation_time=entry["relaxation_time"])
    raise ValueError(f"Unknown pole kind {kind!r} in the material library data.")


@dataclass(frozen=True)
class MaterialRecord:
    """One fitted material, with the provenance of the data behind it.

    Attributes:
        name: Canonical name, e.g. ``"SiO2"``.
        aliases: Other names :func:`get_material` accepts for this entry.
        source_page: Path of the record inside the refractiveindex.info
            database, e.g. ``"main/SiO2/nk/Malitson.yml"``.
        source_reference: The measurement or formula the page reports.
        wavelength_range_m: ``(lambda_min, lambda_max)`` the fit is valid over.
        eps_inf: High-frequency permittivity of the fitted model.
        poles: The fitted poles.
        rms: Root-mean-square :math:`|\\Delta\\varepsilon|` of the fit over its
            range.
        notes: Free-form remarks (temperature, crystal axis, what the fit does
            and does not capture).
    """

    name: str
    aliases: tuple[str, ...]
    source_page: str
    source_reference: str
    wavelength_range_m: tuple[float, float]
    eps_inf: float
    poles: tuple[Pole, ...]
    rms: float
    notes: str = ""

    @property
    def dispersion(self) -> DispersionModel:
        """The entry's poles as a :class:`~fdtdx.dispersion.DispersionModel`."""
        return DispersionModel(poles=self.poles)

    def permittivity(self, wavelength: float) -> complex:
        """Complex relative permittivity at a vacuum wavelength (metres).

        Args:
            wavelength: Vacuum wavelength in metres.

        Returns:
            complex: :math:`\\varepsilon(\\omega)` in the ``exp(-i omega t)``
            convention, so a lossy material has a positive imaginary part.
        """
        omega = 2.0 * math.pi * c_light / wavelength
        return self.dispersion.permittivity(omega, eps_inf=self.eps_inf)

    def refractive_index(self, wavelength: float) -> complex:
        """Complex refractive index :math:`n + ik` at a vacuum wavelength (metres).

        Args:
            wavelength: Vacuum wavelength in metres.

        Returns:
            complex: ``sqrt(permittivity(wavelength))``.
        """
        import numpy as np

        return complex(np.sqrt(self.permittivity(wavelength)))

    def to_json(self) -> dict[str, Any]:
        """Serialize to the plain-dict form stored in the library JSON."""
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "source_page": self.source_page,
            "source_reference": self.source_reference,
            "wavelength_range_m": list(self.wavelength_range_m),
            "eps_inf": self.eps_inf,
            "poles": [_pole_to_json(p) for p in self.poles],
            "rms": self.rms,
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, entry: dict[str, Any]) -> "MaterialRecord":
        """Rebuild a record from the plain-dict form stored in the library JSON.

        Args:
            entry: One value of the JSON file's ``"materials"`` mapping.

        Returns:
            MaterialRecord: The reconstructed record.
        """
        return cls(
            name=entry["name"],
            aliases=tuple(entry.get("aliases", ())),
            source_page=entry["source_page"],
            source_reference=entry["source_reference"],
            wavelength_range_m=(float(entry["wavelength_range_m"][0]), float(entry["wavelength_range_m"][1])),
            eps_inf=float(entry["eps_inf"]),
            poles=tuple(_pole_from_json(p) for p in entry["poles"]),
            rms=float(entry["rms"]),
            notes=entry.get("notes", ""),
        )


def _load_library(path: Path) -> dict[str, MaterialRecord]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    return {name: MaterialRecord.from_json(entry) for name, entry in doc.get("materials", {}).items()}


#: The library, keyed by canonical name. Empty if the data file is missing.
MATERIALS: dict[str, MaterialRecord] = _load_library(DATA_PATH)


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for name, record in MATERIALS.items():
        for key in (name, *record.aliases):
            index[key.casefold()] = name
    return index


def list_materials() -> list[str]:
    """Canonical names of every material in the library, sorted.

    Returns:
        list[str]: The keys of :data:`MATERIALS`. Use ``MATERIALS[name]`` for
        the full record (provenance, range, fit residual).
    """
    return sorted(MATERIALS)


def get_material(name: str, wavelength: float | None = None) -> Material:
    """Build a :class:`~fdtdx.materials.Material` from the library.

    Args:
        name: Canonical name or alias, matched case-insensitively.
        wavelength: ``None`` (default) returns the **dispersive** material —
            ``permittivity = eps_inf`` plus the fitted
            :class:`~fdtdx.dispersion.DispersionModel`, valid across the
            entry's whole range. A wavelength in metres instead returns a
            **non-dispersive** material frozen at that wavelength:
            ``permittivity = Re eps(omega)`` with the absorption carried as an
            equivalent conductivity ``sigma = omega eps0 Im eps(omega)``, which
            is the cheaper choice for a narrowband simulation.

    Returns:
        Material: Ready to attach to a simulation object.

    Raises:
        KeyError: If ``name`` is not in the library.
        ValueError: If a single-wavelength material is requested where
            ``Re eps < 0`` (a metal in its plasmonic band): a negative constant
            permittivity is unconditionally unstable in explicit FDTD, so the
            dispersive form is the only correct one there.
    """
    key = _alias_index().get(name.casefold())
    if key is None:
        raise KeyError(f"Unknown material {name!r}. Available: {', '.join(list_materials())}")
    record = MATERIALS[key]

    if wavelength is None:
        return Material(permittivity=record.eps_inf, dispersion=record.dispersion)

    lo, hi = record.wavelength_range_m
    if not (lo <= wavelength <= hi):
        warnings.warn(
            f"{record.name} was fitted over {lo * 1e6:.4g}-{hi * 1e6:.4g} um; "
            f"{wavelength * 1e6:.4g} um is outside that range and the pole model does not "
            "extrapolate reliably.",
            UserWarning,
            stacklevel=2,
        )
    eps = record.permittivity(wavelength)
    if eps.real <= 0.0:
        raise ValueError(
            f"{record.name} has Re(eps) = {eps.real:.4g} <= 0 at {wavelength * 1e6:.4g} um, which is "
            "unconditionally unstable as a constant permittivity in explicit FDTD. Use "
            f"get_material({record.name!r}) without a wavelength to get the dispersive model."
        )
    omega = 2.0 * math.pi * c_light / wavelength
    sigma = omega * eps0 * eps.imag
    return Material(permittivity=eps.real, electric_conductivity=max(sigma, 0.0))
