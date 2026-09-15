#!/usr/bin/env python
"""Rebuild ``src/fdtdx/data/materials_library.json`` from a refractiveindex.info clone.

The refractiveindex.info database is dedicated to the public domain (CC0 1.0),
so its numbers may be used freely. The pole models written here are *our* fits
of that data, produced by :mod:`fdtdx.dispersion_fit` — no coefficients are
taken from another simulator's material file.

Usage::

    python scripts/build_material_library.py --database /path/to/refractiveindex/database
    python scripts/build_material_library.py --only Si,SiO2 --dry-run

A page missing from the clone is skipped with a message rather than failing the
build, so a partial clone still produces a usable library.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fdtdx.constants import c as c_light  # noqa: E402
from fdtdx.dispersion_fit import fit_dispersion, read_refractiveindex_yaml  # noqa: E402
from fdtdx.materials_library import DATA_PATH, MaterialRecord  # noqa: E402

DEFAULT_DB = Path(
    os.environ.get("FDTDX_REFRACTIVEINDEX_DB", Path.home() / "Projects/ReferenceSolvers/refractiveindex/database")
)


@dataclass(frozen=True)
class Entry:
    """One material to fit: where the data is, over what band, and with what poles."""

    name: str
    page: str
    lam_min: float
    lam_max: float
    num_poles: int
    kinds: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    notes: str = ""
    num_points: int = 120
    max_starts: int = 10


#: The first set. Bands are chosen where each material is actually used in
#: photonics and where the source record is valid; a lossless (Sellmeier) fit is
#: used inside a transparency window, where the data cannot constrain a damping
#: rate, and Drude+Lorentz for the metals.
ENTRIES: tuple[Entry, ...] = (
    Entry(
        name="Si",
        page="main/Si/nk/Salzberg.yml",
        lam_min=1.36e-6,
        lam_max=1.7e-6,
        num_poles=2,
        kinds=("sellmeier",),
        aliases=("silicon", "c-Si"),
        notes="Crystalline silicon at 26 C, telecom band. Transparent here; absorption is not modelled.",
    ),
    Entry(
        name="Si_visible",
        page="main/Si/nk/Green-2008.yml",
        lam_min=0.5e-6,
        lam_max=1.1e-6,
        num_poles=3,
        kinds=("lorentz",),
        aliases=("silicon_visible",),
        notes="Crystalline silicon at 300 K including band-to-band absorption; visible to near-IR.",
    ),
    Entry(
        name="SiO2",
        page="main/SiO2/nk/Malitson.yml",
        lam_min=0.4e-6,
        lam_max=2.0e-6,
        num_poles=2,
        kinds=("sellmeier",),
        aliases=("silica", "fused silica", "glass", "oxide"),
        notes="Fused silica at 20 C.",
    ),
    Entry(
        name="Si3N4",
        page="main/Si3N4/nk/Luke.yml",
        lam_min=0.4e-6,
        lam_max=2.0e-6,
        num_poles=2,
        kinds=("sellmeier",),
        aliases=("silicon nitride", "nitride"),
        notes="Stoichiometric LPCVD silicon nitride.",
    ),
    Entry(
        name="Al2O3",
        page="main/Al2O3/nk/Malitson-o.yml",
        lam_min=0.4e-6,
        lam_max=2.0e-6,
        num_poles=2,
        kinds=("sellmeier",),
        aliases=("sapphire", "alumina", "sapphire_o"),
        notes="Sapphire, ordinary ray. The extraordinary ray differs by ~0.008 in n.",
    ),
    Entry(
        name="TiO2",
        page="main/TiO2/nk/Devore-o.yml",
        lam_min=0.45e-6,
        lam_max=1.5e-6,
        num_poles=2,
        kinds=("sellmeier",),
        aliases=("titania", "rutile", "TiO2_o"),
        notes="Rutile TiO2, ordinary ray.",
    ),
    Entry(
        name="LiNbO3_o",
        page="main/LiNbO3/nk/Zelmon-o.yml",
        lam_min=0.5e-6,
        lam_max=4.0e-6,
        num_poles=3,
        kinds=("sellmeier",),
        aliases=("lithium niobate", "LiNbO3", "LN_o"),
        notes="Congruent LiNbO3, ordinary ray (n_o). Pair with LiNbO3_e for the uniaxial tensor.",
    ),
    Entry(
        name="LiNbO3_e",
        page="main/LiNbO3/nk/Zelmon-e.yml",
        lam_min=0.5e-6,
        lam_max=4.0e-6,
        num_poles=3,
        kinds=("sellmeier",),
        aliases=("LN_e",),
        notes="Congruent LiNbO3, extraordinary ray (n_e).",
    ),
    Entry(
        name="Ge",
        page="main/Ge/nk/Burnett.yml",
        lam_min=2.0e-6,
        lam_max=14.0e-6,
        num_poles=2,
        kinds=("sellmeier",),
        aliases=("germanium",),
        notes="Germanium in its mid-IR transparency window; opaque below ~1.9 um.",
    ),
    Entry(
        name="InP",
        page="main/InP/nk/Pettit.yml",
        lam_min=1.0e-6,
        lam_max=4.0e-6,
        num_poles=2,
        kinds=("sellmeier",),
        aliases=("indium phosphide",),
        notes="InP above the band edge (transparent); the record is a lossless Sellmeier formula.",
    ),
    Entry(
        name="GaAs",
        page="main/GaAs/nk/Skauli.yml",
        lam_min=1.0e-6,
        lam_max=6.0e-6,
        num_poles=3,
        kinds=("sellmeier",),
        aliases=("gallium arsenide",),
        notes="GaAs at 22 C in its transparency window.",
    ),
    Entry(
        name="Au",
        page="main/Au/nk/Johnson.yml",
        lam_min=0.6e-6,
        lam_max=1.6e-6,
        num_poles=2,
        kinds=("drude", "lorentz"),
        aliases=("gold",),
        notes="Johnson & Christy gold, NIR only. The interband transitions below ~0.6 um are not modelled.",
        max_starts=12,
    ),
    Entry(
        name="Ag",
        page="main/Ag/nk/Johnson.yml",
        lam_min=0.4e-6,
        lam_max=1.6e-6,
        num_poles=2,
        kinds=("drude", "lorentz"),
        aliases=("silver",),
        notes="Johnson & Christy silver.",
        max_starts=12,
    ),
    Entry(
        name="Al",
        page="main/Al/nk/Rakic-LD.yml",
        lam_min=0.4e-6,
        lam_max=2.0e-6,
        num_poles=3,
        kinds=("drude", "lorentz"),
        aliases=("aluminium", "aluminum"),
        notes="Rakic Lorentz-Drude aluminium, refitted from the tabulated n/k on that page.",
        max_starts=12,
    ),
    Entry(
        name="Cu",
        page="main/Cu/nk/Johnson.yml",
        lam_min=0.6e-6,
        lam_max=1.6e-6,
        num_poles=2,
        kinds=("drude", "lorentz"),
        aliases=("copper",),
        notes="Johnson & Christy copper, NIR only (interband absorption below ~0.6 um is not modelled).",
        max_starts=12,
    ),
    Entry(
        name="H2O",
        page="main/H2O/nk/Hale.yml",
        lam_min=0.4e-6,
        lam_max=1.6e-6,
        num_poles=2,
        kinds=("sellmeier",),
        aliases=("water",),
        notes=(
            "Liquid water at 25 C in the visible / near-IR, where it is effectively transparent "
            "(k < 1e-4). Use H2O_farIR for the absorbing far-IR response."
        ),
    ),
    Entry(
        name="H2O_farIR",
        page="main/H2O/nk/Hale.yml",
        lam_min=15e-6,
        lam_max=200e-6,
        num_poles=4,
        kinds=("debye", "lorentz"),
        aliases=("water_farIR", "water_thz"),
        notes=(
            "Liquid water at 25 C from the far-IR into the sub-THz; the library's Debye example. "
            "The Debye pole carries the relaxation background that the librational Lorentz bands sit on. "
            "Its fitted relaxation time is an effective far-IR one, not the ~8 ps microwave relaxation, "
            "which lies outside this band."
        ),
        num_points=160,
        max_starts=14,
    ),
    Entry(
        name="PMMA",
        page="organic/(C5H8O2)n - poly(methyl methacrylate)/nk/Sultanova.yml",
        lam_min=0.44e-6,
        lam_max=1.05e-6,
        num_poles=1,
        kinds=("sellmeier",),
        aliases=("poly(methyl methacrylate)", "acrylic", "resist"),
        notes="PMMA at 20 C, Sellmeier fit of the Sultanova data.",
    ),
)


def _reference(path: Path) -> str:
    """First sentence(s) of the record's REFERENCES block, stripped of HTML."""
    import re

    import yaml

    with path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    raw = str(doc.get("REFERENCES", "")).strip()
    text = re.sub(r"<[^>]+>", "", raw)
    text = " ".join(text.split())
    return text[:400]


def build_entry(entry: Entry, database: Path) -> MaterialRecord | None:
    """Fit one entry; returns ``None`` when the page is missing from the clone."""
    path = database / "data" / entry.page
    if not path.is_file():
        print(f"  SKIP {entry.name}: {entry.page} not in the clone")
        return None

    lam_all, n_all, k_all = read_refractiveindex_yaml(path, num_points=max(entry.num_points, 60))
    sel = (lam_all >= entry.lam_min) & (lam_all <= entry.lam_max)
    lam, n, k = lam_all[sel], n_all[sel], k_all[sel]
    if lam.size < 2 * (1 + 3 * entry.num_poles):
        # A formula record was sampled over its whole validity range; re-sample
        # inside the requested band so the fit is not starved of points.
        lam = np.linspace(
            max(entry.lam_min, float(lam_all.min())), min(entry.lam_max, float(lam_all.max())), entry.num_points
        )
        lam, n, k = read_refractiveindex_yaml(path, wavelengths_m=lam)
    if lam.size < 4:
        print(f"  SKIP {entry.name}: only {lam.size} points inside {entry.lam_min:.3g}-{entry.lam_max:.3g} m")
        return None

    result = fit_dispersion(
        lam,
        n,
        k,
        num_poles=entry.num_poles,
        kinds=entry.kinds,
        max_starts=entry.max_starts,
        seed=0,
    )
    omega = 2.0 * math.pi * c_light / lam
    eps_fit = np.array([result.eps_inf + result.model.susceptibility(float(w)) for w in omega])
    n_err = float(np.max(np.abs(np.sqrt(eps_fit) - (n + 1j * k))))
    print(
        f"  {entry.name:<11s} {lam.size:>4d} pts  {entry.lam_min * 1e6:.3g}-{entry.lam_max * 1e6:.3g} um  "
        f"rms(eps)={result.rms:.3g}  max|dn|={n_err:.3g}  passive={result.passive}"
    )
    if not result.passive:
        print(f"    WARNING: {entry.name} fit is not passive")

    return MaterialRecord(
        name=entry.name,
        aliases=entry.aliases,
        source_page=entry.page,
        source_reference=_reference(path),
        wavelength_range_m=(float(lam.min()), float(lam.max())),
        eps_inf=result.eps_inf,
        poles=result.model.poles,
        rms=result.rms,
        notes=entry.notes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB, help="path to a refractiveindex.info clone")
    parser.add_argument("--output", type=Path, default=DATA_PATH, help="where to write the JSON")
    parser.add_argument("--only", type=str, default="", help="comma-separated subset of entry names to build")
    parser.add_argument("--dry-run", action="store_true", help="fit and report, but do not write the file")
    args = parser.parse_args()

    if not (args.database / "data").is_dir():
        parser.error(f"{args.database} does not look like a refractiveindex.info clone (no data/ directory)")

    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    entries = [e for e in ENTRIES if not wanted or e.name in wanted]
    print(f"Fitting {len(entries)} material(s) from {args.database}")

    records: dict[str, MaterialRecord] = {}
    for entry in entries:
        record = build_entry(entry, args.database)
        if record is not None:
            records[record.name] = record

    doc = {
        "schema": 1,
        "generated_by": "scripts/build_material_library.py",
        "source": "refractiveindex.info database (CC0 1.0). Fits produced by fdtdx.dispersion_fit.",
        "materials": {name: records[name].to_json() for name in sorted(records)},
    }
    if args.dry_run:
        print(f"(dry run) {len(records)} entries, not written")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"Wrote {len(records)} entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
