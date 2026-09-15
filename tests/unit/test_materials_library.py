"""Unit tests for the fitted material library.

The library data is checked in, so these run without a refractiveindex.info
clone; the one test that cross-checks against the raw database record skips
when no clone is present.
"""

import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from fdtdx.dispersion import DebyePole, DrudePole
from fdtdx.dispersion_fit import check_stability, read_refractiveindex_yaml
from fdtdx.materials import Material
from fdtdx.materials_library import DATA_PATH, MATERIALS, MaterialRecord, get_material, list_materials

_DB_ENV = os.environ.get("FDTDX_REFRACTIVEINDEX_DB")
_DB_CANDIDATES = [
    Path(_DB_ENV) if _DB_ENV else None,
    Path.home() / "Projects/ReferenceSolvers/refractiveindex/database",
]
_DB = next((p for p in _DB_CANDIDATES if p is not None and (p / "data").is_dir()), None)

# Malitson 1965 fused silica, the public-domain coefficients behind the SiO2
# entry's source page. Kept here so the reference value does not need a clone.
_MALITSON_B = (0.6961663, 0.4079426, 0.8974794)
_MALITSON_SQRT_C_UM = (0.0684043, 0.1162414, 9.896161)


def _malitson_n(lam_um: float) -> float:
    n_sq = 1.0 + sum(b * lam_um**2 / (lam_um**2 - c**2) for b, c in zip(_MALITSON_B, _MALITSON_SQRT_C_UM, strict=True))
    return math.sqrt(n_sq)


class TestLibraryContents:
    def test_library_is_populated(self):
        names = list_materials()
        assert len(names) >= 15
        assert names == sorted(names)
        for expected in ("Si", "SiO2", "Si3N4", "Al2O3", "TiO2", "LiNbO3_o", "LiNbO3_e", "Ge", "InP", "GaAs"):
            assert expected in names, f"{expected} missing from the library"
        for metal in ("Au", "Ag", "Al", "Cu"):
            assert metal in names

    def test_every_entry_carries_provenance(self):
        for name, record in MATERIALS.items():
            assert record.name == name
            assert record.source_page.endswith(".yml")
            assert len(record.source_reference) > 20, f"{name} has no usable reference"
            lo, hi = record.wavelength_range_m
            assert 0.0 < lo < hi
            assert record.eps_inf >= 1.0
            assert record.poles, f"{name} has no poles"
            assert record.rms >= 0.0

    def test_metals_use_a_drude_pole(self):
        for metal in ("Au", "Ag", "Al", "Cu"):
            assert any(isinstance(p, DrudePole) for p in MATERIALS[metal].poles), metal

    def test_water_is_the_debye_example(self):
        record = MATERIALS["H2O_farIR"]
        assert any(isinstance(p, DebyePole) for p in record.poles)
        assert "Debye" in record.notes


class TestPassivityAndStability:
    def test_every_entry_is_passive_over_its_range(self):
        for name, record in MATERIALS.items():
            lo, hi = record.wavelength_range_m
            lam = np.linspace(lo, hi, 400)
            eps = np.array([record.permittivity(float(x)) for x in lam])
            assert np.min(eps.imag) >= -1e-9 * max(1.0, float(np.max(np.abs(eps)))), (
                f"{name} has Im(eps) < 0 inside its fitted range"
            )

    def test_every_entry_is_stable_at_a_typical_time_step(self):
        # 10 nm cells at Courant 0.99/sqrt(3) -> dt ~ 1.9e-17 s; use a coarser
        # 2e-17 s so the check is not accidentally generous.
        for name, record in MATERIALS.items():
            ok, msg = check_stability(record.dispersion, dt=2e-17)
            assert ok, f"{name}: {msg}"


class TestGetMaterial:
    def test_sio2_permittivity_matches_the_database_record(self):
        expected = _malitson_n(1.55) ** 2
        eps = MATERIALS["SiO2"].permittivity(1550e-9)
        assert eps.real == pytest.approx(expected, abs=1e-3)
        assert eps.imag == pytest.approx(0.0, abs=1e-12)
        # ... and through the Material the user actually gets
        material = get_material("SiO2", wavelength=1550e-9)
        assert material.permittivity[0] == pytest.approx(expected, abs=1e-3)

    @pytest.mark.skipif(_DB is None, reason="no refractiveindex.info database clone found")
    def test_sio2_matches_the_raw_record_from_the_clone(self):
        assert _DB is not None
        record = MATERIALS["SiO2"]
        _lam, n, _k = read_refractiveindex_yaml(_DB / "data" / record.source_page, wavelengths_m=np.array([1.55e-6]))
        assert record.permittivity(1.55e-6).real == pytest.approx(float(n[0]) ** 2, abs=1e-3)

    def test_dispersive_material_by_default(self):
        material = get_material("SiO2")
        assert isinstance(material, Material)
        assert material.is_dispersive
        assert material.dispersion is not None
        assert material.permittivity[0] == pytest.approx(MATERIALS["SiO2"].eps_inf)

    def test_constant_material_carries_loss_as_conductivity(self):
        record = MATERIALS["Si_visible"]
        lam = 0.6e-6
        eps = record.permittivity(lam)
        assert eps.imag > 0.0, "test premise: silicon absorbs in the visible"
        material = get_material("Si_visible", wavelength=lam)
        assert not material.is_dispersive
        assert material.permittivity[0] == pytest.approx(eps.real)
        omega = 2.0 * math.pi * 299792458.0 / lam
        expected_sigma = omega * 8.8541878128e-12 * eps.imag
        assert material.electric_conductivity[0] == pytest.approx(expected_sigma, rel=1e-6)

    def test_aliases_are_case_insensitive(self):
        for alias in ("silica", "Fused Silica", "GLASS"):
            assert get_material(alias).permittivity[0] == pytest.approx(MATERIALS["SiO2"].eps_inf)
        assert get_material("gold").dispersion is not None

    def test_unknown_material_raises(self):
        with pytest.raises(KeyError, match="Unknown material"):
            get_material("unobtainium")

    def test_metal_refuses_a_constant_permittivity(self):
        with pytest.raises(ValueError, match="unconditionally unstable"):
            get_material("Au", wavelength=1.55e-6)

    def test_out_of_range_wavelength_warns(self):
        with pytest.warns(UserWarning, match="outside that range"):
            get_material("Si", wavelength=0.5e-6)

    def test_refractive_index_helper(self):
        assert MATERIALS["SiO2"].refractive_index(1.55e-6).real == pytest.approx(_malitson_n(1.55), abs=1e-3)


class TestJsonRoundTrip:
    def test_record_round_trips(self):
        for name, record in MATERIALS.items():
            again = MaterialRecord.from_json(json.loads(json.dumps(record.to_json())))
            assert again == record, f"{name} did not round-trip"

    def test_file_round_trips(self):
        with DATA_PATH.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        assert doc["schema"] == 1
        assert "CC0" in doc["source"]
        rebuilt = {name: MaterialRecord.from_json(entry) for name, entry in doc["materials"].items()}
        assert rebuilt == MATERIALS

    def test_every_pole_kind_serialises(self):
        kinds = {p["kind"] for record in MATERIALS.values() for p in record.to_json()["poles"]}
        assert {"sellmeier", "lorentz", "drude", "debye"} <= kinds


def test_library_material_runs_through_the_coefficient_builder():
    """A library material must survive the engine's setup path unchanged."""
    from fdtdx.materials import compute_allowed_dispersive_coefficients, compute_max_dispersive_poles

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no permittivity warnings from the library entries
        mats = {"air": Material(permittivity=1.0), "sio2": get_material("SiO2"), "au": get_material("Au")}
    max_poles = compute_max_dispersive_poles(mats)
    assert max_poles >= 2
    c1, c2, c3 = compute_allowed_dispersive_coefficients(
        mats, dt=2e-17, max_num_poles=max_poles, num_components=1, coupling_components=1
    )
    assert np.all(np.isfinite(c1)) and np.all(np.isfinite(c2)) and np.all(np.isfinite(c3))
