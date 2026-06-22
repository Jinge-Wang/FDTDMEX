"""Validation for modal decomposition at a monitor (the PIC transmission analysis).

Contract:
* Decomposing the through-port field of a straight single-mode Si waveguide puts almost all power in
  the fundamental TE0 mode, and the per-mode power fractions sum to 1.
* The TE0 transmission ``|α_TE0 / α_in|²`` equals the through detector's own |overlap|² (the same
  quantity ``calculate_sparam`` reports) — i.e. the decomposition is self-consistent with the
  ModeOverlapDetector.
* The mode cache round-trips: a second decomposition reuses the cached modes (and a different
  cross-section is rejected by the fingerprint check).
"""

from __future__ import annotations

import numpy as np
import pytest

import fdtdx
from fdtdx.fdtd.stop_conditions import EnergyThresholdCondition
from fdtdx.objects.static_material.polygon import ExtrudedPolygon
from fdtdx.utils.mode_expansion import _cross_section_fingerprint, _load_mode_cache, compute_mode_expansion
from fdtdx.utils.sparams import PortSpec, determine_input_norm_detector_name, setup_sparams_simulation

pytestmark = pytest.mark.validation


def _straight_bus():
    """A short straight Si strip waveguide (extruded along x), single-mode at 1.55 µm."""
    wg, slab = 0.40e-6, 0.22e-6
    verts = np.array([[-wg / 2, -slab / 2], [wg / 2, -slab / 2], [wg / 2, slab / 2], [-wg / 2, slab / 2]])
    bus = ExtrudedPolygon(
        vertices=verts, axis=0, material_name="si",
        materials={"si": fdtdx.Material(permittivity=12.25), "air": fdtdx.Material(permittivity=1.0)},
        partial_real_shape=(4.0e-6, None, None),
    )
    return bus


def _run_and_decompose(modes, cache_path=None):
    res, LX, LY, LZ = 90e-9, 4.0e-6, 1.6e-6, 0.66e-6
    yb = LY / 2
    bus = _straight_bus()
    o, a, c = setup_sparams_simulation(
        polygons=[(bus, (LX / 2, yb, LZ / 2))],
        input_ports=[PortSpec(center=(0.7e-6, yb, LZ / 2), axis=0, direction="+", width=1.2e-6, height=0.5e-6, name="in")],
        output_ports=[PortSpec(center=(LX - 0.7e-6, yb, LZ / 2), axis=0, direction="+", width=1.2e-6, height=0.5e-6, name="thru")],
        wavelength=1.55e-6, resolution=res, max_time=200e-15, domain_size=(LX, LY, LZ), pml_layers=6,
    )
    a, o, _ = fdtdx.apply_params(a, o, {})
    _, sr = fdtdx.run_fdtd(
        arrays=a, objects=o, config=c, show_progress=False,
        stopping_condition=EnergyThresholdCondition(min_steps=round(c.time_steps_total / 5)),
    )
    states = sr.detector_states
    in_name = determine_input_norm_detector_name("in", o)
    alpha_in = complex(o[in_name].compute_overlap(states[in_name])[0])
    decomp = compute_mode_expansion(o["thru"], states["thru"], a, c, modes, input_overlap=alpha_in, cache_path=cache_path)
    # The through detector's own (fundamental TE0) |overlap / input|² — the same quantity the
    # ModeOverlapDetector reports — used here only to check decomposition self-consistency.
    te0_self_T = abs(complex(o["thru"].compute_overlap(states["thru"])[0]) / alpha_in) ** 2
    return decomp, te0_self_T, o, a, c, states, alpha_in


def test_modal_decomposition_te0_dominates():
    modes = [("te", 0), ("te", 1), ("tm", 0)]
    decomp, te0_self_T, *_ = _run_and_decompose(modes)

    by_label = {ch.label: ch for ch in decomp.channels}
    # Fundamental TE0 carries essentially all the transmitted power on a single-mode strip.
    assert by_label["TE0"].power_fraction > 0.9
    assert by_label["TE0"].transmission > by_label["TE1"].transmission
    assert by_label["TE0"].transmission > by_label["TM0"].transmission
    # Power fractions are a partition of the captured power.
    assert abs(sum(ch.power_fraction for ch in decomp.channels) - 1.0) < 1e-6
    # TE0 transmission matches the ModeOverlapDetector's own |overlap|² (self-consistency).
    assert abs(by_label["TE0"].transmission - te0_self_T) < 1e-6


def test_mode_cache_roundtrip(tmp_path):
    modes = [("te", 0), ("te", 1)]
    cache = tmp_path / "modes.npz"
    decomp1, _, objects, arrays, config, states, alpha_in = _run_and_decompose(modes, cache_path=cache)
    assert decomp1.n_computed == 2 and decomp1.n_cached == 0
    assert cache.exists()

    # Second decomposition of the same monitor reuses the cache (no re-solve).
    decomp2 = compute_mode_expansion(
        objects["thru"], states["thru"], arrays, config, modes, input_overlap=alpha_in, cache_path=cache
    )
    assert decomp2.n_cached == 2 and decomp2.n_computed == 0
    # Identical results from the cached path.
    assert abs(decomp1.channels[0].transmission - decomp2.channels[0].transmission) < 1e-9

    # Fingerprint guard: a wrong-shaped cross-section is rejected (returns None → would recompute).
    bogus = _cross_section_fingerprint(np.ones((1, 3, 3, 1)), 90e-9, 1.0e14, "+")
    assert _load_mode_cache(cache, bogus) is None
