"""Physics simulation tests for dispersive (Lorentz / Drude) materials.

Normal-incidence transmission/reflection through a semi-infinite dispersive
half-space at a single frequency. The dispersive ADE update must reproduce
the analytic Fresnel coefficient computed from the DispersionModel's
susceptibility at the test frequency.

Layout mirrors ``test_fresnel.py`` — 3x3 periodic transverse, PMLs in z,
``UniformPlaneSource`` in +z, one transmission-side Poynting flux detector.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.constants import c as c0

_WAVELENGTH = 1e-6
_OMEGA = 2.0 * np.pi * c0 / _WAVELENGTH  # ≈ 1.884e15 rad/s
_RESOLUTION = 25e-9  # 40 cells/λ in vacuum
_PML_CELLS = 10
_DOMAIN_XY = 3 * _RESOLUTION
_DOMAIN_Z = 5e-6
_Z_CELLS = round(_DOMAIN_Z / _RESOLUTION)  # = 200

_SOURCE_Z = _PML_CELLS + 2  # = 12
_INTERFACE_Z = 100
_DET_T_Z = 140

# Layout for tests with a source fully embedded in a uniform dispersive medium.
_UNIFORM_SOURCE_Z = 60
_UNIFORM_FWD_Z = 90
_UNIFORM_BWD_Z = 30

_DIEL_CELLS_Z = _Z_CELLS - _INTERFACE_Z  # = 100 cells

_SIM_TIME = 120e-15
_TOLERANCE = 0.05

_DT_APPROX = 0.99 * _RESOLUTION / (c0 * np.sqrt(3))
_STEPS_PER_PERIOD = round(_WAVELENGTH / (c0 * _DT_APPROX))
_N_AVG_STEPS = 10 * _STEPS_PER_PERIOD


# ---------------------------------------------------------------------------
# Dispersion models used by the tests
# ---------------------------------------------------------------------------


def _lorentz_model():
    """A Lorentz pole whose resonance is well above the test frequency, so
    Im(ε) is tiny at ω and the medium behaves like a low-loss dielectric."""
    # omega_0 = 2*omega, Δε chosen so that ε_inf + Re(χ) ≈ 4 (n ≈ 2)
    omega_0 = 2.0 * _OMEGA
    gamma = 1e13
    # At ω = omega_0/2, Re(χ) = Δε·ω₀²/(ω₀² - ω²) = Δε · 4/3
    # For ε_inf=1 and target ε=4, we want Re(χ) = 3 → Δε = 9/4 = 2.25
    delta_eps = 2.25
    return fdtdx.DispersionModel(
        poles=(fdtdx.LorentzPole(resonance_frequency=omega_0, damping=gamma, delta_epsilon=delta_eps),)
    )


def _drude_model():
    """A Drude pole with ω_p ≫ ω, damping small compared to ω — gives ε with
    a large negative real part, i.e. a highly reflective metallic response."""
    omega_p = 5.0 * _OMEGA
    gamma = 0.05 * _OMEGA
    return fdtdx.DispersionModel(poles=(fdtdx.DrudePole(plasma_frequency=omega_p, damping=gamma),))


def _fresnel_transmission_semi_infinite(eps_complex: complex) -> float:
    """Power transmission coefficient from vacuum into a semi-infinite
    medium with complex permittivity ``eps_complex`` at normal incidence.

    T = Re(n2) · |t|^2 with t = 2 / (1 + n2) and n2 = sqrt(eps_complex).
    """
    n2 = np.sqrt(eps_complex)
    t = 2.0 / (1.0 + n2)
    return float(np.real(n2) * np.abs(t) ** 2)


# ---------------------------------------------------------------------------
# Scene builders — mirror test_fresnel.py
# ---------------------------------------------------------------------------


def _build_base():
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
        time=_SIM_TIME,
        dtype=jnp.float32,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(_DOMAIN_XY, _DOMAIN_XY, _DOMAIN_Z),
    )
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={
            "min_x": "periodic",
            "max_x": "periodic",
            "min_y": "periodic",
            "max_y": "periodic",
        },
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_SOURCE_Z,)),
        ]
    )
    objects.append(source)

    return objects, constraints, config, volume


def _add_flux_det(name, z_idx, volume, objects, constraints):
    det = fdtdx.PoyntingFluxDetector(
        name=name,
        partial_grid_shape=(None, None, 1),
        direction="+",
        reduce_volume=True,
        plot=False,
    )
    constraints.extend(
        [
            det.same_size(volume, axes=(0, 1)),
            det.place_at_center(volume, axes=(0, 1)),
            det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(z_idx,)),
        ]
    )
    objects.append(det)


def _add_half_space(material, volume, objects, constraints):
    """Fill cells [_INTERFACE_Z, _Z_CELLS) with ``material``."""
    slab = fdtdx.UniformMaterialObject(
        partial_grid_shape=(None, None, _DIEL_CELLS_Z),
        material=material,
    )
    constraints.extend(
        [
            slab.same_size(volume, axes=(0, 1)),
            slab.place_at_center(volume, axes=(0, 1)),
            slab.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_INTERFACE_Z,)),
        ]
    )
    objects.append(slab)


def _fill_domain(material, volume, objects, constraints):
    """Fill the entire z-extent of the domain with ``material``."""
    slab = fdtdx.UniformMaterialObject(
        partial_grid_shape=(None, None, _Z_CELLS),
        material=material,
    )
    constraints.extend(
        [
            slab.same_size(volume, axes=(0, 1)),
            slab.place_at_center(volume, axes=(0, 1)),
            slab.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(0,)),
        ]
    )
    objects.append(slab)


def _run(objects, constraints, config):
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj_container, config=config, key=key)
    return arrays


def _mean_flux(arrays, name):
    flux = np.array(arrays.detector_states[name]["poynting_flux"][:, 0])
    return float(np.mean(flux[-_N_AVG_STEPS:]))


# ---------------------------------------------------------------------------
# Lorentz tests
# ---------------------------------------------------------------------------


def test_lorentz_transmission_matches_fresnel():
    """Semi-infinite Lorentz dielectric transmits as predicted by Fresnel.

    Two-run normalization: vacuum reference establishes S0, Lorentz run gives
    S_T, and T_measured = S_T / S0 is compared to the analytic Fresnel
    transmission evaluated from the model's own susceptibility at the source
    frequency.
    """
    model = _lorentz_model()
    eps_inf = 1.0
    eps_omega = eps_inf + complex(model.susceptibility(_OMEGA))
    T_analytic = _fresnel_transmission_semi_infinite(eps_omega)

    # Reference run: vacuum everywhere
    obj0, con0, cfg0, vol0 = _build_base()
    _add_flux_det("flux_t", _DET_T_Z, vol0, obj0, con0)
    S0 = _mean_flux(_run(obj0, con0, cfg0), "flux_t")

    # Dispersive run
    obj1, con1, cfg1, vol1 = _build_base()
    material = fdtdx.Material(permittivity=eps_inf, dispersion=model)
    _add_half_space(material, vol1, obj1, con1)
    _add_flux_det("flux_t", _DET_T_Z, vol1, obj1, con1)
    S_T = _mean_flux(_run(obj1, con1, cfg1), "flux_t")

    assert S0 > 0, f"Reference flux zero: {S0}"
    assert S_T > 0, f"Dispersive transmitted flux zero: {S_T}"

    T_measured = S_T / S0
    rel_err = abs(T_measured - T_analytic) / T_analytic
    assert rel_err < _TOLERANCE, (
        f"Lorentz T_measured={T_measured:.4f}, "
        f"T_analytic={T_analytic:.4f} (eps={eps_omega}), "
        f"rel_err={rel_err:.3f} > {_TOLERANCE}"
    )


def test_lorentz_permittivity_sanity():
    """Quick unit-level sanity check on the Lorentz model itself so failures
    in the simulation test are easier to attribute."""
    model = _lorentz_model()
    eps_inf = 1.0
    eps = eps_inf + model.susceptibility(_OMEGA)
    # Target was Re(ε) ≈ 4, Im(ε) small
    assert abs(eps.real - 4.0) < 0.05
    assert eps.imag > 0  # causal absorption sign
    assert eps.imag < 0.01 * eps.real


# ---------------------------------------------------------------------------
# Drude test
# ---------------------------------------------------------------------------


def _build_embedded_source(source_z: int):
    """Same as ``_build_base`` but with the plane source placed at an arbitrary
    z-coordinate so it can be embedded inside a uniform medium."""
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
        time=_SIM_TIME,
        dtype=jnp.float32,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(_DOMAIN_XY, _DOMAIN_XY, _DOMAIN_Z),
    )
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={
            "min_x": "periodic",
            "max_x": "periodic",
            "min_y": "periodic",
            "max_y": "periodic",
        },
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    objects.extend(bound_dict.values())

    wave = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(source_z,)),
        ]
    )
    objects.append(source)

    return objects, constraints, config, volume


def test_plane_source_inside_lorentz_medium_has_correct_impedance():
    """A TFSF source embedded in a uniform Lorentz medium must inject the
    impedance of the medium at the carrier frequency, not of vacuum. If the
    impedance is matched, the backward-scattered flux is close to zero; if
    the source used ``eps_inf`` (the pre-fix behavior) the impedance
    mismatch would reflect ~10 % of the injected power into the backward
    half-space.
    """
    model = _lorentz_model()
    eps_inf = 1.0
    eps_omega = eps_inf + complex(model.susceptibility(_OMEGA))
    # Sanity: the medium must be a meaningfully different impedance from vacuum
    assert abs(np.sqrt(eps_omega.real) - 1.0) > 0.5, "Test premise weak: Lorentz is too close to vacuum"

    obj, con, cfg, vol = _build_embedded_source(_UNIFORM_SOURCE_Z)
    material = fdtdx.Material(permittivity=eps_inf, dispersion=model)
    _fill_domain(material, vol, obj, con)
    _add_flux_det("flux_fwd", _UNIFORM_FWD_Z, vol, obj, con)
    _add_flux_det("flux_bwd", _UNIFORM_BWD_Z, vol, obj, con)

    arrays = _run(obj, con, cfg)
    S_fwd = _mean_flux(arrays, "flux_fwd")
    S_bwd = _mean_flux(arrays, "flux_bwd")

    assert S_fwd > 0, f"Forward flux should be positive, got {S_fwd}"
    # Both detectors have direction='+'; a backward wave registers as a
    # negative flux on the '-' side of the source. Take the magnitude.
    ratio = abs(S_bwd) / abs(S_fwd)
    assert ratio < 0.02, (
        f"Backward/forward flux ratio {ratio:.4f} exceeds 2% — the source "
        "impedance is not matched to the dispersive medium."
    )


def test_drude_metal_is_highly_reflective():
    """A Drude half-space with ω_p ≫ ω reflects ≳ 90 % of incident power.

    Measures transmitted flux and checks it is a small fraction of the
    vacuum-reference flux, matching the Fresnel prediction for the complex
    permittivity at the source frequency within 5 %.
    """
    model = _drude_model()
    eps_inf = 1.0
    eps_omega = eps_inf + complex(model.susceptibility(_OMEGA))
    T_analytic = _fresnel_transmission_semi_infinite(eps_omega)
    # Sanity: Drude above plasma limit should reflect strongly → T small
    assert T_analytic < 0.05, f"Test premise wrong: Drude T_analytic={T_analytic:.3f} not small"

    obj0, con0, cfg0, vol0 = _build_base()
    _add_flux_det("flux_t", _DET_T_Z, vol0, obj0, con0)
    S0 = _mean_flux(_run(obj0, con0, cfg0), "flux_t")

    obj1, con1, cfg1, vol1 = _build_base()
    material = fdtdx.Material(permittivity=eps_inf, dispersion=model)
    _add_half_space(material, vol1, obj1, con1)
    _add_flux_det("flux_t", _DET_T_Z, vol1, obj1, con1)
    S_T = _mean_flux(_run(obj1, con1, cfg1), "flux_t")

    assert S0 > 0, f"Reference flux zero: {S0}"

    T_measured = S_T / S0
    # Absolute rather than relative tolerance because T_analytic is close to 0
    assert abs(T_measured - T_analytic) < _TOLERANCE, (
        f"Drude T_measured={T_measured:.4f}, T_analytic={T_analytic:.4f}, "
        f"|diff|={abs(T_measured - T_analytic):.3f} > {_TOLERANCE}"
    )


# ---------------------------------------------------------------------------
# Debye tests
# ---------------------------------------------------------------------------

# Relaxation time placed at omega*tau = 1 — the knee of the Debye response,
# where Im(chi) is maximal and the model is most distinguishable from a
# constant dielectric. tau = 1/omega is ~11 time steps, so the relaxation is
# resolved by the grid.
_DEBYE_DELTA_EPS = 4.0
_DEBYE_TAU = 1.0 / _OMEGA
_DEBYE_EPS_INF = 2.25

# Two detectors inside the medium, used to measure the decay rate over a fixed
# span. Both sit at the same sub-cell offset, so the 500 nm separation carries
# no half-cell ambiguity (an absolute depth would).
_DEBYE_DECAY_Z0 = _INTERFACE_Z + 5
_DEBYE_DECAY_Z1 = _INTERFACE_Z + 25
_DEBYE_DECAY_SPAN = (_DEBYE_DECAY_Z1 - _DEBYE_DECAY_Z0) * _RESOLUTION
# The reflection detector sits *behind* the source: for a one-way TFSF plane
# source that half-space is the scattered-field region, so it carries the
# reflected wave alone. A detector between source and interface would read the
# total field (incident minus reflected) instead.
_DEBYE_SOURCE_Z = 40
_REFLECT_Z = 25


def _debye_model():
    return fdtdx.DispersionModel(poles=(fdtdx.DebyePole(delta_epsilon=_DEBYE_DELTA_EPS, relaxation_time=_DEBYE_TAU),))


def _fresnel_reflection_semi_infinite(eps_complex: complex) -> float:
    """Power reflection at normal incidence from vacuum into ``eps_complex``."""
    n2 = np.sqrt(eps_complex)
    return float(np.abs((1.0 - n2) / (1.0 + n2)) ** 2)


def test_debye_permittivity_sanity():
    """Unit-level sanity on the Debye model so simulation failures are easier
    to attribute: at omega*tau = 1 the susceptibility is
    delta_epsilon * (1 + i) / 2 in the exp(-i omega t) convention."""
    model = _debye_model()
    chi = model.susceptibility(_OMEGA)
    assert chi.real == pytest.approx(_DEBYE_DELTA_EPS / 2.0, rel=1e-12)
    assert chi.imag == pytest.approx(_DEBYE_DELTA_EPS / 2.0, rel=1e-12)
    assert chi.imag > 0  # causal absorption sign
    eps = _DEBYE_EPS_INF + chi
    assert eps == pytest.approx(4.25 + 2.0j, rel=1e-12)


def test_debye_reflection_matches_fresnel():
    """A Debye half-space reflects as predicted by Fresnel.

    Reflection, not deep transmission, is the quantity compared against the
    analytic complex index: it is a surface quantity, so it carries none of
    the absorption-depth ambiguity of a detector placed inside a medium whose
    absorption length is a few cells. The TFSF plane source radiates only in
    +z, so a flux detector between the source and the interface sees the
    reflected wave alone.
    """
    model = _debye_model()
    eps_omega = _DEBYE_EPS_INF + complex(model.susceptibility(_OMEGA))
    R_analytic = _fresnel_reflection_semi_infinite(eps_omega)
    assert R_analytic > 0.1, f"Test premise weak: Debye R_analytic={R_analytic:.3f} too small"

    # Reference run: vacuum everywhere; forward flux = incident power.
    obj0, con0, cfg0, vol0 = _build_embedded_source(_DEBYE_SOURCE_Z)
    _add_flux_det("flux_t", _DET_T_Z, vol0, obj0, con0)
    S0 = _mean_flux(_run(obj0, con0, cfg0), "flux_t")

    obj1, con1, cfg1, vol1 = _build_embedded_source(_DEBYE_SOURCE_Z)
    material = fdtdx.Material(permittivity=_DEBYE_EPS_INF, dispersion=model)
    _add_half_space(material, vol1, obj1, con1)
    _add_flux_det("flux_r", _REFLECT_Z, vol1, obj1, con1)
    S_R = _mean_flux(_run(obj1, con1, cfg1), "flux_r")

    assert S0 > 0, f"Reference flux zero: {S0}"
    R_measured = abs(S_R) / S0
    rel_err = abs(R_measured - R_analytic) / R_analytic
    assert rel_err < _TOLERANCE, (
        f"Debye R_measured={R_measured:.4f}, R_analytic={R_analytic:.4f} "
        f"(eps={eps_omega}), rel_err={rel_err:.3f} > {_TOLERANCE}"
    )


def test_debye_absorption_rate_matches_analytic_index():
    """The wave decays inside the Debye medium at the analytic rate.

    Two flux detectors a fixed 500 nm apart inside the half-space give
    ``Im(n)`` from ``S(z1)/S(z0) = exp(-2 k0 Im(n) dz)`` without any
    absolute-depth ambiguity. This also pins the sign of the susceptibility:
    an anti-causal sign would make the field grow instead of decay.

    Tolerance is looser than the Fresnel tests because the ADE recurrence
    samples ``E`` half a step early (see
    ``DebyePole.recurrence_coefficients_axes``), which biases the realized
    ``Im(n)`` up by ``~omega*dt/2`` — about 5 % at this resolution.
    """
    model = _debye_model()
    eps_omega = _DEBYE_EPS_INF + complex(model.susceptibility(_OMEGA))
    n_analytic = np.sqrt(eps_omega)
    k0 = _OMEGA / c0

    obj, con, cfg, vol = _build_embedded_source(_DEBYE_SOURCE_Z)
    material = fdtdx.Material(permittivity=_DEBYE_EPS_INF, dispersion=model)
    _add_half_space(material, vol, obj, con)
    _add_flux_det("flux_a", _DEBYE_DECAY_Z0, vol, obj, con)
    _add_flux_det("flux_b", _DEBYE_DECAY_Z1, vol, obj, con)
    arrays = _run(obj, con, cfg)
    S_a = _mean_flux(arrays, "flux_a")
    S_b = _mean_flux(arrays, "flux_b")

    assert S_a > 0 and S_b > 0, f"Fluxes inside the medium must be positive, got {S_a}, {S_b}"
    assert S_b < S_a, f"Field must decay in an absorbing medium, got {S_a} -> {S_b}"
    im_n_measured = -np.log(S_b / S_a) / (2.0 * k0 * _DEBYE_DECAY_SPAN)
    rel_err = abs(im_n_measured - n_analytic.imag) / n_analytic.imag
    assert rel_err < 0.10, (
        f"Debye Im(n) measured={im_n_measured:.4f}, analytic={n_analytic.imag:.4f}, rel_err={rel_err:.3f} > 0.10"
    )
