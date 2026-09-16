"""Simulation test: the resonance finder reads a Fabry-Perot slab's mode off a short ring-down.

Layout (25 nm cells, z is the propagation axis, 3x3 periodic cells in x and y):

.. code-block:: text

      cells   0 -  15 : PML
      cells  16 -  75 : vacuum (source at z = 36)
      cells  76 - 115 : dielectric slab, eps = 12  (n = 3.4641, 40 cells = 1.0 um)
      cells 116 - 175 : vacuum
      cells 176 - 191 : PML

A short Gaussian plane-wave pulse centred on the m = 4 longitudinal resonance excites the slab; a
one-cell ``FieldDetector`` seven cells inside the slab records Ex, switched on only after the source
has died away, so the record is a clean ring-down.

Checks
------
1. ``find_resonances`` puts the mode on the analytic Fabry-Perot line ``f_m = m c / (2 n L)``, up to
   the grid's numerical dispersion. That shift is predictable from the cell count: for axial
   propagation the numerical wavenumber exceeds the exact one by ``(k*d)**2 (1 - S**2) / 24``, with
   ``k*d = 2 pi / 20`` for the 20 cells per wavelength inside the slab and ``S = c dt / (n d)`` the
   medium Courant number. That is 0.40 % here, and the resonance of a cavity of fixed geometric
   length moves down by the same fraction. The test allows 1.5x that.
2. The mode's Q matches the analytic pole Q, ``m pi / (2 ln(1/r))`` with ``r = (n-1)/(n+1)``.
3. ``q_from_ringdown`` agrees with the finder's Q to 20 %.

The run is forced onto the JAX engine: this test is about the analysis utility, so it should read
the same numbers on every platform rather than exercising the backend routing.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx
from fdtdx.constants import c as C_LIGHT
from fdtdx.utils.resonance import find_resonances, q_from_ringdown

# ── Geometry ──────────────────────────────────────────────────────────────────
_RES = 25e-9
_EPS = 12.0
_N = float(np.sqrt(_EPS))
_SLAB_CELLS = 40
_PML_CELLS = 16
_VAC_CELLS = 60
_NZ = 2 * _PML_CELLS + 2 * _VAC_CELLS + _SLAB_CELLS
_SLAB_START = _PML_CELLS + _VAC_CELLS
_SOURCE_Z = _PML_CELLS + 20
_PROBE_Z = _SLAB_START + 7

# ── Analytic cavity ───────────────────────────────────────────────────────────
_L = _SLAB_CELLS * _RES
_FSR = C_LIGHT / (2.0 * _N * _L)  # free spectral range
_ORDER = 4  # longitudinal mode order; 20 cells per wavelength inside the slab
_F0 = _ORDER * _FSR
_R_FACET = (_N - 1.0) / (_N + 1.0)
_Q_ANALYTIC = _ORDER * np.pi / (2.0 * np.log(1.0 / _R_FACET))
_TAU = _Q_ANALYTIC / (2.0 * np.pi * _F0)  # field-envelope lifetime

# ── Excitation ────────────────────────────────────────────────────────────────
# A third of a free spectral range: broad enough to be a short pulse, narrow enough that the
# neighbouring orders are driven a hundred times weaker.
_SIGMA_F = _FSR / 3.0
_SIGMA_T = 1.0 / (2.0 * np.pi * _SIGMA_F)
_RECORD_START = 13.0 * _SIGMA_T  # the pulse peaks at 6 sigma_t and is over well before this
_SIM_TIME = _RECORD_START + 12.0 * _TAU


def _build():
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_RES),
        time=_SIM_TIME,
        dtype=jnp.float32,
    )
    objects, constraints = [], []

    volume = fdtdx.SimulationVolume(partial_real_shape=(3 * _RES, 3 * _RES, _NZ * _RES))
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={"min_x": "periodic", "max_x": "periodic", "min_y": "periodic", "max_y": "periodic"},
    )
    bound_dict, bound_constraints = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(bound_constraints)
    objects.extend(bound_dict.values())

    slab = fdtdx.UniformMaterialObject(
        name="slab",
        partial_grid_shape=(None, None, _SLAB_CELLS),
        material=fdtdx.Material(permittivity=_EPS),
    )
    constraints.extend(
        [
            slab.same_size(volume, axes=(0, 1)),
            slab.place_at_center(volume, axes=(0, 1)),
            slab.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_SLAB_START,)),
        ]
    )
    objects.append(slab)

    source = fdtdx.UniformPlaneSource(
        name="source",
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(frequency=_F0),
        temporal_profile=fdtdx.GaussianPulseProfile(
            center_wave=fdtdx.WaveCharacter(frequency=_F0),
            spectral_width=fdtdx.WaveCharacter(frequency=_SIGMA_F),
        ),
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

    probe = fdtdx.FieldDetector(
        name="probe",
        components=("Ex",),
        reduce_volume=True,
        partial_grid_shape=(1, 1, 1),
        switch=fdtdx.OnOffSwitch(start_time=_RECORD_START),
    )
    constraints.extend(
        [
            probe.place_at_center(volume, axes=(0, 1)),
            probe.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_PROBE_Z,)),
        ]
    )
    objects.append(probe)

    return objects, constraints, config


def _ringdown() -> tuple[np.ndarray, float]:
    """Run the slab and return the recorded Ex ring-down plus its sample spacing."""
    objects, constraints, config = _build()
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    with fdtdx.use_backend("jax"):
        _, final = fdtdx.run_fdtd(
            arrays=arrays,
            objects=obj_container,
            config=config,
            key=key,
            show_progress=False,
        )
    signal = np.asarray(final.detector_states["probe"]["fields"])[:, 0].astype(float)
    return signal, float(config.time_step_duration)


def _predicted_dispersion_shift(dt: float) -> float:
    """Relative downward shift of the resonance from the grid's axial numerical dispersion."""
    cells_per_wavelength = 2.0 * _SLAB_CELLS / _ORDER  # 20 cells at m = 4
    courant_in_medium = C_LIGHT * dt / (_N * _RES)
    return (2.0 * np.pi / cells_per_wavelength) ** 2 * (1.0 - courant_in_medium**2) / 24.0


def test_fabry_perot_resonance_and_q():
    """The finder lands on the analytic line and Q, and the ring-down estimate confirms Q."""
    signal, dt = _ringdown()

    assert signal.size > 500, "the detector recorded too little of the ring-down"
    assert np.abs(signal).max() > 0.0, "no field reached the probe"

    modes = find_resonances(signal, dt, _F0 - 0.45 * _FSR, _F0 + 0.45 * _FSR, error_threshold=1e-2)
    assert modes, "no resonance found in the band around the m = 4 line"

    mode = max(modes, key=lambda m: m.amplitude)
    shift = (mode.frequency - _F0) / _F0
    tolerance = 1.5 * _predicted_dispersion_shift(dt)
    assert abs(shift) < tolerance, (
        f"resonance at {mode.frequency:.6e} Hz, analytic line {_F0:.6e} Hz: "
        f"relative shift {shift:+.4f}, allowed {tolerance:.4f} "
        f"(1.5x the predicted grid-dispersion shift at {2 * _SLAB_CELLS / _ORDER:.0f} cells/wavelength)"
    )
    assert shift < 0.0, "numerical dispersion shifts a fixed-length cavity's resonance down, not up"

    assert mode.decay_rate > 0.0, "a passive cavity must ring down, not up"
    assert abs(mode.q - _Q_ANALYTIC) / _Q_ANALYTIC < 0.10, (
        f"Q = {mode.q:.3f}, analytic Fabry-Perot pole Q = {_Q_ANALYTIC:.3f}"
    )

    q_envelope = q_from_ringdown(signal, dt, mode.frequency)
    assert abs(q_envelope - mode.q) / mode.q < 0.20, (
        f"ring-down envelope Q = {q_envelope:.3f} disagrees with the finder's Q = {mode.q:.3f} by more than 20 %"
    )
