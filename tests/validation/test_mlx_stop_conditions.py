"""Two-backend stop-condition parity: the MLX (Metal) forward loop vs the JAX-CPU oracle.

Runs the *same* placed simulation — a small vacuum box with CPML and a point dipole driven by a
short Gaussian pulse — through both backends via ``fdtdx.use_backend`` and checks the stop
contract (refs #39):

- ``EnergyThresholdCondition`` stops both backends on the same energy crossing. JAX samples the
  condition every step; MLX samples it every ``STOP_CHECK_EVERY`` steps (the loop's ``mx.eval``
  cadence), so MLX may overshoot by up to one check interval and never more.
- When the crossing is already satisfied at ``min_steps`` and ``min_steps`` sits on the check
  cadence, both backends stop on exactly the same step — then E, H and the recorded detector
  array must agree to the usual float32 parity tolerance.
- Detector arrays keep the full ``time_steps_total`` shape on both backends, with the rows for
  steps that never ran left at their ``reset()`` zeros.
- ``TimeStepCondition`` runs exactly ``time_steps_total`` steps on MLX.

Skipped off Apple Silicon / without mlx. The Claude Code sandbox hides the Metal GPU, so this file
has to be run outside it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.backend.dispatch import STOP_CHECK_EVERY
from fdtdx.backend.platform import is_apple_silicon, mlx_available
from fdtdx.fdtd.stop_conditions import (
    DetectorConvergenceCondition,
    EnergyThresholdCondition,
    TimeStepCondition,
)

pytestmark = [
    pytest.mark.validation,
    pytest.mark.skipif(
        not (is_apple_silicon() and mlx_available()),
        reason="MLX (Metal) backend requires Apple Silicon + mlx",
    ),
]

_RES = 50e-9
_PML = 8
_N = 32
_STEPS = 600
_RTOL = 1e-3

# Calibrated on this setup against the JAX-CPU oracle (threshold sweep): the pulse peaks around
# step 100, after which the domain's total energy falls through 1.5e-2 at step 136, 1e-2 at step
# 139 and 7e-3 at step 154, then settles on a residual floor between 5e-3 and 7e-3 that the
# 600-step run never leaves. One threshold, two min_steps:
_THRESHOLD = 1e-2
# ...crossed mid-run, in between two check points, so MLX overshoots by less than one interval.
_CROSSING_MIN_STEPS = 100
# ...already satisfied at a min_steps that sits on the check cadence, so both backends stop on
# exactly that step and the fields at the stop are directly comparable.
_ALIGNED_MIN_STEPS = 144
assert _ALIGNED_MIN_STEPS % STOP_CHECK_EVERY == 0


def _build():
    dt = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_RES), time=1e-15).time_step_duration
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_RES), time=_STEPS * dt, dtype=jnp.float32)
    objects, constraints = [], []
    vol = fdtdx.SimulationVolume(partial_real_shape=(_N * _RES,) * 3)
    objects.append(vol)
    bdict, clist = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML), vol)
    constraints.extend(clist)
    objects.extend(bdict.values())
    # A short Gaussian pulse: the injected energy is finite, so the total energy decays once the
    # pulse has left the box -- the regime EnergyThresholdCondition is meant for.
    src = fdtdx.PointDipoleSource(
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=1e-6),
        temporal_profile=fdtdx.GaussianPulseProfile(
            spectral_width=fdtdx.WaveCharacter(frequency=1e14),
            center_wave=fdtdx.WaveCharacter(wavelength=1e-6),
        ),
        polarization=2,
        amplitude=1.0,
    )
    constraints.append(src.place_at_center(vol, axes=(0, 1, 2)))
    objects.append(src)
    det = fdtdx.EnergyDetector(name="energy", reduce_volume=True, plot=False)
    constraints.extend([det.same_size(vol, axes=(0, 1, 2)), det.place_at_center(vol, axes=(0, 1, 2))])
    objects.append(det)
    return objects, constraints, config


@pytest.fixture(scope="module")
def placed():
    key = jax.random.PRNGKey(0)
    objects, constraints, config = _build()
    oc, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, oc, _ = fdtdx.apply_params(arrays, oc, params, key)
    return arrays, oc, config, key


def _run(placed, backend, condition):
    arrays, oc, config, key = placed
    with fdtdx.use_backend(backend):
        step, out = fdtdx.run_fdtd(
            arrays=arrays,
            objects=oc,
            config=config,
            key=key,
            stopping_condition=condition,
            show_progress=False,
        )
    return int(step), out


def _rel(j, m):
    j, m = np.asarray(j), np.asarray(m)
    return float(np.abs(j - m).max() / (np.abs(j).max() + 1e-30))


def test_energy_threshold_stops_within_one_check_interval(placed):
    """A crossing between two check points: MLX overshoots by less than one interval, never more."""
    _, _, config, _ = placed
    cond = EnergyThresholdCondition(threshold=_THRESHOLD, min_steps=_CROSSING_MIN_STEPS)
    step_j, arr_j = _run(placed, "jax", cond)
    step_m, arr_m = _run(placed, "mlx", cond)

    # Both must actually stop early, or the calibration above has drifted and the rest is vacuous.
    assert _CROSSING_MIN_STEPS <= step_j < config.time_steps_total, step_j
    assert step_m < config.time_steps_total, step_m
    # MLX samples the condition only on its check cadence, so it stops on the first check point at
    # or after the JAX stop step -- never earlier, never a full interval later.
    assert step_m % STOP_CHECK_EVERY == 0
    assert step_j <= step_m < step_j + STOP_CHECK_EVERY, (step_j, step_m)

    # Detector shape contract: the full time_steps_total buffer on both backends, with the rows for
    # steps that never ran left at their reset() zeros.
    rec_j = np.asarray(arr_j.detector_states["energy"]["energy"])
    rec_m = np.asarray(arr_m.detector_states["energy"]["energy"])
    assert rec_j.shape == rec_m.shape == (config.time_steps_total, 1)
    assert not np.any(rec_j[step_j:]), "JAX left non-zero rows past its stop step"
    assert not np.any(rec_m[step_m:]), "MLX left non-zero rows past its stop step"
    # Every step both backends ran agrees element-wise.
    assert _rel(rec_j[:step_j], rec_m[:step_j]) < _RTOL


def test_aligned_stop_matches_jax_element_wise(placed):
    """min_steps on the check cadence: identical stop step, so E/H/detectors must match exactly."""
    _, _, config, _ = placed
    cond = EnergyThresholdCondition(threshold=_THRESHOLD, min_steps=_ALIGNED_MIN_STEPS)
    step_j, arr_j = _run(placed, "jax", cond)
    step_m, arr_m = _run(placed, "mlx", cond)

    assert step_j == _ALIGNED_MIN_STEPS, step_j  # energy is already under the threshold there
    assert step_m == step_j
    assert step_m < config.time_steps_total

    assert _rel(arr_j.fields.E, arr_m.fields.E) < _RTOL, "E mismatch at the stop"
    assert _rel(arr_j.fields.H, arr_m.fields.H) < _RTOL, "H mismatch at the stop"

    rec_j = np.asarray(arr_j.detector_states["energy"]["energy"])
    rec_m = np.asarray(arr_m.detector_states["energy"]["energy"])
    assert rec_j.shape == rec_m.shape == (config.time_steps_total, 1)
    assert _rel(rec_j, rec_m) < _RTOL
    assert not np.any(rec_j[step_j:]) and not np.any(rec_m[step_m:])


def test_time_step_condition_runs_every_step(placed):
    """TimeStepCondition on MLX runs exactly time_steps_total steps and records every row."""
    _, _, config, _ = placed
    step_m, arr_m = _run(placed, "mlx", TimeStepCondition())
    assert step_m == config.time_steps_total

    # Same as passing no condition at all.
    step_none, arr_none = _run(placed, "mlx", None)
    assert step_none == config.time_steps_total
    rec_m = np.asarray(arr_m.detector_states["energy"]["energy"])
    rec_none = np.asarray(arr_none.detector_states["energy"]["energy"])
    assert rec_m.shape == (config.time_steps_total, 1)
    assert np.array_equal(rec_m, rec_none)
    # The last row carries the residual energy of a run that went all the way to the end -- no
    # early break left trailing reset() zeros behind.
    assert rec_m[-1, 0] > 0.0


def test_detector_convergence_condition_matches_jax(placed):
    """DetectorConvergenceCondition served from the live MLX detector buffers.

    Two ends of the contract, both independent of how fast the signal actually converges: a
    threshold so large that the spectral distance is under it as soon as ``min_steps`` is reached
    (both backends stop right there, since ``min_steps`` sits on the check cadence), and a
    threshold of zero, which nothing satisfies (both run to the end).
    """
    _, _, config, _ = placed
    kwargs = dict(
        detector_name="energy",
        wave_character=fdtdx.WaveCharacter(wavelength=1e-6),
        prev_periods=2,
        min_steps=_ALIGNED_MIN_STEPS,
    )

    step_j, arr_j = _run(placed, "jax", DetectorConvergenceCondition(threshold=1e9, **kwargs))
    step_m, arr_m = _run(placed, "mlx", DetectorConvergenceCondition(threshold=1e9, **kwargs))
    assert step_j == step_m == _ALIGNED_MIN_STEPS
    assert _rel(arr_j.fields.E, arr_m.fields.E) < _RTOL
    assert _rel(arr_j.fields.H, arr_m.fields.H) < _RTOL

    step_j, _ = _run(placed, "jax", DetectorConvergenceCondition(threshold=0.0, **kwargs))
    step_m, _ = _run(placed, "mlx", DetectorConvergenceCondition(threshold=0.0, **kwargs))
    assert step_j == step_m == config.time_steps_total
