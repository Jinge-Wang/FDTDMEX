# tests/unit/fdtd/test_mlx_stop_plan.py
"""Stop-plan derivation + the MLX feature gate for stopping conditions (refs #39).

Pure host-side logic: ``fdtdx.mlx.stop`` imports ``mlx.core`` only inside ``make_stop_check``,
so these run on any platform (no Metal device needed).
"""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from fdtdx.backend.dispatch import _unsupported_reason
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.fdtd.container import ArrayContainer, FieldState
from fdtdx.fdtd.stop_conditions import (
    DetectorConvergenceCondition,
    EnergyThresholdCondition,
    StoppingCondition,
    TimeStepCondition,
)
from fdtdx.mlx.stop import (
    UNSUPPORTED_REASON,
    aligned_check_every,
    build_stop_plan,
    stop_condition_unsupported_reason,
)

pytestmark = pytest.mark.unit

# ~525 time steps
_CONFIG = SimulationConfig(time=100e-11, grid=UniformGrid(spacing=1e-3), courant_factor=0.99)
_TOTAL = _CONFIG.time_steps_total
_DET = "watched"


def _arrays(detector_states=None):
    return ArrayContainer(
        fields=FieldState(E=jnp.ones((3, 4, 4, 4)), H=jnp.ones((3, 4, 4, 4)), psi_E={}, psi_H={}),
        inv_permittivities=jnp.ones((4, 4, 4)),
        inv_permeabilities=jnp.ones((4, 4, 4)),
        detector_states={} if detector_states is None else detector_states,
        recording_state=None,
    )


def _state(arrays=None):
    return (jnp.asarray(0, dtype=jnp.int32), _arrays() if arrays is None else arrays)


def _objects(forward_detectors=()):
    """Minimal stand-in for ObjectContainer: only the attributes the gate/plan read.

    ``detectors`` stays empty so the dispatcher's unrelated detector-type whitelist doesn't trip on
    the stand-in objects; the stop gate reads ``forward_detectors`` only.
    """
    return SimpleNamespace(
        forward_detectors=list(forward_detectors),
        detectors=[],
        sources=[],
        bloch_objects=[],
    )


def _fake_detector(name=_DET, reduce_volume=True, latent=_TOTAL):
    d = SimpleNamespace(name=name, reduce_volume=reduce_volume)
    d._num_latent_time_steps = lambda: latent
    return d


class _CustomCondition(StoppingCondition):
    """A user-defined subclass the MLX path cannot reduce to a plan."""

    def setup(self, state, config, objects):
        return self

    def _validate(self, state, config, objects):
        pass

    def __call__(self, state, config, objects):
        return jnp.asarray(True)


# ---------------------------------------------------------------------------
# Plan derivation
# ---------------------------------------------------------------------------


def test_no_condition_gives_no_plan():
    assert build_stop_plan(None, _state(), _CONFIG, _objects(), check_every=8) is None


def test_time_step_plan_runs_every_step():
    plan = build_stop_plan(TimeStepCondition(), _state(), _CONFIG, _objects(), check_every=8)
    assert plan is not None
    assert plan.kind == "time"
    assert plan.max_steps == _TOTAL
    # A pure time plan never checks anything, so the loop keeps its original synchronisation points.
    assert plan.needs_checks is False
    assert aligned_check_every(plan, 8) == 0


def test_energy_plan_copies_resolved_numbers():
    cond = EnergyThresholdCondition(threshold=2.5e-7, min_steps=40, max_steps=300)
    plan = build_stop_plan(cond, _state(), _CONFIG, _objects(), check_every=8)
    assert plan is not None
    assert (plan.kind, plan.threshold, plan.min_steps, plan.max_steps) == ("energy", 2.5e-7, 40, 300)
    assert plan.needs_checks is True


def test_energy_plan_uses_upstream_defaults():
    plan = build_stop_plan(EnergyThresholdCondition(), _state(), _CONFIG, _objects(), check_every=8)
    assert plan is not None
    assert plan.min_steps == round(_TOTAL * 0.1)  # upstream default
    assert plan.max_steps == _TOTAL
    assert plan.threshold == pytest.approx(1e-6)


def test_energy_plan_caps_max_steps_at_time_steps_total():
    cond = EnergyThresholdCondition(threshold=1e-6, min_steps=10, max_steps=_TOTAL + 1000)
    plan = build_stop_plan(cond, _state(), _CONFIG, _objects(), check_every=8)
    assert plan is not None
    assert plan.max_steps == _TOTAL


def test_energy_plan_propagates_upstream_validation():
    with pytest.raises(ValueError, match="must be positive"):
        build_stop_plan(EnergyThresholdCondition(threshold=-1.0), _state(), _CONFIG, _objects(), check_every=8)


def test_detector_plan_derives_samples_per_period():
    period = 5e-11
    arrays = _arrays({_DET: {"energy": jnp.zeros((_TOTAL, 1))}})
    cond = DetectorConvergenceCondition(
        detector_name=_DET,
        wave_character=WaveCharacter(period=period),
        prev_periods=2,
        threshold=1e-5,
        min_steps=800,
    )
    plan = build_stop_plan(cond, _state(arrays), _CONFIG, _objects([_fake_detector()]), check_every=8)
    assert plan is not None
    assert plan.kind == "detector"
    assert plan.detector_name == _DET
    assert plan.spp == round(period / _CONFIG.time_step_duration)
    assert plan.prev_periods == 2
    assert (plan.min_steps, plan.threshold) == (800, 1e-5)
    # __call__ caps on config.time_steps_total, not on the condition's own max_steps attribute.
    assert plan.max_steps == _TOTAL


# ---------------------------------------------------------------------------
# Check cadence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "eval_every", "expected"),
    [(8, 8, 8), (1, 8, 8), (16, 8, 16), (20, 8, 16), (7, 4, 4), (12, 4, 12)],
)
def test_check_cadence_snaps_to_a_multiple_of_eval_every(requested, eval_every, expected):
    plan = build_stop_plan(EnergyThresholdCondition(), _state(), _CONFIG, _objects(), check_every=requested)
    assert aligned_check_every(plan, eval_every) == expected


def test_no_plan_means_no_checks():
    assert aligned_check_every(None, 8) == 0


# ---------------------------------------------------------------------------
# Feature gate: what still falls back to JAX
# ---------------------------------------------------------------------------


def test_supported_conditions_pass_the_gate():
    objects = _objects([_fake_detector()])
    for cond in (TimeStepCondition(), EnergyThresholdCondition(threshold=1e-6, min_steps=10)):
        assert stop_condition_unsupported_reason(cond, _CONFIG, objects) is None
        assert _unsupported_reason(_CONFIG, objects, cond) is None
    assert stop_condition_unsupported_reason(None, _CONFIG, objects) is None


def test_unsupported_subclass_still_falls_back():
    objects = _objects()
    cond = _CustomCondition()
    assert stop_condition_unsupported_reason(cond, _CONFIG, objects) == UNSUPPORTED_REASON
    assert _unsupported_reason(_CONFIG, objects, cond) == UNSUPPORTED_REASON


def test_subclass_of_a_supported_condition_falls_back():
    """Exact-type match: a subclass may override __call__, which a plan would misrepresent."""

    class TighterEnergy(EnergyThresholdCondition):
        pass

    cond = TighterEnergy(threshold=1e-6, min_steps=10)
    assert stop_condition_unsupported_reason(cond, _CONFIG, _objects()) == UNSUPPORTED_REASON


def test_detector_convergence_gate():
    cond = DetectorConvergenceCondition(
        detector_name=_DET, wave_character=WaveCharacter(period=5e-11), prev_periods=2, threshold=1e-5, min_steps=800
    )
    # Supported: a forward detector, reduce_volume, recording on every step.
    assert stop_condition_unsupported_reason(cond, _CONFIG, _objects([_fake_detector()])) is None
    # Not a forward detector the MLX loop records.
    assert "not a forward detector" in str(stop_condition_unsupported_reason(cond, _CONFIG, _objects()))
    # Volume not reduced -> the readings aren't the (T, 1) array the condition slices.
    reason = stop_condition_unsupported_reason(cond, _CONFIG, _objects([_fake_detector(reduce_volume=False)]))
    assert "reduce_volume" in str(reason)
    # Not recording every step -> the time->row map isn't the identity.
    reason = stop_condition_unsupported_reason(cond, _CONFIG, _objects([_fake_detector(latent=_TOTAL // 2)]))
    assert "every time step" in str(reason)
