"""ModeTracker: following one physical mode across a sweep instead of a position in a sorted list.

Every check needs float64; a central finite difference of an effective index has no signal in
complex64.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.core.physics.mode_adjoint import ModeSolveSettings, ModeTracker, mode_neff

FREQ = 299792458.0 / 1.55e-6


@pytest.fixture
def float64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _two_uncoupled_guides(eps_b: float) -> jnp.ndarray:
    """Two 360 nm square cores 1.2 um apart in oxide; only the second one's index is swept.

    The separation is wide enough that the two do not couple, so guide A's effective index is a
    constant of the sweep to six decimals - which makes it obvious whether the caller is still
    looking at guide A after the crossing.
    """
    eps = np.full((44, 20), 2.25)
    eps[4:10, 7:13] = 10.5
    eps[34:40, 7:13] = eps_b
    return jnp.asarray(eps)[None, None, :, :]


def _settings(**kwargs) -> ModeSolveSettings:
    base = {"frequency": FREQ, "resolution": 60e-9, "mode_index": 0}
    base.update(kwargs)
    return ModeSolveSettings.create(**base)


class TestTrackingAcrossACrossing:
    def test_a_fixed_index_swaps_modes_and_the_tracker_does_not(self, float64):
        settings = _settings()
        tracker = ModeTracker(settings, num_candidates=4)
        fixed, tracked = [], []
        for eps_b in (9.0, 10.0, 11.0, 12.0):
            permittivity = _two_uncoupled_guides(eps_b)
            fixed.append(complex(mode_neff(permittivity, settings)).real)
            tracked.append(complex(tracker.neff(permittivity)).real)

        # Guide A does not move: the tracker reports the same index all the way through.
        assert max(tracked) - min(tracked) < 1e-6
        # The fixed index starts reporting guide B once guide B overtakes guide A.
        assert fixed[0] == pytest.approx(tracked[0], abs=1e-9)
        assert fixed[-1] > tracked[-1] * 1.05
        assert tracker.mode_index > 0

    def test_the_overlap_of_the_tracked_mode_stays_high(self, float64):
        tracker = ModeTracker(_settings(), num_candidates=4)
        overlaps = []
        for eps_b in (9.0, 10.0, 11.0, 12.0):
            tracker.neff(_two_uncoupled_guides(eps_b))
            if tracker.last_match is not None:
                overlaps.append(tracker.last_match.overlap)
        assert min(overlaps) > 0.9

    def test_the_first_step_uses_the_configured_mode_index(self, float64):
        tracker = ModeTracker(_settings(mode_index=1), num_candidates=3)
        permittivity = _two_uncoupled_guides(9.0)
        assert tracker.mode_index == 1
        assert tracker.last_match is None
        first = complex(tracker.neff(permittivity))
        assert first == pytest.approx(complex(mode_neff(permittivity, _settings(mode_index=1))), abs=1e-9)
        assert tracker.last_match is None

    def test_reset_forgets_the_tracked_mode(self, float64):
        tracker = ModeTracker(_settings(), num_candidates=4)
        for eps_b in (9.0, 12.0):
            tracker.neff(_two_uncoupled_guides(eps_b))
        assert tracker.mode_index > 0
        tracker.reset()
        assert tracker.mode_index == 0
        assert tracker.last_match is None

    def test_a_step_that_loses_the_mode_is_refused_when_a_gate_is_set(self, float64):
        """One candidate and a large jump: the tracked mode is not in the list and the gate fires."""
        tracker = ModeTracker(_settings(), num_candidates=1, min_overlap=0.9)
        tracker.neff(_two_uncoupled_guides(9.0))
        with pytest.raises(ValueError, match="best mode overlap"):
            tracker.neff(_two_uncoupled_guides(12.0))


class TestTrackedIndexIsDifferentiable:
    def test_the_gradient_of_the_tracked_index_matches_a_finite_difference(self, float64):
        """Sweeping guide A itself: the tracked mode is the one that moves, and its gradient is real."""

        def cross_section(scale):
            eps = np.full((44, 20), 2.25)
            eps[34:40, 7:13] = 11.0
            base = jnp.asarray(eps)[None, None, :, :]
            core = np.zeros((44, 20))
            core[4:10, 7:13] = 10.5
            return base + scale * jnp.asarray(core)[None, None, :, :]

        tracker = ModeTracker(_settings(), num_candidates=4)
        tracker.neff(cross_section(1.0))
        settings = tracker.settings.with_mode_index(tracker.mode_index)

        def neff_of(scale):
            return mode_neff(cross_section(scale), settings).real

        step = 1e-5
        finite_difference = (float(neff_of(1.0 + step)) - float(neff_of(1.0 - step))) / (2 * step)
        gradient = float(jax.grad(lambda s: mode_neff(cross_section(s), settings).real)(1.0))
        assert gradient != 0.0
        assert gradient == pytest.approx(finite_difference, rel=1e-4)

    def test_solve_returns_the_tracked_mode_fields(self, float64):
        tracker = ModeTracker(_settings(), num_candidates=4)
        tracker.neff(_two_uncoupled_guides(9.0))
        solution = tracker.solve(_two_uncoupled_guides(12.0))
        assert solution.E.shape[0] == 3
        assert complex(solution.neff).real == pytest.approx(2.2478, abs=1e-3)
