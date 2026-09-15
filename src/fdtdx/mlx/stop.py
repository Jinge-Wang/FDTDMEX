"""Stopping conditions on the MLX (Metal) forward path.

``fdtdx.fdtd.stop_conditions`` evaluates its condition *inside* the JAX ``while_loop``: once per
step, on traced arrays, for free. The MLX loop is an eager Python ``for`` loop, so an every-step
evaluation would force a GPU sync every step and destroy the pipelining the loop depends on.

So the condition is reduced **before the loop** to a plain-data :class:`StopPlan` (threshold,
min/max steps, check cadence) and evaluated only at the loop's existing ``mx.eval``
synchronisation points — ``check_every`` is snapped to a multiple of ``eval_every``, so a check
costs one extra reduction plus one scalar ``.item()`` on a boundary where the loop already
synchronises, and nothing at all on the other steps.

**The stop contract vs JAX.** Identical except for the sampling cadence:

- *Step counting.* ``run_fdtd`` returns the number of steps actually executed, the same quantity
  the JAX ``while_loop`` carries in ``state[0]``. The condition is read as "after ``s`` steps have
  run": ``EnergyThresholdCondition`` stops at the first ``s >= min_steps`` whose post-step total
  energy is below the threshold, capped at ``max_steps``.
- *Detector buffers.* Rows are sized for ``time_steps_total`` and rows for steps that never ran
  keep the zeros ``ArrayContainer.reset()`` left there — exactly what the JAX early-stop path
  leaves behind. Nothing is truncated or masked, on either backend.
- *The one difference.* A check is only sampled every ``check_every`` steps, so the MLX run can
  overshoot the JAX stop step by up to ``check_every - 1`` steps.

Supported: :class:`TimeStepCondition`, :class:`EnergyThresholdCondition` and
:class:`DetectorConvergenceCondition` (whose readings come from the frozen detector buffers the
loop is already filling). Any other subclass keeps the JAX fallback with the existing warn-once
reason; the match is on the *exact* type, because a subclass may override ``__call__``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Reason string handed back to the dispatcher for conditions the MLX path cannot evaluate.
UNSUPPORTED_REASON = "custom stopping_condition not supported by the MLX backend yet"


@dataclass(frozen=True)
class StopPlan:
    """Plain-data reduction of a ``StoppingCondition``, derived once before the time loop."""

    #: ``"time"`` | ``"energy"`` | ``"detector"``.
    kind: str
    #: Hard cap; the loop never runs past this many steps.
    max_steps: int
    #: No early stop before this many steps have run.
    min_steps: int = 0
    #: Energy threshold / spectral-distance threshold, depending on ``kind``.
    threshold: float = 0.0
    #: Steps between checks; snapped to a multiple of the loop's ``eval_every``.
    check_every: int = 8
    #: ``"detector"`` only: the watched detector, its samples-per-period and reference periods.
    detector_name: str | None = None
    spp: int = 0
    prev_periods: int = 0

    @property
    def needs_checks(self) -> bool:
        """Whether the loop has to evaluate anything at the check cadence (false for a pure time plan)."""
        return self.kind != "time"


def _condition_kind(stopping_condition: Any) -> str | None:
    """Map a condition instance to a plan kind, or ``None`` when the MLX path can't evaluate it.

    Local imports mirror the dispatcher's convention (no load-time cycle). The match is on the
    exact type: a subclass may override ``__call__`` with arbitrary logic, which a plan derived
    from its attributes would silently misrepresent.
    """
    from fdtdx.fdtd.stop_conditions import (
        DetectorConvergenceCondition,
        EnergyThresholdCondition,
        TimeStepCondition,
    )

    kinds = {TimeStepCondition: "time", EnergyThresholdCondition: "energy", DetectorConvergenceCondition: "detector"}
    return kinds.get(type(stopping_condition))


def _detector_condition_reason(stopping_condition: Any, config, objects) -> str | None:
    """Why ``DetectorConvergenceCondition`` can't be served from the MLX detector buffers, if so.

    The condition reads a ``(time_steps_total, 1)`` reading array indexed by absolute time step.
    The MLX buffers meet that contract only for a *forward* detector with ``reduce_volume=True``
    that records on every step (then the plan's time->row map is the identity, as on JAX).
    """
    name = stopping_condition.detector_name
    match = None
    for d in objects.forward_detectors:
        if d.name == name:
            match = d
            break
    if match is None:
        return (
            f"DetectorConvergenceCondition watches detector {name!r}, "
            "which is not a forward detector recorded by the MLX backend"
        )
    if not getattr(match, "reduce_volume", False):
        return "DetectorConvergenceCondition needs reduce_volume=True on the watched detector (MLX backend)"
    if int(match._num_latent_time_steps()) != int(config.time_steps_total):
        return "DetectorConvergenceCondition needs a detector recording on every time step (MLX backend)"
    return None


def stop_condition_unsupported_reason(stopping_condition: Any, config, objects) -> str | None:
    """Return why this stopping condition forces the JAX fallback, or ``None`` if MLX can run it."""
    if stopping_condition is None:
        return None
    kind = _condition_kind(stopping_condition)
    if kind is None:
        return UNSUPPORTED_REASON
    if kind == "detector":
        return _detector_condition_reason(stopping_condition, config, objects)
    return None


def build_stop_plan(stopping_condition: Any, state, config, objects, *, check_every: int) -> StopPlan | None:
    """Derive the :class:`StopPlan` for ``stopping_condition``, or ``None`` when there is none.

    ``state`` is the reset ``(0, arrays)`` pair the JAX engine also feeds to ``setup()``, so the
    condition's own defaults (``min_steps``, ``max_steps``, samples-per-period) and its validation
    are the ones upstream defines — the plan only copies the resolved numbers out.
    """
    if stopping_condition is None:
        return None
    kind = _condition_kind(stopping_condition)
    if kind is None:  # pragma: no cover - guarded by the dispatcher
        raise NotImplementedError(UNSUPPORTED_REASON)

    cond = stopping_condition.setup(state, config, objects)
    total = int(config.time_steps_total)
    if kind == "time":
        return StopPlan(kind="time", max_steps=total, check_every=check_every)
    if kind == "energy":
        # The JAX loop caps at both `while_loop(max_steps=time_steps_total)` and the condition's
        # own `curr < max_steps`, so the effective cap is the smaller of the two.
        return StopPlan(
            kind="energy",
            max_steps=min(total, int(cond.max_steps)),
            min_steps=int(cond.min_steps),
            threshold=float(cond.threshold),
            check_every=check_every,
        )
    # DetectorConvergenceCondition.__call__ caps on `config.time_steps_total` (it does not read its
    # own max_steps attribute) -- mirror that exactly rather than the attribute.
    return StopPlan(
        kind="detector",
        max_steps=total,
        min_steps=int(cond.min_steps),
        threshold=float(cond.threshold),
        check_every=check_every,
        detector_name=str(cond.detector_name),
        spp=int(cond._spp),
        prev_periods=int(cond.prev_periods),
    )


def aligned_check_every(plan: StopPlan | None, eval_every: int) -> int:
    """Steps between stop checks, snapped down to a multiple of ``eval_every``; ``0`` = never.

    Snapping is what keeps the hot path free of extra synchronisation: every check then lands on a
    step where the loop already calls ``mx.eval``. A plan asking for less than one eval interval
    gets one eval interval.
    """
    if plan is None or not plan.needs_checks:
        return 0
    return max(eval_every, (plan.check_every // eval_every) * eval_every)


def should_stop(plan: StopPlan, state, detector_buffers: dict, steps_done: int) -> bool:
    """Evaluate ``plan`` after ``steps_done`` steps; ``True`` means break out of the time loop.

    Costs one device reduction and one scalar ``.item()`` sync. Called only at the loop's
    ``mx.eval`` boundaries, where the arrays are evaluated anyway.
    """
    import mlx.core as mx  # local: keeps the gating helpers above importable without a Metal device

    if plan.kind == "time" or steps_done < plan.min_steps:
        return False

    if plan.kind == "energy":
        from fdtdx.mlx.metrics import compute_energy_mlx

        # Same formula as EnergyThresholdCondition: sum of compute_energy over the *whole* domain,
        # PML region included, from the post-step E/H and the inverse permittivity/permeability.
        total_energy = mx.sum(compute_energy_mlx(state.E, state.H, state.inv_eps, state.inv_mu))
        return bool(total_energy.item() < plan.threshold)

    # DetectorConvergenceCondition: L2 distance between the amplitude spectrum of the last full
    # period and the mean of the `prev_periods` before it, read straight out of the live buffer.
    assert plan.detector_name is not None
    readings = next(iter(detector_buffers[plan.detector_name].values()))  # (time_steps_total, 1)
    spp, periods, total = plan.spp, plan.prev_periods, readings.shape[0]
    start_ref = min(max(steps_done - (periods + 1) * spp, 0), total - periods * spp)
    start_last = min(max(steps_done - spp, 0), total - spp)

    ref = readings[start_ref : start_ref + periods * spp, 0].reshape(periods, spp)
    last = readings[start_last : start_last + spp, 0]
    fft_ref = mx.abs(mx.fft.rfft(mx.mean(ref, axis=0), n=spp))
    fft_last = mx.abs(mx.fft.rfft(last, n=spp))
    distance = mx.sqrt(mx.sum(mx.square(fft_ref - fft_last)))
    return bool(distance.item() < plan.threshold)
