"""Two-way drivers: the outer loop that makes a coupled pair a fixed point, and its schedules.

A two-way coupling is a fixed point. One outer iteration composes every physics solve in the loop —
for the thermal/electromagnetic case an electromagnetic solve, the absorbed-power assembly, a
thermal solve and the thermo-optic map — and returns a new state; the coupled solution is the state
that maps to itself. Nothing here knows what the state is: it is any array (or scalar) the caller's
``step`` consumes and returns.

Two things this module exists to keep straight, both measured on an independent one-dimensional
etalon model (Apple M4 Pro):

* **Damped Picard tracks a branch; Anderson does not.** Plain Picard failed to converge in 500
  iterations at a strongly driven operating point, at every damping from 1.0 down to 0.3, while
  Anderson of depth 3 converged in 14. But Anderson started cold and Anderson started hot converged
  to *different* fixed points at the same drive (peak temperature rise 63.93 K against 24.91 K). So
  a continuation that is meant to follow one branch takes small steps with damped Picard from the
  previous solution, and Anderson is turned on only to polish a residual once the branch is fixed.
* **Bistability is the observable, not a convergence nuisance.** Two coexisting states at one drive
  differed by a factor 2.56 in absorbed fraction. :func:`continuation_sweep` sweeps a control
  parameter up and then down, each point started from its neighbour's solution, and reports where
  the two directions disagree — which is the hysteresis window.

Every run produces one JSON-serialisable block (:func:`coupling_convergence_report`) so coupled
cases are comparable in a grader: iteration count, residual history, the wall clock of each
sub-solve, the mixing used, and the branch label.

:func:`continuation_schedule` is the other schedule a coupled optimisation runs on: the projection
sharpness and grayscale penalty a density-based topology optimisation ramps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np


def _as_vector(x: Any) -> tuple[np.ndarray, Callable[[np.ndarray], Any]]:
    """Flatten a state to a 1-D float array and return the map back to its own type."""
    if np.isscalar(x) or (isinstance(x, np.generic)):
        value = float(np.asarray(x, dtype=np.float64).reshape(-1)[0])
        return np.array([value], dtype=np.float64), lambda v: float(v[0])
    array = np.asarray(x, dtype=np.float64)
    shape = array.shape
    return array.reshape(-1).copy(), lambda v: v.reshape(shape)


def _call_step(step: Callable[[Any], Any], x: Any) -> tuple[Any, dict[str, Any] | None]:
    """``step`` may return the new state, or the new state and a dictionary about the sub-solves."""
    out = step(x)
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        return out[0], out[1]
    return out, None


@dataclass
class CouplingConvergenceReport:
    """What one fixed-point solve did, in a form a grader can serialise.

    Attributes:
        converged (bool): Whether the fixed-point residual fell below the tolerance.
        iterations (int): Outer iterations actually taken.
        reason (str): ``"tolerance"``, ``"max_iter"``, ``"monitor"`` or ``"not_finite"``.
        residual (float): The final fixed-point residual ``max|G(x) - x|``.
        residual_history (list[float]): One entry per outer iteration.
        update_history (list[float]): ``max|x_{k+1} - x_k|`` per iteration, which is what damping
            changes; the residual is what convergence is judged on, so both are kept.
        monitor_history (list[float]): The caller's secondary observable per iteration, empty when
            no monitor was given. One is worth giving, because the absorbed fraction can still be
            moving when the temperature norm has stalled.
        wall_per_iteration (list[float]): Seconds per outer iteration.
        wall_total (float): Seconds for the whole solve.
        sub_solve_info (list[dict]): Whatever ``step`` returned alongside the state, one per
            iteration; the place a caller records the wall clock of each physics solve.
        mixing (dict): ``{"mix": float, "anderson": int | None}``.
        tolerance (float): The tolerance used.
        max_iter (int): The cap used.
        branch (str | None): A label the caller attaches, e.g. ``"up"`` or ``"cold-start"``.
    """

    converged: bool = False
    iterations: int = 0
    reason: str = "max_iter"
    residual: float = float("inf")
    residual_history: list[float] = field(default_factory=list)
    update_history: list[float] = field(default_factory=list)
    monitor_history: list[float] = field(default_factory=list)
    wall_per_iteration: list[float] = field(default_factory=list)
    wall_total: float = 0.0
    sub_solve_info: list[dict[str, Any]] = field(default_factory=list)
    mixing: dict[str, Any] = field(default_factory=dict)
    tolerance: float = 0.0
    max_iter: int = 0
    branch: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """The JSON-serialisable block."""
        return {
            "converged": bool(self.converged),
            "iterations": int(self.iterations),
            "reason": str(self.reason),
            "residual": float(self.residual),
            "residual_history": [float(v) for v in self.residual_history],
            "update_history": [float(v) for v in self.update_history],
            "monitor_history": [float(v) for v in self.monitor_history],
            "wall_per_iteration": [float(v) for v in self.wall_per_iteration],
            "wall_total": float(self.wall_total),
            "sub_solve_info": list(self.sub_solve_info),
            "mixing": dict(self.mixing),
            "tolerance": float(self.tolerance),
            "max_iter": int(self.max_iter),
            "branch": self.branch,
        }


def coupling_convergence_report(
    report: CouplingConvergenceReport,
    branch: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The one convergence block a coupled case writes into its results JSON.

    Args:
        report (CouplingConvergenceReport): What :func:`alternating_solver` returned.
        branch (str | None): Overrides the report's own branch label.
        extra (dict | None): Case-specific entries merged in (the drive value, the grid, the
            engine commit); they must be JSON-serialisable, which is not checked here.

    Returns:
        dict: The block. Keys are stable across tracks so a grader can compare coupled cases.
    """
    block = report.as_dict()
    if branch is not None:
        block["branch"] = branch
    if extra:
        block.update(extra)
    return block


def alternating_solver(
    step: Callable[[Any], Any],
    x0: Any,
    mix: float = 1.0,
    anderson: int | None = None,
    tol: float = 1e-10,
    max_iter: int = 200,
    monitor: Callable[[Any], float] | None = None,
    monitor_tol: float | None = None,
    branch: str | None = None,
) -> tuple[Any, CouplingConvergenceReport]:
    """Drive ``x = G(x)`` with damped Picard and, optionally, Anderson mixing.

    Damped Picard is ``x_{k+1} = x_k + mix (G(x_k) - x_k)``. With ``anderson=m`` the update is the
    type-II Anderson step over the last ``m`` residual differences, with ``mix`` as its relaxation:
    it converges where Picard cannot, and it does not respect the branch, so use it to polish and
    not to track (see the module docstring).

    Args:
        step: ``G``. Takes the state and returns the new state, or ``(new_state, info)`` where
            ``info`` is a dictionary recorded per iteration — the natural place for the wall clock
            of each physics sub-solve.
        x0: The starting state: a float, or any array-like. The return has the same shape.
        mix (float): Damping in ``(0, 1]``. ``1.0`` is plain Picard.
        anderson (int | None): Anderson depth, or ``None`` for pure damped Picard. Depth 3 was
            enough on the etalon model where Picard failed at every damping.
        tol (float): Convergence on the fixed-point residual ``max|G(x) - x|``, which does not
            depend on the damping.
        max_iter (int): Cap on outer iterations.
        monitor: Optional secondary observable ``x -> float`` recorded per iteration.
        monitor_tol (float | None): When given, the run also requires the monitor's relative change
            between iterations to fall below this before reporting convergence.
        branch (str | None): Label recorded in the report.

    Returns:
        tuple: ``(x, report)`` with ``x`` in the same layout as ``x0``.

    Raises:
        ValueError: If ``mix`` is outside ``(0, 1]``, ``anderson`` is not positive, ``tol`` is not
            positive, or ``max_iter`` is not positive.
    """
    if not (0.0 < mix <= 1.0):
        raise ValueError(f"mix must lie in (0, 1], got {mix}")
    if anderson is not None and anderson < 1:
        raise ValueError(f"anderson depth must be at least 1, got {anderson}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")

    x_vec, rebuild = _as_vector(x0)
    report = CouplingConvergenceReport(
        mixing={"mix": float(mix), "anderson": None if anderson is None else int(anderson)},
        tolerance=float(tol),
        max_iter=int(max_iter),
        branch=branch,
    )
    history_x: list[np.ndarray] = []
    history_f: list[np.ndarray] = []
    started = time.perf_counter()
    previous_monitor: float | None = None

    for iteration in range(1, max_iter + 1):
        tick = time.perf_counter()
        g_state, info = _call_step(step, rebuild(x_vec))
        g_vec, _ = _as_vector(g_state)
        f_vec = g_vec - x_vec
        residual = float(np.max(np.abs(f_vec))) if f_vec.size else 0.0

        if not np.all(np.isfinite(g_vec)):
            report.reason = "not_finite"
            report.residual = residual
            report.iterations = iteration
            report.wall_per_iteration.append(time.perf_counter() - tick)
            report.wall_total = time.perf_counter() - started
            if info is not None:
                report.sub_solve_info.append(info)
            return rebuild(x_vec), report

        if anderson is None or not history_f:
            new_vec = x_vec + mix * f_vec
        else:
            # type-II Anderson: gamma minimises ||f_k - dF gamma|| over the last `depth`
            # residual differences, and the update is x_k + mix f_k - (dX + mix dF) gamma
            depth = min(int(anderson), len(history_f))
            cols_f = [history_f[i + 1] - history_f[i] for i in range(len(history_f) - depth, len(history_f) - 1)]
            cols_x = [history_x[i + 1] - history_x[i] for i in range(len(history_x) - depth, len(history_x) - 1)]
            cols_f.append(f_vec - history_f[-1])
            cols_x.append(x_vec - history_x[-1])
            dF = np.stack(cols_f, axis=-1)
            dX = np.stack(cols_x, axis=-1)
            try:
                gamma, *_ = np.linalg.lstsq(dF, f_vec, rcond=None)
            except np.linalg.LinAlgError:
                gamma = np.zeros(dF.shape[1])
            new_vec = x_vec + mix * f_vec - (dX + mix * dF) @ gamma

        history_x.append(x_vec.copy())
        history_f.append(f_vec.copy())
        if anderson is not None and len(history_f) > int(anderson) + 1:
            history_x.pop(0)
            history_f.pop(0)

        update = float(np.max(np.abs(new_vec - x_vec))) if new_vec.size else 0.0
        x_vec = new_vec
        report.residual_history.append(residual)
        report.update_history.append(update)
        report.wall_per_iteration.append(time.perf_counter() - tick)
        if info is not None:
            report.sub_solve_info.append(info)

        monitor_ok = True
        if monitor is not None:
            value = float(monitor(rebuild(x_vec)))
            report.monitor_history.append(value)
            if monitor_tol is not None:
                if previous_monitor is None:
                    monitor_ok = False
                else:
                    scale = max(abs(previous_monitor), 1e-300)
                    monitor_ok = abs(value - previous_monitor) / scale <= monitor_tol
            previous_monitor = value

        report.iterations = iteration
        report.residual = residual
        if residual <= tol:
            if monitor_ok:
                report.converged = True
                report.reason = "tolerance"
                break
            report.reason = "monitor"

    report.wall_total = time.perf_counter() - started
    return rebuild(x_vec), report


def _observable(x: Any) -> float:
    """Default scalar summary of a state: the state itself, or its largest magnitude."""
    if np.isscalar(x) or isinstance(x, np.generic):
        return float(np.asarray(x, dtype=np.float64).reshape(-1)[0])
    array = np.asarray(x, dtype=np.float64)
    return float(np.max(np.abs(array))) if array.size else 0.0


def continuation_sweep(
    driver: Callable[[float, Any], Any],
    values: Sequence[float],
    both_directions: bool = True,
    x0: Any = None,
    key: Callable[[Any], float] | None = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
) -> dict[str, Any]:
    """Sweep a control parameter, each point started from its neighbour, and report the hysteresis.

    The upward sweep runs ``values`` in the given order starting from ``x0``; each point starts
    from the previous point's converged state. The downward sweep runs them in reverse, starting
    from the upward sweep's last state. Two states at the same control value that differ by more
    than the tolerance are two branches, which is the observable a bistable coupled device is
    graded on. The same machinery answers a different question for an optimisation — whether a
    design falls into another local optimum on a restart.

    Args:
        driver: ``driver(value, start)`` returning the converged state, or ``(state, info)``. It is
            the caller's job to make this a *small* continuation step with damped Picard; Anderson
            jumps branches (module docstring).
        values (Sequence[float]): The control parameter values, in sweep-up order.
        both_directions (bool): Run the downward sweep too. With ``False`` only ``"up"`` is filled
            and no hysteresis is reported.
        x0: The starting state for the first upward point.
        key: ``state -> float``, the scalar the two directions are compared on. Defaults to the
            state itself for a scalar and its largest magnitude otherwise.
        rtol (float): Relative tolerance for calling two branches different.
        atol (float): Absolute tolerance for the same.

    Returns:
        dict: ``{"values", "up", "down", "states_up", "states_down", "hysteresis"}``. Each of
        ``up`` and ``down`` is a list aligned with ``values`` holding
        ``{"value", "observable", "info"}``. ``hysteresis`` holds ``detected``, ``window``
        (the lowest and highest control value where the directions disagree, or ``None``),
        ``num_disagreeing``, ``max_relative_difference`` and ``max_ratio``.

    Raises:
        ValueError: If ``values`` is empty.
    """
    controls = [float(v) for v in values]
    if not controls:
        raise ValueError("values must hold at least one control value")
    observe = key or _observable

    def _drive(value: float, start: Any) -> tuple[Any, dict[str, Any] | None]:
        out = driver(value, start)
        if isinstance(out, tuple) and len(out) == 2:
            second = out[1]
            if isinstance(second, CouplingConvergenceReport):
                return out[0], second.as_dict()
            if isinstance(second, dict):
                return out[0], second
        return out, None

    states_up: list[Any] = []
    up: list[dict[str, Any]] = []
    start = x0
    for value in controls:
        state, info = _drive(value, start)
        states_up.append(state)
        up.append({"value": value, "observable": float(observe(state)), "info": info})
        start = state

    result: dict[str, Any] = {
        "values": controls,
        "up": up,
        "down": [],
        "states_up": states_up,
        "states_down": [],
        "hysteresis": {
            "detected": False,
            "window": None,
            "num_disagreeing": 0,
            "max_relative_difference": 0.0,
            "max_ratio": 1.0,
        },
    }
    if not both_directions:
        return result

    states_down: list[Any] = [None] * len(controls)
    down: list[dict[str, Any]] = [{} for _ in controls]
    start = states_up[-1]
    for index in range(len(controls) - 1, -1, -1):
        value = controls[index]
        state, info = _drive(value, start)
        states_down[index] = state
        down[index] = {"value": value, "observable": float(observe(state)), "info": info}
        start = state
    result["down"] = down
    result["states_down"] = states_down

    disagreeing: list[float] = []
    max_relative = 0.0
    max_ratio = 1.0
    for index, value in enumerate(controls):
        a = up[index]["observable"]
        b = down[index]["observable"]
        scale = max(abs(a), abs(b), 1e-300)
        relative = abs(a - b) / scale
        if abs(a - b) > atol + rtol * scale:
            disagreeing.append(value)
        max_relative = max(max_relative, relative)
        if min(abs(a), abs(b)) > 0.0:
            max_ratio = max(max_ratio, max(abs(a), abs(b)) / min(abs(a), abs(b)))
    result["hysteresis"] = {
        "detected": bool(disagreeing),
        "window": [min(disagreeing), max(disagreeing)] if disagreeing else None,
        "num_disagreeing": len(disagreeing),
        "max_relative_difference": float(max_relative),
        "max_ratio": float(max_ratio),
    }
    return result


def continuation_schedule(
    beta: float,
    alpha: float,
    iteration: int,
    every: int = 50,
    factor: float = 1.5,
    alpha_start: float = 1.5,
    beta_max: float | None = None,
    alpha_max: float | None = None,
) -> tuple[float, float]:
    """Projection sharpness and grayscale penalty at a given optimization iteration.

    The schedule of Jokisch, Christiansen & Sigmund (JOSA B 41(2), A18 (2024), Table 3): ``beta``
    starts at its initial value and is multiplied by ``factor`` every ``every`` iterations; the
    grayscale penalty ``alpha`` is zero until the first continuation step, jumps to ``alpha_start``
    there and is then multiplied by ``factor`` on every later step. Their published values are
    ``beta = 5`` (10 for the large-cladding study), ``factor = 1.5`` and ``alpha_start = 1.5``.

    Args:
        beta (float): Initial projection sharpness, the value used before the first step.
        alpha (float): Initial grayscale penalty; the paper starts it at zero.
        iteration (int): Zero-based iteration counter.
        every (int): Iterations between continuation steps.
        factor (float): Multiplier applied at each step.
        alpha_start (float): Value ``alpha`` takes at the first continuation step, when it starts
            from zero. Ignored if ``alpha`` is already non-zero, which is then ramped like ``beta``.
        beta_max (float | None): Optional cap on ``beta``.
        alpha_max (float | None): Optional cap on ``alpha``.

    Returns:
        tuple[float, float]: ``(beta, alpha)`` at that iteration.

    Raises:
        ValueError: If ``every`` is not positive or ``iteration`` is negative.
    """
    if every <= 0:
        raise ValueError("the continuation interval must be positive")
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    steps = int(iteration) // int(every)
    beta_now = float(beta) * factor**steps
    if alpha != 0.0:
        alpha_now = float(alpha) * factor**steps
    elif steps == 0:
        alpha_now = 0.0
    else:
        alpha_now = float(alpha_start) * factor ** (steps - 1)
    if beta_max is not None:
        beta_now = min(beta_now, float(beta_max))
    if alpha_max is not None:
        alpha_now = min(alpha_now, float(alpha_max))
    return beta_now, alpha_now
