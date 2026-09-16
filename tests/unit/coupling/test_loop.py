"""Fixed-point drivers on the scalar bistable map the two-way thermal case is built around.

The map is the etalon fixed point of ``W1_design.md`` section 6 reduced to one dimension: the
absorbed fraction of an absorbing Fabry-Perot etalon is an Airy comb in the round-trip phase, the
absorbed power heats the slab, and the thermo-optic coefficient moves the phase, so

    dT = G(dT) = P / (1 + F sin^2(pi (dT - dT_offset) / dT_FSR))

For a large enough drive ``P`` this has three fixed points over one thermal free spectral range and
the device is bistable. Every reference number below is computed from the map itself by dense
bracketing and bisection, not from an engine, so the checks are exact statements about the drivers:
damped Picard reaches the branch the start belongs to, plain Picard fails where Anderson succeeds,
and the continuation sweep reports the hysteresis window.
"""

import json

import numpy as np
import pytest

from fdtdx.coupling.loop import (
    CouplingConvergenceReport,
    alternating_solver,
    continuation_sweep,
    coupling_convergence_report,
)

DT_FSR = 41.667  # K, the thermal free spectral range of the Phase 1 etalon
FINESSE_F = 3.0  # the Airy coefficient of the reduced map
DT_OFFSET = 18.0  # K, the temperature rise that puts the etalon on resonance

#: Drives inside the bistable window of the reduced map, found by counting its roots.
BISTABLE_WINDOW = (17.09, 19.68)
GATE_DRIVE = 18.4  # inside the window; roots 6.9790, 12.8106, 18.3595

#: A steeper member of the same family whose single fixed point has G' = -6.52, so damped Picard
#: diverges at every damping from 1.0 down to 0.3 (|1 - mix (1 - G')| > 1) and Anderson does not.
STEEP = {"drive": 45.0, "finesse": 100.0, "offset": 8.0}


def etalon_map(drive: float, finesse: float = FINESSE_F, offset: float = DT_OFFSET):
    """One outer iteration of the reduced etalon loop, ``dT -> dT``."""

    def g(dT: float) -> float:
        phase = np.pi * (float(dT) - offset) / DT_FSR
        return float(drive / (1.0 + finesse * np.sin(phase) ** 2))

    return g


def fixed_points(
    drive: float,
    finesse: float = FINESSE_F,
    offset: float = DT_OFFSET,
    lo: float = 0.0,
    hi: float = 60.0,
    samples: int = 400_001,
):
    """Every root of ``G(x) - x`` on ``[lo, hi]``, by dense bracketing plus bisection."""
    g = etalon_map(drive, finesse, offset)
    grid = np.linspace(lo, hi, samples)
    values = drive / (1.0 + finesse * np.sin(np.pi * (grid - offset) / DT_FSR) ** 2) - grid
    roots = []
    for index in np.nonzero(values[:-1] * values[1:] < 0.0)[0]:
        a, b, fa = float(grid[index]), float(grid[index + 1]), float(values[index])
        for _ in range(80):
            m = 0.5 * (a + b)
            fm = g(m) - m
            if fa * fm <= 0.0:
                b = m
            else:
                a, fa = m, fm
        roots.append(0.5 * (a + b))
    return roots


# ---------------------------------------------------------------------------
# The map itself: the reference the driver tests are graded against
# ---------------------------------------------------------------------------


def test_the_map_is_bistable_at_the_gate_drive_and_single_valued_outside_the_window():
    roots = fixed_points(GATE_DRIVE)
    assert len(roots) == 3, roots
    lower, middle, upper = roots
    assert (lower, middle, upper) == pytest.approx((6.9790, 12.8106, 18.3595), abs=1e-3)
    # the two coexisting states differ by much more than any tolerance
    assert upper / lower == pytest.approx(2.631, abs=1e-2)
    # the middle root is the unstable one and the outer two are stable under Picard
    g = etalon_map(GATE_DRIVE)
    h = 1e-6

    def slope(x):
        return (g(x + h) - g(x - h)) / (2 * h)

    assert abs(slope(middle)) > 1.0
    assert abs(slope(lower)) < 1.0 and abs(slope(upper)) < 1.0
    assert len(fixed_points(BISTABLE_WINDOW[0] - 0.2)) == 1
    assert len(fixed_points(BISTABLE_WINDOW[1] + 0.2)) == 1


# ---------------------------------------------------------------------------
# alternating_solver
# ---------------------------------------------------------------------------


def test_damped_picard_reaches_the_branch_its_start_belongs_to():
    drive = GATE_DRIVE
    lower, _, upper = fixed_points(drive)
    step = etalon_map(drive)

    cold, cold_report = alternating_solver(step, 0.0, mix=0.5, tol=1e-12, max_iter=500, branch="cold")
    hot, hot_report = alternating_solver(step, 18.0, mix=0.5, tol=1e-12, max_iter=500, branch="hot")

    assert cold_report.converged and hot_report.converged
    assert cold == pytest.approx(lower, abs=1e-9)
    assert hot == pytest.approx(upper, abs=1e-9)
    assert cold_report.branch == "cold" and hot_report.branch == "hot"
    assert cold_report.residual <= 1e-12
    assert len(cold_report.residual_history) == cold_report.iterations
    assert cold_report.residual_history[-1] < cold_report.residual_history[0]


def test_anderson_converges_where_plain_picard_and_every_damping_fail():
    """The Phase 1 finding, reproduced on the steep member of the same family."""
    step = etalon_map(STEEP["drive"], STEEP["finesse"], STEEP["offset"])
    roots = fixed_points(STEEP["drive"], STEEP["finesse"], STEEP["offset"])
    assert len(roots) == 1
    assert roots[0] == pytest.approx(10.4284, abs=1e-3)

    start = 12.0
    picard = {}
    for mix in (1.0, 0.7, 0.5, 0.3):
        _, report = alternating_solver(step, start, mix=mix, tol=1e-10, max_iter=500)
        picard[mix] = report.converged
    assert not any(picard.values()), picard

    for depth in (3, 5):
        x, report = alternating_solver(step, start, mix=1.0, anderson=depth, tol=1e-10, max_iter=500)
        assert report.converged, f"Anderson depth {depth} did not converge"
        assert report.iterations < 200
        assert x == pytest.approx(roots[0], abs=1e-8)
        assert step(x) == pytest.approx(x, abs=1e-9)
        assert report.mixing == {"mix": 1.0, "anderson": depth}


def test_anderson_does_not_respect_the_branch():
    """Why the continuation must step with damped Picard: Anderson jumps between branches."""
    drive = GATE_DRIVE
    lower, _, upper = fixed_points(drive)
    step = etalon_map(drive)
    picard_cold, _ = alternating_solver(step, 0.0, mix=0.5, tol=1e-12, max_iter=500)
    picard_hot, _ = alternating_solver(step, 18.0, mix=0.5, tol=1e-12, max_iter=500)
    assert picard_cold == pytest.approx(lower, abs=1e-9)
    assert picard_hot == pytest.approx(upper, abs=1e-9)

    landed = set()
    for start in np.linspace(0.0, 20.0, 11):
        x, report = alternating_solver(step, float(start), mix=1.0, anderson=3, tol=1e-11, max_iter=300)
        if report.converged:
            landed.add(round(x, 6))
    assert len(landed) >= 2, f"Anderson landed only on {landed}"


def test_a_vector_state_keeps_its_shape_and_a_linear_map_converges():
    matrix = np.array([[0.5, 0.1], [0.0, 0.4]])
    offset = np.array([1.0, 2.0])
    exact = np.linalg.solve(np.eye(2) - matrix, offset)
    x, report = alternating_solver(lambda v: matrix @ v + offset, np.zeros(2), mix=1.0, tol=1e-13)
    assert report.converged
    assert x.shape == (2,)
    np.testing.assert_allclose(x, exact, atol=1e-11)

    grid = np.zeros((2, 3))
    y, _ = alternating_solver(lambda v: 0.5 * v + 1.0, grid, mix=1.0, tol=1e-13)
    assert y.shape == (2, 3)
    np.testing.assert_allclose(y, 2.0, atol=1e-11)


def test_the_step_may_report_its_sub_solves_and_a_monitor_gates_convergence():
    drive = GATE_DRIVE
    g = etalon_map(drive)
    walls = []

    def step(x):
        walls.append(len(walls))
        return g(x), {"em_wall": 0.03, "thermal_wall": 0.007, "iteration": len(walls)}

    _, report = alternating_solver(step, 0.0, mix=0.5, tol=1e-12, max_iter=500, monitor=lambda v: g(v) / drive)
    assert report.converged
    assert len(report.sub_solve_info) == report.iterations
    assert report.sub_solve_info[0]["em_wall"] == 0.03
    assert len(report.monitor_history) == report.iterations
    assert len(report.wall_per_iteration) == report.iterations
    assert report.wall_total >= 0.0

    # a monitor tolerance that can never be met keeps the run going to the cap
    _, gated = alternating_solver(
        etalon_map(drive),
        0.0,
        mix=0.5,
        tol=1e-12,
        max_iter=25,
        monitor=lambda v: float(np.random.default_rng(int(v * 1e6) % 97).normal()),
        monitor_tol=1e-30,
    )
    assert not gated.converged
    assert gated.reason in ("monitor", "max_iter")


def test_the_solver_refuses_a_nonsense_configuration_and_reports_a_blow_up():
    step = etalon_map(GATE_DRIVE)
    for kwargs in ({"mix": 0.0}, {"mix": 1.5}, {"anderson": 0}, {"tol": 0.0}, {"max_iter": 0}):
        with pytest.raises(ValueError):
            alternating_solver(step, 0.0, **kwargs)
    _, report = alternating_solver(lambda x: 10.0 * x + 1.0, 1.0, mix=1.0, tol=1e-12, max_iter=2000)
    assert not report.converged
    assert report.reason in ("not_finite", "max_iter")


# ---------------------------------------------------------------------------
# The report block
# ---------------------------------------------------------------------------


def test_the_convergence_block_is_json_serialisable_and_carries_the_agreed_keys():
    _, report = alternating_solver(etalon_map(GATE_DRIVE), 0.0, mix=0.5, tol=1e-12, branch="up")
    block = coupling_convergence_report(report, extra={"drive": GATE_DRIVE, "engine": "waveEMFDFD@1330c2a"})
    for key in (
        "converged",
        "iterations",
        "residual",
        "residual_history",
        "wall_per_iteration",
        "wall_total",
        "sub_solve_info",
        "mixing",
        "branch",
    ):
        assert key in block
    assert block["branch"] == "up"
    assert block["drive"] == GATE_DRIVE
    assert json.loads(json.dumps(block))["engine"] == "waveEMFDFD@1330c2a"
    assert coupling_convergence_report(CouplingConvergenceReport(), branch="down")["branch"] == "down"


# ---------------------------------------------------------------------------
# continuation_sweep
# ---------------------------------------------------------------------------


def _picard_driver(value, start):
    x, report = alternating_solver(
        etalon_map(value), 0.0 if start is None else start, mix=0.4, tol=1e-11, max_iter=4000
    )
    return x, report


def test_the_sweep_reports_the_hysteresis_window_the_map_actually_has():
    drives = [float(v) for v in np.arange(14.0, 24.01, 0.5)]
    sweep = continuation_sweep(_picard_driver, drives, both_directions=True, x0=0.0)

    hysteresis = sweep["hysteresis"]
    assert hysteresis["detected"]
    assert hysteresis["num_disagreeing"] > 3
    assert hysteresis["max_ratio"] > 2.0

    # every drive the sweep flags as two-valued really has three fixed points, and every drive it
    # calls single-valued really has one
    lo, hi = hysteresis["window"]
    for record_up, record_down in zip(sweep["up"], sweep["down"]):
        drive = record_up["value"]
        n_roots = len(fixed_points(drive))
        two_valued = abs(record_up["observable"] - record_down["observable"]) > 1e-6 * max(
            abs(record_up["observable"]), abs(record_down["observable"])
        )
        if two_valued:
            # three roots inside the window; a drive sitting on a fold shows the tangency as two
            assert n_roots >= 2, f"drive {drive}: sweep says two branches, the map has {n_roots} roots"
            assert lo - 1e-12 <= drive <= hi + 1e-12
    assert len(fixed_points(lo)) >= 2 and len(fixed_points(hi)) >= 2
    assert BISTABLE_WINDOW[0] - 1.0 <= lo and hi <= BISTABLE_WINDOW[1] + 1.0
    # the sweep's window sits inside the map's own bistable range
    single_below = [d for d in drives if d < lo and len(fixed_points(d)) == 1]
    assert single_below, "expected a single-valued region below the window"


def test_the_sweep_reuses_the_previous_solution_and_keeps_the_states():
    drives = [10.0, 12.0, 14.0]
    starts = []

    def driver(value, start):
        starts.append(start)
        x, _ = alternating_solver(etalon_map(value), 0.0 if start is None else start, mix=0.4, tol=1e-11)
        return x

    sweep = continuation_sweep(driver, drives, both_directions=True, x0=0.0)
    assert starts[0] == 0.0
    assert starts[1] == sweep["states_up"][0]
    assert starts[3] == sweep["states_up"][-1]  # the downward sweep starts hot
    assert len(sweep["states_up"]) == len(sweep["states_down"]) == 3
    assert [r["value"] for r in sweep["down"]] == drives  # aligned with `values`, not reversed


def test_a_one_directional_sweep_reports_no_hysteresis_and_an_empty_sweep_is_refused():
    sweep = continuation_sweep(_picard_driver, [10.0, 12.0], both_directions=False, x0=0.0)
    assert sweep["down"] == []
    assert sweep["hysteresis"]["detected"] is False
    with pytest.raises(ValueError, match="at least one control value"):
        continuation_sweep(_picard_driver, [], x0=0.0)


def test_a_monotone_map_shows_no_hysteresis_in_either_direction():
    def driver(value, start):
        x, _ = alternating_solver(lambda t: 0.5 * t + value, 0.0 if start is None else start, tol=1e-13)
        return x

    sweep = continuation_sweep(driver, [1.0, 2.0, 3.0, 4.0], both_directions=True, x0=0.0)
    assert not sweep["hysteresis"]["detected"]
    assert sweep["hysteresis"]["max_relative_difference"] < 1e-9
    np.testing.assert_allclose([r["observable"] for r in sweep["up"]], [2.0, 4.0, 6.0, 8.0], atol=1e-10)
