"""The MLX forward time-loop driver.

Plain Python ``for`` loop over time steps; per step mirrors ``fdtdx.fdtd.forward.forward``:
update E -> inject electric sources -> update H -> inject magnetic sources -> record
detectors. Detectors see interpolated E and the interpolated time-averaged H (H_prev is the
H captured at the start of the step), matching ``update_detector_states``. The lazy MLX
graph is bounded with periodic ``mx.eval``. ``mx.compile`` of the step body is a later
optimization; M1 runs eager for clarity / debuggability.
"""

from __future__ import annotations

import mlx.core as mx

from fdtdx.mlx.accumulate import update_detectors
from fdtdx.mlx.curl import pad_zero
from fdtdx.mlx.detector_freeze import DetectorPlan
from fdtdx.mlx.inject import inject_sources_E, inject_sources_H
from fdtdx.mlx.interpolate import interpolate_fields_mlx
from fdtdx.mlx.source_freeze import SourcePlan
from fdtdx.mlx.state import MLXState
from fdtdx.mlx.update import update_E_mlx, update_H_mlx


def run_forward_mlx(
    state: MLXState,
    source_plans: list[SourcePlan],
    detector_plans: list[DetectorPlan],
    detector_buffers: dict[str, dict[str, mx.array]],
    num_steps: int,
    c: float,
    simulate_boundaries: bool = True,
    eval_every: int = 8,
) -> tuple[MLXState, dict[str, dict[str, mx.array]]]:
    """Advance ``state`` ``num_steps`` steps, recording detectors; return state + buffers."""
    record = bool(detector_plans)

    for n in range(num_steps):
        H_prev = state.H

        E, psi_E = update_E_mlx(state, c, simulate_boundaries)
        E = inject_sources_E(E, source_plans, n)
        state.E = E
        state.psi_E = psi_E

        H, psi_H = update_H_mlx(state, c, simulate_boundaries)
        H = inject_sources_H(H, source_plans, n)
        state.H = H
        state.psi_H = psi_H

        if record:
            E_interp, H_interp = interpolate_fields_mlx(pad_zero(state.E), pad_zero((H_prev + state.H) / 2.0))
            update_detectors(
                detector_plans, detector_buffers, E_interp, H_interp, state.E, state.H, state.inv_eps, state.inv_mu, n
            )

        if (n + 1) % eval_every == 0:
            leaves = [state.E, state.H, state.psi_E, state.psi_H]
            for bufs in detector_buffers.values():
                leaves.extend(bufs.values())
            mx.eval(*leaves)

    leaves = [state.E, state.H, state.psi_E, state.psi_H]
    for bufs in detector_buffers.values():
        leaves.extend(bufs.values())
    mx.eval(*leaves)
    return state, detector_buffers
