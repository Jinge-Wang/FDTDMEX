"""The MLX forward time-loop driver.

Plain Python ``for`` loop over time steps; per step mirrors ``fdtdx.fdtd.forward.forward``:
update E -> inject electric sources -> update H -> inject magnetic sources -> (record
detectors). The lazy MLX graph is bounded with periodic ``mx.eval``. ``mx.compile`` of the
step body is a later optimization; M1 runs eager for clarity / debuggability.
"""

from __future__ import annotations

import mlx.core as mx

from fdtdx.mlx.inject import inject_sources_E, inject_sources_H
from fdtdx.mlx.source_freeze import SourcePlan
from fdtdx.mlx.state import MLXState
from fdtdx.mlx.update import update_E_mlx, update_H_mlx


def run_forward_mlx(
    state: MLXState,
    source_plans: list[SourcePlan],
    num_steps: int,
    c: float,
    simulate_boundaries: bool = True,
    eval_every: int = 8,
) -> MLXState:
    """Advance ``state`` ``num_steps`` steps in place and return it."""
    for n in range(num_steps):
        E, psi_E = update_E_mlx(state, c, simulate_boundaries)
        E = inject_sources_E(E, source_plans, n)
        state.E = E
        state.psi_E = psi_E

        H, psi_H = update_H_mlx(state, c, simulate_boundaries)
        H = inject_sources_H(H, source_plans, n)
        state.H = H
        state.psi_H = psi_H

        if (n + 1) % eval_every == 0:
            mx.eval(state.E, state.H, state.psi_E, state.psi_H)

    mx.eval(state.E, state.H, state.psi_E, state.psi_H)
    return state
