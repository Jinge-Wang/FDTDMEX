"""The MLX forward time-loop driver.

Plain Python ``for`` loop over time steps; per step mirrors ``fdtdx.fdtd.forward.forward``:
update E -> inject electric sources -> update H -> inject magnetic sources -> record detectors.
Detectors see interpolated E and the interpolated time-averaged H (H_prev is the H captured at the
start of the step), matching ``update_detector_states``.

The per-step E-core and H-core (curl -> field update -> CPML recurrence) are wrapped in
``mx.compile`` so the chain of element-wise ops fuses into few kernels instead of each op streaming
to DRAM. **Source injection stays host-gated and runs *between* the two compiled cores** (E-core ->
inject E -> H-core -> inject H), so (a) the compiled graph is static across steps and (b) the
leapfrog ordering is preserved exactly: the H update reads the source-injected E^{n+1}. The cores
are functional (return new arrays), keeping the Yee update race-free. The lazy MLX graph is bounded
with periodic ``mx.eval``.
"""

from __future__ import annotations

import mlx.core as mx

from fdtdx.mlx.accumulate import update_detectors
from fdtdx.mlx.curl import pad_fields_mlx
from fdtdx.mlx.detector_freeze import DetectorPlan
from fdtdx.mlx.inject import inject_sources_E, inject_sources_H
from fdtdx.mlx.interpolate import interpolate_fields_mlx
from fdtdx.mlx.source_freeze import SourcePlan
from fdtdx.mlx.state import MLXState
from fdtdx.mlx.update import _update_E, _update_H


def _build_cores(state: MLXState, c: float, sb: bool, compile_step: bool):
    """Build the per-step E-core/H-core. Time-invariant arrays are captured as constants so the
    only graph inputs are the time-varying fields (E, H, ψ); ``mx.compile`` then fuses the body."""
    inv_eps, sigma_E = state.inv_eps, state.sigma_E
    inv_mu, sigma_H = state.inv_mu, state.sigma_H
    a, b, ik = state.cpml_a, state.cpml_b, state.inv_kappa
    mfwd, mbwd = state.metric_fwd, state.metric_bwd
    per, awid, ext = state.periodic_axes, state.aniso_widths, state.cpml_extents

    def e_core(E, H, psi_E):
        return _update_E(E, H, psi_E, inv_eps, sigma_E, a, b, ik, mbwd, per, ext, awid, c, sb)

    def h_core(E, H, psi_H):
        return _update_H(E, H, psi_H, inv_mu, sigma_H, a, b, ik, mfwd, per, ext, awid, c, sb)

    if compile_step:
        return mx.compile(e_core), mx.compile(h_core)
    return e_core, h_core


def run_forward_mlx(
    state: MLXState,
    source_plans: list[SourcePlan],
    detector_plans: list[DetectorPlan],
    detector_buffers: dict[str, dict[str, mx.array]],
    num_steps: int,
    c: float,
    simulate_boundaries: bool = True,
    eval_every: int = 8,
    compile_step: bool = True,
) -> tuple[MLXState, dict[str, dict[str, mx.array]]]:
    """Advance ``state`` ``num_steps`` steps, recording detectors; return state + buffers."""
    record = bool(detector_plans)
    e_core, h_core = _build_cores(state, c, simulate_boundaries, compile_step)

    for n in range(num_steps):
        H_prev = state.H

        E, psi_E = e_core(state.E, state.H, state.psi_E)
        E = inject_sources_E(E, source_plans, n)
        state.E = E
        state.psi_E = psi_E

        H, psi_H = h_core(state.E, state.H, state.psi_H)
        H = inject_sources_H(H, source_plans, n)
        state.H = H
        state.psi_H = psi_H

        if record:
            E_interp, H_interp = interpolate_fields_mlx(
                pad_fields_mlx(state.E, state.periodic_axes),
                pad_fields_mlx((H_prev + state.H) / 2.0, state.periodic_axes),
                state.interp_widths,
            )
            update_detectors(
                detector_plans, detector_buffers, E_interp, H_interp, state.E, state.H, state.inv_eps, state.inv_mu, n
            )

        if (n + 1) % eval_every == 0:
            leaves = [state.E, state.H, *state.psi_E, *state.psi_H]
            for bufs in detector_buffers.values():
                leaves.extend(bufs.values())
            mx.eval(*leaves)

    leaves = [state.E, state.H, state.psi_E, state.psi_H]
    for bufs in detector_buffers.values():
        leaves.extend(bufs.values())
    mx.eval(*leaves)
    return state, detector_buffers
