from __future__ import annotations

from collections.abc import Callable

import jax

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.default_key import default_key
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, SimulationState
from fdtdx.fdtd.fdtd import checkpointed_fdtd, reversible_fdtd
from fdtdx.fdtd.stop_conditions import StoppingCondition


def run_fdtd(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array | None = None,
    stopping_condition: StoppingCondition | None = None,
    show_progress: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SimulationState:
    # FDTDMEX MLX (Metal) injection: on Apple Silicon, a supported forward-only run is
    # handed to the native MLX time loop. Returns a completed SimulationState, or None to
    # fall through to the unchanged JAX engine below. Local import avoids a load-time cycle.
    from fdtdx.backend.dispatch import maybe_run_mlx_forward

    _mlx_state = maybe_run_mlx_forward(arrays, objects, config, key, stopping_condition)
    if _mlx_state is not None:
        return _mlx_state

    key = default_key(key)
    if stopping_condition is not None:
        if config.gradient_config is not None:
            raise NotImplementedError(
                "Custom stopping conditions are not yet compatible with gradient computation. "
                "Set config.gradient_config to None or use default time-based stopping by "
                "setting stopping_condition=None."
            )

    if config.gradient_config is None:
        # only forward simulation, use standard while loop of checkpointed fdtd
        return checkpointed_fdtd(
            arrays=arrays,
            objects=objects,
            config=config,
            key=key,
            stopping_condition=stopping_condition,
            show_progress=show_progress,
            progress_callback=progress_callback,
        )
    if config.gradient_config.method == "reversible":
        return reversible_fdtd(
            arrays=arrays,
            objects=objects,
            config=config,
            key=key,
            show_progress=show_progress,
            progress_callback=progress_callback,
        )
    elif config.gradient_config.method == "checkpointed":
        return checkpointed_fdtd(
            arrays=arrays,
            objects=objects,
            config=config,
            key=key,
            stopping_condition=stopping_condition,
            show_progress=show_progress,
            progress_callback=progress_callback,
        )
    else:
        raise Exception(f"Unknown gradient computation method: {config.gradient_config.method}")
