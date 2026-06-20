"""Array bridge: ArrayContainer <-> MLXState.

Bridge IN converts the (already reset) JAX/numpy ArrayContainer to a plain-MLX state,
precomputing the time-invariant CPML coefficients on the host. Bridge OUT writes the MLX
results back into a copy of the template ArrayContainer so all downstream fdtdx code
(detector reading, plotting, S-params) consumes an indistinguishable result.
"""

from __future__ import annotations

import jax.numpy as jnp
import mlx.core as mx
import numpy as np

from fdtdx.constants import eps0
from fdtdx.mlx.pml import precompute_cpml_coeffs
from fdtdx.mlx.state import MLXState


def _to_mx(x) -> mx.array:
    return mx.array(np.ascontiguousarray(np.asarray(x)))


def _to_jnp(x) -> jnp.ndarray:
    return jnp.asarray(np.array(x))


def to_mlx_state(arrays, config) -> MLXState:
    """Convert a (reset) :class:`ArrayContainer` to an :class:`MLXState`."""
    dt = float(config.time_step_duration)
    a, b, inv_kappa = precompute_cpml_coeffs(
        np.asarray(arrays.alpha), np.asarray(arrays.kappa), np.asarray(arrays.sigma), dt, eps0
    )

    inv_mu = arrays.inv_permeabilities
    if hasattr(inv_mu, "ndim") and getattr(inv_mu, "ndim", 0) > 0:
        inv_mu_state = _to_mx(inv_mu)
    else:
        inv_mu_state = float(inv_mu)

    return MLXState(
        E=_to_mx(arrays.fields.E),
        H=_to_mx(arrays.fields.H),
        psi_E=_to_mx(arrays.fields.psi_E),
        psi_H=_to_mx(arrays.fields.psi_H),
        inv_eps=_to_mx(arrays.inv_permittivities),
        inv_mu=inv_mu_state,
        cpml_a=_to_mx(a),
        cpml_b=_to_mx(b),
        inv_kappa=_to_mx(inv_kappa),
        sigma_E=None if arrays.electric_conductivity is None else _to_mx(arrays.electric_conductivity),
        sigma_H=None if arrays.magnetic_conductivity is None else _to_mx(arrays.magnetic_conductivity),
    )


def buffers_to_detector_states(buffers: dict[str, dict[str, mx.array]]) -> dict[str, dict]:
    """Convert MLX detector buffers back to host (jnp) detector_states."""
    return {name: {key: _to_jnp(buf) for key, buf in bufs.items()} for name, bufs in buffers.items()}


def to_array_container(template_arrays, state: MLXState, detector_states=None):
    """Write MLX field results (and optional detector states) back into the container."""
    arrays = template_arrays
    arrays = arrays.aset("fields->E", _to_jnp(state.E))
    arrays = arrays.aset("fields->H", _to_jnp(state.H))
    arrays = arrays.aset("fields->psi_E", _to_jnp(state.psi_E))
    arrays = arrays.aset("fields->psi_H", _to_jnp(state.psi_H))
    if detector_states is not None:
        arrays = arrays.aset("detector_states", detector_states)
    return arrays
