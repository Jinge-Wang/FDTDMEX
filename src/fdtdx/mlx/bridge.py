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

from fdtdx.constants import c as c_light
from fdtdx.constants import eps0
from fdtdx.mlx.curl import _AX, _slab_take, slab_to_full
from fdtdx.mlx.pml import detect_pml_slabs, precompute_cpml_coeffs
from fdtdx.mlx.state import MLXState


def _to_mx(x) -> mx.array:
    return mx.array(np.ascontiguousarray(np.asarray(x)))


def _to_jnp(x) -> jnp.ndarray:
    return jnp.asarray(np.array(x))


def _broadcast_axis(arr: np.ndarray, axis: int) -> np.ndarray:
    """Reshape a 1-D per-cell array so it broadcasts along ``axis`` of a (Nx, Ny, Nz) field."""
    shape = [1, 1, 1]
    shape[axis] = arr.shape[0]
    return np.ascontiguousarray(arr.reshape(shape).astype(np.float32))


def _grid_metrics(config, periodic_axes):
    """Precompute non-uniform-grid metric scales and interpolation/averaging weights.

    Returns ``(metric_fwd, metric_bwd, interp_widths, aniso_widths)``. On a uniform grid the
    metric tuples are scalar ``1.0`` (the curl fast path) and the weight tables are ``None``
    (plain means), so the uniform code path is byte-for-byte the M3 path.

    Ports ``fdtdx.core.physics.curl._metric_scale`` (forward stencil for ``curl_E``, backward /
    dual-width stencil for ``curl_H``) and ``_backward_edge_average``'s half-width weights, and
    adds padded per-axis cell widths for the genuinely-new spacing-weighted anisotropic average.
    """
    if not config.has_nonuniform_grid:
        return (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), None, None

    grid = config.resolved_grid
    assert grid is not None
    reference_spacing = float(c_light * config.time_step_duration / config.courant_number)

    metric_fwd: list = []
    metric_bwd: list = []
    interp_widths: list = []
    aniso_widths: list = []
    for axis in range(3):
        widths = np.asarray(grid.cell_widths(axis), dtype=np.float64)  # (N,)
        prev = np.concatenate([widths[:1], widths[:-1]])
        dual = 0.5 * (widths + prev)

        metric_fwd.append(_to_mx(_broadcast_axis(reference_spacing / widths, axis)))
        metric_bwd.append(_to_mx(_broadcast_axis(reference_spacing / dual, axis)))
        interp_widths.append((_to_mx(_broadcast_axis(0.5 * widths, axis)), _to_mx(_broadcast_axis(0.5 * prev, axis))))

        # Padded (length N+2) widths for the off-diagonal aniso average, matching the field
        # padding: wrap on periodic axes, replicate the edge width on zero-padded (PML) axes.
        if periodic_axes[axis]:
            wpad = np.concatenate([widths[-1:], widths, widths[:1]])
        else:
            wpad = np.concatenate([widths[:1], widths, widths[-1:]])
        aniso_widths.append(_to_mx(_broadcast_axis(wpad, axis)))

    return tuple(metric_fwd), tuple(metric_bwd), tuple(interp_widths), tuple(aniso_widths)


def to_mlx_state(arrays, config, periodic_axes: tuple = (False, False, False), objects=None) -> MLXState:
    """Convert a (reset) :class:`ArrayContainer` to an :class:`MLXState`.

    ``objects`` (the :class:`ObjectContainer`) is used only to freeze the PEC/PMC keep-masks (Phase
    3); pass ``None`` to skip them.
    """
    dt = float(config.time_step_duration)
    a, b, inv_kappa = precompute_cpml_coeffs(
        np.asarray(arrays.alpha), np.asarray(arrays.kappa), np.asarray(arrays.sigma), dt, eps0
    )
    # slab-CPML: ψ and the CPML correction are confined to the PML boundary slabs. Detect each
    # axis's slab extent, then carry ψ only there (component i on the slabs perpendicular to _AX[i]).
    extents = tuple(detect_pml_slabs(a, b, inv_kappa))
    psi_E_full = _to_mx(arrays.fields.psi_E)
    psi_H_full = _to_mx(arrays.fields.psi_H)
    psi_E_slabs = tuple(_slab_take(psi_E_full[i], _AX[i], *extents[_AX[i]]) for i in range(6))
    psi_H_slabs = tuple(_slab_take(psi_H_full[i], _AX[i], *extents[_AX[i]]) for i in range(6))

    inv_mu = arrays.inv_permeabilities
    if hasattr(inv_mu, "ndim") and getattr(inv_mu, "ndim", 0) > 0:
        inv_mu_state = _to_mx(inv_mu)
    else:
        inv_mu_state = float(inv_mu)

    metric_fwd, metric_bwd, interp_widths, aniso_widths = _grid_metrics(config, periodic_axes)

    pec_keep = pmc_keep = None
    if objects is not None:
        from fdtdx.mlx.boundary_mask import freeze_boundary_masks

        pec_keep, pmc_keep = freeze_boundary_masks(objects, tuple(np.asarray(arrays.fields.E).shape))

    # Drude-Lorentz (ADE) dispersion: c1/c2/c3 are time-invariant (precomputed on the JAX side in
    # initialization), P_curr/P_prev are zeroed by reset(). Carry all five as mx.arrays; absent (None)
    # for non-dispersive simulations, leaving the whole ADE path inert.
    disp_c1 = disp_c2 = disp_c3 = disp_P_curr = disp_P_prev = None
    if arrays.dispersive_c1 is not None:
        disp_c1 = _to_mx(arrays.dispersive_c1)
        disp_c2 = _to_mx(arrays.dispersive_c2)
        disp_c3 = _to_mx(arrays.dispersive_c3)
        disp_P_curr = _to_mx(arrays.dispersive_P_curr)
        disp_P_prev = _to_mx(arrays.dispersive_P_prev)

    return MLXState(
        E=_to_mx(arrays.fields.E),
        H=_to_mx(arrays.fields.H),
        psi_E=psi_E_slabs,
        psi_H=psi_H_slabs,
        inv_eps=_to_mx(arrays.inv_permittivities),
        inv_mu=inv_mu_state,
        cpml_a=_to_mx(a),
        cpml_b=_to_mx(b),
        inv_kappa=_to_mx(inv_kappa),
        sigma_E=None if arrays.electric_conductivity is None else _to_mx(arrays.electric_conductivity),
        sigma_H=None if arrays.magnetic_conductivity is None else _to_mx(arrays.magnetic_conductivity),
        periodic_axes=periodic_axes,
        cpml_extents=extents,
        metric_fwd=metric_fwd,
        metric_bwd=metric_bwd,
        interp_widths=interp_widths,
        aniso_widths=aniso_widths,
        pec_keep=pec_keep,
        pmc_keep=pmc_keep,
        dispersive_c1=disp_c1,
        dispersive_c2=disp_c2,
        dispersive_c3=disp_c3,
        dispersive_P_curr=disp_P_curr,
        dispersive_P_prev=disp_P_prev,
    )


def buffers_to_detector_states(buffers: dict[str, dict[str, mx.array]]) -> dict[str, dict]:
    """Convert MLX detector buffers back to host (jnp) detector_states."""
    return {name: {key: _to_jnp(buf) for key, buf in bufs.items()} for name, bufs in buffers.items()}


def to_array_container(template_arrays, state: MLXState, detector_states=None):
    """Write MLX field results (and optional detector states) back into the container."""
    arrays = template_arrays
    arrays = arrays.aset("fields->E", _to_jnp(state.E))
    arrays = arrays.aset("fields->H", _to_jnp(state.H))
    # slab-CPML: scatter the per-component ψ slabs back into full (6, Nx, Ny, Nz) arrays (zeros in
    # the interior) so the host container is indistinguishable from the full-domain engine.
    shape = state.E.shape  # (3, Nx, Ny, Nz)
    ext = state.cpml_extents
    psi_E_full = mx.stack([slab_to_full(state.psi_E[i], _AX[i], *ext[_AX[i]], shape[1 + _AX[i]]) for i in range(6)])
    psi_H_full = mx.stack([slab_to_full(state.psi_H[i], _AX[i], *ext[_AX[i]], shape[1 + _AX[i]]) for i in range(6)])
    arrays = arrays.aset("fields->psi_E", _to_jnp(psi_E_full))
    arrays = arrays.aset("fields->psi_H", _to_jnp(psi_H_full))
    # Dispersive (ADE) polarization write-back — keeps the host container's state consistent with the
    # full-domain JAX engine (E/H/detectors are the parity bar; this is for completeness/resumption).
    if state.dispersive_P_curr is not None:
        arrays = arrays.aset("dispersive_P_curr", _to_jnp(state.dispersive_P_curr))
        arrays = arrays.aset("dispersive_P_prev", _to_jnp(state.dispersive_P_prev))
    if detector_states is not None:
        arrays = arrays.aset("detector_states", detector_states)
    return arrays
