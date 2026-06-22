"""MLX E/H field updates (isotropic/diagonal fast path + full-anisotropic 9-tensor path).

Translation of ``fdtdx.fdtd.update.update_E`` / ``update_H``. The non-9-tensor branch is the
component-wise fast path; the 9-tensor branch builds per-cell A/B matrices and uses the
spacing-weighted off-diagonal averaging (see :mod:`fdtdx.mlx.aniso`). Each curl is metric-scaled
per axis (``metric_bwd``/``metric_fwd``; ``1.0`` on uniform grids) and computed pad-free
(:mod:`fdtdx.mlx.curl`). Source injection and detector recording are handled by the loop driver.
Conductivity follows Schneider ch. 3.12.

``update_E_mlx``/``update_H_mlx`` are thin wrappers over the pure array functions ``_update_E`` /
``_update_H`` (all inputs explicit), so the loop driver can wrap those in ``mx.compile`` — the
time-invariant material/CPML/metric arrays become captured constants and only E/H/ψ flow through.
"""

from __future__ import annotations

import mlx.core as mx

from fdtdx.constants import eta0
from fdtdx.mlx.aniso import (
    avg_anisotropic_E_component_mlx,
    avg_anisotropic_H_component_mlx,
    compute_anisotropic_update_matrices_mlx,
    expand_to_3x3_mlx,
)
from fdtdx.mlx.curl import curl_E_mlx, curl_H_mlx, pad_fields_mlx
from fdtdx.mlx.state import MLXState


def _is_full_tensor(arr) -> bool:
    return isinstance(arr, mx.array) and arr.ndim > 0 and arr.shape[0] == 9


def _update_E(
    E,
    H,
    psi_E,
    inv_eps,
    sigma_E,
    cpml_a,
    cpml_b,
    inv_kappa,
    metric_bwd,
    periodic_axes,
    extents,
    aniso_widths,
    c,
    sb,
    disp_c1=None,
    disp_c2=None,
    disp_c3=None,
    P_curr=None,
    P_prev=None,
):
    """Pure E update from ``dE/dt = (1/eps) curl(H)``. All inputs explicit (compilable).

    Returns ``(E_new, psi_E_new)`` normally, or ``(E_new, psi_E_new, P_curr_new, P_prev_new)`` when
    Drude-Lorentz dispersion is active (``P_curr is not None``). Dispersion is only ever iso/diagonal
    (fdtdx forbids it with off-diagonal tensors), so the ADE term lives only in the fast path; the
    ``P_curr is not None`` guard is evaluated at trace time, so the non-dispersive graph is unchanged.
    """
    curl, psi_E_new = curl_H_mlx(H, psi_E, cpml_a, cpml_b, inv_kappa, sb, metric_bwd, periodic_axes, extents)

    if not _is_full_tensor(inv_eps) and not _is_full_tensor(sigma_E):
        # E^n is needed by the ADE recurrence (fdtdx uses the *pre-update* field), so keep it before
        # overwriting E with the curl update.
        E_old = E
        factor = 1.0
        if sigma_E is not None:
            factor = 1.0 - c * sigma_E * eta0 * inv_eps / 2.0
        E = factor * E_old + c * curl * inv_eps
        if P_curr is not None:
            # ADE: per-pole P_new = c1*P_curr + c2*P_prev + c3*E^n; back-action E += inv_eps*Σ(P_curr-P_new).
            # disp_c* are (poles,1,N,N,N) and broadcast over E_old's 3 components → (poles,3,N,N,N).
            P_new = disp_c1 * P_curr + disp_c2 * P_prev + disp_c3 * E_old
            E = E + inv_eps * mx.sum(P_curr - P_new, axis=0)
        if sigma_E is not None:
            E = E / (1.0 + c * sigma_E * eta0 * inv_eps / 2.0)
        if P_curr is not None:
            # swap: P_curr_new <- P_new, P_prev_new <- P_curr (old)
            return E, psi_E_new, P_new, P_curr
        return E, psi_E_new

    return (
        _update_aniso(
            E, curl, inv_eps, sigma_E, c, eta0, add=True, periodic_axes=periodic_axes, aniso_widths=aniso_widths
        ),
        psi_E_new,
    )


def _update_H(
    E, H, psi_H, inv_mu, sigma_H, cpml_a, cpml_b, inv_kappa, metric_fwd, periodic_axes, extents, aniso_widths, c, sb
):
    """Pure ``(H_new, psi_H_new)`` from ``dH/dt = -(1/mu) curl(E)``. All inputs explicit (compilable)."""
    curl, psi_H_new = curl_E_mlx(E, psi_H, cpml_a, cpml_b, inv_kappa, sb, metric_fwd, periodic_axes, extents)

    if not _is_full_tensor(inv_mu) and not _is_full_tensor(sigma_H):
        factor = 1.0
        if sigma_H is not None:
            factor = 1.0 - c * sigma_H / eta0 * inv_mu / 2.0
        H = factor * H - c * curl * inv_mu
        if sigma_H is not None:
            H = H / (1.0 + c * sigma_H / eta0 * inv_mu / 2.0)
        return H, psi_H_new

    return (
        _update_aniso(
            H, curl, inv_mu, sigma_H, c, 1.0 / eta0, add=False, periodic_axes=periodic_axes, aniso_widths=aniso_widths
        ),
        psi_H_new,
    )


def update_E_mlx(state: MLXState, c: float, simulate_boundaries: bool = True) -> tuple[mx.array, mx.array]:
    """Return ``(E_new, psi_E_new)`` from ``dE/dt = (1/eps) curl(H)`` (eager wrapper over ``_update_E``)."""
    return _update_E(
        state.E,
        state.H,
        state.psi_E,
        state.inv_eps,
        state.sigma_E,
        state.cpml_a,
        state.cpml_b,
        state.inv_kappa,
        state.metric_bwd,
        state.periodic_axes,
        state.cpml_extents,
        state.aniso_widths,
        c,
        simulate_boundaries,
    )


def update_H_mlx(state: MLXState, c: float, simulate_boundaries: bool = True) -> tuple[mx.array, mx.array]:
    """Return ``(H_new, psi_H_new)`` from ``dH/dt = -(1/mu) curl(E)`` (eager wrapper over ``_update_H``)."""
    return _update_H(
        state.E,
        state.H,
        state.psi_H,
        state.inv_mu,
        state.sigma_H,
        state.cpml_a,
        state.cpml_b,
        state.inv_kappa,
        state.metric_fwd,
        state.periodic_axes,
        state.cpml_extents,
        state.aniso_widths,
        c,
        simulate_boundaries,
    )


def _update_aniso(
    F, curl, inv_material, sigma, c: float, eta_factor: float, add: bool, periodic_axes, aniso_widths=None
):
    """Full-anisotropic E (add=True) or H (add=False) update via per-cell A/B matrices.

    ``F`` and ``curl`` are the un-padded (3, Nx, Ny, Nz) fields. Off-diagonal terms use the
    other components averaged to this component's Yee location (spacing-weighted on non-uniform
    grids via ``aniso_widths``).
    """
    inv_t = expand_to_3x3_mlx(inv_material)
    sigma_t = expand_to_3x3_mlx(sigma) if sigma is not None else None
    A, B = compute_anisotropic_update_matrices_mlx(inv_t, sigma_t, c, eta_factor)

    avg_fn = avg_anisotropic_E_component_mlx if add else avg_anisotropic_H_component_mlx

    def avg(field_pad, component, location):
        return avg_fn(field_pad, component, location, aniso_widths)

    Fp = pad_fields_mlx(F, periodic_axes)
    Cp = pad_fields_mlx(curl, periodic_axes)

    # F[other] averaged to each diagonal component's location.
    Fx_y, Fx_z = avg(Fp, 0, 1), avg(Fp, 0, 2)
    Fy_x, Fy_z = avg(Fp, 1, 0), avg(Fp, 1, 2)
    Fz_x, Fz_y = avg(Fp, 2, 0), avg(Fp, 2, 1)
    Cx_y, Cx_z = avg(Cp, 0, 1), avg(Cp, 0, 2)
    Cy_x, Cy_z = avg(Cp, 1, 0), avg(Cp, 1, 2)
    Cz_x, Cz_y = avg(Cp, 2, 0), avg(Cp, 2, 1)

    a_term_x = A[0, 0] * F[0] + A[0, 1] * Fy_x + A[0, 2] * Fz_x
    a_term_y = A[1, 0] * Fx_y + A[1, 1] * F[1] + A[1, 2] * Fz_y
    a_term_z = A[2, 0] * Fx_z + A[2, 1] * Fy_z + A[2, 2] * F[2]

    b_term_x = B[0, 0] * curl[0] + B[0, 1] * Cy_x + B[0, 2] * Cz_x
    b_term_y = B[1, 0] * Cx_y + B[1, 1] * curl[1] + B[1, 2] * Cz_y
    b_term_z = B[2, 0] * Cx_z + B[2, 1] * Cy_z + B[2, 2] * curl[2]

    if add:
        Fx, Fy, Fz = a_term_x + b_term_x, a_term_y + b_term_y, a_term_z + b_term_z
    else:
        Fx, Fy, Fz = a_term_x - b_term_x, a_term_y - b_term_y, a_term_z - b_term_z

    return mx.stack([Fx, Fy, Fz], axis=0)
