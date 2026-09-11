"""Tidy3D-free full-vectorial waveguide mode solver.

A drop-in replacement for the Tidy3D mode-solver call used in
:mod:`fdtdx.core.physics.modes`. :func:`fdtdmex_mode_computation_wrapper` mirrors the signature and
``list[ModeTupleType]`` return of ``tidy3d_mode_computation_wrapper`` (Tidy3D's z-propagation
convention, pre-eta0 scaling) so the surrounding ``compute_mode`` post-processing — axis rotation,
eta0-scaling, Poynting normalisation — is unchanged.

Scope (stages 1-2): straight waveguide; uniform and rectilinear transverse grids; permittivity
tensors carrying the two transverse off-diagonal entries ``eps_xy`` / ``eps_yx`` in addition to the
diagonal; diagonal permeability. Bends are deferred, and so are the four entries that couple a
transverse axis to the propagation axis (see below).

Which off-diagonal entries the solver can carry, and why
--------------------------------------------------------

The transverse-E formulation is an eigenproblem *linear in* ``n_eff**2``. That is possible only when
the medium has a mirror symmetry about the cross-section plane, i.e. when

.. code-block:: text

    eps_xz = eps_zx = eps_yz = eps_zy = 0.

With those entries present the bulk dispersion relation picks up odd powers of the propagation
constant (forward and backward modes stop being mirror images), so no operator whose eigenvalue is
``-(n_eff)**2`` can represent them; an exact treatment needs the four-field ``[Ex, Ey, hx, hy]``
formulation, whose eigenvalue is ``n_eff`` itself and whose operator is twice as large. The backend
therefore **warns** (:class:`ModeLongitudinalOffdiagWarning`), naming the entry and its magnitude,
and solves the cross-section with those four entries dropped. ``eps_xy`` and ``eps_yx`` are carried
exactly. :data:`TOL_TENSORIAL` is the magnitude above which the warning fires; below it the entries
are dropped silently.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List, Literal, NamedTuple, Sequence

import numpy as np

from fdtdx.constants import c
from fdtdx.core.physics.mode_backend.operator import build_derivative_matrices
from fdtdx.core.physics.mode_backend.solve import solve_modes_diagonal

if TYPE_CHECKING:
    from fdtdx.core.physics.modes import ModeTupleType

#: Magnitude above which a dropped longitudinal off-diagonal entry (xz / zx / yz / zy) is reported.
#: Until phase 2 of the mode-solver track this was the threshold above which the whole
#: cross-section was refused; the transverse entries are now carried exactly and only the four
#: longitudinal ones are dropped, so the refusal became this warning.
TOL_TENSORIAL = 1e-6

#: Names of the tensor entries in the row-major 9-component layout.
_ENTRY_NAMES = ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")

#: Flat indices of the four entries that couple a transverse axis to the propagation axis.
_LONGITUDINAL_OFFDIAG = (2, 5, 6, 7)  # xz, yz, zx, zy


class ModeLongitudinalOffdiagWarning(UserWarning):
    """A cross-section carries ``eps_xz`` / ``eps_zx`` / ``eps_yz`` / ``eps_zy``, which are dropped.

    Those four entries cannot enter an eigenproblem that is linear in ``n_eff**2`` (see the module
    docstring). The solve continues on the remaining five components, which are exact.
    """


class TensorComponents(NamedTuple):
    """Flattened permittivity components the mode operator consumes.

    Attributes:
        xx: Flattened ``eps_xx`` at the Yee Ex locations.
        yy: Flattened ``eps_yy``.
        zz: Flattened ``eps_zz``.
        xy: Flattened ``eps_xy``, or ``None`` when the medium has no transverse off-diagonal.
        yx: Flattened ``eps_yx``, or ``None``.
        dropped: ``{entry name: largest magnitude}`` for the longitudinal entries that were dropped.
    """

    xx: np.ndarray
    yy: np.ndarray
    zz: np.ndarray
    xy: np.ndarray | None
    yx: np.ndarray | None
    dropped: dict[str, float]


def _tensor_components(cross_section, nx: int, ny: int, what: str = "permittivity") -> TensorComponents:
    """Split a ``1 / 3 / 9`` component cross-section into the components the operator uses.

    The expansion of the ``1 / 3 / 9`` component layout to a 3x3 tensor is done in numpy at
    ``complex128`` rather than through :func:`fdtdx.core.misc.expand_to_3x3`. The mode solve is
    unconditionally double precision, and ``expand_to_3x3`` runs in JAX: under the default (x32)
    JAX configuration ``jnp.asarray`` of a ``complex128`` cross-section silently returns
    ``complex64``, which would round the material before the operator is assembled.

    Args:
        cross_section: Rotated cross-section, shape ``(1 | 3 | 9, Nx, Ny)``.
        nx (int): First transverse cell count.
        ny (int): Second transverse cell count.
        what (str): Name of the material, used in the warning message.

    Returns:
        TensorComponents: The flattened components, with ``xy`` / ``yx`` set to ``None`` when the
        cross-section has no transverse off-diagonal entry at all (so the assembly takes the
        diagonal path entry for entry rather than by cancellation).

    Raises:
        ValueError: If the component axis is not 1, 3 or 9 long.
    """
    arr = np.asarray(cross_section).astype(np.complex128)
    ncomp = arr.shape[0]
    dropped: dict[str, float] = {}
    off_xy: np.ndarray | None = None
    off_yx: np.ndarray | None = None
    if ncomp == 1:
        diagonal = (arr[0], arr[0], arr[0])
    elif ncomp == 3:
        diagonal = (arr[0], arr[1], arr[2])
    elif ncomp == 9:
        diagonal = (arr[0], arr[4], arr[8])
        if arr.size:
            for flat in _LONGITUDINAL_OFFDIAG:
                magnitude = float(np.max(np.abs(arr[flat])))
                if magnitude > TOL_TENSORIAL:
                    dropped[_ENTRY_NAMES[flat]] = magnitude
            if np.any(arr[1] != 0.0) or np.any(arr[3] != 0.0):
                off_xy, off_yx = arr[1], arr[3]
    else:
        raise ValueError(f"cross-section component axis must be 1, 3 or 9 long, got {ncomp}")
    if dropped:
        listing = ", ".join(f"{name} (largest |entry| {value:.4g})" for name, value in sorted(dropped.items()))
        warnings.warn(
            f"the {what} cross-section carries the longitudinal off-diagonal entries {listing}, above "
            f"TOL_TENSORIAL={TOL_TENSORIAL:g}. They are dropped: an eigenproblem linear in n_eff**2 "
            "cannot represent a medium without a mirror plane at the cross-section, and the exact "
            "treatment needs the four-field formulation. The transverse entries xy / yx are carried "
            "exactly, and every diagonal entry is unchanged.",
            ModeLongitudinalOffdiagWarning,
            stacklevel=3,
        )

    def flat(component: np.ndarray) -> np.ndarray:
        return component.reshape(nx, ny).ravel()

    return TensorComponents(
        xx=flat(diagonal[0]),
        yy=flat(diagonal[1]),
        zz=flat(diagonal[2]),
        xy=None if off_xy is None else flat(off_xy),
        yx=None if off_yx is None else flat(off_yx),
        dropped=dropped,
    )


def _diag_components(cross_section, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the flattened xx/yy/zz components, refusing any off-diagonal entry.

    Used for the permeability, which the transverse-E operator carries on its diagonal only: the
    dual of the ``q_ep`` block that carries ``eps_xy`` is a ``p_mu`` block that would carry
    ``mu_xy``, and it is not implemented because nothing in the fork produces a tensorial
    permeability.

    Args:
        cross_section: Rotated cross-section, shape ``(1 | 3 | 9, Nx, Ny)``.
        nx (int): First transverse cell count.
        ny (int): Second transverse cell count.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Flattened ``xx``, ``yy``, ``zz`` components.

    Raises:
        NotImplementedError: If an off-diagonal component exceeds :data:`TOL_TENSORIAL`.
        ValueError: If the component axis is not 1, 3 or 9 long.
    """
    arr = np.asarray(cross_section).astype(np.complex128)
    if arr.shape[0] == 9 and arr.size:
        tensor = arr.reshape(3, 3, *arr.shape[1:])
        off = ~np.eye(3, dtype=bool)
        if float(np.max(np.abs(tensor[off]))) > TOL_TENSORIAL:
            raise NotImplementedError(
                "The fdtdmex mode backend carries a diagonal permeability only; the cross-section "
                "has significant off-diagonal permeability components. Set mode_backend='tidy3d' "
                "(requires the optional tidy3d dependency) for a fully tensorial permeability."
            )
    components = _tensor_components(arr, nx, ny, what="permeability")
    return components.xx, components.yy, components.zz


def fdtdmex_mode_computation_wrapper(
    frequency: float,
    permittivity_cross_section,
    coords: Sequence[np.ndarray],
    direction: Literal["+", "-"],
    permeability_cross_section=None,
    target_neff: float | None = None,
    num_modes: int = 10,
    bend_radius: float | None = None,
    bend_axis: int | None = None,
    plane_center: tuple[float, float] | None = None,
    symmetry: tuple[int, int] = (0, 0),
) -> List[ModeTupleType]:
    """Compute waveguide modes with the native full-vectorial FD solver.

    Args mirror ``tidy3d_mode_computation_wrapper``: ``coords`` are the two transverse cell-edge
    arrays in micrometres; ``permittivity_cross_section`` / ``permeability_cross_section`` are the
    rotated cross-sections (1/3/9 components x Nx x Ny) in Tidy3D's transverse convention.

    Returns:
        ``list[ModeTupleType]`` sorted by descending ``Re(n_eff)``, in the same convention as the
        Tidy3D wrapper (so ``compute_mode``'s downstream handling is unchanged).
    """
    # Lazy import breaks the modes.py <-> mode_backend import cycle.
    from fdtdx.core.physics.modes import ModeTupleType

    if bend_radius is not None:
        raise NotImplementedError(
            "Bend modes (conformal transform) are not implemented in the fdtdmex mode backend; "
            "use mode_backend='tidy3d' for bends."
        )

    perm = np.asarray(permittivity_cross_section)
    nx, ny = perm.shape[1], perm.shape[2]
    eps = _tensor_components(perm, nx, ny)
    eps_xx, eps_yy, eps_zz = eps.xx, eps.yy, eps.zz

    mu = np.asarray(permeability_cross_section) if permeability_cross_section is not None else None
    if mu is None or mu.ndim < 3:
        # Uniform (scalar) permeability: broadcast to a flat per-cell array.
        mu_val = 1.0 if mu is None else complex(np.asarray(mu).reshape(-1)[0])
        mu_xx = mu_yy = mu_zz = np.full(nx * ny, mu_val, dtype=np.complex128)
    else:
        mu_xx, mu_yy, mu_zz = _diag_components(mu, nx, ny)

    coords_x_m = np.asarray(coords[0], dtype=np.float64) * 1e-6
    coords_y_m = np.asarray(coords[1], dtype=np.float64) * 1e-6
    if len(coords_x_m) != nx + 1 or len(coords_y_m) != ny + 1:
        raise ValueError("coords length must be one more than the cross-section size on each axis")

    k0 = 2.0 * np.pi * frequency / c

    # Shift-invert target. Without one, aim just above the largest real index in the cross-section:
    # the guided modes sit below it and the shift then picks up the highest-index ones. With one,
    # aim exactly at it - the caller is asking for the modes nearest that index, which is the whole
    # point of the argument (a metal-clad or plasmonic guide has no useful "largest real index").
    if target_neff is None:
        # With a transverse off-diagonal the bound is the largest eigenvalue of the 2x2 transverse
        # block, not the largest diagonal entry: a rotated uniaxial tensor hides its extraordinary
        # index in the off-diagonal, and aiming below it loses the mode the caller wants.
        if eps.xy is not None and eps.yx is not None:
            half_trace = 0.5 * (eps_xx + eps_yy)
            radicand = 0.25 * (eps_xx - eps_yy) ** 2 + eps.xy * eps.yx
            transverse_max = np.max(np.real(half_trace + np.emath.sqrt(radicand)))
            n_max = float(np.sqrt(max(float(transverse_max), float(np.max(np.real(eps_zz))))))
        else:
            n_max = float(np.sqrt(np.max(np.real([eps_xx, eps_yy, eps_zz]))))
        neff_guess = n_max * (1.0 + 1e-6) + 1e-6
    else:
        neff_guess = float(target_neff)

    dmin_pmc = (symmetry[0] == 1, symmetry[1] == 1)
    der_mats = build_derivative_matrices(coords_x_m, coords_y_m, dmin_pmc=dmin_pmc)

    E, H, neff, keff = solve_modes_diagonal(
        eps_xx,
        eps_yy,
        eps_zz,
        mu_xx,
        mu_yy,
        mu_zz,
        der_mats,
        k0=k0,
        num_modes=num_modes,
        neff_guess=neff_guess,
        direction=direction,
        eps_xy=eps.xy,
        eps_yx=eps.yx,
    )

    n_solved = E.shape[2]
    modes: list[ModeTupleType] = []
    for i in range(n_solved):
        modes.append(
            ModeTupleType(
                neff=complex(neff[i] + 1j * keff[i]),
                Ex=E[0, :, i].reshape(nx, ny),
                Ey=E[1, :, i].reshape(nx, ny),
                Ez=E[2, :, i].reshape(nx, ny),
                Hx=H[0, :, i].reshape(nx, ny),
                Hy=H[1, :, i].reshape(nx, ny),
                Hz=H[2, :, i].reshape(nx, ny),
            )
        )
    return modes
