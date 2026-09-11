"""Tidy3D-free full-vectorial waveguide mode solver.

A drop-in replacement for the Tidy3D mode-solver call used in
:mod:`fdtdx.core.physics.modes`. :func:`fdtdmex_mode_computation_wrapper` mirrors the signature and
``list[ModeTupleType]`` return of ``tidy3d_mode_computation_wrapper`` (Tidy3D's z-propagation
convention, pre-eta0 scaling) so the surrounding ``compute_mode`` post-processing — axis rotation,
eta0-scaling, Poynting normalisation — is unchanged.

Scope: straight waveguide; uniform and rectilinear transverse grids; the **whole** permittivity
tensor, all nine entries; diagonal permeability. Bends are deferred.

The two formulations, and which one a cross-section gets
--------------------------------------------------------

The **transverse-E** formulation (:mod:`fdtdx.core.physics.mode_backend.solve`) has eigenvector
``[Ex; Ey]`` and eigenvalue ``-(n_eff)**2``. An eigenvalue that is a function of ``n_eff**2`` alone
can only describe a medium with a mirror symmetry about the cross-section plane, i.e. one with

.. code-block:: text

    eps_xz = eps_zx = eps_yz = eps_zy = 0,

because otherwise the bulk dispersion relation gains odd powers of the propagation constant and
forward and backward modes stop being mirror images. It carries ``eps_xy`` and ``eps_yx`` exactly.

The **four-component** formulation (:mod:`fdtdx.core.physics.mode_backend.full_tensor`) has
eigenvector ``[Ex; Ey; hx; hy]`` and eigenvalue ``n_eff`` itself, linear, so the four longitudinal
entries enter exactly. Its operator is ``4N x 4N`` instead of ``2N x 2N``. With those four entries
zero it is block anti-diagonal and its square is the transverse operator entry for entry
(``mat_transverse = -A B``), so the two agree to round-off.

A cross-section with any non-zero longitudinal entry takes the four-component path automatically;
everything else takes the transverse one, on exactly the code path it took before. ``formulation``
overrides the choice: ``"transverse"`` asks for the old behaviour deliberately and then drops the
four entries with a :class:`ModeLongitudinalOffdiagWarning` naming each one and its magnitude,
which is worth doing when the entries are known to be negligible and the ``4N`` operator is not
worth its cost. :data:`TOL_TENSORIAL` is the magnitude above which that warning fires.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List, Literal, NamedTuple, Sequence

import numpy as np

from fdtdx.constants import c
from fdtdx.core.physics.mode_backend.full_tensor import solve_modes_full_tensor
from fdtdx.core.physics.mode_backend.operator import build_average_matrices, build_derivative_matrices
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

#: Allowed values of the ``formulation`` argument.
FORMULATIONS = ("auto", "transverse", "full")


class ModeLongitudinalOffdiagWarning(UserWarning):
    """``eps_xz`` / ``eps_zx`` / ``eps_yz`` / ``eps_zy`` are present and are being dropped.

    Since the four-component formulation landed this fires only when the caller asked for
    ``formulation="transverse"`` on a cross-section that carries those entries. They cannot enter an
    eigenproblem linear in ``n_eff**2`` (see the module docstring); the solve continues on the
    remaining five components, which are exact.
    """


class TensorComponents(NamedTuple):
    """Flattened permittivity components the mode operator consumes.

    Attributes:
        xx: Flattened ``eps_xx`` at the Yee Ex locations.
        yy: Flattened ``eps_yy``.
        zz: Flattened ``eps_zz``.
        xy: Flattened ``eps_xy``, or ``None`` when the medium has no transverse off-diagonal.
        yx: Flattened ``eps_yx``, or ``None``.
        longitudinal: ``{entry name: flattened entry}`` for ``xz`` / ``yz`` / ``zx`` / ``zy``, empty
            when all four are identically zero.
        magnitudes: ``{entry name: largest magnitude}`` for the longitudinal entries above
            :data:`TOL_TENSORIAL`; what the warning names when they are dropped.
    """

    xx: np.ndarray
    yy: np.ndarray
    zz: np.ndarray
    xy: np.ndarray | None
    yx: np.ndarray | None
    longitudinal: dict[str, np.ndarray]
    magnitudes: dict[str, float]

    def as_dict(self) -> dict[str, np.ndarray]:
        """The nine components keyed by entry name, with absent ones left out.

        Returns:
            dict[str, np.ndarray]: What
            :func:`fdtdx.core.physics.mode_backend.full_tensor.solve_modes_full_tensor` consumes.
        """
        out = {"xx": self.xx, "yy": self.yy, "zz": self.zz}
        if self.xy is not None and self.yx is not None:
            out["xy"], out["yx"] = self.xy, self.yx
        out.update(self.longitudinal)
        return out


def _tensor_components(cross_section, nx: int, ny: int, what: str = "permittivity") -> TensorComponents:
    """Split a ``1 / 3 / 9`` component cross-section into the components the operator uses.

    The expansion of the ``1 / 3 / 9`` component layout to a 3x3 tensor is done in numpy at
    ``complex128`` rather than through :func:`fdtdx.core.misc.expand_to_3x3`. The mode solve is
    unconditionally double precision, and ``expand_to_3x3`` runs in JAX: under the default (x32)
    JAX configuration ``jnp.asarray`` of a ``complex128`` cross-section silently returns
    ``complex64``, which would round the material before the operator is assembled.

    Nothing is dropped and nothing is warned about here: the caller decides the formulation and
    :func:`drop_longitudinal` is what reports a deliberate drop.

    Args:
        cross_section: Rotated cross-section, shape ``(1 | 3 | 9, Nx, Ny)``.
        nx (int): First transverse cell count.
        ny (int): Second transverse cell count.
        what (str): Name of the material, unused here and kept for the caller's message.

    Returns:
        TensorComponents: The flattened components. ``xy`` / ``yx`` are ``None`` and
        ``longitudinal`` is empty when the cross-section has no such entry at all, so the assembly
        takes the narrower path entry for entry rather than by cancellation.

    Raises:
        ValueError: If the component axis is not 1, 3 or 9 long.
    """
    del what
    arr = np.asarray(cross_section).astype(np.complex128)
    ncomp = arr.shape[0]
    magnitudes: dict[str, float] = {}
    longitudinal: dict[str, np.ndarray] = {}
    off_xy: np.ndarray | None = None
    off_yx: np.ndarray | None = None
    if ncomp == 1:
        diagonal = (arr[0], arr[0], arr[0])
    elif ncomp == 3:
        diagonal = (arr[0], arr[1], arr[2])
    elif ncomp == 9:
        diagonal = (arr[0], arr[4], arr[8])
        if arr.size:
            if any(np.any(arr[flat] != 0.0) for flat in _LONGITUDINAL_OFFDIAG):
                for flat in _LONGITUDINAL_OFFDIAG:
                    name = _ENTRY_NAMES[flat]
                    longitudinal[name] = arr[flat]
                    magnitude = float(np.max(np.abs(arr[flat])))
                    if magnitude > TOL_TENSORIAL:
                        magnitudes[name] = magnitude
            if np.any(arr[1] != 0.0) or np.any(arr[3] != 0.0):
                off_xy, off_yx = arr[1], arr[3]
    else:
        raise ValueError(f"cross-section component axis must be 1, 3 or 9 long, got {ncomp}")

    def flat(component: np.ndarray) -> np.ndarray:
        return component.reshape(nx, ny).ravel()

    return TensorComponents(
        xx=flat(diagonal[0]),
        yy=flat(diagonal[1]),
        zz=flat(diagonal[2]),
        xy=None if off_xy is None else flat(off_xy),
        yx=None if off_yx is None else flat(off_yx),
        longitudinal={name: flat(value) for name, value in longitudinal.items()},
        magnitudes=magnitudes,
    )


def use_full_tensor(components: TensorComponents, formulation: str) -> bool:
    """Decide which formulation a cross-section gets.

    Args:
        components (TensorComponents): The split cross-section.
        formulation (str): ``"auto"``, ``"transverse"`` or ``"full"``.

    Returns:
        bool: ``True`` for the four-component path.

    Raises:
        ValueError: On an unknown ``formulation``.
    """
    if formulation not in FORMULATIONS:
        raise ValueError(f"formulation must be one of {FORMULATIONS}, got {formulation!r}")
    if formulation == "full":
        return True
    if formulation == "transverse":
        return False
    return bool(components.longitudinal)


def drop_longitudinal(components: TensorComponents, what: str = "permittivity") -> None:
    """Report a deliberate drop of the longitudinal entries, naming each one and its magnitude.

    Args:
        components (TensorComponents): The split cross-section.
        what (str): Name of the material, used in the message.
    """
    if not components.magnitudes:
        return
    listing = ", ".join(
        f"{name} (largest |entry| {value:.4g})" for name, value in sorted(components.magnitudes.items())
    )
    warnings.warn(
        f"the {what} cross-section carries the longitudinal off-diagonal entries {listing}, above "
        f"TOL_TENSORIAL={TOL_TENSORIAL:g}, and formulation='transverse' was asked for. They are "
        "dropped: an eigenproblem linear in n_eff**2 cannot represent a medium without a mirror "
        "plane at the cross-section. Use formulation='auto' (the default) or 'full' for the "
        "four-component operator, which carries them exactly.",
        ModeLongitudinalOffdiagWarning,
        stacklevel=3,
    )


def transverse_index_bound(
    components: TensorComponents,
    mu_xx: np.ndarray | None = None,
    mu_yy: np.ndarray | None = None,
) -> float:
    """Largest ``n**2`` the cross-section can support, used to aim the shift-invert solve.

    For propagation along the mode axis with no transverse variation, eliminating the longitudinal
    components leaves ``n**2 E_t = diag(mu_yy, mu_xx) S E_t`` with ``S`` the Schur complement
    ``eps_tt - eps_tz eps_zz^-1 eps_zt``. Both factors matter. A rotated uniaxial tensor hides its
    extraordinary index inside ``S`` — in the transverse off-diagonal and in the longitudinal
    entries — so reading the bound off the permittivity diagonal alone aims the shift below the mode
    the caller wants; and a *magnetically* anisotropic cross-section, which this solver supports
    (only an off-diagonal permeability is refused), moves it again, by a factor of up to
    ``max(mu_xx, mu_yy)``.

    Args:
        components (TensorComponents): The split permittivity cross-section.
        mu_xx (np.ndarray | None): Flattened ``mu_xx``; ``None`` means unity.
        mu_yy (np.ndarray | None): Flattened ``mu_yy``; ``None`` means unity.

    Returns:
        float: The largest real ``n**2``, never below what ``eps_zz`` alone would give.
    """
    xx, yy, zz = components.xx, components.yy, components.zz
    zero = np.zeros_like(xx)
    xy = zero if components.xy is None else components.xy
    yx = zero if components.yx is None else components.yx
    longitudinal = components.longitudinal
    if longitudinal:
        xz, yz = longitudinal["xz"], longitudinal["yz"]
        zx, zy = longitudinal["zx"], longitudinal["zy"]
        s11 = xx - xz * zx / zz
        s12 = xy - xz * zy / zz
        s21 = yx - yz * zx / zz
        s22 = yy - yz * zy / zz
    else:
        s11, s12, s21, s22 = xx, xy, yx, yy
    row_x = np.ones_like(xx) if mu_yy is None else np.asarray(mu_yy)
    row_y = np.ones_like(xx) if mu_xx is None else np.asarray(mu_xx)
    m11, m12 = row_x * s11, row_x * s12
    m21, m22 = row_y * s21, row_y * s22
    half_trace = 0.5 * (m11 + m22)
    radicand = 0.25 * (m11 - m22) ** 2 + m12 * m21
    largest = float(np.max(np.real(half_trace + np.emath.sqrt(radicand))))
    return max(largest, float(np.max(np.real(np.maximum(row_x, row_y) * zz))))


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
    formulation: Literal["auto", "transverse", "full"] = "auto",
) -> List[ModeTupleType]:
    """Compute waveguide modes with the native full-vectorial FD solver.

    Args mirror ``tidy3d_mode_computation_wrapper``: ``coords`` are the two transverse cell-edge
    arrays in micrometres; ``permittivity_cross_section`` / ``permeability_cross_section`` are the
    rotated cross-sections (1/3/9 components x Nx x Ny) in Tidy3D's transverse convention.
    ``formulation`` selects the operator: ``"auto"`` takes the four-component one exactly when the
    cross-section has a non-zero longitudinal off-diagonal entry, ``"transverse"`` forces the
    ``2N`` operator and drops those entries with a warning, ``"full"`` forces the ``4N`` one.

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
    full_tensor = use_full_tensor(eps, formulation)
    if not full_tensor:
        drop_longitudinal(eps)
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
        n_max = float(np.sqrt(transverse_index_bound(eps, mu_xx, mu_yy)))
        neff_guess = n_max * (1.0 + 1e-6) + 1e-6
    else:
        neff_guess = float(target_neff)

    dmin_pmc = (symmetry[0] == 1, symmetry[1] == 1)
    der_mats = build_derivative_matrices(coords_x_m, coords_y_m, dmin_pmc=dmin_pmc)

    if full_tensor:
        avg_mats = build_average_matrices(coords_x_m, coords_y_m, dmin_pmc=dmin_pmc)
        E, H, neff, keff = solve_modes_full_tensor(
            eps.as_dict(),
            mu_xx,
            mu_yy,
            mu_zz,
            der_mats,
            avg_mats,
            k0=k0,
            num_modes=num_modes,
            neff_guess=neff_guess,
            direction=direction,
        )
    else:
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
