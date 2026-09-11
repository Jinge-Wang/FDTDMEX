"""A differentiable effective index: ``compute_mode`` wrapped in a ``jax.custom_vjp``.

:func:`fdtdx.core.physics.modes.compute_mode` reaches its backend through ``jax.pure_callback`` and
wraps every array argument in ``jax.lax.stop_gradient``, so ``jax.grad`` through it returns zero
without raising. Measured on a 40 x 30 cross-section at 40 nm (Si core 12.0 in SiO2 2.25, 1.55 um,
scaling the whole permittivity by a scalar): ``jax.grad`` gives ``0.000000`` where a central finite
difference gives ``2.040029``. Anyone who wires the mode solver into an optimizer without reading
the source gets a silent no-op.

:func:`mode_neff` closes that hole. It is a ``jax.custom_vjp`` around the same solve whose backward
is the first-order (reciprocity) eigenvalue sensitivity evaluated on the field the forward already
returned, so it costs no extra solve:

.. math::

    \\frac{\\partial n_{\\mathrm{eff}}}{\\partial \\varepsilon_c(\\mathbf{r})}
    = \\frac{1}{2}\\, s_c\\, \\frac{E_c(\\mathbf{r})^2\\, w(\\mathbf{r})}
      {\\int (\\mathbf{E}_t \\times \\mathbf{H}_t)\\cdot \\hat{e}_p \\, \\mathrm{d}A}

with ``s_c = -1`` on the propagation axis ``p`` and ``+1`` on the two transverse axes, ``w`` the cell
area, and *unconjugated* products throughout. The unconjugated form is the Lorentz-reciprocity one
(the partner field is the backward mode, which carries ``-E_p`` and ``-H_t``); it is the version that
stays correct for a complex permittivity, and because the eigenvalue is holomorphic in the
permittivity entries a single backward yields ``d Re(n_eff)/d eps`` and ``d Im(n_eff)/d eps``
together -- phase and loss off one solve.

The same formula is what the ring case's ``perturbation_theory_shift`` uses for a resonance
(``analysis_reports/agent_reports/T4_ring_thermal_tuning_coupled.md``, agreement 0.34 %); the
denominator differs because a resonance shifts in frequency at fixed structure while a guided mode
shifts in index at fixed frequency, so the normalization here is the modal power rather than the
stored electric energy.

Scope: component axes of length 1, 3 and 9. The native mode backend carries the two transverse
off-diagonal entries of a permittivity tensor exactly, so ``d n_eff / d eps_ab`` exists for every
entry of the 9-component tier; the reciprocity formula above generalises to the bilinear form
``0.5 s_a E_a E_b w / flux``. The four entries that couple a transverse axis to the propagation axis
are dropped by the solver with a warning (see :mod:`fdtdx.core.physics.mode_backend`), so their
sensitivity is reported but not acted on. The permeability is treated as a constant. The returned
mode fields are differentiable through :func:`mode_solve`'s default path.

A cross-section with a transverse axis of exactly two cells triggers ``compute_mode``'s
two-dimensional collapse. That path used to raise a shape error inside its own callback; it is fixed
and the slab cross-section now solves and differentiates like any other.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.core.jax.utils import is_jax_tracer
from fdtdx.core.physics.modes import compute_mode, compute_modes

__all__ = [
    "ModeMatch",
    "ModeSolution",
    "ModeSolveSettings",
    "ModeTracker",
    "carries_longitudinal_entries",
    "mode_neff",
    "mode_neff_parts",
    "mode_overlap",
    "mode_sensitivity",
    "mode_solve",
    "track_mode",
    "tracked_mode_index",
]


@dataclass(frozen=True)
class ModeSolveSettings:
    """Everything about a mode solve that is not differentiated.

    Hashable so it can ride in ``jax.custom_vjp``'s ``nondiff_argnums``. Build it with
    :func:`ModeSolveSettings.create`, which accepts the coordinate arrays as sequences.

    Attributes:
        frequency: Operating frequency in Hz.
        resolution: Uniform transverse cell size in metres, or ``None`` when ``transverse_coords``
            is given.
        transverse_coords: Cell-edge coordinates in metres of the two axes transverse to
            propagation, as tuples so the settings stay hashable, or ``None`` for a uniform grid.
        direction: Propagation direction, ``"+"`` or ``"-"``.
        mode_index: Index into the modes sorted by descending ``Re(n_eff)``.
        filter_pol: Restrict the sorted list to ``"te"`` or ``"tm"`` before indexing.
        permeability: Uniform relative permeability; held constant by the backward.
        symmetry: Per-transverse-axis mirror condition at the min edge (0 = PEC, 1 = PMC).
        mode_backend: ``"fdtdmex"`` or ``"tidy3d"``; ``None`` uses the package default.
        double_precision: Ask the solver for complex128 fields and index. Needs
            ``jax_enable_x64``; a finite-difference check is not meaningful without it.
        formulation: Which native mode operator to assemble — ``"auto"``, ``"transverse"`` or
            ``"full"``; see :func:`fdtdx.core.physics.modes.compute_mode`. On a nine-component
            cross-section anything but ``"transverse"`` means the four longitudinal entries are
            carried, and the index gradient then comes off the matrix-level adjoint rather than the
            field-level reciprocity integral (see :func:`_sensitivity_from_fields`).
    """

    frequency: float
    resolution: float | None = None
    transverse_coords: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    direction: Literal["+", "-"] = "+"
    mode_index: int = 0
    filter_pol: Literal["te", "tm"] | None = None
    permeability: float = 1.0
    symmetry: tuple[int, int] = (0, 0)
    mode_backend: Literal["fdtdmex", "tidy3d"] | None = None
    double_precision: bool = True
    formulation: Literal["auto", "transverse", "full"] = "auto"

    @staticmethod
    def create(
        *,
        frequency: float,
        resolution: float | None = None,
        transverse_coords: Sequence[Any] | None = None,
        direction: Literal["+", "-"] = "+",
        mode_index: int = 0,
        filter_pol: Literal["te", "tm"] | None = None,
        permeability: float = 1.0,
        symmetry: tuple[int, int] = (0, 0),
        mode_backend: Literal["fdtdmex", "tidy3d"] | None = None,
        double_precision: bool | None = None,
        formulation: Literal["auto", "transverse", "full"] = "auto",
    ) -> "ModeSolveSettings":
        """Build settings, converting coordinate arrays to hashable tuples.

        Args:
            frequency (float): Operating frequency in Hz.
            resolution (float | None): Uniform transverse cell size in metres.
            transverse_coords (Sequence[Any] | None): Pair of cell-edge coordinate arrays in metres.
            direction (Literal["+", "-"]): Propagation direction.
            mode_index (int): Index into the sorted mode list.
            filter_pol (Literal["te", "tm"] | None): Optional polarization filter.
            permeability (float): Uniform relative permeability.
            symmetry (tuple[int, int]): Mirror condition at the min edge of each transverse axis.
            mode_backend (Literal["fdtdmex", "tidy3d"] | None): Mode-solver backend.
            double_precision (bool | None): Force complex128; ``None`` follows ``jax_enable_x64``.
            formulation (Literal["auto", "transverse", "full"]): Which native mode operator to
                assemble.

        Returns:
            ModeSolveSettings: The frozen, hashable settings object.

        Raises:
            ValueError: If neither or both of ``resolution`` and ``transverse_coords`` are given.
        """
        if (resolution is None) == (transverse_coords is None):
            raise ValueError("give exactly one of resolution (uniform grid) and transverse_coords")
        coords: tuple[tuple[float, ...], tuple[float, ...]] | None = None
        if transverse_coords is not None:
            if len(transverse_coords) != 2:
                raise ValueError(f"transverse_coords must hold exactly two arrays, got {len(transverse_coords)}")
            c0, c1 = (tuple(float(v) for v in np.asarray(c).reshape(-1)) for c in transverse_coords)
            coords = (c0, c1)
        if double_precision is None:
            double_precision = bool(jax.config.jax_enable_x64)
        return ModeSolveSettings(
            frequency=float(frequency),
            resolution=None if resolution is None else float(resolution),
            transverse_coords=coords,
            direction=direction,
            mode_index=int(mode_index),
            filter_pol=filter_pol,
            permeability=float(permeability),
            symmetry=(int(symmetry[0]), int(symmetry[1])),
            mode_backend=mode_backend,
            double_precision=bool(double_precision),
            formulation=formulation,
        )

    def with_mode_index(self, mode_index: int) -> "ModeSolveSettings":
        """Return a copy that selects a different mode.

        Args:
            mode_index (int): The new index into the sorted mode list.

        Returns:
            ModeSolveSettings: A copy with ``mode_index`` replaced.
        """
        return ModeSolveSettings(
            frequency=self.frequency,
            resolution=self.resolution,
            transverse_coords=self.transverse_coords,
            direction=self.direction,
            mode_index=int(mode_index),
            filter_pol=self.filter_pol,
            permeability=self.permeability,
            symmetry=self.symmetry,
            mode_backend=self.mode_backend,
            double_precision=self.double_precision,
            formulation=self.formulation,
        )


class ModeSolution(NamedTuple):
    """One solved mode.

    Attributes:
        neff: Complex effective index (``Re`` the phase index, ``Im`` the attenuation index).
        E: Electric field, shape ``(3, nx, ny, nz)`` with the propagation axis of length one.
        H: Magnetic field, same shape.
    """

    neff: jax.Array
    E: jax.Array
    H: jax.Array


class ModeMatch(NamedTuple):
    """Which candidate mode is the continuation of a reference mode.

    Attributes:
        index: Index of the best-overlapping candidate.
        overlap: Its normalized overlap magnitude, in ``[0, 1]``.
        overlaps: The overlap magnitude of every candidate, in the order given.
    """

    index: int
    overlap: float
    overlaps: np.ndarray


# ------------------------------------------------------------------------------------------------
# geometry helpers
# ------------------------------------------------------------------------------------------------


def _propagation_axis(shape: tuple[int, ...]) -> int:
    """Return the index (0/1/2) of the singleton spatial axis of a ``(ncomp, nx, ny, nz)`` array.

    Args:
        shape (tuple[int, ...]): Shape of the permittivity array.

    Returns:
        int: The propagation axis.

    Raises:
        ValueError: If the shape is not a valid cross-section layout.
    """
    if len(shape) != 4 or shape[0] not in (1, 3, 9):
        raise ValueError(f"permittivity must have shape (1, 3 or 9, nx, ny, nz), got {shape}")
    singletons = [axis for axis, dim in enumerate(shape[1:]) if dim == 1]
    if len(singletons) != 1:
        raise ValueError(f"exactly one of the three spatial axes must have length 1, got shape {shape}")
    return singletons[0]


def _area_weights(settings: ModeSolveSettings, shape: tuple[int, ...], prop_axis: int) -> np.ndarray:
    """Cell areas of the transverse plane, broadcastable to ``(nx, ny, nz)``.

    A constant factor cancels between the numerator and the denominator of the sensitivity, so the
    uniform grid needs no weights at all; a rectilinear grid does.

    Args:
        settings (ModeSolveSettings): The solve settings carrying the grid.
        shape (tuple[int, ...]): Shape of the permittivity array.
        prop_axis (int): Propagation axis.

    Returns:
        np.ndarray: The area weights.
    """
    if settings.transverse_coords is None:
        return np.ones((1, 1, 1))
    transverse = [axis for axis in range(3) if axis != prop_axis]
    d0 = np.diff(np.asarray(settings.transverse_coords[0], dtype=np.float64))
    d1 = np.diff(np.asarray(settings.transverse_coords[1], dtype=np.float64))
    area = d0[:, None] * d1[None, :]
    weight_shape = [1, 1, 1]
    weight_shape[transverse[0]] = area.shape[0]
    weight_shape[transverse[1]] = area.shape[1]
    return area.reshape(weight_shape)


# ------------------------------------------------------------------------------------------------
# the custom_vjp
# ------------------------------------------------------------------------------------------------


def _inverse_permittivity(eps: jax.Array) -> jax.Array:
    """Invert the material array: elementwise on the diagonal tiers, a 3x3 inverse on the 9th.

    Args:
        eps (jax.Array): Relative permittivity, shape ``(1, 3 or 9, nx, ny, nz)``.

    Returns:
        jax.Array: The inverse permittivity in the same layout, which is what ``compute_mode`` takes.
    """
    if eps.shape[0] != 9:
        return 1.0 / eps
    tensor = eps.reshape(3, 3, *eps.shape[1:])
    to_last = (2, 3, 4, 0, 1)
    to_first = (3, 4, 0, 1, 2)
    return jnp.linalg.inv(tensor.transpose(to_last)).transpose(to_first).reshape(9, *eps.shape[1:])


def _solve(eps_re: jax.Array, eps_im: jax.Array, settings: ModeSolveSettings) -> tuple[jax.Array, ...]:
    """Run ``compute_mode`` on ``eps_re + 1j eps_im`` and return ``(re_neff, im_neff, E, H)``.

    Args:
        eps_re (jax.Array): Real part of the diagonal permittivity, shape ``(1 or 3, nx, ny, nz)``.
        eps_im (jax.Array): Imaginary part, same shape.
        settings (ModeSolveSettings): The solve settings.

    Returns:
        tuple[jax.Array, ...]: ``(Re n_eff, Im n_eff, E, H)``.
    """
    dtype = jnp.float64 if settings.double_precision else jnp.float32
    eps = jnp.asarray(eps_re) + 1j * jnp.asarray(eps_im)
    coords = None
    if settings.transverse_coords is not None:
        coords = [jnp.asarray(c, dtype=jnp.float64) for c in settings.transverse_coords]
    field_E, field_H, neff = compute_mode(
        frequency=settings.frequency,
        inv_permittivities=_inverse_permittivity(eps),
        inv_permeabilities=settings.permeability,
        resolution=settings.resolution,
        direction=settings.direction,
        mode_index=settings.mode_index,
        filter_pol=settings.filter_pol,
        dtype=dtype,
        transverse_coords=coords,
        symmetry=settings.symmetry,
        mode_backend=settings.mode_backend,
        mode_formulation=settings.formulation,
    )
    return jnp.real(neff), jnp.imag(neff), field_E, field_H


def _solve_candidates(
    eps_re: jax.Array,
    eps_im: jax.Array,
    settings: ModeSolveSettings,
    num_modes: int,
    drop_spurious: bool = False,
) -> jax.Array:
    """Solve one cross-section and return the E fields of its first ``num_modes`` modes.

    One call to :func:`fdtdx.core.physics.modes.compute_modes`, i.e. one eigen-solve, rather than
    one solve per candidate.

    Args:
        eps_re (jax.Array): Real part of the diagonal permittivity, shape ``(1 or 3, nx, ny, nz)``.
        eps_im (jax.Array): Imaginary part, same shape.
        settings (ModeSolveSettings): The solve settings; ``mode_index`` is ignored.
        num_modes (int): How many modes to return.
        drop_spurious (bool): Remove non-physical modes from the candidate list first.

    Returns:
        jax.Array: The E fields, shape ``(num_modes, 3, nx, ny, nz)``.
    """
    dtype = jnp.float64 if settings.double_precision else jnp.float32
    eps = jnp.asarray(eps_re) + 1j * jnp.asarray(eps_im)
    coords = None
    if settings.transverse_coords is not None:
        coords = [jnp.asarray(c, dtype=jnp.float64) for c in settings.transverse_coords]
    field_E, _, _ = compute_modes(
        frequency=settings.frequency,
        inv_permittivities=_inverse_permittivity(eps),
        inv_permeabilities=settings.permeability,
        num_modes=num_modes,
        resolution=settings.resolution,
        direction=settings.direction,
        filter_pol=settings.filter_pol,
        dtype=dtype,
        transverse_coords=coords,
        symmetry=settings.symmetry,
        mode_backend=settings.mode_backend,
        mode_formulation=settings.formulation,
        drop_spurious=drop_spurious,
    )
    return jax.lax.stop_gradient(field_E)


def _sensitivity_from_fields(
    field_E: jax.Array,
    field_H: jax.Array,
    settings: ModeSolveSettings,
    ncomp: int,
    prop_axis: int,
) -> jax.Array:
    """First-order ``d n_eff / d eps`` per cell and component, shaped like the permittivity.

    On the 9-component tier the same reciprocity integral gives every entry of the *transverse*
    block, because the first-order shift of a holomorphic eigenvalue is the bilinear form

    .. math::

        \\delta n_{\\mathrm{eff}} = \\frac{1}{2}\\,
        \\frac{\\int \\tilde{\\mathbf{E}} \\cdot \\delta\\varepsilon \\cdot \\mathbf{E}\\, w\\,dA}
             {\\int (\\mathbf{E}_t \\times \\mathbf{H}_t)\\cdot \\hat{e}_p \\, dA}

    with the partner field ``E~`` the backward mode's. Writing that as ``E`` with its propagation
    component negated — which is what makes ``d n_eff / d eps_ab = 0.5 s_a E_a E_b w / flux``,
    unconjugated — assumes the medium has a mirror plane at the cross-section. It therefore holds
    for the transverse block and for the transverse formulation, which is where this function is
    used; the four-component tier takes the exact matrix-level adjoint instead (see
    :func:`mode_neff_parts`), because with a longitudinal entry present the backward mode is a
    genuinely different field and the reciprocity integral written this way gets the ``zx`` and
    ``zy`` entries wrong by a sign.

    Args:
        field_E (jax.Array): Mode electric field, shape ``(3, nx, ny, nz)``.
        field_H (jax.Array): Mode magnetic field, same shape.
        settings (ModeSolveSettings): The solve settings (grid, direction).
        ncomp (int): Component count of the permittivity array (1, 3 or 9).
        prop_axis (int): Propagation axis.

    Returns:
        jax.Array: Complex sensitivity of the same shape as the permittivity array.
    """
    weights = jnp.asarray(_area_weights(settings, (ncomp, *field_E.shape[1:]), prop_axis), dtype=field_E.real.dtype)
    ax_a, ax_b = (prop_axis + 1) % 3, (prop_axis + 2) % 3
    flux = jnp.sum((field_E[ax_a] * field_H[ax_b] - field_E[ax_b] * field_H[ax_a]) * weights)
    if settings.direction == "-":
        # The "-" mode is returned with -H_t and -E_p, which flips the sign of the flux integral
        # while the physical sensitivity is unchanged.
        flux = -flux
    signs = np.ones(3)
    signs[prop_axis] = -1.0
    sign_array = jnp.asarray(signs, dtype=field_E.real.dtype)[:, None, None, None]
    if ncomp == 9:
        # d n_eff / d eps_ab, row-major, with the sign on the *row* index (the partner field).
        rows = sign_array * field_E
        outer = rows[:, None] * field_E[None, :]
        # The partner field used here is the mode with its propagation component negated, which is
        # the backward mode only when the medium has a mirror plane at the cross-section, i.e. when
        # the four longitudinal entries vanish. The transverse formulation drops them, so the solved
        # n_eff genuinely does not depend on them and their sensitivity is zero *for this solver*;
        # the continuum value is not zero, which is what the formulation choice is about.
        is_prop = np.arange(3) == prop_axis
        carried = jnp.asarray(~(is_prop[:, None] ^ is_prop[None, :]), dtype=field_E.real.dtype)
        outer = outer * carried[:, :, None, None, None]
        return 0.5 * outer.reshape(9, *field_E.shape[1:]) * weights[None] / flux
    per_component = 0.5 * sign_array * field_E**2 * weights[None] / flux
    if ncomp == 1:
        # One number per cell drives all three diagonal entries, so the three add.
        return jnp.sum(per_component, axis=0, keepdims=True)
    return per_component


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _neff_parts(eps_re: jax.Array, eps_im: jax.Array, settings: ModeSolveSettings) -> tuple[jax.Array, jax.Array]:
    """``(Re n_eff, Im n_eff)`` as two real scalars, differentiable in both permittivity parts.

    Args:
        eps_re (jax.Array): Real part of the diagonal permittivity.
        eps_im (jax.Array): Imaginary part.
        settings (ModeSolveSettings): The solve settings.

    Returns:
        tuple[jax.Array, jax.Array]: ``(Re n_eff, Im n_eff)``.
    """
    re, im, _, _ = _solve(eps_re, eps_im, settings)
    return re, im


def _neff_parts_fwd(eps_re, eps_im, settings):
    re, im, field_E, field_H = _solve(eps_re, eps_im, settings)
    ncomp = eps_re.shape[0]
    prop_axis = _propagation_axis(eps_re.shape)
    grad = _sensitivity_from_fields(field_E, field_H, settings, ncomp, prop_axis)
    return (re, im), (grad,)


def _neff_parts_bwd(settings, residuals, cotangents):
    del settings
    (grad,) = residuals
    ct_re, ct_im = cotangents
    g_re, g_im = jnp.real(grad), jnp.imag(grad)
    # n_eff is holomorphic in eps, so d n_eff / d Re(eps) = g and d n_eff / d Im(eps) = 1j g.
    d_eps_re = ct_re * g_re + ct_im * g_im
    d_eps_im = -ct_re * g_im + ct_im * g_re
    return d_eps_re, d_eps_im


_neff_parts.defvjp(_neff_parts_fwd, _neff_parts_bwd)


# ------------------------------------------------------------------------------------------------
# public interface
# ------------------------------------------------------------------------------------------------


def _split(permittivity: Any) -> tuple[jax.Array, jax.Array]:
    """Split a possibly-complex permittivity array into real and imaginary parts.

    Args:
        permittivity (Any): Array-like of shape ``(1 or 3, nx, ny, nz)``.

    Returns:
        tuple[jax.Array, jax.Array]: ``(real part, imaginary part)``.
    """
    eps = jnp.asarray(permittivity)
    if jnp.issubdtype(eps.dtype, jnp.complexfloating):
        return jnp.real(eps), jnp.imag(eps)
    return eps, jnp.zeros_like(eps)


def carries_longitudinal_entries(permittivity: Any, settings: ModeSolveSettings) -> bool:
    """Whether the solve these settings ask for carries ``eps_xz`` / ``eps_zx`` / ``eps_yz`` / ``eps_zy``.

    It is a property of the *layout* and the formulation, not of the values, and that is deliberate:
    under ``formulation="auto"`` a nine-component cross-section whose longitudinal entries happen to
    be exactly zero still has a non-zero derivative with respect to them, because perturbing one
    moves the solve onto the operator that carries it. Reporting zero there would match the operator
    that ran and disagree with a finite difference through the same front end. It also matches what
    ``jax.grad`` does: the permittivity is a tracer inside the backward and cannot be inspected, so
    the differentiable front end assumes the entries are carried as well.

    The cost is that ``mode_sensitivity`` and the callback path are out of reach for a nine-component
    cross-section under ``"auto"``, including one whose tensor is only transversely off-diagonal.
    ``formulation="transverse"`` asks for the old behaviour explicitly and gets it.

    Args:
        permittivity (Any): The permittivity array.
        settings (ModeSolveSettings): The solve settings.

    Returns:
        bool: ``True`` when the four longitudinal entries are carried.
    """
    ncomp = int(jnp.asarray(permittivity).shape[0])
    return ncomp == 9 and settings.formulation != "transverse"


def _solve_used_the_full_operator(permittivity: Any, settings: ModeSolveSettings) -> bool:
    """Whether the solve that *ran* assembled the four-component operator.

    The value-aware question, as opposed to :func:`carries_longitudinal_entries`'s layout-aware one.
    They differ on a nine-component cross-section whose four longitudinal entries are all exactly
    zero: the solve took the transverse operator (so a report about *that* solve is well defined),
    while the derivative with respect to those entries is not the transverse operator's, because
    perturbing one moves the solve. Reporting functions want this predicate; chain-rule components
    want the other one.

    Args:
        permittivity (Any): The permittivity array.
        settings (ModeSolveSettings): The solve settings.

    Returns:
        bool: ``True`` when the four-component operator was assembled.
    """
    array = jnp.asarray(permittivity)
    if int(array.shape[0]) != 9 or settings.formulation == "transverse":
        return False
    if settings.formulation == "full" or is_jax_tracer(array):
        return True
    prop_axis = _propagation_axis(array.shape)
    tensor = np.asarray(array).reshape(3, 3, *array.shape[1:])
    is_prop = np.arange(3) == prop_axis
    return bool(np.any(tensor[is_prop[:, None] ^ is_prop[None, :]] != 0.0))


def mode_neff_parts(permittivity: Any, settings: ModeSolveSettings) -> tuple[jax.Array, jax.Array]:
    """Effective index of one mode as a differentiable ``(Re, Im)`` pair.

    Both outputs are real, so ``jax.grad`` applies to either without a complex-derivative
    convention getting in the way. The permittivity may be real or complex; when it is complex the
    gradient with respect to its real and imaginary parts comes off the same backward.

    Where the backward comes from depends on the tier. For one, three or nine components *without*
    the longitudinal entries it is the field-level reciprocity integral evaluated on the field the
    forward already returned, so it costs no extra solve. With the longitudinal entries carried,
    that integral's partner field is no longer the mode with its propagation component negated, so
    the index instead comes off the JAX-native pipeline, where ``jax.grad`` contracts the *discrete*
    operator against a solved left eigenvector and is exact for every entry by construction. That
    path runs the same eigen-solve; what it costs is the second Arnoldi run for the left
    eigenvector.

    Args:
        permittivity (Any): Relative permittivity, shape ``(1, 3 or 9, nx, ny, nz)`` with the
            propagation axis of length one. Complex entries are allowed (a lossy or metallic cell
            has ``Re eps < 0``, which the material table cannot express but the array can).
        settings (ModeSolveSettings): Frequency, grid, mode index and backend.

    Returns:
        tuple[jax.Array, jax.Array]: ``(Re n_eff, Im n_eff)``.

    Raises:
        ValueError: If the longitudinal entries are carried and the settings put the exact path out
            of reach, since the reciprocity integral would then be wrong rather than merely coarse.
    """
    eps_re, eps_im = _split(permittivity)
    _propagation_axis(eps_re.shape)
    if carries_longitudinal_entries(permittivity, settings):
        obstacle = _differentiable_path_obstacle(settings)
        if obstacle is not None:
            raise ValueError(
                "a nine-component cross-section carrying the longitudinal permittivity entries "
                f"needs the exact matrix-level adjoint, and it is out of reach here: {obstacle}. "
                "Use formulation='transverse' to drop those entries deliberately (their "
                "sensitivity is then reported as zero, which is what that solve depends on)."
            )
        solution = _mode_solve_differentiable(eps_re, eps_im, settings)
        return jnp.real(solution.neff), jnp.imag(solution.neff)
    return _neff_parts(eps_re, eps_im, settings)


def mode_neff(permittivity: Any, settings: ModeSolveSettings) -> jax.Array:
    """Complex effective index of one mode, differentiable in the permittivity.

    Args:
        permittivity (Any): Diagonal relative permittivity, shape ``(1 or 3, nx, ny, nz)``.
        settings (ModeSolveSettings): Frequency, grid, mode index and backend.

    Returns:
        jax.Array: Complex scalar ``n_eff``. Take ``jnp.real`` / ``jnp.imag`` for phase and loss;
        differentiating either goes through :func:`mode_neff_parts`.
    """
    re, im = mode_neff_parts(permittivity, settings)
    return re + 1j * im


def _differentiable_path_obstacle(settings: ModeSolveSettings) -> str | None:
    """Why :func:`fdtdx.core.physics.mode_backend.jax_modes.compute_modes_jax` cannot serve, if so."""
    if settings.mode_backend == "tidy3d":
        return "mode_backend='tidy3d' is a callback by construction"
    if settings.filter_pol is not None:
        return "filter_pol selects modes by inspecting field values, which is not a static reordering"
    if any(value == 1 for value in settings.symmetry):
        return "a magnetic (PMC) min-edge wall has no closed-form left eigenvector"
    if not jax.config.jax_enable_x64:
        return "the differentiable path is complex128 and needs jax_enable_x64"
    return None


def mode_solve(
    permittivity: Any,
    settings: ModeSolveSettings,
    differentiable_fields: bool | None = None,
) -> ModeSolution:
    """Solve one mode and return the index together with its fields.

    By default the solve runs through the JAX-native pipeline
    (:func:`fdtdx.core.physics.mode_backend.jax_modes.compute_modes_jax`), so the **fields** carry a
    permittivity gradient and an overlap-integral objective such as ``log10 sum |E|^2`` is
    differentiable. The eigen-solve itself is still ARPACK, behind one ``custom_vjp`` whose backward
    covers the eigenvectors with one bordered shifted solve per mode.

    Args:
        permittivity (Any): Relative permittivity, shape ``(1, 3 or 9, nx, ny, nz)``.
        settings (ModeSolveSettings): Frequency, grid, mode index and backend.
        differentiable_fields (bool | None): ``None`` takes the differentiable path when the
            settings allow it and warns when falling back to the callback path; ``True`` requires
            it and raises otherwise; ``False`` always takes the callback path, whose fields carry
            ``jax.lax.stop_gradient`` as they always did.

    Returns:
        ModeSolution: The complex index and the E and H fields.

    Raises:
        ValueError: If ``differentiable_fields=True`` and the settings are outside its scope.
    """
    eps_re, eps_im = _split(permittivity)
    _propagation_axis(eps_re.shape)
    obstacle = _differentiable_path_obstacle(settings)
    if differentiable_fields is not False:
        if obstacle is None:
            return _mode_solve_differentiable(eps_re, eps_im, settings)
        if differentiable_fields is True:
            raise ValueError(f"differentiable mode fields are not available here: {obstacle}")
        warnings.warn(
            f"mode_solve fell back to the callback path, so its fields carry no gradient: {obstacle}. "
            "Pass differentiable_fields=False to silence this, or mode_neff for the index gradient.",
            UserWarning,
            stacklevel=2,
        )
    re, im, field_E, field_H = _solve(eps_re, eps_im, settings)
    return ModeSolution(
        neff=re + 1j * im,
        E=jax.lax.stop_gradient(field_E),
        H=jax.lax.stop_gradient(field_H),
    )


def _mode_solve_differentiable(
    eps_re: jax.Array,
    eps_im: jax.Array,
    settings: ModeSolveSettings,
) -> ModeSolution:
    """One mode off the JAX-native pipeline, fields included in the gradient."""
    from fdtdx.core.physics.mode_backend.jax_modes import compute_modes_jax

    eps = jnp.asarray(eps_re) + 1j * jnp.asarray(eps_im)
    coords = None if settings.transverse_coords is None else [np.asarray(c) for c in settings.transverse_coords]
    field_E, field_H, neff = compute_modes_jax(
        frequency=settings.frequency,
        inv_permittivities=_inverse_permittivity(eps),
        inv_permeabilities=settings.permeability,
        num_modes=settings.mode_index + 1,
        resolution=settings.resolution,
        transverse_coords=coords,
        direction=settings.direction,
        symmetry=settings.symmetry,
        formulation=settings.formulation,
    )
    index = settings.mode_index
    return ModeSolution(neff=neff[index], E=field_E[index], H=field_H[index])


def mode_sensitivity(permittivity: Any, settings: ModeSolveSettings) -> tuple[jax.Array, jax.Array]:
    """The mode index and the raw per-cell ``d n_eff / d eps``, without going through ``jax.grad``.

    Useful for reporting a calibration table, for chaining into a hand-written adjoint, and for
    reading the sensitivity map itself (it is the mode's own energy density, so it shows where a
    design change can move the index at all).

    Args:
        permittivity (Any): Diagonal relative permittivity, shape ``(1 or 3, nx, ny, nz)``.
        settings (ModeSolveSettings): Frequency, grid, mode index and backend.

    Returns:
        tuple[jax.Array, jax.Array]: ``(complex n_eff, complex sensitivity)``; the sensitivity has
        the same shape as ``permittivity``.

    Raises:
        ValueError: When the solve assembled the four-component operator, where the reciprocity
            integral's partner field is not the backward mode. A nine-component cross-section whose
            four longitudinal entries are all exactly zero is *not* refused: it takes the transverse
            operator, for which the integral is exact. That is a looser test than the one
            :func:`mode_neff_parts` uses, deliberately — this function reports on the solve that
            ran, while that one is a link in a chain rule and has to describe the solve that would
            run after a perturbation.
    """
    eps_re, eps_im = _split(permittivity)
    prop_axis = _propagation_axis(eps_re.shape)
    if _solve_used_the_full_operator(permittivity, settings):
        raise ValueError(
            "mode_sensitivity is the field-level reciprocity integral, whose partner field is the "
            "mode with its propagation component negated. That is the backward mode only when the "
            "cross-section has a mirror plane, and a nine-component tensor carrying eps_xz / "
            "eps_zx / eps_yz / eps_zy does not. Use jax.grad(mode_neff), which contracts the "
            "discrete operator against a solved left eigenvector, or formulation='transverse'."
        )
    re, im, field_E, field_H = _solve(eps_re, eps_im, settings)
    grad = _sensitivity_from_fields(field_E, field_H, settings, eps_re.shape[0], prop_axis)
    return re + 1j * im, grad


# ------------------------------------------------------------------------------------------------
# mode tracking
# ------------------------------------------------------------------------------------------------


def mode_overlap(field_a: Any, field_b: Any) -> float:
    """Normalized overlap magnitude of two mode fields, in ``[0, 1]``.

    ``|<a, b>| / (||a|| ||b||)`` with the conjugate on the first argument, so it is blind to the
    arbitrary global phase and amplitude an eigensolver returns.

    Args:
        field_a (Any): First field, any shape.
        field_b (Any): Second field, the same shape.

    Returns:
        float: The overlap magnitude.

    Raises:
        ValueError: If the two fields differ in shape.
    """
    a = np.asarray(field_a)
    b = np.asarray(field_b)
    if a.shape != b.shape:
        raise ValueError(f"the two fields differ in shape: {a.shape} vs {b.shape}")
    norm = np.sqrt(np.sum(np.abs(a) ** 2)) * np.sqrt(np.sum(np.abs(b) ** 2))
    if norm == 0.0:
        return 0.0
    return float(np.abs(np.sum(np.conj(a) * b)) / norm)


def track_mode(previous_field: Any, candidates: Sequence[Any], min_overlap: float = 0.0) -> ModeMatch:
    """Pick the candidate that continues ``previous_field``, by field overlap.

    Sorting by ``Re(n_eff)`` reorders the mode list the moment two indices cross, which happens
    routinely during a parameter sweep or inside a finite-difference step. Selecting by overlap with
    the previous iteration's field follows one physical mode instead.

    Args:
        previous_field (Any): The field of the mode being followed, from the previous parameter
            value. Any shape, as long as the candidates match it.
        candidates (Sequence[Any]): The fields of the modes solved at the new parameter value.
        min_overlap (float): Raise if the best overlap falls below this. The default of ``0.0``
            never raises; ``0.5`` is a reasonable gate for a sweep with small steps.

    Returns:
        ModeMatch: The selected index, its overlap and every candidate's overlap.

    Raises:
        ValueError: If ``candidates`` is empty, or the best overlap is below ``min_overlap``.
    """
    if len(candidates) == 0:
        raise ValueError("track_mode needs at least one candidate")
    overlaps = np.array([mode_overlap(previous_field, c) for c in candidates], dtype=np.float64)
    index = int(np.argmax(overlaps))
    best = float(overlaps[index])
    if best < min_overlap:
        raise ValueError(
            f"the best mode overlap is {best:.3f}, below the required {min_overlap:.3f}: the mode "
            "being tracked is not in the candidate list. Solve more modes, or take a smaller step."
        )
    return ModeMatch(index=index, overlap=best, overlaps=overlaps)


@dataclass
class ModeTracker:
    """Follow one physical mode across a parameter sweep. **This is the way to run a sweep.**

    Indexing the list sorted by ``Re(n_eff)`` is only stable while no two modes cross. They cross
    routinely - a heater sweep, a width sweep, even the two evaluations of a finite difference - and
    a fixed ``mode_index`` then silently starts reporting a different physical mode. Measured on the
    thermo-optic phase shifter, that is a 25 % error in ``d n_eff / d T`` (D2 Slide 6).

    The tracker keeps the previous step's field and selects, at every new parameter value, the
    candidate with the largest overlap with it - so tracking is the default here rather than an
    option a caller has to remember. The first call has nothing to track against and uses
    ``settings.mode_index``; every later call follows the mode.

    .. code-block:: python

        tracker = ModeTracker(settings)
        for temperature in sweep:
            neff = tracker.neff(permittivity_at(temperature))   # differentiable in the permittivity
            print(tracker.last_match.overlap, tracker.mode_index)

    Cost: two eigen-solves per step - one for the candidate list, one for the differentiable index
    at the selected mode. The candidate solve is a single call to
    :func:`fdtdx.core.physics.modes.compute_modes`, not one solve per candidate.

    Attributes:
        settings: The solve settings. Its ``mode_index`` seeds the first step only.
        num_candidates: How many modes to compare against the previous field.
        min_overlap: Raise when the best overlap falls below this - a sweep step so large that the
            tracked mode is not in the candidate list at all. ``0.5`` is a reasonable gate.
        drop_spurious: Remove non-physical modes from the candidate list; on by default here,
            because a wall artefact in the list can win an overlap contest against nothing.
        mode_index: The index selected at the last step.
        last_match: The last :class:`ModeMatch`, or ``None`` before the second step.
    """

    settings: ModeSolveSettings
    num_candidates: int = 4
    min_overlap: float = 0.0
    drop_spurious: bool = True
    mode_index: int = -1
    last_match: ModeMatch | None = None
    _reference: np.ndarray | None = None

    def __post_init__(self):
        if self.mode_index < 0:
            self.mode_index = int(self.settings.mode_index)

    def reset(self) -> None:
        """Forget the tracked mode, so the next call starts from ``settings.mode_index`` again."""
        self._reference = None
        self.last_match = None
        self.mode_index = int(self.settings.mode_index)

    def select(self, permittivity: Any) -> ModeSolveSettings:
        """Choose the mode index that continues the tracked mode and remember its field.

        Args:
            permittivity (Any): Diagonal relative permittivity at the new parameter value.

        Returns:
            ModeSolveSettings: A copy of ``settings`` whose ``mode_index`` is the selected one.
        """
        eps_re, eps_im = _split(permittivity)
        _propagation_axis(eps_re.shape)
        if self._reference is None:
            settings = self.settings.with_mode_index(self.mode_index)
            self._reference = np.asarray(mode_solve(permittivity, settings).E)
            self.last_match = None
            return settings
        fields = np.asarray(
            _solve_candidates(
                eps_re,
                eps_im,
                self.settings,
                self.num_candidates,
                drop_spurious=self.drop_spurious,
            )
        )
        match = track_mode(self._reference, list(fields), min_overlap=self.min_overlap)
        self.mode_index = match.index
        self.last_match = match
        self._reference = fields[match.index]
        return self.settings.with_mode_index(match.index)

    def neff(self, permittivity: Any) -> jax.Array:
        """Complex effective index of the tracked mode, differentiable in the permittivity.

        Args:
            permittivity (Any): Diagonal relative permittivity at the new parameter value.

        Returns:
            jax.Array: The complex ``n_eff`` of the mode that continues the tracked one.
        """
        return mode_neff(permittivity, self.select(permittivity))

    def solve(self, permittivity: Any) -> ModeSolution:
        """Solve the tracked mode and return its index and fields (fields not differentiable).

        Args:
            permittivity (Any): Diagonal relative permittivity at the new parameter value.

        Returns:
            ModeSolution: The complex index and the E and H fields of the tracked mode.
        """
        return mode_solve(permittivity, self.select(permittivity))


def tracked_mode_index(
    permittivity: Any,
    settings: ModeSolveSettings,
    previous_field: Any,
    num_candidates: int = 4,
    min_overlap: float = 0.0,
) -> ModeMatch:
    """Solve the first ``num_candidates`` modes and report which one continues ``previous_field``.

    Runs eagerly (it inspects the fields), so it cannot sit inside ``jax.jit``. Feed its ``index``
    into :meth:`ModeSolveSettings.with_mode_index` and call :func:`mode_neff` with that, which keeps
    the differentiable path a single solve.

    Costs **one** eigen-solve regardless of ``num_candidates``: the backend already computes a
    dozen eigenpairs per call, and :func:`fdtdx.core.physics.modes.compute_modes` hands the sorted
    list back instead of throwing all but one away.

    Args:
        permittivity (Any): Diagonal relative permittivity at the new parameter value.
        settings (ModeSolveSettings): The solve settings; its ``mode_index`` is ignored.
        previous_field (Any): The tracked mode's E field at the previous parameter value.
        num_candidates (int): How many modes to compare.
        min_overlap (float): Gate passed to :func:`track_mode`.

    Returns:
        ModeMatch: The selected index, its overlap and every candidate's overlap.
    """
    eps_re, eps_im = _split(permittivity)
    _propagation_axis(eps_re.shape)
    fields = np.asarray(_solve_candidates(eps_re, eps_im, settings, num_candidates))
    return track_mode(np.asarray(previous_field), list(fields), min_overlap=min_overlap)
