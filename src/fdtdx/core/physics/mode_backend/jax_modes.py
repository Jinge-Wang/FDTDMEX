"""A ``compute_modes``-shaped front end with the gradient connected to the fields as well.

:func:`fdtdx.core.physics.modes.compute_modes` reaches its backend through ``jax.pure_callback``
and wraps every array argument in ``jax.lax.stop_gradient``: the whole solve, the axis rotation and
the field reconstruction all happen in numpy inside the callback, so nothing downstream of it
carries a permittivity gradient. :mod:`fdtdx.core.physics.mode_adjoint` closes that for the effective
index by re-deriving the sensitivity from the returned fields; the fields themselves stayed frozen.

This module is the same pipeline written so that the gradient crosses it: the tensor inversion, the
axis rotation, the two-dimensional collapse, the field reconstruction and the Poynting
normalisation are all ``jnp``, and only the eigen-solve is opaque - behind the single ``custom_vjp``
of :mod:`fdtdx.core.physics.mode_backend.jax_solve`, whose backward now covers the eigenvectors as
well as the eigenvalues. So ``d(fields) / d eps`` is available and an overlap-integral objective is
differentiable.

What it does not do, and why. It is the ``fdtdmex`` backend only (Tidy3D is a callback by
construction), straight waveguides only (no bend transform), and electric walls only (the mode
adjoint's left eigenvector is a PEC statement). ``filter_pol`` and the spurious-mode filter are not
applied: both select modes by inspecting field values, which is a data-dependent reordering and
would need the selection to be static. Select with ``target_neff`` and ``mode_index``, or track the
mode with :class:`fdtdx.core.physics.mode_adjoint.ModeTracker`.

The transverse grid is *static*: the difference matrices are built in scipy from the cell-edge
coordinates. Coordinates are geometry, not a design variable, so that costs nothing. The
shift-invert target is not static - it reaches the eigen-solve as a runtime scalar - so it is
derived from the permittivity even under ``jax.grad``, and ``target_neff`` only overrides it.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.constants import c, eta0
from fdtdx.core.jax.utils import is_jax_tracer
from fdtdx.core.physics.metrics import normalize_by_poynting_flux
from fdtdx.core.physics.mode_backend import TOL_TENSORIAL, ModeLongitudinalOffdiagWarning
from fdtdx.core.physics.mode_backend.jax_solve import solve_modes_diagonal_jax
from fdtdx.core.physics.mode_backend.operator import build_derivative_matrices, primal_dual_steps

__all__ = ["compute_modes_jax"]

#: Component permutations that put the propagation axis last, matching ``modes.py``.
_ROTATE_DIAGONAL = {0: [1, 2, 0], 1: [0, 2, 1], 2: [0, 1, 2]}
_ROTATE_TENSOR = {
    0: [4, 5, 3, 7, 8, 6, 1, 2, 0],
    1: [0, 2, 1, 6, 8, 7, 3, 5, 4],
    2: [0, 1, 2, 3, 4, 5, 6, 7, 8],
}

#: Flat indices of the longitudinal off-diagonal entries in the rotated (solver-frame) layout.
_LONGITUDINAL = {2: "xz", 5: "yz", 6: "zx", 7: "zy"}


def _permittivity_from_inverse(inv_permittivities: jax.Array) -> jax.Array:
    """Invert the material array, as a genuine 3x3 inverse on the 9-component tier."""
    if inv_permittivities.shape[0] != 9:
        return 1.0 / inv_permittivities
    tensor = inv_permittivities.reshape(3, 3, *inv_permittivities.shape[1:])
    to_last = (2, 3, 4, 0, 1)
    to_first = (3, 4, 0, 1, 2)
    inverted = jnp.linalg.inv(tensor.transpose(to_last)).transpose(to_first)
    return inverted.reshape(9, *inv_permittivities.shape[1:])


def _required(components: dict[str, jax.Array | None], key: str, what: str) -> jax.Array:
    """The component the diagonal operator cannot do without, or a clear error."""
    value = components[key]
    if value is None:
        raise ValueError(f"the {what} has no {key!r} component to assemble the mode operator from")
    return value


def _components(rotated: jax.Array, nx: int, ny: int, what: str) -> dict[str, jax.Array | None]:
    """Flatten a rotated cross-section into the five components the operator consumes."""
    ncomp = rotated.shape[0]
    if ncomp == 1:
        diagonal = (rotated[0], rotated[0], rotated[0])
        off: tuple[jax.Array, jax.Array] | None = None
    elif ncomp == 3:
        diagonal = (rotated[0], rotated[1], rotated[2])
        off = None
    elif ncomp == 9:
        diagonal = (rotated[0], rotated[4], rotated[8])
        off = (rotated[1], rotated[3])
        if not is_jax_tracer(rotated):
            concrete = np.asarray(rotated)
            reported = {
                name: float(np.max(np.abs(concrete[flat])))
                for flat, name in _LONGITUDINAL.items()
                if float(np.max(np.abs(concrete[flat]))) > TOL_TENSORIAL
            }
            if reported:
                listing = ", ".join(f"{k} (largest |entry| {v:.4g})" for k, v in sorted(reported.items()))
                warnings.warn(
                    f"the {what} cross-section carries the longitudinal off-diagonal entries "
                    f"{listing}, above TOL_TENSORIAL={TOL_TENSORIAL:g}; they are dropped. See "
                    "fdtdx.core.physics.mode_backend for why an eigenproblem linear in n_eff**2 "
                    "cannot represent them.",
                    ModeLongitudinalOffdiagWarning,
                    stacklevel=3,
                )
    else:
        raise ValueError(f"{what} component axis must be 1, 3 or 9 long, got {ncomp}")

    def flat(component: jax.Array) -> jax.Array:
        return jnp.asarray(component, dtype=jnp.complex128).reshape(nx * ny)

    return {
        "xx": flat(diagonal[0]),
        "yy": flat(diagonal[1]),
        "zz": flat(diagonal[2]),
        "xy": None if off is None else flat(off[0]),
        "yx": None if off is None else flat(off[1]),
    }


def _shift_invert_guess(components: dict[str, jax.Array | None]) -> jax.Array:
    """Largest index the cross-section can support, from the transverse block and ``eps_zz``.

    Traced, not concrete: the shift reaches the eigen-solve as a runtime scalar, so this works
    inside ``jax.grad`` and ``jax.jit`` without the caller naming a target. With a transverse
    off-diagonal the bound is the largest eigenvalue of the 2x2 transverse block, not the largest
    diagonal entry - a rotated uniaxial tensor hides its extraordinary index in the off-diagonal.

    Args:
        components (dict[str, jax.Array | None]): The five flattened permittivity components.

    Returns:
        jax.Array: A real scalar just above the largest supportable index.
    """
    xx = jnp.asarray(components["xx"])
    yy = jnp.asarray(components["yy"])
    zz = jnp.asarray(components["zz"])
    largest = jnp.max(jnp.real(jnp.stack((xx, yy, zz))))
    off_xy, off_yx = components["xy"], components["yx"]
    if off_xy is not None and off_yx is not None:
        half_trace = 0.5 * (xx + yy)
        radicand = 0.25 * (xx - yy) ** 2 + jnp.asarray(off_xy) * jnp.asarray(off_yx)
        transverse = jnp.max(jnp.real(half_trace + jnp.sqrt(radicand)))
        largest = jnp.maximum(largest, transverse)
    return jnp.sqrt(largest) * (1.0 + 1e-6) + 1e-6


def compute_modes_jax(
    frequency: float,
    inv_permittivities: Any,
    inv_permeabilities: Any = 1.0,
    num_modes: int = 1,
    resolution: float | None = None,
    transverse_coords: Sequence[Any] | None = None,
    direction: Literal["+", "-"] = "+",
    target_neff: float | None = None,
    symmetry: tuple[int, int] = (0, 0),
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Solve a cross-section's modes with the gradient connected to the fields.

    The return contract is the one of :func:`fdtdx.core.physics.modes.compute_modes`: ``E`` and
    ``H`` of shape ``(num_modes, 3, nx, ny, nz)`` in the *physical* axis frame with the propagation
    axis of length one, normalised to unit Poynting flux one mode at a time, and ``n_eff`` of shape
    ``(num_modes,)``, sorted by descending ``Re(n_eff)`` or by distance from ``target_neff``.

    Args:
        frequency (float): Operating frequency in Hz.
        inv_permittivities (Any): Inverse relative permittivity, shape ``(1 | 3 | 9, nx, ny, nz)``
            with exactly one spatial axis of length one (the propagation axis).
        inv_permeabilities (Any): Inverse relative permeability: a scalar, or an array of the same
            layout with 1 or 3 components. A tensorial permeability is refused.
        num_modes (int): How many modes to return.
        resolution (float | None): Uniform transverse cell size in metres, or ``None`` when
            ``transverse_coords`` is given.
        transverse_coords (Sequence[Any] | None): Cell-edge coordinates in metres of the two
            transverse axes. Concrete (they are geometry, and the difference matrices are scipy).
        direction (str): ``"+"`` or ``"-"``.
        target_neff (float | None): Shift-invert target and sort key. Required when the
            permittivity is traced.
        symmetry (tuple[int, int]): Per-transverse-axis min-edge wall, 0 = electric, 1 = magnetic.
            Magnetic walls are refused: the mode adjoint's left eigenvector is a PEC statement.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: ``(E, H, n_eff)``.

    Raises:
        ValueError: On a layout the front end does not accept, or when the shift cannot be derived.
        NotImplementedError: On a magnetic wall.
    """
    permittivities = _permittivity_from_inverse(jnp.asarray(inv_permittivities))
    if permittivities.ndim != 4:
        raise ValueError(f"permittivity must have shape (1|3|9, nx, ny, nz), got {permittivities.shape}")
    spatial = permittivities.shape[1:]
    singletons = [axis for axis, dim in enumerate(spatial) if dim == 1]
    if len(singletons) != 1:
        raise ValueError(f"exactly one spatial axis must have length one, got shape {permittivities.shape}")
    propagation_axis = singletons[0]
    transverse_axes = [axis for axis in range(3) if axis != propagation_axis]

    if transverse_coords is None:
        if resolution is None:
            raise ValueError("resolution is required when transverse_coords is not given")
        coords_m = [np.arange(spatial[axis] + 1) * float(resolution) for axis in transverse_axes]
    else:
        coords_m = [np.asarray(jax.lax.stop_gradient(coord), dtype=np.float64) for coord in transverse_coords]
        for index, (coord, axis) in enumerate(zip(coords_m, transverse_axes, strict=True)):
            if coord.ndim != 1 or coord.shape[0] != spatial[axis] + 1:
                raise ValueError(f"transverse_coords[{index}] must be 1-D with length {spatial[axis] + 1}")

    cross = jnp.take(permittivities, indices=0, axis=propagation_axis + 1)
    rotation = _ROTATE_TENSOR if cross.shape[0] == 9 else _ROTATE_DIAGONAL
    if cross.shape[0] in (3, 9):
        cross = cross[jnp.array(rotation[propagation_axis]), :, :]

    if not np.isscalar(inv_permeabilities) and jnp.asarray(inv_permeabilities).ndim > 0:
        mu_array = jnp.asarray(inv_permeabilities)
        if mu_array.shape[0] == 9:
            raise NotImplementedError("the differentiable mode path carries a diagonal permeability only")
        mu_cross = jnp.take(1.0 / mu_array, indices=0, axis=propagation_axis + 1)
        if mu_cross.shape[0] == 3:
            mu_cross = mu_cross[jnp.array(_ROTATE_DIAGONAL[propagation_axis]), :, :]
    else:
        mu_cross = jnp.full((1, *cross.shape[1:]), 1.0 / float(inv_permeabilities))

    # Two-dimensional collapse: an invariant transverse axis of exactly two cells is solved on one
    # cell and repeated back, exactly as compute_mode does.
    collapsed_axis = None
    full_coords_m = [coord.copy() for coord in coords_m]
    cross_shape = cross.shape[1:]
    if 2 in cross_shape:
        collapsed_axis = list(cross_shape).index(2)
        keep = (
            (slice(None), slice(None), slice(0, 1)) if collapsed_axis == 1 else (slice(None), slice(0, 1), slice(None))
        )
        cross = cross[keep]
        mu_cross = mu_cross[keep]
        coords_m[collapsed_axis] = coords_m[collapsed_axis][:2]
    nx, ny = cross.shape[1], cross.shape[2]

    if any(s == 1 for s in symmetry):
        raise NotImplementedError(
            "the differentiable mode path supports electric (PEC) min-edge walls only; its backward "
            "uses a left eigenvector that is a PEC statement. Solve the full cross-section instead."
        )

    eps = _components(cross, nx, ny, "permittivity")
    mu = _components(mu_cross, nx, ny, "permeability")
    if mu["xy"] is not None:
        raise NotImplementedError("the differentiable mode path carries a diagonal permeability only")

    der_mats = build_derivative_matrices(coords_m[0], coords_m[1])
    cell_steps = (primal_dual_steps(coords_m[0]), primal_dual_steps(coords_m[1]))
    guess = jnp.asarray(float(target_neff)) if target_neff is not None else _shift_invert_guess(eps)
    k0 = 2.0 * np.pi * float(frequency) / c

    field_e, field_h, neff, keff = solve_modes_diagonal_jax(
        _required(eps, "xx", "permittivity"),
        _required(eps, "yy", "permittivity"),
        _required(eps, "zz", "permittivity"),
        _required(mu, "xx", "permeability"),
        _required(mu, "yy", "permeability"),
        _required(mu, "zz", "permeability"),
        der_mats,
        cell_steps,
        k0=k0,
        num_modes=max(2 * num_modes + 10, num_modes),
        neff_guess=guess,
        direction=direction,
        eps_xy=eps["xy"],
        eps_yx=eps["yx"],
    )
    n_complex = neff + 1j * keff
    if target_neff is not None:
        order = jnp.argsort(jnp.abs(jnp.real(n_complex) - float(target_neff)))
        field_e, field_h, n_complex = field_e[:, :, order], field_h[:, :, order], n_complex[order]
    if n_complex.shape[0] < num_modes:
        raise ValueError(f"the eigen-solve returned {n_complex.shape[0]} modes, fewer than the {num_modes} asked for")
    field_e = field_e[:, :, :num_modes]
    field_h = field_h[:, :, :num_modes] * eta0
    n_complex = n_complex[:num_modes]

    # (3, N, M) in the solver frame -> (M, 3, nx, ny, nz) in the physical frame.
    def to_physical(field: jax.Array, is_magnetic: bool) -> jax.Array:
        stacked = field.reshape(3, nx, ny, field.shape[2]).transpose(3, 0, 1, 2)
        if propagation_axis == 0:
            stacked = stacked[:, jnp.array([2, 0, 1])]
        elif propagation_axis == 1:
            stacked = stacked[:, jnp.array([0, 2, 1])]
            if is_magnetic:
                stacked = -stacked
        if collapsed_axis is not None:
            stacked = jnp.repeat(stacked, 2, axis=collapsed_axis + 2)
        return jnp.expand_dims(stacked, axis=propagation_axis + 2)

    mode_e = to_physical(field_e, is_magnetic=False)
    mode_h = to_physical(field_h, is_magnetic=True)

    area_weights = None
    if transverse_coords is not None:
        area = np.diff(full_coords_m[0])[:, None] * np.diff(full_coords_m[1])[None, :]
        weight_shape = [1, 1, 1]
        weight_shape[transverse_axes[0]] = area.shape[0]
        weight_shape[transverse_axes[1]] = area.shape[1]
        area_weights = jnp.asarray(area.reshape(weight_shape))

    normalized = [
        normalize_by_poynting_flux(mode_e[i], mode_h[i], axis=propagation_axis, area_weights=area_weights)
        for i in range(num_modes)
    ]
    return (
        jnp.stack([e for e, _ in normalized], axis=0),
        jnp.stack([h for _, h in normalized], axis=0),
        n_complex,
    )
