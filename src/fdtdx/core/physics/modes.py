import os
from collections import namedtuple
from types import SimpleNamespace
from typing import TYPE_CHECKING, List, Literal, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from loguru import logger

from fdtdx.constants import eta0
from fdtdx.core.axis import get_transverse_axes
from fdtdx.core.jax.utils import is_jax_tracer
from fdtdx.core.misc import expand_to_3x3
from fdtdx.core.physics.metrics import normalize_by_poynting_flux
from fdtdx.core.physics.mode_backend.bend import transform_cross_section
from fdtdx.core.physics.symmetry import (
    mirror_edge_coordinates,
    mirror_material_cross_section,
    project_onto_parity,
    restrict_to_kept_half,
)

if TYPE_CHECKING:  # the dispersion module pulls in the JAX operator; keep it off the import path
    from fdtdx.core.physics.mode_backend.dispersion import ModeDispersion

#: Default mode-solver backend. ``"fdtdmex"`` selects the native, Tidy3D-free full-vectorial FD
#: solver (Phase 4 Track A); ``"tidy3d"`` selects the legacy Tidy3D path (optional dependency, kept
#: for fully tensorial media, bends, and dev-time cross-checks). Override with the
#: ``FDTDMEX_MODE_BACKEND`` environment variable or the ``mode_backend`` argument of
#: :func:`compute_mode`.
_DEFAULT_MODE_BACKEND = "fdtdmex"

ModeTupleType = namedtuple("ModeTupleType", ["neff", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
"""A named tuple containing the mode fields and effective index.

Attributes:
    neff: Complex effective refractive index of the mode
    Ex: x-component of the electric field
    Ey: y-component of the electric field
    Ez: z-component of the electric field
    Hx: x-component of the magnetic field
    Hy: y-component of the magnetic field
    Hz: z-component of the magnetic field
"""


def compute_mode_polarization_fraction(
    mode: ModeTupleType,
    tangential_axes: tuple[int, int],
    pol: Literal["te", "tm"],
) -> float:
    """Mode polarization fraction.

    Args:
        mode (ModeTupleType): a ModeTupleType instance
        tangential_axes (tuple[int, int]): indices of transverse E-field component axes.
        pol (Literal["te", "tm"]): "te" or "tm" determines which axis is 'E1'

    Returns:
        float: Polarization fraction between 0 and 1.
    """

    E_fields = [mode.Ex, mode.Ey, mode.Ez]
    E1 = E_fields[tangential_axes[0]]
    E2 = E_fields[tangential_axes[1]]

    if pol == "te":
        numerator = np.sum(np.abs(E1) ** 2)
    elif pol == "tm":
        numerator = np.sum(np.abs(E2) ** 2)
    else:
        raise ValueError(f"pol must be 'te' or 'tm', but got {pol}")

    denominator = np.sum(np.abs(E1) ** 2 + np.abs(E2) ** 2) + 1e-18
    return numerator / denominator


def sort_modes(
    modes: list[ModeTupleType],
    filter_pol: Literal["te", "tm"] | None,
    tangential_axes: tuple[int, int],
    target_neff: float | None = None,
) -> list[ModeTupleType]:
    """
    Sort modes by polarization.

    Args:
        modes (list[ModeTupleType]): list of modes.
        filter_pol (Literal["te", "tm"] | None): If not none, sort by polarization specificaton.
        tangential_axes (tuple[int, int]): indices of transverse E-field component axes.
        target_neff (float | None, optional): When given, order by increasing distance of
            ``Re(n_eff)`` from this value instead of by decreasing ``Re(n_eff)``, so that index 0 is
            the mode nearest the target. Defaults to None (the historical descending order).

    Returns:
        list[ModeTupleType]: sorted list of modes.
    """
    if target_neff is None:

        def key(mode: ModeTupleType) -> float:
            return -float(np.real(mode.neff))
    else:
        aim = float(target_neff)

        def key(mode: ModeTupleType) -> float:
            return abs(float(np.real(mode.neff)) - aim)

    if filter_pol is None:
        return sorted(modes, key=key)

    def is_matching(mode):
        frac = compute_mode_polarization_fraction(mode, tangential_axes, filter_pol)
        return frac >= 0.5

    matching = [m for m in modes if is_matching(m)]
    non_matching = [m for m in modes if not is_matching(m)]

    return sorted(matching, key=key) + sorted(non_matching, key=key)


class SpuriousModeVerdict(NamedTuple):
    """Why one mode was rejected by :func:`filter_spurious_modes`.

    Attributes:
        index: Position in the sorted list the mode was rejected from.
        neff: Its complex effective index.
        reason: ``"index_above_material"`` or ``"energy_at_the_walls"``.
        detail: The number the rejection was made on - the material index bound, or the fraction of
            the electric energy sitting in the one-cell ring against the walls.
    """

    index: int
    neff: complex
    reason: str
    detail: float


def wall_energy_fraction(mode: ModeTupleType) -> float:
    """Fraction of the mode's electric energy in the one-cell ring against the outer walls.

    The finite-difference operator closes both transverse axes with electric walls, and it admits
    solutions that live almost entirely on those wall cells. They are artefacts of the
    discretization, not modes of the structure, and this is what tells them apart: a guided mode
    decays exponentially toward the wall and a box mode is spread over the whole cross-section, so
    both leave only a small fraction here, while a wall artefact leaves nearly all of it.

    Args:
        mode (ModeTupleType): One solved mode.

    Returns:
        float: The fraction, in ``[0, 1]``. Zero when the cross-section is too small to have a ring
        (a collapsed 2-D axis).
    """
    energy = np.abs(mode.Ex) ** 2 + np.abs(mode.Ey) ** 2 + np.abs(mode.Ez) ** 2
    energy = np.asarray(energy)
    total = float(np.sum(energy))
    if total <= 0.0 or energy.ndim != 2 or min(energy.shape) < 3:
        return 0.0
    interior = float(np.sum(energy[1:-1, 1:-1]))
    return float((total - interior) / total)


def filter_spurious_modes(
    modes: list[ModeTupleType],
    max_material_index: float,
    index_tolerance: float = 1e-6,
    max_wall_energy_fraction: float = 0.5,
) -> tuple[list[ModeTupleType], list[SpuriousModeVerdict]]:
    """Drop the modes the discrete operator admits but the structure does not support.

    Two gates, both conservative enough that a physical mode is never the one that goes:

    1. ``Re(n_eff)`` above the largest material index in the cross-section. No mode of a
       source-free dielectric cross-section can propagate faster than its own densest material.
    2. More than ``max_wall_energy_fraction`` of the electric energy in the one-cell ring against
       the outer walls (see :func:`wall_energy_fraction`).

    The first gate has one honest exception: a **metal-clad or plasmonic** guide, where the cladding
    has ``Re(eps) < 0``, does support surface modes above every dielectric index in the picture. Pass
    a larger ``max_material_index`` there - :func:`compute_mode` derives it from ``sqrt(max |eps|)``
    rather than ``sqrt(max Re eps)`` as soon as any cell has a negative real permittivity, which is
    the loosest bound that is still a bound.

    Args:
        modes (list[ModeTupleType]): The sorted mode list.
        max_material_index (float): Largest refractive index present in the cross-section.
        index_tolerance (float): Relative slack on the index bound.
        max_wall_energy_fraction (float): Wall-energy fraction above which a mode is rejected.

    Returns:
        tuple[list[ModeTupleType], list[SpuriousModeVerdict]]: The kept modes, in order, and one
        verdict per rejected mode.
    """
    kept: list[ModeTupleType] = []
    dropped: list[SpuriousModeVerdict] = []
    bound = float(max_material_index) * (1.0 + index_tolerance)
    for position, mode in enumerate(modes):
        neff = complex(mode.neff)
        if neff.real > bound:
            dropped.append(SpuriousModeVerdict(position, neff, "index_above_material", bound))
            continue
        fraction = wall_energy_fraction(mode)
        if fraction > max_wall_energy_fraction:
            dropped.append(SpuriousModeVerdict(position, neff, "energy_at_the_walls", fraction))
            continue
        kept.append(mode)
    return kept, dropped


def _resolve_mode_backend(mode_backend: Literal["fdtdmex", "tidy3d"] | None) -> str:
    """Resolve the active mode backend from the argument, env var, or the module default."""
    resolved: str = (
        mode_backend if mode_backend is not None else os.environ.get("FDTDMEX_MODE_BACKEND", _DEFAULT_MODE_BACKEND)
    )
    if resolved not in ("fdtdmex", "tidy3d"):
        raise ValueError(f"mode_backend must be 'fdtdmex' or 'tidy3d', got {resolved!r}")
    return resolved


def _dispatch_mode_solver(mode_backend: str, **kwargs) -> List[ModeTupleType]:
    """Call the selected mode backend, auto-routing fdtdmex's deferred cases to Tidy3D if present.

    The fdtdmex backend raises :class:`NotImplementedError` for a tensorial permeability and for
    bends. When that happens we transparently fall back to the Tidy3D solver if it is installed;
    otherwise the original error is re-raised so the user gets an actionable message. The
    ``formulation`` argument is fdtdmex's alone and is dropped on the Tidy3D route, which is fully
    tensorial by construction.
    """
    formulation = kwargs.pop("formulation", "auto")
    if mode_backend == "tidy3d":
        return tidy3d_mode_computation_wrapper(**kwargs)

    from fdtdx.core.physics.mode_backend import fdtdmex_mode_computation_wrapper

    try:
        return fdtdmex_mode_computation_wrapper(formulation=formulation, **kwargs)
    except NotImplementedError:
        try:
            import tidy3d  # noqa: F401
        except ImportError:
            raise
        return tidy3d_mode_computation_wrapper(**kwargs)


def _collapsed_cross_section_shape(cross_shape: tuple[int, int]) -> tuple[int, int]:
    """Transverse shape the solver actually sees, after ``compute_mode``'s 2-D collapse.

    A transverse axis of exactly two cells is treated as an invariant (two-dimensional) direction
    and collapsed to a single cell before the solve; the mode is repeated back over the two cells
    afterwards. The rule has to be known outside the callback because it decides how many
    eigenpairs the operator has.

    Args:
        cross_shape (tuple[int, int]): The two transverse cell counts.

    Returns:
        tuple[int, int]: The cell counts the mode operator is built on.
    """
    if 2 not in cross_shape:
        return cross_shape
    collapsed_axis = cross_shape.index(2)
    out = list(cross_shape)
    out[collapsed_axis] = 1
    return (out[0], out[1])


def max_solvable_modes(cross_shape: tuple[int, int]) -> int:
    """Largest number of modes the finite-difference operator of this cross-section can return.

    The transverse-E operator is ``2 N x 2 N`` for ``N`` cells, and the shift-invert Arnoldi
    iteration needs at least two Krylov vectors beyond the ones it returns, so at most ``2 N - 2``
    eigenpairs come back from one solve.

    Args:
        cross_shape (tuple[int, int]): The two transverse cell counts (before any 2-D collapse).

    Returns:
        int: The cap on ``num_modes``.
    """
    nx, ny = _collapsed_cross_section_shape(cross_shape)
    return max(1, 2 * nx * ny - 2)


def _mode_arrays(
    *,
    frequency: float,
    inv_permittivities: jax.Array,
    inv_permeabilities: jax.Array | float,
    resolution: float | None,
    direction: Literal["+", "-"],
    selected: tuple[int, ...],
    num_solver_modes: int,
    filter_pol: Literal["te", "tm"] | None,
    dtype: jnp.dtype,
    bend_radius: float | None,
    bend_axis: int | None,
    symmetry: tuple[int, int],
    transverse_coords: Sequence[jax.Array] | None,
    mode_backend: Literal["fdtdmex", "tidy3d"] | None,
    target_neff: float | None,
    drop_spurious: bool = False,
    mode_formulation: Literal["auto", "transverse", "full"] = "auto",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Solve one cross-section and return the modes at the positions listed in ``selected``.

    The shared implementation of :func:`compute_mode` (one mode) and :func:`compute_modes` (several
    modes off the same solve). Every returned array carries a leading axis of length
    ``len(selected)``.

    Args:
        frequency (float): Operating frequency in Hz.
        inv_permittivities (jax.Array): Inverse relative permittivity, shape ``(1|3|9, nx, ny, nz)``.
        inv_permeabilities (jax.Array | float): Inverse relative permeability, array or scalar.
        resolution (float | None): Uniform grid spacing in metres, or None with ``transverse_coords``.
        direction (Literal["+", "-"]): Propagation direction.
        selected (tuple[int, ...]): Positions in the sorted mode list to return.
        num_solver_modes (int): How many modes to ask the backend for.
        filter_pol (Literal["te", "tm"] | None): Optional polarization filter.
        dtype (jnp.dtype): Float dtype of the simulation; fixes the complex output dtype.
        bend_radius (float | None): Waveguide bend radius in metres.
        bend_axis (int | None): Physical axis normal to the plane of the bend (Tidy3D's convention);
            the radius grows along the *other* transverse axis.
        symmetry (tuple[int, int]): Mirror condition at the min edge of each transverse axis.
        transverse_coords (Sequence[jax.Array] | None): Cell-edge coordinates in metres.
        mode_backend (Literal["fdtdmex", "tidy3d"] | None): Mode-solver backend.
        target_neff (float | None): Shift-invert target and, when given, the sort key.
        drop_spurious (bool): Remove non-physical modes from the sorted list before indexing.
        mode_formulation (Literal["auto", "transverse", "full"]): Which native mode operator to
            assemble; see :func:`compute_mode`.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: ``(E, H, n_eff)`` with shapes ``(M, 3, nx, ny, nz)``,
        ``(M, 3, nx, ny, nz)`` and ``(M,)`` for ``M = len(selected)``.
    """
    # Input validation
    if (
        not (inv_permittivities.ndim == 4 and inv_permittivities.shape[0] in [1, 3, 9])
        or sum(dim == 1 for dim in inv_permittivities.shape[1:]) != 1
    ):
        raise Exception(f"Invalid shape of inv_permittivities: {inv_permittivities.shape}")
    if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
        if (
            not (inv_permeabilities.ndim == 4 and inv_permeabilities.shape[0] in [1, 3, 9])
            or sum(dim == 1 for dim in inv_permeabilities.shape[1:]) != 1
        ):
            raise Exception(f"Invalid shape of inv_permeabilities: {inv_permeabilities.shape}")
    if (bend_radius is None) != (bend_axis is None):
        raise ValueError("bend_radius and bend_axis must both be set or both be None")

    backend = _resolve_mode_backend(mode_backend)
    np_complex_dtype = np.complex128 if dtype == jnp.float64 else np.complex64

    def mode_helper(permittivity, permeability, c0_um, c1_um):
        coords = [np.asarray(c0_um), np.asarray(c1_um)]

        # Implicitly detect 2D mode if any transverse dimension is exactly 2
        mode_2d = 2 in permittivity.shape[1:]  # permittivity.shape=(N_comp, dim1, dim2)

        if mode_2d:
            collapsed_axis = permittivity.shape[1:].index(2)
            sl = (slice(None), slice(None), [0]) if collapsed_axis == 1 else (slice(None), [0], slice(None))

            assert np.allclose(permittivity, permittivity[sl]), "Permittivity is not uniform across the collapsed axis!"
            assert len(coords[collapsed_axis]) == 3, (
                f"Assumption: Permittivity {permittivity.shape[1:]=}+1 matches ({coords[0].shape=}, {coords[1].shape})"
            )
            permittivity = permittivity[sl]

            if isinstance(permeability, np.ndarray) and permeability.ndim > 0:
                permeability = permeability[sl]

            # Adjust coordinates for the collapsed dimension
            coords[collapsed_axis] = coords[collapsed_axis][:2]

        collapsed_shape = permittivity.shape[1:]

        if bend_radius is not None:
            assert bend_axis is not None
            transverse_axes = get_transverse_axes(propagation_axis)
            tidy3d_bend_axis = transverse_axes.index(bend_axis)
            bend_radius_um = bend_radius / 1e-6
            plane_center = (float(0.5 * (coords[0][0] + coords[0][-1])), float(0.5 * (coords[1][0] + coords[1][-1])))
        else:
            tidy3d_bend_axis = None
            bend_radius_um = None
            plane_center = None

        if bend_radius_um is not None and backend == "fdtdmex" and permittivity.shape[0] in (1, 3):
            # The native backend has no bend of its own: the curvature is removed here, by the
            # conformal / transformation-optics map, and it then solves an ordinary straight guide.
            # A fully tensorial cross-section is left alone so that it still routes to Tidy3D.
            assert tidy3d_bend_axis is not None and plane_center is not None
            permittivity, permeability, coords = transform_cross_section(
                permittivity,
                permeability,
                coords,
                bend_radius=bend_radius_um,
                bend_axis=tidy3d_bend_axis,
                plane_center=plane_center,
            )
            permittivity = np.asarray(permittivity)
            permeability = np.asarray(permeability)
            coords = [np.asarray(coords[0]), np.asarray(coords[1])]
            bend_radius_um, tidy3d_bend_axis, plane_center = None, None, None

        modes = _dispatch_mode_solver(
            backend,
            frequency=frequency,
            permittivity_cross_section=permittivity,
            permeability_cross_section=permeability,
            coords=coords,
            direction=direction,
            num_modes=num_solver_modes,
            target_neff=target_neff,
            bend_radius=bend_radius_um,
            bend_axis=tidy3d_bend_axis,
            plane_center=plane_center,
            symmetry=symmetry,
            formulation=mode_formulation,
        )

        # sort modes by polarization
        # tidy3d assumes propagation in the z-direction. The tangential axes are therefore x and y.
        modes = sort_modes(modes, filter_pol, (0, 1), target_neff=target_neff)

        if drop_spurious:
            array = np.asarray(permittivity)
            diagonal = array[[0, 4, 8]] if array.shape[0] == 9 else array
            if np.min(np.real(diagonal)) < 0.0:
                # A metal in the cross-section supports surface modes above every dielectric index
                # present, so the real part is not a bound there; |eps| still is.
                bound = float(np.sqrt(np.max(np.abs(diagonal))))
            else:
                bound = float(np.sqrt(np.max(np.real(diagonal))))
            modes, rejected = filter_spurious_modes(modes, bound)
            if rejected:
                summary = ", ".join(f"{verdict.neff.real:.6g}" for verdict in rejected)
                logger.warning(
                    f"the mode solver dropped {len(rejected)} spurious mode(s) of the "
                    f"{collapsed_shape} cross-section (index bound {bound:.6g}); their n_eff: {summary}"
                )
                for verdict in rejected:
                    logger.debug(
                        f"  spurious mode at sorted position {verdict.index}: n_eff {verdict.neff:.6g}, "
                        f"{verdict.reason} ({verdict.detail:.4g})"
                    )

        if max(selected) >= len(modes):
            raise ValueError(
                f"mode index {max(selected)} was requested but the backend returned only "
                f"{len(modes)} modes for a {collapsed_shape} cross-section"
                + (" after the spurious-mode filter" if drop_spurious else "")
            )

        def rotate(mode: ModeTupleType) -> tuple[np.ndarray, np.ndarray]:
            if propagation_axis == 0:
                mode_E, mode_H = (
                    np.stack([mode.Ez, mode.Ex, mode.Ey], axis=0).astype(np_complex_dtype),
                    np.stack([mode.Hz, mode.Hx, mode.Hy], axis=0).astype(np_complex_dtype),
                )
            elif propagation_axis == 1:
                mode_E, mode_H = (
                    np.stack([mode.Ex, mode.Ez, mode.Ey], axis=0).astype(np_complex_dtype),
                    -np.stack([mode.Hx, mode.Hz, mode.Hy], axis=0).astype(np_complex_dtype),
                )
            elif propagation_axis == 2:
                mode_E, mode_H = (
                    np.stack([mode.Ex, mode.Ey, mode.Ez], axis=0).astype(np_complex_dtype),
                    np.stack([mode.Hx, mode.Hy, mode.Hz], axis=0).astype(np_complex_dtype),
                )
            else:
                raise Exception("This should never happen")

            if mode_2d:
                # Re-expand the collapsed dimension. Backends disagree on whether the collapsed
                # (length-one) axis survives their own reshape - the native backend keeps it, the
                # tidy3d wrapper squeezes it away - so pin the shape first and only then repeat.
                mode_E = mode_E.reshape(3, *collapsed_shape)
                mode_H = mode_H.reshape(3, *collapsed_shape)
                mode_E = np.repeat(mode_E, 2, axis=collapsed_axis + 1)
                mode_H = np.repeat(mode_H, 2, axis=collapsed_axis + 1)
            return mode_E, mode_H

        rotated = [rotate(modes[i]) for i in selected]
        mode_E = np.stack([e for e, _ in rotated], axis=0)
        mode_H = np.stack([h for _, h in rotated], axis=0)
        neff = np.stack([np.asarray(modes[i].neff) for i in selected], axis=0).astype(np_complex_dtype)
        return mode_E, mode_H, neff

    # compute input to tidy3d Mode solver
    if inv_permittivities.shape[0] == 9:
        eps = expand_to_3x3(inv_permittivities)
        # Invert the 3x3 matrix
        perm = (2, 3, 4, 0, 1)  # (3, 3, nx, ny, nz) -> (nx, ny, nz, 3, 3)
        inv_perm = (3, 4, 0, 1, 2)  # (nx, ny, nz, 3, 3) -> (3, 3, nx, ny, nz)
        permittivities = (
            jnp.linalg.inv(eps.transpose(perm)).transpose(inv_perm).reshape(9, *inv_permittivities.shape[1:])
        )
    else:
        permittivities = 1 / inv_permittivities
    other_axes = [a for a in range(1, 4) if permittivities.shape[a] != 1]
    propagation_axis = permittivities.shape[1:].index(1)
    if transverse_coords is None:
        if resolution is None:
            raise ValueError("resolution is required when transverse_coords is not provided")
        # Uniform grid: build concrete coordinate arrays in µm and pass as callback args.
        c0_um = jnp.asarray(np.arange(permittivities.shape[other_axes[0]] + 1) * resolution / 1e-6)
        c1_um = jnp.asarray(np.arange(permittivities.shape[other_axes[1]] + 1) * resolution / 1e-6)
        normalization_area_weights = None
    else:
        if len(transverse_coords) != 2:
            raise ValueError(
                f"transverse_coords must contain exactly two coordinate arrays, got {len(transverse_coords)}"
            )
        # Shape validation uses .shape which is always concrete, even for JAX tracers.
        expected_lengths = [permittivities.shape[dim] + 1 for dim in other_axes]
        for axis_idx, (coord, expected_length) in enumerate(zip(transverse_coords, expected_lengths, strict=True)):
            if coord.ndim != 1 or coord.shape[0] != expected_length:
                raise ValueError(
                    f"transverse_coords[{axis_idx}] must be 1D with length {expected_length}, got {coord.shape}"
                )
        # Convert to µm for tidy3d; keep as JAX arrays so jax.jit can trace through.
        c0_um = jnp.asarray(transverse_coords[0]) / 1e-6
        c1_um = jnp.asarray(transverse_coords[1]) / 1e-6
        # area_2d in m²: use jnp.diff so this works with traced JAX arrays.
        area_2d = (
            jnp.diff(jnp.asarray(transverse_coords[0]))[:, None] * jnp.diff(jnp.asarray(transverse_coords[1]))[None, :]
        ).astype(dtype)
        weight_shape = [1, 1, 1]
        weight_shape[other_axes[0] - 1] = area_2d.shape[0]
        weight_shape[other_axes[1] - 1] = area_2d.shape[1]
        normalization_area_weights = area_2d.reshape(weight_shape)
    permittivity_squeezed = jnp.take(
        permittivities,
        indices=0,
        axis=propagation_axis + 1,
    )

    # Rotate permittivity components to match tidy3d coordinate system
    # tidy3d assumes propagation along z, so we need to map physical axes to tidy3d axes:
    # - tidy3d x → first transverse axis
    # - tidy3d y → second transverse axis
    # - tidy3d z → propagation axis
    if propagation_axis == 0:
        # propagation along x: tidy3d (x,y,z) → physical (y,z,x)
        perm_idx = [1, 2, 0]
        perm_idx_full_anisotropy = [4, 5, 3, 7, 8, 6, 1, 2, 0]
    elif propagation_axis == 1:
        # propagation along y: tidy3d (x,y,z) → physical (x,z,y)
        perm_idx = [0, 2, 1]
        perm_idx_full_anisotropy = [0, 2, 1, 6, 8, 7, 3, 5, 4]
    else:  # propagation_axis == 2
        # propagation along z: tidy3d (x,y,z) → physical (x,y,z)
        perm_idx = [0, 1, 2]
        perm_idx_full_anisotropy = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # Only apply rotation if anisotropic (3 components)
    if permittivity_squeezed.shape[0] == 3:
        permittivity_squeezed = permittivity_squeezed[jnp.array(perm_idx), :, :]
    if permittivity_squeezed.shape[0] == 9:
        permittivity_squeezed = permittivity_squeezed[jnp.array(perm_idx_full_anisotropy), :, :]

    jnp_complex_dtype = jnp.complex128 if dtype == jnp.float64 else jnp.complex64
    n_selected = len(selected)
    result_shape_dtype = (
        jnp.zeros((n_selected, 3, *permittivity_squeezed.shape[1:]), dtype=jnp_complex_dtype),
        jnp.zeros((n_selected, 3, *permittivity_squeezed.shape[1:]), dtype=jnp_complex_dtype),
        jnp.zeros(shape=(n_selected,), dtype=jnp_complex_dtype),
    )

    if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0 and inv_permeabilities.shape[0] == 9:
        mu = expand_to_3x3(inv_permeabilities)
        # Invert the 3x3 matrix
        perm = (2, 3, 4, 0, 1)  # (3, 3, nx, ny, nz) -> (nx, ny, nz, 3, 3)
        inv_perm = (3, 4, 0, 1, 2)  # (nx, ny, nz, 3, 3) -> (3, 3, nx, ny, nz)
        permeabilities = (
            jnp.linalg.inv(mu.transpose(perm)).transpose(inv_perm).reshape(9, *inv_permeabilities.shape[1:])
        )
    else:
        permeabilities = 1 / inv_permeabilities
    if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
        permeability_squeezed = jnp.take(
            permeabilities,
            indices=0,
            axis=propagation_axis + 1,
        )
        # Apply same rotation to permeability if anisotropic
        if permeability_squeezed.shape[0] == 3:
            permeability_squeezed = permeability_squeezed[jnp.array(perm_idx), :, :]
        if permeability_squeezed.shape[0] == 9:
            permeability_squeezed = permeability_squeezed[jnp.array(perm_idx_full_anisotropy), :, :]
    else:  # float
        permeability_squeezed = permeabilities

    # pure callback to tidy3d is necessary to work in jitted environment.
    # c0_um and c1_um are passed as explicit args so JAX materialises them to
    # concrete numpy arrays before calling mode_helper, allowing np.asarray()
    # inside the callback without raising TracerArrayConversionError.
    mode_E_raw, mode_H_raw, eff_idx = jax.pure_callback(
        mode_helper,
        result_shape_dtype,
        jax.lax.stop_gradient(permittivity_squeezed),
        jax.lax.stop_gradient(permeability_squeezed),
        jax.lax.stop_gradient(c0_um),
        jax.lax.stop_gradient(c1_um),
    )
    mode_E = jnp.expand_dims(mode_E_raw, axis=propagation_axis + 2)
    mode_H = jnp.expand_dims(mode_H_raw, axis=propagation_axis + 2)

    # The solver returns H scaled by -1j/eta0; restore the standard H units expected downstream.
    mode_H = mode_H * eta0

    # Every mode carries its own arbitrary eigenvector amplitude, so normalize them one at a time.
    normalized = [
        normalize_by_poynting_flux(
            mode_E[i],
            mode_H[i],
            axis=propagation_axis,
            area_weights=normalization_area_weights,
        )
        for i in range(n_selected)
    ]
    mode_E_norm = jnp.stack([e for e, _ in normalized], axis=0)
    mode_H_norm = jnp.stack([h for _, h in normalized], axis=0)

    return mode_E_norm, mode_H_norm, eff_idx


def compute_mode(
    frequency: float,
    inv_permittivities: jax.Array,  # shape (nx, ny, nz)
    inv_permeabilities: jax.Array | float,
    resolution: float | None = None,
    direction: Literal["+", "-"] = "+",
    mode_index: int = 0,
    filter_pol: Literal["te", "tm"] | None = None,
    dtype: jnp.dtype = jnp.float32,
    bend_radius: float | None = None,
    bend_axis: int | None = None,
    symmetry: tuple[int, int] = (0, 0),
    transverse_coords: Sequence[jax.Array] | None = None,
    mode_backend: Literal["fdtdmex", "tidy3d"] | None = None,
    target_neff: float | None = None,
    drop_spurious: bool = False,
    mode_formulation: Literal["auto", "transverse", "full"] = "auto",
) -> tuple[
    jax.Array,  # E
    jax.Array,  # H
    jax.Array,  # complex propagation constant
]:
    """Compute optical modes of a waveguide cross-section.

    By default modes are sorted by their effective index. The mode_index argument indexes this sorted list of modes and
    returns the desired mode. With filter_pol, it is also possible to only index a specific polarization.

    Args:
        frequency (float): Operating frequency in Hz
        inv_permittivities (jax.Array): 3D array of inverse relative permittivity values
        inv_permeabilities (jax.Array | float): 3D array of inverse relative permittivity values or single float for
            uniform permeability distribution.
        resolution (float | None): Uniform-grid spacing in metres. Required when ``transverse_coords`` is not
            provided (uniform-grid path). Ignored when ``transverse_coords`` is given. Defaults to None.
        direction (Literal["+", "-"]): Propagation direction, either "+" or "-".
        mode_index (int, optional): Index of the mode to compute. Defaults to 0.
        filter_pol (Literal["te", "tm"] | None, optional). If not None, modes are filtered by polarization.
        dtype (jnp.dtype, optional): Float dtype of the simulation. Controls whether mode fields are returned
            as complex64 (float32) or complex128 (float64). Defaults to jnp.float32. The solve itself is
            always done in double precision.
        bend_radius (float | None, optional): Bend radius of the waveguide in meters. Must be set together with
            bend_axis. When set, the mode solver uses a conformal transformation to account for the bend. Defaults to
            None (straight waveguide).
        bend_axis (int | None, optional): Physical axis index (0/1/2) **normal to the plane in which the bend
            lies** - a ring in the xy-plane bends about z. The radial direction is then the *other* transverse
            axis, and the sign of ``bend_radius`` says which way the radius grows along it. This is Tidy3D's
            ``ModeSpec.bend_axis`` convention, which both backends follow. Must differ from the propagation axis.
            Required when bend_radius is set. Defaults to None.
        symmetry (tuple[int, int], optional): Symmetry-plane condition at the *min* edge of each transverse axis,
            in the order of the two non-propagation physical axes (increasing index). ``0`` imposes a PEC mirror
            (electric wall — the tidy3d default), ``1`` imposes a PMC mirror (magnetic wall). Use this when the
            waveguide sits on a symmetry plane of a reduced (half/quarter) domain so the mode solver reproduces the
            same boundary the FDTD uses there. For a +x-propagating TE mode on a y/z quarter domain with PEC at y=0
            and PMC at the z Si-mid plane, pass ``(0, 1)``. Defaults to ``(0, 0)`` (PEC on both, i.e. no symmetry).
        transverse_coords: Optional pair of physical edge-coordinate arrays, in metres, for the two axes transverse
            to propagation. Each array must have one more entry than the corresponding transverse cell count.
            When provided, the mode solver receives the non-uniform rectilinear grid directly.
            JAX arrays are accepted; the numpy conversion happens inside the callback so the function
            remains compatible with ``jax.jit``.
        mode_backend (Literal["fdtdmex", "tidy3d"] | None, optional): Mode-solver backend. Defaults to None
            (environment variable ``FDTDMEX_MODE_BACKEND``, else the package default).
        target_neff (float | None, optional): Effective index to aim the solve at. It is the shift-invert
            target of the eigensolver, so the returned modes are the ones nearest it rather than the ones
            of highest index, and the sorted list is then ordered by increasing distance from it - i.e.
            ``mode_index=0`` selects the mode *nearest* the target. Without it the shift is guessed from the
            largest real permittivity in the cross-section, which is wrong for a metal-clad or plasmonic
            guide. Defaults to None (guess the shift, sort by decreasing ``Re(n_eff)``).
        drop_spurious (bool, optional): Remove non-physical modes - effective index above the
            largest material index, or the electric energy piled up against the outer walls - from
            the sorted list before ``mode_index`` selects from it, reporting each one it drops.
            See :func:`filter_spurious_modes`. Off by default because dropping a mode renumbers
            the list every caller indexes into. Defaults to False.
        mode_formulation (Literal["auto", "transverse", "full"], optional): Which mode operator the
            native backend assembles. ``"auto"`` takes the four-component (``4N``) one exactly when
            the cross-section carries a non-zero longitudinal off-diagonal permittivity entry
            (``eps_xz`` / ``eps_zx`` / ``eps_yz`` / ``eps_zy``) and the transverse-E (``2N``) one
            otherwise. ``"transverse"`` forces the cheaper operator and drops those entries with a
            warning; ``"full"`` forces the larger one. Ignored by the Tidy3D backend. Defaults to
            ``"auto"``.

    Returns:
        Tuple[jax.Array, jax.Array, jax.Array]:
            Tuple of E, H field and the effective index as complex-valued jax arrays.
    """
    mode_E, mode_H, eff_idx = _mode_arrays(
        frequency=frequency,
        inv_permittivities=inv_permittivities,
        inv_permeabilities=inv_permeabilities,
        resolution=resolution,
        direction=direction,
        selected=(mode_index,),
        num_solver_modes=2 * (mode_index + 1) + 10,
        filter_pol=filter_pol,
        dtype=dtype,
        bend_radius=bend_radius,
        bend_axis=bend_axis,
        symmetry=symmetry,
        transverse_coords=transverse_coords,
        mode_backend=mode_backend,
        target_neff=target_neff,
        drop_spurious=drop_spurious,
        mode_formulation=mode_formulation,
    )
    return mode_E[0], mode_H[0], eff_idx[0]


def compute_modes(
    frequency: float,
    inv_permittivities: jax.Array,
    inv_permeabilities: jax.Array | float,
    num_modes: int,
    resolution: float | None = None,
    direction: Literal["+", "-"] = "+",
    filter_pol: Literal["te", "tm"] | None = None,
    dtype: jnp.dtype = jnp.float32,
    bend_radius: float | None = None,
    bend_axis: int | None = None,
    symmetry: tuple[int, int] = (0, 0),
    transverse_coords: Sequence[jax.Array] | None = None,
    mode_backend: Literal["fdtdmex", "tidy3d"] | None = None,
    target_neff: float | None = None,
    drop_spurious: bool = False,
    mode_formulation: Literal["auto", "transverse", "full"] = "auto",
) -> tuple[
    jax.Array,  # E, shape (num_modes, 3, nx, ny, nz)
    jax.Array,  # H, same shape
    jax.Array,  # complex effective indices, shape (num_modes,)
]:
    """Return the first ``num_modes`` modes of a cross-section from a *single* solve.

    :func:`compute_mode` asks the backend for a dozen modes and hands back one, so a caller that
    needs a short list of candidates - mode tracking across a parameter step, a polarization sweep,
    a degeneracy check - pays one eigen-solve per candidate for a list the solver already had. This
    returns the whole sorted list off one solve. The ordering, the polarization filter, the 2-D
    collapse, the eta0 scaling and the per-mode Poynting normalization are the same as
    ``compute_mode``'s, so ``compute_modes(...)[k]`` is ``compute_mode(mode_index=k)`` up to the
    Arnoldi iteration seeing a slightly different Krylov space.

    Args:
        frequency (float): Operating frequency in Hz.
        inv_permittivities (jax.Array): Inverse relative permittivity, shape ``(1|3|9, nx, ny, nz)``.
        inv_permeabilities (jax.Array | float): Inverse relative permeability, array or scalar.
        num_modes (int): How many modes to return, counted from the front of the sorted list.
        resolution (float | None, optional): Uniform grid spacing in metres. Defaults to None.
        direction (Literal["+", "-"], optional): Propagation direction. Defaults to ``"+"``.
        filter_pol (Literal["te", "tm"] | None, optional): Polarization filter. Defaults to None.
        dtype (jnp.dtype, optional): Float dtype of the simulation. Defaults to jnp.float32.
        bend_radius (float | None, optional): Waveguide bend radius in metres. Defaults to None.
        bend_axis (int | None, optional): Physical axis normal to the plane of the bend. Defaults to None.
        symmetry (tuple[int, int], optional): Min-edge mirror condition. Defaults to ``(0, 0)``.
        transverse_coords (Sequence[jax.Array] | None, optional): Cell-edge coordinates in metres.
            Defaults to None.
        mode_backend (Literal["fdtdmex", "tidy3d"] | None, optional): Backend. Defaults to None.
        target_neff (float | None, optional): Shift-invert target; see :func:`compute_mode`.
            Defaults to None.
        drop_spurious (bool, optional): Remove non-physical modes before slicing the list; see
            :func:`compute_mode`. Defaults to False.
        mode_formulation (Literal["auto", "transverse", "full"], optional): Which native mode
            operator to assemble; see :func:`compute_mode`. Defaults to ``"auto"``.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: ``(E, H, n_eff)`` with a leading axis of length
        ``num_modes``, sorted the same way ``compute_mode`` sorts.

    Raises:
        ValueError: If ``num_modes`` is not positive, or exceeds what the discrete operator of this
            cross-section can supply (``2 N - 2`` for ``N`` transverse cells).
    """
    if num_modes < 1:
        raise ValueError(f"num_modes must be at least 1, got {num_modes}")
    spatial = tuple(inv_permittivities.shape[1:])
    if len(spatial) != 3 or sum(dim == 1 for dim in spatial) != 1:
        raise Exception(f"Invalid shape of inv_permittivities: {inv_permittivities.shape}")
    propagation_axis = spatial.index(1)
    cross_shape = tuple(dim for axis, dim in enumerate(spatial) if axis != propagation_axis)
    available = max_solvable_modes((cross_shape[0], cross_shape[1]))
    if num_modes > available:
        raise ValueError(
            f"num_modes={num_modes} exceeds the {available} modes a {cross_shape} cross-section "
            "can supply; the transverse-E operator has 2 N degrees of freedom and the Arnoldi "
            "iteration keeps two of them."
        )
    # Match the padding compute_mode(mode_index=num_modes - 1) would have used, so the sorted list
    # this returns is the one that caller would have seen.
    num_solver_modes = min(2 * num_modes + 10, available)
    return _mode_arrays(
        frequency=frequency,
        inv_permittivities=inv_permittivities,
        inv_permeabilities=inv_permeabilities,
        resolution=resolution,
        direction=direction,
        selected=tuple(range(num_modes)),
        num_solver_modes=num_solver_modes,
        filter_pol=filter_pol,
        dtype=dtype,
        bend_radius=bend_radius,
        bend_axis=bend_axis,
        symmetry=symmetry,
        transverse_coords=transverse_coords,
        mode_backend=mode_backend,
        target_neff=target_neff,
        drop_spurious=drop_spurious,
        mode_formulation=mode_formulation,
    )


def _cross_section_for_backend(
    inv_permittivities: jax.Array,
    inv_permeabilities: jax.Array | float,
    resolution: float | None,
    transverse_coords: Sequence[jax.Array] | None,
    bend_radius: float | None,
    bend_axis: int | None,
):
    """Turn the front end's arrays into what the native mode backend takes.

    The same preparation ``_mode_arrays`` does inside its callback - invert, drop the propagation
    axis, rotate the components into the backend's (transverse, transverse, propagation) order,
    collapse an invariant two-cell axis and remove a bend - but in one place and without the
    ``pure_callback``, so the differentiable path can use it. Everything stays in JAX, so a
    permittivity gradient survives.

    Args:
        inv_permittivities (jax.Array): Inverse relative permittivity, shape ``(1|3, nx, ny, nz)``.
        inv_permeabilities (jax.Array | float): Inverse relative permeability.
        resolution (float | None): Uniform grid spacing in metres, or None with transverse_coords.
        transverse_coords (Sequence[jax.Array] | None): Cell-edge coordinates in metres.
        bend_radius (float | None): Signed bend radius in metres.
        bend_axis (int | None): Physical axis normal to the plane of the bend.

    Returns:
        tuple: ``(permittivity, permeability, coords_m, propagation_axis)`` with the two materials of
        shape ``(1|3, Nx, Ny)`` and ``coords_m`` the two cell-edge arrays in metres.

    Raises:
        NotImplementedError: On a fully tensorial (9-component) cross-section.
        ValueError: On an invalid shape or a missing resolution.
    """
    if inv_permittivities.ndim != 4 or inv_permittivities.shape[0] not in (1, 3):
        raise NotImplementedError(
            f"the native mode backend's differentiable path takes isotropic or diagonally "
            f"anisotropic media, got a cross-section of shape {inv_permittivities.shape}"
        )
    if sum(dim == 1 for dim in inv_permittivities.shape[1:]) != 1:
        raise ValueError(f"Invalid shape of inv_permittivities: {inv_permittivities.shape}")
    if (bend_radius is None) != (bend_axis is None):
        raise ValueError("bend_radius and bend_axis must both be set or both be None")

    permittivities = 1.0 / inv_permittivities
    propagation_axis = permittivities.shape[1:].index(1)
    other_axes = [a for a in range(1, 4) if permittivities.shape[a] != 1]
    permittivity = jnp.take(permittivities, indices=0, axis=propagation_axis + 1)
    if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
        permeability = jnp.take(1.0 / inv_permeabilities, indices=0, axis=propagation_axis + 1)
    else:
        permeability = jnp.asarray(1.0 / inv_permeabilities, dtype=jnp.complex128).reshape(1, 1, 1)

    # Rotate the components into the backend's convention (propagation last), as _mode_arrays does.
    order = {0: [1, 2, 0], 1: [0, 2, 1], 2: [0, 1, 2]}[propagation_axis]
    if permittivity.shape[0] == 3:
        permittivity = permittivity[jnp.array(order), :, :]
    if permeability.shape[0] == 3:
        permeability = permeability[jnp.array(order), :, :]

    if transverse_coords is None:
        if resolution is None:
            raise ValueError("resolution is required when transverse_coords is not provided")
        coords_m = [np.arange(permittivities.shape[axis] + 1) * resolution for axis in other_axes]
    else:
        coords_m = [np.asarray(coord, dtype=np.float64) for coord in transverse_coords]

    # A transverse axis of exactly two cells is an invariant direction: solve one cell of it.
    cross_shape = permittivity.shape[1:]
    if 2 in cross_shape:
        collapsed = cross_shape.index(2)
        permittivity = permittivity[:, :, :1] if collapsed == 1 else permittivity[:, :1, :]
        if permeability.ndim == 3 and permeability.shape[1:] == cross_shape:
            permeability = permeability[:, :, :1] if collapsed == 1 else permeability[:, :1, :]
        coords_m[collapsed] = coords_m[collapsed][:2]

    if bend_radius is not None:
        assert bend_axis is not None
        transverse_axes = get_transverse_axes(propagation_axis)
        plane_center = tuple(float(0.5 * (coord[0] + coord[-1])) for coord in coords_m)
        permittivity, permeability, coords_um = transform_cross_section(
            permittivity,
            permeability,
            [coord / 1e-6 for coord in coords_m],
            bend_radius=bend_radius / 1e-6,
            bend_axis=transverse_axes.index(bend_axis),
            plane_center=tuple(value / 1e-6 for value in plane_center),
        )
        coords_m = [np.asarray(coord) * 1e-6 for coord in coords_um]

    return permittivity, permeability, coords_m, propagation_axis


def group_index(
    frequency: float,
    inv_permittivities: jax.Array,
    inv_permeabilities: jax.Array | float,
    resolution: float | None = None,
    mode_index: int = 0,
    filter_pol: Literal["te", "tm"] | None = None,
    bend_radius: float | None = None,
    bend_axis: int | None = None,
    transverse_coords: Sequence[jax.Array] | None = None,
    target_neff: float | None = None,
    num_modes: int | None = None,
    symmetry: tuple[int, int] = (0, 0),
) -> "ModeDispersion":
    """Effective index and group index of one mode, from a single eigen-solve.

    ``n_g = n_eff + omega d n_eff / d omega``, with the frequency derivative taken through the mode
    operator's own explicit dependence on ``k0`` rather than by solving at three frequencies and
    differencing. The operator is exactly ``D + S / k0**2``, so one assembly serves every frequency
    and ``jax.grad`` turns the eigen-solve's existing backward into ``d lambda / d k0`` - one
    contraction, no second solve. See :mod:`fdtdx.core.physics.mode_backend.dispersion`.

    The material is used as handed in, i.e. non-dispersive at ``frequency``; a dispersive medium
    contributes an extra term the caller can chain on with the same machinery.

    Args:
        frequency (float): Operating frequency in Hz.
        inv_permittivities (jax.Array): Inverse relative permittivity, shape ``(1|3, nx, ny, nz)``
            with the propagation axis of length one.
        inv_permeabilities (jax.Array | float): Inverse relative permeability.
        resolution (float | None, optional): Uniform grid spacing in metres. Defaults to None.
        mode_index (int, optional): Which mode of the sorted list. Defaults to 0.
        filter_pol (Literal["te", "tm"] | None, optional): Polarization filter, same convention as
            :func:`compute_mode`. Defaults to None.
        bend_radius (float | None, optional): Signed bend radius in metres. Defaults to None.
        bend_axis (int | None, optional): Physical axis normal to the plane of the bend. Defaults
            to None.
        transverse_coords (Sequence[jax.Array] | None, optional): Cell-edge coordinates in metres,
            for a non-uniform grid. Defaults to None.
        target_neff (float | None, optional): Sort by distance from this index instead of by
            descending ``Re(n_eff)``. Defaults to None.
        num_modes (int | None, optional): How many eigenpairs to solve for before selecting.
            Defaults to ``2 (mode_index + 1) + 10``, the padding :func:`compute_mode` uses.
        symmetry (tuple[int, int], optional): Min-edge mirror condition per transverse axis. A
            magnetic wall (``1``) is refused: the backward has no exact left eigenvector there.

    Returns:
        ModeDispersion: ``neff``, ``group_index``, ``dneff_domega`` and the selected position.

    Raises:
        NotImplementedError: On a fully tensorial cross-section or a magnetic wall.
        ValueError: If ``jax_enable_x64`` is off - the whole differentiable path is double
            precision, and a complex64 operator moves ``n_eff`` at the 1e-7 level.
    """
    from fdtdx.core.physics.mode_backend.dispersion import mode_dispersion
    from fdtdx.core.physics.mode_backend.operator import build_derivative_matrices, primal_dual_steps

    permittivity, permeability, coords_m, _ = _cross_section_for_backend(
        inv_permittivities, inv_permeabilities, resolution, transverse_coords, bend_radius, bend_axis
    )
    nx, ny = permittivity.shape[1], permittivity.shape[2]

    def components(array):
        array = jnp.asarray(array, dtype=jnp.complex128)
        if array.shape[0] == 1 or array.size == 1:
            flat = jnp.broadcast_to(array.reshape(-1)[:1] if array.size == 1 else array[0], (nx, ny)).reshape(-1)
            return flat, flat, flat
        return tuple(array[i].reshape(-1) for i in range(3))

    eps_xx, eps_yy, eps_zz = components(permittivity)
    mu_xx, mu_yy, mu_zz = components(permeability)
    dmin_pmc = (symmetry[0] == 1, symmetry[1] == 1)
    der_mats = build_derivative_matrices(coords_m[0], coords_m[1], dmin_pmc=dmin_pmc)
    cell_steps = (primal_dual_steps(coords_m[0]), primal_dual_steps(coords_m[1]))
    if target_neff is None:
        neff_guess = float(np.sqrt(np.max(np.real(np.asarray(jax.lax.stop_gradient(eps_xx)))))) * (1.0 + 1e-6) + 1e-6
    else:
        neff_guess = float(target_neff)
    return mode_dispersion(
        eps_xx,
        eps_yy,
        eps_zz,
        mu_xx,
        mu_yy,
        mu_zz,
        der_mats,
        cell_steps,
        frequency=frequency,
        num_modes=num_modes if num_modes is not None else 2 * (mode_index + 1) + 10,
        neff_guess=neff_guess,
        mode_index=mode_index,
        target_neff=target_neff,
        filter_pol=filter_pol,
        dmin_pmc=dmin_pmc,
    )


def _check_parity_residual(residual: jax.Array, walls: dict[int, int], object_name: str) -> None:
    """Diagnose a parity projection that removed too much of the mode.

    Skipped when ``residual`` is a JAX tracer. A mode source or mode-overlap detector that overlaps a
    :class:`~fdtdx.Device` solves its mode inside :func:`fdtdx.apply_params`, which callers routinely
    trace (``jax.jit`` around an optimization step), and there the residual has no value yet:
    concretizing it would raise, and comparing it would fire on a tracer. The check cannot be hoisted
    out of the trace either — the residual depends on the device permittivity being traced over. It is
    setup guidance, not something the solve depends on, so an eager ``apply_params`` surfaces it; and
    ``place_objects`` already applies (hence checks) every mode object that does not overlap a device,
    which is the overwhelming majority.

    Args:
        residual (jax.Array): Fraction of the mode the projection removed.
        walls (dict[int, int]): Mirror axis to wall type, for the message.
        object_name (str): Name used in diagnostics.

    Raises:
        ValueError: If the projection removed (almost) the whole mode, i.e. the configured wall types
            are incompatible with the selected mode.
    """
    if is_jax_tracer(residual):
        return
    value = float(residual)
    wall_description = ", ".join(f"{'xyz'[axis]}={'PMC' if wall == 1 else 'PEC'}" for axis, wall in walls.items())
    if value > 0.9:
        raise ValueError(
            f"The mode selected for '{object_name}' has (almost) none of the symmetry imposed by the "
            f"walls on {wall_description}: projecting it onto the admissible parity removes "
            f"{value:.1%} of the mode. The wall types do not match this mode - flip the sign of "
            f"config.symmetry on those axes, pick a different mode_index/filter_pol, or run without "
            f"config.symmetry."
        )
    if value > 0.3:
        logger.warning(
            f"The mode of '{object_name}' is only approximately symmetric about the walls on "
            f"{wall_description}: the parity projection removed {value:.1%} of it. Check that the "
            f"structure is mirror-symmetric there and that the wall types match the mode. A coarsely "
            f"resolved cross-section alone can account for a residual of a few tens of percent - the "
            f"mode solver samples materials on its staggered grid, so its discrete mode is only "
            f"mirror-symmetric to first order in the cell size."
        )


def compute_mode_symmetry_reduced(
    *,
    mirrored_axes: tuple[int, ...],
    walls: dict[int, int],
    frequency: float,
    inv_permittivities: jax.Array,
    inv_permeabilities: jax.Array | float,
    resolution: float | None = None,
    direction: Literal["+", "-"] = "+",
    mode_index: int = 0,
    filter_pol: Literal["te", "tm"] | None = None,
    dtype: jnp.dtype = jnp.float32,
    bend_radius: float | None = None,
    bend_axis: int | None = None,
    transverse_coords: Sequence[jax.Array] | None = None,
    object_name: str = "mode object",
    mode_backend: Literal["fdtdmex", "tidy3d"] | None = None,
    target_neff: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Solve a mode on a symmetry-reduced cross-section by way of the full cross-section.

    A reduced simulation replaces the discarded half by the mirror image of the kept half, so the
    mode it supports is the full-domain mode restricted to the kept half. Rather than asking the
    mode solver for a *symmetric* solve on the reduced cross-section, this mirrors the reduced
    material arrays back to the full cross-section, solves there, and restricts the result. The
    detour matters: the mode solver interprets the permittivity arrays on its own staggered Yee
    grid, while FDTDX rasterizes materials per cell and hands the same array to every component, so
    a solve on the reduced cross-section is inconsistent with the full one at first order in the
    cell size (an ``neff`` error of several percent for a high-index waveguide). Going through the
    full cross-section reproduces exactly the mode the unreduced simulation would inject.

    The solved mode is then projected onto the parity subspace the walls admit (see
    :func:`~fdtdx.core.physics.symmetry.project_onto_parity`), which both removes the small
    non-symmetric residue of the discrete mode and detects a wall type that does not match the mode
    at all. Finally the fields are renormalized to unit Poynting flux through the *reduced* plane,
    keeping the convention that a mode source launches unit power through the plane it occupies.

    Both steps are limited by the same discretization the detour above avoids for ``neff``: because
    the solver samples materials on its staggered grid, the discrete mode of a mirror-symmetric
    cross-section is itself only symmetric to first order in the cell size, so its flux does not
    split exactly evenly between the two halves. Measured for a 400x200 nm Si waveguide at 1.55 um:
    0.446 / 0.554 along the solver's first transverse axis at 25 nm (0.473 / 0.527 at 12.5 nm), and
    0.499 / 0.501 along its second. Renormalizing over the reduced plane therefore leaves the
    returned profile up to ~6% (25 nm) resp. ~3% (12.5 nm) above ``sqrt(2**k)`` times the restriction
    of the full-domain mode on such an axis, and the parity projection removes a few tenths of a
    percent of its norm. Both vanish with refinement; the flux convention is exact by construction at
    every resolution.

    Bend-axis convention, settled 2026-09-09: ``bend_axis`` is the axis *normal* to the plane of the
    bend, as in Tidy3D's ``ModeSpec.bend_axis``, so the radius - and with it the refractive index the
    bend transform scales - grows along the *other* transverse axis, the radial one. A mirror plane
    normal to the radial axis is therefore the one the bend destroys, and a mirror plane normal to
    ``bend_axis`` itself survives the bend untouched. Until this date the guard below read
    ``bend_axis`` as the radial axis, so it refused the safe pairing and let the unsafe one through.

    Args:
        mirrored_axes (tuple[int, ...]): Physical axes clipped by a symmetry plane.
        walls (dict[int, int]): Mirror axis to wall type (``-1`` PEC, ``+1`` PMC).
        frequency (float): Operating frequency in Hz.
        inv_permittivities (jax.Array): Reduced inverse permittivity on the mode plane.
        inv_permeabilities (jax.Array | float): Reduced inverse permeability on the mode plane.
        resolution (float | None): Uniform grid spacing, required without ``transverse_coords``.
        direction (Literal["+", "-"]): Propagation direction.
        mode_index (int): Index into the sorted mode list.
        filter_pol (Literal["te", "tm"] | None): Optional polarization filter.
        dtype (jnp.dtype): Float dtype of the simulation.
        bend_radius (float | None): Waveguide bend radius, with ``bend_axis``.
        bend_axis (int | None): Physical axis normal to the plane of the bend (Tidy3D's convention);
            the radius grows along the *other* transverse axis, which is the one the bend transform
            makes asymmetric.
        transverse_coords (Sequence[jax.Array] | None): Reduced transverse edge coordinates, or
            None on a uniform grid.
        object_name (str): Name used in diagnostics.
        mode_backend (Literal["fdtdmex", "tidy3d"] | None, optional): Mode-solver backend forwarded to
            :func:`compute_mode`. Defaults to None (environment or fork default).
        target_neff (float | None, optional): Effective index to aim the solve at, forwarded to
            :func:`compute_mode`. Defaults to None.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: ``(E, H, effective_index)`` on the reduced
        cross-section.

    Raises:
        ValueError: If a symmetry plane mirrors the bend's radial axis (the transverse axis that is
            not ``bend_axis``), or if the parity projection removes almost the entire mode, which
            means the configured wall types are incompatible with the selected mode. The latter is
            only detectable where the residual is concrete, i.e. not inside ``jax.jit`` (see
            :func:`_check_parity_residual`).
    """
    propagation_axis = next(a for a in range(3) if inv_permittivities.shape[1:][a] == 1)
    if bend_radius is not None and bend_axis is not None:
        transverse_axes = get_transverse_axes(propagation_axis)
        if bend_axis in transverse_axes:
            radial_axis = transverse_axes[1 - transverse_axes.index(bend_axis)]
            if radial_axis in mirrored_axes:
                raise ValueError(
                    f"'{object_name}' bends in the plane normal to the {'xyz'[bend_axis]}-axis, so "
                    f"its radius grows along the {'xyz'[radial_axis]}-axis, and config.symmetry "
                    f"mirrors that same {'xyz'[radial_axis]}-axis. The bend is modelled by a "
                    f"conformal transformation that scales the refractive index across the radial "
                    f"axis, so the transformed cross-section is not mirror-symmetric about a plane "
                    f"normal to it: the mode has no definite parity there and the reduced "
                    f"simulation, which replaces the discarded half by the mirror of the kept half, "
                    f"cannot represent it. Drop config.symmetry on the {'xyz'[radial_axis]}-axis, or "
                    f"put the symmetry plane normal to the {'xyz'[bend_axis]}-axis - a bend leaves "
                    f"that one mirror-symmetric."
                )
    full_inv_permittivities = mirror_material_cross_section(inv_permittivities, mirrored_axes)
    if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
        full_inv_permeabilities: jax.Array | float = mirror_material_cross_section(inv_permeabilities, mirrored_axes)
    else:
        full_inv_permeabilities = inv_permeabilities

    full_transverse_coords = transverse_coords
    if transverse_coords is not None:
        transverse_axes = get_transverse_axes(propagation_axis)
        full_transverse_coords = [
            mirror_edge_coordinates(coords) if axis in mirrored_axes else coords
            for axis, coords in zip(transverse_axes, transverse_coords, strict=True)
        ]

    mode_E, mode_H, eff_index = compute_mode(
        frequency=frequency,
        inv_permittivities=full_inv_permittivities,
        inv_permeabilities=full_inv_permeabilities,
        resolution=resolution,
        direction=direction,
        mode_index=mode_index,
        filter_pol=filter_pol,
        dtype=dtype,
        bend_radius=bend_radius,
        bend_axis=bend_axis,
        symmetry=(0, 0),
        transverse_coords=full_transverse_coords,
        mode_backend=mode_backend,
        target_neff=target_neff,
    )

    mode_E, residual_E = project_onto_parity(mode_E, "E", walls)
    mode_H, residual_H = project_onto_parity(mode_H, "H", walls)
    _check_parity_residual(jnp.maximum(residual_E, residual_H), walls, object_name)

    mode_E = restrict_to_kept_half(mode_E, mirrored_axes)
    mode_H = restrict_to_kept_half(mode_H, mirrored_axes)

    area_weights = None
    if transverse_coords is not None:
        widths = [jnp.diff(jnp.asarray(coords)) for coords in transverse_coords]
        area_2d = widths[0][:, None] * widths[1][None, :]
        weight_shape = [1, 1, 1]
        for local_axis, axis in enumerate(get_transverse_axes(propagation_axis)):
            weight_shape[axis] = area_2d.shape[local_axis]
        area_weights = area_2d.reshape(weight_shape).astype(mode_E.real.dtype)
    mode_E, mode_H = normalize_by_poynting_flux(
        mode_E,
        mode_H,
        axis=propagation_axis,
        area_weights=area_weights,
    )
    return mode_E, mode_H, eff_index


def _is_reciprocal_tensor(components: Sequence[ArrayLike], tol: float = 1e-6) -> bool:
    """Check whether a material tensor given as 9 row-major components is symmetric (reciprocal)."""
    return bool(
        np.max(np.abs(np.asarray(components[1]) - np.asarray(components[3]))) <= tol
        and np.max(np.abs(np.asarray(components[2]) - np.asarray(components[6]))) <= tol
        and np.max(np.abs(np.asarray(components[5]) - np.asarray(components[7]))) <= tol
    )


def tidy3d_mode_computation_wrapper(
    frequency: float,
    permittivity_cross_section: ArrayLike,
    coords: List[np.ndarray],
    direction: Literal["+", "-"],
    permeability_cross_section: ArrayLike | float | None = None,
    target_neff: float | None = None,
    angle_theta: float = 0.0,
    angle_phi: float = 0.0,
    num_modes: int = 10,
    precision: Literal["single", "double"] = "double",
    bend_radius: float | None = None,
    bend_axis: int | None = None,
    plane_center: tuple[float, float] | None = None,
    symmetry: tuple[int, int] = (0, 0),
) -> List[ModeTupleType]:
    """Compute optical modes of a waveguide cross-section.

    This function uses the Tidy3D mode solver to compute the optical modes of a given
    waveguide cross-section defined by its permittivity distribution.

    Args:
        frequency (float): Operating frequency in Hz
        permittivity_cross_section (jax.Array): 2D array of relative permittivity values
        coords (List[np.ndarray]): List of coordinate arrays [x, y] defining the grid
        direction (Literal["+", "-"], optional): Propagation direction, either "+" or "-"
        permeability_cross_section (jax.Array | float | None, optional): 2D array of relative permeability values.
            Defauts to None.
        target_neff (float | None, optional): Target effective index to search around. Defaults to None.
        angle_theta (float, optional): Polar angle in radians. Defaults to 0.0.
        angle_phi (float, optional): Azimuthal angle in radians. Defaults to 0.0.
        num_modes (int, optional): Number of modes to compute. Defaults to 10.
        precision (Literal["single", "double"], optional): Numerical precision. Defaults to "double".
        bend_radius (float | None, optional): Bend radius in microns (tidy3d units). Defaults to None.
        bend_axis (int | None, optional): Axis index (0 or 1) of the center of curvature in tidy3d's transverse
            coordinate frame. Defaults to None.
        plane_center (tuple[float, float] | None, optional): Center of the mode plane in the same units as coords.
            Required by tidy3d when bend_radius is set. Defaults to None.
        symmetry (tuple[int, int], optional): Per-transverse-axis symmetry condition at the min edge, forwarded to
            the tidy3d mode solver. ``1`` imposes a PMC (magnetic) wall there; ``0`` (default) leaves the solver's
            PEC (electric) wall. Order matches ``coords``. Defaults to ``(0, 0)``.

    Notes:
        tidy3d assumes propagation in z-direction. The output fields should be handled accordingly.

    Returns:
        List[ModeTupleType]: List of computed modes sorted by decreasing real part of
            effective index. Each mode contains the field components and effective index.
    """
    try:
        from tidy3d.components.mode.solver import compute_modes as _compute_modes
    except ImportError as exc:  # pragma: no cover - exercised only without tidy3d installed
        raise ImportError(
            "mode_backend='tidy3d' requires the optional 'tidy3d' dependency. Install it with "
            "'pip install tidy3d' / 'uv add tidy3d', or use the default native backend "
            "(mode_backend='fdtdmex')."
        ) from exc

    # see https://docs.flexcompute.com/projects/tidy3d/en/latest/_autosummary/tidy3d.ModeSpec.html#tidy3d.ModeSpec
    mode_spec = SimpleNamespace(
        # Note that the filter_pol argument is not used here since it does not work from tidy3d
        num_modes=num_modes,
        target_neff=target_neff,
        num_pml=(0, 0),
        angle_theta=angle_theta,
        angle_phi=angle_phi,
        bend_radius=bend_radius,
        bend_axis=bend_axis,
        precision=precision,
        track_freq="central",
        group_index_step=False,
    )
    permittivity_cross_section = jnp.asarray(permittivity_cross_section)
    permittivity_cross_section = expand_to_3x3(permittivity_cross_section)
    permittivity_cross_section = permittivity_cross_section.reshape(9, *permittivity_cross_section.shape[2:])
    eps_cross = [
        permittivity_cross_section[0],
        permittivity_cross_section[1],
        permittivity_cross_section[2],
        permittivity_cross_section[3],
        permittivity_cross_section[4],
        permittivity_cross_section[5],
        permittivity_cross_section[6],
        permittivity_cross_section[7],
        permittivity_cross_section[8],
    ]
    mu_cross = None
    if permeability_cross_section is not None:
        permeability_cross_section = jnp.asarray(permeability_cross_section)
        permeability_cross_section = expand_to_3x3(permeability_cross_section)
        permeability_cross_section = permeability_cross_section.reshape(9, *permeability_cross_section.shape[2:])

        mu_cross = [
            permeability_cross_section[0],
            permeability_cross_section[1],
            permeability_cross_section[2],
            permeability_cross_section[3],
            permeability_cross_section[4],
            permeability_cross_section[5],
            permeability_cross_section[6],
            permeability_cross_section[7],
            permeability_cross_section[8],
        ]

    if direction == "-" and (
        angle_theta != 0.0
        or angle_phi != 0.0
        or not _is_reciprocal_tensor(eps_cross)
        or (mu_cross is not None and not _is_reciprocal_tensor(mu_cross))
    ):
        raise NotImplementedError(
            "Backward ('-') modes are derived from the forward solve via the reciprocity transformation, "
            "which requires symmetric material tensors and normal incidence."
        )

    # Always solve forward: tidy3d >= 2.9 normalizes backward modes with a 1/sqrt of their
    # negative self-flux, tainting them with a spurious global phase of +-i.
    EH, neffs, _ = _compute_modes(
        eps_cross=eps_cross,
        coords=coords,
        freq=frequency,
        precision=precision,
        mode_spec=mode_spec,
        direction="+",
        mu_cross=mu_cross,
        plane_center=plane_center,
        symmetry=symmetry,
    )
    ((Ex, Ey, Ez), (Hx, Hy, Hz)) = EH.squeeze()

    # Backward modes: reciprocity transformation (E_z -> -E_z, H_t -> -H_t) of the forward mode
    if direction == "-":
        Ez, Hx, Hy = -Ez, -Hx, -Hy

    if num_modes == 1:
        modes = [
            ModeTupleType(
                Ex=Ex,
                Ey=Ey,
                Ez=Ez,
                Hx=Hx,
                Hy=Hy,
                Hz=Hz,
                neff=float(neffs.real) + 1j * float(neffs.imag),
            )
            for _ in range(num_modes)
        ]
    else:
        modes = [
            ModeTupleType(
                Ex=Ex[..., i],
                Ey=Ey[..., i],
                Ez=Ez[..., i],
                Hx=Hx[..., i],
                Hy=Hy[..., i],
                Hz=Hz[..., i],
                neff=neffs[i],
            )
            for i in range(num_modes)
        ]
    return modes
