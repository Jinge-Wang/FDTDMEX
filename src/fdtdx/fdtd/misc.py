from typing import Sequence

import jax
import jax.numpy as jnp

from fdtdx.fdtd.container import ArrayContainer
from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer


def collect_boundary_interfaces(
    arrays: ArrayContainer,
    pml_objects: Sequence[PerfectlyMatchedLayer],
    fields_to_collect: Sequence[str] = ("E", "H"),
) -> dict[str, jax.Array]:
    """Collects field values at PML boundary interfaces.

    Extracts field values at the interfaces between PML regions and the main simulation
    volume. This is used to enable time-reversible automatic differentiation by saving
    boundary values that would otherwise be lost due to PML absorption.

    Args:
        arrays (ArrayContainer): Container holding the field arrays (E, H fields)
        pml_objects (Sequence[PerfectlyMatchedLayer]): Sequence of PML objects defining boundary regions
        fields_to_collect (Sequence[str], optional): Which fields to collect values for (default: E and H fields)

    Returns:
        dict[str, jax.Array]: Dictionary mapping "{pml_name}_{field_str}" to array of interface field values
    """
    res = {}
    for field_str in fields_to_collect:
        arr: jax.Array = getattr(arrays.fields, field_str)
        for pml in pml_objects:
            cur_slice = arr[:, *pml.interface_slice()]
            res[f"{pml.name}_{field_str}"] = cur_slice
    return res


def add_boundary_interfaces(
    arrays: ArrayContainer,
    values: dict[str, jax.Array],
    pml_objects: Sequence[PerfectlyMatchedLayer],
    fields_to_add: Sequence[str] = ("E", "H"),
) -> ArrayContainer:
    """Adds saved field values back to PML boundary interfaces.

    Restores previously collected field values at the interfaces between PML regions
    and the main simulation volume. This is the inverse operation to collect_boundary_interfaces()
    and is used during time-reversed automatic differentiation.

    Args:
        arrays (ArrayContainer): Container holding the field arrays to update
        values (dict[str, jax.Array]): Dictionary of saved interface values from collect_boundary_interfaces()
        pml_objects (Sequence[PerfectlyMatchedLayer]): Sequence of PML objects defining boundary regions
        fields_to_add (Sequence[str], optional): Which fields to restore values for (default: E and H fields)

    Returns:
        ArrayContainer: Updated ArrayContainer with restored interface field values
    """
    for field_str in fields_to_add:
        arr: jax.Array = getattr(arrays.fields, field_str)
        for pml in pml_objects:
            val = values[f"{pml.name}_{field_str}"]
            arr = arr.at[:, *pml.interface_slice()].set(val)
        arrays = arrays.aset(f"fields->{field_str}", arr)

    return arrays


def compute_anisotropic_update_matrices(
    inv_material_prop: jax.Array,
    sigma: jax.Array | None,
    c: float,
    eta_factor: float,
) -> tuple[jax.Array, jax.Array]:
    """Computes the A and B matrices for anisotropic FDTD updates.

    Args:
        inv_material_prop (jax.Array): Inverse material property tensor (3, 3, Nx, Ny, Nz)
        sigma (jax.Array | None): Conductivity tensor (3, 3, Nx, Ny, Nz) or None
        c (float): Courant number
        eta_factor (float): eta0 for electric, 1/eta0 for magnetic

    Returns:
        tuple[jax.Array, jax.Array]: A and B matrices
    """

    M1 = jnp.eye(3)[:, :, None, None, None]
    M2 = jnp.eye(3)[:, :, None, None, None]
    if sigma is not None:
        factor = c * eta_factor / 2 * jnp.einsum("ijxyz,jkxyz->ikxyz", inv_material_prop, sigma)
        M1 += factor
        M2 -= factor
    perm = (2, 3, 4, 0, 1)  # (3, 3, Nx, Ny, Nz) -> (Nx, Ny, Nz, 3, 3)
    inv_perm = (3, 4, 0, 1, 2)  # (Nx, Ny, Nz, 3, 3) -> (3, 3, Nx, Ny, Nz)
    A = jnp.linalg.solve(M1.transpose(perm), M2.transpose(perm)).transpose(inv_perm)
    B = c * jnp.linalg.solve(M1.transpose(perm), inv_material_prop.transpose(perm)).transpose(inv_perm)

    return A, B


def compute_anisotropic_update_matrices_reverse(
    inv_material_prop: jax.Array,
    sigma: jax.Array | None,
    c: float,
    eta_factor: float,
) -> tuple[jax.Array, jax.Array]:
    """Computes the A and B matrices for reverse anisotropic FDTD updates.

    Args:
        inv_material_prop (jax.Array): Inverse material property tensor (3, 3, Nx, Ny, Nz)
        sigma (jax.Array | None): Conductivity tensor (3, 3, Nx, Ny, Nz) or None
        c (float): Courant number
        eta_factor (float): eta0 for electric, 1/eta0 for magnetic

    Returns:
        tuple[jax.Array, jax.Array]: A and B matrices
    """
    M1 = jnp.eye(3)[:, :, None, None, None]
    M2 = jnp.eye(3)[:, :, None, None, None]
    if sigma is not None:
        factor = c * eta_factor / 2 * jnp.einsum("ijxyz,jkxyz->ikxyz", inv_material_prop, sigma)
        M1 += factor
        M2 -= factor
    perm = (2, 3, 4, 0, 1)  # (3, 3, Nx, Ny, Nz) -> (Nx, Ny, Nz, 3, 3)
    inv_perm = (3, 4, 0, 1, 2)  # (Nx, Ny, Nz, 3, 3) -> (3, 3, Nx, Ny, Nz)
    A = jnp.linalg.solve(M2.transpose(perm), M1.transpose(perm)).transpose(inv_perm)
    B = c * jnp.linalg.solve(M2.transpose(perm), inv_material_prop.transpose(perm)).transpose(inv_perm)

    return A, B


def avg_anisotropic_E_component(
    field: jax.Array,
    component: int,
    location: int,
    aniso_widths: tuple[jax.Array, jax.Array, jax.Array] | None = None,
) -> jax.Array:
    """Averages an E field component onto another component's Yee location.

    On a uniform grid this is the four-point mean of the staggered samples. On a non-uniform
    grid the center-to-edge half-step along the component axis is weighted by the local cell
    widths; the location axis is already edge-aligned and keeps an unweighted midpoint.

    Args:
        field (jax.Array): E field to average (3, Nx, Ny, Nz)
        component (int): Component to average, 0 for Ex, 1 for Ey, 2 for Ez
        location (int): Location to calculate average, 0 for Ex, 1 for Ey, 2 for Ez
        aniso_widths (tuple | None): Per-axis padded cell widths from
            ``get_anisotropic_averaging_widths``, broadcast along their axis, or None on a
            uniform grid. None (the default) selects the unweighted four-point mean.

    Returns:
        jax.Array: Averaged E field component
    """

    samples = field[component]
    if aniso_widths is None:
        return (
            (
                samples
                + jnp.roll(samples, -1, axis=location)
                + jnp.roll(samples, 1, axis=component)
                + jnp.roll(samples, (-1, 1), axis=(location, component))
            )
            / 4
        )[1:-1, 1:-1, 1:-1]
    centered = 0.5 * (samples + jnp.roll(samples, -1, axis=location))
    width = aniso_widths[component]
    previous_width = jnp.roll(width, 1, axis=component)
    on_edge = (centered * previous_width + jnp.roll(centered, 1, axis=component) * width) / (width + previous_width)
    return on_edge[1:-1, 1:-1, 1:-1]


#: For each E row ``c``, the two ``(partner component, vertex entry index)`` pairs that couple into
#: it. The vertex array stores the symmetric tensor's independent off-diagonal entries in the order
#: ``(xy, xz, yz)``, so row x reads xy and xz, row y reads xy and yz, row z reads xz and yz — and
#: every entry is read by exactly the two rows it couples, from the same array. The partner's own
#: axis is its component index: ``D_y`` samples straddle a vertex along y, ``D_z`` along z.
OFFDIAG_ROW_PARTNERS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((1, 0), (2, 1)),
    ((0, 0), (2, 2)),
    ((0, 1), (1, 2)),
)


def _padded_window(array: jax.Array, offsets: tuple[int, int, int]) -> jax.Array:
    """The interior of a halo-padded array, shifted by whole cells on each spatial axis.

    ``array`` has its three trailing axes padded by one cell each way. Offset ``0`` returns the
    interior, ``+1`` the neighbour one cell up, ``-1`` the neighbour one cell down.
    """
    lead = array.ndim - 3
    index: list[slice] = [slice(None)] * lead
    for axis in range(3):
        extent = array.shape[lead + axis] - 2
        offset = offsets[axis]
        index.append(slice(1 + offset, 1 + offset + extent))
    return array[tuple(index)]


def _padded_width_window(widths: jax.Array, axis: int, offset: int, extent: int) -> jax.Array:
    """One axis's padded cell widths, shifted by whole cells, still broadcasting along that axis."""
    index: list[slice] = [slice(None)] * 3
    index[axis] = slice(1 + offset, 1 + offset + extent)
    return widths[tuple(index)]


def add_offdiag_correction(
    E: jax.Array,
    increment_pad: jax.Array,
    offdiag_pad: jax.Array,
    aniso_widths: tuple[jax.Array, jax.Array, jax.Array] | None = None,
) -> jax.Array:
    """Add the vertex-placed off-diagonal term to an otherwise diagonal E update.

    This is Meep's ``OFFDIAG`` stencil (Werner & Cary 2007), written out for all three rows. Let
    ``u_xy[i,j,k]`` be the ``xy`` entry of the smoothed inverse-permittivity tensor at the vertex
    ``(e_x[i], e_y[j], e_z[k])`` and ``K`` the quantity the diagonal update multiplies by
    ``inv_eps``. Then, on a uniform grid::

        E_x[i,j,k] += 0.25 * ( u_xy[i  ,j,k] * (K_y[i  ,j,k] + K_y[i  ,j-1,k])
                             + u_xy[i+1,j,k] * (K_y[i+1,j,k] + K_y[i+1,j-1,k]) )
                    + 0.25 * ( u_xz[i  ,j,k] * (K_z[i  ,j,k] + K_z[i  ,j,k-1])
                             + u_xz[i+1,j,k] * (K_z[i+1,j,k] + K_z[i+1,j,k-1]) )

    and the same with the indices rotated for the other two rows. Read it as: at each of the two
    vertices bracketing the ``E_x`` point along its own axis (weights exactly 1/2 and 1/2 on any
    grid, because the component point is the midpoint of its own two edges), average the two ``K_y``
    samples straddling that vertex along the partner's axis, multiply by the vertex's own entry, and
    average the two vertex products.

    On a non-uniform grid only the partner average is re-weighted: the two samples are neighbouring
    cell centres and the vertex is the edge between them, so the weight is the dual-cell weight
    ``(v[j] w[j-1] + v[j-1] w[j]) / (w[j-1] + w[j])`` that
    :func:`avg_anisotropic_E_component` already uses.

    Both coupled rows read one shared entry — ``E_x <- K_y`` and ``E_y <- K_x`` both take
    ``u_xy`` at the vertex they share — so the assembled map is its own transpose entry for entry.
    That is the whole point of the placement: with the row written at each component's own pixel
    instead, the two rows read different boxes and the map is 1-5% asymmetric.

    Args:
        E (jax.Array): ``(3, Nx, Ny, Nz)`` field after the elementwise diagonal update.
        increment_pad (jax.Array): ``(3, Nx+2, Ny+2, Nz+2)`` halo-padded ``K``, the quantity the
            diagonal branch multiplies by ``inv_eps`` (``c * curl`` plus the folded ADE polarization
            delta where there is one).
        offdiag_pad (jax.Array): ``(3, Nx+2, Ny+2, Nz+2)`` halo-padded vertex entries
            ``(xy, xz, yz)``.
        aniso_widths (tuple | None): Per-axis padded cell widths from
            :func:`fdtdx.fdtd.update.get_anisotropic_averaging_widths`, or None on a uniform grid.

    Returns:
        jax.Array: ``(3, Nx, Ny, Nz)`` field with the off-diagonal term added.
    """
    rows = []
    for component in range(3):
        row = None
        for partner, entry in OFFDIAG_ROW_PARTNERS[component]:
            for near in (0, 1):
                # The two vertices bracketing the E_c point along c's own axis: the cell's own two
                # edges, so the sample point is their exact midpoint on any grid and the weight is
                # 1/2 either way.
                offsets = [0, 0, 0]
                offsets[component] = near
                near_offsets = (offsets[0], offsets[1], offsets[2])
                vertex = _padded_window(offdiag_pad[entry], near_offsets)
                upper = _padded_window(increment_pad[partner], near_offsets)
                lower_offsets = list(offsets)
                lower_offsets[partner] -= 1
                lower = _padded_window(increment_pad[partner], (lower_offsets[0], lower_offsets[1], lower_offsets[2]))
                if aniso_widths is None:
                    straddling = 0.5 * (upper + lower)
                else:
                    extent = increment_pad.shape[1 + partner] - 2
                    width_upper = _padded_width_window(aniso_widths[partner], partner, offsets[partner], extent)
                    width_lower = _padded_width_window(aniso_widths[partner], partner, offsets[partner] - 1, extent)
                    straddling = (upper * width_lower + lower * width_upper) / (width_upper + width_lower)
                term = 0.5 * vertex * straddling
                row = term if row is None else row + term
        assert row is not None  # every row has two partners
        rows.append(row)
    return E + jnp.stack(rows, axis=0)


def avg_anisotropic_H_component(
    field: jax.Array,
    component: int,
    location: int,
    aniso_widths: tuple[jax.Array, jax.Array, jax.Array] | None = None,
) -> jax.Array:
    """Averages an H field component onto another component's Yee location.

    On a uniform grid this is the four-point mean of the staggered samples. On a non-uniform
    grid the center-to-edge half-step along the location axis is weighted by the local cell
    widths; the component axis is already edge-aligned and keeps an unweighted midpoint.

    Args:
        field (jax.Array): H field to average (3, Nx, Ny, Nz)
        component (int): Component to average, 0 for Hx, 1 for Hy, 2 for Hz
        location (int): Location to calculate average, 0 for Hx, 1 for Hy, 2 for Hz
        aniso_widths (tuple | None): Per-axis padded cell widths from
            ``get_anisotropic_averaging_widths``, broadcast along their axis, or None on a
            uniform grid. None (the default) selects the unweighted four-point mean.

    Returns:
        jax.Array: Averaged H field component
    """

    samples = field[component]
    if aniso_widths is None:
        return (
            (
                samples
                + jnp.roll(samples, 1, axis=location)
                + jnp.roll(samples, -1, axis=component)
                + jnp.roll(samples, (1, -1), axis=(location, component))
            )
            / 4
        )[1:-1, 1:-1, 1:-1]
    width = aniso_widths[location]
    previous_width = jnp.roll(width, 1, axis=location)
    on_edge = (samples * previous_width + jnp.roll(samples, 1, axis=location) * width) / (width + previous_width)
    centered = 0.5 * (on_edge + jnp.roll(on_edge, -1, axis=component))
    return centered[1:-1, 1:-1, 1:-1]
