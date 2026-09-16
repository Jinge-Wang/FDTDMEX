"""The loader's material arrays out onto named Cartesian lattices, for an external solver.

This is the inverse direction of :mod:`fdtdx.coupling.fem`: that module brings an external
field *in* at the Yee points, this one hands the assembled material *out* at the same points. It is
the piece a frequency-domain (FDFD) solver, a mode solver or a reference engine needs to be given
exactly the structure the fork's loader built — sub-pixel smoothing, interface blends and all —
rather than re-rasterising the same geometry a second time with its own conventions.

Nothing here interpolates. Each of the three E lattices is returned on its own Cartesian axes,
which is what the Yee grid already is; the caller gets the values and the cell edges and can rebuild
the lattice coordinates with :func:`fdtdx.coupling.lattice.lattice_coordinates`.

Per-slot convention, verified against Kronos ``waveEMFDFD`` (commit 1330c2a) in Phase 1 of the
coupling work and recorded here because it is the whole point of the export::

    fdtdx lattice   Yee position of the component      waveEMFDFD 2x-grid slot
    E0 (E_x)        ((i + 1/2) dx,  j dy,       k dz)  _eps_2x[1::2, 0::2, 0::2]   ERxx
    E1 (E_y)        ( i dx,        (j + 1/2) dy, k dz) _eps_2x[0::2, 1::2, 0::2]   ERyy
    E2 (E_z)        ( i dx,         j dy, (k + 1/2) dz) _eps_2x[0::2, 0::2, 1::2]  ERzz

``fdtdx``'s ``E_OFFSETS = ((.5, 0, 0), (0, .5, 0), (0, 0, .5))``
(:mod:`fdtdx.core.physics.geometry_raster`) is the same lattice as the ``ERxx/ERyy/ERzz`` slot
comment of ``fdfdSim.py:28-30``: no permutation and no half-cell shift, so slot ``c`` of the
external solver takes lattice ``E<c>`` unchanged. The one frame difference is the origin — fdtdx
centres its grid on the simulation volume, waveEMFDFD spans ``[0, S]`` — which is a translation of
the returned ``edges``, not a re-indexing of the values.

Two things the export deliberately does not do:

* It does not invert the off-diagonal tier. The loader stores the *inverse* permittivity, and the
  off-diagonal entries under ``include_offdiag=True`` are entries of ``eps^-1``, not of ``eps``;
  inverting a tensor entry by entry is wrong and the caller must never do it. They are returned so
  a consumer can either use them or measure what it is dropping — see :func:`offdiag_drop_ratio`.
* It does not fold the conductivity into the permittivity by itself. The loader's
  ``inv_permittivities`` is real and all loss lives in ``electric_conductivity``;
  :func:`complex_permittivity_slots` is the explicit one-line step that combines them at a stated
  angular frequency.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from fdtdx.constants import eps0 as EPS0
from fdtdx.coupling.lattice import grid_edges

#: The lattices this module exports the diagonal permittivity tier on.
E_LATTICES: tuple[str, str, str] = ("E0", "E1", "E2")

#: Order of the off-diagonal entries the loader keeps on the vertex lattice.
OFFDIAG_ENTRY_NAMES: tuple[str, str, str] = ("xy", "xz", "yz")


def _as_numpy(array: Any) -> np.ndarray:
    return np.asarray(np.array(array, copy=False), dtype=np.float64)


def _resolve_grid(grid: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Edges from a ``RectilinearGrid``, a ``SimulationConfig`` holding one, or three edge arrays."""
    if hasattr(grid, "grid") and not hasattr(grid, "edges"):
        grid = grid.grid
    return grid_edges(grid)


def yee_arrays_to_cartesian(
    arrays: Any,
    grid: Any,
    lattices: Sequence[str] = E_LATTICES,
    include_offdiag: bool = False,
) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The assembled permittivity per Yee lattice, on Cartesian axes, with the grid's cell edges.

    Args:
        arrays: The loader's array container (anything carrying ``inv_permittivities`` and,
            optionally, ``inv_permittivity_offdiag``), or a plain inverse-permittivity array. The
            array is ``(3, Nx, Ny, Nz)`` on the diagonal tier and ``(1, Nx, Ny, Nz)`` on the
            isotropic tier (``material_sampling="box"``), where the one component is repeated on
            all three lattices.
        grid: The resolved ``RectilinearGrid``, a ``SimulationConfig`` holding one, or a triple of
            cell-edge arrays, in metres.
        lattices (Sequence[str]): Which of ``"E0"``, ``"E1"``, ``"E2"`` to export. Any subset, in
            any order; the returned keys are exactly these names.
        include_offdiag (bool): Also return the vertex off-diagonal tier under the key ``"V"``.

    Returns:
        tuple: ``(values, edges)``.

        ``values[lattice]`` is ``(Nx, Ny, Nz)`` float64 **permittivity** (``1 / inv_permittivity``)
        at that lattice's points, in the loader's own index order, so ``values["E0"][i, j, k]`` sits
        at ``lattice_coordinates(edges, "E0")`` entry ``(i, j, k)``. With ``include_offdiag`` the extra
        key ``"V"`` is ``(3, Nx, Ny, Nz)`` and holds the ``(xy, xz, yz)`` entries of the **inverse**
        permittivity tensor on the cell vertices, exactly as the loader stored them.

        ``edges`` is the triple of cell-edge arrays the values refer to.

    Raises:
        ValueError: If a name outside ``"E0"``/``"E1"``/``"E2"`` is requested, if the arrays and
            the grid disagree on the shape, if a diagonal entry is not strictly positive, or if
            ``include_offdiag`` is asked for a scene the loader built without an off-diagonal tier.
    """
    edges = _resolve_grid(grid)
    inv = getattr(arrays, "inv_permittivities", arrays)
    inv = _as_numpy(inv)
    if inv.ndim != 4 or inv.shape[0] not in (1, 3):
        raise ValueError(f"inv_permittivities must have shape (1 or 3, Nx, Ny, Nz), got {inv.shape}")
    shape = tuple(int(e.size) - 1 for e in edges)
    if tuple(inv.shape[1:]) != shape:
        raise ValueError(f"the arrays are {tuple(inv.shape[1:])} cells but the grid edges say {shape}")
    if not np.all(inv > 0.0):
        raise ValueError("inv_permittivities has a non-positive entry; the export cannot invert it")

    values: dict[str, np.ndarray] = {}
    for lattice in lattices:
        if lattice not in E_LATTICES:
            raise ValueError(f"lattices must be a subset of {E_LATTICES}, got {lattice!r}")
        component = int(lattice[1]) if inv.shape[0] == 3 else 0
        values[lattice] = np.ascontiguousarray(1.0 / inv[component])

    if include_offdiag:
        offdiag = getattr(arrays, "inv_permittivity_offdiag", None)
        if offdiag is None:
            raise ValueError(
                "include_offdiag=True but the scene has no off-diagonal tier; it is written only by "
                "material_sampling='yee_smooth' with yee_smooth_offdiag_placement='node'"
            )
        offdiag = _as_numpy(offdiag)
        if offdiag.shape != (3, *shape):
            raise ValueError(f"inv_permittivity_offdiag must be {(3, *shape)}, got {offdiag.shape}")
        values["V"] = np.ascontiguousarray(offdiag)
    return values, edges


def offdiag_drop_ratio(arrays: Any) -> dict[str, float]:
    """How much a diagonal-only consumer loses by dropping the vertex off-diagonal tier.

    Dropping an off-diagonal entry ``c`` from a 2x2 block whose diagonal entries differ by ``d``
    moves the eigenvalue by ``c^2 / d``: second order while the block is well split, but first
    order (``a +/- c``) as ``d -> 0``. A nearly isotropic base — silica, silicon, and therefore any
    thermo-optic scene — sits near that degenerate limit, so the ratio, not the bare off-diagonal
    magnitude, is what a case has to gate on. The Phase 1b cross-review fixed the gate at 0.1,
    where the squared error passes 1 %.

    Args:
        arrays: The loader's array container.

    Returns:
        dict: ``max_abs_offdiag`` (largest ``|eps^-1_offdiag|``), ``max_diag_spread`` (largest
        per-cell spread of the three diagonal ``eps^-1`` entries), ``ratio`` (their quotient,
        ``inf`` when the spread is zero and the off-diagonals are not, ``0.0`` when both are), and
        ``num_nonzero`` (off-diagonal entries the loader actually wrote).
    """
    inv = _as_numpy(getattr(arrays, "inv_permittivities", arrays))
    offdiag = getattr(arrays, "inv_permittivity_offdiag", None)
    if offdiag is None:
        return {"max_abs_offdiag": 0.0, "max_diag_spread": 0.0, "ratio": 0.0, "num_nonzero": 0}
    offdiag = _as_numpy(offdiag)
    max_off = float(np.abs(offdiag).max()) if offdiag.size else 0.0
    spread = float((inv.max(axis=0) - inv.min(axis=0)).max()) if inv.size else 0.0
    if max_off == 0.0:
        ratio = 0.0
    elif spread == 0.0:
        ratio = float("inf")
    else:
        ratio = max_off / spread
    return {
        "max_abs_offdiag": max_off,
        "max_diag_spread": spread,
        "ratio": ratio,
        "num_nonzero": int(np.count_nonzero(offdiag)),
    }


def complex_permittivity_slots(
    values: Mapping[str, np.ndarray],
    conductivity: Any,
    omega: float,
    eps0: float = EPS0,
) -> dict[str, np.ndarray]:
    """Fold the loader's electric conductivity into the exported permittivity as its imaginary part.

    The loader's ``inv_permittivities`` is real; every loss channel it has is carried separately in
    ``electric_conductivity`` in S/m. A frequency-domain solver wants one complex number per slot,
    so under the ``exp(-i omega t)`` convention a positive imaginary part means loss::

        eps_c = eps_c + i sigma_c / (omega eps0)

    Args:
        values (Mapping[str, np.ndarray]): The ``values`` dictionary of
            :func:`yee_arrays_to_cartesian`. A ``"V"`` entry, if present, is passed through
            untouched: the off-diagonal tier has no conductivity counterpart.
        conductivity: ``(Nx, Ny, Nz)`` (isotropic, broadcast over the three lattices) or
            ``(3, Nx, Ny, Nz)`` (per lattice) in S/m, or ``None`` for a lossless scene.
        omega (float): Angular frequency in rad/s (or the run's nondimensional equivalent, as long
            as ``eps0`` uses the same system).
        eps0 (float): Vacuum permittivity; pass ``1.0`` for a nondimensional scene.

    Returns:
        dict: The same keys, complex128 for the E lattices, unchanged for ``"V"``.

    Raises:
        ValueError: If ``omega`` is not positive or the conductivity's shape fits neither form.
    """
    if omega <= 0.0:
        raise ValueError(f"omega must be positive, got {omega}")
    out: dict[str, np.ndarray] = {}
    if conductivity is None:
        for name, array in values.items():
            out[name] = np.asarray(array) if name == "V" else np.asarray(array, dtype=np.complex128)
        return out
    sigma = _as_numpy(conductivity)
    reference = next(iter(array for name, array in values.items() if name != "V"), None)
    if reference is None:
        raise ValueError("values holds no E lattice to fold a conductivity into")
    shape = tuple(reference.shape)
    if sigma.shape == shape:
        sigma = np.broadcast_to(sigma[None], (3, *shape))
    elif sigma.shape != (3, *shape):
        raise ValueError(f"electric_conductivity must be {shape} or {(3, *shape)}, got {sigma.shape}")
    for name, array in values.items():
        if name == "V":
            out[name] = np.asarray(array)
            continue
        out[name] = np.asarray(array, dtype=np.complex128) + 1j * sigma[int(name[1])] / (omega * eps0)
    return out
