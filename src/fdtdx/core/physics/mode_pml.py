"""Alert when a mode cross-section reaches into the simulation's PML.

The mode solver has no PML of its own: it closes both transverse axes with electric walls and
solves a lossless, bounded problem. A mode plane that reaches into the simulation's absorbing layer
is therefore solved on a cross-section that does not exist - the loader extends the surrounding
materials into the PML region, so the solver sees plain dielectric where the FDTD run will apply
complex coordinate stretching.

How bad that is depends on the permittivity tier, and the two cases are handled differently:

* **diagonal tier** (1 or 3 components). The extended material is a meaningful continuation of the
  structure, so the solved mode is a usable approximation of the guided mode - its tail is confined
  by a PEC wall instead of being absorbed, which shifts ``n_eff`` by however much of the mode sits
  out there. This raises a loud warning naming the overlap in cells.
* **tensorial tier** (9 components with real off-diagonal entries). There is no meaningful
  continuation: an off-diagonal tensor inside a stretched-coordinate region is not the same object
  as the same tensor in free space, and the native backend refuses tensorial media anyway. This
  raises :class:`ValueError`.

The check is a pure function of the mode plane's grid slice and the PML boxes, so it can be used
from a placement check, from a case script, or in a unit test without building a simulation.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Sequence

import numpy as np
from loguru import logger

__all__ = [
    "ModePlanePmlOverlapWarning",
    "check_mode_plane_pml_overlap",
    "cross_section_is_tensorial",
    "pml_boxes_from_boundary_config",
    "pml_overlap_cells",
]

#: Off-diagonal magnitude above which a 9-component cross-section counts as fully tensorial. Matches
#: :data:`fdtdx.core.physics.mode_backend.TOL_TENSORIAL`, which is what the backend itself refuses on.
TENSORIAL_TOLERANCE = 1e-6

_AXIS_NAMES = ("x", "y", "z")


class ModePlanePmlOverlapWarning(UserWarning):
    """A mode plane overlaps the simulation's PML and was solved as if it did not."""


def pml_boxes_from_boundary_config(
    boundary_config: Any,
    volume_grid_shape: tuple[int, int, int],
) -> list[tuple[str, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]]:
    """Grid boxes of the PML layers described by a :class:`~fdtdx.BoundaryConfig`.

    Args:
        boundary_config (Any): Anything exposing ``get_type_dict()`` and ``get_dict()`` with the
            six ``min_x`` ... ``max_z`` keys, i.e. a :class:`~fdtdx.BoundaryConfig`.
        volume_grid_shape (tuple[int, int, int]): Cell counts of the simulation volume.

    Returns:
        list[tuple[str, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]]: One
        ``(side name, grid slice tuple)`` entry per side whose boundary type is ``"pml"`` and whose
        thickness is positive.
    """
    types: Mapping[str, str] = boundary_config.get_type_dict()
    thicknesses: Mapping[str, int] = boundary_config.get_dict()
    boxes = []
    for axis, axis_name in enumerate(_AXIS_NAMES):
        extent = int(volume_grid_shape[axis])
        for side in ("min", "max"):
            key = f"{side}_{axis_name}"
            if str(types.get(key, "")).lower() != "pml":
                continue
            thickness = int(thicknesses.get(key, 0))
            if thickness <= 0:
                continue
            span = (0, min(thickness, extent)) if side == "min" else (max(0, extent - thickness), extent)
            full = [(0, int(volume_grid_shape[a])) for a in range(3)]
            full[axis] = span
            boxes.append((key, (full[0], full[1], full[2])))
    return boxes


def pml_overlap_cells(
    grid_slice_tuple: Sequence[tuple[int, int]],
    pml_boxes: Sequence[tuple[str, Sequence[tuple[int, int]]]],
) -> dict[str, int]:
    """Cells of the mode plane that lie inside each PML box.

    Args:
        grid_slice_tuple (Sequence[tuple[int, int]]): The mode plane's ``((lo, hi), ...)`` grid box.
        pml_boxes (Sequence[tuple[str, Sequence[tuple[int, int]]]]): ``(name, grid box)`` pairs, as
            returned by :func:`pml_boxes_from_boundary_config` or read off placed
            :class:`~fdtdx.PerfectlyMatchedLayer` objects.

    Returns:
        dict[str, int]: Number of overlapping cells per PML box, omitting boxes that do not
        intersect the plane at all.
    """
    overlaps: dict[str, int] = {}
    for name, box in pml_boxes:
        cells = 1
        for axis in range(3):
            low = max(int(grid_slice_tuple[axis][0]), int(box[axis][0]))
            high = min(int(grid_slice_tuple[axis][1]), int(box[axis][1]))
            cells *= max(0, high - low)
            if cells == 0:
                break
        if cells > 0:
            overlaps[name] = int(cells)
    return overlaps


def cross_section_is_tensorial(permittivity: Any, tol: float = TENSORIAL_TOLERANCE) -> bool:
    """Whether a permittivity array carries significant off-diagonal tensor components.

    Args:
        permittivity (Any): Array with a leading component axis of length 1, 3 or 9.
        tol (float): Off-diagonal magnitude above which the tier counts as tensorial.

    Returns:
        bool: ``True`` only for a 9-component array with an off-diagonal entry above ``tol``.
    """
    array = np.asarray(permittivity)
    if array.ndim == 0 or array.shape[0] != 9:
        return False
    tensor = array.reshape(3, 3, *array.shape[1:])
    off_diagonal = ~np.eye(3, dtype=bool)
    if tensor.size == 0:
        return False
    return bool(np.max(np.abs(tensor[off_diagonal])) > tol)


def check_mode_plane_pml_overlap(
    *,
    grid_slice_tuple: Sequence[tuple[int, int]],
    pml_boxes: Sequence[tuple[str, Sequence[tuple[int, int]]]],
    tensorial: bool,
    object_name: str = "mode object",
) -> dict[str, int]:
    """Raise or warn when a mode plane reaches into the PML.

    Args:
        grid_slice_tuple (Sequence[tuple[int, int]]): The mode plane's grid box.
        pml_boxes (Sequence[tuple[str, Sequence[tuple[int, int]]]]): ``(name, grid box)`` pairs.
        tensorial (bool): Whether the cross-section carries off-diagonal tensor components; see
            :func:`cross_section_is_tensorial`.
        object_name (str): Name used in the message.

    Returns:
        dict[str, int]: The overlap per PML box, empty when there is none.

    Raises:
        ValueError: If the plane overlaps a PML and the cross-section is fully tensorial.
    """
    overlaps = pml_overlap_cells(grid_slice_tuple, pml_boxes)
    if not overlaps:
        return {}
    detail = ", ".join(f"{name}: {cells} cells" for name, cells in sorted(overlaps.items()))
    if tensorial:
        raise ValueError(
            f"the mode plane of '{object_name}' overlaps the simulation's PML ({detail}) and its "
            "cross-section is fully tensorial. The mode solver has no PML: it closes the plane with "
            "electric walls and would solve a tensor material sitting where the FDTD run applies "
            "complex coordinate stretching, which is not the same object. Move the mode plane inside "
            "the PML-free region, enlarge the domain, or project the permittivity onto its diagonal "
            "first (which downgrades this to a warning)."
        )
    message = (
        f"the mode plane of '{object_name}' overlaps the simulation's PML ({detail}). The mode "
        "solver has no PML: it closes the cross-section with electric walls, so the part of the mode "
        "that reaches the absorbing layer is reflected instead of absorbed and n_eff is shifted by "
        "however much of the mode sits out there. The diagonal permittivity the loader extends into "
        "the PML region still makes the solve meaningful, which is why this is a warning and not an "
        "error - check that the mode is confined well inside the PML-free region."
    )
    warnings.warn(message, ModePlanePmlOverlapWarning, stacklevel=2)
    logger.warning(message)
    return overlaps
