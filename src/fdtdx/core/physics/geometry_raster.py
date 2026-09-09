"""Field-loaded (per-Yee-component) material sampling from continuous object shapes.

This module is the ``material_sampling="yee"`` half of the static-material assembly. Where the
default path rounds every object's box to whole cells, draws the shape inside that box on the cell
centres, and broadcasts the one resulting mask to all three field components, this path does the
opposite: it keeps every object's requested metric extent continuous (see
:mod:`fdtdx.fdtd.metric_shadow`) and asks, at each Yee component's own position, which object is in
front there.

Two facts fix the whole design.

**Where each component lives.** ``calculate_spatial_offsets_yee`` gives the half-offsets and
``calculate_time_offset_yee`` fixes what an offset means in metres (offset ``0`` is
``edges[:-1]``, offset ``0.5`` is the cell centres). For array cell ``(i, j, k)`` on a rectilinear
grid with edges ``e`` and centres ``c``:

======  ==========================
comp    metric position
======  ==========================
``Ex``  ``(c_x[i], e_y[j], e_z[k])``
``Ey``  ``(e_x[i], c_y[j], e_z[k])``
``Ez``  ``(e_x[i], e_y[j], c_z[k])``
``Hx``  ``(e_x[i], c_y[j], c_z[k])``
``Hy``  ``(c_x[i], e_y[j], c_z[k])``
``Hz``  ``(c_x[i], c_y[j], e_z[k])``
======  ==========================

The update multiplies component-wise at the same index, so ``inv_permittivities[c][i, j, k]`` is the
material that ``E_c`` sees at the position in that table, and likewise for ``H_c``.

**Who wins where objects overlap.** The default path writes static objects sorted by
``placement_order`` and lets each one overwrite the cells it covers — that is how carving works.
This path keeps the identical ordering and reads it as a priority: later in the write order is in
front. The simulation volume (``placement_order = -1000``) is written first and therefore acts as
the scene background.

Stage A is point sampling only. A cell straddling an interface takes whichever material contains its
component position; there is no fill fraction and no Kottke blend yet, so the geometry is exact but
the interface is still staircased. ``material_sampling="yee_smooth"`` adds the fill-fraction blend
on top, for the permittivity on the E lattices and for the permeability on the H lattices
(:mod:`fdtdx.core.physics.geometry_smooth`).

**Periodic axes.** When an axis carries a periodic or Bloch boundary every object is also evaluated
one lattice vector each way, so a shape crossing that face reappears on the other side. The object
is never translated: its bounding interval is shifted to window the lattice and the query points are
shifted back into the object's own frame. An object and its images share one entry and one priority,
so they cannot outrank each other and the ascending write order is unchanged. An axis the simulation
is invariant along is excluded, because fdtdx's 2-D convention is a single periodic cell there.
"""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from fdtdx.core.grid import RectilinearGrid
from fdtdx.materials import (
    Material,
    compute_allowed_dispersive_coefficients,
    compute_allowed_electric_conductivities,
    compute_allowed_magnetic_conductivities,
    compute_allowed_permeabilities,
    compute_allowed_permittivities,
    compute_ordered_names,
)
from fdtdx.objects.object import SimulationObject
from fdtdx.objects.static_material.static import (
    SimulationVolume,
    StaticMultiMaterialObject,
    UniformMaterialObject,
)

#: The two static-object base classes the scene can query (``contains`` / ``material_at`` /
#: ``normal_at``): what ``ObjectContainer.static_material_objects`` returns.
StaticObject = UniformMaterialObject | StaticMultiMaterialObject

#: Half-cell offsets of the three E components, in the order (x, y, z) per component.
E_OFFSETS: tuple[tuple[float, float, float], ...] = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5))
#: Half-cell offsets of the three H components, in the order (x, y, z) per component.
H_OFFSETS: tuple[tuple[float, float, float], ...] = ((0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))

#: Half-cell offsets of the cell-vertex lattice, where the off-diagonal Kottke entries live under
#: ``yee_smooth_offdiag_placement="node"``. It is **one** lattice, not three: the off-diagonal entry
#: of row ``c`` sits half a cell back along ``c``'s own axis from the ``E_c`` point, which for all
#: three components is the same primary-grid vertex ``(i, j, k)``. The three entries are repeated so
#: the tuple can be indexed by component like the other two.
V_OFFSETS: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),) * 3

#: Maximum number of lattice points evaluated in one ``contains`` call, to bound peak memory.
_CHUNK_POINTS = 4_000_000


def yee_lattice_coordinates(
    grid: RectilinearGrid,
    field: str,
    component: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the three 1-D metric coordinate arrays of one Yee component's sample lattice.

    Args:
        grid (RectilinearGrid): The resolved simulation grid.
        field (str): ``"E"``, ``"H"`` or ``"V"`` (the cell-vertex lattice).
        component (int): Component index 0, 1 or 2. Ignored for ``"V"``, which is one lattice.

    Returns:
        tuple: ``(x, y, z)`` coordinate arrays of lengths ``(Nx, Ny, Nz)``, in metres.

    Raises:
        ValueError: If ``field`` is not ``"E"``, ``"H"`` or ``"V"``.
    """
    if field == "E":
        offsets = E_OFFSETS[component]
    elif field == "H":
        offsets = H_OFFSETS[component]
    elif field == "V":
        offsets = V_OFFSETS[component]
    else:
        raise ValueError(f"field must be 'E', 'H' or 'V', got {field!r}")
    coords = []
    for axis in range(3):
        edges = np.asarray(grid.edges(axis), dtype=float)
        if offsets[axis] == 0.0:
            coords.append(edges[:-1])
        else:
            coords.append(0.5 * (edges[:-1] + edges[1:]))
    return coords[0], coords[1], coords[2]


def cell_center_coordinates(grid: RectilinearGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the shared cell-centre lattice the legacy ``"box"`` path samples on."""
    coords = []
    for axis in range(3):
        edges = np.asarray(grid.edges(axis), dtype=float)
        coords.append(0.5 * (edges[:-1] + edges[1:]))
    return coords[0], coords[1], coords[2]


@dataclass
class SceneEntry:
    """One static object as the scene sees it: a continuous shape with a priority."""

    #: Object name, for reporting.
    name: str
    #: The placed object itself; queried through ``contains`` / ``material_at``.
    obj: StaticObject
    #: Position in the write order. Higher wins.
    priority: int
    #: Map from the object's own material index to the scene-global material index.
    local_to_global: np.ndarray
    #: Continuous per-axis ``(lower, upper)`` bounds in metres, used to window the evaluation.
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


@dataclass
class Scene:
    """The full static scene: a global material list plus the prioritised object entries."""

    #: Global material dict, keyed by a scene-unique name.
    materials: dict[str, Material]
    #: Entries in ascending priority (the order the box path writes them).
    entries: list[SceneEntry]
    #: Global material index of the simulation volume, used as the background.
    background_index: int


def _material_signature(material: Material) -> tuple:
    """Value signature of a material, used to detect two definitions sharing one name."""
    return (
        tuple(material.permittivity),
        tuple(material.permeability),
        tuple(material.electric_conductivity),
        tuple(material.magnetic_conductivity),
        repr(material.dispersion),
    )


def _uniform_material_key(obj: SimulationObject) -> str:
    """Scene-global key for the single material of a ``UniformMaterialObject``."""
    return f"__uniform__::{obj.name}"


def build_scene(static_objects: Sequence[StaticObject]) -> Scene:
    """Merge every static object's materials into one global list and order the objects by priority.

    Args:
        static_objects (Sequence[SimulationObject]): The container's ``static_material_objects``.

    Returns:
        Scene: The global material list plus one entry per object, in ascending write order.

    Raises:
        ValueError: If two objects give the same material name two different definitions, or if no
            simulation volume is present.
    """
    ordered = sorted(static_objects, key=lambda o: o.placement_order)

    materials: dict[str, Material] = {}

    def _register(name: str, material: Material) -> None:
        existing = materials.get(name)
        if existing is not None and _material_signature(existing) != _material_signature(material):
            raise ValueError(
                f"Material name {name!r} is defined twice with different properties. The yee scene "
                "loader needs one global material list; rename one of them."
            )
        materials[name] = material

    for obj in ordered:
        if isinstance(obj, UniformMaterialObject):
            _register(_uniform_material_key(obj), obj.material)
        elif isinstance(obj, StaticMultiMaterialObject):
            for name, material in obj.materials.items():
                _register(name, material)
        else:
            raise ValueError(f"Unknown static object type in the yee scene loader: {type(obj).__name__}")

    global_names = compute_ordered_names(materials)
    global_index = {name: idx for idx, name in enumerate(global_names)}

    entries: list[SceneEntry] = []
    background_index = None
    for priority, obj in enumerate(ordered):
        if isinstance(obj, UniformMaterialObject):
            lut = np.asarray([global_index[_uniform_material_key(obj)]], dtype=np.int32)
        else:
            assert isinstance(obj, StaticMultiMaterialObject)
            lut = np.asarray([global_index[name] for name in compute_ordered_names(obj.materials)], dtype=np.int32)
        entries.append(
            SceneEntry(
                name=obj.name,
                obj=obj,
                priority=priority,
                local_to_global=lut,
                bounds=obj.metric_bounds,
            )
        )
        if isinstance(obj, SimulationVolume):
            background_index = int(lut[0])
    if background_index is None:
        raise ValueError("The yee scene loader needs a SimulationVolume to use as the scene background.")
    return Scene(materials=materials, entries=entries, background_index=background_index)


#: Fraction of the smallest cell width by which every sample point is shifted before the containment
#: tests. A shape face that lands exactly on a lattice point (a 500 nm box on a 25 nm grid, or a 2-D
#: object whose extrusion is one cell thick) would otherwise be decided by the float32 rounding of the
#: grid edges against the float64 metric shadow, differently at every resolution. Shifting the points by
#: +tol on every axis makes every face half-open in the same direction: a point on a lower face is inside,
#: a point on an upper face is outside, for boxes, slabs and polygon edges alike. 1e-3 of a cell is far
#: above the float32 coordinate error (about 1e-7 relative) and far below anything physical.
_TIE_NUDGE_FRACTION = 1e-3


def _nudge_off_ties(coords: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift the lattice coordinates by a fixed fraction of the smallest cell width (see above)."""
    widths = [np.min(np.diff(c)) for c in coords if c.size > 1]
    if not widths:
        return coords
    tol = _TIE_NUDGE_FRACTION * float(min(widths))
    return (
        np.asarray(coords[0], dtype=float) + tol,
        np.asarray(coords[1], dtype=float) + tol,
        np.asarray(coords[2], dtype=float) + tol,
    )


def _axis_window(coords: np.ndarray, lower: float, upper: float) -> tuple[int, int]:
    """Index range of the lattice samples that fall in the half-open metric interval."""
    start = int(np.searchsorted(coords, lower, side="left"))
    stop = int(np.searchsorted(coords, upper, side="left"))
    return start, max(start, stop)


#: Packed code of the identity periodic shift, ``9*(0+1) + 3*(0+1) + (0+1)``.
SHIFT_IDENTITY = 13


def grid_periods(grid: RectilinearGrid) -> tuple[float, float, float]:
    """Metric period of each axis: the full extent of the grid, which is what wrap padding wraps."""
    periods = []
    for axis in range(3):
        edges = np.asarray(grid.edges(axis), dtype=float)
        periods.append(float(edges[-1] - edges[0]))
    return periods[0], periods[1], periods[2]


def periodic_image_axes(
    grid: RectilinearGrid,
    periodic_axes: tuple[bool, bool, bool],
) -> tuple[bool, bool, bool]:
    """Axes that get periodic images of the geometry: periodic *and* actually resolved.

    A periodic flag alone is not enough. fdtdx's 2-D convention is one cell with periodic boundaries
    on the third axis, so every 2-D scene reports a periodic z. Replicating objects along an axis
    the simulation is invariant along would put images outside the one-cell-thick shapes, report a
    spurious second material and move the statistics of every recorded 2-D run.

    Args:
        grid (RectilinearGrid): The resolved simulation grid.
        periodic_axes (tuple): Which axes carry a periodic or Bloch boundary.

    Returns:
        tuple: The three flags with the invariant axes cleared.
    """
    flags = [bool(periodic_axes[axis]) and grid.shape[axis] > 1 for axis in range(3)]
    return (flags[0], flags[1], flags[2])


def entry_shifts(
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    periodic_axes: tuple[bool, bool, bool],
    periods: tuple[float, float, float],
) -> list[np.ndarray]:
    """Candidate periodic translations of one object, the identity first.

    One lattice vector each way per periodic axis, which is libctl's own bound (``LOOP_PERIODIC``
    runs every axis from -1 to +1 and is the only periodic search in the library). A candidate is
    *not* pruned against the domain here: on a periodic axis the evaluated coordinates reach below
    the first edge, so an image that misses the domain can still cover part of the domain-edge
    pixel. The window test the caller already does prunes it for free against the coordinates
    actually being evaluated.

    Args:
        bounds (tuple): The object's own per-axis ``(lower, upper)`` metric bounds.
        periodic_axes (tuple): Axes that get images, from :func:`periodic_image_axes`.
        periods (tuple): Metric period per axis.

    Returns:
        list: ``(3,)`` float arrays; the first is always the zero shift.

    Raises:
        ValueError: If an object is more than two periods long on a periodic axis, where one
            lattice vector each way no longer covers it.
    """
    per_axis: list[list[float]] = []
    for axis in range(3):
        options = [0.0]
        if periodic_axes[axis] and periods[axis] > 0.0:
            extent = bounds[axis][1] - bounds[axis][0]
            if extent > 2.0 * periods[axis] * (1.0 + 1e-9):
                raise ValueError(
                    f"Object extent {extent} on periodic axis {axis} exceeds two periods "
                    f"({periods[axis]}); one lattice vector each way no longer covers the domain."
                )
            options.extend([-periods[axis], periods[axis]])
        per_axis.append(options)
    shifts = []
    for shift_x in per_axis[0]:
        for shift_y in per_axis[1]:
            for shift_z in per_axis[2]:
                shifts.append(np.array([shift_x, shift_y, shift_z], dtype=float))
    return shifts


def shift_code(shift: np.ndarray, periods: tuple[float, float, float]) -> int:
    """Pack a shift as ``9*(mx+1) + 3*(my+1) + (mz+1)``; the identity is :data:`SHIFT_IDENTITY`."""
    code = 0
    for axis in range(3):
        step = 0 if periods[axis] <= 0.0 else round(float(shift[axis]) / periods[axis])
        code = code * 3 + (step + 1)
    return code


def unpack_shift_codes(codes: np.ndarray, periods: tuple[float, float, float]) -> np.ndarray:
    """Metric shift per entry, from the packed codes of :func:`shift_code`."""
    packed = np.asarray(codes, dtype=np.int64)
    out = np.zeros((packed.shape[0], 3), dtype=float)
    out[:, 0] = (packed // 9 - 1) * periods[0]
    out[:, 1] = ((packed // 3) % 3 - 1) * periods[1]
    out[:, 2] = (packed % 3 - 1) * periods[2]
    return out


def _entry_material_at(entry: SceneEntry, points: np.ndarray) -> np.ndarray:
    """Global material index of one entry at each point (constant for every object today)."""
    obj = entry.obj
    if isinstance(obj, UniformMaterialObject):
        return np.full(points.shape[:-1], entry.local_to_global[0], dtype=np.int32)
    assert isinstance(obj, StaticMultiMaterialObject)
    return entry.local_to_global[np.asarray(obj.material_at(points), dtype=np.int64)]


def front_material_indices(
    scene: Scene,
    coords: tuple[np.ndarray, np.ndarray, np.ndarray],
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
    periods: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Resolve, at every lattice point, the material of the highest-priority object containing it.

    Objects are applied in ascending priority (the box path's write order), each one overwriting the
    points it contains, so the last object to claim a point wins — the same result carving gets from
    sequential writes, evaluated at one lattice instead of at cell centres.

    Args:
        scene (Scene): The scene from :func:`build_scene`.
        coords (tuple): The three 1-D lattice coordinate arrays from
            :func:`yee_lattice_coordinates`.

    Returns:
        np.ndarray: ``int32`` array of shape ``(len(x), len(y), len(z))`` with global material
        indices.
    """
    return front_indices(scene, coords, periodic_axes, periods)[0]


def front_indices(
    scene: Scene,
    coords: tuple[np.ndarray, np.ndarray, np.ndarray],
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
    periods: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve the material, the winning object and its periodic image at every lattice point.

    Same pass as :func:`front_material_indices`, which is a thin wrapper over this. The second
    output is the winning entry's position in ``scene.entries`` (its priority), which the sub-pixel
    smoothing step needs to know *which shape* bounds the material inside a pixel — a material index
    alone does not identify an object. The third says which periodic image of that shape won, which
    the smoothing step needs to take the fill fraction and the normal in the right frame.

    On a periodic axis every object is also tested one lattice vector each way. The object itself is
    never moved: its bounding interval is shifted to window the lattice, and the query points are
    shifted back into the object's own frame, which is what libctl and Meep both do. An object and
    its images are one entry with one priority, so they cannot outrank each other, and the ascending
    write order is unchanged.

    Args:
        scene (Scene): The scene from :func:`build_scene`.
        coords (tuple): The three 1-D lattice coordinate arrays.
        periodic_axes (tuple): Axes to replicate along, from :func:`periodic_image_axes`.
        periods (tuple): Metric period per axis, from :func:`grid_periods`.

    Returns:
        tuple: ``(material_index, entry_index, shift_code)``, the first two ``int32`` and the third
        ``int8``, all of shape ``(len(x), len(y), len(z))``. Points claimed by no object carry the
        background material, entry ``-1`` and the identity shift.
    """
    shape = (coords[0].size, coords[1].size, coords[2].size)
    front = np.full(shape, scene.background_index, dtype=np.int32)
    owner = np.full(shape, -1, dtype=np.int32)
    shifts = np.full(shape, SHIFT_IDENTITY, dtype=np.int8)
    # Nudge first, then subtract the shift. A shift is an exact translation by one period, so
    # (x + tol) - L carries the same +tol offset relative to the image's faces as x + tol does
    # relative to the original's; the half-open convention is then identical in every image. Doing
    # it the other way round would compute the tolerance from the shifted array and could round
    # differently at the seam.
    coords = _nudge_off_ties(coords)

    for entry in scene.entries:
        for shift in entry_shifts(entry.bounds, periodic_axes, periods):
            moved = bool(np.any(shift != 0.0))
            windows = [
                _axis_window(coords[axis], entry.bounds[axis][0] + shift[axis], entry.bounds[axis][1] + shift[axis])
                for axis in range(3)
            ]
            sizes = [w[1] - w[0] for w in windows]
            if any(size == 0 for size in sizes):
                continue
            code = np.int8(shift_code(shift, periods))
            plane = max(1, sizes[1] * sizes[2])
            chunk = max(1, min(sizes[0], _CHUNK_POINTS // plane))
            y_coords = coords[1][windows[1][0] : windows[1][1]]
            z_coords = coords[2][windows[2][0] : windows[2][1]]
            for begin in range(windows[0][0], windows[0][1], chunk):
                end = min(begin + chunk, windows[0][1])
                x_coords = coords[0][begin:end]
                block = np.stack(np.meshgrid(x_coords, y_coords, z_coords, indexing="ij"), axis=-1)
                if moved:
                    block = block - shift
                mask = np.asarray(entry.obj.contains(block), dtype=bool)
                if not mask.any():
                    continue
                target = (
                    slice(begin, end),
                    slice(windows[1][0], windows[1][1]),
                    slice(windows[2][0], windows[2][1]),
                )
                values = _entry_material_at(entry, block)
                front[target] = np.where(mask, values, front[target])
                owner[target] = np.where(mask, np.int32(entry.priority), owner[target])
                shifts[target] = np.where(mask, code, shifts[target])
    return front, owner, shifts


def box_mode_material_indices(
    scene: Scene,
    volume_shape: tuple[int, int, int],
) -> np.ndarray:
    """Reproduce the legacy ``"box"`` path's per-cell material assignment, for comparison only.

    Each object's rounded box is filled with its own cell-centre voxel mask, written in the same
    ascending order, exactly as ``_init_arrays`` writes the arrays today. Used to report how many
    Yee points the two sampling modes disagree on; it never feeds the simulation.

    Args:
        scene (Scene): The scene from :func:`build_scene`.
        volume_shape (tuple[int, int, int]): The simulation grid shape.

    Returns:
        np.ndarray: ``int32`` array of shape ``volume_shape`` with global material indices.
    """
    front = np.full(volume_shape, scene.background_index, dtype=np.int32)
    for entry in scene.entries:
        obj = entry.obj
        grid_slice = obj.grid_slice
        if isinstance(obj, UniformMaterialObject):
            front[grid_slice] = entry.local_to_global[0]
            continue
        assert isinstance(obj, StaticMultiMaterialObject)
        mask = np.asarray(obj.get_voxel_mask_for_shape(), dtype=bool)
        local = np.asarray(obj.get_material_mapping(), dtype=np.int64)
        values = entry.local_to_global[local]
        front[grid_slice] = np.where(mask, values, front[grid_slice])
    return front


@dataclass
class YeeSceneArrays:
    """Host-side material arrays assembled from per-Yee-point sampling."""

    inv_permittivities: np.ndarray
    #: ``(3, Nx, Ny, Nz)`` off-diagonal entries ``(xy, xz, yz)`` of the smoothed inverse
    #: permittivity on the cell-vertex lattice, or ``None`` unless the vertex (node) placement is
    #: active. Zero wherever no interface was blended.
    inv_permittivity_offdiag: np.ndarray | None
    inv_permeabilities: np.ndarray | None
    electric_conductivity: np.ndarray | None
    magnetic_conductivity: np.ndarray | None
    dispersive_c1: np.ndarray | None
    dispersive_c2: np.ndarray | None
    dispersive_c3: np.ndarray | None
    front_E: np.ndarray
    front_H: np.ndarray | None
    #: Diagnostic: how many sampled points carry a different material than the ``"box"`` path.
    sampling_difference: dict[str, Any]


def _diagonal_table(values: Sequence[tuple[float, ...]]) -> np.ndarray:
    """Stack a per-material property list into an ``(M, ncomp)`` table."""
    return np.asarray(values, dtype=np.float64)


def _row_of_inverse_tensor(table_9: np.ndarray) -> np.ndarray:
    """Invert every material's 3x3 property tensor, returning shape ``(M, 3, 3)``."""
    matrices = table_9.reshape(table_9.shape[0], 3, 3)
    return np.linalg.inv(matrices)


def _assemble_property(
    front: np.ndarray,
    table: np.ndarray,
    num_components: int,
    invert: bool,
) -> np.ndarray:
    """Gather a per-material property table onto the per-component front-material arrays.

    Row ``c`` of a material's tensor is what multiplies the curl to give component ``c``, so the
    whole row belongs at ``E_c``'s (or ``H_c``'s) own position — the only self-consistent choice.

    Args:
        front (np.ndarray): ``(3, Nx, Ny, Nz)`` global material index per component.
        table (np.ndarray): ``(M, 3)`` diagonal or ``(M, 9)`` full-tensor property values.
        num_components (int): 3 for the diagonal tier, 9 for the full-tensor tier.
        invert (bool): Whether to invert the property (permittivity/permeability) or gather it
            directly (conductivity).

    Returns:
        np.ndarray: ``(num_components, Nx, Ny, Nz)``.
    """
    spatial = front.shape[1:]
    out = np.zeros((num_components, *spatial), dtype=np.float64)
    if num_components == 3:
        diag = table if table.shape[1] == 3 else table[:, (0, 4, 8)]
        for component in range(3):
            gathered = diag[front[component], component]
            out[component] = 1.0 / gathered if invert else gathered
        return out
    full = table if table.shape[1] == 9 else _expand_diagonal_to_9(table)
    source = _row_of_inverse_tensor(full) if invert else full.reshape(full.shape[0], 3, 3)
    for component in range(3):
        rows = source[front[component], component, :]
        for j in range(3):
            out[3 * component + j] = rows[..., j]
    return out


def _expand_diagonal_to_9(table_3: np.ndarray) -> np.ndarray:
    """Widen an ``(M, 3)`` diagonal table into an ``(M, 9)`` row-major tensor table."""
    out = np.zeros((table_3.shape[0], 9), dtype=np.float64)
    for idx, position in enumerate((0, 4, 8)):
        out[:, position] = table_3[:, idx]
    return out


def load_scene_on_yee_lattices(
    static_objects: Sequence[StaticObject],
    grid: RectilinearGrid,
    volume_shape: tuple[int, int, int],
    num_perm_components: int,
    num_permeability_components: int | None,
    num_electric_cond_components: int | None,
    num_magnetic_cond_components: int | None,
    num_dispersive_poles: int,
    num_disp_components: int,
    num_disp_coupling_components: int,
    conductivity_spacing: float | None,
    time_step_duration: float,
    smooth: bool = False,
    supersample: int = 8,
    full_tensor: bool = False,
    report_box_difference: bool = False,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
    offdiag_on_vertices: bool = False,
    offdiag_placement: str = "node",
) -> YeeSceneArrays:
    """Assemble every static material array by sampling the scene at the Yee component positions.

    Args:
        static_objects (Sequence[SimulationObject]): The container's ``static_material_objects``.
        grid (RectilinearGrid): The resolved simulation grid.
        volume_shape (tuple[int, int, int]): Grid shape of the simulation volume.
        num_perm_components (int): 3 or 9.
        num_permeability_components (int | None): 3 or 9, or ``None`` when permeability stays scalar.
        num_electric_cond_components (int | None): 3 or 9, or ``None`` when there is no array.
        num_magnetic_cond_components (int | None): 3 or 9, or ``None`` when there is no array.
        num_dispersive_poles (int): Pole count the arrays are padded to; 0 disables dispersion.
        num_disp_components (int): Component count of ``c1``/``c2``.
        num_disp_coupling_components (int): Component count of ``c3`` (3 or 9).
        conductivity_spacing (float | None): Scale factor applied to conductivities.
        time_step_duration (float): Simulation time step, for the dispersive recurrence.
        smooth (bool): Replace the point sample by the Kottke blend at two-material pixels
            (``material_sampling="yee_smooth"``). The permittivity is smoothed on the three E
            lattices and the permeability, when its array exists, on the three H lattices.
            Conductivity and dispersion stay point-sampled on both.
        supersample (int): Samples per axis used for a fill fraction or a normal that no shape can
            answer analytically.
        full_tensor (bool): Request the 9-component tier. The row form is used whenever
            ``num_perm_components`` is 9, however that tier was reached.
        periodic_axes (tuple): Which axes carry a periodic or Bloch boundary. An object crossing
            such a face is also evaluated one lattice vector each way, so it reappears on the other
            side, and the domain-edge smoothing pixel becomes the full dual box instead of being
            clipped. An axis the simulation is invariant along is excluded, since fdtdx's 2-D
            convention is a single periodic cell there.
        offdiag_on_vertices (bool): Put the three off-diagonal Kottke entries on the cell-vertex
            lattice, in ``inv_permittivity_offdiag``. The permittivity array itself then stays on
            the diagonal tier and the update applies the vertex entries as an additive correction.
        offdiag_placement (str): Which vertex placement, when ``offdiag_on_vertices`` is set.
            ``"node"`` takes the entries straight off the vertex dual cells and leaves the diagonal
            entries bit-identical to a run without the flag. ``"node_avg"`` instead reads them off
            the component pixels and averages four of them onto each vertex, so every entry comes
            from the component boxes. ``"vertex_all"`` computes the whole tensor on the vertex dual
            cells and rebuilds the diagonal entries from it, so every entry comes from the vertex
            boxes. All three assemble an exactly symmetric D-to-E map.
        report_box_difference (bool): Also rasterise the scene the legacy ``"box"`` way and count
            how many Yee points the two modes disagree on. Off by default: it is a second pass over
            every object plus another ``int32`` copy of the domain, and nothing in the simulation
            reads the answer. It is cheaper than the Yee pass it is compared against (it only fills
            each object's rounded box), so the added cost is a fraction rather than a doubling.

    Returns:
        YeeSceneArrays: The host-side arrays plus the front-material index arrays.
    """
    scene = build_scene(static_objects)
    materials = scene.materials
    image_axes = periodic_image_axes(grid, periodic_axes)
    periods = grid_periods(grid)

    need_H = num_permeability_components is not None or num_magnetic_cond_components is not None

    resolved_E = [front_indices(scene, yee_lattice_coordinates(grid, "E", c), image_axes, periods) for c in range(3)]
    front_E = np.stack([r[0] for r in resolved_E], axis=0)
    owner_E = np.stack([r[1] for r in resolved_E], axis=0)
    shift_E = np.stack([r[2] for r in resolved_E], axis=0)
    front_H = None
    owner_H = None
    shift_H = None
    if need_H:
        resolved_H = [
            front_indices(scene, yee_lattice_coordinates(grid, "H", c), image_axes, periods) for c in range(3)
        ]
        front_H = np.stack([r[0] for r in resolved_H], axis=0)
        owner_H = np.stack([r[1] for r in resolved_H], axis=0)
        shift_H = np.stack([r[2] for r in resolved_H], axis=0)

    difference: dict[str, Any] = {"box_difference_reported": report_box_difference}
    if report_box_difference:
        box_front = box_mode_material_indices(scene, volume_shape)
        difference["num_points_E"] = int(front_E.size)
        difference["num_differing_E"] = int(np.count_nonzero(front_E != box_front[None, ...]))
        difference["num_differing_E_per_component"] = [int(np.count_nonzero(front_E[c] != box_front)) for c in range(3)]
        if front_H is not None:
            difference["num_points_H"] = int(front_H.size)
            difference["num_differing_H"] = int(np.count_nonzero(front_H != box_front[None, ...]))

    permittivity_table = _diagonal_table(
        compute_allowed_permittivities(materials, diagonally_anisotropic=num_perm_components == 3)
    )
    inv_permittivities = _assemble_property(front_E, permittivity_table, num_perm_components, invert=True)
    inv_permittivity_offdiag = None
    vertex_placement = offdiag_placement if (smooth and offdiag_on_vertices) else None
    if vertex_placement in ("node_avg", "vertex_all") and num_perm_components != 3:
        # Both rebuild the 3-component diagonal array themselves, so they cannot also be the
        # 9-component tier. The initialization gate never lets this combination through.
        raise ValueError(
            f"offdiag_placement={vertex_placement!r} needs the 3-component permittivity tier, "
            f"got num_perm_components={num_perm_components}"
        )

    if smooth and vertex_placement == "vertex_all":
        # Every entry off the vertex dual cells: the whole tensor is smoothed there and the diagonal
        # is put back at the component points as the mean of the two vertices bracketing each one.
        # The component lattices are not smoothed at all on this path, so there is one pass, not two.
        from fdtdx.core.physics.geometry_smooth import (
            pixel_diagonals_from_vertex_tensor,
            smooth_permittivity_tensor_on_vertex_lattice,
        )

        vertex_tensor, offdiag_stats = smooth_permittivity_tensor_on_vertex_lattice(
            scene=scene,
            grid=grid,
            supersample=supersample,
            periodic_axes=image_axes,
        )
        inv_permittivities = pixel_diagonals_from_vertex_tensor(vertex_tensor, image_axes)
        inv_permittivity_offdiag = np.ascontiguousarray(vertex_tensor[3:])
        # One pass produced both halves, so both counters name the same numbers.
        difference["smoothing"] = offdiag_stats.as_dict()
        difference["smoothing_offdiag"] = offdiag_stats.as_dict()
    elif smooth and vertex_placement == "node_avg":
        # Every entry off the component pixels: the full Kottke row is smoothed there, the diagonal
        # entries are taken straight out of it (entry (c, c) of a row-major 3x3 lives at 4*c, so the
        # values are the ones the diagonal tier writes) and the off-diagonal entries are averaged
        # onto the vertices. The 9-component array is scratch; it never reaches the simulation.
        from fdtdx.core.physics.geometry_smooth import (
            smooth_property_on_yee_pixels,
            vertex_offdiagonals_from_pixel_rows,
        )

        rows, smoothing_stats = smooth_property_on_yee_pixels(
            scene=scene,
            grid=grid,
            field="E",
            property_kind="permittivity",
            front_material=front_E,
            front_owner=owner_E,
            front_shift=shift_E,
            periodic_axes=image_axes,
            inverse_property=_assemble_property(front_E, permittivity_table, 9, invert=True),
            supersample=supersample,
            full_tensor=True,
        )
        inv_permittivities = np.ascontiguousarray(rows[(0, 4, 8), ...])
        inv_permittivity_offdiag = vertex_offdiagonals_from_pixel_rows(rows, image_axes)
        difference["smoothing"] = smoothing_stats.as_dict()
        difference["smoothing_offdiag"] = smoothing_stats.as_dict()
    elif smooth:
        from fdtdx.core.physics.geometry_smooth import smooth_property_on_yee_pixels

        inv_permittivities, smoothing_stats = smooth_property_on_yee_pixels(
            scene=scene,
            grid=grid,
            field="E",
            property_kind="permittivity",
            front_material=front_E,
            front_owner=owner_E,
            front_shift=shift_E,
            periodic_axes=image_axes,
            inverse_property=inv_permittivities,
            # A 9-component array must always be written as the full Kottke row: entry (c, c) of a
            # row-major 3x3 lives at 4*c, not at c. The config flag's only job is to force the tier.
            supersample=supersample,
            full_tensor=num_perm_components == 9,
        )
        difference["smoothing"] = smoothing_stats.as_dict()

    if vertex_placement == "node":
        from fdtdx.core.physics.geometry_smooth import smooth_offdiagonal_on_vertex_lattice

        inv_permittivity_offdiag, offdiag_stats = smooth_offdiagonal_on_vertex_lattice(
            scene=scene,
            grid=grid,
            supersample=supersample,
            periodic_axes=image_axes,
        )
        difference["smoothing_offdiag"] = offdiag_stats.as_dict()
    elif vertex_placement is not None and vertex_placement not in ("node_avg", "vertex_all"):
        raise ValueError(f"unknown off-diagonal placement {vertex_placement!r}")

    inv_permeabilities = None
    if num_permeability_components is not None:
        assert front_H is not None
        inv_permeabilities = _assemble_property(
            front_H,
            _diagonal_table(
                compute_allowed_permeabilities(materials, diagonally_anisotropic=num_permeability_components == 3)
            ),
            num_permeability_components,
            invert=True,
        )
        if smooth:
            # The permeability gets the identical treatment on the three H lattices. There is no
            # separate "has_mu" test: the array exists exactly when some material is magnetic
            # (fdtdx's all_objects_non_magnetic, which is Meep's has_mu), so a mu = 1 scene never
            # reaches this branch and is untouched by construction.
            from fdtdx.core.physics.geometry_smooth import smooth_property_on_yee_pixels

            assert owner_H is not None and shift_H is not None
            inv_permeabilities, permeability_stats = smooth_property_on_yee_pixels(
                scene=scene,
                grid=grid,
                field="H",
                property_kind="permeability",
                front_material=front_H,
                front_owner=owner_H,
                front_shift=shift_H,
                periodic_axes=image_axes,
                inverse_property=inv_permeabilities,
                supersample=supersample,
                full_tensor=num_permeability_components == 9,
            )
            difference["smoothing_H"] = permeability_stats.as_dict()

    electric_conductivity = None
    if num_electric_cond_components is not None:
        assert conductivity_spacing is not None
        electric_conductivity = (
            _assemble_property(
                front_E,
                _diagonal_table(
                    compute_allowed_electric_conductivities(
                        materials, diagonally_anisotropic=num_electric_cond_components == 3
                    )
                ),
                num_electric_cond_components,
                invert=False,
            )
            * conductivity_spacing
        )

    magnetic_conductivity = None
    if num_magnetic_cond_components is not None:
        assert conductivity_spacing is not None
        assert front_H is not None
        magnetic_conductivity = (
            _assemble_property(
                front_H,
                _diagonal_table(
                    compute_allowed_magnetic_conductivities(
                        materials, diagonally_anisotropic=num_magnetic_cond_components == 3
                    )
                ),
                num_magnetic_cond_components,
                invert=False,
            )
            * conductivity_spacing
        )

    dispersive_c1 = dispersive_c2 = dispersive_c3 = None
    if num_dispersive_poles > 0:
        allowed_c1, allowed_c2, allowed_c3 = compute_allowed_dispersive_coefficients(
            materials,
            dt=time_step_duration,
            max_num_poles=num_dispersive_poles,
            num_components=num_disp_components,
            coupling_components=num_disp_coupling_components,
        )
        dispersive_c1 = _gather_dispersive(front_E, np.asarray(allowed_c1), num_disp_components)
        dispersive_c2 = _gather_dispersive(front_E, np.asarray(allowed_c2), num_disp_components)
        dispersive_c3 = _gather_dispersive(front_E, np.asarray(allowed_c3), num_disp_coupling_components)

    return YeeSceneArrays(
        inv_permittivities=inv_permittivities,
        inv_permittivity_offdiag=inv_permittivity_offdiag,
        inv_permeabilities=inv_permeabilities,
        electric_conductivity=electric_conductivity,
        magnetic_conductivity=magnetic_conductivity,
        dispersive_c1=dispersive_c1,
        dispersive_c2=dispersive_c2,
        dispersive_c3=dispersive_c3,
        front_E=front_E,
        front_H=front_H,
        sampling_difference=difference,
    )


def _gather_dispersive(front_E: np.ndarray, table: np.ndarray, num_components: int) -> np.ndarray:
    """Gather ``(M, poles, ncomp)`` dispersive coefficients onto ``(poles, ncomp, Nx, Ny, Nz)``.

    Component ``c`` of the recurrence belongs at ``E_c``'s position. A 9-component coupling tier
    keeps the three entries of row ``c`` together, mirroring the permittivity rule.
    """
    num_poles = table.shape[1]
    spatial = front_E.shape[1:]
    out = np.zeros((num_poles, num_components, *spatial), dtype=np.float64)
    for component in range(3):
        gathered = table[front_E[component]]  # (Nx, Ny, Nz, poles, ncomp)
        gathered = np.moveaxis(gathered, (-2, -1), (0, 1))  # (poles, ncomp, Nx, Ny, Nz)
        if num_components == 3:
            out[:, component] = gathered[:, component]
        elif num_components == 9:
            for j in range(3):
                out[:, 3 * component + j] = gathered[:, 3 * component + j]
        else:
            raise ValueError(f"Unsupported dispersive component count {num_components} for yee sampling.")
    return out
