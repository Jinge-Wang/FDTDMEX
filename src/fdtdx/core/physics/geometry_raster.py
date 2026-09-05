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
the interface is still staircased.
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

#: Half-cell offsets of the three E components, in the order (x, y, z) per component.
E_OFFSETS: tuple[tuple[float, float, float], ...] = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5))
#: Half-cell offsets of the three H components, in the order (x, y, z) per component.
H_OFFSETS: tuple[tuple[float, float, float], ...] = ((0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))

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
        field (str): ``"E"`` or ``"H"``.
        component (int): Component index 0, 1 or 2.

    Returns:
        tuple: ``(x, y, z)`` coordinate arrays of lengths ``(Nx, Ny, Nz)``, in metres.

    Raises:
        ValueError: If ``field`` is not ``"E"`` or ``"H"``.
    """
    if field == "E":
        offsets = E_OFFSETS[component]
    elif field == "H":
        offsets = H_OFFSETS[component]
    else:
        raise ValueError(f"field must be 'E' or 'H', got {field!r}")
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
    obj: SimulationObject
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


def build_scene(static_objects: Sequence[SimulationObject]) -> Scene:
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
    return tuple(np.asarray(c, dtype=float) + tol for c in coords)  # type: ignore[return-value]


def _axis_window(coords: np.ndarray, lower: float, upper: float) -> tuple[int, int]:
    """Index range of the lattice samples that fall in the half-open metric interval."""
    start = int(np.searchsorted(coords, lower, side="left"))
    stop = int(np.searchsorted(coords, upper, side="left"))
    return start, max(start, stop)


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
    return front_indices(scene, coords)[0]


def front_indices(
    scene: Scene,
    coords: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve both the material and the winning object at every lattice point.

    Same pass as :func:`front_material_indices`, which is a thin wrapper over this. The second
    output is the winning entry's position in ``scene.entries`` (its priority), which the sub-pixel
    smoothing step needs to know *which shape* bounds the material inside a pixel — a material index
    alone does not identify an object.

    Args:
        scene (Scene): The scene from :func:`build_scene`.
        coords (tuple): The three 1-D lattice coordinate arrays.

    Returns:
        tuple: ``(material_index, entry_index)``, both ``int32`` of shape
        ``(len(x), len(y), len(z))``. Points claimed by no object carry the background material and
        entry ``-1``.
    """
    shape = (coords[0].size, coords[1].size, coords[2].size)
    front = np.full(shape, scene.background_index, dtype=np.int32)
    owner = np.full(shape, -1, dtype=np.int32)
    coords = _nudge_off_ties(coords)

    for entry in scene.entries:
        windows = [_axis_window(coords[axis], entry.bounds[axis][0], entry.bounds[axis][1]) for axis in range(3)]
        sizes = [w[1] - w[0] for w in windows]
        if any(size == 0 for size in sizes):
            continue
        plane = max(1, sizes[1] * sizes[2])
        chunk = max(1, min(sizes[0], _CHUNK_POINTS // plane))
        y_coords = coords[1][windows[1][0] : windows[1][1]]
        z_coords = coords[2][windows[2][0] : windows[2][1]]
        for begin in range(windows[0][0], windows[0][1], chunk):
            end = min(begin + chunk, windows[0][1])
            x_coords = coords[0][begin:end]
            block = np.stack(np.meshgrid(x_coords, y_coords, z_coords, indexing="ij"), axis=-1)
            mask = np.asarray(entry.obj.contains(block), dtype=bool)
            if not mask.any():
                continue
            target = (slice(begin, end), slice(windows[1][0], windows[1][1]), slice(windows[2][0], windows[2][1]))
            values = _entry_material_at(entry, block)
            front[target] = np.where(mask, values, front[target])
            owner[target] = np.where(mask, np.int32(entry.priority), owner[target])
    return front, owner


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
    static_objects: Sequence[SimulationObject],
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
            (``material_sampling="yee_smooth"``). Conductivity and dispersion stay point-sampled.
        supersample (int): Samples per axis used for a fill fraction or a normal that no shape can
            answer analytically.
        full_tensor (bool): Request the 9-component tier. The row form is used whenever
            ``num_perm_components`` is 9, however that tier was reached.

    Returns:
        YeeSceneArrays: The host-side arrays plus the front-material index arrays.
    """
    scene = build_scene(static_objects)
    materials = scene.materials

    need_H = num_permeability_components is not None or num_magnetic_cond_components is not None

    resolved_E = [front_indices(scene, yee_lattice_coordinates(grid, "E", c)) for c in range(3)]
    front_E = np.stack([r[0] for r in resolved_E], axis=0)
    owner_E = np.stack([r[1] for r in resolved_E], axis=0)
    front_H = None
    if need_H:
        front_H = np.stack(
            [front_material_indices(scene, yee_lattice_coordinates(grid, "H", c)) for c in range(3)],
            axis=0,
        )

    box_front = box_mode_material_indices(scene, volume_shape)
    difference = {
        "num_points_E": int(front_E.size),
        "num_differing_E": int(np.count_nonzero(front_E != box_front[None, ...])),
        "num_differing_E_per_component": [int(np.count_nonzero(front_E[c] != box_front)) for c in range(3)],
    }
    if front_H is not None:
        difference["num_points_H"] = int(front_H.size)
        difference["num_differing_H"] = int(np.count_nonzero(front_H != box_front[None, ...]))

    inv_permittivities = _assemble_property(
        front_E,
        _diagonal_table(compute_allowed_permittivities(materials, diagonally_anisotropic=num_perm_components == 3)),
        num_perm_components,
        invert=True,
    )
    if smooth:
        from fdtdx.core.physics.geometry_smooth import smooth_inverse_permittivity_on_yee_pixels

        inv_permittivities, smoothing_stats = smooth_inverse_permittivity_on_yee_pixels(
            scene=scene,
            grid=grid,
            front_material=front_E,
            front_owner=owner_E,
            inv_permittivities=inv_permittivities,
            # A 9-component array must always be written as the full Kottke row: entry (c, c) of a
            # row-major 3x3 lives at 4*c, not at c. The config flag's only job is to force the tier.
            supersample=supersample,
            full_tensor=num_perm_components == 9,
        )
        difference["smoothing"] = smoothing_stats.as_dict()

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
