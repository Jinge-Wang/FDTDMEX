"""M4 validation: non-uniform (rectilinear) grids on the MLX (Metal) backend.

Three things are checked:

1. **Element-wise parity vs the JAX-CPU oracle** for isotropic and diagonal-anisotropic
   materials on a stretched grid. fdtdx's own curl (``_metric_scale``) and detector
   interpolation (``_backward_edge_average``) are *already* spacing-weighted, and the MLX port
   mirrors them, so these must agree to float32 tolerance (the same bar as the uniform parity
   suite).

2. **2nd-order convergence of the spacing-weighted off-diagonal anisotropic average** -- the
   genuinely-new physics. fdtdx leaves ``avg_anisotropic_*`` *unweighted* even on non-uniform
   grids (1st-order on a graded mesh), so this is validated against analytic linear-interpolation
   accuracy, not element-wise vs fdtdx. On a strongly graded (alternating-width) mesh the weighted
   average converges at 2nd order while the unweighted one is only 1st order -- the weighted form
   reconstructs a linear field exactly.

3. **End-to-end finiteness** of a full 9-tensor crystal on a (mildly) stretched grid, exercising
   the whole non-uniform anisotropic path (bridge widths -> update -> loop).

Skipped off Apple Silicon / without mlx.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.backend.platform import is_apple_silicon, mlx_available

pytestmark = [
    pytest.mark.validation,
    pytest.mark.skipif(
        not (is_apple_silicon() and mlx_available()),
        reason="MLX (Metal) backend requires Apple Silicon + mlx",
    ),
]

_RES = 50e-9
_PML = 8
_NZ = 60
_WL = 1.0e-6
_RTOL = 1e-3


def _rel(j, m):
    j, m = np.asarray(j), np.asarray(m)
    return float(np.abs(j - m).max() / (np.abs(j).max() + 1e-30))


def _stretched_z_edges(nz=_NZ, amp=0.12):
    """A mildly stretched z grid (smoothly varying widths), same total length as uniform."""
    cells = np.arange(nz, dtype=float)
    widths = _RES * (1.0 + amp * np.sin(2.0 * np.pi * (cells + 0.5) / nz))
    widths *= (nz * _RES) / widths.sum()
    return jnp.asarray(np.concatenate([[0.0], np.cumsum(widths)]), dtype=jnp.float32)


def _run_both(objects, constraints, config):
    key = jax.random.PRNGKey(0)
    oc, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, oc, _ = fdtdx.apply_params(arrays, oc, params, key)
    with fdtdx.use_backend("jax"):
        _, arr_j = fdtdx.run_fdtd(arrays=arrays, objects=oc, config=config, key=key, show_progress=False)
    with fdtdx.use_backend("mlx"):
        _, arr_m = fdtdx.run_fdtd(arrays=arrays, objects=oc, config=config, key=key, show_progress=False)
    return arr_j, arr_m


def _real_z(obj, zedges, z_idx):
    """A lower-side z placement at the physical coordinate of grid edge ``z_idx`` (non-uniform-safe).

    Index-space ``set_grid_coordinates`` is rejected on non-uniform grids, so placement uses
    physical coordinates. Both backends run the same placed config, so the snapped plane is
    identical for the parity comparison.
    """
    return fdtdx.RealCoordinateConstraint(object=obj.name, axes=(2,), sides=("-",), coordinates=(float(zedges[z_idx]),))


def _stretched_slab_case(material):
    """Periodic-x/y, PML-z stretched grid with a plane wave through a material slab + detectors."""
    zedges = np.asarray(_stretched_z_edges())
    config = fdtdx.SimulationConfig(
        grid=fdtdx.RectilinearGrid.custom(
            x_edges=jnp.linspace(0.0, 3 * _RES, 4),
            y_edges=jnp.linspace(0.0, 3 * _RES, 4),
            z_edges=jnp.asarray(zedges),
        ),
        time=16e-15,
        dtype=jnp.float32,
    )
    objects, constraints = [], []
    vol = fdtdx.SimulationVolume(partial_real_shape=(3 * _RES, 3 * _RES, _NZ * _RES))
    objects.append(vol)
    bcfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML,
        override_types={"min_x": "periodic", "max_x": "periodic", "min_y": "periodic", "max_y": "periodic"},
    )
    bdict, clist = fdtdx.boundary_objects_from_config(bcfg, vol)
    constraints.extend(clist)
    objects.extend(bdict.values())

    slab = fdtdx.UniformMaterialObject(partial_grid_shape=(None, None, 20), material=material)
    constraints.extend(
        [slab.same_size(vol, axes=(0, 1)), slab.place_at_center(vol, axes=(0, 1)), _real_z(slab, zedges, _PML + 6)]
    )
    objects.append(slab)

    wave = fdtdx.WaveCharacter(wavelength=_WL)
    src = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1), wave_character=wave, direction="+", fixed_E_polarization_vector=(1, 0, 0)
    )
    constraints.extend(
        [src.same_size(vol, axes=(0, 1)), src.place_at_center(vol, axes=(0, 1)), _real_z(src, zedges, _PML + 2)]
    )
    objects.append(src)

    fdet = fdtdx.FieldDetector(name="F", reduce_volume=True, plot=False)
    constraints.extend([fdet.same_size(vol, axes=(0, 1, 2)), fdet.place_at_center(vol, axes=(0, 1, 2))])
    objects.append(fdet)
    ph = fdtdx.PhasorDetector(
        name="ph",
        partial_grid_shape=(None, None, 1),
        wave_characters=(wave,),
        reduce_volume=True,
        components=("Ex", "Hy"),
    )
    constraints.extend(
        [ph.same_size(vol, axes=(0, 1)), ph.place_at_center(vol, axes=(0, 1)), _real_z(ph, zedges, _NZ - _PML - 4)]
    )
    objects.append(ph)
    return _run_both(objects, constraints, config)


def test_nonuniform_isotropic_matches_jax():
    """Stretched grid + isotropic dielectric slab: element-wise parity vs JAX-CPU."""
    arr_j, arr_m = _stretched_slab_case(fdtdx.Material(permittivity=2.25))
    for name in ("E", "H", "psi_E", "psi_H"):
        assert _rel(getattr(arr_j.fields, name), getattr(arr_m.fields, name)) < _RTOL, f"field {name}"
    assert _rel(arr_j.detector_states["F"]["fields"], arr_m.detector_states["F"]["fields"]) < _RTOL
    assert _rel(arr_j.detector_states["ph"]["phasor"], arr_m.detector_states["ph"]["phasor"]) < _RTOL


def test_nonuniform_diagonal_matches_jax():
    """Stretched grid + diagonal-anisotropic slab: element-wise parity vs JAX-CPU."""
    arr_j, arr_m = _stretched_slab_case(fdtdx.Material(permittivity=(2.25, 4.0, 1.0)))
    for name in ("E", "H"):
        assert _rel(getattr(arr_j.fields, name), getattr(arr_m.fields, name)) < _RTOL, f"field {name}"
    assert _rel(arr_j.detector_states["ph"]["phasor"], arr_m.detector_states["ph"]["phasor"]) < _RTOL


# --- 2nd-order convergence of the spacing-weighted off-diagonal anisotropic average ----------


def _alternating_edges(length, n_pairs, ratio=2.0):
    """Strongly graded 1-D mesh: widths alternate w, ratio*w (O(1) adjacent-cell ratio at all scales)."""
    w = length / (n_pairs * (1.0 + ratio))
    widths = np.tile([w, ratio * w], n_pairs).astype(np.float64)
    edges = np.concatenate([[0.0], np.cumsum(widths)])
    return edges, widths


def _aniso_x_interp_error(n_pairs, weighted):
    """Max error of ``avg_anisotropic_E_component`` (Ex -> Ey location) reconstructing a smooth
    field at the backward x-edges, on an alternating-width mesh.

    The averaged field is constant along the (unweighted) location axis, so only the
    center->edge step along the component (x) axis is exercised. The weighted form must converge
    at 2nd order; the unweighted one is 1st order.
    """
    import mlx.core as mx

    from fdtdx.mlx.aniso import avg_anisotropic_E_component_mlx

    length = 1.0
    edges, widths = _alternating_edges(length, n_pairs)
    nx = widths.shape[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    k = 2.0 * np.pi / length
    f_center = np.sin(k * centers)  # Ex samples (x at cell centers)
    f_edge = np.sin(k * edges)  # analytic value at the x-edges

    ny = nz = 3
    field = np.zeros((3, nx, ny, nz), dtype=np.float32)
    field[0] = np.broadcast_to(f_center[:, None, None], (nx, ny, nz))
    field_pad = mx.array(np.pad(field, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="edge"))

    if weighted:
        wpad = np.concatenate([widths[:1], widths, widths[-1:]]).astype(np.float32)
        aniso_widths = (
            mx.array(wpad.reshape(nx + 2, 1, 1)),
            mx.array(np.ones((1, ny + 2, 1), np.float32)),
            mx.array(np.ones((1, 1, nz + 2), np.float32)),
        )
    else:
        aniso_widths = None

    out = np.asarray(avg_anisotropic_E_component_mlx(field_pad, 0, 1, aniso_widths))
    got = out[:, ny // 2, nz // 2]  # value at backward x-edge of each cell: got[i] ~ x_edges[i]
    # Interior edges (between two real cells) are i = 1 .. nx-1.
    return float(np.abs(got[1:nx] - f_edge[1:nx]).max())


def _order(errors):
    return float(np.mean([np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]))


def test_weighted_aniso_offdiagonal_second_order():
    """Weighted off-diagonal average is 2nd-order on a graded mesh; unweighted is 1st-order."""
    levels = [8, 16, 32, 64]
    err_weighted = [_aniso_x_interp_error(n, weighted=True) for n in levels]
    err_unweighted = [_aniso_x_interp_error(n, weighted=False) for n in levels]

    order_w = _order(err_weighted)
    order_u = _order(err_unweighted)

    assert order_w > 1.8, f"weighted average not 2nd-order: order={order_w:.3f}, errors={err_weighted}"
    assert order_u < 1.3, f"unweighted average should be ~1st-order: order={order_u:.3f}, errors={err_unweighted}"
    # The genuinely-new physics is strictly better on a graded mesh.
    assert err_weighted[-1] < 0.2 * err_unweighted[-1], (err_weighted[-1], err_unweighted[-1])


def test_nonuniform_full_anisotropy_runs_on_mlx():
    """A full 9-tensor crystal slab on a stretched grid runs to completion without NaNs on MLX.

    The off-diagonal coupling is in the x-z block, so the spacing-weighted off-diagonal average is
    exercised on the *stretched* z-axis (the Ez<->Ex average weights along z). This is the genuinely
    non-uniform anisotropic path end-to-end (bridge widths -> update -> loop). The result is *not*
    compared element-wise to the JAX run: fdtdx leaves the off-diagonal average unweighted on
    non-uniform grids, so the two backends legitimately differ on the stretched axis -- here we only
    require both to stay finite and non-trivial.
    """
    eps_tensor = ((2.5, 0.0, 0.3), (0.0, 3.0, 0.0), (0.3, 0.0, 4.0))  # mild x-z off-diagonal (stable; see Quirk A)
    arr_j, arr_m = _stretched_slab_case(fdtdx.Material(permittivity=eps_tensor))
    assert np.asarray(arr_m.inv_permittivities).shape[0] == 9
    for arr in (arr_j, arr_m):
        assert np.isfinite(np.asarray(arr.fields.E)).all()
        assert np.isfinite(np.asarray(arr.fields.H)).all()
    assert np.abs(np.asarray(arr_m.fields.E)).max() > 0.0
