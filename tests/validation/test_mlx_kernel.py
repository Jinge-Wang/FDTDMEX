"""Phase 2 M2 parity: the custom Metal E/H kernel path vs the MLX-op cores and the JAX-CPU oracle.

Two checks per case:
- **Kernel-core vs MLX-op-core (same engine), rel < 1e-4** — runs ``run_forward_mlx`` twice on
  identical fresh state with ``use_metal_kernel`` off/on. Isolates the kernel (indexing, ghost,
  slab-CPML hybrid) from the physics; the MLX-op path is already JAX-validated. Asserts the kernel
  path actually engaged (``kernels.KERNEL_CORES_BUILT``).
- **Kernel path vs forced-JAX, rel < 1e-3** — the end-to-end physics bar, with the kernel forced on
  via ``FDTDMEX_METAL_KERNEL`` through the real dispatcher.

Covers the eligible surface: isotropic + CPML, diagonal-anisotropic + CPML, and periodic (x/y) +
CPML (z). Skipped off Apple Silicon / without mlx.
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
_DOMAIN = 24 * _RES
_TIME = 12e-15


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / (np.abs(a).max() + 1e-30))


def _placed(objects, constraints, config):
    key = jax.random.PRNGKey(0)
    oc, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, oc, _ = fdtdx.apply_params(arrays, oc, params, key)
    return arrays, oc, config


def _mlx_out(arrays, oc, config, use_metal_kernel):
    """Replicate dispatch._run_mlx_forward but with an explicit kernel flag; return the container."""
    from fdtdx.fdtd.update import get_wrap_padding_axes
    from fdtdx.mlx.bridge import buffers_to_detector_states, to_array_container, to_mlx_state
    from fdtdx.mlx.detector_freeze import allocate_buffers, freeze_detectors
    from fdtdx.mlx.loop import run_forward_mlx
    from fdtdx.mlx.source_freeze import freeze_sources

    arr = arrays.reset()
    periodic_axes = get_wrap_padding_axes(oc)
    state = to_mlx_state(arr, config, periodic_axes)
    source_plans = freeze_sources(oc, config, arr)
    detector_plans = freeze_detectors(oc, config)
    detector_buffers = allocate_buffers(detector_plans)
    num_steps = int(config.time_steps_total)
    c = float(config.courant_number)
    state, detector_buffers = run_forward_mlx(
        state,
        source_plans,
        detector_plans,
        detector_buffers,
        num_steps,
        c,
        simulate_boundaries=True,
        use_metal_kernel=use_metal_kernel,
    )
    detector_states = buffers_to_detector_states(detector_buffers) if detector_plans else None
    return to_array_container(arr, state, detector_states)


def _assert_kernel_matches_ops(objects, constraints, config, rtol=1e-4):
    """Kernel-core vs MLX-op-core on identical fresh state; assert the kernel path engaged."""
    import fdtdx.mlx.kernels as kernels

    arrays, oc, config = _placed(objects, constraints, config)
    before = kernels.KERNEL_CORES_BUILT
    out_ops = _mlx_out(arrays, oc, config, use_metal_kernel=False)
    out_ker = _mlx_out(arrays, oc, config, use_metal_kernel=True)
    assert kernels.KERNEL_CORES_BUILT == before + 1, "kernel path did not engage (fell back to MLX ops)"

    for name in ("E", "H", "psi_E", "psi_H"):
        assert _rel(getattr(out_ops.fields, name), getattr(out_ker.fields, name)) < rtol, f"field {name}"
    return arrays, oc, config


def _assert_kernel_matches_jax(arrays, oc, config, field_names=("E", "H"), rtol=1e-3):
    with fdtdx.use_backend("jax"):
        _, arr_j = fdtdx.run_fdtd(
            arrays=arrays, objects=oc, config=config, key=jax.random.PRNGKey(0), show_progress=False
        )
    out_ker = _mlx_out(arrays, oc, config, use_metal_kernel=True)
    for name in field_names:
        assert _rel(getattr(arr_j.fields, name), getattr(out_ker.fields, name)) < rtol, f"field {name} vs JAX"


def _vacuum_objects():
    objects, constraints = [], []
    vol = fdtdx.SimulationVolume(partial_real_shape=(_DOMAIN,) * 3)
    objects.append(vol)
    bdict, clist = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML), vol)
    constraints.extend(clist)
    objects.extend(bdict.values())
    src = fdtdx.PointDipoleSource(
        partial_grid_shape=(1, 1, 1), wave_character=fdtdx.WaveCharacter(wavelength=1e-6), polarization=2
    )
    constraints.append(src.place_at_center(vol, axes=(0, 1, 2)))
    objects.append(src)
    det = fdtdx.EnergyDetector(name="energy", reduce_volume=True, plot=False)
    constraints.extend([det.same_size(vol, axes=(0, 1, 2)), det.place_at_center(vol, axes=(0, 1, 2))])
    objects.append(det)
    return objects, constraints


def test_kernel_isotropic_cpml():
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_RES), time=_TIME, dtype=jnp.float32)
    objects, constraints = _vacuum_objects()
    arrays, oc, config = _assert_kernel_matches_ops(objects, constraints, config)
    _assert_kernel_matches_jax(arrays, oc, config)


def test_kernel_diagonal_anisotropy_cpml():
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_RES), time=_TIME, dtype=jnp.float32)
    objects, constraints = [], []
    vol = fdtdx.SimulationVolume(
        partial_real_shape=(_DOMAIN,) * 3, material=fdtdx.Material(permittivity=(2.0, 3.0, 4.0))
    )
    objects.append(vol)
    bdict, clist = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML), vol)
    constraints.extend(clist)
    objects.extend(bdict.values())
    src = fdtdx.PointDipoleSource(
        partial_grid_shape=(1, 1, 1), wave_character=fdtdx.WaveCharacter(wavelength=1e-6), polarization=0
    )
    constraints.append(src.place_at_center(vol, axes=(0, 1, 2)))
    objects.append(src)

    arrays, oc, config = _assert_kernel_matches_ops(objects, constraints, config)
    _assert_kernel_matches_jax(arrays, oc, config)


def _stretched_z_edges(nz, res, amp=0.12):
    """A mildly stretched z grid (smoothly varying widths), same total length as uniform."""
    cells = np.arange(nz, dtype=float)
    widths = res * (1.0 + amp * np.sin(2.0 * np.pi * (cells + 0.5) / nz))
    widths *= (nz * res) / widths.sum()
    return jnp.asarray(np.concatenate([[0.0], np.cumsum(widths)]), dtype=jnp.float32)


def _nonuniform_case(material, polarization):
    """Periodic-x/y, PML-z stretched-z grid with a material slab + a plane-wave source (M3 metric)."""
    nz = 60
    zedges = _stretched_z_edges(nz, _RES)
    config = fdtdx.SimulationConfig(
        grid=fdtdx.RectilinearGrid.custom(
            x_edges=jnp.linspace(0.0, 3 * _RES, 4),
            y_edges=jnp.linspace(0.0, 3 * _RES, 4),
            z_edges=zedges,
        ),
        time=16e-15,
        dtype=jnp.float32,
    )
    objects, constraints = [], []
    vol = fdtdx.SimulationVolume(partial_real_shape=(3 * _RES, 3 * _RES, nz * _RES))
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
        [
            slab.same_size(vol, axes=(0, 1)),
            slab.place_at_center(vol, axes=(0, 1)),
            fdtdx.RealCoordinateConstraint(
                object=slab.name, axes=(2,), sides=("-",), coordinates=(float(zedges[_PML + 6]),)
            ),
        ]
    )
    objects.append(slab)

    src = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=1e-6),
        direction="+",
        fixed_E_polarization_vector=tuple(1 if a == polarization else 0 for a in range(3)),
    )
    constraints.extend(
        [
            src.same_size(vol, axes=(0, 1)),
            src.place_at_center(vol, axes=(0, 1)),
            fdtdx.RealCoordinateConstraint(
                object=src.name, axes=(2,), sides=("-",), coordinates=(float(zedges[_PML + 2]),)
            ),
        ]
    )
    objects.append(src)
    return objects, constraints, config


def test_kernel_nonuniform_isotropic_cpml():
    objects, constraints, config = _nonuniform_case(fdtdx.Material(permittivity=2.25), polarization=0)
    arrays, oc, config = _assert_kernel_matches_ops(objects, constraints, config)
    _assert_kernel_matches_jax(arrays, oc, config)


def test_kernel_nonuniform_diagonal_cpml():
    objects, constraints, config = _nonuniform_case(fdtdx.Material(permittivity=(2.25, 4.0, 1.0)), polarization=0)
    arrays, oc, config = _assert_kernel_matches_ops(objects, constraints, config)
    _assert_kernel_matches_jax(arrays, oc, config)


def test_kernel_periodic_cpml():
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_RES), time=16e-15, dtype=jnp.float32)
    objects, constraints = [], []
    vol = fdtdx.SimulationVolume(partial_real_shape=(3 * _RES, 3 * _RES, 40 * _RES))
    objects.append(vol)
    bcfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML,
        override_types={"min_x": "periodic", "max_x": "periodic", "min_y": "periodic", "max_y": "periodic"},
    )
    bdict, clist = fdtdx.boundary_objects_from_config(bcfg, vol)
    constraints.extend(clist)
    objects.extend(bdict.values())
    diel = fdtdx.UniformMaterialObject(
        partial_grid_shape=(None, None, 20), material=fdtdx.Material(permittivity=(2.25, 4.0, 1.0))
    )
    constraints.extend(
        [
            diel.same_size(vol, axes=(0, 1)),
            diel.place_at_center(vol, axes=(0, 1)),
            diel.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(14,)),
        ]
    )
    objects.append(diel)
    src = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=1e-6),
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
    )
    constraints.extend(
        [
            src.same_size(vol, axes=(0, 1)),
            src.place_at_center(vol, axes=(0, 1)),
            src.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_PML + 2,)),
        ]
    )
    objects.append(src)

    arrays, oc, config = _assert_kernel_matches_ops(objects, constraints, config)
    _assert_kernel_matches_jax(arrays, oc, config)
