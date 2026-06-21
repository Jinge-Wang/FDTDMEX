"""Two-backend element-wise parity: the MLX (Metal) forward loop vs the JAX-CPU oracle.

Runs the *same* placed simulation through both backends on one Apple-Silicon machine via
``fdtdx.use_backend`` and asserts the final fields and detector_states agree to float32
tolerance. JAX-Metal is unusable, so the forced-JAX run is the CPU oracle (conftest pins
``JAX_PLATFORMS=cpu``). Skipped off Apple Silicon / without mlx.

Covers the M1 surface: vacuum/isotropic + CPML + point dipole + EnergyDetector
(reduce_volume and full-volume), plus AUTO routing engaging MLX.
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
_RTOL = 1e-3


def _build():
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_RES), time=_TIME, dtype=jnp.float32)
    objects, constraints = [], []
    vol = fdtdx.SimulationVolume(partial_real_shape=(_DOMAIN,) * 3)
    objects.append(vol)
    bdict, clist = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML), vol)
    constraints.extend(clist)
    objects.extend(bdict.values())
    src = fdtdx.PointDipoleSource(
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=1e-6),
        polarization=2,
        amplitude=1.0,
    )
    constraints.append(src.place_at_center(vol, axes=(0, 1, 2)))
    objects.append(src)
    for name, reduce_volume in (("energy", True), ("energy_full", False)):
        det = fdtdx.EnergyDetector(name=name, reduce_volume=reduce_volume, plot=False)
        constraints.extend([det.same_size(vol, axes=(0, 1, 2)), det.place_at_center(vol, axes=(0, 1, 2))])
        objects.append(det)
    return objects, constraints, config


def _placed():
    key = jax.random.PRNGKey(0)
    objects, constraints, config = _build()
    oc, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, oc, _ = fdtdx.apply_params(arrays, oc, params, key)
    return arrays, oc, config, key


def _rel(j, m):
    j, m = np.asarray(j), np.asarray(m)
    return float(np.abs(j - m).max() / (np.abs(j).max() + 1e-30))


def test_mlx_matches_jax_cpu():
    arrays, oc, config, key = _placed()
    with fdtdx.use_backend("jax"):
        _, arr_j = fdtdx.run_fdtd(arrays=arrays, objects=oc, config=config, key=key, show_progress=False)
    with fdtdx.use_backend("mlx"):
        _, arr_m = fdtdx.run_fdtd(arrays=arrays, objects=oc, config=config, key=key, show_progress=False)

    for name in ("E", "H", "psi_E", "psi_H"):
        assert _rel(getattr(arr_j.fields, name), getattr(arr_m.fields, name)) < _RTOL, f"field {name} mismatch"

    for det in ("energy", "energy_full"):
        assert _rel(arr_j.detector_states[det]["energy"], arr_m.detector_states[det]["energy"]) < _RTOL, (
            f"detector {det} mismatch"
        )


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


def test_field_and_poynting_detectors_match_jax():
    """FieldDetector + PoyntingFluxDetector (offset plane, real flux) parity."""
    n, c = 32, 16
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_RES), time=16e-15, dtype=jnp.float32)
    objects, constraints = [], []
    vol = fdtdx.SimulationVolume(partial_real_shape=(n * _RES,) * 3)
    objects.append(vol)
    bdict, clist = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML), vol)
    constraints.extend(clist)
    objects.extend(bdict.values())
    src = fdtdx.PointDipoleSource(
        partial_grid_shape=(1, 1, 1), wave_character=fdtdx.WaveCharacter(wavelength=1e-6), polarization=0
    )
    constraints.append(src.place_at_center(vol, axes=(0, 1, 2)))
    objects.append(src)

    fdet = fdtdx.FieldDetector(name="F", reduce_volume=True, plot=False)
    constraints.extend([fdet.same_size(vol, axes=(0, 1, 2)), fdet.place_at_center(vol, axes=(0, 1, 2))])
    objects.append(fdet)
    for name, reduce_volume in (("PF", True), ("PFm", False)):
        pf = fdtdx.PoyntingFluxDetector(
            name=name, partial_grid_shape=(None, None, 1), direction="+", reduce_volume=reduce_volume, plot=False
        )
        constraints.extend(
            [
                pf.same_size(vol, axes=(0, 1)),
                pf.place_at_center(vol, axes=(0, 1)),
                pf.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(c + 5,)),
            ]
        )
        objects.append(pf)

    arr_j, arr_m = _run_both(objects, constraints, config)
    assert _rel(arr_j.detector_states["F"]["fields"], arr_m.detector_states["F"]["fields"]) < _RTOL
    for name in ("PF", "PFm"):
        assert _rel(arr_j.detector_states[name]["poynting_flux"], arr_m.detector_states[name]["poynting_flux"]) < _RTOL


def test_conductive_material_matches_jax():
    """Lossy (iso electric-conductivity) material parity."""
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_RES), time=_TIME, dtype=jnp.float32)
    objects, constraints = [], []
    mat = fdtdx.Material(permittivity=2.0, electric_conductivity=0.05)
    vol = fdtdx.SimulationVolume(partial_real_shape=(_DOMAIN,) * 3, material=mat)
    objects.append(vol)
    bdict, clist = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML), vol)
    constraints.extend(clist)
    objects.extend(bdict.values())
    src = fdtdx.PointDipoleSource(
        partial_grid_shape=(1, 1, 1), wave_character=fdtdx.WaveCharacter(wavelength=1e-6), polarization=2
    )
    constraints.append(src.place_at_center(vol, axes=(0, 1, 2)))
    objects.append(src)

    arr_j, arr_m = _run_both(objects, constraints, config)
    for name in ("E", "H"):
        assert _rel(getattr(arr_j.fields, name), getattr(arr_m.fields, name)) < _RTOL


def _plane_case(source):
    n, pml = 32, 8
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=100e-9), time=30e-15, dtype=jnp.float32)
    objects, constraints = [], []
    vol = fdtdx.SimulationVolume(partial_real_shape=(n * 100e-9,) * 3)
    objects.append(vol)
    bdict, clist = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=pml), vol)
    constraints.extend(clist)
    objects.extend(bdict.values())
    constraints.extend(
        [
            source.same_size(vol, axes=(0, 1)),
            source.place_at_center(vol, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(pml + 2,)),
        ]
    )
    objects.append(source)
    pf = fdtdx.PoyntingFluxDetector(
        name="PF", partial_grid_shape=(None, None, 1), direction="+", reduce_volume=True, plot=False
    )
    constraints.extend(
        [
            pf.same_size(vol, axes=(0, 1)),
            pf.place_at_center(vol, axes=(0, 1)),
            pf.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(n // 2 + 4,)),
        ]
    )
    objects.append(pf)
    return _run_both(objects, constraints, config)


def test_uniform_plane_source_matches_jax():
    """UniformPlaneSource (TFSF) plane wave + downstream flux parity."""
    arr_j, arr_m = _plane_case(
        fdtdx.UniformPlaneSource(
            partial_grid_shape=(None, None, 1),
            fixed_E_polarization_vector=(1, 0, 0),
            wave_character=fdtdx.WaveCharacter(wavelength=1.55e-6),
            direction="+",
        )
    )
    assert _rel(arr_j.fields.E, arr_m.fields.E) < _RTOL
    assert _rel(arr_j.fields.H, arr_m.fields.H) < _RTOL
    assert _rel(arr_j.detector_states["PF"]["poynting_flux"], arr_m.detector_states["PF"]["poynting_flux"]) < _RTOL


def test_gaussian_plane_source_matches_jax():
    """GaussianPlaneSource (TFSF, non-uniform transverse profile) parity."""
    arr_j, arr_m = _plane_case(
        fdtdx.GaussianPlaneSource(
            partial_grid_shape=(None, None, 1),
            fixed_E_polarization_vector=(1, 0, 0),
            wave_character=fdtdx.WaveCharacter(wavelength=1.55e-6),
            direction="+",
            radius=1.2e-6,
            std=1 / 3,
        )
    )
    assert _rel(arr_j.fields.E, arr_m.fields.E) < _RTOL
    assert _rel(arr_j.detector_states["PF"]["poynting_flux"], arr_m.detector_states["PF"]["poynting_flux"]) < _RTOL


def test_phasor_detector_matches_jax():
    """PhasorDetector (complex running DFT), reduce + full, on a plane wave."""
    wc = fdtdx.WaveCharacter(wavelength=1.55e-6)
    n, pml = 32, 8
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=100e-9), time=30e-15, dtype=jnp.float32)
    objects, constraints = [], []
    vol = fdtdx.SimulationVolume(partial_real_shape=(n * 100e-9,) * 3)
    objects.append(vol)
    bdict, clist = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=pml), vol)
    constraints.extend(clist)
    objects.extend(bdict.values())
    src = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        fixed_E_polarization_vector=(1, 0, 0),
        wave_character=wc,
        direction="+",
    )
    constraints.extend(
        [
            src.same_size(vol, axes=(0, 1)),
            src.place_at_center(vol, axes=(0, 1)),
            src.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(pml + 2,)),
        ]
    )
    objects.append(src)
    ph = fdtdx.PhasorDetector(name="ph", wave_characters=(wc,), components=("Ex", "Hy"), reduce_volume=True)
    constraints.extend(
        [
            ph.same_size(vol, axes=(0, 1)),
            ph.place_at_center(vol, axes=(0, 1)),
            ph.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(n // 2,)),
        ]
    )
    objects.append(ph)
    phf = fdtdx.PhasorDetector(name="phf", wave_characters=(wc,), components=("Ex", "Hy"), reduce_volume=False)
    constraints.extend([phf.same_size(vol, axes=(0, 1, 2)), phf.place_at_center(vol, axes=(0, 1, 2))])
    objects.append(phf)

    arr_j, arr_m = _run_both(objects, constraints, config)
    assert np.asarray(arr_j.detector_states["ph"]["phasor"]).dtype == np.complex64
    assert _rel(arr_j.detector_states["ph"]["phasor"], arr_m.detector_states["ph"]["phasor"]) < _RTOL
    assert _rel(arr_j.detector_states["phf"]["phasor"], arr_m.detector_states["phf"]["phasor"]) < _RTOL


def test_diagonal_anisotropy_matches_jax():
    """Diagonal-anisotropic permittivity (3-tensor) with a dipole source."""
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

    arr_j, arr_m = _run_both(objects, constraints, config)
    for name in ("E", "H"):
        assert _rel(getattr(arr_j.fields, name), getattr(arr_m.fields, name)) < _RTOL


def test_periodic_boundaries_match_jax():
    """Periodic (wrap-padded) x/y boundaries + plane wave through a diagonal-anisotropic slab."""
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
    wave = fdtdx.WaveCharacter(wavelength=1e-6)
    src = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1), wave_character=wave, direction="+", fixed_E_polarization_vector=(1, 0, 0)
    )
    constraints.extend(
        [
            src.same_size(vol, axes=(0, 1)),
            src.place_at_center(vol, axes=(0, 1)),
            src.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_PML + 2,)),
        ]
    )
    objects.append(src)
    det = fdtdx.PhasorDetector(
        name="ph",
        partial_grid_shape=(None, None, 1),
        wave_characters=(wave,),
        reduce_volume=True,
        components=("Ex", "Hy"),
    )
    constraints.extend(
        [
            det.same_size(vol, axes=(0, 1)),
            det.place_at_center(vol, axes=(0, 1)),
            det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(20,)),
        ]
    )
    objects.append(det)

    arr_j, arr_m = _run_both(objects, constraints, config)
    for name in ("E", "H"):
        assert _rel(getattr(arr_j.fields, name), getattr(arr_m.fields, name)) < _RTOL
    assert _rel(arr_j.detector_states["ph"]["phasor"], arr_m.detector_states["ph"]["phasor"]) < _RTOL


def test_full_anisotropy_matches_jax():
    """Full-tensor (9-component, off-diagonal) permittivity with a dipole source."""
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_RES), time=_TIME, dtype=jnp.float32)
    objects, constraints = [], []
    eps_tensor = ((2.5, 0.5, 0.0), (0.5, 3.0, 0.0), (0.0, 0.0, 4.0))  # symmetric, positive-definite
    vol = fdtdx.SimulationVolume(partial_real_shape=(_DOMAIN,) * 3, material=fdtdx.Material(permittivity=eps_tensor))
    objects.append(vol)
    bdict, clist = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML), vol)
    constraints.extend(clist)
    objects.extend(bdict.values())
    src = fdtdx.PointDipoleSource(
        partial_grid_shape=(1, 1, 1), wave_character=fdtdx.WaveCharacter(wavelength=1e-6), polarization=0
    )
    constraints.append(src.place_at_center(vol, axes=(0, 1, 2)))
    objects.append(src)
    det = fdtdx.EnergyDetector(name="E", reduce_volume=True, plot=False)
    constraints.extend([det.same_size(vol, axes=(0, 1, 2)), det.place_at_center(vol, axes=(0, 1, 2))])
    objects.append(det)

    arr_j, arr_m = _run_both(objects, constraints, config)
    assert np.asarray(arr_j.inv_permittivities).shape[0] == 9
    for name in ("E", "H"):
        assert _rel(getattr(arr_j.fields, name), getattr(arr_m.fields, name)) < _RTOL, f"field {name}"
    assert _rel(arr_j.detector_states["E"]["energy"], arr_m.detector_states["E"]["energy"]) < _RTOL


def test_auto_routes_to_mlx_on_apple_silicon():
    """With no override, a supported forward run auto-routes to MLX (== forced MLX)."""
    arrays, oc, config, key = _placed()
    _, arr_auto = fdtdx.run_fdtd(arrays=arrays, objects=oc, config=config, key=key, show_progress=False)
    with fdtdx.use_backend("mlx"):
        _, arr_mlx = fdtdx.run_fdtd(arrays=arrays, objects=oc, config=config, key=key, show_progress=False)
    assert np.array_equal(np.asarray(arr_auto.fields.E), np.asarray(arr_mlx.fields.E))
    assert np.array_equal(
        np.asarray(arr_auto.detector_states["energy"]["energy"]),
        np.asarray(arr_mlx.detector_states["energy"]["energy"]),
    )
