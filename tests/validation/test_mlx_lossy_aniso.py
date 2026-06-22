"""Phase 3 item 1: lossy full-anisotropic + 9-tensor conductivity parity (MLX vs JAX-CPU).

These cases route through the MLX-op anisotropic A/B update (``_update_aniso`` ->
``compute_anisotropic_update_matrices_mlx``), which already consumes ``sigma``; the dispatcher gate
was removed in Phase 3. The lossless block-hybrid Metal kernel does not cover lossy media, so these
runs fall back to the MLX-op cores via ``kernel_eligible`` -- correct, just not kernel-accelerated.

Caveat (roadmap "Quirk A"): strongly off-diagonal anisotropy is numerically unstable in the explicit
A/B update in *both* JAX and MLX. Off-diagonals are kept <= 0.5 here; finiteness is asserted to match
both backends rather than loosening the tolerance.
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


def _rel(j, m):
    j, m = np.asarray(j), np.asarray(m)
    return float(np.abs(j - m).max() / (np.abs(j).max() + 1e-30))


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


def _dipole_case(material):
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_RES), time=_TIME, dtype=jnp.float32)
    objects, constraints = [], []
    vol = fdtdx.SimulationVolume(partial_real_shape=(_DOMAIN,) * 3, material=material)
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
    return _run_both(objects, constraints, config)


def _assert_parity(arr_j, arr_m, expect_eps9=False, expect_sigma9=False):
    if expect_eps9:
        assert np.asarray(arr_j.inv_permittivities).shape[0] == 9, "expected a 9-tensor permittivity"
    if expect_sigma9:
        assert np.asarray(arr_j.electric_conductivity).shape[0] == 9, "expected a 9-tensor conductivity"
    for name in ("E", "H"):
        j = np.asarray(getattr(arr_j.fields, name))
        m = np.asarray(getattr(arr_m.fields, name))
        # Finiteness must agree between backends (Quirk A): both finite, or both not.
        assert np.isfinite(j).all() == np.isfinite(m).all(), f"field {name} finiteness mismatch"
        if np.isfinite(j).all():
            assert _rel(j, m) < _RTOL, f"field {name} mismatch"
    assert _rel(arr_j.detector_states["E"]["energy"], arr_m.detector_states["E"]["energy"]) < _RTOL


def test_lossy_full_aniso_matches_jax():
    """9-tensor (off-diagonal) permittivity + isotropic electric conductivity."""
    eps_tensor = ((2.5, 0.5, 0.0), (0.5, 3.0, 0.0), (0.0, 0.0, 4.0))  # symmetric, positive-definite
    material = fdtdx.Material(permittivity=eps_tensor, electric_conductivity=0.05)
    arr_j, arr_m = _dipole_case(material)
    _assert_parity(arr_j, arr_m, expect_eps9=True)


def test_full_tensor_conductivity_matches_jax():
    """Diagonal permittivity + symmetric 9-tensor electric conductivity."""
    sigma_tensor = ((0.05, 0.01, 0.0), (0.01, 0.05, 0.0), (0.0, 0.0, 0.05))  # symmetric, small off-diagonal
    material = fdtdx.Material(permittivity=(2.0, 3.0, 4.0), electric_conductivity=sigma_tensor)
    arr_j, arr_m = _dipole_case(material)
    _assert_parity(arr_j, arr_m, expect_sigma9=True)
