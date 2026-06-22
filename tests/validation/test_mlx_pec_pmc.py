"""Phase 3 item 2: PEC/PMC boundary parity (MLX vs JAX-CPU).

PEC zeros tangential E after each E-update; PMC zeros tangential H after each H-update. The MLX
backend freezes these into multiplicative keep-masks (``fdtdx.mlx.boundary_mask``) applied in the
loop after source injection, matching fdtdx's ``apply_boundary_post_E/H_update`` ordering. Besides
element-wise parity vs the JAX oracle, we assert the tangential components are *exactly* zero on the
boundary face in the MLX result (the physics the boundary enforces).
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


def _run(override_types):
    """Box with a centered dipole and the given boundary override; return (arr_j, arr_m, oc)."""
    key = jax.random.PRNGKey(0)
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_RES), time=_TIME, dtype=jnp.float32)
    objects, constraints = [], []
    vol = fdtdx.SimulationVolume(partial_real_shape=(_DOMAIN,) * 3)
    objects.append(vol)
    bcfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML, override_types=override_types)
    bdict, clist = fdtdx.boundary_objects_from_config(bcfg, vol)
    constraints.extend(clist)
    objects.extend(bdict.values())
    src = fdtdx.PointDipoleSource(
        partial_grid_shape=(1, 1, 1), wave_character=fdtdx.WaveCharacter(wavelength=1e-6), polarization=2
    )
    constraints.append(src.place_at_center(vol, axes=(0, 1, 2)))
    objects.append(src)

    oc, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, oc, _ = fdtdx.apply_params(arrays, oc, params, key)
    with fdtdx.use_backend("jax"):
        _, arr_j = fdtdx.run_fdtd(arrays=arrays, objects=oc, config=config, key=key, show_progress=False)
    with fdtdx.use_backend("mlx"):
        _, arr_m = fdtdx.run_fdtd(arrays=arrays, objects=oc, config=config, key=key, show_progress=False)
    return arr_j, arr_m, oc


def _assert_field_parity(arr_j, arr_m):
    for name in ("E", "H"):
        assert _rel(getattr(arr_j.fields, name), getattr(arr_m.fields, name)) < _RTOL, f"field {name} mismatch"


_ALL_FACES = ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")


def test_pec_matches_jax():
    """PEC cavity (all faces) + dipole: parity and tangential E exactly zero on every PEC face."""
    arr_j, arr_m, oc = _run(dict.fromkeys(_ALL_FACES, "pec"))
    _assert_field_parity(arr_j, arr_m)
    E = np.asarray(arr_m.fields.E)
    assert oc.pec_objects, "expected PEC boundaries to be present"
    for pec in oc.pec_objects:
        c1, c2 = pec.tangential_components
        for comp in (c1, c2):
            assert np.abs(E[comp][pec.grid_slice]).max() == 0.0, (
                f"tangential E[{comp}] not zero on {pec.descriptive_name}"
            )


def test_pmc_matches_jax():
    """PMC cavity (all faces) + dipole: parity and tangential H exactly zero on every PMC face."""
    arr_j, arr_m, oc = _run(dict.fromkeys(_ALL_FACES, "pmc"))
    _assert_field_parity(arr_j, arr_m)
    H = np.asarray(arr_m.fields.H)
    assert oc.pmc_objects, "expected PMC boundaries to be present"
    for pmc in oc.pmc_objects:
        c1, c2 = pmc.tangential_components
        for comp in (c1, c2):
            assert np.abs(H[comp][pmc.grid_slice]).max() == 0.0, (
                f"tangential H[{comp}] not zero on {pmc.descriptive_name}"
            )


def test_pec_cpml_mix_matches_jax():
    """PEC on min_x/max_x, CPML (pml) on the other four faces -- mixed-boundary stress test."""
    arr_j, arr_m, oc = _run({"min_x": "pec", "max_x": "pec"})
    _assert_field_parity(arr_j, arr_m)
    E = np.asarray(arr_m.fields.E)
    for pec in oc.pec_objects:
        c1, c2 = pec.tangential_components
        for comp in (c1, c2):
            assert np.abs(E[comp][pec.grid_slice]).max() == 0.0
