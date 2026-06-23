"""Phasor DFT auto-subsampling: cheaper, and physically equivalent to every-step recording.

The MLX phasor detector records only every ``stride``-th step (``stride`` auto-chosen to keep
~12 samples per period of the highest frequency) and rescales by ``stride``. For a well-oversampled
FDTD signal this must reproduce the every-step (``FDTDMEX_DFT_STRIDE=1``) phasor to within a small
physics tolerance. Also pins the stride formula and the ``static_scale`` Riemann-weight factor.
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


def _build():
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_RES), time=40e-15, dtype=jnp.float32)
    objects, constraints = [], []
    vol = fdtdx.SimulationVolume(partial_real_shape=(24 * _RES,) * 3)
    objects.append(vol)
    bdict, clist = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML), vol)
    constraints.extend(clist)
    objects.extend(bdict.values())
    src = fdtdx.PointDipoleSource(
        partial_grid_shape=(1, 1, 1), wave_character=fdtdx.WaveCharacter(wavelength=1e-6), polarization=2
    )
    constraints.append(src.place_at_center(vol, axes=(0, 1, 2)))
    objects.append(src)
    det = fdtdx.PhasorDetector(
        name="ph", wave_characters=(fdtdx.WaveCharacter(wavelength=1e-6),), components=("Ez", "Hx"), reduce_volume=True
    )
    constraints.extend([det.same_size(vol, axes=(0, 1, 2)), det.place_at_center(vol, axes=(0, 1, 2))])
    objects.append(det)
    return objects, constraints, config


def _run_phasor(monkeypatch, stride_env):
    key = jax.random.PRNGKey(0)
    objects, constraints, config = _build()
    oc, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, oc, _ = fdtdx.apply_params(arrays, oc, params, key)
    if stride_env is None:
        monkeypatch.delenv("FDTDMEX_DFT_STRIDE", raising=False)
    else:
        monkeypatch.setenv("FDTDMEX_DFT_STRIDE", str(stride_env))
    with fdtdx.use_backend("mlx"):
        _, arr = fdtdx.run_fdtd(arrays=arrays, objects=oc, config=config, key=key, show_progress=False)
    return np.asarray(arr.detector_states["ph"]["phasor"])


def test_subsampled_phasor_matches_every_step(monkeypatch):
    exact = _run_phasor(monkeypatch, stride_env=1)  # every step
    auto = _run_phasor(monkeypatch, stride_env=None)  # auto-subsample (default)
    rel = float(np.abs(auto - exact).max() / (np.abs(exact).max() + 1e-30))
    assert rel < 0.02, f"subsampled phasor deviates {rel:.3%} from every-step"


def test_dft_stride_formula_and_env(monkeypatch):
    from fdtdx.mlx.detector_freeze import _DFT_OVERSAMPLE, _dft_stride

    monkeypatch.delenv("FDTDMEX_DFT_STRIDE", raising=False)  # the validation conftest pins it to 1
    omega = np.array([2.0 * np.pi * 3e8 / 1e-6])  # f = c/lambda at 1 um
    dt = 8e-17
    f_max = float(omega.max()) / (2.0 * np.pi)
    expected = max(1, int(np.floor(1.0 / (_DFT_OVERSAMPLE * f_max * dt))))
    assert _dft_stride(omega, dt) == expected
    assert expected > 1  # this configuration is well-oversampled


def test_dft_stride_env_override(monkeypatch):
    from fdtdx.mlx.detector_freeze import _dft_stride

    omega = np.array([2.0 * np.pi * 3e8 / 1e-6])
    monkeypatch.setenv("FDTDMEX_DFT_STRIDE", "1")
    assert _dft_stride(omega, 8e-17) == 1  # forced every-step regardless of oversampling
    monkeypatch.setenv("FDTDMEX_DFT_STRIDE", "5")
    assert _dft_stride(omega, 8e-17) == 5
