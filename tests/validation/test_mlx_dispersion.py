"""Phase 3 item 3: Drude-Lorentz (ADE) dispersion parity (MLX vs JAX-CPU, kernel vs MLX-op cores).

Dispersion threads a polarization ``P`` through the E-side of the loop. fdtdx forbids dispersion with
off-diagonal tensors, so it is always iso/diagonal: a lossless dispersive run rides the Metal
E-kernel's ADE fold; lossy+dispersive falls back to the MLX-op cores (``kernel_eligible``).

Three checks:
- **Kernel-core vs MLX-op-core (lossless), rel < 1e-4** — ``run_forward_mlx`` twice on identical fresh
  state, kernel off/on; isolates the ADE kernel fold from the (JAX-validated) MLX-op ADE. Asserts the
  kernel path actually engaged (``kernels.KERNEL_CORES_BUILT``).
- **MLX vs forced-JAX oracle, rel < 1e-3** — the end-to-end physics bar (runs under whatever
  ``FDTDMEX_METAL_KERNEL`` mode the suite is invoked with).
- **Non-dispersive kernel source unchanged** — guards that the ADE codegen never perturbs the tuned
  non-dispersive E-kernel MSL (the byte-identical guarantee).

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
_DOMAIN = 24 * _RES
_TIME = 12e-15
_RTOL = 1e-3
_KTOL = 1e-4

# Stable poles at this dt (gamma*dt << 2). Lorentz: a mid-IR/visible resonance with modest strength;
# Drude: a free-carrier (metal-like) pole. Parameters mirror tests/unit/test_dispersion.py.
_LORENTZ = fdtdx.LorentzPole(resonance_frequency=1e15, damping=1e13, delta_epsilon=2.0)
_DRUDE = fdtdx.DrudePole(plasma_frequency=1e16, damping=1e14)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / (np.abs(a).max() + 1e-30))


def _build(material):
    """Whole-volume dispersive medium with a central dipole and a volume energy detector."""
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
    return objects, constraints, config


def _placed(material):
    key = jax.random.PRNGKey(0)
    objects, constraints, config = _build(material)
    oc, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, oc, _ = fdtdx.apply_params(arrays, oc, params, key)
    return arrays, oc, config, key


def _run_both(material):
    arrays, oc, config, key = _placed(material)
    # Sanity: the placed container actually carries dispersive coefficients (else we'd test nothing).
    assert arrays.dispersive_c1 is not None, "expected a dispersive material to populate coefficients"
    with fdtdx.use_backend("jax"):
        _, arr_j = fdtdx.run_fdtd(arrays=arrays, objects=oc, config=config, key=key, show_progress=False)
    with fdtdx.use_backend("mlx"):
        _, arr_m = fdtdx.run_fdtd(arrays=arrays, objects=oc, config=config, key=key, show_progress=False)
    return arr_j, arr_m


def _mlx_out(arrays, oc, config, use_metal_kernel):
    """Replicate dispatch._run_mlx_forward with an explicit kernel flag; return the container."""
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
    state, detector_buffers = run_forward_mlx(
        state,
        source_plans,
        detector_plans,
        detector_buffers,
        int(config.time_steps_total),
        float(config.courant_number),
        simulate_boundaries=True,
        use_metal_kernel=use_metal_kernel,
    )
    detector_states = buffers_to_detector_states(detector_buffers) if detector_plans else None
    return to_array_container(arr, state, detector_states)


def _assert_parity(arr_j, arr_m):
    for name in ("E", "H"):
        j = np.asarray(getattr(arr_j.fields, name))
        m = np.asarray(getattr(arr_m.fields, name))
        assert np.isfinite(j).all() and np.isfinite(m).all(), f"field {name} not finite"
        assert _rel(j, m) < _RTOL, f"field {name} mismatch ({_rel(j, m):.2e})"
    # Polarization state, written back by the bridge, must also match the JAX engine.
    for name in ("dispersive_P_curr", "dispersive_P_prev"):
        assert _rel(getattr(arr_j, name), getattr(arr_m, name)) < _RTOL, f"{name} mismatch"
    assert _rel(arr_j.detector_states["E"]["energy"], arr_m.detector_states["E"]["energy"]) < _RTOL


def _disp(poles, permittivity=1.5, **kw):
    return fdtdx.Material(permittivity=permittivity, dispersion=fdtdx.DispersionModel(poles=poles), **kw)


_LOSSLESS_CASES = {
    "lorentz_iso": _disp((_LORENTZ,)),
    "drude_iso": _disp((_DRUDE,)),
    "two_pole_iso": _disp((_LORENTZ, _DRUDE)),
    "lorentz_diag": _disp((_LORENTZ,), permittivity=(1.5, 2.0, 2.5)),
}


@pytest.mark.parametrize("name", list(_LOSSLESS_CASES))
def test_kernel_ade_matches_ops(name):
    """Lossless dispersive: the Metal E-kernel ADE fold must match the MLX-op ADE bit-for-bit (rel <
    1e-4), and the kernel path must actually engage (no silent fallback)."""
    import fdtdx.mlx.kernels as kernels

    arrays, oc, config, _ = _placed(_LOSSLESS_CASES[name])
    before = kernels.KERNEL_CORES_BUILT
    out_ops = _mlx_out(arrays, oc, config, use_metal_kernel=False)
    out_ker = _mlx_out(arrays, oc, config, use_metal_kernel=True)
    assert kernels.KERNEL_CORES_BUILT == before + 1, "ADE kernel path did not engage (fell back to MLX ops)"
    for field in ("E", "H"):
        assert _rel(getattr(out_ops.fields, field), getattr(out_ker.fields, field)) < _KTOL, field
    assert _rel(out_ops.dispersive_P_curr, out_ker.dispersive_P_curr) < _KTOL, "P_curr kernel vs ops"
    assert _rel(out_ops.detector_states["E"]["energy"], out_ker.detector_states["E"]["energy"]) < _KTOL


@pytest.mark.parametrize("name", list(_LOSSLESS_CASES))
def test_dispersion_matches_jax(name):
    """End-to-end physics bar vs the forced-JAX oracle (rel < 1e-3)."""
    _assert_parity(*_run_both(_LOSSLESS_CASES[name]))


def test_lossy_dispersion_matches_jax():
    """Conductivity + dispersion together: kernel-ineligible (sigma) → MLX-op ADE cores vs JAX."""
    mat = _disp((_LORENTZ,), electric_conductivity=0.05)
    _assert_parity(*_run_both(mat))


def test_nondispersive_kernel_source_unchanged():
    """Regression guard: the non-dispersive E-kernel MSL must carry the plain output writes and none
    of the ADE codegen, so the tuned non-dispersive path stays byte-identical."""
    from fdtdx.mlx.kernels import _field_source

    shape, per, ext = (16, 16, 16), (False, False, False), ((0, 0), (0, 0), (0, 0))
    src = _field_source(shape, per, False, None, ext, False, [], forward=False)
    assert "out[idx]       = E[idx]       + cbx*cx;" in src
    for token in ("Pco", "Ppo", "dltx", "dc1"):
        assert token not in src, f"ADE token {token!r} leaked into the non-dispersive kernel"
    # Default (no dispersive args) must equal an explicit dispersive=False build.
    assert src == _field_source(shape, per, False, None, ext, False, [], forward=False, dispersive=False)
