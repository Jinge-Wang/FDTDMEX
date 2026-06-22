"""Validation for the Phase 5 front-end + agentic IO seam (Scene, SceneModel, the HDF5 trio).

Contract:
* ``SceneModel`` round-trips ``Scene → JSON → SceneModel → JsonSetup → place_objects`` and reproduces
  the same placed objects as the direct path.
* ``sim_init → sim_run(backend="mlx") → sim_postproc`` reproduces a direct ``run_fdtd`` *bit-for-bit*
  (the packed payload is the serialized form of the exact same resolved arrays, fed to the same loop).
* ``sim_run(backend="mock")`` writes a schema-valid ``results.hdf5`` without touching the engine.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx

pytestmark = pytest.mark.validation


def _build_objects():
    """A small Si-strip-in-SiO2 setup with a Gaussian plane source + energy detector (MLX-eligible)."""
    config = fdtdx.SimulationConfig(
        time=12e-15, grid=fdtdx.UniformGrid(spacing=80e-9), dtype=jnp.float32, courant_factor=0.99
    )
    constraints, object_list = [], []
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(2.4e-6, 2.4e-6, 2.4e-6), material=fdtdx.Material(permittivity=2.07)
    )
    object_list.append(volume)
    core = fdtdx.UniformMaterialObject(
        partial_real_shape=(2.4e-6, 0.5e-6, 0.3e-6), material=fdtdx.Material(permittivity=12.1)
    )
    constraints += core.same_position_and_size(volume, axes=(0,))
    constraints.append(core.place_relative_to(volume, axes=(1, 2), own_positions=(0, 0), other_positions=(0, 0)))
    object_list.append(core)
    bc = fdtdx.BoundaryConfig.from_uniform_bound(thickness=6, boundary_type="pml")
    bd, cl = fdtdx.boundary_objects_from_config(bc, volume)
    constraints += cl
    object_list += list(bd.values())
    source = fdtdx.GaussianPlaneSource(
        partial_grid_shape=(1, None, None), partial_real_shape=(None, 1.2e-6, 1.2e-6),
        fixed_E_polarization_vector=(0, 1, 0), wave_character=fdtdx.WaveCharacter(wavelength=1.55e-6),
        radius=0.6e-6, std=1 / 3, direction="+",
    )
    constraints.append(
        source.place_relative_to(volume, axes=(0, 1, 2), own_positions=(-1, 0, 0), other_positions=(-0.6, 0, 0))
    )
    object_list.append(source)
    detector = fdtdx.EnergyDetector(name="energy")
    constraints += detector.same_position_and_size(volume)
    object_list.append(detector)
    return config, object_list, constraints


def _direct_energy(config, object_list, constraints):
    objects, arrays, params, cfg, _ = fdtdx.place_objects(
        object_list=object_list, config=config, constraints=constraints
    )
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params)
    _, result = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=cfg, show_progress=False)
    return np.asarray(result.detector_states["energy"]["energy"])


def test_serialize_roundtrip_exact():
    """The generic run-seam serializer round-trips MLX/numpy/slice/dtype leaves exactly."""
    import mlx.core as mx

    from fdtdx.mlx.detector_freeze import DetectorPlan
    from fdtdx.mlx.serialize import deserialize, serialize
    from fdtdx.mlx.source_freeze import SourcePlan

    sp = SourcePlan(
        kind="tfsf", grid_slice=(slice(0, 1), slice(2, 5), slice(None)),
        on_steps=np.array([True, False, True]), sign=-1.0, h_axis=1, v_axis=2,
        spatialE_h=mx.ones((2, 3)), amp_E_h=mx.zeros((4, 2, 3)),
    )
    dp = DetectorPlan(
        name="d", kind="phasor", buffer_key="phasor", grid_slice=(slice(None),),
        on_steps=np.array([1, 0]), time_to_idx=np.array([0, 1]), exact_interp=True, reduce_volume=False,
        buffer_shapes={"phasor": (1, 2, 3)}, buffer_dtypes={"phasor": mx.complex64},
        phasors=mx.zeros((2, 1), dtype=mx.complex64), component_picks=[("E", 0), ("H", 2)], static_scale=2.0,
    )
    skel, arrs = serialize({"sources": [sp], "detectors": [dp], "n": 10})
    out = deserialize(skel, arrs)
    assert out["n"] == 10
    s2, d2 = out["sources"][0], out["detectors"][0]
    assert s2.grid_slice[1] == slice(2, 5) and s2.h_axis == 1
    assert isinstance(s2.spatialE_h, mx.array) and tuple(s2.spatialE_h.shape) == (2, 3)
    assert d2.buffer_dtypes["phasor"] == mx.complex64 and d2.component_picks == [("E", 0), ("H", 2)]


def test_scene_model_roundtrip():
    """SceneModel ↔ JSON ↔ JsonSetup reproduces the object/constraint structure."""
    from fdtdmex.io import SceneModel

    config, object_list, constraints = _build_objects()
    scene = fdtdx.Scene(config).add(*object_list).constrain(constraints)

    model = scene.to_model()
    model2 = SceneModel.model_validate_json(model.model_dump_json())
    setup = model2.to_json_setup()  # validates internally

    assert [type(o).__name__ for o in setup.object_list] == [type(o).__name__ for o in object_list]
    assert len(setup.constraints) == len(constraints)
    assert sum(1 for o in setup.object_list if type(o).__name__ == "SimulationVolume") == 1


def test_scene_run_matches_direct():
    """Scene.run() reproduces the explicit place → apply → run path element-wise."""
    config, object_list, constraints = _build_objects()
    direct = _direct_energy(config, object_list, constraints)

    scene = fdtdx.Scene(config).add(*object_list).constrain(constraints)
    result = scene.run()
    scene_energy = np.asarray(result.detector_states["energy"]["energy"])
    assert np.max(np.abs(direct - scene_energy)) == 0.0


def test_hdf5_trio_bit_identical(tmp_path):
    """sim_init → sim_run(mlx) → sim_postproc reproduces run_fdtd bit-for-bit; mock is schema-valid."""
    import h5py

    from fdtdmex.io import sim_init, sim_postproc, sim_run

    config, object_list, constraints = _build_objects()
    direct = _direct_energy(config, object_list, constraints)

    scene = fdtdx.Scene(config).add(*object_list).constrain(constraints)
    cfg_h5 = tmp_path / "config.hdf5"
    res_h5 = tmp_path / "results.hdf5"
    sim_init(scene, cfg_h5)
    sim_run(cfg_h5, res_h5, backend="mlx")

    with h5py.File(res_h5, "r") as f:
        packed = np.asarray(f["detector_states"]["energy"]["energy"])
    assert np.max(np.abs(direct - packed)) == 0.0

    reduced = sim_postproc(res_h5)
    assert reduced["backend"] == "mlx" and "energy" in reduced["detectors"]

    # Mock backend: same config file, no engine, schema-valid synthetic results.
    mock_h5 = tmp_path / "results_mock.hdf5"
    sim_run(cfg_h5, mock_h5, backend="mock")
    reduced_mock = sim_postproc(mock_h5)
    assert reduced_mock["backend"] == "mock"
    assert reduced_mock["detectors"]["energy"]["energy"]["shape"] == list(direct.shape)
