"""Validation for the Phase 5 front-end + agentic IO seam (Scene, SceneModel, the HDF5 trio).

Contract:
* ``SceneModel`` round-trips ``Scene → JSON → SceneModel → JsonSetup → place_objects`` and reproduces
  the same placed objects as the direct path.
* ``sim_init → sim_run(backend="mlx") → sim_postproc`` reproduces a direct ``run_fdtd`` *bit-for-bit*
  (the packed payload is the serialized form of the exact same resolved arrays, fed to the same loop).
* ``sim_run(backend="mock")`` writes a schema-valid ``results.hdf5`` without touching the engine.
* ``pack → run_simulation_from_hdf5(backend="mock")`` returns immediately, stages a job folder, and
  the detached child advances ``status.json`` ``queued → completed`` + writes ``outputs/result.hdf5``.
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


def _wait_status(status_path, want=("completed", "failed"), timeout=30.0):
    """Poll a job's status.json until it reaches a terminal state (or timeout)."""
    import json
    import time

    deadline = time.time() + timeout
    status = {}
    while time.time() < deadline:
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text())
            except json.JSONDecodeError:
                status = {}  # mid-write; the next atomic replace will land
            if status.get("status") in want:
                return status
        time.sleep(0.05)
    return status


def test_pack_and_launch_mock(tmp_path):
    """pack → run_simulation_from_hdf5(mock): non-blocking, stages job folder, advances status.json."""
    import time

    from fdtdmex.io import pack, run_simulation_from_hdf5, sim_postproc

    config, object_list, constraints = _build_objects()
    scene = fdtdx.Scene(config).add(*object_list).constrain(constraints)

    project = tmp_path / "project"
    bundle = pack(scene, project)
    assert bundle.hdf5_path.exists() and bundle.hdf5_path.parent == project
    assert bundle.config_path is not None and bundle.config_path.exists()
    # Content-addressed + idempotent: re-packing the same scene reuses the name.
    assert pack(scene, project).hdf5_path == bundle.hdf5_path

    jobs = tmp_path / "jobs"
    t0 = time.time()
    handle = run_simulation_from_hdf5(bundle, jobs, backend="mock", name="cold")
    assert time.time() - t0 < 5.0  # returns ~immediately (generous bound for CI)

    # Job folder contract: status.json present at once; bundle + config snapshot staged.
    assert handle.status_path.exists()  # initial "queued" written synchronously
    assert handle.job_dir.parent == jobs
    assert handle.bundle_hdf5.exists()
    assert (handle.job_dir / "config.json").exists()
    assert (handle.job_dir / "outputs").is_dir()

    status = _wait_status(handle.status_path)
    assert status["status"] == "completed", status
    assert status["run_id"] == handle.run_id and status["solver"] == "fdtdmex"
    assert handle.results_path.exists()

    reduced = sim_postproc(handle.results_path)
    assert reduced["backend"] == "mock" and "energy" in reduced["detectors"]

    # A second launch on the same bundle ⇒ a distinct job folder + run id.
    handle2 = run_simulation_from_hdf5(bundle, jobs, backend="mock")
    assert handle2.job_dir != handle.job_dir and handle2.run_id != handle.run_id
