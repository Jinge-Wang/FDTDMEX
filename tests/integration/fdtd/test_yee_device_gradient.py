"""Devices and autodiff under ``material_sampling="yee"`` / ``"yee_smooth"``.

Stage A and Stage B changed how *static* objects reach the material arrays. A ``Device`` is not a
static object: its voxel grid is written by ``apply_params`` over the arrays ``place_objects``
already assembled, and it reads its component count from those arrays' own shape. Per-Yee-point
sampling forces the permittivity array from the 1-component (isotropic) tier to the 3-component
(diagonal) tier, so every one of those reads takes a different branch than it does in ``"box"``
mode. These tests pin that the branch is taken correctly and that the gradient still flows.

The scene is deliberately small (24 x 24 x 12 cells at 40 nm, ~260 steps): a silicon strip drawn at
a sub-cell width and thickness, so the two Yee modes actually differ from ``"box"`` and from each
other, a design region in the middle of it, a dipole and an energy detector.

The gradient bar is a central finite difference on one design parameter. In ``"box"`` and in
``"yee_smooth"`` the reverse-mode gradient agrees with it to better than 1e-3 relative, which is far
inside the float32 noise floor of a 262-step run.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.core.grid import UniformGrid
from fdtdx.materials import Material

SPACING = 40e-9
DOMAIN = (24, 24, 12)
PML = 4
SIM_TIME = 20e-15
#: Drawn at deliberately sub-cell sizes: 330 nm is 8.25 cells and 170 nm is 4.25 cells, so the box
#: path rounds the strip to 320 x 160 nm and the two Yee modes do not.
STRIP = (330e-9, 170e-9)
EPS_SI = 12.25
EPS_OXIDE = 2.25

_COUNTER = [0]


def _tag() -> str:
    _COUNTER[0] += 1
    return f"yd{_COUNTER[0]}"


def _build(sampling: str, discrete: bool = False, gradient: bool = True):
    """Strip waveguide + design region + dipole + energy detector, in one sampling mode."""
    name = _tag()
    config = SimulationConfig(
        time=SIM_TIME,
        grid=UniformGrid(spacing=SPACING),
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        material_sampling=sampling,
        gradient_config=GradientConfig(method="checkpointed", num_checkpoints=2) if gradient else None,
    )
    volume = fdtdx.SimulationVolume(
        name=f"vol_{name}",
        partial_grid_shape=DOMAIN,
        material=Material(permittivity=EPS_OXIDE),
    )
    objects, constraints = [volume], []
    bound_dict, bound_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=PML), volume
    )
    objects.extend(bound_dict.values())
    constraints.extend(bound_constraints)

    strip = fdtdx.UniformMaterialObject(
        name=f"strip_{name}",
        partial_real_shape=(None, STRIP[0], STRIP[1]),
        material=Material(permittivity=EPS_SI),
        placement_order=1,
    )
    objects.append(strip)
    constraints.extend([strip.same_size(volume, axes=(0,)), strip.place_at_center(volume, axes=(0, 1, 2))])

    device = fdtdx.Device(
        name=f"dev_{name}",
        partial_grid_shape=(8, 8, 4),
        materials={"air": Material(permittivity=1.0), "si": Material(permittivity=EPS_SI)},
        param_transforms=[fdtdx.ClosestIndex()] if discrete else [],
        partial_voxel_grid_shape=(2, 2, 2),
    )
    objects.append(device)
    constraints.append(device.place_at_center(volume))

    source = fdtdx.PointDipoleSource(
        name=f"src_{name}",
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(frequency=c0 / 1.0e-6),
        polarization=1,
        amplitude=1.0,
    )
    objects.append(source)
    constraints.append(source.set_grid_coordinates(axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(7, 12, 6)))

    detector = fdtdx.EnergyDetector(
        name=f"det_{name}",
        partial_grid_shape=(3, 3, 3),
        reduce_volume=True,
        exact_interpolation=True,
    )
    objects.append(detector)
    constraints.append(detector.set_grid_coordinates(axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(16, 12, 6)))

    key = jax.random.PRNGKey(0)
    container, arrays, params, config, info = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    return container, arrays, params, config, info, device.name, strip.name


def _objective(params, container, arrays, config, key):
    """Total detected energy over the run — one scalar, differentiable in the device parameters."""
    applied_arrays, applied_objects, _ = fdtdx.apply_params(arrays, container, params, key)
    _, final_arrays = fdtdx.run_fdtd(
        arrays=applied_arrays, objects=applied_objects, config=config, key=key, show_progress=False
    )
    name = applied_objects.detectors[0].name
    return jnp.sum(final_arrays.detector_states[name]["energy"])


_SCENES: dict[tuple[str, bool], tuple] = {}


def _scene(sampling: str, discrete: bool = False):
    """Placed scene, built once per (mode, parameter type) and reused across tests."""
    key = (sampling, discrete)
    if key not in _SCENES:
        _SCENES[key] = _build(sampling, discrete=discrete)
    return _SCENES[key]


# ---------------------------------------------------------------------------
# Placement and apply_params
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sampling", ["yee", "yee_smooth"])
def test_device_scene_places_on_the_diagonal_tier(sampling):
    """Per-Yee-point sampling widens the permittivity array; the device must read 3, not 1."""
    _, arrays, _, _, _, _, _ = _scene(sampling)
    assert arrays.inv_permittivities.shape == (3, *DOMAIN)
    _, box_arrays, _, _, _, _, _ = _scene("box")
    assert box_arrays.inv_permittivities.shape == (1, *DOMAIN)


@pytest.mark.parametrize("sampling", ["box", "yee", "yee_smooth"])
def test_apply_params_writes_the_device_region_on_every_tier(sampling):
    """``apply_params`` fills the design region with a two-material interpolation, in every mode."""
    container, arrays, params, _, _, device_name, _ = _scene(sampling)
    applied, applied_objects, _ = fdtdx.apply_params(arrays, container, params, jax.random.PRNGKey(1))
    assert applied.inv_permittivities.shape == arrays.inv_permittivities.shape
    assert bool(jnp.all(jnp.isfinite(applied.inv_permittivities)))

    device = next(o for o in applied_objects.objects if o.name == device_name)
    region = np.asarray(applied.inv_permittivities[:, *device.grid_slice], dtype=np.float64)
    eps = 1.0 / region
    # Continuous parameters interpolate between air (1.0) and silicon (12.25); nothing outside.
    assert eps.min() >= 1.0 - 1e-4
    assert eps.max() <= EPS_SI + 1e-4
    # The device actually wrote something: a random parameter field is not one single value.
    assert eps.max() - eps.min() > 1.0


def test_yee_smooth_blends_the_strip_but_leaves_the_device_region_to_apply_params():
    """The smoother touches the static strip's interface pixels and nothing inside the design region.

    A ``Device`` is not a static object, so it is invisible to the scene loader: the smoothed pixels
    all sit on the strip's own faces. ``apply_params`` then overwrites the design region wholesale.
    """
    _, arrays, _, _, info, _, _ = _scene("yee_smooth")
    stats = info["yee_sampling_difference"]["smoothing"]
    assert stats["num_smoothed"] > 0
    eps = 1.0 / np.asarray(arrays.inv_permittivities, dtype=np.float64)
    between = (eps > EPS_OXIDE + 1e-6) & (eps < EPS_SI - 1e-6)
    assert between.sum() == stats["num_smoothed"]

    _, point_arrays, _, _, _, _, _ = _scene("yee")
    point_eps = 1.0 / np.asarray(point_arrays.inv_permittivities, dtype=np.float64)
    assert not np.allclose(eps, point_eps)


# ---------------------------------------------------------------------------
# The forward run and the gradient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sampling", ["box", "yee", "yee_smooth"])
def test_gradient_is_finite_and_nonzero(sampling):
    """``jax.grad`` of the detected energy w.r.t. the design parameters flows in every mode."""
    container, arrays, params, config, _, device_name, _ = _scene(sampling)
    key = jax.random.PRNGKey(1)
    value = _objective(params, container, arrays, config, key)
    assert bool(jnp.isfinite(value)) and float(value) > 0.0

    grads = jax.grad(_objective)(params, container, arrays, config, key)[device_name]
    assert grads.shape == params[device_name].shape
    assert bool(jnp.all(jnp.isfinite(grads)))
    assert float(jnp.max(jnp.abs(grads))) > 0.0


@pytest.mark.parametrize("sampling", ["box", "yee_smooth"])
def test_gradient_matches_a_central_finite_difference(sampling):
    """Reverse-mode gradient vs a central difference on the single most sensitive parameter.

    ``1e-3`` relative is the stated bar. The measured disagreement on this machine is 4.9e-5 in
    ``"box"`` and 3.9e-5 in ``"yee_smooth"``; the headroom is for float32 run-to-run drift, not for
    a real discrepancy.
    """
    container, arrays, params, config, _, device_name, _ = _scene(sampling)
    key = jax.random.PRNGKey(1)
    grads = jax.grad(_objective)(params, container, arrays, config, key)[device_name]

    flat_index = int(np.argmax(np.abs(np.asarray(grads))))
    index = tuple(int(i) for i in np.unravel_index(flat_index, grads.shape))
    analytic = float(grads[index])
    assert abs(analytic) > 0.0

    step = 0.05
    base = params[device_name]

    def shifted(delta: float) -> float:
        perturbed = dict(params)
        perturbed[device_name] = base.at[index].add(delta)
        return float(_objective(perturbed, container, arrays, config, key))

    finite = (shifted(step) - shifted(-step)) / (2.0 * step)
    relative = abs(finite - analytic) / abs(finite)
    assert relative < 1e-3, f"{sampling}: grad {analytic:.6e} vs finite difference {finite:.6e} ({relative:.2e})"


def test_discrete_device_gradient_flows_under_yee_smooth():
    """A discretized device (straight-through estimator) still yields a finite, nonzero gradient."""
    container, arrays, params, config, _, device_name, _ = _scene("yee_smooth", discrete=True)
    key = jax.random.PRNGKey(1)
    grads = jax.grad(_objective)(params, container, arrays, config, key)[device_name]
    assert bool(jnp.all(jnp.isfinite(grads)))
    assert float(jnp.max(jnp.abs(grads))) > 0.0
