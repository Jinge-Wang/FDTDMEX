"""A dispersive material at a Kottke-smoothed interface.

What ``material_sampling="yee_smooth"`` does at a pixel where a dispersive material meets a plain
dielectric is a deliberate split:

* the **instantaneous** permittivity (``eps_inf``, which is what ``inv_permittivities`` holds) is
  blended by the Kottke rule, like any other two-material pixel;
* the **pole coefficients** ``c1``/``c2``/``c3`` of the ADE recurrence keep the point sample. They
  are gathered straight from the winning material's table, with no averaging of any kind.

That is a documented approximation, not an oversight: a boundary pixel carries an averaged
``eps_inf`` but a full-strength (or absent) susceptibility, so the resonance it contributes is that
of whichever material won the point sample. Meep averages the susceptibility by volume fraction
instead. Averaging the poles here would need the fill fraction threaded through the dispersive
gather, which Stage B did not do. These tests pin the behaviour that exists, so that changing it is
a deliberate act, and check that the combination produces no NaN and runs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.dispersion import DispersionModel, LorentzPole

SPACING = 40e-9
DOMAIN = (16, 16, 16)
PML = 3
EPS_BG = 2.25
EPS_INF = 4.0
#: 250 nm is 6.25 cells: the slab's two faces land inside a cell, so both are smoothed.
SLAB_THICKNESS = 250e-9

_COUNTER = [0]


def _tag() -> str:
    _COUNTER[0] += 1
    return f"yds{_COUNTER[0]}"


def _dispersive_material() -> fdtdx.Material:
    return fdtdx.Material(
        permittivity=EPS_INF,
        dispersion=DispersionModel(poles=(LorentzPole(resonance_frequency=2e15, damping=1e13, delta_epsilon=1.5),)),
    )


def _build(sampling: str):
    name = _tag()
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=SPACING),
        time=6e-15,
        dtype=jnp.float32,
        material_sampling=sampling,
    )
    volume = fdtdx.SimulationVolume(
        name=f"vol_{name}",
        partial_grid_shape=DOMAIN,
        material=fdtdx.Material(permittivity=EPS_BG),
    )
    objects, constraints = [volume], []
    bound_dict, bound_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=PML), volume
    )
    objects.extend(bound_dict.values())
    constraints.extend(bound_constraints)

    slab = fdtdx.UniformMaterialObject(
        name=f"slab_{name}",
        partial_real_shape=(SLAB_THICKNESS, None, None),
        material=_dispersive_material(),
        placement_order=1,
    )
    objects.append(slab)
    constraints.extend([slab.same_size(volume, axes=(1, 2)), slab.place_at_center(volume, axes=(0, 1, 2))])

    source = fdtdx.PointDipoleSource(
        name=f"src_{name}",
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=1.0e-6),
        polarization=1,
        amplitude=1.0,
    )
    objects.append(source)
    constraints.append(source.set_grid_coordinates(axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(5, 8, 8)))

    key = jax.random.PRNGKey(0)
    return fdtdx.place_objects(object_list=objects, config=config, constraints=constraints, key=key)


_SCENES: dict[str, tuple] = {}


def _scene(sampling: str):
    if sampling not in _SCENES:
        _SCENES[sampling] = _build(sampling)
    return _SCENES[sampling]


def _eps(arrays) -> np.ndarray:
    return 1.0 / np.asarray(arrays.inv_permittivities, dtype=np.float64)


# ---------------------------------------------------------------------------
# What is blended and what is not
# ---------------------------------------------------------------------------


def test_the_permittivity_is_blended_at_the_slab_faces():
    """``eps_inf`` takes intermediate values at exactly the pixels the smoother reports."""
    _, arrays, _, _, info = _scene("yee_smooth")
    eps = _eps(arrays)
    between = (eps > EPS_BG + 1e-6) & (eps < EPS_INF - 1e-6)
    assert between.sum() == info["yee_sampling_difference"]["smoothing"]["num_smoothed"]
    # Two faces, one per side of the slab, over the full transverse cross-section, per component.
    assert between.sum() == 3 * 2 * DOMAIN[1] * DOMAIN[2]
    assert bool(np.all(np.isfinite(eps)))


@pytest.mark.parametrize("field", ["dispersive_c1", "dispersive_c2", "dispersive_c3"])
def test_the_pole_coefficients_stay_point_sampled(field):
    """Every coefficient is either the slab's tabulated value or zero — never anything in between."""
    _, smooth_arrays, _, _, _ = _scene("yee_smooth")
    _, box_arrays, _, _, _ = _scene("box")
    smooth = np.asarray(getattr(smooth_arrays, field), dtype=np.float64)
    box = np.asarray(getattr(box_arrays, field), dtype=np.float64)

    assert np.all(np.isfinite(smooth))
    smooth_values = np.unique(smooth)
    box_values = np.unique(box)
    # The box path can only ever write the table's own entries; the smoothed path writes the same
    # set, at (possibly) different points. A blended coefficient would add a third value.
    assert smooth_values.size <= box_values.size
    np.testing.assert_allclose(smooth_values, box_values[: smooth_values.size], rtol=1e-6, atol=0)


def test_a_smoothed_pixel_carries_a_blended_permittivity_and_an_unblended_pole():
    """The documented split, checked at the pixels where it actually happens."""
    _, arrays, _, _, _ = _scene("yee_smooth")
    eps = _eps(arrays)
    c1 = np.asarray(arrays.dispersive_c1, dtype=np.float64)
    assert c1.shape[1] == 3, "per-Yee-point sampling widens the dispersive tier to 3 components"

    blended = (eps > EPS_BG + 1e-6) & (eps < EPS_INF - 1e-6)
    assert blended.any()
    allowed = np.unique(c1)
    at_blended = c1[0][blended]
    # Every coefficient sitting under a blended permittivity is still one of the exact table values.
    assert np.all(np.isin(np.round(at_blended, 12), np.round(allowed, 12)))
    # And both outcomes really occur among them: some boundary pixels won for the slab, some for the
    # background, so the split is visible rather than vacuous.
    assert at_blended.min() != at_blended.max()


def test_nothing_is_nan_anywhere():
    for sampling in ("box", "yee", "yee_smooth"):
        _, arrays, _, _, _ = _scene(sampling)
        for name in ("inv_permittivities", "dispersive_c1", "dispersive_c2", "dispersive_c3"):
            value = getattr(arrays, name)
            assert value is not None, name
            assert bool(np.all(np.isfinite(np.asarray(value)))), f"{sampling}: {name}"


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sampling", ["box", "yee_smooth"])
def test_a_short_dispersive_run_completes(sampling):
    """The ADE recurrence steps through the smoothed slab without blowing up."""
    container, arrays, params, config, _ = _scene(sampling)
    key = jax.random.PRNGKey(0)
    applied_arrays, applied_objects, _ = fdtdx.apply_params(arrays, container, params, key=key)
    _, final_arrays = fdtdx.run_fdtd(
        arrays=applied_arrays, objects=applied_objects, config=config, key=key, show_progress=False
    )
    for name in ("E", "H"):
        field = np.asarray(getattr(final_arrays.fields, name))
        assert np.all(np.isfinite(field)), f"{sampling}: non-finite {name}"
    assert float(np.max(np.abs(np.asarray(final_arrays.fields.E)))) > 0.0
    for name in ("dispersive_P_curr", "dispersive_P_prev"):
        polarization = np.asarray(getattr(final_arrays.fields, name))
        assert np.all(np.isfinite(polarization)), f"{sampling}: non-finite {name}"
