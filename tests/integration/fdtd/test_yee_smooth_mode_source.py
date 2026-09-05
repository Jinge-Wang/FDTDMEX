"""A ``ModePlaneSource`` on a strip waveguide under ``material_sampling="yee_smooth"``.

The mode solve happens inside ``apply_params``: the source slices the assembled permittivity array
at its own plane and hands that cross-section to the solver. Under ``"yee_smooth"`` that slice is no
longer a two-valued staircase — it is a 3-component diagonal array whose interface pixels carry
blended values that differ between components. This file checks that the native (``"fdtdmex"``)
mode backend accepts that cross-section, that the launched mode is finite and comparable to the one
``"box"`` launches, and that a short forward run completes.

Measured on this machine (JAX CPU, 25 nm cells, 412.5 x 207.5 nm silicon core in oxide,
lambda = 1.55 um, 168 steps):

==========  ===========  ==============================
mode        ``n_eff``    total detected Poynting flux
==========  ===========  ==============================
box         2.1599       3.039e-15
yee         2.3111       2.954e-15
yee_smooth  2.2175       3.021e-15
==========  ===========  ==============================

Read honestly: the three numbers are not converging on each other here, they are three different
devices. ``"box"`` rounds the drawn 412.5 x 207.5 nm core down to 400 x 200 nm; ``"yee"`` keeps the
drawn size but staircases it, and at this grid the point sample lands an extra lattice column inside
the core; ``"yee_smooth"`` keeps the drawn size and averages the boundary pixels. The point of the
test is the third column: whichever cross-section the solver is given, the launched power is finite
and within a few per cent, so nothing about the smoothed array breaks the source.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx

SPACING = 25e-9
WAVELENGTH = 1.55e-6
TRANSVERSE_CELLS = 40
PROPAGATION_CELLS = 10
#: Deliberately not a whole number of cells (16.5 x 8.3), so the three sampling modes differ.
CORE = (412.5e-9, 207.5e-9)
N_SI = 3.48
N_OXIDE = 1.444

_COUNTER = [0]


def _tag() -> str:
    _COUNTER[0] += 1
    return f"yms{_COUNTER[0]}"


def _build(sampling: str):
    name = _tag()
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=SPACING),
        time=8e-15,
        dtype=jnp.float32,
        material_sampling=sampling,
    )
    volume = fdtdx.SimulationVolume(
        name=f"vol_{name}",
        partial_grid_shape=(PROPAGATION_CELLS, TRANSVERSE_CELLS, TRANSVERSE_CELLS),
        material=fdtdx.Material(permittivity=N_OXIDE**2),
    )
    objects, constraints = [volume], []
    bound_dict, bound_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=3), volume
    )
    objects.extend(bound_dict.values())
    constraints.extend(bound_constraints)

    core = fdtdx.UniformMaterialObject(
        name=f"core_{name}",
        partial_real_shape=(None, CORE[0], CORE[1]),
        material=fdtdx.Material(permittivity=N_SI**2),
        placement_order=1,
    )
    objects.append(core)
    constraints.extend([core.same_size(volume, axes=(0,)), core.place_at_center(volume, axes=(0, 1, 2))])

    wave = fdtdx.WaveCharacter(wavelength=WAVELENGTH)
    source = fdtdx.ModePlaneSource(
        name=f"src_{name}",
        partial_grid_shape=(1, None, None),
        wave_character=wave,
        direction="+",
    )
    objects.append(source)
    constraints.extend(
        [
            source.same_size(volume, axes=(1, 2)),
            source.place_at_center(volume, axes=(1, 2)),
            source.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(3,)),
        ]
    )

    flux = fdtdx.PoyntingFluxDetector(
        name=f"flux_{name}",
        partial_grid_shape=(1, None, None),
        direction="+",
        reduce_volume=True,
    )
    objects.append(flux)
    constraints.extend(
        [
            flux.same_size(volume, axes=(1, 2)),
            flux.place_at_center(volume, axes=(1, 2)),
            flux.set_grid_coordinates(axes=(0,), sides=("-",), coordinates=(6,)),
        ]
    )

    key = jax.random.PRNGKey(0)
    container, arrays, params, config, _info = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    applied_arrays, applied_objects, _ = fdtdx.apply_params(arrays, container, params, key=key)
    return applied_arrays, applied_objects, config, source.name, flux.name


_SCENES: dict[str, tuple] = {}


def _scene(sampling: str):
    if sampling not in _SCENES:
        _SCENES[sampling] = _build(sampling)
    return _SCENES[sampling]


def _source(objects, name):
    return next(o for o in objects.objects if o.name == name)


# ---------------------------------------------------------------------------
# What the solver is handed
# ---------------------------------------------------------------------------


def test_the_mode_solver_receives_the_smoothed_cross_section():
    """The plane the source slices carries per-component blended values, not a two-valued staircase."""
    arrays, objects, _, source_name, _ = _scene("yee_smooth")
    source = _source(objects, source_name)
    cross_section = np.asarray(source._inv_permittivity, dtype=np.float64)
    assert cross_section.shape[0] == 3
    eps = 1.0 / cross_section
    blended = (eps > N_OXIDE**2 + 1e-6) & (eps < N_SI**2 - 1e-6)
    assert blended.sum() > 0, "no smoothed pixel reached the source's cross-section"
    # The three components see different pixels: that is the whole point of per-Yee-point sampling.
    assert not np.allclose(eps[0], eps[1])
    assert bool(np.all(np.isfinite(eps)))
    del arrays


def test_the_native_backend_accepts_the_smoothed_cross_section():
    """The diagonal blend keeps every off-diagonal term at zero, so the native solver takes it.

    The native backend refuses an off-diagonal magnitude above its tensorial tolerance and routes
    such a cross-section to Tidy3D, which is an optional dependency. The default diagonal tier never
    produces one: it writes only entry ``(c, c)`` of the Kottke tensor at component ``c``. Only
    ``config.yee_smooth_full_tensor=True`` with a genuinely tilted interface would, and this
    Manhattan geometry has no tilted interface at all.
    """
    _, objects, _, source_name, _ = _scene("yee_smooth")
    source = _source(objects, source_name)
    neff = complex(np.asarray(source._neff))
    assert np.isfinite(neff.real) and np.isfinite(neff.imag)
    assert N_OXIDE < neff.real < N_SI, f"n_eff {neff.real} is not a guided index"
    assert abs(neff.imag) < 1e-9 * max(abs(neff.real), 1.0)
    assert bool(np.all(np.isfinite(np.asarray(source._E))))
    assert bool(np.all(np.isfinite(np.asarray(source._H))))
    assert float(np.max(np.abs(np.asarray(source._E)))) > 0.0


@pytest.mark.parametrize("sampling", ["box", "yee", "yee_smooth"])
def test_the_solved_index_is_guided_in_every_mode(sampling):
    _, objects, _, source_name, _ = _scene(sampling)
    neff = complex(np.asarray(_source(objects, source_name)._neff)).real
    assert N_OXIDE < neff < N_SI


def test_smoothing_moves_the_index_off_the_point_sample():
    """The three sampling modes give three different cross-sections, so three different ``n_eff``."""
    indices = {}
    for sampling in ("box", "yee", "yee_smooth"):
        _, objects, _, source_name, _ = _scene(sampling)
        indices[sampling] = complex(np.asarray(_source(objects, source_name)._neff)).real
    assert abs(indices["yee_smooth"] - indices["yee"]) > 1e-3
    assert abs(indices["yee_smooth"] - indices["box"]) > 1e-3


# ---------------------------------------------------------------------------
# The forward run
# ---------------------------------------------------------------------------


def _launched_flux(sampling: str) -> np.ndarray:
    arrays, objects, config, _, flux_name = _scene(sampling)
    _, final_arrays = fdtdx.run_fdtd(
        arrays=arrays, objects=objects, config=config, key=jax.random.PRNGKey(0), show_progress=False
    )
    return np.asarray(final_arrays.detector_states[flux_name]["poynting_flux"], dtype=np.float64)


def test_launched_power_is_finite_and_comparable_to_box_mode():
    """A short run completes and the smoothed scene launches the same power as the box scene.

    ``5%`` is the bar; the measured difference on this machine is 0.6% against ``"box"`` and 2.3%
    against ``"yee"``. The three runs are three slightly different waveguides, so an exact match is
    not the expectation — a change of order the staircasing error is.
    """
    fluxes = {mode: _launched_flux(mode) for mode in ("box", "yee", "yee_smooth")}
    for mode, flux in fluxes.items():
        assert np.all(np.isfinite(flux)), f"{mode}: non-finite flux"
        assert float(np.max(np.abs(flux))) > 0.0, f"{mode}: nothing was launched"
    totals = {mode: float(np.sum(flux)) for mode, flux in fluxes.items()}
    for mode in ("box", "yee"):
        relative = abs(totals["yee_smooth"] - totals[mode]) / abs(totals[mode])
        assert relative < 0.05, f"yee_smooth flux {totals['yee_smooth']:.4e} vs {mode} {totals[mode]:.4e}"
