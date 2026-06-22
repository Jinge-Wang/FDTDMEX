"""Quickstart — define, see, run, inspect a simulation (notebook style).

Open this in the VS Code interactive window (or Jupyter): the ``# %%`` markers are cells. It walks the
FDTDMEX flow with **inline** matplotlib figures, the same plots a Tidy3D notebook would show:

    define config + objects  ->  plot the setup  ->  plot the material  ->  run  ->  plot the fields
    solve a waveguide mode    ->  plot the mode   ->  assemble an S-matrix ->  plot the S-matrix

Every ``fdtdx.plot_*`` call returns a matplotlib ``Figure`` that renders inline. On Apple Silicon the
forward run auto-routes to the MLX/Metal backend; elsewhere it runs the JAX engine.
"""

# %%
# --- 1. Configure + build the object list -------------------------------------------------------
import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

key = jax.random.PRNGKey(0)
wavelength = 1.55e-6

config = fdtdx.SimulationConfig(
    time=40e-15,
    grid=fdtdx.UniformGrid(spacing=50e-9),
    dtype=jnp.float32,
    courant_factor=0.99,
)

constraints, object_list = [], []

# Simulation volume + background (low-index cladding).
volume = fdtdx.SimulationVolume(
    partial_real_shape=(4e-6, 4e-6, 4e-6),
    material=fdtdx.Material(permittivity=2.07),  # ~SiO2
)
object_list.append(volume)

# A high-index slab through the middle (a crude waveguide core).
core = fdtdx.UniformMaterialObject(
    partial_real_shape=(4e-6, 0.5e-6, 0.3e-6),
    material=fdtdx.Material(permittivity=12.1),  # ~Si
)
constraints += core.same_position_and_size(volume, axes=(0,))
constraints.append(core.place_relative_to(volume, axes=(1, 2), own_positions=(0, 0), other_positions=(0, 0)))
object_list.append(core)

# PML on all sides.
bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=8, boundary_type="pml")
bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
constraints += c_list
object_list += list(bound_dict.values())

# A Gaussian plane source launching into the core, and an energy detector.
source = fdtdx.GaussianPlaneSource(
    partial_grid_shape=(1, None, None),
    partial_real_shape=(None, 2e-6, 2e-6),
    fixed_E_polarization_vector=(0, 1, 0),
    wave_character=fdtdx.WaveCharacter(wavelength=wavelength),
    radius=1e-6,
    std=1 / 3,
    direction="+",
)
constraints.append(source.place_relative_to(volume, axes=(0, 1, 2), own_positions=(-1, 0, 0), other_positions=(-0.6, 0, 0)))
object_list.append(source)

detector = fdtdx.EnergyDetector(name="energy")  # default form keeps the run on the MLX backend
constraints += detector.same_position_and_size(volume)
object_list.append(detector)

# %%
# --- 2. Place the objects, then SEE the setup inline -------------------------------------------
key, subkey = jax.random.split(key)
objects, arrays, params, config, _ = fdtdx.place_objects(
    object_list=object_list, config=config, constraints=constraints, key=subkey
)

# Object layout (XY / XZ / YZ panels).
fig_setup = fdtdx.plot_setup(config=config, objects=objects, exclude_object_list=[detector])
fig_setup  # renders inline

# %%
# --- 3. SEE the material (permittivity) cross-section ------------------------------------------
fig_mat = fdtdx.plot_material_from_side(config=config, arrays=arrays, viewing_side="x")
fig_mat  # renders inline

# %%
# --- 4. RUN the forward simulation -------------------------------------------------------------
arrays, objs, _ = fdtdx.apply_params(arrays, objects, params, key)
_, result = fdtdx.run_fdtd(arrays=arrays, objects=objs, config=config, key=key, show_progress=False)

# Final E/H field slice through the middle (one transverse plane).
mid = result.fields.E.shape[1] // 2
fig_fields = fdtdx.plot_field_slice(E=result.fields.E[:, mid : mid + 1], H=result.fields.H[:, mid : mid + 1])
fig_fields  # renders inline

# %%
# --- 5. Solve a waveguide mode and SEE it (native Tidy3D-free solver) --------------------------
res = 20e-9
nx, ny = 100, 80
xs = (np.arange(nx) - nx / 2 + 0.5) * res
ys = (np.arange(ny) - ny / 2 + 0.5) * res
X, Y = np.meshgrid(xs, ys, indexing="ij")
eps_cs = np.where((np.abs(X) <= 0.25e-6) & (np.abs(Y) <= 0.11e-6), 3.48**2, 1.44**2)
inv_eps = jnp.asarray((1.0 / eps_cs)[None, :, :, None])  # (1, Nx, Ny, 1), z-propagating

E_m, H_m, neff = fdtdx.compute_mode(
    frequency=fdtdx.constants.c / wavelength,
    inv_permittivities=inv_eps,
    inv_permeabilities=1.0,
    resolution=res,
    filter_pol="te",
)
print(f"fundamental TE n_eff = {complex(neff):.4f}")
fig_mode = fdtdx.plot_mode(E_m, H_m, inv_permittivity=inv_eps)
fig_mode  # six components + energy + index cross-section, inline

# %%
# --- 6. Assemble + SEE an S-matrix (here from illustrative values) -----------------------------
# In a real 2-port run these come from fdtdx.calculate_sparams(...); the result class is the same.
sparams = {
    ("through", "in"): np.array([0.96 + 0.05j]),
    ("cross", "in"): np.array([0.04 - 0.02j]),
}
smat = fdtdx.SMatrixResult.from_sparams(sparams, frequencies=[fdtdx.constants.c / wavelength])
print(smat.to_json())  # small, JSON-serializable result an agent/front-end can read
fig_smat = fdtdx.plot_smatrix(smat, value="magnitude")
fig_smat  # renders inline
