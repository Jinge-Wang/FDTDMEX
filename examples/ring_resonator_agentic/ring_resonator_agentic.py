# %% [markdown]
# # ring_resonator_agentic — a microring transmission study, the AGENTIC way
#
# **This is the canonical template for an LLM agent / ag-fdtd notebook.** It reproduces the core of
# `ring_mrm_oband` — bus **TE0 mode** → cold **FDTD run** → through-port **transmission `T(λ)`** and
# loaded **Q** — but it runs the simulation through the **agentic contract**, NOT the in-process engine.
#
# ## The one hard rule
# To PACK and RUN a simulation you use ONLY `fdtdmex.io`:
#
#     from fdtdmex.io import pack, run_simulation_from_hdf5
#     bundle = pack(scene, ".")                                   # resolve + freeze → one config HDF5
#     job    = run_simulation_from_hdf5(bundle, "jobs", ...)      # DETACHED, NON-BLOCKING — returns at once
#
# **Never** call `fdtdx.run_fdtd`, `fdtdx.apply_params`, `fdtdx.place_objects`, or any fdtdx/jax
# simulate primitive to execute a run — those block the kernel and bypass the job system. `fdtdx` is
# used ONLY to BUILD the scene objects (volume, materials, boundaries, sources, detectors, `Scene`)
# and to solve modes (`fdtdx.compute_mode`). Most other examples call `fdtdx.run_fdtd` for fast
# in-memory iteration — copy their **geometry**, but REWRITE the run as the two lines above.
#
# ## Reading results
# `run_simulation_from_hdf5` writes `jobs/<name>/outputs/result.hdf5` when it finishes. To read it:
#   * `sim_postproc(path)` → small per-detector SCALARS (shape, max_abs, mean_abs) — good for magnitudes.
#   * a SPECTRUM (transmission / Poynting flux) needs the **full complex fields**, which the reducer
#     drops — so read them directly with `h5py`: `f["detector_states"][name][key]` are the full arrays.
#
# Run as a script (`python ring_resonator_agentic.py`) or step through the `# %%` cells in a notebook.
# `RING_BACKEND=mock` for a fast GPU-free plumbing check (synthetic fields); default `mlx` is the real
# Metal solve that shows the resonance notch.

# %%
import os
import tempfile
import time

import h5py
import numpy as np

import fdtdx
import gdstk
from fdtdx.objects.static_material.polygon import extruded_polygon_from_gds_path
from fdtdmex.io import pack, run_simulation_from_hdf5  # the ONLY pack/run seam (never fdtdx.run_fdtd)

BACKEND = os.environ.get("RING_BACKEND", "mlx")       # "mlx" (real) | "mock" (fast pipeline check)
RES = 60e-9                                            # grid spacing (coarse keeps the real solve feasible)
SETTLE = 2.0e-12                                       # ring-down time (longer → sharper resonances, slower)
N_SI, N_OX = 3.476, 1.444                             # O-band Si / SiO2
R, WG, LC, CORE_T = 2.5, 0.50, 1.5, 0.22e-6           # ring radius, wg width, racetrack straight, SOI thickness
GAP = 0.12e-6                                          # bus–ring edge gap (coupling knob)
LAMBDA0 = 1.31e-6
WLS = np.linspace(1.285e-6, 1.335e-6, 21)             # O-band sweep (shared by the run + the analysis)
MAT = {"si": fdtdx.Material(permittivity=N_SI**2), "ox": fdtdx.Material(permittivity=N_OX**2)}

# %% [markdown]
# ## 1. Bus TE0 mode — `fdtdx.compute_mode` (this runs IN-PROCESS; it is NOT a simulation run)
#
# The mode solver is a small linear solve, fine to call directly in the kernel. It gives the
# effective index `n_eff` and group index `n_g` of the bus waveguide — the modal context for the ring.

# %%
ny, nz = 64, 48
ys = (np.arange(ny) - ny / 2 + 0.5) * RES
zs = (np.arange(nz) - nz / 2 + 0.5) * RES
Y, Z = np.meshgrid(ys, zs, indexing="ij")
core_cs = (np.abs(Y) <= WG * 1e-6 / 2) & (np.abs(Z) <= CORE_T / 2)
eps_cs = np.where(core_cs, N_SI**2, N_OX**2)


def solve_mode(wl):
    import jax.numpy as jnp
    E, H, neff = fdtdx.compute_mode(
        frequency=fdtdx.constants.c / wl,
        inv_permittivities=jnp.asarray((1.0 / eps_cs)[None, :, :, None]),
        inv_permeabilities=1.0, resolution=RES, filter_pol="te")
    return np.asarray(E), np.asarray(H), complex(neff)


dl = 10e-9
E_m, _, neff0 = solve_mode(LAMBDA0)
n_eff = neff0.real
n_g = n_eff - LAMBDA0 * (solve_mode(LAMBDA0 + dl)[2].real - solve_mode(LAMBDA0 - dl)[2].real) / (2 * dl)
print(f"bus TE0 @ {LAMBDA0*1e9:.0f} nm:  n_eff = {n_eff:.4f}   n_g = {n_g:.4f}")

try:
    import matplotlib.pyplot as plt
    E2 = (np.abs(E_m.reshape(3, ny, nz)) ** 2).sum(axis=0)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.pcolormesh(ys * 1e6, zs * 1e6, E2.T, shading="auto")
    ax.contour(ys * 1e6, zs * 1e6, core_cs.T, levels=[0.5], colors="w", linewidths=0.6)
    ax.set_xlabel("y (µm)"); ax.set_ylabel("z (µm)"); ax.set_title("bus TE0  |E|²")
except Exception:
    pass

# %% [markdown]
# ## 2. Build the ring + bus scene → `pack` → DETACHED run (`run_simulation_from_hdf5`)
#
# Geometry (copy this part from any example): a `gdstk` racetrack ring (outer Si disk + inner oxide
# carve) side-coupled to a straight bus, loaded via `extruded_polygon_from_gds_path`, with input/
# through `PhasorDetector`s. The RUN is the agentic contract — `pack(scene, ".")` then
# `run_simulation_from_hdf5(...)`, which returns immediately while the solver runs detached.

# %%
# --- geometry: GDS with bus (layer 1, idx 0) + racetrack outer Si (idx 1) + inner oxide carve (layer 2) ---
CY = WG + GAP * 1e6 + R                                # bus → ring-centre spacing (µm)
lib = gdstk.Library(unit=1e-6, precision=1e-9)
cell = lib.new_cell("MRM")
cell.add(gdstk.rectangle((-4.0, -WG / 2), (4.0, WG / 2), layer=1))
cell.add(gdstk.racetrack((0, CY), LC, R + WG / 2, layer=1, tolerance=2e-3))
cell.add(gdstk.racetrack((0, CY), LC, R - WG / 2, layer=2, tolerance=2e-3))
gp = os.path.join(tempfile.gettempdir(), "ring_agentic.gds")
lib.write_gds(gp)


def load(layer, idx, mat):
    p = extruded_polygon_from_gds_path(gp, "MRM", layer=layer, polygon_index=idx, axis=2,
                                       material_name=mat, materials=MAT)
    object.__setattr__(p, "partial_real_shape", (*p.partial_real_shape[:2], CORE_T))
    return p


LX, LZ, YBUS, PML = 8.0e-6, 0.8e-6, 0.8e-6, 8
LY = (YBUS * 1e6 + CY + R + WG / 2 + 0.6) * 1e-6
vol = fdtdx.SimulationVolume(
    partial_real_shape=(LX + 2 * PML * RES, LY + 2 * PML * RES, LZ + 2 * PML * RES),
    material=MAT["ox"], name="bg")
ol, cons = [vol], []
bd, bc = fdtdx.boundary_objects_from_config(
    fdtdx.BoundaryConfig.from_uniform_bound(thickness=PML, boundary_type="pml"), vol)
ol += list(bd.values()); cons += bc


def at(obj, off):
    return obj.place_relative_to(vol, axes=(0, 1, 2), own_positions=(0, 0, 0), other_positions=(-1, -1, -1),
                                 margins=(off[0] + PML * RES, off[1] + PML * RES, off[2] + PML * RES))


yring = YBUS + CY * 1e-6
for poly, off in [(load(1, 0, "si"), (LX / 2, YBUS, LZ / 2)),       # bus strip
                  (load(1, 1, "si"), (LX / 2, yring, LZ / 2)),      # outer ring disk (Si)
                  (load(2, 0, "ox"), (LX / 2, yring, LZ / 2))]:     # inner carve (oxide) → annulus
    ol.append(poly); cons.append(at(poly, off))

wcs = tuple(fdtdx.WaveCharacter(wavelength=float(w)) for w in WLS)
cwc = fdtdx.WaveCharacter(wavelength=LAMBDA0)
prof = fdtdx.GaussianPulseProfile(center_wave=cwc, spectral_width=fdtdx.WaveCharacter(wavelength=LAMBDA0 * 18))
W, H = 1.2e-6, 0.5e-6
src = fdtdx.GaussianPlaneSource(partial_grid_shape=(1, None, None), partial_real_shape=(None, W, H),
        fixed_E_polarization_vector=(0, 1, 0), wave_character=cwc, temporal_profile=prof,
        radius=0.5e-6, std=1 / 3, direction="+", name="src")
ol.append(src); cons.append(at(src, (1.0e-6, YBUS, LZ / 2)))
comps = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
inm = fdtdx.PhasorDetector(wave_characters=wcs, components=comps, partial_real_shape=(RES, W, H), name="in")
ol.append(inm); cons.append(at(inm, (1.9e-6, YBUS, LZ / 2)))
thr = fdtdx.PhasorDetector(wave_characters=wcs, components=comps, partial_real_shape=(RES, W, H), name="thru")
ol.append(thr); cons.append(at(thr, (LX - 1.0e-6, YBUS, LZ / 2)))

cfg = fdtdx.SimulationConfig(time=SETTLE, grid=fdtdx.UniformGrid(spacing=RES))

# pack: resolve + rasterize + freeze the declarative scene into one self-contained config HDF5.
# The retry grows the volume by a cell until the PML tiling resolves (a common place_objects quirk).
bundle = None
for _ in range(12):
    try:
        bundle = pack(fdtdx.Scene(cfg).add(*ol).constrain(cons), ".")
        break
    except ValueError:
        s = vol.partial_real_shape
        object.__setattr__(vol, "partial_real_shape", (s[0] + RES, s[1] + RES, s[2]))

# run: DETACHED + NON-BLOCKING. In an ag-fdtd notebook the cell ends here and you watch the
# Simulations panel; as a script we poll the status file to completion before analysing.
job = run_simulation_from_hdf5(bundle, "jobs", simulation_name="ring-cold", name="ring-cold", backend=BACKEND)
print(f"launched ring-cold (backend={BACKEND}, gap={GAP*1e9:.0f} nm) -> {job.job_dir}")

import json
for _ in range(6000):
    st = json.loads(job.status_path.read_text()) if job.status_path.exists() else {}
    if st.get("status") in ("completed", "failed"):
        break
    time.sleep(0.5)
print("run status:", st.get("status"))

# %% [markdown]
# ## 3. Through-port transmission `T(λ)` + resonance Q — read the FULL fields from `result.hdf5`
#
# `sim_postproc(job.results_path)` would give only scalars; transmission needs the complex phasor
# fields, so we read them with `h5py`, compute the net Poynting flux `T(λ) = P_thru / P_in`, and pull
# the loaded Q from the resonance notch's FWHM.

# %%
def load_phasor(f, name):
    g = f["detector_states"][name]
    return {k: np.asarray(g[k]) for k in g.keys()}["phasor"]      # full complex (1, n_freq, 6, *plane)


def net_power(phasor, prop_axis=0):
    ph = phasor[0]                                                # (n_freq, 6, *plane)
    E, H = ph[:, :3], ph[:, 3:]
    ax, ay = [1, 2, 0][prop_axis], [2, 0, 1][prop_axis]
    Sx = E[:, ax] * np.conj(H[:, ay]) - E[:, ay] * np.conj(H[:, ax])
    return 0.5 * np.real(Sx.reshape(Sx.shape[0], -1).sum(axis=1))


with h5py.File(job.results_path, "r") as f:
    P_in = net_power(load_phasor(f, "in"))
    P_thru = net_power(load_phasor(f, "thru"))

wl_nm = WLS * 1e9
T = np.abs(P_thru) / np.maximum(np.abs(P_in), 1e-30)
i0 = int(np.argmin(T))                                            # resonance notch (deepest dip)
lam0, Tmin = wl_nm[i0], float(T[i0])
base = float(np.max(T)); half = (base + Tmin) / 2
lo = next((wl_nm[j] for j in range(i0, 0, -1) if T[j] >= half), wl_nm[0])
hi = next((wl_nm[j] for j in range(i0, len(T)) if T[j] >= half), wl_nm[-1])
fwhm = max(hi - lo, 1e-9)
Q = lam0 / fwhm
print(f"resonance: λ0 = {lam0:.2f} nm   T_min = {Tmin:.3f}   FWHM = {fwhm:.2f} nm   Q ≈ {Q:.0f}")

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(wl_nm, T, "-o", ms=3)
    ax.axvline(lam0, color="r", ls="--", lw=0.8)
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("T = P_thru / P_in")
    ax.set_title(f"through-port transmission (Q ≈ {Q:.0f})"); ax.grid(alpha=0.3)
except Exception:
    pass
