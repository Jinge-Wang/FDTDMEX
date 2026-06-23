"""Generate the showcase images used in the README from the ring-resonator demo.

Run once to (re)create the figures in ``examples/ring_resonator_demo/figures/``::

    uv run python examples/ring_resonator_demo/make_showcase_images.py

Produces: the top-down layout, a 3D voxel view of the resolved permittivity, the injected bus mode,
the modal-transmission (mode-expansion) bars, and the analytical-vs-FDTD ring response. The grid is the
demo's coarse 90 nm — the images are illustrative, not converged.
"""

from __future__ import annotations

import os
import tempfile
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fdtdx
from fdtdx.fdtd.stop_conditions import EnergyThresholdCondition
from fdtdx.objects.static_material.polygon import extruded_polygon_from_gds_path
from fdtdx.utils.plot_material import plot_material_from_side
from fdtdx.utils.plot_setup import plot_setup_from_side
from fdtdx.utils.sparams import PortSpec, determine_input_norm_detector_name, setup_sparams_simulation

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

WAVELENGTH, SLAB_T = 1.55e-6, 0.22e-6
MATERIALS = {"si": fdtdx.Material(permittivity=12.25), "air": fdtdx.Material(permittivity=1.0)}
R, WG, GAP = 1.2, 0.40, 0.10
CY = WG + GAP + R
RES, LX, LZ = 90e-9, 6.4e-6, 0.66e-6
LY = (CY + R + WG + 0.9) * 1e-6
YBUS = 0.5e-6

# --- geometry (gdstk -> extruded polygons) --------------------------------------------------------
import gdstk

lib = gdstk.Library(unit=1e-6, precision=1e-9)
cell = lib.new_cell("RING")
cell.add(
    gdstk.rectangle((-3.0, -WG / 2), (3.0, WG / 2), layer=1),
    gdstk.ellipse((0, CY), R + WG / 2, layer=1, tolerance=2e-3),
    gdstk.ellipse((0, CY), R - WG / 2, layer=2, tolerance=2e-3),
)
GDS = os.path.join(tempfile.gettempdir(), "ring_showcase.gds")
lib.write_gds(GDS)


def polys():
    def load(layer, idx, mat):
        p = extruded_polygon_from_gds_path(GDS, "RING", layer=layer, polygon_index=idx, axis=2, material_name=mat, materials=MATERIALS)
        object.__setattr__(p, "partial_real_shape", (p.partial_real_shape[0], p.partial_real_shape[1], SLAB_T))
        return p

    b, ot, it = load(1, 0, "si"), load(1, 1, "si"), load(2, 0, "air")
    return [(ot, (LX / 2, YBUS + CY * 1e-6, LZ / 2)), (it, (LX / 2, YBUS + CY * 1e-6, LZ / 2)), (b, (LX / 2, YBUS, LZ / 2))]


def make_setup(wavelength):
    return setup_sparams_simulation(
        polygons=polys(),
        input_ports=[PortSpec(center=(0.8e-6, YBUS, LZ / 2), axis=0, direction="+", width=1.2e-6, height=0.5e-6, name="in")],
        output_ports=[PortSpec(center=(LX - 0.8e-6, YBUS, LZ / 2), axis=0, direction="+", width=1.2e-6, height=0.5e-6, name="through")],
        wavelength=wavelength, resolution=RES, max_time=500e-15, domain_size=(LX, LY, LZ), pml_layers=6,
    )


objects, arrays, config = make_setup(WAVELENGTH)

# --- 1a. simulation layout (objects + ports) ------------------------------------------------------
fig = plot_setup_from_side(config=config, objects=objects, viewing_side="z")
fig.savefig(os.path.join(OUT, "ring_layout.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
print("wrote ring_layout.png")

# --- 1b. top-down of the RASTERIZED permittivity (the real ring annulus + bus) --------------------
fig = plot_material_from_side(config=config, arrays=arrays, viewing_side="z")
fig.savefig(os.path.join(OUT, "ring_topdown.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
print("wrote ring_topdown.png")

# --- 2. 3D voxel view of the resolved silicon device layer (PML cropped) ---------------------------
eps = 1.0 / np.clip(np.asarray(arrays.inv_permittivities)[0], 1e-12, None)
pml = 6
core = eps[pml:-pml, pml:-pml, pml:-pml]  # drop PML (and its extended material)
mask = core > 6.0  # silicon
fig = plt.figure(figsize=(7, 4.5))
ax = fig.add_subplot(111, projection="3d")
ax.voxels(mask, facecolors="#2a6fb0", edgecolor="#1c4e7a", linewidth=0.05)
ax.set_box_aspect((mask.shape[0], mask.shape[1], max(mask.shape[2], 6)))
ax.set_axis_off()
ax.view_init(elev=78, azim=-90)
ax.set_title("Resolved silicon device layer (90 nm voxels)")
fig.savefig(os.path.join(OUT, "ring_3d.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
print("wrote ring_3d.png")

# --- 3. injected bus mode -------------------------------------------------------------------------
res_m, nx, ny = 20e-9, 90, 60
X, Y = np.meshgrid((np.arange(nx) - nx / 2 + 0.5) * res_m, (np.arange(ny) - ny / 2 + 0.5) * res_m, indexing="ij")
eps_cs = np.where((np.abs(X) <= WG / 2 * 1e-6) & (np.abs(Y) <= SLAB_T / 2), 3.5**2, 1.0)
inv_cs = (1.0 / eps_cs)[None, :, :, None]
E_m, H_m, _ = fdtdx.compute_mode(frequency=fdtdx.constants.c / WAVELENGTH, inv_permittivities=np.asarray(inv_cs), inv_permeabilities=1.0, resolution=res_m, filter_pol="te")
fig = fdtdx.plot_mode(E_m, H_m, inv_permittivity=np.asarray(inv_cs))
fig.savefig(os.path.join(OUT, "bus_mode.png"), bbox_inches="tight", dpi=130)
plt.close(fig)
print("wrote bus_mode.png")

# --- 4. modal transmission (mode expansion) -------------------------------------------------------
a, o, _ = fdtdx.apply_params(arrays, objects, {})
_, sr = fdtdx.run_fdtd(arrays=a, objects=o, config=config, show_progress=False, stopping_condition=EnergyThresholdCondition(min_steps=round(config.time_steps_total / 3)))
states = sr.detector_states
in_name = determine_input_norm_detector_name("in", o)
alpha_in = complex(o[in_name].compute_overlap(states[in_name])[0])
decomp = fdtdx.compute_mode_expansion(o["through"], states["through"], a, config, [("te", 0), ("te", 1), ("tm", 0)], input_overlap=alpha_in)
decomp.plot(filename=os.path.join(OUT, "mode_expansion.png"))
print("wrote mode_expansion.png")

# --- 5. analytical-vs-FDTD ring response ----------------------------------------------------------
def bus_neff(wl, res):
    nxx, nyy = round(2.0e-6 / res), round(1.2e-6 / res)
    Xg, Yg = np.meshgrid((np.arange(nxx) - nxx / 2 + 0.5) * res, (np.arange(nyy) - nyy / 2 + 0.5) * res, indexing="ij")
    e = np.where((np.abs(Xg) <= WG / 2 * 1e-6) & (np.abs(Yg) <= SLAB_T / 2), 12.25, 1.0)
    _, _, ne = fdtdx.compute_mode(frequency=fdtdx.constants.c / wl, inv_permittivities=np.asarray((1 / e)[None, :, :, None]), inv_permeabilities=1.0, resolution=res, filter_pol="te")
    return float(np.real(ne))


L_ring = 2 * np.pi * R * 1e-6
n_eff = bus_neff(WAVELENGTH, RES)
n_g = n_eff - WAVELENGTH * (bus_neff(WAVELENGTH + 10e-9, RES) - bus_neff(WAVELENGTH - 10e-9, RES)) / 2e-8
FSR = WAVELENGTH**2 / (n_g * L_ring)


def allpass_T(wl, t=0.90, a=0.80):
    # Through-port transmission of an all-pass ring; dips at resonance require round-trip loss a<1
    # (a lossless ring is "all-pass": |T|=1, only the phase resonates).
    phi = 2 * np.pi * n_eff * L_ring / wl
    return (a**2 - 2 * a * t * np.cos(phi) + t**2) / (1 - 2 * a * t * np.cos(phi) + (a * t) ** 2)


sweep = np.linspace(1.50e-6, 1.62e-6, 9)
Ts = []
for wl in sweep:
    o2, a2, c2 = make_setup(wl)
    a2, o2, _ = fdtdx.apply_params(a2, o2, {})
    _, sr2 = fdtdx.run_fdtd(arrays=a2, objects=o2, config=c2, show_progress=False, stopping_condition=EnergyThresholdCondition(min_steps=round(c2.time_steps_total / 3)))
    s2 = sr2.detector_states
    ai = complex(o2[determine_input_norm_detector_name("in", o2)].compute_overlap(s2[determine_input_norm_detector_name("in", o2)])[0])
    Ts.append(fdtdx.compute_mode_expansion(o2["through"], s2["through"], a2, c2, [("te", 0)], input_overlap=ai).channels[0].transmission)

fine = np.linspace(1.50e-6, 1.62e-6, 600)
fig, ax = plt.subplots(figsize=(7, 3.3))
ax.plot(fine * 1e9, [allpass_T(w) for w in fine], "k-", lw=1, label="analytical all-pass (t=0.90)")
ax.plot(sweep * 1e9, Ts, "o", color="tab:blue", label="FDTD mode-expansion T(TE0)")
ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("through transmission"); ax.set_ylim(0, 1.3)
ax.set_title(f"Ring response — analytical FSR ≈ {FSR * 1e9:.0f} nm")
ax.legend(loc="lower right", fontsize=8)
fig.savefig(os.path.join(OUT, "ring_response.png"), bbox_inches="tight", dpi=150)
plt.close(fig)
print("wrote ring_response.png")
print("done →", os.path.abspath(OUT))
