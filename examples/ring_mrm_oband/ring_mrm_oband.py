# %% [markdown]
# # O-band carrier-depletion microring modulator — design verification
#
# A bounded, self-contained study of an **O-band (1310 nm) silicon microring modulator (MRM)**,
# forward-simulated on the **Metal** engine (MLX backend). Five steps:
#
# 1. **Model + mode** — author a racetrack ring + bus, solve the bus TE₀ mode (`n_eff`, `n_g`, `Γ`).
# 2. **Mesh convergence** — cold spectrum from 40 nm down to 20 nm; track the resonance vs grid.
# 3. **Cold run** — through-port `T(λ)`, the resonance (the dip nearest the geometry's reference
#    wavelength), and `|E|²` field maps on / off resonance.
# 4. **Coupling** — through-port `T(λ)` vs bus–ring gap and the extinction-ratio (ER) vs gap trend.
# 5. **Static EO** — a Soref–Bennett free-carrier perturbation → resonance shift vs reverse bias.
#
# **Resolution.** A sensible starting mesh is `λ / (n_eff · 15) ≈ 1310 nm / (2.69·15) ≈ 32 nm`; production
# sign-off uses **20 nm**. We converge 40 → 20 nm and run the chapters at 20 nm.
#
# **Why this recipe.** Mode *sources/detectors* would force the slow JAX/CPU path here, so we excite with a
# broadband **Gaussian** source and read the outlet fields with **phasor monitors**. Transmission is the
# standing-wave-immune **net Poynting flux** with a bus-only reference, `T(λ) = P_thru^ring / P_thru^bus`.
# The FDTD device is a full-etch **strip** (clean, affordable); the rib SOI stack is implicit in the mode /
# EO analysis, where the lateral PN junction lives.

# %%
import os
import tempfile
import time

import gdstk
import matplotlib.pyplot as plt
import numpy as np

import fdtdx
from fdtdx.backend.dispatch import select_backend
from fdtdx.objects.boundaries.initialization import BoundaryConfig, boundary_objects_from_config
from fdtdx.objects.static_material.polygon import extruded_polygon_from_gds_path

try:
    get_ipython().run_line_magic("matplotlib", "inline")  # type: ignore[name-defined]
except Exception:
    pass

FIG = os.path.join(os.path.dirname(__file__) if "__file__" in globals() else ".", "figures")
os.makedirs(FIG, exist_ok=True)

# Materials (O-band): crystalline Si and SiO₂ (BOX + cladding modelled as one oxide background).
N_SI, N_OX = 3.476, 1.444
MAT = {"si": fdtdx.Material(permittivity=N_SI**2), "ox": fdtdx.Material(permittivity=N_OX**2)}

# Device geometry (µm). Compact racetrack to keep the Metal time-loop affordable at 20 nm.
R, WG, LC = 2.5, 0.50, 1.5          # ring radius, waveguide width, racetrack straight length
CORE_T = 0.22e-6                    # SOI device-layer thickness
LAMBDA0 = 1.31e-6                   # O-band design wavelength

# Run configuration. MRM_FAST=1 → a quick coarse smoke run (physics meaningless, exercises the code path).
FAST = os.environ.get("MRM_FAST") == "1"
BAND = (1.285e-6, 1.335e-6, 20 if FAST else 121)         # broadband window (lo, hi, n_points)
SETTLE = 1.5e-12 if FAST else 3.5e-12                    # ring settle time (high-Q needs a long ring-down)
CONV_RES = [60e-9, 40e-9] if FAST else [40e-9, 32e-9, 25e-9, 20e-9]   # mesh-convergence grids
PROD_RES = CONV_RES[-1]                                  # production grid (finest) for cold / fields / EO
GAP_RES = 60e-9 if FAST else 25e-9                       # gap sweep grid (a notch coarser → 4 runs stay feasible)
GAPS = [0.10e-6, 0.26e-6] if FAST else [0.10e-6, 0.18e-6, 0.26e-6, 0.34e-6, 0.42e-6]    # bus–ring gaps (≥2 cells apart)
print(f"figures -> {FIG}  |  FAST={FAST}")


# %% [markdown]
# ## Geometry, scene, and the Metal forward run (helpers)

# %%
def write_gds(gap, with_ring=True):
    """Racetrack ring (outer Si + inner oxide carve) + straight bus. `gap` (m) is the bus–ring edge gap."""
    CY = WG + gap * 1e6 + R                              # bus→ring-centre spacing (µm; gap is metres)
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("MRM")
    cell.add(gdstk.rectangle((-4.0, -WG / 2), (4.0, WG / 2), layer=1))            # bus (layer 1, idx 0)
    if with_ring:
        cell.add(gdstk.racetrack((0, CY), LC, R + WG / 2, layer=1, tolerance=2e-3))   # outer (idx 1)
        cell.add(gdstk.racetrack((0, CY), LC, R - WG / 2, layer=2, tolerance=2e-3))   # inner carve
    path = os.path.join(tempfile.gettempdir(), f"mrm_g{int(gap * 1e9)}_{int(with_ring)}.gds")
    lib.write_gds(path)
    return path, CY


def build_scene(res, gap, band, settle, with_ring=True, xy_wls=None):
    """Strip-ring FDTD scene on Metal: oxide background, Gaussian TE source, in/thru phasor monitors."""
    gp, CY = write_gds(gap, with_ring=with_ring)

    def load(layer, idx, mat):
        p = extruded_polygon_from_gds_path(gp, "MRM", layer=layer, polygon_index=idx, axis=2,
                                           material_name=mat, materials=MAT)
        object.__setattr__(p, "partial_real_shape", (*p.partial_real_shape[:2], CORE_T))
        return p

    LX, LZ = 8.0e-6, 0.8e-6
    YBUS = 0.8e-6                                        # bus offset from the lower interior edge
    LY = (YBUS * 1e6 + CY + R + WG / 2 + 0.6) * 1e-6
    pml = 8
    vol = fdtdx.SimulationVolume(
        partial_real_shape=(LX + 2 * pml * res, LY + 2 * pml * res, LZ + 2 * pml * res),
        material=MAT["ox"], name="bg")
    cons, ol = [], [vol]
    bdict, bcons = boundary_objects_from_config(
        BoundaryConfig.from_uniform_bound(thickness=pml, boundary_type="pml"), vol)
    ol += list(bdict.values()); cons += bcons

    def at(obj, off):
        return obj.place_relative_to(vol, axes=(0, 1, 2), own_positions=(0, 0, 0),
                                     other_positions=(-1, -1, -1),
                                     margins=(off[0] + pml * res, off[1] + pml * res, off[2] + pml * res))

    yring = YBUS + CY * 1e-6
    objs = [(load(1, 0, "si"), (LX / 2, YBUS, LZ / 2))]                          # bus (strip)
    if with_ring:
        objs += [(load(1, 1, "si"), (LX / 2, yring, LZ / 2)),                    # outer ring disk
                 (load(2, 0, "ox"), (LX / 2, yring, LZ / 2))]                    # inner oxide carve
    for poly, off in objs:
        ol.append(poly); cons.append(at(poly, off))

    wls = np.linspace(*band)
    wcs = tuple(fdtdx.WaveCharacter(wavelength=float(w)) for w in wls)
    cwc = fdtdx.WaveCharacter(wavelength=LAMBDA0)
    prof = fdtdx.GaussianPulseProfile(center_wave=cwc,
                                      spectral_width=fdtdx.WaveCharacter(wavelength=LAMBDA0 * 18))
    W, H = 1.2e-6, 0.5e-6                                # source/monitor box: strictly inside the interior
    src = fdtdx.GaussianPlaneSource(
        partial_grid_shape=(1, None, None), partial_real_shape=(None, W, H),
        fixed_E_polarization_vector=(0, 1, 0), wave_character=cwc, temporal_profile=prof,
        radius=0.5e-6, std=1 / 3, direction="+", name="src")
    ol.append(src); cons.append(at(src, (1.0e-6, YBUS, LZ / 2)))                 # ~1 µm off the PML
    comps = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    inm = fdtdx.PhasorDetector(wave_characters=wcs, components=comps, partial_real_shape=(res, W, H), name="in")
    ol.append(inm); cons.append(at(inm, (1.9e-6, YBUS, LZ / 2)))
    thr = fdtdx.PhasorDetector(wave_characters=wcs, components=comps, partial_real_shape=(res, W, H), name="thru")
    ol.append(thr); cons.append(at(thr, (LX - 1.0e-6, YBUS, LZ / 2)))
    if xy_wls is not None:                                                       # in-plane field map (core z)
        xy = fdtdx.PhasorDetector(wave_characters=tuple(fdtdx.WaveCharacter(wavelength=float(w)) for w in xy_wls),
                                  components=("Ex", "Ey", "Ez"), partial_grid_shape=(None, None, 1),
                                  reduce_volume=False, name="xy")
        ol.append(xy)
        cons.append(xy.place_relative_to(vol, axes=2, own_positions=0, other_positions=-1,
                                         margins=LZ / 2 + pml * res))

    cfg = fdtdx.SimulationConfig(time=settle, grid=fdtdx.UniformGrid(spacing=res))
    last = None                          # PML grid-tiling retry: grow the volume by a cell until it resolves
    for _ in range(10):
        try:
            o, a, _, cfg, _ = fdtdx.place_objects(object_list=ol, config=cfg, constraints=cons)
            break
        except ValueError as e:
            last = e; s = vol.partial_real_shape
            object.__setattr__(vol, "partial_real_shape", (s[0] + res, s[1] + res, s[2]))
    else:
        raise last
    a = fdtdx.extend_material_to_pml(objects=o, arrays=a)
    a, o, _ = fdtdx.apply_params(a, o, {})
    return o, a, cfg, wls


def net_power(state, prop_axis=0):
    """Per-frequency net Poynting flux  ½·Re ∮ (ExH*)·n̂ dA  through the monitor plane."""
    ph = np.asarray(state["phasor"])[0]                 # (n_freq, 6, *plane)
    E, H = ph[:, :3], ph[:, 3:]
    ax, ay = [1, 2, 0][prop_axis], [2, 0, 1][prop_axis]
    Sx = E[:, ax] * np.conj(H[:, ay]) - E[:, ay] * np.conj(H[:, ax])
    return 0.5 * np.real(Sx.reshape(Sx.shape[0], -1).sum(axis=1))


def run_scene(res, gap, band, settle, with_ring=True, xy_wls=None):
    o, a, cfg, wls = build_scene(res, gap, band, settle, with_ring=with_ring, xy_wls=xy_wls)
    t0 = time.time()
    _, r = fdtdx.run_fdtd(arrays=a, objects=o, config=cfg, show_progress=False)
    s = r.detector_states
    out = {"wls": wls, "P_thru": net_power(s["thru"]), "wall": time.time() - t0,
           "grid": tuple(int(v) for v in a.inv_permittivities.shape[1:])}
    if xy_wls is not None:
        ph = np.asarray(s["xy"]["phasor"])
        out["xy_E2"] = (np.abs(ph[0]) ** 2).sum(axis=1)[:, :, :, 0]              # (n_freq, nx, ny)
        invE = np.asarray(a.inv_permittivities)
        out["eps_xy"] = 1.0 / invE[0, :, :, invE.shape[3] // 2]
    print(f"  {'ring' if with_ring else 'bus '} res={res * 1e9:.0f}nm gap={gap * 1e9:.0f}nm "
          f"grid={out['grid']} backend={select_backend(a, o, cfg, None)} wall={out['wall']:.0f}s")
    return out


_BUSREF = {}


def transmission(res, gap, band, settle, xy_wls=None):
    """Two-run net-Poynting transmission T(λ) = P_thru(ring) / P_thru(bus-only). Bus ref cached per grid."""
    if res not in _BUSREF:
        _BUSREF[res] = run_scene(res, 0.18e-6, band, settle, with_ring=False)["P_thru"]
    ring = run_scene(res, gap, band, settle, with_ring=True, xy_wls=xy_wls)
    return ring["wls"], ring["P_thru"] / _BUSREF[res], ring


def resonance_metrics(wls, T, target, edge_frac=0.15):
    """Fit the deepest dip nearest `target`: returns (λ0, T_min, baseline, FWHM, Q, ER_dB).

    Band edges are excluded (low pulse power → noisy ratio); `target` (the geometry's reference
    wavelength) selects the resonance order so the same dip is tracked across grids and gaps.
    """
    n = len(wls); m = max(2, int(n * edge_frac))
    w, t = wls[m:n - m], T[m:n - m]
    i0 = int(np.argmin(np.where(np.abs(w - target) < 11e-9, t, 2.0)))   # deepest dip within ±11 nm of target
    baseline = min(float(np.max(t)), 1.0)
    tmin, lam0 = float(t[i0]), float(w[i0])
    half = 0.5 * (baseline + tmin)
    lo = i0
    while lo > 0 and t[lo] < half:
        lo -= 1
    hi = i0
    while hi < len(t) - 1 and t[hi] < half:
        hi += 1
    fwhm = abs(w[hi] - w[lo]) if hi > lo else float(w[1] - w[0])
    return lam0, tmin, baseline, fwhm, (lam0 / fwhm if fwhm else float("nan")), 10 * np.log10(baseline / max(tmin, 1e-6))


# %% [markdown]
# # 1 — Model and waveguide mode
# The bus propagates along **x**, so its cross-section is the **y–z plane** (y = width, z = thickness). We
# solve the fundamental TE₀ mode of the as-simulated strip (full 220 nm etch, oxide-clad) and read off
# `n_eff`, the group index `n_g = n_eff − λ·dn_eff/dλ`, and the modal confinement `Γ`. The geometry then
# fixes a **reference resonance** `λ_ref = n_eff·L/m` (L = round-trip length) used to locate the dip later.

# %%
MODE_RES = 10e-9


def bus_cross_section(res):
    """Strip bus cross-section in the y–z plane → (eps[ny,nz], ys, masks)."""
    WG_m = WG * 1e-6                                     # WG is in µm; the grid is in metres
    ny, nz = round(2.0e-6 / res), round(1.2e-6 / res)
    ys = (np.arange(ny) - ny / 2 + 0.5) * res
    zs = (np.arange(nz) - nz / 2 + 0.5) * res
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    core = (np.abs(Y) <= WG_m / 2) & (np.abs(Z) <= CORE_T / 2)
    eps = np.where(core, N_SI**2, N_OX**2)
    dep = (np.abs(Y) <= 0.11e-6) & core                 # lateral-junction depletion window (y≈0)
    return eps, ys, {"si": core, "dep": dep}


def solve_bus_mode(wl):
    import jax.numpy as jnp
    eps, ys, masks = bus_cross_section(MODE_RES)
    E, H, neff = fdtdx.compute_mode(frequency=fdtdx.constants.c / wl,
                                    inv_permittivities=jnp.asarray((1.0 / eps)[None, :, :, None]),
                                    inv_permeabilities=1.0, resolution=MODE_RES, filter_pol="te")
    return E, H, complex(neff), eps, ys, masks


E_m, H_m, neff0, eps_cs, ys_cs, masks = solve_bus_mode(LAMBDA0)
dl = 10e-9
n_eff = neff0.real
n_g = n_eff - LAMBDA0 * (solve_bus_mode(LAMBDA0 + dl)[2].real - solve_bus_mode(LAMBDA0 - dl)[2].real) / (2 * dl)

Em = np.asarray(E_m).reshape(3, *eps_cs.shape)
u = (eps_cs[None] * np.abs(Em) ** 2).sum(axis=0)        # modal energy density (ny, nz)
core_cs = masks["si"]
GAMMA_CORE = float(u[core_cs].sum() / u.sum())
GAMMA_DEP = float(u[masks["dep"]].sum() / u.sum())

L_ring = (2 * np.pi * R + 2 * LC) * 1e-6                 # round-trip length (m)
m_order = round(n_eff * L_ring / LAMBDA0)
LAM_REF = n_eff * L_ring / m_order                       # geometry's reference resonance wavelength
print(f"bus TE0 @ {LAMBDA0 * 1e9:.0f} nm:  n_eff = {n_eff:.4f}   n_g = {n_g:.4f}")
print(f"Γ_core = {GAMMA_CORE:.3f}   Γ_dep = {GAMMA_DEP:.3f}   λ_ref(straight n_eff) ≈ {LAM_REF * 1e9:.1f} nm "
      f"(m={m_order}); resonances tracked near the design λ {LAMBDA0 * 1e9:.0f} nm (bends lower n_eff)")

# Mode figure: index cross-section + |E|².
fig_mode, (am1, am2) = plt.subplots(1, 2, figsize=(11, 3.6))
ys_um = ys_cs * 1e6
zs_um = (np.arange(eps_cs.shape[1]) - eps_cs.shape[1] / 2 + 0.5) * MODE_RES * 1e6
for ax, data, title in ((am1, eps_cs, "index n²"), (am2, u / u.max(), "modal |E|²")):
    ax.pcolormesh(ys_um, zs_um, data.T, cmap="viridis" if title.startswith("index") else "inferno", shading="auto")
    ax.set_aspect("equal"); ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.45, 0.45)
    ax.set_xlabel("y — width (µm)"); ax.set_ylabel("z — thickness (µm)"); ax.set_title(title)
am2.contour(ys_um, zs_um, eps_cs.T, levels=[6.0], colors="cyan", linewidths=0.6)
fig_mode.suptitle(f"Bus TE₀ mode  —  n_eff={n_eff:.3f},  n_g={n_g:.2f},  Γ_core={GAMMA_CORE:.2f}")
fig_mode.savefig(os.path.join(FIG, "mode.png"), dpi=120, bbox_inches="tight")
fig_mode

# %% [markdown]
# Simulation setup — top-down permittivity (racetrack ring side-coupled to the bus).

# %%
_o, _a, _cfg, _ = build_scene(PROD_RES, 0.18e-6, (1.30e-6, 1.32e-6, 3), 100e-15)
_eps = 1.0 / np.asarray(_a.inv_permittivities)[0, :, :, np.asarray(_a.inv_permittivities).shape[3] // 2]
print("production grid:", tuple(int(v) for v in _a.inv_permittivities.shape[1:]),
      "| forward backend:", select_backend(_a, _o, _cfg, None))
fig_set, axs = plt.subplots(figsize=(7.0, 4.6))
_lx, _ly = _eps.shape[0] * PROD_RES * 1e6, _eps.shape[1] * PROD_RES * 1e6  # centered axes (origin at domain center)
axs.imshow(_eps.T, origin="lower", extent=[-_lx / 2, _lx / 2, -_ly / 2, _ly / 2],
           cmap="viridis", aspect="equal")
axs.set_title("Simulation setup — top-down ε (ring + bus)"); axs.set_xlabel("x (µm)"); axs.set_ylabel("y (µm)")
fig_set.savefig(os.path.join(FIG, "setup.png"), dpi=120, bbox_inches="tight")
fig_set

# %% [markdown]
# # 2 — Mesh convergence (40 → 20 nm)
# The cold spectrum is re-run from 40 nm down to 20 nm. We track the resonance nearest `λ_ref` and its
# loaded Q vs grid: as the mesh refines, FDTD numerical dispersion shrinks and both settle toward their
# physical values. The finest (20 nm) run is reused as the production cold run in §3.

# %%
conv = {}
for res in CONV_RES:
    w, T, ring = transmission(res, 0.18e-6, BAND, SETTLE)
    lam0, tmin, base, fwhm, Q, ER = resonance_metrics(w, T, LAMBDA0)
    conv[res] = {"w": w, "T": T, "lam0": lam0, "Q": Q, "ER": ER, "ring": ring}
    print(f"  res {res * 1e9:.0f} nm -> λ_res {lam0 * 1e9:.2f} nm,  Q {Q:.0f},  ER {ER:.1f} dB")

fig_cv, (ac1, ac2) = plt.subplots(1, 2, figsize=(11, 3.8))
rr = [r * 1e9 for r in CONV_RES]
ac1.plot(rr, [conv[r]["lam0"] * 1e9 for r in CONV_RES], "-o", color="tab:green")
ac1.axhline(LAMBDA0 * 1e9, color="0.6", ls="--", lw=0.8, label="design λ (1310 nm)")
ac1.set_xlabel("grid spacing (nm)"); ac1.set_ylabel("resonance λ_res (nm)")
ac1.set_title("Resonance convergence"); ac1.legend(fontsize=8); ac1.invert_xaxis()
ac2.plot(rr, [conv[r]["Q"] for r in CONV_RES], "-o", color="tab:green")
ac2.set_xlabel("grid spacing (nm)"); ac2.set_ylabel("loaded Q"); ac2.set_title("Loaded-Q convergence"); ac2.invert_xaxis()
fig_cv.savefig(os.path.join(FIG, "convergence.png"), dpi=120, bbox_inches="tight")
fig_cv

# %% [markdown]
# # 3 — Cold run: spectrum, resonance, Q, and extinction ratio
# At the production grid: the through-port `T(λ)` with the fitted loaded Q and extinction ratio. The
# `|E|²` field maps that show the resonant field trapped circulating in the ring are rendered separately
# at the operating gap — see "Operating-gap field maps" after the gap sweep (`field_maps_100nm.py`).

# %%
prod = conv[PROD_RES]
wls, T_cold = prod["w"], prod["T"]
lam0, tmin, base, fwhm, Q0, ER0 = resonance_metrics(wls, T_cold, LAMBDA0)
cc = slice(3, -3)
print(f"cold: λ_res {lam0 * 1e9:.2f} nm, Q {Q0:.0f}, ER {ER0:.1f} dB, FSR≈{LAMBDA0**2 / (n_g * L_ring) * 1e9:.1f} nm")

fig_cold, axc = plt.subplots(figsize=(7.4, 3.8))
axc.plot(wls[cc] * 1e9, T_cold[cc], "-", color="tab:blue")
axc.axvline(lam0 * 1e9, color="tab:red", ls="--", lw=0.8)
axc.annotate(f"Q ≈ {Q0:.0f}\nER ≈ {ER0:.1f} dB", xy=(lam0 * 1e9, tmin), xytext=(lam0 * 1e9 + 4, 0.45),
             fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.7))
axc.set_xlabel("wavelength (nm)"); axc.set_ylabel("through transmission T"); axc.set_ylim(0, 1.1)
axc.set_title(f"Cold through-port spectrum ({PROD_RES * 1e9:.0f} nm, 180 nm gap)")
fig_cold.savefig(os.path.join(FIG, "cold_spectrum.png"), dpi=120, bbox_inches="tight")
fig_cold

# %% [markdown]
# # 4 — Coupling: through-port spectra and ER vs bus–ring gap
# Sweeping the gap traces the coupling regime. For an all-pass ring the on-resonance depth is
# `T_min = (a−t)²/(1−a·t)²`: the gap sets the self-coupling `t` (smaller gap → smaller `t`), while the
# round-trip amplitude `a` is fixed by the ring loss; extinction peaks at critical coupling `t = a`. This
# lossy compact ring is **under-coupled across the whole sweep** (`t ≈ 0.78→0.97` vs `a ≈ 0.52`), so ER is
# deepest at the **smallest** gap and falls as the gap widens. The operating gap is the deepest (max ER).

# %%
gap_T, gap_m = {}, {}
for g in GAPS:
    w, Tg, _ = transmission(GAP_RES, g, BAND, SETTLE)
    gap_T[g] = (w, Tg); gap_m[g] = resonance_metrics(w, Tg, LAMBDA0)
    print(f"  gap {g * 1e9:.0f} nm -> T_min {gap_m[g][1]:.3f}, ER {gap_m[g][5]:.1f} dB, Q {gap_m[g][4]:.0f}")
op_gap = max(GAPS, key=lambda g: gap_m[g][5])
print(f"operating gap = {op_gap * 1e9:.0f} nm (max ER ≈ {gap_m[op_gap][5]:.1f} dB)")

fig_gap, (ag1, ag2) = plt.subplots(1, 2, figsize=(12, 4.2))
for g in GAPS:
    w, Tg = gap_T[g]; ag1.plot(w[cc] * 1e9, Tg[cc], "-", label=f"{g * 1e9:.0f} nm")
ag1.set_xlabel("wavelength (nm)"); ag1.set_ylabel("through transmission T"); ag1.set_ylim(0, 1.1)
ag1.legend(title="bus–ring gap", fontsize=8); ag1.set_title("Through-port spectra vs gap")
ag2.plot([g * 1e9 for g in GAPS], [gap_m[g][5] for g in GAPS], "-o", color="tab:purple")
ag2.axvline(op_gap * 1e9, color="tab:red", ls="--", lw=0.8, label="operating gap")
ag2.set_xlabel("bus–ring gap (nm)"); ag2.set_ylabel("extinction ratio (dB)")
ag2.set_title("ER vs gap — coupling control"); ag2.legend(fontsize=8)
fig_gap.savefig(os.path.join(FIG, "gap_sweep.png"), dpi=120, bbox_inches="tight")
fig_gap

# %% [markdown]
# ## Operating-gap field maps
# At the operating gap (100 nm — the deepest-extinction point of the sweep) the resonant field is
# clearest: `|E|²` at the silicon-core mid-plane circulates **inside the ring** on resonance and **passes
# straight through the bus** off resonance. Produced by a separate ~6 min Metal run,
# [`field_maps_100nm.py`](field_maps_100nm.py): it records the through-port spectrum and the field at the
# same wavelengths and reads on-resonance = the through-port dip (≈1307 nm, the resonance the gap sweep
# shows) and off-resonance = the transmission peak (≈1298 nm). Shown here rather than re-run in the study.

# %%
from IPython.display import Image  # noqa: E402

_fm = os.path.join(FIG, "field_maps_100nm.png")
Image(filename=_fm) if os.path.exists(_fm) else print(f"run field_maps_100nm.py to generate {_fm}")

# %% [markdown]
# # 5 — Static electro-optic response (Soref–Bennett)
# Reverse bias widens the lateral-junction depletion region, sweeping carriers **out** of the mode →
# silicon index **up** → resonance **red-shifts** (loss drops slightly). We weight the bulk Soref–Bennett
# change (O-band coefficients, Nedeljkovic *et al.* 2011) by the modal overlap with the *newly depleted*
# shell, `Γ(W(V)) − Γ(W(0))`. Optical-only — not an RF/thermal/transient prediction.

# %%
q, eps0, kT_q = 1.602e-19, 8.854e-12, 0.02585
EPS_SI, NI, NA, ND = 11.7, 1.0e10, 2.0e18, 2.0e18       # silicon; PDK doping (cm⁻³)
V_bi = kT_q * np.log(NA * ND / NI**2)


def depletion_width(Vr):
    """Abrupt-junction depletion width (m) at reverse bias |Vr| (V)."""
    return np.sqrt(2 * EPS_SI * eps0 * (V_bi + Vr) / q * (1 / (NA * 1e6) + 1 / (ND * 1e6)))


# Γ within |y| < half-width, interpolated from the cumulative modal energy (smooth, not mode-grid-quantized).
_absy = np.sort(np.abs(ys_cs))
_cum = np.cumsum(((u * core_cs).sum(axis=1))[np.argsort(np.abs(ys_cs))])


def gamma_halfwidth(hw):
    return float(np.interp(hw, _absy, _cum) / u.sum())


# Bulk index change for removing the doping-level carriers; 0.5 = abrupt-junction symmetry (half n-, half p-side).
dn_remove = 0.5 * (3.0e-22 * ND**1.011 + 5.4e-18 * NA**0.838)
Vsweep = np.linspace(0, 6, 25)
g0 = gamma_halfwidth(depletion_width(0.0) / 2)
dlam = np.array([LAMBDA0 * dn_remove * (gamma_halfwidth(depletion_width(V) / 2) - g0) / n_g for V in Vsweep])
print(f"V_bi {V_bi:.2f} V, W: {depletion_width(0) * 1e9:.0f}→{depletion_width(6) * 1e9:.0f} nm; "
      f"tuning ≈ {dlam[-1] / 6 * 1e12:.1f} pm/V (Δλ(6V) ≈ +{dlam[-1] * 1e12:.0f} pm, red shift)")

fig_eo, (ae1, ae2) = plt.subplots(1, 2, figsize=(12, 4.0))
ae1.plot(Vsweep, dlam * 1e12, "-o", ms=3, color="tab:green")
ae1.set_xlabel("reverse bias |V| (V)"); ae1.set_ylabel("resonance shift Δλ_res (pm)")
ae1.set_title("Static EO tuning (red shift)")
for V, col in [(0.0, "tab:blue"), (3.0, "tab:orange"), (6.0, "tab:red")]:
    w_op, T_op = gap_T[op_gap]
    ae2.plot(w_op[cc] * 1e9 + float(np.interp(V, Vsweep, dlam * 1e9)), T_op[cc], "-", color=col, label=f"|V|={V:.0f} V")
ae2.set_xlabel("wavelength (nm)"); ae2.set_ylabel("through transmission T"); ae2.set_ylim(0, 1.1)
ae2.legend(fontsize=8); ae2.set_title(f"EO modulation at the operating gap ({op_gap * 1e9:.0f} nm)")
fig_eo.savefig(os.path.join(FIG, "eo_response.png"), dpi=120, bbox_inches="tight")
fig_eo

# %% [markdown]
# ## Operating point (optical-only)

# %%
operating_point = {
    "lambda_res_nm": round(lam0 * 1e9, 3), "Q_loaded": round(float(Q0)),
    "ER_dB": round(float(gap_m[op_gap][5]), 2), "operating_gap_nm": round(op_gap * 1e9),
    "n_eff": round(n_eff, 4), "n_g": round(n_g, 4), "Gamma_core": round(GAMMA_CORE, 3),
    "tuning_pm_per_V": round(dlam[-1] / 6 * 1e12, 2), "dlambda_6V_pm": round(dlam[-1] * 1e12, 1),
    "scope": "optical-only free-carrier prediction; NOT an RF/thermal/transient result",
}
np.savez(os.path.join(FIG, "operating_point.npz"), wls=wls, T_cold=T_cold, Vsweep=Vsweep, dlam_pm=dlam * 1e12)
for k, v in operating_point.items():
    print(f"  {k}: {v}")
