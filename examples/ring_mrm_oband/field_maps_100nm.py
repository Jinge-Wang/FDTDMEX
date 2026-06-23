"""Standalone generator for the operating-gap (100 nm) field-trapping figure.

Runs one forward simulation of the racetrack ring + bus at the **operating gap (100 nm)** — the
deepest-extinction point of the gap sweep — recording the through-port spectrum AND an in-plane phasor
monitor at the silicon-core mid-plane at the same wavelengths, then plots ``|E|²`` on resonance (the
through-port dip; light circulating in the ring) versus off resonance (the transmission peak; light
passing straight through the bus). Writes ``figures/field_maps_100nm.png`` (the filename carries the
gap size).

This is intentionally separate from the main study (``ring_mrm_oband.py``): the field map is the only
chapter that benefits from being shown at the operating gap rather than the 180 nm cold-run gap, and
isolating it keeps the main notebook's runtime down. The geometry/scene helpers are copied verbatim
from the main script so the device is byte-for-byte identical.

Run (≈25 min at the 25 nm gap-sweep grid on the current Metal engine; set MRM_FMRES for a coarser/
faster preview):

    cd examples/ring_mrm_oband
    uv run --extra viz python field_maps_100nm.py
"""

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

FIG = os.path.join(os.path.dirname(__file__) if "__file__" in globals() else ".", "figures")
os.makedirs(FIG, exist_ok=True)

# Materials + geometry — identical to ring_mrm_oband.py.
N_SI, N_OX = 3.476, 1.444
MAT = {"si": fdtdx.Material(permittivity=N_SI**2), "ox": fdtdx.Material(permittivity=N_OX**2)}
R, WG, LC = 2.5, 0.50, 1.5
CORE_T = 0.22e-6
LAMBDA0 = 1.31e-6

# Run configuration for this figure.
GAP = 0.10e-6                                    # operating gap (deepest extinction in the sweep)
RES = float(os.environ.get("MRM_FMRES", 25e-9))  # the gap-sweep grid, so the resonance matches gap_sweep.png
SETTLE = 3.5e-12                                 # match the production settle (ring needs a long ring-down)

# Sample the through-port phasor monitor AND the in-plane field at the SAME wavelengths, so on/off
# resonance are read straight off the through-port dip/peak (identical to the gap-sweep methodology) and
# the field slices are available at exactly those wavelengths — no re-derivation from field intensity.
WLS = np.linspace(1.298e-6, 1.320e-6, 18)        # brackets the 100 nm-gap resonance (~1308 nm at 25 nm)
BAND = (float(WLS[0]), float(WLS[-1]), len(WLS))  # in/thru monitors sample the same grid as the field map
XY_WLS = WLS


def write_gds(gap, with_ring=True):
    """Racetrack ring (outer Si + inner oxide carve) + straight bus. `gap` (m) is the bus–ring edge gap."""
    CY = WG + gap * 1e6 + R
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("MRM")
    cell.add(gdstk.rectangle((-4.0, -WG / 2), (4.0, WG / 2), layer=1))
    if with_ring:
        cell.add(gdstk.racetrack((0, CY), LC, R + WG / 2, layer=1, tolerance=2e-3))
        cell.add(gdstk.racetrack((0, CY), LC, R - WG / 2, layer=2, tolerance=2e-3))
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
    YBUS = 0.8e-6
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
    objs = [(load(1, 0, "si"), (LX / 2, YBUS, LZ / 2))]
    if with_ring:
        objs += [(load(1, 1, "si"), (LX / 2, yring, LZ / 2)),
                 (load(2, 0, "ox"), (LX / 2, yring, LZ / 2))]
    for poly, off in objs:
        ol.append(poly); cons.append(at(poly, off))

    wls = np.linspace(*band)
    wcs = tuple(fdtdx.WaveCharacter(wavelength=float(w)) for w in wls)
    cwc = fdtdx.WaveCharacter(wavelength=LAMBDA0)
    prof = fdtdx.GaussianPulseProfile(center_wave=cwc,
                                      spectral_width=fdtdx.WaveCharacter(wavelength=LAMBDA0 * 18))
    W, H = 1.2e-6, 0.5e-6
    src = fdtdx.GaussianPlaneSource(
        partial_grid_shape=(1, None, None), partial_real_shape=(None, W, H),
        fixed_E_polarization_vector=(0, 1, 0), wave_character=cwc, temporal_profile=prof,
        radius=0.5e-6, std=1 / 3, direction="+", name="src")
    ol.append(src); cons.append(at(src, (1.0e-6, YBUS, LZ / 2)))
    comps = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    inm = fdtdx.PhasorDetector(wave_characters=wcs, components=comps, partial_real_shape=(res, W, H), name="in")
    ol.append(inm); cons.append(at(inm, (1.9e-6, YBUS, LZ / 2)))
    thr = fdtdx.PhasorDetector(wave_characters=wcs, components=comps, partial_real_shape=(res, W, H), name="thru")
    ol.append(thr); cons.append(at(thr, (LX - 1.0e-6, YBUS, LZ / 2)))
    if xy_wls is not None:
        xy = fdtdx.PhasorDetector(wave_characters=tuple(fdtdx.WaveCharacter(wavelength=float(w)) for w in xy_wls),
                                  components=("Ex", "Ey", "Ez"), partial_grid_shape=(None, None, 1),
                                  reduce_volume=False, name="xy")
        ol.append(xy)
        cons.append(xy.place_relative_to(vol, axes=2, own_positions=0, other_positions=-1,
                                         margins=LZ / 2 + pml * res))

    cfg = fdtdx.SimulationConfig(time=settle, grid=fdtdx.UniformGrid(spacing=res))
    last = None
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
    """Per-frequency net Poynting flux  ½·Re ∮ (E×H*)·n̂ dA  through the monitor plane (E/H phasors)."""
    ph = np.asarray(state["phasor"])[0]                 # (n_freq, 6, *plane)
    E, H = ph[:, :3], ph[:, 3:]
    ax, ay = [1, 2, 0][prop_axis], [2, 0, 1][prop_axis]
    Sx = E[:, ax] * np.conj(H[:, ay]) - E[:, ay] * np.conj(H[:, ax])
    return 0.5 * np.real(Sx.reshape(Sx.shape[0], -1).sum(axis=1))


def run_scene(res, gap, band, settle, xy_wls):
    o, a, cfg, wls = build_scene(res, gap, band, settle, with_ring=True, xy_wls=xy_wls)
    t0 = time.time()
    _, r = fdtdx.run_fdtd(arrays=a, objects=o, config=cfg, show_progress=False)
    s = r.detector_states
    P_thru = net_power(s["thru"])                                # through-port net Poynting flux vs λ
    ph = np.asarray(s["xy"]["phasor"])
    xy_E2 = (np.abs(ph[0]) ** 2).sum(axis=1)[:, :, :, 0]          # (n_freq, nx, ny)
    invE = np.asarray(a.inv_permittivities)
    eps_xy = 1.0 / invE[0, :, :, invE.shape[3] // 2]             # (nx, ny)
    print(f"  ring res={res * 1e9:.0f}nm gap={gap * 1e9:.0f}nm grid={tuple(int(v) for v in invE.shape[1:])} "
          f"backend={select_backend(a, o, cfg, None)} wall={time.time() - t0:.0f}s")
    return xy_E2, eps_xy, P_thru, wls


def make_figure(E2, eps_xy, on_idx=0, off_idx=1):
    """Two-panel |E|²: on-resonance (on_idx) and off-resonance (off_idx) bins of XY_WLS."""
    si_xy = eps_xy > 4.0
    fig, axf = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, k, lab in zip(axf, (on_idx, off_idx), ("on-resonance", "off-resonance")):
        mp = E2[k].T
        p = np.percentile(mp, 99.5)
        ax.imshow(np.clip(mp / (p + 1e-30), 0, 1) ** 0.5, origin="lower", aspect="equal", cmap="inferno")
        ax.contour(si_xy.T.astype(float), levels=[0.5], colors="cyan", linewidths=0.4, alpha=0.5)
        ax.set_title(f"|E|²  {lab}  ({XY_WLS[k] * 1e9:.1f} nm)")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Operating gap ({GAP * 1e9:.0f} nm): light circulating in the ring on resonance; "
                 f"passing to the through port off resonance")
    out = os.path.join(FIG, f"field_maps_{GAP * 1e9:.0f}nm.png")    # filename carries the gap size
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  wrote {out}")


def main():
    print(f"figures -> {FIG}  |  {GAP * 1e9:.0f} nm operating gap, res={RES * 1e9:.0f} nm")
    E2, eps_xy, P_thru, wls = run_scene(RES, GAP, BAND, SETTLE, XY_WLS)
    nm = wls * 1e9
    # On/off-resonance straight from the through-port spectrum (same definition as the gap-sweep chart):
    # the resonance is the deepest dip, off-resonance the highest-transmission wavelength.
    on_idx, off_idx = int(P_thru.argmin()), int(P_thru.argmax())
    print("  through-port P(λ): " + "  ".join(f"{nm[i]:.0f}:{P_thru[i] / P_thru.max():.2f}" for i in range(len(nm))))
    print(f"  resonance {nm[on_idx]:.1f} nm (through-port dip)   off {nm[off_idx]:.1f} nm (peak)")
    np.savez(os.path.join(FIG, f"field_maps_{GAP * 1e9:.0f}nm_data.npz"),
             E2=E2, eps_xy=eps_xy, P_thru=P_thru, wls_nm=nm)    # so the on/off pick can be re-plotted
    make_figure(E2, eps_xy, on_idx, off_idx)


if __name__ == "__main__":
    main()
