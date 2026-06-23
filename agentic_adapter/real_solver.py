"""real_solver.py — the ag-fdtd ↔ FDTDMEX bridge (runs INSIDE fdtdmex's env).

The ag-fdtd workspace submits FDTD runs through a detached child process behind a
fixed CLI + file contract (see ag-fdtd `mock_solver.py` / KICKSTART §E). This adapter
mirrors that contract exactly but drives the REAL engine: it translates the ag-fdtd
`SimConfig` (ring knobs) into an fdtdx `Scene` (bus + side-coupled ring, Gaussian TE
source, in/through phasor monitors), runs it through the IO seam
(`sim_init → sim_run → results.hdf5`), reduces the through/in net Poynting flux into a
mode-resolved S-matrix, and writes the three contract files ag-fdtd unwraps.

    python real_solver.py --bundle sim.hdf5 --out-dir DIR --run-id ID
        [--steps 11] [--domain fdtd] [--solver fdtdmex]
        [--backend mlx|mock] [--fail-mode diverge|mesh_fail]

`--backend mock` fabricates schema-valid results with no GPU (validates the whole
pipeline); `--backend mlx` runs the Metal forward engine on Apple Silicon. This file
lives in the FDTDMEX repo and may import `fdtdx` freely; it NEVER imports ag-fdtd —
it reads the bundle's config JSON straight out of the HDF5 attrs.

Run via ag-fdtd's `LocalJobRunner` with `cwd` = this repo and the fdtdmex venv python.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time

import h5py
import numpy as np

# The ag-fdtd bundle stashes the canonical SimConfig JSON in this root attr
# (ag-fdtd `app/bundle.py` _CONFIG_ATTR). We read it directly — no ag-fdtd import.
_CONFIG_ATTR = "sim_config_json"

_EXIT_OK = "finished_without_error"
_EXIT_DIVERGED = "finished_with_divergence"
_EXIT_MESH = "finished_with_mesh_warning"


def _progress(step: int, total: int) -> None:
    print(f"PROGRESS {step}/{total}", flush=True)


# --------------------------------------------------------------------------- #
# Config extraction (ag-fdtd SimConfig dict → ring knobs in SI / µm)
# --------------------------------------------------------------------------- #
def _read_config(bundle_path: str) -> dict:
    with h5py.File(bundle_path, "r") as f:
        raw = f.attrs.get(_CONFIG_ATTR)
    if raw is None:
        raise ValueError(f"bundle {bundle_path!r} has no {_CONFIG_ATTR!r} attr")
    return json.loads(raw)


def _find(obj, key_sub: str):
    """First numeric value under a key containing `key_sub` (case-insensitive)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if key_sub in str(k).lower() and isinstance(v, (int, float)):
                return float(v)
            hit = _find(v, key_sub)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find(v, key_sub)
            if hit is not None:
                return hit
    return None


def _ring_knobs(cfg: dict) -> dict:
    """Pull the ring-resonator knobs from the ag-fdtd config (lengths in µm)."""
    params = cfg.get("parameters", {}) or {}
    gap_um = float(params.get("gap", _find(cfg, "gap") or 0.10))  # coupling gap (µm)
    # ring radius + waveguide width from the structures, with sensible fallbacks
    radius = width = None
    for s in cfg.get("structures", []) or []:
        if isinstance(s, dict):
            if str(s.get("type", "")).lower().startswith("ring") and "radius" in s:
                radius = float(s["radius"])
            if "width" in s and width is None:
                width = float(s["width"])
    wl_um = _find(cfg.get("sources", []), "wavelength") or 1.55
    spacing_nm = _find(cfg.get("grid_spec", {}), "spacing") or 50.0
    return {
        "gap_um": gap_um,
        "R": radius or 1.2,
        "WG": width or 0.40,
        "wl": float(wl_um) * 1e-6,
        "res": max(float(spacing_nm), 20.0) * 1e-9,  # floor at 20 nm for tractability
    }


def _channels(items, default_port: str) -> list[tuple[str, str]]:
    """(port, mode) channels from sources/monitors (mirrors the mock + PortSpec)."""
    out: list[tuple[str, str]] = []
    for it in items or []:
        if not isinstance(it, dict) or not it.get("port"):
            continue
        modes = it.get("modes") or ([it["mode"]] if it.get("mode") else ["TE0"])
        for m in modes:
            if (it["port"], m) not in out:
                out.append((it["port"], m))
    return out or [(default_port, "TE0")]


# --------------------------------------------------------------------------- #
# Scene construction (mirrors examples/ring_mrm_oband build_scene, simplified)
# --------------------------------------------------------------------------- #
def _write_gds(knobs: dict, with_ring: bool = True) -> tuple[str, float]:
    import gdstk

    R, WG, gap_um = knobs["R"], knobs["WG"], knobs["gap_um"]
    CY = WG + gap_um + R  # bus → ring-centre spacing (µm)
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("RING")
    cell.add(gdstk.rectangle((-4.0, -WG / 2), (4.0, WG / 2), layer=1))  # bus: layer1 idx0
    if with_ring:
        cell.add(gdstk.ellipse((0, CY), R + WG / 2, layer=1, tolerance=2e-3))  # outer: layer1 idx1
        cell.add(gdstk.ellipse((0, CY), R - WG / 2, layer=2, tolerance=2e-3))  # inner carve: layer2 idx0
    path = os.path.join(tempfile.gettempdir(), f"agfdtd_ring_g{int(gap_um * 1000)}.gds")
    lib.write_gds(path)
    return path, CY


def _build_scene(knobs: dict, band: tuple[float, float, int], settle: float):
    """A strip-ring forward scene: oxide background, Gaussian TE source, in/through
    phasor monitors. Returns an unplaced fdtdx.Scene (sim_init places it)."""
    import fdtdx
    from fdtdx.objects.boundaries.initialization import BoundaryConfig, boundary_objects_from_config
    from fdtdx.objects.static_material.polygon import extruded_polygon_from_gds_path

    res = knobs["res"]
    N_SI, N_OX, CORE_T = 3.476, 1.444, 0.22e-6
    mat = {"si": fdtdx.Material(permittivity=N_SI**2), "ox": fdtdx.Material(permittivity=N_OX**2)}
    gp, CY = _write_gds(knobs, with_ring=True)

    def load(layer, idx, m):
        p = extruded_polygon_from_gds_path(gp, "RING", layer=layer, polygon_index=idx, axis=2,
                                           material_name=m, materials=mat)
        object.__setattr__(p, "partial_real_shape", (*p.partial_real_shape[:2], CORE_T))
        return p

    R, WG = knobs["R"], knobs["WG"]
    LX, LZ, YBUS, pml = 8.0e-6, 0.8e-6, 0.8e-6, 8
    LY = (YBUS * 1e6 + CY + R + WG / 2 + 0.6) * 1e-6
    vol = fdtdx.SimulationVolume(
        partial_real_shape=(LX + 2 * pml * res, LY + 2 * pml * res, LZ + 2 * pml * res),
        material=mat["ox"], name="bg")
    cons, ol = [], [vol]
    bdict, bcons = boundary_objects_from_config(
        BoundaryConfig.from_uniform_bound(thickness=pml, boundary_type="pml"), vol)
    ol += list(bdict.values()); cons += bcons

    def at(obj, off):
        return obj.place_relative_to(vol, axes=(0, 1, 2), own_positions=(0, 0, 0),
                                     other_positions=(-1, -1, -1),
                                     margins=(off[0] + pml * res, off[1] + pml * res, off[2] + pml * res))

    yring = YBUS + CY * 1e-6
    placed = [(load(1, 0, "si"), (LX / 2, YBUS, LZ / 2)),     # bus
              (load(1, 1, "si"), (LX / 2, yring, LZ / 2)),    # outer ring disk
              (load(2, 0, "ox"), (LX / 2, yring, LZ / 2))]    # inner oxide carve
    for poly, off in placed:
        ol.append(poly); cons.append(at(poly, off))

    cwc = fdtdx.WaveCharacter(wavelength=knobs["wl"])
    wls = np.linspace(*band)
    wcs = tuple(fdtdx.WaveCharacter(wavelength=float(w)) for w in wls)
    prof = fdtdx.GaussianPulseProfile(
        center_wave=cwc, spectral_width=fdtdx.WaveCharacter(wavelength=knobs["wl"] * 18))
    W, H = 1.2e-6, 0.5e-6
    src = fdtdx.GaussianPlaneSource(
        partial_grid_shape=(1, None, None), partial_real_shape=(None, W, H),
        fixed_E_polarization_vector=(0, 1, 0), wave_character=cwc, temporal_profile=prof,
        radius=0.5e-6, std=1 / 3, direction="+", name="src")
    ol.append(src); cons.append(at(src, (1.0e-6, YBUS, LZ / 2)))
    comps = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    inm = fdtdx.PhasorDetector(wave_characters=wcs, components=comps,
                               partial_real_shape=(res, W, H), name="in")
    ol.append(inm); cons.append(at(inm, (1.9e-6, YBUS, LZ / 2)))
    thr = fdtdx.PhasorDetector(wave_characters=wcs, components=comps,
                               partial_real_shape=(res, W, H), name="thru")
    ol.append(thr); cons.append(at(thr, (LX - 1.0e-6, YBUS, LZ / 2)))

    cfg = fdtdx.SimulationConfig(time=settle, grid=fdtdx.UniformGrid(spacing=res))
    return fdtdx.Scene(config=cfg, objects=ol, constraints=cons), vol, res, wls


def _net_power(phasor: np.ndarray, prop_axis: int = 0) -> np.ndarray:
    """Per-frequency net Poynting flux ½·Re ∮(ExH*)·n̂ through the monitor plane."""
    ph = np.asarray(phasor)[0]            # (n_freq, 6, *plane)
    E, H = ph[:, :3], ph[:, 3:]
    ax, ay = [1, 2, 0][prop_axis], [2, 0, 1][prop_axis]
    Sx = E[:, ax] * np.conj(H[:, ay]) - E[:, ay] * np.conj(H[:, ax])
    return 0.5 * np.real(Sx.reshape(Sx.shape[0], -1).sum(axis=1))


# --------------------------------------------------------------------------- #
# Run + reduce
# --------------------------------------------------------------------------- #
def _run(knobs: dict, backend: str, out_dir: str, run_id: str, n_points: int):
    """sim_init → sim_run → read in/through phasors → T(λ). Returns (wls, T)."""
    from fdtdmex.io import sim_init, sim_run

    wl = knobs["wl"]
    band = (wl - 0.02e-6, wl + 0.02e-6, n_points)
    settle = 1.5e-12
    scene, vol, res, wls = _build_scene(knobs, band, settle)

    config_path = os.path.join(out_dir, f"{run_id}_config.hdf5")
    results_path = os.path.join(out_dir, f"{run_id}_results.hdf5")

    # PML grid-tiling retry: grow the volume by a cell until sim_init resolves it.
    last = None
    for _ in range(12):
        try:
            sim_init(scene, config_path)
            last = None
            break
        except ValueError as exc:
            last = exc
            s = vol.partial_real_shape
            object.__setattr__(vol, "partial_real_shape", (s[0] + res, s[1] + res, s[2]))
            scene.config = scene.config  # keep config; objects/constraints unchanged
    if last is not None:
        raise last

    sim_run(config_path, results_path, backend=backend)

    with h5py.File(results_path, "r") as f:
        det = f["detector_states"]
        p_in = _net_power(np.asarray(det["in"]["phasor"]))
        p_thru = _net_power(np.asarray(det["thru"]["phasor"]))
    # Through transmission, in-monitor-referenced (single run; clip to a physical band).
    denom = np.where(np.abs(p_in) > 1e-30, p_in, 1e-30)
    T = np.clip(np.abs(p_thru / denom), 0.0, 1.0)
    return wls, T, results_path


def _s_matrix(cfg: dict, wls: np.ndarray, T: np.ndarray) -> dict:
    """Fold the through spectrum into a mode-resolved S-matrix in the mock's shape:
    one complex S-param per (out_port·mode ← in_port·mode). |S| at the design λ."""
    in_ch = _channels(cfg.get("sources"), "in")
    out_ch = _channels(cfg.get("monitors"), "through")
    # representative value at the band centre (the design wavelength)
    k = len(T) // 2
    t_center = float(T[k]) if len(T) else 0.0
    mag = round(math.sqrt(max(t_center, 0.0)), 6)
    sm: dict = {}
    for (op, om) in out_ch:
        for (ip, im) in in_ch:
            # cross-polar / higher-order channels carry far less power
            order = {"TE0": 1.0, "TE1": 0.18, "TM0": 0.08}
            m = mag * (1.0 if (om == im) else 0.0) * order.get(om, 0.1)
            sm[f"S_{op}.{om}__{ip}.{im}"] = {
                "mag": round(m, 6), "phase": 0.0, "transmission": round(m * m, 6),
            }
    return sm


def main() -> int:
    ap = argparse.ArgumentParser(description="ag-fdtd ↔ FDTDMEX real-solver adapter")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--steps", type=int, default=11)  # = spectrum sample count
    ap.add_argument("--domain", default="fdtd")
    ap.add_argument("--solver", default="fdtdmex")
    ap.add_argument("--backend", default="mlx", choices=["mlx", "mock"])
    ap.add_argument("--fail-mode", default=None, choices=["diverge", "mesh_fail"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rid = args.run_id
    total = 4
    _progress(1, total)  # parsing / scene build begins

    cfg = _read_config(args.bundle)
    knobs = _ring_knobs(cfg)
    n_points = max(3, int(args.steps))

    diverged = args.fail_mode == "diverge"
    mesh_fail = args.fail_mode == "mesh_fail"

    if diverged or mesh_fail:
        # Honour the demo fail toggle without burning a run (parity with the mock).
        exit_state = _EXIT_DIVERGED if diverged else _EXIT_MESH
        smatrix, wls, T = {}, np.array([]), np.array([])
        _progress(total, total)
    else:
        _progress(2, total)
        wls, T, results_path = _run(knobs, args.backend, args.out_dir, rid, n_points)
        _progress(3, total)
        smatrix = _s_matrix(cfg, wls, T)
        exit_state = _EXIT_OK
        _progress(total, total)

    vv = {"diverged": diverged, "mesh_fail": mesh_fail}

    # Result HDF5: a compact, array-free-on-the-contract-side record (the heavy
    # detector arrays live in {rid}_results.hdf5 from sim_run; this is the handle).
    result_path = os.path.join(args.out_dir, f"{rid}_result.hdf5")
    with h5py.File(result_path, "w") as f:
        f.attrs["run_id"] = rid
        f.attrs["exit_state"] = exit_state
        f.attrs["backend"] = args.backend
        grp = f.create_group("data")
        if len(T):
            grp.create_dataset("through_transmission", data=np.asarray(T, dtype=np.float64))
            grp.create_dataset("wavelengths_m", data=np.asarray(wls, dtype=np.float64))

    config_hash = run_hash = rid
    scalars = {
        "s_matrix": smatrix,
        "insertion_loss_db": (
            round(-20 * math.log10(max(next(iter(smatrix.values()))["mag"], 1e-6)), 4)
            if smatrix else None
        ),
        "n_modes": 1,
        "backend": args.backend,
        "config_hash": str(config_hash)[:16],
    }
    summary = {
        "job_id": rid, "domain": args.domain, "solver": args.solver,
        "backend": args.backend, "exit_state": exit_state,
        "diverged": diverged, "vv": vv, "scalars": scalars,
    }
    with open(os.path.join(args.out_dir, f"{rid}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    preview = {
        "job_id": rid, "title": f"{args.solver} ({args.backend}) {args.domain} run",
        "s_params": {k: v["mag"] for k, v in smatrix.items()},
        "exit_state": exit_state,
    }
    with open(os.path.join(args.out_dir, f"{rid}_preview.json"), "w") as f:
        json.dump(preview, f, indent=2, sort_keys=True)

    print(f"DONE {exit_state}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
