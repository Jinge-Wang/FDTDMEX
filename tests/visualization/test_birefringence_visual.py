"""Visualization + physics test: birefringence (double refraction) on the MLX backend.

An angled Gaussian beam, polarized at 45 deg, passes through a uniaxial-anisotropic slab
(optic axis along y) embedded in vacuum with PML on the outside. The two polarizations see
different indices -- n_o along x, n_e along y -- so inside the slab Ex and Ey acquire
different wavevectors (different wavelengths) and refract differently. That is the signature
of birefringence.

Runs on the MLX (Metal) backend, asserts the measured ordinary/extraordinary wavevectors in
the slab match n_o / n_e, and saves a figure (Re(Ex), Re(Ey), |E|) to the output dir
(``$FDTDMEX_VIZ_DIR`` or ``<repo>/outputs``).
"""

import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.backend.platform import is_apple_silicon, mlx_available

pytestmark = [
    pytest.mark.validation,
    pytest.mark.skipif(
        not (is_apple_silicon() and mlx_available()),
        reason="MLX (Metal) backend requires Apple Silicon + mlx",
    ),
]

_RES = 50e-9
_PML = 8
_N = 72
_WL = 1.0e-6
_N_O = 1.5  # ordinary index (x-polarization)
_N_E = 2.2  # extraordinary index (y-polarization = optic axis)
_Z0 = 18  # slab start (z cell)
_ZT = 40  # slab thickness (z cells)


def _build():
    crystal = fdtdx.Material(permittivity=(_N_O**2, _N_E**2, 1.0))  # uniaxial, optic axis along y
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=_RES), time=60e-15, dtype=jnp.float32, courant_factor=0.85
    )
    objects, constraints = [], []
    vol = fdtdx.SimulationVolume(partial_real_shape=(_N * _RES,) * 3)
    objects.append(vol)
    bcfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML, override_types={"min_y": "periodic", "max_y": "periodic"}
    )
    bdict, clist = fdtdx.boundary_objects_from_config(bcfg, vol)
    constraints.extend(clist)
    objects.extend(bdict.values())

    # Anisotropic slab in the middle (full transverse, finite z), vacuum before/after, PML outside.
    blk = fdtdx.UniformMaterialObject(partial_grid_shape=(None, None, _ZT), material=crystal)
    constraints.extend(
        [
            blk.same_size(vol, axes=(0, 1)),
            blk.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_Z0,)),
            blk.place_at_center(vol, axes=(0, 1)),
        ]
    )
    objects.append(blk)

    # Angled (18 deg azimuth) Gaussian beam, polarized 45 deg so both o and e are excited.
    src = fdtdx.GaussianPlaneSource(
        partial_grid_shape=(None, None, 1),
        fixed_E_polarization_vector=(1, 1, 0),
        wave_character=fdtdx.WaveCharacter(wavelength=_WL),
        direction="+",
        radius=1.4e-6,
        std=1 / 3,
        azimuth_angle=18.0,
    )
    constraints.extend(
        [
            src.same_size(vol, axes=(0, 1)),
            src.place_at_center(vol, axes=(0, 1)),
            src.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_PML + 3,)),
        ]
    )
    objects.append(src)

    # Plane monitor: steady-state phasor of Ex, Ey over the whole volume.
    mon = fdtdx.PhasorDetector(
        name="mon",
        wave_characters=(fdtdx.WaveCharacter(wavelength=_WL),),
        components=("Ex", "Ey"),
        reduce_volume=False,
        plot=False,
    )
    constraints.extend([mon.same_size(vol, axes=(0, 1, 2)), mon.place_at_center(vol, axes=(0, 1, 2))])
    objects.append(mon)
    return objects, constraints, config


def _kz_in_slab(comp_phasor, cy):
    """Measure |k_z| of a component along the beam column inside the slab (rad/cell)."""
    f = comp_phasor[:, cy, :]  # (x, z) complex
    zr = np.arange(_Z0 + 3, _Z0 + _ZT - 3)
    xc = int(np.argmax(np.abs(f[:, _Z0 + 5])))  # x column of peak amplitude at slab entrance
    phase = np.unwrap(np.angle(f[xc, zr]))
    return abs(np.polyfit(zr, phase, 1)[0])


def _output_dir() -> pathlib.Path:
    default = pathlib.Path(__file__).resolve().parents[2] / "outputs"
    d = pathlib.Path(os.environ.get("FDTDMEX_VIZ_DIR", default))
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_birefringence_double_refraction_mlx():
    objects, constraints, config = _build()
    key = jax.random.PRNGKey(0)
    oc, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, oc, _ = fdtdx.apply_params(arrays, oc, params, key)
    with fdtdx.use_backend("mlx"):
        _, arr = fdtdx.run_fdtd(arrays=arrays, objects=oc, config=config, key=key, show_progress=False)

    ph = np.asarray(arr.detector_states["mon"]["phasor"])[0, 0]  # (component, x, y, z)
    cy = _N // 2
    assert np.isfinite(ph).all(), "simulation diverged"

    k_o = _kz_in_slab(ph[0], cy)  # Ex, ordinary
    k_e = _kz_in_slab(ph[1], cy)  # Ey, extraordinary
    # Birefringence: the extraordinary wave has a larger wavevector (shorter wavelength).
    assert k_e / k_o > 1.25, f"no birefringence: k_e/k_o = {k_e / k_o:.3f}"
    # And each matches its analytic z-wavevector 2*pi*n*RES/wl within 15%.
    k_o_analytic = 2 * np.pi * _N_O * _RES / _WL
    k_e_analytic = 2 * np.pi * _N_E * _RES / _WL
    assert abs(k_o - k_o_analytic) / k_o_analytic < 0.15, f"k_o={k_o:.4f} vs {k_o_analytic:.4f}"
    assert abs(k_e - k_e_analytic) / k_e_analytic < 0.15, f"k_e={k_e:.4f} vs {k_e_analytic:.4f}"

    # Save the visualization.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    re_ex = np.real(ph[0][:, cy, :]).T
    re_ey = np.real(ph[1][:, cy, :]).T
    tot = np.sqrt(np.abs(ph[0][:, cy, :]) ** 2 + np.abs(ph[1][:, cy, :]) ** 2).T
    vmax = max(np.abs(re_ex).max(), np.abs(re_ey).max())
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.6))
    panels = [
        (axs[0], re_ex, f"Re(Ex) ordinary  n={_N_O}", "RdBu_r", vmax),
        (axs[1], re_ey, f"Re(Ey) extraordinary  n={_N_E}", "RdBu_r", vmax),
        (axs[2], tot, "|E_transverse| (angled beam)", "inferno", None),
    ]
    for a, data, title, cmap, vm in panels:
        kw = dict(vmin=-vm, vmax=vm) if vm is not None else {}
        im = a.imshow(data, origin="lower", aspect="equal", cmap=cmap, extent=[0, _N, 0, _N], **kw)
        a.axhline(_Z0, color="lime", ls="--", lw=1)
        a.axhline(_Z0 + _ZT, color="lime", ls="--", lw=1)
        a.set_xlabel("x (cells)")
        a.set_ylabel("z (cells, propagation +)")
        a.set_title(title)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(
        f"Birefringence / double refraction through a uniaxial slab (MLX/Metal)\n"
        f"k_e/k_o measured = {k_e / k_o:.2f}  (n_e/n_o = {_N_E / _N_O:.2f})",
        fontsize=12,
    )
    fig.tight_layout()
    out = _output_dir() / "birefringence_mlx.png"
    fig.savefig(out, dpi=115)
    plt.close(fig)
    assert out.exists()
