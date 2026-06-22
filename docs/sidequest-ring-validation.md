# Side quest — ring-resonator validation showcase

**Start a fresh session here.** This is a self-contained task: produce a *physically convincing* ring
resonator characterization and fold it into the demo notebook + README. It is a **documentation +
validation** task — we are **not** rewriting or improving engine code. When done, return to the main plan
in [`ACTION_PLAN.md`](../ACTION_PLAN.md).

## Why

The current demo ([`examples/ring_resonator_demo.ipynb`](../examples/ring_resonator_demo.ipynb)) runs at a
coarse **90 nm** grid where the 0.1 µm coupling gap is ~1 cell, so the ring is effectively fused to the bus
and the resonances are noisy and unverifiable. We want a higher-resolution study that demonstrates real
ring physics and verifies it.

## Deliverables (five figures → `docs/images/`, linked from notebook + README)

1. **Through-port transmission `T(λ)`** — a broadband spectrum showing a clear resonance **dip**; annotate
   the fitted loaded **Q** and **extinction ratio (ER)**.
2. **`|E|²` field maps** at resonance vs off-resonance — an **x–y plane** monitor at the slab mid-plane,
   showing light **trapped in the ring** on resonance and **passing through** off resonance. Excite with a
   **Gaussian pulse**.
3. **Grid convergence** — **loaded Q vs grid resolution** (fit the dip at each resolution). As the grid
   refines, numerical loss drops and Q trends toward the physical value.
4. **Spectra vs bus–ring gap** — `T(λ)` overlaid for several gaps (coupling control).
5. **ER vs gap** — extinction ratio (dB) vs gap, showing the under-/critical-/over-coupling trend.

## Validated methodology (use this — already worked out; don't rediscover)

- **One broadband run, not a wavelength sweep.** A single `GaussianPulseProfile` `ModePlaneSource` + a
  **multi-frequency** `ModeOverlapDetector` (a DFT monitor) yields the whole `T(λ)` spectrum from one
  simulation. Pass `wave_characters=tuple(WaveCharacter(wavelength=w) for w in band)`.
- **Normalization: input monitor, single run.** Normalize the through-port overlap by a second
  `ModeOverlapDetector` placed **~3 cells past the source** (`same_position(src, grid_margins=(3,0,0))`):
  `T(f) = |α_through(f) / α_in(f)|²`. The two-run reference method (bus-only) was **worse** at coarse
  resolution (it amplifies noise → `T>1`). Restrict analysis to the **central band** where the pulse has
  power; band edges give `T>1` artifacts.
- **Settle the ring.** Use `time ≈ 3 ps` so resonances build up and ring down (high-Q needs longer).
- **Resolution / gap.** **35 nm with a 0.20 µm gap (~6 cells)** gives a clean spectrum: dip **T≈0.20 at
  1550 nm**, baseline ~0.9, **ER ≈ 12.7 dB**. Keep **gap ≥ 0.16 µm** and **res ≤ 40 nm**. The headline
  demo can stay coarse/fast; add this as a clearly-labeled *higher-resolution validation* section.
- **Runtime.** JAX runs the forward loop on the **Metal GPU** here, so the time loop is seconds; wall time
  is dominated by **JIT compile** (~5–90 s per distinct config, variable). A 35 nm run (~950k cells,
  ~45k steps) is ~4 s warm. Budget **~30–90 s/run**; the full five-figure suite is **~15–20 min**.
- Mode sources/detectors route the forward run to **JAX** (not MLX) — expected, and fine here.

## Working build skeleton (distilled from the validated probe)

```python
import numpy as np, tempfile, os, gdstk, fdtdx
from fdtdx.objects.static_material.polygon import extruded_polygon_from_gds_path
from fdtdx.objects.boundaries.initialization import BoundaryConfig, boundary_objects_from_config

def run_ring(RES, GAP, band, max_time, with_ring=True):
    """Broadband ring sim → (wls, T(λ)). band = (lo, hi, n) in metres."""
    R, WG, SLAB_T = 1.2, 0.40, 0.22e-6          # ring radius, wg width, slab thickness (µm-ish)
    CY = WG + GAP + R                            # bus→ring-center spacing (real GAP between edges)
    MAT = {"si": fdtdx.Material(permittivity=12.25), "air": fdtdx.Material(permittivity=1.0)}
    lib = gdstk.Library(unit=1e-6, precision=1e-9); cell = lib.new_cell("R")
    shapes = [gdstk.rectangle((-3.2, -WG / 2), (3.2, WG / 2), layer=1)]
    if with_ring:
        shapes += [gdstk.ellipse((0, CY), R + WG / 2, layer=1, tolerance=2e-3),
                   gdstk.ellipse((0, CY), R - WG / 2, layer=2, tolerance=2e-3)]
    cell.add(*shapes); gp = os.path.join(tempfile.gettempdir(), "ring.gds"); lib.write_gds(gp)

    def load(layer, idx, mat):
        p = extruded_polygon_from_gds_path(gp, "R", layer=layer, polygon_index=idx, axis=2,
                                           material_name=mat, materials=MAT)
        object.__setattr__(p, "partial_real_shape", (p.partial_real_shape[0], p.partial_real_shape[1], SLAB_T))
        return p

    LX, LZ = 6.6e-6, 0.66e-6; LY = (CY + R + WG + 0.9) * 1e-6; YBUS = 0.6e-6; pml = 8
    vol = fdtdx.SimulationVolume(partial_real_shape=(LX + 2 * pml * RES, LY + 2 * pml * RES, LZ + 2 * pml * RES),
                                 material=MAT["air"], name="bg")
    cons, ol = [], [vol]
    bd, bc = boundary_objects_from_config(BoundaryConfig.from_uniform_bound(thickness=pml, boundary_type="pml"), vol)
    ol += list(bd.values()); cons += bc

    def ctr(o, off):
        return o.place_relative_to(vol, axes=(0, 1, 2), own_positions=(0, 0, 0), other_positions=(-1, -1, -1),
                                   margins=(off[0] + pml * RES, off[1] + pml * RES, off[2] + pml * RES))

    objs = [(load(1, 0, "si"), (LX / 2, YBUS, LZ / 2))]
    if with_ring:
        objs = [(load(1, 1, "si"), (LX / 2, YBUS + CY * 1e-6, LZ / 2)),   # outer Si disk
                (load(2, 0, "air"), (LX / 2, YBUS + CY * 1e-6, LZ / 2)),  # inner air disk carves the hole
                (load(1, 0, "si"), (LX / 2, YBUS, LZ / 2))]               # bus
    for poly, off in objs:
        ol.append(poly); cons.append(ctr(poly, off))

    wls = np.linspace(*band); wcs = tuple(fdtdx.WaveCharacter(wavelength=float(w)) for w in wls)
    cw = fdtdx.WaveCharacter(wavelength=1.56e-6)
    prof = fdtdx.GaussianPulseProfile(center_wave=cw, spectral_width=fdtdx.WaveCharacter(wavelength=1.56e-6 * 18))
    W, Hh = 1.4e-6, 0.5e-6
    src = fdtdx.ModePlaneSource(mode_index=0, filter_pol="te", direction="+", temporal_profile=prof,
                                wave_character=cw, partial_real_shape=(RES, W, Hh), name="in")
    ol.append(src); cons.append(ctr(src, (0.9e-6, YBUS, LZ / 2)))
    inm = fdtdx.ModeOverlapDetector(mode_index=0, filter_pol="te", direction="+", wave_characters=wcs,
                                    partial_real_shape=(RES, W, Hh), name="in_norm")
    ol.append(inm); cons.append(inm.same_position(src, grid_margins=(3, 0, 0)))
    thr = fdtdx.ModeOverlapDetector(mode_index=0, filter_pol="te", direction="+", wave_characters=wcs,
                                    partial_real_shape=(RES, W, Hh), name="thru")
    ol.append(thr); cons.append(ctr(thr, (LX - 0.9e-6, YBUS, LZ / 2)))
    # For field maps: add a PhasorDetector with partial_grid_shape=(None,None,1) at z=slab-mid,
    # wave_characters=[dip, off_res], components=("Ex","Ey","Ez"); |E|² = Σ|phasor_E|².

    cfg = fdtdx.SimulationConfig(time=max_time, grid=fdtdx.UniformGrid(spacing=RES))
    o, a, _, cfg, _ = fdtdx.place_objects(object_list=ol, config=cfg, constraints=cons)
    a = fdtdx.extend_material_to_pml(objects=o, arrays=a)
    a, o, _ = fdtdx.apply_params(a, o, {})
    _, r = fdtdx.run_fdtd(arrays=a, objects=o, config=cfg, show_progress=False)
    s = r.detector_states
    T = np.abs(np.asarray(o["thru"].compute_overlap(s["thru"])) / np.asarray(o["in_norm"].compute_overlap(s["in_norm"]))) ** 2
    return wls, T

# Example: wls, T = run_ring(35e-9, 0.20e-6, (1.53e-6, 1.59e-6, 60), 3000e-15)
```

## Fitting Q / ER from a spectrum

Find the deepest dip `λ0 = argmin T`, a local **baseline** (off-resonance max near the dip), and `T_min`.
Half-depth level `= (baseline + T_min)/2`; **FWHM** = separation of the two wavelengths where `T` crosses
that level around `λ0`; **Q = λ0 / FWHM**; **ER = 10·log10(baseline / T_min)** dB.

## Reuse

- [`src/fdtdx/utils/mode_expansion.py`](../src/fdtdx/utils/mode_expansion.py) — `compute_mode_expansion`
  (mode-expansion monitor; the spectrum here uses `ModeOverlapDetector.compute_overlap` directly across
  many frequencies, which is the same overlap integral).
- [`examples/make_showcase_images.py`](../examples/make_showcase_images.py) — current figure generator;
  add the new figures here (or a sibling `examples/ring_characterization.py`) writing to `docs/images/`.
- The demo notebook is generated from the `.py` (percent-format) via a small nbformat script, then
  executed with `jupyter nbconvert --to notebook --execute --inplace`. Kernel: `FDTDMEX (.venv)`.

## Acceptance

- `T(λ)` shows ≥1 clean resonance dip with baseline near the off-resonance transmission; Q and ER quoted.
- Field maps clearly show energy **inside the ring** on resonance and not off resonance.
- Q-vs-resolution shows a sensible convergence trend.
- ER-vs-gap shows the coupling trend.
- Notebook re-executed (higher-resolution validation section added), README showcase updated with the new
  figures, `uvx ruff check` clean, existing `tests/validation` still pass.

## When done

Update [`ACTION_PLAN.md`](../ACTION_PLAN.md) if anything changed, then resume the main task there: the
orchestration layer (MCP server, mode-sources-on-Metal, web UI).
```bash
uv run --extra viz python examples/ring_characterization.py   # generate figures
uv run --with pytest pytest tests/validation -q               # confirm nothing regressed
uvx ruff format src/fdtdx src/fdtdmex && uvx ruff check src/fdtdx src/fdtdmex
```
