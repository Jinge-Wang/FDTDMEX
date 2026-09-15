# graded_grid_slab — mesh override regions on a dielectric slab

A plane wave at 1 µm hits an `eps = 4` half space at normal incidence. The same problem is run on
three z discretizations and the measured transmission is compared:

| grid | z cells | what it is |
| --- | --- | --- |
| `coarse` | 80 | 50 nm everywhere (20 cells per vacuum wavelength, 10 inside the slab) |
| `fine` | 320 | 12.5 nm everywhere along z |
| `graded` | 220 | 50 nm background, a 12.5 nm `RefinementRegion` over the slab, graded back to 50 nm on the vacuum side |

Each grid is run twice — once with the slab, once in vacuum — and the transmission is the ratio of
the two time-averaged Poynting fluxes at the same detector. That removes the source amplitude and
the detector's own grid weighting, so the three numbers are comparable with each other and with the
Fresnel value `4n / (1 + n)^2 = 8/9`.

```
python examples/graded_grid_slab/graded_grid_slab.py
```

## Output

```
GradedGrid: background 50 nm, 1 refinement region(s), max ratio 1.4, 3,520 cells total
  x: 4 cells, width 50 nm .. 50 nm, largest neighbour ratio 1.000
  y: 4 cells, width 50 nm .. 50 nm, largest neighbour ratio 1.000
  z: 220 cells, width 12.5 nm .. 49.85 nm, largest neighbour ratio 1.399
  ratio bound honoured

    grid   z cells          T    vs fine
  coarse        80   0.876847   1.11e-02
    fine       320   0.887917
  graded       220   0.888307   3.90e-04
analytic             0.888889
```

The graded grid lands 3.9e-4 from the uniformly fine answer with 220 z-cells instead of 320, while
the coarse background alone is 1.1e-2 off. The transverse axes and the physical PML thickness are
identical in all three runs, so the only variable is the z cell size.

## Two things the numbers taught us

**Refine the material, not just the interface.** The wavelength inside `eps = 4` is half the vacuum
one, so a 50 nm background leaves only 10 cells per wavelength there. An earlier version of this
example refined only `[-300 nm, +700 nm]` — the interface and the detector — and measured
`T = 0.8959`, a 0.8 % bias, because the grading transition then sat inside the high-index medium.
Covering the whole slab removes it. Choosing cell sizes per material automatically is issue #34;
here the region is placed by hand.

**Pin both faces of an object in metres.** The slab's faces are set with a single
`RealCoordinateConstraint` carrying both sides. A single metric *length* would be converted to a
cell count using the cells at the domain's lower corner, which on a graded grid are not the cells
the object sits on.
