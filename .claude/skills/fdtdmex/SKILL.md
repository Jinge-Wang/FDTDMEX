---
name: fdtdmex
description: FDTDMEX framework knowledge — MLX/Metal FDTD conventions, Yee grid layout, functional race-free updates, non-uniform-grid spacing, and the forward time-loop pattern. Use when writing or modifying fdtdmex engine code.
user-invocable: false
---

# FDTDMEX Framework Knowledge

Forward-first FDTD on **MLX** (Apple Silicon / Metal). Inverse design is out of scope (stays on CUDA/JAX). Authoritative physics: `docs/physics.md`; porting recipe: see the `porting-from-fdtdx` skill.

## Array & dtype conventions

- Field arrays are `(3, Nx, Ny, Nz)` — index 0 is the vector component (x,y,z).
- Material arrays store **inverse** permittivity/permeability to avoid division in the hot loop: `(1, …)` isotropic, `(3, …)` diagonal-anisotropic, `(9, …)` full-tensor (row-major xx,xy,xz,…). Sizing is **global**: if any object is full-anisotropic, all material arrays become 9-component.
- Real fields are `float32`; complex (Bloch/phasor) use `complex64`. MLX supports complex arrays and complex FFT (`mx.fft`).

## Functional / out-of-place = race-free

MLX ops return new arrays. Each step computes a **new** `E` from the **old** `E`/`H` and returns it — never mutate in place. This guarantees no read-after-write hazard between neighbouring cells (the framework effectively double-buffers). Do not hand-roll ping-pong buffers or atomics.

```python
E_new = factor * E_old + c * curl_H * inv_eps   # new buffer; old E still intact for neighbours
```

## Yee grid + leapfrog

Staggered E (integer steps) and H (half steps). Single step order: update E (curl H) → update H (curl E) → inject sources → record detectors. eta0-normalized H (impedance folded into the update coefficients). See `docs/physics.md` for the exact stencils and coefficients.

## Non-uniform grids are first-class

Every curl / interpolation / update **must** take per-axis Yee cell-size arrays (primal `Δ` and dual `Δ̃` spacings) and use **spacing-weighted** finite differences and interpolation. The naive unweighted 4-point average (as in FDTDX) is only 1st-order on graded meshes. See `docs/nonuniform-grid.md`. Carry spacing through the engine API from the start.

## Time loop

```python
@mx.compile
def step(state):           # one E/H/source/detector update
    ...
    return state

for t in range(num_steps):
    state = step(state)
    if t % EVAL_EVERY == 0:
        mx.eval(state)     # bound the lazy graph; avoids unbounded memory growth
```

## Material tensor consumption

Full-anisotropic updates do a per-cell 3×3 solve `E_new = A·E_old + B·curl(H)` where off-diagonal terms couple components living at different Yee locations — so they need **spacing-weighted interpolation** of the other components to the target location (see `fdtd/misc.py` in `../fdtdx` for the structure; improve the averaging to be spacing-weighted). Subpixel smoothing (WS-C) emits exactly these per-cell tensors, including for nominally isotropic geometry at tilted interfaces.

## Common pitfalls

- Forgetting to thread cell-size arrays → silently 1st-order on non-uniform grids.
- Relying on in-place writes → breaks the functional/race-free guarantee on MLX.
- Letting the lazy graph grow unbounded → periodic `mx.eval`.
- Assuming MLX has `eig`/`eigh` for complex on GPU — it does not; the mode solver runs on host (numpy/scipy). See the `porting-from-fdtdx` skill for the full "do not port" list.
