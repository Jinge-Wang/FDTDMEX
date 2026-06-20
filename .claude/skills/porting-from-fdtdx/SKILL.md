---
name: porting-from-fdtdx
description: Recipe for porting FDTDX (JAX) kernels to FDTDMEX (MLX) — the JAX→MLX API mapping, the array-bridge pattern, and what to deliberately NOT port. Use when translating engine code from ../fdtdx.
user-invocable: false
---

# Porting FDTDX (JAX) → FDTDMEX (MLX)

Reference source: `../fdtdx` (MIT). Port the **forward numerical hot loop** only; reuse or skip the rest.

## What to port (the hot loop, ~1.5–3k lines)

| FDTDX file | Port to | Notes |
|---|---|---|
| `core/physics/curl.py` | `fdtdmex/fdtd/` | finite-diff curl; make spacing-weighted (non-uniform) |
| `fdtd/update.py` (E/H, incl. 9-tensor path) | `fdtdmex/fdtd/` | functional out-of-place; keep the iso/diag fast path + full-anisotropic path |
| `fdtd/misc.py` (`compute_anisotropic_update_matrices`, `avg_anisotropic_*`) | `fdtdmex/fdtd/` | per-cell 3×3 solve + off-diagonal interpolation → spacing-weighted |
| `objects/boundaries/perfectly_matched_layer.py` | `fdtdmex/fdtd/` (pml) | CPML ψ recurrences |
| `objects/sources/*` (inject path) | `fdtdmex/sources/` | only the source types you need |
| `objects/detectors/*` | `fdtdmex/detectors/` | phasor → complex; diffractive → `mx.fft` |
| `dispersion.py` (ADE) | `fdtdmex/materials/` | Lorentz/Drude pole recurrences |
| `fdtd/initialization.py` | `fdtdmex/io/` array bridge | reference for material-array shapes/sizing |

## What to deliberately NOT port

- `jax.custom_vjp` reversible gradient, `eqxi.while_loop` checkpointed loop, `backward.py` — **no
  autodiff on Metal** (inverse design stays on clusters). Use a plain Python time loop.
- `pytreeclass.TreeClass` / `.aset()` / `.at[].set()` immutable-pytree model — use plain MLX arrays
  in lightweight dataclasses/dicts.
- `jax.sharding` / multi-GPU — single-machine, unified memory.
- `jax.pure_callback` to Tidy3D mode solver — write your own (WS-B).

## JAX → MLX API mapping

| JAX | MLX |
|---|---|
| `jnp.array`, `jnp.zeros`, slicing | `mx.array`, `mx.zeros`, slicing |
| `jnp.roll` (Yee shifts) | `mx.roll` |
| `arr.at[idx].set(v)` | `arr[idx] = v` (eager) or rebuild via concat/`mx.where` |
| `jax.jit` | `mx.compile` |
| `jnp.fft.*` | `mx.fft.*` (complex supported) |
| `jnp.linalg.solve` (per-cell 3×3) | small/analytic inverse, or `mx.linalg.solve` if available; **complex `eig` is NOT on GPU → host numpy/scipy** |
| `jax.lax.cond` (source/detector gating) | Python `if` on a host-known flag, or `mx.where` mask |
| `jax.random` | `mx.random` |

## Array bridge (reuse FDTDX's front end)

Run FDTDX's `place_objects` / `apply_params` / `_init_arrays` **on CPU** to get the material/PML arrays and source temporal profiles as plain arrays, then `np.asarray(...) → mx.array(...)`. This reuses geometry/constraints/GDS/PML-profile computation without porting them. Pin a FDTDX commit (it is a moving target, pre-PyTorch-refactor).

## Numerics caveats

- MLX defaults to `float32`; verify stability vs FDTDX float32/float64 on a known case.
- Validate every ported kernel against FDTDX-on-CPU element-wise (see the `physics-validation` skill). Generated FDTD kernels often pass smoke tests but fail physics (sign/index/Courant bugs).
