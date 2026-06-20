# Porting Notes: FDTDX (JAX) → FDTDMEX (MLX)

Companion to the `porting-from-fdtdx` skill. Reference source: `../fdtdx` (MIT).

## Strategy

Port the **forward numerical hot loop** (~1.5–3k lines of FDTDX's ~25k) and **reuse** FDTDX's host-side front end (geometry, constraints, GDS, PML profiles, source temporal profiles) via a plain-array bridge. Skip everything tied to differentiable inverse design.

## The race-free advantage (why MLX makes this easy)

FDTDX updates are **functional / out-of-place**: each step computes a new `E` from the *old* `E`/`H` and returns it; the anisotropic averages read the *old* padded `E`. Nothing mutates in place, so there is **no read-after-write hazard** between neighbouring cells — the framework effectively double-buffers. MLX is the same functional model, so FDTDMEX inherits race-freedom **for free**: no ping-pong buffers, no atomics, no update-ordering constraints (the exact pain points of a hand-CUDA kernel). Cost is transient ~2× field memory + bandwidth, reclaimed by `mx.compile` fusion / buffer reuse — negligible on unified memory.

## What to port

| FDTDX | FDTDMEX target | Notes |
|---|---|---|
| `core/physics/curl.py` | `fdtd/` | make spacing-weighted (non-uniform) |
| `fdtd/update.py` (E/H, iso/diag + 9-tensor) | `fdtd/` | keep both fast path and full-anisotropic path |
| `fdtd/misc.py` (`compute_anisotropic_update_matrices`, `avg_anisotropic_*`) | `fdtd/` | per-cell 3×3 solve; off-diagonal interp → spacing-weighted |
| `objects/boundaries/perfectly_matched_layer.py` | `fdtd/` (pml) | CPML ψ recurrences |
| `objects/sources/*` (inject path) | `sources/` | only needed source types |
| `objects/detectors/*` | `detectors/` | phasor→complex; diffractive→`mx.fft` |
| `dispersion.py` | `materials/` | ADE Lorentz/Drude pole recurrences |
| `fdtd/initialization.py` | `io/` bridge | reference for material-array shapes/sizing (1/3/9) |

## What NOT to port

- `jax.custom_vjp` reversible gradient; `eqxi.while_loop` checkpointing; `fdtd/backward.py` — no on-device autodiff. Plain Python time loop instead.
- `pytreeclass.TreeClass`, `.aset()`, `.at[].set()` — use plain MLX arrays in light dataclasses.
- `jax.sharding` / multi-GPU; `jax.pure_callback` to Tidy3D (write your own mode solver).

## JAX → MLX API cheatsheet

`jnp.*`→`mx.*`; `jnp.roll`→`mx.roll`; `jax.jit`→`mx.compile`; `jnp.fft`→`mx.fft`; `jax.random`→ `mx.random`; `arr.at[i].set(v)`→`arr[i] = v` (eager) or rebuild; `lax.cond` gate→Python `if` on a host flag or `mx.where`. **Complex `eig`/`eigh` is not on the MLX GPU** → mode solver runs on host (scipy/numpy).

## Array bridge

Run FDTDX `place_objects`/`apply_params`/`_init_arrays` on **CPU** → obtain `inv_permittivities`, `inv_permeabilities`, conductivities, PML `alpha/kappa/sigma`, source profiles as arrays → `np.asarray(x) → mx.array(...)`. Pin a FDTDX commit (it's pre-PyTorch-refactor and changing).

## Pitfalls

- MLX defaults to `float32`; verify stability vs FDTDX. Use `float64`/`complex128` on host where the mode solver needs it.
- Validate every kernel element-wise vs FDTDX-on-CPU; generated kernels pass smoke tests but fail physics (sign/index/Courant/PML-coefficient bugs).
- Don't let the lazy graph grow unbounded in the time loop — periodic `mx.eval`.
