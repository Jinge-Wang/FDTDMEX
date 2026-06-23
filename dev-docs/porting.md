# Porting fdtdx (JAX) → FDTDMEX (MLX)

Everything about moving a piece of fdtdx's forward engine onto the MLX/Metal backend: the strategy, the JAX→MLX mapping, the array bridge, the step-by-step recipe, and three worked examples. Companion to the `porting-from-fdtdx` skill. Reference sources: the sibling read-only clones `../fdtdx` (MIT) and `../meep` (GPL — consulted, not copied). The [roadmap](roadmap.md) points here for any "widen the MLX surface" work.

## Strategy

Port the **forward numerical hot loop** (~1.5–3k lines of fdtdx's ~25k) and **reuse** fdtdx's host-side front end (geometry, constraints, GDS, PML profiles, source temporal profiles) through a plain-array bridge. Skip everything tied to differentiable inverse design — that stays on JAX/CUDA.

## The race-free advantage (why MLX makes this easy)

fdtdx's updates are **functional / out-of-place**: each step computes a new `E` from the *old* `E`/`H` and returns it; the anisotropic averages read the *old* padded `E`. Nothing mutates in place, so there is **no read-after-write hazard** between neighbouring cells — the framework effectively double-buffers. MLX is the same functional model, so FDTDMEX inherits race-freedom **for free**: no ping-pong buffers, no atomics, no update-ordering constraints (the exact pain points of a hand-written CUDA kernel). The cost is a transient ~2x field memory + bandwidth, reclaimed by MLX's caching allocator / `mx.compile` buffer reuse — negligible on unified memory.

## What to port

| fdtdx | FDTDMEX target | Notes |
|---|---|---|
| `core/physics/curl.py` | `mlx/curl.py` | make spacing-weighted (non-uniform grids) |
| `fdtd/update.py` (E/H, iso/diag + 9-tensor) | `mlx/update.py` | keep both the fast path and the full-anisotropic path |
| `fdtd/misc.py` (`compute_anisotropic_update_matrices`, `avg_anisotropic_*`) | `mlx/aniso.py` | per-cell 3x3 solve; off-diagonal interp → spacing-weighted |
| `objects/boundaries/perfectly_matched_layer.py` | `mlx/pml.py` | CPML ψ recurrences |
| `objects/sources/*` (inject path) | `mlx/source_freeze.py` + `inject.py` | only the needed source types |
| `objects/detectors/*` | `mlx/detector_freeze.py` + `accumulate.py` | phasor→complex; diffractive→`mx.fft` |
| `dispersion.py` | `mlx/state.py` + `update.py` + `kernels.py` | ADE Lorentz/Drude pole recurrences |
| `fdtd/initialization.py` | `mlx/bridge.py` | reference for material-array shapes/sizing (1/3/9) |

## What NOT to port

- `jax.custom_vjp` reversible gradient; `eqxi.while_loop` checkpointing; `fdtd/backward.py` — no on-device autodiff. A plain Python time loop replaces them.
- `pytreeclass.TreeClass`, `.aset()`, `.at[].set()` — use plain MLX arrays in light dataclasses.
- `jax.sharding` / multi-GPU; `jax.pure_callback` to Tidy3D (a native mode solver replaces it).

## JAX → MLX API cheatsheet

`jnp.*`→`mx.*`; `jnp.roll`→`mx.roll`; `jax.jit`→`mx.compile`; `jnp.fft`→`mx.fft`; `jax.random`→ `mx.random`; `arr.at[i].set(v)`→`arr[i] = v` (eager) or rebuild; `lax.cond` gate→Python `if` on a host flag or `mx.where`. **Complex `eig`/`eigh` is not on the MLX GPU** → the mode solver runs on the host (scipy/numpy).

## Array bridge

Run fdtdx `place_objects` / `apply_params` / `_init_arrays` on **CPU** → obtain `inv_permittivities`, `inv_permeabilities`, conductivities, PML `alpha/kappa/sigma`, and source profiles as arrays → `np.asarray(x)` → `mx.array(...)`. The seam is `mlx/bridge.py` (`to_mlx_state` / `to_array_container`). Pin an fdtdx commit (it is pre-PyTorch-refactor and still changing).

## The recurring recipe

Every forward feature follows the same six steps:

1. **Translate the kernel** from `src/fdtdx/fdtd/update.py` (JAX) to the MLX engine ([`src/fdtdx/mlx/`](../src/fdtdx/mlx/)), staying *functional / out-of-place* (compute a new array, return it).
2. **Precompute time-invariant coefficients on the host** in the bridge ([`src/fdtdx/mlx/bridge.py`](../src/fdtdx/mlx/bridge.py)) — anything that depends only on material + `dt`, not on the field, is computed once with numpy and carried into `MLXState` (exactly like the CPML a/b coefficients).
3. **Carry any new per-cell time-varying state** in `MLXState` ([`src/fdtdx/mlx/state.py`](../src/fdtdx/mlx/state.py)) and thread it through the loop ([`src/fdtdx/mlx/loop.py`](../src/fdtdx/mlx/loop.py)).
4. **Bridge it back** only if downstream fdtdx reads it (`to_array_container`); internal recurrences (e.g. the ADE polarization) do not need to round-trip.
5. **Un-gate** in [`src/fdtdx/backend/dispatch.py`](../src/fdtdx/backend/dispatch.py) (`_unsupported_reason` / `_unsupported_reason_arrays`).
6. **Add a `validation`-marked element-wise parity test** vs forced-JAX-CPU in `tests/validation/` (reuse the `_run_both` construction from `test_mlx_parity.py`). The bar is `rel < 1e-3` (float32) on E, H, ψ, and detector states. Marginal failure → **raise resolution, never loosen tolerance**.

The forced-JAX oracle is `with fdtdx.use_backend("jax")` (CPU) vs `use_backend("mlx")` (Metal) — the same case through both backends on one Mac.

## Worked examples

These three features were ported with the recipe above and are now supported; they are kept as concrete templates for the next port (each illustrates a different amount of new state).

### A. Lossy full-anisotropic + 9-tensor conductivity — *un-gate + validate*

The MLX anisotropic A/B path ([`mlx/aniso.py`](../src/fdtdx/mlx/aniso.py) `compute_anisotropic_update_matrices_mlx`) was already general over σ — it builds the lossy `A = m1⁻¹(I−factor)`, `B = c·m1⁻¹·inv_material` path with the analytic per-cell `inv3x3`, and `expand_to_3x3_mlx` already accepts a 9-component σ. Only the dispatcher refused to route to it. So the port was: remove the two gates in `_unsupported_reason_arrays`, confirm the bridge carries σ (it does, for any leading dim), and add parity tests (a 9-tensor ε with a modest off-diagonal plus an isotropic conductivity; a diagonal ε with a 9-tensor conductivity). The lesson worth keeping: **most of the effort was the test, not the kernel** — always check what the existing MLX path already covers before writing new code.

### B. PEC / PMC boundaries — *per-step masking pass*

Each PEC/PMC boundary is a 1-cell face object that zeros the two tangential components (E for PEC, H for PMC) in its `grid_slice`, applied after the field update. There are no time-invariant coefficients and no dynamic state — it is pure masking. The port freezes, on the host, a `(3, Nx, Ny, Nz)` float32 **keep-mask** (`0.0` where a face zeros a component, `1.0` elsewhere; [`mlx/boundary_mask.py`](../src/fdtdx/mlx/boundary_mask.py)), carries it in `MLXState`, and applies it in the loop as one multiply `E = E * pec_keep` — branch-free and `mx.compile`-friendly. **Ordering is the only subtlety:** match fdtdx and mask **after** the field update *and* source injection, so a source can't leave a nonzero tangential E on a PEC wall.

### C. Drude–Lorentz dispersion (ADE) — *host coeffs + P-history + one E-term*

`compute_pole_coefficients(poles, dt)` returns time-invariant `c1, c2, c3` → host-precompute and carry them, plus the zeroed polarization history `P_curr`/`P_prev`, in `MLXState`. Inside the E-update's iso/diagonal branch (dispersion is iso/diagonal only — fdtdx forbids it with off-diagonal tensors), after the lossless update and before the conductivity divisor:

```
P_new     = c1*P_curr + c2*P_prev + c3*E          # (num_poles, 3, Nx, Ny, Nz) via broadcast
delta_sum = sum over poles of (P_curr - P_new)     # (3, Nx, Ny, Nz)
E         = E + inv_eps * delta_sum
P_prev   <- P_curr ;  P_curr <- P_new              # shift history
```

`c3 = 0` in non-dispersive cells, so with `P = 0` the term is inert outside dispersive regions (like the CPML coefficients away from the PML). The new piece vs the other ports is **carrying mutable per-step P-history** through the loop — thread it exactly like `psi_E`/`psi_H`, and include the P arrays in the periodic-`mx.eval` leaf list so the lazy graph stays bounded. This recurrence was also folded into the Metal E-kernel (`kernels.py` `_ade_lines`) so dispersive media ride the bandwidth floor; the non-dispersive kernel stays byte-identical. Beyond parity, check a Drude slab's reflectance against the analytic Fresnel value — a coefficient/sign error can agree between two ports but not with physics.

## Pitfalls

- MLX defaults to `float32`; verify stability vs fdtdx. Use `float64`/`complex128` on the host where the mode solver needs it.
- Validate every kernel element-wise vs fdtdx-on-CPU; generated kernels pass smoke tests but fail physics (sign/index/Courant/PML-coefficient bugs).
- Don't let the lazy graph grow unbounded in the time loop — periodic `mx.eval`.
- Watch the JAX clamps: OOB integer indexing (isotropic `inv_eps[axis]` → component 0), and the H-source update samples the temporal profile at the `+0.5` half step.
- If a NaN appears, it should appear in **both** backends identically (some upstream updates are not unconditionally stable — see the roadmap's robustness notes). Assert `np.isfinite(...)` matches rather than chasing it, and never loosen tolerances to hide divergence.

## Definition of done (per feature)

- The gate in `dispatch.py` is removed (or narrowed) and AUTO routing engages MLX on Apple Silicon.
- A `validation`-marked element-wise parity test vs forced-JAX-CPU passes at `rel 1e-3` (float32), plus at least one physics sanity check where cheap (Fresnel/cavity).
- `uvx ruff format/check src/fdtdx/mlx src/fdtdx/backend` clean.
- The roadmap's surface table and CLAUDE.md's "Deferred → falls back to JAX" list are updated to move the feature from *gated* to *supported*.
