# Action Plan — Widening the MLX surface (ADE · lossy full-anisotropic · PEC/PMC)

> ✅ **All three landed (Phase 3 complete).** This is the original how-to spec, kept as the
> implementation record. As-built notes and results live in [ACTION_PLAN.md](../ACTION_PLAN.md)
> ("Phase 3"). One deviation from the plan below: ADE was also **folded into the Metal E-kernel**
> (not just the MLX-op cores) so dispersive media ride the bandwidth floor; the non-dispersive kernel
> stays byte-identical. Drude/Lorentz only — there is **no `DebyePole`** in upstream fdtdx.

> **For the agent picking this up in a fresh session.** Read this top-to-bottom, then
> [CLAUDE.md](../CLAUDE.md), [roadmap.md](roadmap.md) ("Widening the MLX surface"), and the skill
> [`.claude/skills/porting-from-fdtdx`](../.claude/skills/porting-from-fdtdx). These three features
> **already exist in upstream fdtdx's JAX engine**; the dispatcher only gates them to JAX because the
> MLX *kernel* isn't ported (or, for lossy-aniso, only the gate is missing). Each is the **same
> JAX→MLX port pattern as M1–M4** — no MEEP, no new physics. Do them in the order below
> (ascending effort); each is independently shippable.

---

## 0. Orientation (what "porting one of these" means here)

The recurring recipe (identical to M1–M4):

1. **Translate the kernel** from `src/fdtdx/fdtd/update.py` (JAX) to the MLX engine
   (`src/fdtdx/mlx/`), staying *functional / out-of-place* (compute a new array, return it).
2. **Precompute time-invariant coefficients on the host** in the bridge
   ([`src/fdtdx/mlx/bridge.py`](../src/fdtdx/mlx/bridge.py)) exactly like the CPML a/b coefficients —
   anything that depends only on material + dt, not on the field, is computed once with numpy and
   carried into `MLXState`.
3. **Carry any new per-cell time-varying state** in `MLXState`
   ([`src/fdtdx/mlx/state.py`](../src/fdtdx/mlx/state.py)) and thread it through the loop
   ([`src/fdtdx/mlx/loop.py`](../src/fdtdx/mlx/loop.py)).
4. **Bridge it back** if downstream fdtdx reads it (`to_array_container`); ADE's polarization is an
   internal recurrence and does **not** need to round-trip.
5. **Un-gate** in [`src/fdtdx/backend/dispatch.py`](../src/fdtdx/backend/dispatch.py)
   (`_unsupported_reason` / `_unsupported_reason_arrays`).
6. **Add a `validation`-marked element-wise parity test** vs forced-JAX-CPU in
   `tests/validation/` (reuse the `_run_both` construction pattern from
   `tests/validation/test_mlx_parity.py`). Marginal failure → raise resolution, **not** loosen
   tolerance.

Backend forcing for the oracle: `with fdtdx.use_backend("jax")` (CPU) vs `use_backend("mlx")`
(Metal), same case, one Mac. The validation bar is `rel < 1e-3` (float32) on E, H, psi, and detector
states.

The three features and the current gates that decline them
([`dispatch.py`](../src/fdtdx/backend/dispatch.py)):

| Feature | Gate today | True remaining work |
|---|---|---|
| **Lossy full-anisotropic + 9-tensor conductivity** | `_unsupported_reason_arrays` returns "lossy full-anisotropic…"/"full-anisotropic … conductivity (9-tensor)…" | **un-gate + validate** — the kernel is already there |
| **PEC / PMC boundaries** | `_unsupported_reason` returns "PEC/PMC boundaries not supported…" when `objects.pec_objects or objects.pmc_objects` | a per-step field-masking pass in the loop + un-gate + validate |
| **Drude–Lorentz dispersion (ADE)** | `_unsupported_reason_arrays` returns "dispersive (ADE) materials…" when `arrays.dispersive_c1 is not None` | host-precompute c1/c2/c3, carry P-history in state, one extra E-update term + un-gate + validate |

**Do them in this order: (1) lossy-aniso → (2) PEC/PMC → (3) ADE.** Ascending effort; (1) is mostly
a test, (3) adds genuine per-step state.

---

## 1. Lossy full-anisotropic + 9-tensor conductivity — *un-gate + validate* (lowest effort)

**Where it lives in fdtdx:** the 9-tensor A/B update in `update.py` (`else` branch of `update_E`),
which already takes `sigma_E`/`sigma_H` and forms `A, B = compute_anisotropic_update_matrices(...)`.

**What's already done on MLX (verify, don't rebuild):** the MLX anisotropic path is *already
general over σ*:
- [`src/fdtdx/mlx/update.py`](../src/fdtdx/mlx/update.py) `_update_aniso(...)` takes `sigma` and
  calls `compute_anisotropic_update_matrices_mlx(inv_t, sigma_t, c, eta_factor)`.
- [`src/fdtdx/mlx/aniso.py`](../src/fdtdx/mlx/aniso.py) `compute_anisotropic_update_matrices_mlx`
  already builds the lossy `A = m1⁻¹(I−factor)`, `B = c·m1⁻¹·inv_material` path (with the analytic
  per-cell `inv3x3`), and `expand_to_3x3_mlx` already accepts a **9-component** σ.

So the kernel handles lossy full-anisotropic **and** 9-tensor (full) conductivity today; only the
dispatcher refuses to route to it.

**Steps:**
1. **Remove the two gates** in `_unsupported_reason_arrays`
   ([`dispatch.py:102`](../src/fdtdx/backend/dispatch.py)):
   - the `(eps9 or mu9) and has_conductivity` → "lossy full-anisotropic" early return;
   - the `sigma.shape[0] == 9` → "full-anisotropic … conductivity (9-tensor)" loop.
   Leave the dispersive gate. (Double-check no *other* still-unsupported combo slips through — e.g.
   lossy-aniso **and** dispersive together should still decline on the dispersive gate.)
2. **Confirm the bridge already carries σ** — it does: `to_mlx_state` sets
   `sigma_E`/`sigma_H` from `arrays.electric/magnetic_conductivity` for any leading dim. No change.
3. **Add parity tests** `tests/validation/test_mlx_lossy_aniso.py`:
   - `test_lossy_full_aniso_matches_jax`: a 9-tensor ε with a **modest** off-diagonal (≤0.5, see
     caveat) **plus** an isotropic `electric_conductivity` (e.g. 0.05) filling the volume; dipole
     source; assert E/H/detector parity `< 1e-3` vs forced-JAX.
   - `test_full_tensor_conductivity_matches_jax`: diagonal ε + a **9-tensor** `electric_conductivity`
     (symmetric, small off-diagonal); assert parity.
   - (optional) magnetic-conductivity analogue.

**Effort:** ~0.5 day, mostly writing/validating tests.

**Caveat (do not paper over):** **Quirk A** — strongly off-diagonal (9-tensor) anisotropy is
numerically unstable in the *explicit* A/B update and **NaNs even at low Courant**, and it does so in
*pure JAX too* (it's upstream behavior, not an MLX bug — see roadmap "Exposed potential issues").
Keep the test off-diagonals modest (≤0.5) so the parity test measures *correctness*, not the
instability. If a NaN appears, it should appear in **both** backends identically — assert that
(`np.isfinite(...)` matches) rather than chasing it. Do **not** loosen tolerances to hide divergence.

---

## 2. PEC / PMC boundaries — *per-step masking pass* (low effort)

**Where it lives in fdtdx:**
[`src/fdtdx/objects/boundaries/pec.py`](../src/fdtdx/objects/boundaries/pec.py) /
[`pmc.py`](../src/fdtdx/objects/boundaries/pmc.py). Each boundary is a **1-cell-thick** face object
with:
- `tangential_components -> (int, int)` — the two field-component indices tangential to the face
  (PEC zeros **E** tangential; PMC zeros **H** tangential);
- `grid_slice -> (sx, sy, sz)` — the slice of the domain the face occupies;
- `apply_post_E_update(E)` (PEC) / `apply_post_H_update(H)` (PMC) — sets those components to 0 in
  that slice, applied **after** the corresponding field update each step.

In the JAX forward loop these `apply_post_*` passes run after `update_E` / `update_H`. There are no
time-invariant coefficients and no extra dynamic state — it is pure masking.

**Steps:**
1. **Freeze the masks on the host** — add `freeze_pec_pmc(objects)` (new
   `src/fdtdx/mlx/boundary_mask.py`, mirroring `freeze_sources`/`freeze_detectors`). Walk
   `objects.pec_objects` and `objects.pmc_objects`; for each, record `(component_indices,
   grid_slice)`. Two cheap representations, pick one:
   - **(a) index list** — carry `list[(comp:int, slice_tuple)]` and apply with MLX slice-assignment;
   - **(b) precomputed boolean keep-mask** — build, once, a `(3, Nx, Ny, Nz)` float32 `pec_keep`
     (and `pmc_keep`) that is `0.0` on every (component, cell) a PEC/PMC face zeros and `1.0`
     elsewhere; then the per-step op is a single multiply `E = E * pec_keep`. **(b) is preferred**
     for MLX — it is branch-free, allocation-light, fuses into the lazy graph, and trivially
     `mx.compile`-able later (no Python-side slicing per step). Build the mask with numpy from each
     face's `tangential_components` + `grid_slice`, then `_to_mx` it.
2. **Carry the masks in `MLXState`** (`pec_keep: mx.array | None`, `pmc_keep: mx.array | None`,
   default `None`).
3. **Apply in the loop** ([`loop.py`](../src/fdtdx/mlx/loop.py)): immediately **after**
   `inject_sources_E` (and before storing `state.E`), `if state.pec_keep is not None: E = E *
   state.pec_keep`; symmetrically `H = H * state.pmc_keep` after `inject_sources_H`. Order matters —
   match fdtdx: PEC masks **after** the E-update *and* source injection (so a source can't leave a
   nonzero tangential E on a PEC wall); same for PMC/H. Verify against fdtdx's forward step ordering.
4. **Bridge:** masks are inputs only; nothing new to bridge out.
5. **Un-gate** the `if objects.pec_objects or objects.pmc_objects:` early return in
   `_unsupported_reason` ([`dispatch.py:70`](../src/fdtdx/backend/dispatch.py)).
6. **Parity test** `tests/validation/test_mlx_pec_pmc.py`:
   - `test_pec_cavity_matches_jax`: box with one or more faces overridden to PEC
     (`BoundaryConfig.from_uniform_bound(..., override_types={"min_x": "pec", ...})`), dipole inside,
     run, assert E/H parity `< 1e-3` vs forced-JAX **and** that tangential E is exactly 0 on the PEC
     slice (`np.abs(E[tangential, slice]).max() == 0`).
   - `test_pmc_face_matches_jax`: analogous with a PMC face, asserting tangential H is 0 there.
   - A mixed PEC+CPML box is a good third case (PEC on `min_z`, CPML elsewhere).

**Effort:** ~0.5–1 day. The only subtlety is **ordering** (mask after source injection) and getting
each face's `grid_slice`→mask mapping right for all 6 faces / partial faces.

---

## 3. Drude–Lorentz dispersion (ADE) — *host coeffs + P-history + one E-term* (low–medium)

**Where it lives in fdtdx:** [`src/fdtdx/dispersion.py`](../src/fdtdx/dispersion.py)
(`compute_pole_coefficients`, the `DrudePole`/`LorentzPole` model — no Debye upstream) +
the ADE block in `update.py` (the `if arrays.dispersive_P_curr is not None:` branch inside
`update_E`, lines ~180–194). It lives **only in the iso/diagonal fast path** (not the 9-tensor
branch) — so the MLX port targets the non-full-tensor branch of `update_E_mlx`.

**The math (already factored for porting):**
- `compute_pole_coefficients(poles, dt)` returns `c1, c2, c3` of shape `(num_poles,)` — depends only
  on the poles (ω₀, γ, coupling) and `dt`, i.e. **fully time-invariant → host-precompute**. In the
  placed `ArrayContainer` these are broadcast to `(num_poles, 1, Nx, Ny, Nz)`.
- Per step, inside the E-update (right after the lossless `E = factor·E + c·curl·inv_eps`, **before**
  the conductivity divisor):
  ```
  P_new     = c1*P_curr + c2*P_prev + c3*E          # (num_poles, 3, Nx, Ny, Nz) via broadcast
  delta_sum = sum over poles of (P_curr - P_new)     # (3, Nx, Ny, Nz)
  E         = E + inv_eps * delta_sum
  P_prev   <- P_curr ;  P_curr <- P_new              # shift history
  ```
- `c3 = 0` in every **non**-dispersive cell, so with `P_curr = P_prev = 0` (the reset state) the term
  is identically zero outside dispersive regions — it is inert elsewhere, exactly like the CPML
  coeffs are inert away from the PML. (Confirmed by the fdtdx comment at `update.py:176`.)
- `reset()` zeros `dispersive_P_curr`/`dispersive_P_prev` (container.py) — the MLX `arrays.reset()`
  at the top of `_run_mlx_forward` already does this, so the bridged-in P arrays start at zero.

**Steps:**
1. **Host-precompute / bridge-in** ([`bridge.py`](../src/fdtdx/mlx/bridge.py)): if
   `arrays.dispersive_c1 is not None`, `_to_mx` the (already-host-resident) `dispersive_c1/c2/c3` and
   the zeroed `dispersive_P_curr`/`dispersive_P_prev` into the state. (The c-coeffs come pre-built on
   the placed container — no need to call `compute_pole_coefficients` yourself; just carry them. Only
   `dispersive_inv_c2` is reverse-pass-only and **not needed** for forward.)
2. **Extend `MLXState`** ([`state.py`](../src/fdtdx/mlx/state.py)) with `disp_c1, disp_c2, disp_c3,
   P_curr, P_prev` (all `mx.array | None`, default `None`).
3. **Add the ADE term** in `update_E_mlx`'s **non-full-tensor branch**
   ([`update.py`](../src/fdtdx/mlx/update.py)): after `E = factor*state.E + c*curl*inv_eps` and
   before the `/(1 + …)` conductivity divisor, do the `P_new`/`delta_sum`/`E += inv_eps*delta_sum`
   block above. **Return the updated `P_curr`/`P_prev`** alongside `(E, psi_E)` (the loop must shift
   them into the state each step). Mind the broadcast: `disp_c*` are `(num_poles, 1, …)`, `P_*` are
   `(num_poles, 3, …)`; right-aligned broadcasting yields `(num_poles, 3, …)` — replicate fdtdx's
   reshape-free form (`update.py:187-190`) so MLX broadcasting matches element-wise.
4. **Thread P-history through the loop** ([`loop.py`](../src/fdtdx/mlx/loop.py)): `update_E_mlx` now
   also returns `(P_curr_new, P_prev_new)`; store them on `state` right after `state.psi_E = psi_E`.
   Include the P arrays in the periodic-`mx.eval` leaf list so the lazy graph stays bounded.
5. **No bridge-out needed** — P is an internal recurrence; fdtdx does not read final P for forward
   results. (If a future detector needs it, add it then.)
6. **Un-gate** the `arrays.dispersive_c1 is not None` early return in `_unsupported_reason_arrays`
   ([`dispatch.py:115`](../src/fdtdx/backend/dispatch.py)). Keep declining **dispersive + 9-tensor**
   if that combo isn't covered (ADE lives in the fast path only) — add a narrower gate if needed.
7. **Parity test** `tests/validation/test_mlx_dispersion.py`:
   - Build a slab with a dispersive material (a single **Drude** pole is the simplest; a **Lorentz**
     pole exercises `c1`/`c2` fully). Plane-wave or dipole source; run; assert E/H **and** a
     `PhasorDetector` parity `< 1e-3` vs forced-JAX-CPU.
   - A physics check beyond parity: a thin Drude (metal) slab's **reflectance** vs the analytic
     Fresnel value at one frequency (reuse the `tests/visualization` two-run normalization pattern),
     to catch a coefficient/sign error that happens to agree between the two ports.

**Effort:** ~1–2 days. The new piece vs M1–M4 is **carrying mutable per-step P-history** through the
loop (M1–M4 only mutated E/H/psi); follow the existing `psi_E`/`psi_H` threading exactly.

**Caveats:** dispersive **plane sources** (`_temporal_H_filter`) are a *separate* still-gated feature
(the source-side filter) — keep that gate. Start with a non-dispersive source illuminating a
dispersive material. Watch float32: the ADE recurrence can be stiff for high-ω Lorentz poles —
if a marginal parity failure appears, raise resolution / shorten dt (Courant), don't loosen `rtol`.

---

## 4. Definition of done (per feature)

- The corresponding gate in `dispatch.py` is removed (or narrowed) and AUTO routing engages MLX for
  the feature on Apple Silicon.
- A `validation`-marked element-wise parity test vs forced-JAX-CPU passes at `rtol 1e-3` (float32),
  plus at least one *physics* sanity check where cheap (Fresnel/cavity).
- `uvx ruff format/check src/fdtdx/mlx src/fdtdx/backend` clean.
- `roadmap.md`'s "Widening the MLX surface" row and CLAUDE.md's "Deferred → falls back to JAX" list
  are updated to move the feature from *gated* to *supported*.
- A one-paragraph note in the PR describing what was ported and the validation evidence.

## 5. Suggested sequencing & rationale

1. **Lossy full-anisotropic / 9-tensor σ** first — kernel already exists, so it's a fast win that
   also stress-tests the anisotropic A/B path under conductivity (and documents Quirk A's blast
   radius) before ADE leans on the same fast path.
2. **PEC/PMC** next — self-contained masking, no coefficient math, unlocks cavity / waveguide demos.
3. **ADE** last of the three — the only one that adds per-step mutable state; do it once the loop /
   state threading is fresh in mind and (ideally) after the `mx.compile` perf pass lands, so the new
   per-step P-update is written compile-friendly from the start (host-side pole gating, P arrays as
   compiled carry).

None of these depend on or block WS-B/C/D; pull whichever a use case demands.
