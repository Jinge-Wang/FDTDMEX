# Roadmap

Where FDTDMEX is and where it's going, grouped by capability area rather than by development phase. Code generation is cheap; **physics validation and Metal performance are the long pole**, so estimates are part-time and rough. The single new-contributor entry point is [ACTION_PLAN.md](ACTION_PLAN.md); porting a feature off fdtdx's JAX engine is covered in detail in [porting.md](porting.md).

## What's done (vs fdtdx)

FDTDMEX keeps fdtdx's entire front end and adds a native forward backend. Relative to upstream fdtdx, the forward path gains:

- **Metal/MLX forward engine at the memory-bandwidth floor.** Custom Metal E/H kernels run the time loop and are default-on (`FDTDMEX_METAL_KERNEL=0` forces the MLX-op cores). ~6.5–7x over JAX-CPU for iso/diagonal at N ≥ 128, no plateau with resolution. Depth: [performance.md](../docs/performance.md).
- **Materials & physics on Metal.** Isotropic / diagonal / full-tensor (9-component) anisotropy, lossy + 9-tensor conductivity, and Drude–Lorentz (ADE) dispersion (folded into the E-kernel). Depth: [materials-anisotropy.md](../docs/materials-anisotropy.md).
- **Boundaries.** CPML, periodic (real / Bloch-k0), and PEC/PMC — all on the Metal path.
- **2nd-order non-uniform (graded) grids.** Spacing-weighted curl, interpolation, and off-diagonal averaging — *more correct* than upstream's unweighted (1st-order) average. Depth: [nonuniform-grid.md](../docs/nonuniform-grid.md).
- **Native, Tidy3D-free mode solver + mode-expansion monitor**, Kottke **subpixel smoothing**, a `Scene` front end with interactive 3-D viz, and a **portable HDF5 hand-off** (`sim_init` → `sim_run` → `sim_postproc`). Depth: [mode-solver.md](../docs/mode-solver.md), [subpixel-smoothing.md](../docs/subpixel-smoothing.md), [mcp-and-ui.md](../docs/mcp-and-ui.md).

All of the above is element-wise parity-validated against the forced-JAX-CPU oracle (`tests/validation/`); fdtdx's own physics tests also pass auto-routed to MLX.

## Coming next

Grouped by what the feature is and why we want it. None of these block each other unless noted.

### Orchestration / agentic workspace (the current focus)
**Why:** make the solver LLM-drivable — an agent describes a setup, runs it, and reads back small results — without ever handling large arrays. The declarative schema and the HDF5 `sim_*` contract are done; what remains is the **MCP server** (introspect the API → build/validate a `SceneModel` → run → fetch `sim_postproc`) and, as the long tail, a locally-hostable **web 3-D editor**. See [mcp-and-ui.md](../docs/mcp-and-ui.md).

### Deeper engine performance
**Why:** the bulk update already runs at the Metal memory-bandwidth floor, so the next gains must beat the memory wall itself, not the per-byte rate. This is the second major track alongside the agentic workflow, and it targets a **much larger speedup over the current engine** (fp32 throughout, every step parity-validated). Levers, largest first: **interior temporal blocking** (advance several time steps per DRAM pass — the only sub-floor lever, the deep-algorithm work); **per-subdivision material compaction** (a homogeneous tile carries a single scalar/tensor instead of a per-cell array, with the saving scaling by the material's component count — biggest on the full-tensor anisotropic path); **monitor-traffic reduction** (defer the *linear* interpolation to the end, and auto-subsample the phasor DFT to a small factor above its Nyquist rate); **threadgroup-memory tiling + SIMD-group** neighbour-sharing (the strided x/y neighbours) and reductions (for volume-integrating monitors and mode overlap). The full plan — model, profiling, ranked strategies, the layout decision, and staged targets — is in [performance-roadmap.md](performance-roadmap.md); see also [porting.md](porting.md) (JAX→MLX kernel recipe) and [performance.md](../docs/performance.md) (roofline and current numbers).

### Mode sources / detectors on Metal
**Why:** today a mode source + mode-overlap detector forces the forward loop onto JAX/CPU, so the full photonic-IC workflow (and the ring showcase) can't run on Metal end-to-end. The native mode *solver* they call is already done; porting the mode-source injection and overlap detector into the MLX freeze seam (`mlx/source_freeze.py` / `mlx/detector_freeze.py`) closes the gap. Same [porting](porting.md) recipe as the engine features.

### Bloch / complex (nonzero-k) propagation
**Why:** band-structure and obliquely-periodic problems need complex fields with a Bloch phase. This is the one remaining same-port engine feature — promote the MLX forward path to complex64 end-to-end and validate against the JAX complex oracle. Demand-driven. (Gradients stay out of scope — forward-only.)

### Widening the MLX surface (demand-driven ports)
Features that **already exist in fdtdx's JAX engine** and just need the JAX→MLX port (no new physics). The big three — lossy full-anisotropic + 9-tensor conductivity, PEC/PMC, and Drude–Lorentz ADE — are **done** and serve as the worked examples in [porting.md](porting.md). Slot any remaining one in whenever a use case demands it; each is independently shippable and follows the same recipe.

### New physics (needs a MEEP reference or a fresh derivation)
Not in fdtdx, so these come with their own validation reference rather than a parity oracle:

- **Subpixel-smoothing auto-integration.** The Kottke smoother exists as a standalone utility; wiring it into `place_objects` (host-side supersampling, opt-in, default off to preserve parity) would make it automatic. See [subpixel-smoothing.md](../docs/subpixel-smoothing.md).
- **Near-to-far-field.** Port `../meep/src/near2far.cpp`; self-contained and optional.
- **χ² nonlinearity (Pockels / SHG; LiNbO₃).** A local nonlinear-polarization term in the E-update, `P_NL,i = ε₀ Σ_jk χ²_ijk E_j E_k` — straightforward in a forward time-domain solver (no autodiff entanglement). The χ² tensor is itself anisotropic. ~3–5 d impl + ~1 wk SHG validation. See the χ² note in [materials-anisotropy.md](../docs/materials-anisotropy.md).
- **Anisotropic + dispersive media (low priority).** Upstream fdtdx forbids the combination, so there is **no parity oracle** — it needs a MEEP-style derivation. MEEP's reference: each Lorentz/Drude pole carries a symmetric 3x3 susceptibility tensor `σ` (`sigma_diag` + `sigma_offdiag`), and the ADE polarization update for component `c` sums tensor-weighted contributions of all three field components, the off-diagonal ones Yee-averaged to `c`'s location (the same averaging our 9-tensor update already does). Code: [`../meep/src/susceptibility.cpp`](../../meep/src/susceptibility.cpp), `lorentzian_susceptibility::update_P`; API: `LorentzianSusceptibility` (`sigma_diag`/`sigma_offdiag`) in [`../meep/python/geom.py`](../../meep/python/geom.py). The motivating case is **lithium niobate** (anisotropic + dispersive + χ²) — a clean implementation would be a real FDTDMEX differentiator. Approach: extend the iso/diagonal ADE recurrence so each pole's `c3` becomes a 3x3 tensor with Yee-averaged off-diagonal E reads, validated vs MEEP.

### Mode-solver extensions
Off-diagonal (fully tensorial) cross-sections — the 4Nx4N complex eigenproblem — and bends / leaky modes route to Tidy3D today when it's installed; native support is a later expansion. See [mode-solver.md](../docs/mode-solver.md).

## Known robustness issues (worth a dedicated stability study)

Both **reproduce in pure JAX** (`fdtdx.use_backend("jax")`), so they are upstream fdtdx behaviour, not MLX-port bugs — but they bound what the engine can be trusted with.

- **Strongly off-diagonal (9-tensor) anisotropy is unstable.** A full-tensor ε with a large off-diagonal element (e.g. optic axis at 45°) diverges to NaN even at Courant 0.3; a small off-diagonal (≲0.5) is stable. Likely the explicit per-cell A/B update is not unconditionally stable for strong coupling — wants a von-Neumann analysis and possibly a symmetrized average or sub-Courant factor.
- **Finite-aperture `GaussianPlaneSource` is unstable.** A Gaussian with a *partial* transverse aperture NaNs, while a full-aperture one (radius-controlled) is stable. Suspect the TFSF profile / energy normalization for partial apertures and thin transverse dimensions.

Reproductions: vary the crystal off-diagonal / source aperture in `tests/visualization/test_birefringence_visual.py` and force JAX.

## Strategic note

Upstream fdtdx is being rewritten in PyTorch ("The Big Refactor," disc. #349) with no timeline. PyTorch's MPS backend has no FFT and weak complex support, so it would not give good native Metal anyway. This forward MLX engine is independent of that timeline and reusable as the seed of a future MLX backend.
