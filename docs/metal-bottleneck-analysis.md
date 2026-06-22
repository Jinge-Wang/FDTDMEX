# Metal GPU bottleneck analysis — measured, not inferred

> **Purpose.** The ACTION_PLAN proposed four eager-path fixes on the strength of a *wall-clock*
> decomposition (perf-baseline §1a) that was never verified at the GPU. This document replaces that
> inference with measurement: a real achieved-bandwidth roofline, a round-trip accounting of the
> real engine, and a 2×2 (compile × CPML) decomposition across N. It ends with an evidence-ranked
> fix list and a go/no-go that **reorders** the ACTION_PLAN and **kills** one of its Phase-2 ideas.
>
> Repro harnesses (added this phase, no engine changes):
> [`benchmarks/profile_metal.py`](../benchmarks/profile_metal.py) (roofline + dispatch),
> [`benchmarks/profile_engine.py`](../benchmarks/profile_engine.py) (real-engine round-trips + the
> 2×2 predictive check), [`benchmarks/profile_memory.py`](../benchmarks/profile_memory.py) (value proof).
> Machine: Apple M4 Pro, `applegpu_g16s`, MLX 0.31.2, float32.

## TL;DR

The engine is **memory-traffic-bound on redundant data**, not bandwidth-limited, not dispatch-starved,
not layout/coalescing-limited. At N=192 the per-step time equals **~99 full-array DRAM round-trips**
at the measured **240 GB/s** roofline, when only ~5–8 are physically necessary — **the bus is ~85%
saturated moving redundant intermediates.** The fix is to *move less data*, and **no single change is
a silver bullet**:

| lever | measured gain | what it removes | maps to |
|---|---|---|---|
| `mx.compile` the step body | **1.52×** (all N) | fuses ~35 *intermediate* round-trips into registers | ACTION_PLAN Fix 1.2 |
| **+ slab-CPML** | **1.4× more → 2.1× stacked** | the ~24 carried-ψ round-trips `compile` *cannot* fuse | ACTION_PLAN Fix 1.4 |
| **+ drop pad / ψ-stack / ×1 guards** | ~1.9× more → **~4× → ~440 Mcs/s** | the remaining 44→23 round-trips | ACTION_PLAN Fix 1.1 + 1.3 |

**Go:** Phase 1 is justified — every fix targets measured redundant traffic, and stacked they reach
~440 Mcs/s (~2.3× JAX-CPU's ~190). **But reorder it**: `compile` first (cheap, and makes the rest
compile-friendly), then **slab-CPML is co-critical, not "do-last"** (it is the binding post-compile
constraint). **Kill** the Phase-2 component-last layout experiment — it measures **1.00×** here.

---

## Method note: why the old story needed re-measuring

perf-baseline §1a derived "~3% of 273 GB/s" from an *assumed* 8-arrays/cell minimum — a
back-of-envelope, never a counter reading. `xctrace`/Xcode GPU traces are **unavailable on this machine** (Command-Line Tools
only). So instead of per-kernel counters we use a **round-trip (RT) model**: measure the achieved
coalesced bandwidth (240 GB/s), then express each variant's per-step time as the number of full
`(3,N³)` read+write round-trips it equals at that bandwidth. Because the bus is saturated (shown
below), RT is an honest proxy for "how much redundant data the step moves," and the whole ladder
reconciles in RT.

## The roofline (the denominator the docs lacked)

`profile_metal.py`, N=192, known-traffic ops (a copy/roll of an M-byte array moves 2M bytes):

- **coalesced copy: 240 GB/s = 88% of the 273 spec.** This is the real ceiling.
- `mx.roll` along inner/mid/outer axes: 20 / 227 / 236 GB/s. The 20 GB/s inner-axis figure is an
  artifact of a *self-chained pure roll*; on the engine's actual `y - roll(y)` difference pattern,
  **roll-diff vs slice-diff is 0.89–1.13× across all axes** — roll is *not* a culprit.
- **layout: stencil6 in `(3,N,N,N)` vs `(N,N,N,3)` = 1.00×.** No coalescing penalty from the
  component-leading layout.

## The four hypotheses — verdicts

| # | hypothesis | verdict | evidence |
|---|---|---|---|
| H1 | dispatch/encode-starved (GPU idle) | **minor** | `compile` (collapses ~130→1 kernels) buys only **1.5×**; `eval_every=8` is already at plateau (sync-every-step costs ~20%). If GPU were starved, fusion would buy far more. |
| **H2** | **redundant-traffic-bound (bus busy on waste)** | **CONFIRMED** | N=192: 70 ms/step = **99 RT** at 240 GB/s; ~5–8 RT are necessary. RT≈const ~95–100 at N=96/192/256 → pinned to roofline. |
| H3 | uncoalesced `(3,N,N,N)` layout | **REJECTED** | layout swap = 1.00×; copy already hits 88% of peak. |
| H4 | throughput collapses with N (CPML fraction grows) | **REFINED/partly wrong** | throughput *plateaus* at ~100 Mcs/s (116→105→100 for N=96→192→256 is a fading small-N cache bonus). CPML is a **constant ~25%** of traffic at all N, not a growing one. |

## The round-trip model (everything reconciles)

Real engine + the 2×2 predictive check (`profile_engine.py`, iso, consistent across N=96/192/256;
numbers below are N=192):

```
                              Mcs/s   RT/step   note (realized via profile_engine.py, N=192 iso)
eager, CPML on (pre-Phase-1)   ~105     ~99     original engine (pad+roll) — baseline
eager, CPML on (pad-free)       130      77     drop-pad slice-diff alone (Fix 1.3)
compiled, CPML on ← DEFAULT     211      47     + mx.compile E/H cores (Fix 1.1) — LANDED, 2.0×
compiled, CPML off              338      30     slab-CPML headroom (the ψ traffic removed)
lean (no CPML/pad/ψ-stack)     ~440     ~23     MLX-op ceiling (kernel-floor estimate)
necessary (R/W E,H + mat)      ~600+    ~5–8    one fused custom kernel per field (Phase 2)
```

Toggling CPML in the compiled loop removes **~17 RT** (47→30); in the eager pad-free loop it removes
~30 (77→47) — the carried-ψ recurrence is ~a quarter of all traffic at every stage. After compile +
drop-pad the default path **stalls at 47 RT because ψ_E/ψ_H and the ψ-stack are carried state that
must round-trip to DRAM — fusion cannot remove them.** That is *why* slab-CPML (compute ψ only on the
~8-cell boundary slabs, not the full N³) is the next lever — it targets exactly the ~17 RT compile
leaves on the floor, and `profile_engine` already shows the CPML-off path at 338 Mcs/s.

## Evidence-ranked fix list (reorders the ACTION_PLAN)

> **STATUS (branch `mlx-fork`, commit `053a590`):** items 1 + 3 **LANDED together** — pad-free
> slice-diff curl + compiled E/H cores. Default path 105 → **211 Mcs/s (2.0×, now > JAX-CPU)**,
> 99 → 47 RT, all 14 validation tests green. **Item 2 (slab-CPML) is next** (→ ~330 toward ~440).

1. **`mx.compile` the per-step core (Fix 1.2) — do first.** 1.52× at all N, low risk, and it makes
   the remaining fixes compile-friendly. Hoist source/detector gating host-side; inputs/outputs
   E,H,ψ_E,ψ_H; materials captured.
2. **Slab-CPML (Fix 1.4) — promote to co-critical.** The binding post-compile constraint (~24 RT).
   Store/append ψ only on the 6 boundary slabs. ~1.4× *on top of* compile (stacked **2.1×**). Note:
   constant gain across N (not N-growing, contra perf-baseline §1a).
3. **Drop per-step pad + guard ψ-stack + guard ×1 metric (Fix 1.1 + 1.3) — cleanup to the ceiling.**
   Takes 44→~23 RT (~1.9× more), reaching the lean **~440 Mcs/s**. The pad removal is the win here,
   *not* roll→slice (which is ~1.0× on the difference itself).

**Do NOT do:** the component-last `(N,N,N,3)` layout (Phase-2 Task 2.2) — measured **1.00×**.
Dispatch/async/`eval_every` tuning — already at plateau.

## Go / no-go

- **The decision is binary: commit to *near-full* Phase 1 (→ ~440, 2.3× CPU) or don't start.**
  With no memory moat, the only justification is throughput, and the cheap half (compile + slab-CPML
  → ~225) is only 1.2× over the zero-maintenance JAX-CPU path — not worth a parallel engine. The win
  is real but lives in the *last* fixes (no-pad / ψ-stack / metric guards that take 44→23 RT).
- **Phase 1: GO if committed**, reordered as above. Stacked, the fixes reach ~440 Mcs/s ≈ **2.3×
  JAX-CPU**, and unlike today the result is compile-fused so it should hold (or improve) with N. Land
  them as separate commits (compile → slab-CPML → cleanup), re-running `profile_engine.py` after each
  so each RT reduction is auditable. Stop early only if a fix fails to remove its predicted RT.
- **Phase 2 (custom `mx.fast.metal_kernel`): conditional GO, redefined.** Justified *only* as "fuse
  the entire step including the CPML recurrence into ~1 kernel" to collapse ~23→~5–8 RT (→ 600+
  Mcs/s) — i.e. the lever past ~440 if Phase 1 proves insufficient. The **layout** rationale for
  Phase 2 is killed by the 1.00× measurement.

## Value proof — there is NO memory moat; the case is throughput-only

Subprocess-isolated peak memory (`profile_memory.py`, clean per-process high-water, 20 steps):

| N | cells | MLX GPU peak | MLX proc RSS | JAX-CPU proc RSS |
|---|---|---|---|---|
| 96 | 0.9M | 1.4 GB | 1.5 | 1.4 |
| 192 | 7.1M | 2.6 GB | 6.7 | 5.9 |
| 256 | 16.8M | 4.6 GB | 15.0 | 12.9 |
| 320 | 32.8M | 9.1 GB | 23.1 | 20.5 |
| 384 | 56.6M | 15.6 GB | 28.6 | **28.0** |

**The "JAX-CPU can't fit large domains" argument is false.** Total process memory is ~identical at
every N (JAX even slightly lower); both fit N=384 (56.6 M cells) on this 51.5 GB machine. The earlier
"3–4× more" was an artifact of comparing JAX RSS to MLX *GPU-only* peak in a polluted shared process.
So MLX's value rests **entirely on throughput**, and that bar matters:

| state | MLX Mcs/s | vs JAX-CPU ~190 |
|---|---|---|
| today | ~100 | **CPU wins 1.9×** (the original observation) |
| compile + slab-CPML (measured 2.1×) | ~225 | only **1.2×** over CPU — marginal |
| full stack (~4×: + no-pad / ψ-stack / metric guards) | ~440 | **2.3×** over CPU — decisive |
| custom fused kernel (Phase 2, ~5–8 RT) | 600+ | ~3×+ over CPU |

→ MLX only *clearly* beats the zero-maintenance JAX-CPU path if Phase 1 is taken **near-fully to
~440**. The easy wins alone (compile+slab, ~225) barely edge CPU and would not justify a parallel
engine on their own.

## Reproduce

```bash
uv run python benchmarks/profile_metal.py  --N 192 --iters 100      # roofline + dispatch
uv run python benchmarks/profile_engine.py --N 192 --steps 200      # real-engine RT + 2×2
for b in mlx jax; do for N in 96 192 256 320 384; do
  uv run python benchmarks/profile_memory.py --backend $b --N $N --steps 20; done; done
```
