# Parallel tasks

Three independent workstreams toward the north star — **fold Metal acceleration back into upstream fdtdx and
retire the fork** — spun out so agents can run them in parallel. Full analysis + decision in
[`../research/jax-mps-eval.md`](../research/jax-mps-eval.md) and
[`../research/hybrid-and-optimization-plan.md`](../research/hybrid-and-optimization-plan.md).

| Task | What | Where it lands | Risk/effort |
|---|---|---|---|
| [1](task-1-jaxmps-custom-kernel.md) | Make **jax-mps** run our hand-rolled Metal kernel (custom_call / FFI) | jax-mps fork/PR | build-heavy; the linchpin |
| [2](task-2-monitor-upstream.md) | Port the **monitor 3.9× win** to fdtdx's JAX detectors | PR to upstream fdtdx | low, clean |
| [3](task-3-fdtdx-mps-canvas.md) | Turn on **Apple-GPU support in fdtdx** via jax-mps (no kernel — the "canvas") | PR to upstream fdtdx | low |

Order: Tasks 2 and 3 are low-effort, independent, and valuable now; Task 1 is the linchpin that decides
whether the fork can be retired. Tasks 1 and 3 both touch how fdtdx runs under the `mps` platform — coordinate.
