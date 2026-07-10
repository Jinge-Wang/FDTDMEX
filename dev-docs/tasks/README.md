# Parallel tasks

Three independent workstreams toward the north star — **fold Metal acceleration back into upstream fdtdx and
retire the fork** — spun out so agents can run them in parallel. Full analysis + decision in
[`../research/jax-mps-eval.md`](../research/jax-mps-eval.md) and
[`../research/hybrid-and-optimization-plan.md`](../research/hybrid-and-optimization-plan.md).

| Task | What | Where it lands | Risk/effort |
|---|---|---|---|
| [2](task-2-monitor-upstream.md) | Port the **monitor 3.9× win** to fdtdx's JAX detectors | PR to upstream fdtdx | low, clean — **do now** |
| [3](task-3-fdtdx-mps-canvas.md) | Turn on **Apple-GPU support in fdtdx** via jax-mps (no kernel — the "canvas") | PR to upstream fdtdx | low — **do now** |
| [1](task-1-jaxmps-custom-kernel.md) | Make **jax-mps** run our hand-rolled Metal kernel (custom_call) | jax-mps fork/PR | build-heavy; **gated research bet — may postpone** |

**Order (revised after the 2026-07 jax-mps audit):** do **Tasks 2 & 3 now** (low-risk, independent, valuable
regardless) and keep improving the fork's **kernel** (the durable asset). **Task 1 is a months-scale research
bet, not a quick win** — the generic-kernel hook (#203) is an *unanswered, unsubmitted* proposal, and injecting
a kernel must clear jax-mps's MLX-resource race (#169) + no-donation cost. It is the **go/no-go spike that
gates retiring the fork**; start it deliberately and consider postponing full investment until the low-risk
wins land or the maintainer signals on #203. Tasks 1 & 3 both touch the `mps` platform — coordinate.
