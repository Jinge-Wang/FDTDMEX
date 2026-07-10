# Engine research harnesses

Reproducible drivers for the jax-mps / applejax vs MLX-engine study. Full results, method, and the
decision are in [`../../dev-docs/research/jax-mps-eval.md`](../../dev-docs/research/jax-mps-eval.md).

- `engine_matrix.py` — forward-FDTD throughput across JAX-CPU / jax-mps / MLX-op / fused Metal kernel.
- `inverse_design_adjoint.py` — `jax.value_and_grad` through `run_fdtd` (checkpointed adjoint), to test
  whether inverse design runs on a given JAX platform (CPU vs the `mps` plugin).

Both reuse `benchmarks/bench_forward.py` and select the engine only via env vars — **no engine edits**.

## Plugin environments (isolated; prebuilt wheels, no source build)

```
# jax-mps (shares the fork's jax 0.10.x + MLX)
uv venv --python 3.13 env-jaxmps
uv pip install --python env-jaxmps/bin/python "jax==0.10.1" "jaxlib==0.10.1" jax-mps -e /path/to/FDTDMEX

# applejax (older jax 0.9.x; MPSGraph) — currently crashes natively on the fdtdx graph
uv venv --python 3.13 env-applejax
uv pip install --python env-applejax/bin/python "jax==0.9.2" "jaxlib==0.9.2" applejax -e /path/to/FDTDMEX
```

Note: the fork's `config`/`sharding` assume `cpu`/`gpu`/`METAL` platforms; the scripts apply a small
in-process shim so fdtdx runs under the `mps` plugin platform. Making the fork recognize `mps` natively
would remove the need for the shim (small, autodiff-safe).
