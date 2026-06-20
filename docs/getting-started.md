# Getting Started

This guide is written for someone **new to MLX and JAX**. It covers what the dependencies are, how to set up the environment, how to confirm you're actually running on the Metal GPU, and a
mental-model mapping so the codebase reads clearly.

## Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4…). MLX's GPU backend is Metal-only; there is no CUDA path.
- **Recent macOS** (Metal-capable; macOS 13.5+ recommended).
- **Python 3.11–3.13**.
- **[uv](https://docs.astral.sh/uv/)** — a fast Python package/environment manager (replaces
  `pip` + `venv` + `pip-tools`). Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

## What each dependency is and why

| Package | Role |
|---|---|
| **mlx** | Apple's array framework. Think "NumPy + JAX-style transforms" with a **Metal GPU backend** and **unified memory** (CPU and GPU share one address space — no host↔device copies). This is the compute engine. |
| **numpy** | Host-side arrays and the interop bridge to/from MLX (`mx.array(np_array)` / `np.array(mx_array)`). |
| **scipy** | Sparse eigensolver for the mode solver (WS-B), plus FFTs/special functions on the host. |
| **gdstk** | Read GDSII layouts (planar photonics geometry). |
| **trimesh** | STL / triangle-mesh geometry I/O. |
| **pydantic** | Declarative, validated config schema — the Tidy3D-like front end (WS-D). |
| **h5py** | HDF5 for large field-result files. |
| **rich / loguru** | Console output, progress, logging. |
| *extras* `viz` | matplotlib, plotly, pyvista, trame — 2D/3D and web visualization. |
| *extras* `mcp` | the MCP server for LLM orchestration (WS-D). |
| *extras* `validation` | **jax** — used only as a CPU cross-check oracle (run the same case through FDTDX). |
| *extras* `dev` | pytest, ruff, pre-commit. |

## Setup

```bash
# from the repo root
uv sync                       # core deps, creates .venv automatically
uv sync --extra dev           # + test/lint tooling
uv sync --extra viz           # + visualization
uv sync --all-extras          # everything

uv run python -c "import fdtdmex; print('ok')"   # import sanity check
uv run pytest                 # run the test suite
```

`uv run <cmd>` executes inside the managed environment — you don't need to manually activate it.

## Verify MLX is using the Metal GPU

```python
import mlx.core as mx
print(mx.default_device())          # expect Device(gpu, 0) on Apple Silicon
a = mx.ones((4096, 4096))
b = (a @ a)
mx.eval(b)                          # force evaluation (MLX is lazy)
print(b.dtype, b.shape)
```

You can force placement with `mx.set_default_device(mx.gpu)` (or `mx.cpu`). Unified memory means an `mx.array` is usable from both CPU and GPU without explicit transfers.

## MLX ↔ JAX mental model (for readers coming from JAX or NumPy)

MLX deliberately resembles JAX's functional style:

| Concept | JAX | MLX |
|---|---|---|
| array | `jnp.array` | `mx.array` |
| compile/fuse | `jax.jit(f)` | `mx.compile(f)` |
| gradient | `jax.grad(f)` | `mx.grad(f)` |
| value+grad | `jax.value_and_grad` | `mx.value_and_grad` |
| vectorize | `jax.vmap` | `mx.vmap` |
| custom VJP | `jax.custom_vjp` | `mx.custom_function` |
| FFT | `jnp.fft` | `mx.fft` |
| RNG | `jax.random` | `mx.random` |

**Two big differences to keep in mind:**
1. **Lazy evaluation.** MLX builds a graph and only computes on `mx.eval(...)` (or when you print/ convert to numpy). In the time loop we call `mx.eval` periodically to bound graph/memory growth.
2. **No traced control flow.** JAX has `lax.scan` / `lax.while_loop`; MLX does not. We use a plain Python `for` loop over time steps with the per-step body wrapped in `mx.compile`.

FDTDMEX does **not** use MLX's autodiff (forward-only on Metal). Differentiable inverse design stays on JAX/CUDA clusters — see [architecture.md](architecture.md).

## Next
- Skim [physics.md](physics.md) and [porting-notes.md](porting-notes.md) before touching engine code.
- Build order and current status: [roadmap.md](roadmap.md).
