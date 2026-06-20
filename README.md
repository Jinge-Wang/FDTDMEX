# FDTDMEX

**Forward-first, Metal-native FDTD electromagnetics on Apple Silicon.**

FDTDMEX is a finite-difference time-domain (FDTD) Maxwell solver built on [MLX](https://github.com/ml-explore/mlx) — Apple's array framework — to run **natively on the Metal GPU with unified memory**. It targets a specific, underserved niche: **fast, large *forward* simulations on a single Mac**, without a CUDA GPU or a cloud-solver subscription.

> **Status: pre-implementation scaffold.** This repository currently contains the architecture, docs, agent rules, and importable package stubs. No physics kernels are implemented yet. See [docs/roadmap.md](docs/roadmap.md).

## Why this exists

Most differentiable FDTD tooling (e.g. [FDTDX](https://github.com/ymahlau/fdtdx)) is built on JAX, whose Metal backend is unusable on macOS (no JIT). The strongest case for Apple Silicon here is **memory, not just compute**: a fully-anisotropic simulation stores a 3×3 permittivity *tensor per voxel* — ~9× the isotropic footprint — which saturates the PCIe bus or VRAM of a single CUDA GPU. Apple's **unified memory** (up to 512 GB) lets the GPU address the entire domain with no host↔device streaming. FDTDMEX leans into that.

Design priorities:
- **Forward simulation on Metal** (inverse design stays on CUDA/JAX clusters — it needs cluster-scale parallelism).
- **Full-tensor anisotropic, spatially heterogeneous materials** as a first-class citizen.
- **Non-uniform grids done right** — spacing-weighted curl and interpolation (2nd-order on graded meshes), see [docs/nonuniform-grid.md](docs/nonuniform-grid.md).
- **Subpixel smoothing** of static geometry (Kottke/Farjadpour), a pre-time-stepping step.
- An **LLM-orchestratable** front end (declarative config + MCP server + web UI).

New to MLX/JAX? Start with **[docs/getting-started.md](docs/getting-started.md)**.

## Workstreams

| | Workstream | Summary |
|---|---|---|
| **WS-A** | Forward MLX engine | Curl, E/H update (isotropic→full-anisotropic), CPML, sources, detectors, time loop. Non-uniform-grid-aware. |
| **WS-B** | Mode solver front end | 2D-Yee full-vectorial FD eigensolver + mode overlap; injection ported from FDTDX's TFSF. |
| **WS-C** | Subpixel smoothing | Kottke/Farjadpour effective-tensor averaging as a host pre-step feeding WS-A. |
| **WS-D** | Orchestration | Declarative config, MCP server, git-like history, locally-hosted web UI / 3D editor. |

See [docs/architecture.md](docs/architecture.md) for the data flow and dependency graph.

## Install (once the package has content)

Requires **Apple Silicon** (M-series) and recent macOS. Uses [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                      # core
uv sync --extra viz          # + matplotlib/plotly/pyvista/trame
uv sync --extra validation   # + jax (CPU cross-check oracle)
uv sync --extra dev          # + pytest/ruff/pre-commit
```

Quickstart (placeholder — API not implemented yet):

```python
import fdtdmex  # noqa: F401  (currently stubs)
```

## Repository layout

```
src/fdtdmex/      core package (backend, core, fdtd, materials[, smoothing],
                  geometry, sources, detectors, modes, io, viz)
server/           MCP server (WS-D)
web/              web UI (WS-D, placeholder)
docs/             architecture, physics, porting notes, workstream specs
tests/            unit / integration / validation
examples/         runnable examples (later)
benchmarks/       performance harnesses (later)
reference/        pointers to ../fdtdx (MIT) and ../meep (GPL) — not vendored
.claude/skills/   vibe-coding skills (framework, porting, validation)
```

## License & attribution

Provisionally **Apache-2.0** (see [LICENSE](LICENSE), [NOTICE](NOTICE)) — final licensing is undecided; see [docs/licensing.md](docs/licensing.md). FDTDMEX draws heavily on **FDTDX** (MIT) for its numerical design and partially consults **MEEP** (GPL) for subpixel smoothing and near-to-far-field math. Those projects are referenced, not vendored.
