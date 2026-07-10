"""Apples-to-apple forward-FDTD engine matrix — JAX-CPU / jax-mps / MLX-op / fused Metal kernel.

Reuses ``benchmarks/bench_forward.py`` helpers; selects the engine purely by env (no engine edits).
One engine per process (JAX_PLATFORMS is fixed at import). See dev-docs/research/jax-mps-eval.md.

    # JAX-CPU baseline
    JAX_PLATFORMS=cpu FDTDMEX_BACKEND=jax   python engine_matrix.py jax isotropic 48,64,96 60 2
    # jax-mps (install jax-mps into a py3.13 env with jax 0.10.x)
    JAX_PLATFORMS=mps JAX_MPS_ASYNC_DISPATCH=1 FDTDMEX_BACKEND=jax python engine_matrix.py jax isotropic 48,64 60 1
    # MLX-op cores / fused Metal kernel (run in the fork's own CPU-default env)
    FDTDMEX_BACKEND=mlx FDTDMEX_METAL_KERNEL=0 python engine_matrix.py mlx isotropic 48,64 60 2
    FDTDMEX_BACKEND=mlx FDTDMEX_METAL_KERNEL=1 python engine_matrix.py mlx isotropic 48,64 60 2
"""

import argparse
import os
import sys

# Non-intrusive shim for the jax-mps 'mps' platform (fdtdx config/sharding assume cpu/gpu/METAL).
if os.environ.get("JAX_PLATFORMS") == "mps":
    import jax

    _orig_devices = jax.devices
    jax.devices = lambda backend=None: _orig_devices()
    _orig_update = jax.config.update
    jax.config.update = lambda k, v, *a, **kw: None if k == "jax_platform_name" else _orig_update(k, v, *a, **kw)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_forward as bf  # noqa: E402
import jax  # noqa: E402

backend = sys.argv[1]
material = sys.argv[2]
sizes = [int(x) for x in sys.argv[3].split(",")]
steps = int(sys.argv[4])
repeats = int(sys.argv[5]) if len(sys.argv) > 5 else 2
args = argparse.Namespace(
    spacing=50e-9, wavelength=1e-6, courant=0.99, pml=8, detector="none", steps=steps, repeats=repeats
)

label = f"{backend}/METAL_KERNEL={os.environ.get('FDTDMEX_METAL_KERNEL', '-')}/JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', 'default')}"
print(f"# engine={label}  jax_devices={[str(d) for d in jax.devices()]}  material={material} steps={steps}", flush=True)
for n in sizes:
    try:
        rec = bf.time_cell(material, n, backend, args)
        print(
            f"N={n:4d} cells={rec['cells']:>9d}  {rec['throughput_mcellsteps_s']:9.1f} Mcs/s  "
            f"median={rec['time_s_median']:.4f}s  finite={rec['finite']}  status={rec['status']}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        print(f"N={n:4d}  FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        traceback.print_exc()
        break
