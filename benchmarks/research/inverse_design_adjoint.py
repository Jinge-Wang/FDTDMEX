"""Minimal inverse-design / adjoint probe: jax.value_and_grad through run_fdtd (checkpointed
reversible engine), the FDTD backward pass the fork's MLX path drops. Runs on whatever JAX
platform is active, so use it to test JAX-CPU vs a plugin (JAX_PLATFORMS=mps). See
dev-docs/research/jax-mps-eval.md.

    JAX_PLATFORMS=cpu                          python inverse_design_adjoint.py 32 40
    JAX_PLATFORMS=mps JAX_MPS_ASYNC_DISPATCH=1 python inverse_design_adjoint.py 48 60
"""

import argparse
import os
import sys
import time

if os.environ.get("JAX_PLATFORMS") == "mps":
    import jax

    _od = jax.devices
    jax.devices = lambda backend=None: _od()
    _ou = jax.config.update
    jax.config.update = lambda k, v, *a, **kw: None if k == "jax_platform_name" else _ou(k, v, *a, **kw)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_forward as bf  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import fdtdx  # noqa: E402
from fdtdx.config import GradientConfig  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 40
args = argparse.Namespace(spacing=50e-9, wavelength=1e-6, courant=0.99, pml=6, detector="none", steps=steps, repeats=1)

print(f"# inverse-design adjoint  N={N} steps={steps}  jax_devices={[str(d) for d in jax.devices()]}", flush=True)
arrays, oc, config, key, info = bf.build_case("isotropic", N, args)
config = config.aset("gradient_config", GradientConfig(method="checkpointed", num_checkpoints=8))
base_inv_eps = arrays.inv_permittivities


def loss(scale):
    a2 = arrays.aset("inv_permittivities", base_inv_eps * scale)
    _, out = fdtdx.run_fdtd(arrays=a2, objects=oc, config=config, key=key, show_progress=False)
    return jnp.mean(out.fields.E**2)


try:
    t0 = time.perf_counter()
    val, g = jax.value_and_grad(loss)(1.0)
    jax.block_until_ready(g)
    dt = time.perf_counter() - t0
    print(
        f"OK  loss={float(val):.6e}  dL/dscale={float(g):.6e}  finite={bool(jnp.isfinite(g))}  time={dt:.3f}s",
        flush=True,
    )
    eps = 1e-3
    fd = (float(loss(1.0 + eps)) - float(loss(1.0 - eps))) / (2 * eps)
    print(f"FD-check dL/dscale~={fd:.6e}  rel_err={abs(fd - float(g)) / (abs(fd) + 1e-30):.3e}", flush=True)
except Exception as e:  # noqa: BLE001
    import traceback

    print(f"FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
    traceback.print_exc()
