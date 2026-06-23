"""PEC/PMC boundary keep-masks for the MLX forward loop.

PEC zeros tangential E after each E-update; PMC zeros tangential H after each H-update (fdtdx
``PerfectElectricConductor.apply_post_E_update`` / ``PerfectMagneticConductor.apply_post_H_update``).
These are pure 1-cell-face masking ops -- no coefficients, no dynamic state -- so we freeze them once
into multiplicative keep-masks: ``0.0`` on every ``(component, cell)`` a boundary zeros, ``1.0``
elsewhere. Multiplying the post-injection field by the mask each step reproduces the zeroing exactly,
branch-free and allocation-light, and composes with both the Metal kernel and the MLX-op cores
(the mask lives in the loop, outside the cores).

The masks are built by running an all-ones field through fdtdx's *own* boundary appliers, so the
zeroed cells are bit-identical to the JAX engine -- no reimplementation of ``grid_slice`` /
``tangential_components`` here, and any future field-zeroing boundary is picked up automatically.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def freeze_boundary_masks(objects, grid_shape: tuple) -> tuple[mx.array | None, mx.array | None]:
    """Return ``(pec_keep, pmc_keep)`` multiplicative masks, or ``None`` where the boundary is absent.

    ``grid_shape`` is the field shape ``(3, Nx, Ny, Nz)``. Each mask is a float32 ``mx.array`` equal
    to ``0.0`` where the PEC/PMC boundaries zero the E/H tangential components and ``1.0`` elsewhere.
    ``None`` is returned when there are no PEC (resp. PMC) objects, so the loop skips the multiply.
    """
    import jax.numpy as jnp

    from fdtdx.fdtd.update import apply_boundary_post_E_update, apply_boundary_post_H_update

    pec_keep = pmc_keep = None
    ones = jnp.ones(grid_shape, dtype=jnp.float32)
    if objects.pec_objects:
        pec_keep = mx.array(np.ascontiguousarray(np.asarray(apply_boundary_post_E_update(ones, objects))))
    if objects.pmc_objects:
        pmc_keep = mx.array(np.ascontiguousarray(np.asarray(apply_boundary_post_H_update(ones, objects))))
    return pec_keep, pmc_keep
