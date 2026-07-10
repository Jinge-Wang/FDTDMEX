"""Host-side assembly of the (time-invariant) CPML recurrence coefficients.

Upstream fdtdx #384 ("Refactor PML: attach constants to obj") moved the CPML ``a``/``b``/
``1/kappa`` coefficients out of the global ``ArrayContainer`` (they no longer live in
``arrays.alpha/kappa/sigma``) and onto each :class:`PerfectlyMatchedLayer` object as
``pml_a_E``/``pml_b_E``/``inv_kappa_E`` (+ ``_H``), sized to that PML's slab. The fork's Metal
kernel still consumes a single global ``(6, Nx, Ny, Nz)`` coefficient array (E-side axis-``k``
in channel ``k``, H-side in ``k+3``), so :func:`build_cpml_coeffs_from_pml_objects` re-assembles
that global view from the per-object arrays. All quantities are time-invariant, so this runs
once on the host (numpy) and the MLX curl never recomputes them per step. Computed in the field
dtype (float32 by default) to match the JAX path.
"""

from __future__ import annotations

import numpy as np

#: Map a PML object's ``axis`` to the two channels its ``(psi_1, psi_2)`` tuple occupies in the
#: fork's 6-channel derivative-term ψ layout (``_AX = (1, 2, 2, 0, 0, 1)``). Derived from the
#: ``step_cpml`` call order in ``fdtdx.core.physics.curl.curl_E``/``curl_H``: for a PML on axis
#: ``a`` the first corrected derivative (``d_a_F_j`` → ``psi_1``) and second (``d_a_F_i`` →
#: ``psi_2``) land in these channels. Used to round-trip ψ to/from the per-object dict that
#: upstream #379 now stores in ``arrays.fields.psi_E/psi_H``.
PML_AXIS_TO_PSI_CHANNELS: dict[int, tuple[int, int]] = {0: (3, 4), 1: (5, 0), 2: (1, 2)}


def build_cpml_coeffs_from_pml_objects(
    pml_objects, field_shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the global ``(a, b, inv_kappa)`` CPML arrays, each shape ``(6, Nx, Ny, Nz)``.

    Channels ``0..2`` hold the E-side axis-``0..2`` profile, ``3..5`` the H-side, matching the
    layout ``detect_pml_slabs`` and the Metal kernel expect. Each PML on ``axis`` writes its E
    coefficients into channel ``axis`` and its H coefficients into ``axis+3`` over its
    ``grid_slice``; every other cell keeps the trivial ``(a=0, b=1, inv_kappa=1)`` value, exactly
    like the old full-grid profiles outside the PML shell.
    """
    nx, ny, nz = field_shape
    dtype = np.float32
    a = np.zeros((6, nx, ny, nz), dtype=dtype)
    b = np.ones((6, nx, ny, nz), dtype=dtype)
    inv_kappa = np.ones((6, nx, ny, nz), dtype=dtype)
    for pml in pml_objects:
        k = int(pml.axis)
        sl = tuple(pml.grid_slice)
        a[k][sl] = np.asarray(pml.pml_a_E, dtype=dtype)
        b[k][sl] = np.asarray(pml.pml_b_E, dtype=dtype)
        inv_kappa[k][sl] = np.asarray(pml.inv_kappa_E, dtype=dtype)
        a[k + 3][sl] = np.asarray(pml.pml_a_H, dtype=dtype)
        b[k + 3][sl] = np.asarray(pml.pml_b_H, dtype=dtype)
        inv_kappa[k + 3][sl] = np.asarray(pml.inv_kappa_H, dtype=dtype)
    return a, b, inv_kappa


def detect_pml_slabs(a: np.ndarray, b: np.ndarray, inv_kappa: np.ndarray, pad: int = 1) -> list[tuple[int, int]]:
    """Per-axis ``(lo, hi)`` PML slab thickness: the CPML correction is confined to indices
    ``[0:lo]`` and ``[N-hi:N]`` along each axis ``k``.

    ``a``/``b``/``inv_kappa`` have shape ``(6, Nx, Ny, Nz)``; index ``k`` is the E-side axis-``k``
    profile, ``k+3`` the H-side. Each is non-trivial (``a≠0`` / ``b≠1`` / ``inv_kappa≠1``) only in
    the two slabs perpendicular to axis ``k``. The slab correction ``(inv_kappa-1)·d + ψ`` is
    exactly zero wherever all three are trivial, so the detected support is exact; ``pad`` widens
    each slab by a safety margin of provably-zero cells (cheap, keeps it exact under float ramps).
    """
    slabs: list[tuple[int, int]] = []
    for k in range(3):
        active = np.zeros(a.shape[1:], dtype=bool)
        for idx in (k, k + 3):
            active |= a[idx] != 0.0
            active |= b[idx] != 1.0
            active |= inv_kappa[idx] != 1.0
        other = tuple(ax for ax in range(3) if ax != k)
        mask = active.any(axis=other)  # 1-D along axis k
        n = int(mask.shape[0])
        lo = 0
        while lo < n and mask[lo]:
            lo += 1
        hi = 0
        while hi < n and mask[n - 1 - hi]:
            hi += 1
        # widen by `pad` provably-zero cells, but never overlap (lo+hi <= n)
        if lo:
            lo = min(n, lo + pad)
        if hi:
            hi = min(n, hi + pad)
        if lo + hi > n:
            lo, hi = n, 0  # whole axis is PML (degenerate tiny domain) -> one slab covers it
        slabs.append((lo, hi))
    return slabs
