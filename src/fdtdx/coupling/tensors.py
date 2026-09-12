"""Voigt notation: symmetric tensors as 6-vectors, and the constitutive matrices that read them.

A photoelastic or electro-optic response is written with the symmetric pair of tensor indices
contracted into one index running 0..5, in the order ``(xx, yy, zz, yz, xz, xy)``, which is what
:class:`~fdtdx.coupling.responses.PhotoelasticResponse` and
:class:`~fdtdx.coupling.responses.PockelsResponse` expect. Four things live here:

* the assembly, :func:`voigt_from_tensor` and its inverse, with the *engineering* shear convention
  a strain uses (the last three entries are twice the tensor components) kept separate from the
  plain convention a stress or a ``d(1/eps)`` uses. Getting that factor of two wrong changes a
  shear-driven index change by exactly a factor of two and nothing else flags it, so it is a named
  argument here rather than a convention in a docstring;
* :func:`voigt_permute`, which re-expresses a ``(6, 6)``, ``(6, 3)`` or ``(6,)`` object in a frame
  whose axes are a signed permutation of the old ones, and :func:`permute_tensor`, its companion
  for the base permittivity. That is the crystal-cut question: a published electro-optic or
  photoelastic matrix is given in the crystal's own frame, and the device puts the crystal axes on
  whichever grid axes the layout needs. The material's own tensor and its response matrix have to
  be permuted together, or the two end up in different frames;
* :func:`photoelastic_from_stress_optic`, the exact isotropic conversion of a Maxwell-Neumann
  stress-optic pair ``(B1, B2)`` — the form every vendor stress-optic model is written in — into
  the ``(6, 6)`` photoelastic matrix that a strain-driven response consumes;
* :func:`voigt_samples_from_tensor`, which turns a sampled tensor field into the 6-component
  samples a response reads, rotating the components into the Yee frame *before* contracting them.

Nothing here imports DOLFINx or JAX; it is arithmetic on NumPy arrays.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from fdtdx.coupling.frames import PointTransform, RadialPlaneTransform, _as_square
from fdtdx.coupling.lattice import YeeLatticeSamples, lattice_points

#: The tensor index pair each Voigt index stands for, in the order the response classes use.
VOIGT_ORDER: tuple[tuple[int, int], ...] = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))

#: Human-readable names of the six Voigt entries, aligned with :data:`VOIGT_ORDER`.
VOIGT_LABELS: tuple[str, ...] = ("xx", "yy", "zz", "yz", "xz", "xy")

_VOIGT_OF_PAIR: dict[tuple[int, int], int] = {}
for _index, (_i, _j) in enumerate(VOIGT_ORDER):
    _VOIGT_OF_PAIR[(_i, _j)] = _index
    _VOIGT_OF_PAIR[(_j, _i)] = _index


def _checked_permutation(perm: Sequence[int], signs: Sequence[float] | None) -> tuple[tuple[int, ...], np.ndarray]:
    axes = tuple(int(a) for a in perm)
    if sorted(axes) != [0, 1, 2]:
        raise ValueError(f"perm must reorder (0, 1, 2), got {perm}")
    if signs is None:
        return axes, np.ones(3, dtype=np.float64)
    s = np.asarray(signs, dtype=np.float64).reshape(-1)
    if s.shape != (3,) or not np.all(np.abs(np.abs(s) - 1.0) < 1e-12):
        raise ValueError(f"signs must be three entries of +1 or -1, got {signs}")
    return axes, s


def voigt_from_tensor(tensor: np.ndarray, engineering: bool = True) -> np.ndarray:
    """Contract symmetric tensors into 6-vectors in the order ``(xx, yy, zz, yz, xz, xy)``.

    Args:
        tensor (np.ndarray): ``(..., 3, 3)`` or ``(..., 2, 2)``, or the flattened ``(..., 9)`` /
            ``(..., 4)`` forms ``FemField.evaluate`` returns. A 2-D tensor is lifted with a zero
            third row and column, which for a strain is the plane-strain reading; state a non-zero
            out-of-plane entry upstream (``FemField.symmetric_gradient_of(..., out_of_plane=...)``)
            rather than here, so the modelling choice sits with the solve.
        engineering (bool): Double the three shear entries, which is the strain convention the
            photoelastic matrix is tabulated against. ``False`` for a stress or a ``d(1/eps)``.

    Returns:
        np.ndarray: ``(..., 6)``.

    Raises:
        ValueError: If the input is not symmetric-tensor shaped.
    """
    t = _as_square(tensor)
    if t.shape[-1] == 2:
        padded = np.zeros((*t.shape[:-2], 3, 3), dtype=np.float64)
        padded[..., :2, :2] = t
        t = padded
    out = np.empty((*t.shape[:-2], 6), dtype=np.float64)
    for index, (i, j) in enumerate(VOIGT_ORDER):
        out[..., index] = 0.5 * (t[..., i, j] + t[..., j, i])
    if engineering:
        out[..., 3:] *= 2.0
    return out


def tensor_from_voigt(vector: np.ndarray, engineering: bool = True) -> np.ndarray:
    """Expand 6-vectors back into symmetric ``(..., 3, 3)`` tensors; the inverse of :func:`voigt_from_tensor`."""
    v = np.asarray(vector, dtype=np.float64)
    if v.shape[-1] != 6:
        raise ValueError(f"a Voigt vector needs 6 entries on its last axis, got {v.shape}")
    out = np.zeros((*v.shape[:-1], 3, 3), dtype=np.float64)
    for index, (i, j) in enumerate(VOIGT_ORDER):
        entry = v[..., index]
        if engineering and index >= 3:
            entry = 0.5 * entry
        out[..., i, j] = entry
        out[..., j, i] = entry
    return out


def voigt_index_permutation(perm: Sequence[int], signs: Sequence[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """How the six Voigt entries move when the axes are permuted (and optionally flipped).

    With new axis ``i`` taken from old axis ``perm[i]`` and an optional sign per new axis, a
    second-rank tensor transforms as ``T_new[i, j] = s_i s_j T_old[perm[i], perm[j]]``. A signed
    permutation maps diagonal entries to diagonal entries and shears to shears, so in Voigt form
    the whole thing collapses to ``u_new[I] = sign[I] * u_old[index[I]]`` — and the same index map
    serves the engineering-shear convention, because the factor of two rides along with the entry.

    Args:
        perm (Sequence[int]): A reordering of ``(0, 1, 2)``; new axis ``i`` is old axis ``perm[i]``.
        signs (Sequence[float] | None): ``+1`` or ``-1`` per new axis; all ``+1`` when ``None``.

    Returns:
        tuple: ``(index, sign)``, both length 6.
    """
    axes, s = _checked_permutation(perm, signs)
    index = np.zeros(6, dtype=np.int64)
    sign = np.zeros(6, dtype=np.float64)
    for out_index, (i, j) in enumerate(VOIGT_ORDER):
        index[out_index] = _VOIGT_OF_PAIR[(axes[i], axes[j])]
        sign[out_index] = s[i] * s[j]
    return index, sign


def permute_tensor(tensor: np.ndarray, perm: Sequence[int], signs: Sequence[float] | None = None) -> np.ndarray:
    """``T_new[i, j] = s_i s_j T_old[perm[i], perm[j]]`` on ``(..., 3, 3)`` tensors.

    The companion of :func:`voigt_permute` for the base permittivity: a crystal-cut permutation has
    to be applied to the material's own tensor and to its response matrix together, or the two end
    up in different frames.
    """
    axes, s = _checked_permutation(perm, signs)
    t = np.asarray(tensor, dtype=np.float64)
    if t.shape[-2:] != (3, 3):
        raise ValueError(f"permute_tensor needs (..., 3, 3), got {t.shape}")
    order = list(axes)
    picked = t[..., order, :][..., :, order]
    return picked * np.outer(s, s)


def voigt_permute(matrix: np.ndarray, perm: Sequence[int], signs: Sequence[float] | None = None) -> np.ndarray:
    """Re-express a Voigt object in a frame whose axes are a signed permutation of the old ones.

    Covers the three shapes a response carries: the ``(6, 6)`` photoelastic matrix ``p``, the
    ``(6, 3)`` electro-optic matrix ``r`` (rows Voigt, columns vector components), and a bare
    ``(6,)`` Voigt vector. The permutation is the crystal cut: for an x-cut crystal whose optic
    axis lies along grid ``x``, grid axis 0 is crystal axis 2, so ``perm=(2, 0, 1)``.

    Args:
        matrix (np.ndarray): ``(6,)``, ``(6, 3)`` or ``(6, 6)`` in the old frame.
        perm (Sequence[int]): New axis ``i`` is old axis ``perm[i]``.
        signs (Sequence[float] | None): ``+1`` or ``-1`` per new axis.

    Returns:
        np.ndarray: The same shape, in the new frame.

    Raises:
        ValueError: For any other shape, or an invalid permutation or sign list.
    """
    axes, s = _checked_permutation(perm, signs)
    index, sign = voigt_index_permutation([int(a) for a in axes], [float(w) for w in s])
    index = np.asarray(index)
    axes_idx = np.asarray([int(a) for a in axes])
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape == (6,):
        return sign * m[index]
    if m.shape == (6, 3):
        return sign[:, None] * s[None, :] * m[np.ix_(index, axes_idx)]
    if m.shape == (6, 6):
        return sign[:, None] * sign[None, :] * m[np.ix_(index, index)]
    raise ValueError(f"voigt_permute handles (6,), (6, 3) and (6, 6) objects, got {m.shape}")


def cubic_photoelastic_matrix(p11: float, p12: float, p44: float) -> np.ndarray:
    """The ``(6, 6)`` photoelastic matrix of a cubic crystal (silicon, germanium, GaAs).

    Three independent entries in the crystal frame: ``p11`` on the diagonal of the normal block,
    ``p12`` off it, ``p44`` on each shear entry. Rows and columns are in :data:`VOIGT_ORDER`, and
    the strain it multiplies carries engineering shears.
    """
    p = np.zeros((6, 6), dtype=np.float64)
    p[:3, :3] = float(p12)
    np.fill_diagonal(p[:3, :3], float(p11))
    p[3, 3] = p[4, 4] = p[5, 5] = float(p44)
    return p


def isotropic_photoelastic_matrix(p11: float, p12: float) -> np.ndarray:
    """The ``(6, 6)`` photoelastic matrix of an isotropic medium (fused silica, a polymer).

    Isotropy is the cubic form with the shear entry fixed by the other two,
    ``p44 = (p11 - p12) / 2`` — the same relation the elastic constants of an isotropic solid obey.
    """
    return cubic_photoelastic_matrix(p11, p12, 0.5 * (float(p11) - float(p12)))


def photoelastic_from_stress_optic(B1: float, B2: float, n: float, E: float, nu: float) -> np.ndarray:
    """Exact isotropic conversion of a Maxwell-Neumann stress-optic pair into a photoelastic matrix.

    A vendor stress-optic model is written on stress,
    ``dn_i = -[B1 sigma_i + B2 (sigma_j + sigma_k)]``, while a mechanical solve naturally produces
    strain and the fork's response is written on strain,
    ``d(1/n^2)_I = sum_J p_IJ S_J``. For an isotropic medium the two are the same physics:
    substituting Hooke's law and ``dn = -(n^3 / 2) d(1/n^2)`` gives

    ``B1 = n^3 / (2 E) * (p11 - 2 nu p12)`` and ``B2 = n^3 / (2 E) * (p12 - nu (p11 + p12))``,

    a 2x2 linear system this function inverts. Converting is preferable to adding a second response
    class: it is exact, it keeps one code path in the engine, and it puts the material data
    disagreement (published ``p`` values and published ``(B1, B2)`` values do not agree to better
    than about 10 % for silica) where it belongs, in the inputs.

    Args:
        B1 (float): Stress-optic coefficient for the stress along the polarisation, m^2/N.
        B2 (float): Stress-optic coefficient for the two transverse stresses, m^2/N.
        n (float): Unstressed refractive index.
        E (float): Young's modulus, Pa (same unit system as ``B1``, ``B2``).
        nu (float): Poisson's ratio.

    Returns:
        np.ndarray: The ``(6, 6)`` isotropic photoelastic matrix, dimensionless.

    Raises:
        ValueError: If ``nu`` makes the system singular (``1 - nu - 2 nu^2 = 0``, i.e. ``nu = 0.5``).
    """
    scale = 2.0 * float(E) / float(n) ** 3
    b1, b2 = scale * float(B1), scale * float(B2)
    nu = float(nu)
    determinant = (1.0 - nu) - 2.0 * nu * nu
    if abs(determinant) < 1e-15:
        raise ValueError(f"the stress-optic system is singular at nu={nu} (incompressible medium)")
    p11 = ((1.0 - nu) * b1 + 2.0 * nu * b2) / determinant
    p12 = (nu * b1 + b2) / determinant
    return isotropic_photoelastic_matrix(p11, p12)


def stress_optic_from_photoelastic(p11: float, p12: float, n: float, E: float, nu: float) -> tuple[float, float]:
    """``(B1, B2)`` from an isotropic ``(p11, p12)``; the inverse of :func:`photoelastic_from_stress_optic`."""
    factor = float(n) ** 3 / (2.0 * float(E))
    nu = float(nu)
    return (
        factor * (float(p11) - 2.0 * nu * float(p12)),
        factor * (float(p12) - nu * (float(p11) + float(p12))),
    )


def voigt_samples_from_tensor(
    samples: YeeLatticeSamples,
    transform: PointTransform | RadialPlaneTransform | None = None,
    engineering: bool = True,
    name: str = "S",
    unit: str | None = None,
) -> YeeLatticeSamples:
    """Turn sampled mesh-frame tensors into the 6-component samples a response consumes.

    Two steps that must not be separated: the components are rotated into the Yee frame by the same
    transform that mapped the positions
    (:meth:`fdtdx.coupling.frames.PointTransform.apply_values` with ``rank=2``), and only then
    contracted into Voigt form. Doing the contraction first would put the shear entries on the
    wrong axes.

    Args:
        samples (YeeLatticeSamples): Per lattice, ``(Nx, Ny, Nz, m, m)`` or the flattened
            ``(Nx, Ny, Nz, m * m)`` mesh-frame tensors.
        transform (PointTransform | RadialPlaneTransform | None): The one used to sample; identity
            when ``None``.
        engineering (bool): Engineering shears, i.e. a strain. See :func:`voigt_from_tensor`.
        name (str): Field name of the result (``"S"`` is what ``PhotoelasticResponse`` reads).
        unit (str | None): Unit label of the result; ``None`` (dimensionless) for a strain.

    Returns:
        YeeLatticeSamples: The same lattices and coverage, with ``(Nx, Ny, Nz, 6)`` values.
    """
    transform = transform or PointTransform()
    values: dict[str, np.ndarray] = {}
    for lattice, block in samples.values.items():
        points, shape = lattice_points(samples.edges, lattice)
        flat = np.asarray(block, dtype=np.float64).reshape((int(np.prod(shape)), -1))
        rotated = transform.apply_values(flat, rank=2, points=points)
        values[lattice] = voigt_from_tensor(rotated, engineering=engineering).reshape((*shape, 6))
    provenance: dict[str, Any] = dict(samples.provenance)
    provenance["voigt"] = {
        "order": list(VOIGT_LABELS),
        "engineering_shear": bool(engineering),
        "from_field": samples.name,
    }
    return YeeLatticeSamples(
        edges=samples.edges,
        values=values,
        covered={lattice: np.array(mask, copy=True) for lattice, mask in samples.covered.items()},
        name=name,
        unit=unit,
        transform=dict(samples.transform),
        provenance=provenance,
    )
