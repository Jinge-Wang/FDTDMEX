"""Bend modes: the conformal map that turns a curved guide into a straight one.

A guide bent with radius ``R`` has no translational invariance, so it has no propagation constant
in the ordinary sense. The classical way out (Heiblum & Harris, IEEE J. Quantum Electron. 11, 75
(1975)) is a coordinate map that trades the curvature for a graded index: with the azimuthal
dependence written as ``exp(-i m phi)`` and the *reference radius* ``R`` fixed at the centre of the
mode plane, the substitution

.. code-block:: text

    u = R ln(r / R),     n_eq(u) = n(r) * (r / R) = n(r) * exp(u / R),     beta = m / R

turns the scalar wave equation of the bend into the scalar wave equation of a straight guide with
index ``n_eq``. The permittivity therefore carries ``exp(2 u / R)``, and the eigenvalue of the
straight problem is the *azimuthal* one: ``n_eff = beta / k0 = m / (k0 R)``, an effective index
defined **at the reference radius**, not at the mode's own centroid.

Three forms of the same statement are implemented, and they are **not** interchangeable. The bend
shift ``n_eff(R) - n_eff(infinity)`` is itself second order in ``1 / R``, so a form that is only
first-order faithful gets that shift wrong by an O(1) factor at *every* radius, not just at small
ones. Measured against a cylindrical-Bessel solution of the same guide (0.5 um core, index 2.448 in
1.444, 1.55 um), as a relative error of the bend shift:

``"tensor"`` (default)
    Transformation optics, exact for the *full vector* problem and on the physical grid. The map
    ``w = R phi`` has Jacobian ``diag(1, 1, R/r)``, so ``eps' = J eps J^T / det J`` gives

    .. code-block:: text

        eps_t -> eps_t * (r / R)        (both transverse components)
        eps_w -> eps_w * (R / r)        (the propagation component)

    and the same for ``mu``. Nothing is approximated and nothing is polarisation-specific: an
    isotropic medium becomes diagonally anisotropic, which is a tier the mode operator already
    supports. This is the form Tidy3D's mode solver uses
    (``tidy3d/components/mode/transforms.py:radial_transform``). Measured error of the bend shift:
    **2e-4 and 7e-5** for the two polarisations at R = 5 um, which is the extrapolation floor of the
    comparison rather than a property of the form.

``"conformal"``
    Heiblum & Harris as written: the *index* alone carries the bend, ``eps -> eps (r/R)^2``, and the
    cell edges are relabelled ``u = R ln(r/R)`` (cells map onto cells, so no material is resampled;
    the radial grid simply becomes non-uniform, which the difference matrices already handle). This
    is a scalar statement and it holds for the polarisation whose electric field is **out of the
    bend plane** - there the transformation optics above reduces to exactly this. For the other
    polarisation the same substitution belongs on ``mu``, not on ``eps``, and using it on ``eps``
    is wrong by a fixed fraction: measured **1e-4** (out of plane) against **6.4e-2** (in plane),
    both radius-independent. The grid moves, so this form is differentiable in the permittivity but
    **not** in ``R``.

``"exponential"``
    The literal ``n(x) exp(x / R)`` with ``x`` the *physical* offset and the grid left alone, i.e.
    ``eps * exp(2 x / R)``. It is the form usually quoted, it agrees with the other two only to
    first order in ``x / R``, and that is not good enough for a quantity that is second order:
    measured **+54 %** (out of plane) and **+80 %** (in plane) on the bend shift, at both R = 5 um
    and R = 10 um. It is here to be compared against, not to be used.

Sign convention (stated once, because it is the part that is easy to get wrong)
    ``bend_axis`` is the transverse axis **normal to the plane in which the bend lies** - the ring's
    own symmetry axis - and the radial direction is therefore *the other* transverse axis. This is
    Tidy3D's convention (``ModeSpec.bend_axis``: "index into the two tangential axes defining the
    normal to the plane in which the bend lies"), and it is what the mode front end already hands
    to the Tidy3D backend. ``bend_radius`` is signed and is measured from the centre of the mode
    plane to the centre of curvature along the radial axis: with ``R > 0`` the radius grows with the
    radial coordinate, ``r = R + s``, and with ``R < 0`` it grows the other way. Everything below is
    written in terms of the positive ratio ``r / R = 1 + s / R``, so no case distinction is needed.

Limits, all of which matter at small ``R``
    - The map is exact; the *discretisation* is not. The equivalent index grows without bound away
      from the centre of curvature, so beyond the radiation caustic ``r_c = R n_eff / n_clad`` the
      mode is not bound. A PEC- or PMC-walled window turns that continuum into a discrete real
      spectrum: the solver returns ``Im n_eff = 0`` (up to round-off) and the leakage shows up
      instead as box modes and avoided crossings that move when the window moves. Radiation loss
      needs a PML in the mode plane, which is set aside for this track.
    - **A bent solve converges at first order in the cell size, not second.** Every component is
      scaled at the cell centre, because that is where the caller's one-value-per-cell material
      lives, while the components it feeds sit half a cell apart on the Yee grid; the mismatch is
      ``dx / (2 R)`` in the scale factor and it is odd in the radial coordinate, so it does not
      cancel. Measured against a cylindrical-Bessel solution of the same guide: the error is
      proportional to ``dx / R`` and one Richardson step in ``dx`` removes it. Sampling each
      component at its nominal Yee position instead moves the coefficient by a quarter and does not
      remove it - the base profile is cell-centred as well.
    - Material dispersion is ignored: the permittivity handed in is the one at the solve frequency.
"""

from __future__ import annotations

from typing import Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.core.jax.utils import is_jax_tracer

__all__ = [
    "BEND_FORMS",
    "DEFAULT_BEND_FORM",
    "BentCrossSection",
    "bend_radius_ratio",
    "transform_cross_section",
]

#: The three implementations of the map; see the module docstring.
BEND_FORMS = ("tensor", "conformal", "exponential")

#: The one used when a caller does not say. Exact for the vector problem, physical grid, and it
#: keeps the whole transform differentiable in the bend radius.
DEFAULT_BEND_FORM = "tensor"

BendForm = Literal["tensor", "conformal", "exponential"]


class BentCrossSection(tuple):
    """The transformed cross-section: ``(permittivity, permeability, coords)``.

    A plain 3-tuple subclass so it unpacks like one but says what it is in a traceback.

    Attributes:
        permittivity: The equivalent permittivity, ``(3, Nx, Ny)`` for the tensor form and the same
            component count as the input for the two scalar forms.
        permeability: The equivalent permeability, in the same layout.
        coords: The two cell-edge arrays in micrometres, relabelled by the conformal form and
            passed through unchanged by the other two.
    """

    __slots__ = ()

    def __new__(cls, permittivity, permeability, coords):
        return super().__new__(cls, (permittivity, permeability, coords))

    @property
    def permittivity(self):
        return self[0]

    @property
    def permeability(self):
        return self[1]

    @property
    def coords(self):
        return self[2]


def _module(*values):
    """Return ``jnp`` if any argument is a JAX value, else ``numpy``.

    The transform runs in two places: inside the front end's ``pure_callback``, where the arrays are
    numpy and must stay ``complex128`` whatever the process's ``jax_enable_x64`` setting is, and on
    the differentiable path, where the permittivity and the radius may be tracers. Dispatching on the
    values keeps one implementation for both instead of a JAX copy that silently downcasts.

    Args:
        *values: Candidate arrays or scalars.

    Returns:
        The array module to use.
    """
    for value in values:
        if isinstance(value, jax.Array) or is_jax_tracer(value):
            return jnp
    return np


def bend_radius_ratio(
    edges: np.ndarray | jax.Array,
    bend_radius: float | jax.Array,
    plane_center: float | jax.Array,
    at: Literal["cell", "conformal_cell", "lower_edge", "edges"] = "cell",
) -> jax.Array | np.ndarray:
    """The dimensionless local radius ``r / R = 1 + s / R`` along one axis.

    Args:
        edges (np.ndarray | jax.Array): ``N + 1`` cell-edge coordinates of the radial axis, in the
            same length unit as ``bend_radius`` and ``plane_center``.
        bend_radius (float | jax.Array): Signed bend radius ``R``, from the centre of the mode plane
            to the centre of curvature along this axis.
        plane_center (float | jax.Array): Coordinate of the mode plane's centre on this axis; this
            is the point where ``r = R`` and where the reported ``n_eff`` is defined.
        at (Literal["cell", "conformal_cell", "lower_edge", "edges"]): Where to evaluate it.
            ``"cell"`` is the physical cell centre, ``"conformal_cell"`` the cell centre in the
            conformal coordinate (the geometric mean of the edge ratios), ``"lower_edge"`` the
            cell's lower edge, ``"edges"`` all ``N + 1`` edges.

    Returns:
        jax.Array | np.ndarray: ``N`` ratios, or ``N + 1`` for ``at="edges"``.

    Raises:
        ValueError: If ``bend_radius`` is zero, if ``at`` is unknown, or if the window reaches the
            centre of curvature so that a ratio is not positive (checked for a concrete radius only;
            a traced one cannot be).
    """
    xp = _module(edges, bend_radius, plane_center)
    if not (isinstance(bend_radius, jax.Array) or is_jax_tracer(bend_radius)) and float(bend_radius) == 0.0:
        raise ValueError("bend_radius must be non-zero; use bend_radius=None for a straight guide")
    edge_ratio = 1.0 + (xp.asarray(edges) - plane_center) / bend_radius
    if not is_jax_tracer(edge_ratio) and bool(np.any(np.asarray(edge_ratio) <= 0.0)):
        raise ValueError(
            "the mode plane reaches the centre of curvature: the conformal map needs r/R > 0 over "
            "the whole window, so the transverse window must be narrower than |bend_radius| on the "
            "side facing the centre of curvature"
        )
    if at == "edges":
        return edge_ratio
    if at == "lower_edge":
        return edge_ratio[:-1]
    if at == "conformal_cell":
        return xp.sqrt(edge_ratio[:-1] * edge_ratio[1:])
    if at == "cell":
        return 0.5 * (edge_ratio[:-1] + edge_ratio[1:])
    raise ValueError(f"unknown sampling point {at!r}")


def _as_three_components(array, nx: int, ny: int, xp) -> jax.Array | np.ndarray:
    """Broadcast a 1- or 3-component cross-section (or a scalar) to ``(3, Nx, Ny)``."""
    if array is None:
        return xp.ones((3, nx, ny), dtype=xp.complex128)
    arr = xp.asarray(array)
    if arr.ndim == 0 or arr.size == 1:
        return xp.full((3, nx, ny), arr.reshape(()), dtype=xp.complex128)
    if arr.ndim != 3:
        raise ValueError(f"cross-section must be (Ncomp, Nx, Ny), got shape {arr.shape}")
    arr = arr.astype(xp.complex128)
    if arr.shape[0] == 1:
        return arr * xp.ones((3, nx, ny), dtype=xp.complex128)
    if arr.shape[0] == 3:
        return arr
    raise NotImplementedError(
        f"the bend transform handles isotropic and diagonally-anisotropic media (1 or 3 components), "
        f"got {arr.shape[0]}. A fully tensorial cross-section transforms with the same Jacobian but "
        f"its off-diagonal entries move too, which the diagonal mode operator cannot carry."
    )


def _scaled(array, factors, radial_axis: int, nx: int, ny: int, xp):
    """Multiply a (broadcast) ``(3, Nx, Ny)`` diagonal material by one factor per component."""
    components = _as_three_components(array, nx, ny, xp)
    shape = (nx, 1) if radial_axis == 0 else (1, ny)
    stacked = xp.stack([xp.asarray(f).reshape(shape) for f in factors], axis=0)
    return components * stacked


def transform_cross_section(
    permittivity,
    permeability,
    coords: Sequence[np.ndarray],
    bend_radius: float | jax.Array,
    bend_axis: int,
    plane_center: Sequence[float],
    form: BendForm = DEFAULT_BEND_FORM,
) -> BentCrossSection:
    """Map a bent guide onto an equivalent straight one, before any operator is assembled.

    The cross-section is in the mode backend's convention: propagation along the third (``z``)
    component, the two transverse axes first, components ordered ``(xx, yy, zz)``.

    Args:
        permittivity: Rotated permittivity cross-section, ``(1 | 3, Nx, Ny)``.
        permeability: Rotated permeability cross-section, ``(1 | 3, Nx, Ny)``, a scalar, or None.
        coords (Sequence[np.ndarray]): The two cell-edge arrays (micrometres), lengths ``Nx + 1``
            and ``Ny + 1``.
        bend_radius (float | jax.Array): Signed bend radius in micrometres, from the centre of the
            mode plane to the centre of curvature along the radial axis. May be traced.
        bend_axis (int): ``0`` or ``1``, the transverse axis **normal to the plane of the bend**;
            the radial axis is the other one.
        plane_center (Sequence[float]): Centre of the mode plane on the two transverse axes, in
            micrometres. The radial one is where ``r = R``, i.e. where ``n_eff`` is defined.
        form (BendForm): ``"tensor"``, ``"conformal"`` or ``"exponential"``; see the module
            docstring.

    Returns:
        BentCrossSection: ``(permittivity, permeability, coords)`` for the equivalent straight guide.

    Raises:
        ValueError: On an unknown ``form``, a zero radius, or a window that reaches the centre of
            curvature.
        NotImplementedError: On a fully tensorial (9-component) cross-section.
    """
    if form not in BEND_FORMS:
        raise ValueError(f"bend form must be one of {BEND_FORMS}, got {form!r}")
    if bend_axis not in (0, 1):
        raise ValueError(f"bend_axis must be 0 or 1 (the transverse axis normal to the bend), got {bend_axis}")
    radial_axis = 1 - int(bend_axis)
    xp = _module(permittivity, permeability, bend_radius)
    nx = len(coords[0]) - 1
    ny = len(coords[1]) - 1
    edges = np.asarray(coords[radial_axis], dtype=np.float64)
    center = float(plane_center[radial_axis])
    # Every component is scaled at the *cell centre*, because that is where the caller's material
    # array lives: one value per cell, reused for components whose Yee sample points are half a cell
    # apart. That mismatch is what makes a bent solve first order in the cell size rather than
    # second (see the module docstring). Sampling each component at its nominal Yee position instead
    # does not remove it - it moves it by 24 % - because the base profile is cell-centred too; only
    # a staggered material from the loader would.
    ratio = bend_radius_ratio(edges, bend_radius, center, at="conformal_cell" if form == "conformal" else "cell")

    out_coords = [np.asarray(coords[0], dtype=np.float64), np.asarray(coords[1], dtype=np.float64)]

    if form == "tensor":
        # Transformation optics: eps_t *= r/R on the two transverse components, eps_w *= R/r on the
        # propagation one. The grid does not move, so this form is differentiable in R as well.
        factors = (ratio, ratio, 1.0 / ratio)
    elif form == "conformal":
        # Heiblum & Harris: the index alone carries the bend, eps -> eps (r/R)^2, and the edges are
        # relabelled u = R ln(r/R). Cells map onto cells, so no material is resampled; the radial
        # grid becomes non-uniform, which the difference matrices already handle. The grid is built
        # outside the traced path, so this form is differentiable in the permittivity but not in R.
        factors = (ratio**2, ratio**2, ratio**2)
        if is_jax_tracer(bend_radius):
            raise ValueError(
                "the conformal form relabels the grid, and the difference matrices are built outside "
                "the traced path, so its bend radius cannot be a tracer. Use form='tensor', which "
                "leaves the grid alone and is differentiable in the radius."
            )
        radius_value = float(jax.lax.stop_gradient(bend_radius))
        edge_ratio = np.asarray(bend_radius_ratio(edges, radius_value, center, at="edges"))
        out_coords[radial_axis] = radius_value * np.log(edge_ratio)
    else:  # "exponential"
        # The literal n exp(x/R) on the physical grid, isotropic: eps * exp(2 x / R), with
        # x / R = r / R - 1, so it agrees with the other two forms only to first order.
        scale = xp.exp(2.0 * (ratio - 1.0))
        factors = (scale, scale, scale)

    eps = _scaled(permittivity, factors, radial_axis, nx, ny, xp)
    if form == "tensor":
        # Transformation optics moves the permeability the same way; the two scalar forms are index
        # transformations and leave it alone (which is what makes them polarisation-specific).
        mu = _scaled(permeability, factors, radial_axis, nx, ny, xp)
    else:
        mu = _as_three_components(permeability, nx, ny, xp)
    return BentCrossSection(eps, mu, (out_coords[0], out_coords[1]))
