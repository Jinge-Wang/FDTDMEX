"""Dispersive material models for FDTDX.

Provides a generic Auxiliary Differential Equation (ADE) dispersion
abstraction for linear materials. The concrete pole types are Lorentz,
Drude and Sellmeier (all second-order) plus Debye (first-order);
Lorentz and Drude combine freely as a "Drude-Lorentz" model.

Physics
-------
Each pole contributes a 2nd-order ODE for the normalized polarization
``p = P / eps_0`` (same units as E):

.. math::
    \\ddot{p}_p + \\gamma_p \\dot{p}_p + \\omega_{0,p}^2 p_p = K_p E

Lorentz pole (resonance :math:`\\omega_0`, damping :math:`\\gamma`,
strength :math:`\\Delta\\varepsilon`):

.. math::
    \\chi_p(\\omega) = \\frac{\\Delta\\varepsilon \\cdot \\omega_0^2}{\\omega_0^2 - \\omega^2 - i\\gamma\\omega}

Drude pole (plasma frequency :math:`\\omega_p`, damping :math:`\\gamma`;
special case of Lorentz with :math:`\\omega_0 = 0`):

.. math::
    \\chi_p(\\omega) = -\\frac{\\omega_p^2}{\\omega^2 + i\\gamma\\omega}

The unified pole parameterization stores ``(omega_0, gamma, coupling_sq)``
where ``coupling_sq`` is the effective squared coupling frequency
:math:`K = \\Delta\\varepsilon \\omega_0^2` (Lorentz) or :math:`\\omega_p^2`
(Drude), both in (rad/s)^2.

Discrete update
---------------
Central differences at integer time ``n``:

.. math::
    p_p^{n+1} = c_1 p_p^{n} + c_2 p_p^{n-1} + c_3 E^{n}

with coefficients derived from the unified pole parameters and time
step ``dt``:

.. math::
    c_1 = \\frac{2 - \\omega_0^2 \\Delta t^2}{1 + \\gamma \\Delta t / 2}, \\quad
    c_2 = -\\frac{1 - \\gamma \\Delta t / 2}{1 + \\gamma \\Delta t / 2}, \\quad
    c_3 = \\frac{K \\Delta t^2}{1 + \\gamma \\Delta t / 2}

**Forward unit-circle (Jury) stability** constrains the coefficients and is
enforced in :func:`compute_pole_coefficients_per_axis` (only on axes where the
pole actually couples, since a zero-coupling axis keeps its polarization
identically zero). The roots of :math:`z^2 - c_1 z - c_2 = 0` lie inside the
unit circle iff :math:`|c_2| < 1` *and* :math:`|c_1| < 1 - c_2`. The first
holds for every :math:`\\gamma \\Delta t > 0` (:math:`c_2 = 0` at
:math:`\\gamma \\Delta t = 2` and :math:`|c_2| \\to 1` only as
:math:`\\gamma \\Delta t \\to 0` or :math:`\\infty`), so it is not the binding
constraint. The second is algebraically equivalent to
:math:`\\omega_0 \\Delta t < 2` (independent of :math:`\\gamma`), which is
therefore the stability bound.

Anisotropic (per-axis) dispersion
---------------------------------
Every pole parameter accepts either a scalar (isotropic, applied to all
three axes) or a 3-tuple ``(x, y, z)`` giving a different value per grid
axis. This yields a diagonally anisotropic susceptibility tensor
:math:`\\chi(\\omega) = \\mathrm{diag}(\\chi_x, \\chi_y, \\chi_z)` — enough to
model uniaxial/biaxial crystals and hyperbolic media (e.g. hBN) whose
optical axes align with the grid. A pole that only acts on one axis is
expressed by zeroing its strength on the others, e.g.
``LorentzPole(resonance_frequency=w0, damping=g, delta_epsilon=(2.25, 0.0, 0.0))``:
with zero coupling the polarization on that axis stays identically zero.

Oriented (off-diagonal) dispersion
----------------------------------
A pole may additionally carry an ``orientation`` unit vector ``u``: it then
acts as a single 1D oscillator along ``u`` and contributes the coupling
tensor :math:`K\\, u u^T` — off-diagonal for non-axis-aligned directions.
This models rotated/tilted crystals and monoclinic media (shear phonon
polaritons), where each IR-active phonon oscillates along its own,
generally non-orthogonal, direction. :meth:`DispersionModel.rotated`
converts a per-axis model into oriented poles for the common case of a
crystal rotated relative to the grid. Oriented dispersion runs through the
fully anisotropic update path.

Pole hooks (non-second-order poles)
-----------------------------------
The recurrence itself is generic in ``(c1, c2, c3)``, so a pole type that is
*not* a damped harmonic oscillator only has to override two hooks:

* :meth:`Pole.recurrence_coefficients_axes` — the per-axis ``(c1, c2, c3)``,
  defaulting to the second-order formulas above, and
* :meth:`Pole.susceptibility_axes` — the analytic
  :math:`\\chi_a(\\omega)`, defaulting to the Lorentzian form.

:class:`DebyePole` (first order, :math:`\\chi = \\Delta\\varepsilon /
(1 - i\\omega\\tau)`) overrides both; :class:`SellmeierPole` is a lossless
Lorentz pole and only supplies the unified triplet.

Because the engine (and the Metal fold) threads nothing but ``c1``, ``c2`` and
``c3``, the paths that reconstruct :math:`\\chi(\\omega)` from stored
coefficients alone — :func:`susceptibility_from_coefficients` (used by the
source impedance correction and by the mode solver) and
:func:`compute_eps_spectrum_from_coefficients` — tell the two pole orders
apart by ``c2``: a second-order pole has
:math:`c_2 = -(1 - \\gamma \\Delta t / 2) / D`, which is exactly zero only at
the unphysical :math:`\\gamma \\Delta t = 2` (nudged one ulp off zero there),
so ``c2 == 0`` marks a first-order pole.

Gradients
---------
Dispersive simulations currently support only the ``checkpointed`` gradient
method; the ``reversible`` method rejects them (reversing the ADE polarization
recurrence is under active development).
"""

from __future__ import annotations

from abc import ABC
from typing import NoReturn

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.constants import c as c_light
from fdtdx.constants import eps0
from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field


def _broadcast_axis_param(value: float | tuple) -> tuple:
    """Normalize a pole parameter to a per-axis 3-tuple ``(x, y, z)``.

    Scalars are broadcast to all three axes; 3-tuples pass through unchanged.
    """
    if isinstance(value, tuple):
        if len(value) != 3:
            raise ValueError(
                f"Per-axis pole parameters must be a scalar or a 3-tuple (x, y, z), got a tuple of length {len(value)}."
            )
        return value
    return (value, value, value)


def _is_uniform(axes: tuple) -> bool:
    return bool(axes[0] == axes[1] == axes[2])


def _as_rotation_matrix(rotation: tuple) -> np.ndarray:
    """Build and validate a 3x3 rotation matrix from a nested tuple or Euler angles."""
    if isinstance(rotation, tuple) and len(rotation) == 3 and not any(isinstance(v, tuple) for v in rotation):
        alpha, beta, gamma = (float(v) for v in rotation)
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
        ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
        rz = np.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]])
        r_mat = rz @ ry @ rx
    else:
        r_mat = np.asarray(rotation, dtype=np.float64)
        if r_mat.shape != (3, 3):
            raise ValueError(
                f"rotation must be a 3x3 nested tuple or a 3-tuple of Euler angles, got shape {r_mat.shape}."
            )
    if not np.allclose(r_mat @ r_mat.T, np.eye(3), atol=1e-9) or not np.isclose(np.linalg.det(r_mat), 1.0, atol=1e-9):
        raise ValueError("rotation must be a proper rotation matrix (orthogonal with determinant +1).")
    return r_mat


def _signed_permutation(r_mat: np.ndarray, tol: float = 1e-12) -> tuple[int, int, int] | None:
    """Detect a signed axis permutation: returns ``perm`` with grid axis ``perm[a]``
    receiving crystal axis ``a``, or ``None`` if the rotation is not a permutation."""
    perm = []
    for a in range(3):
        col = r_mat[:, a]
        nonzero = np.flatnonzero(np.abs(col) > tol)
        if len(nonzero) != 1 or not np.isclose(abs(col[nonzero[0]]), 1.0, atol=tol):
            return None
        perm.append(int(nonzero[0]))
    return (perm[0], perm[1], perm[2])


def _permute_pole_axes(p: "Pole", perm: tuple[int, int, int]) -> "Pole":
    """Remap a per-axis pole's parameters under a signed axis permutation (sign is
    irrelevant: the coupling enters as ``u u^T``)."""

    def _remap(value):
        axes = _broadcast_axis_param(value)
        out = [axes[0]] * 3
        for a in range(3):
            out[perm[a]] = axes[a]
        return (out[0], out[1], out[2])

    if isinstance(p, LorentzPole):
        return LorentzPole(
            resonance_frequency=_remap(p.resonance_frequency),
            damping=_remap(p.damping),
            delta_epsilon=_remap(p.delta_epsilon),
        )
    if isinstance(p, DrudePole):
        return DrudePole(plasma_frequency=_remap(p.plasma_frequency), damping=_remap(p.damping))
    if isinstance(p, SellmeierPole):
        return SellmeierPole(B=_remap(p.B), C=_remap(p.C))
    if isinstance(p, DebyePole):
        return DebyePole(delta_epsilon=_remap(p.delta_epsilon), relaxation_time=_remap(p.relaxation_time))
    raise TypeError(
        f"Cannot rotate pole of type {type(p).__name__}; construct oriented poles directly for custom pole types."
    )


def _oriented_pole_for_axis(p: "Pole", axis: int, direction: tuple[float, float, float]) -> "Pole":
    """Extract the 1D oscillator of a per-axis pole along ``axis`` as an oriented pole."""
    if isinstance(p, LorentzPole):
        w = _broadcast_axis_param(p.resonance_frequency)
        g = _broadcast_axis_param(p.damping)
        de = _broadcast_axis_param(p.delta_epsilon)
        return LorentzPole(
            resonance_frequency=float(w[axis]),
            damping=float(g[axis]),
            delta_epsilon=float(de[axis]),
            orientation=direction,
        )
    if isinstance(p, DrudePole):
        wp = _broadcast_axis_param(p.plasma_frequency)
        g = _broadcast_axis_param(p.damping)
        return DrudePole(plasma_frequency=float(wp[axis]), damping=float(g[axis]), orientation=direction)
    if isinstance(p, SellmeierPole):
        b = _broadcast_axis_param(p.B)
        cc = _broadcast_axis_param(p.C)
        return SellmeierPole(B=float(b[axis]), C=float(cc[axis]), orientation=direction)
    if isinstance(p, DebyePole):
        de = _broadcast_axis_param(p.delta_epsilon)
        tau = _broadcast_axis_param(p.relaxation_time)
        return DebyePole(delta_epsilon=float(de[axis]), relaxation_time=float(tau[axis]), orientation=direction)
    raise TypeError(
        f"Cannot rotate pole of type {type(p).__name__}; construct oriented poles directly for custom pole types."
    )


@autoinit
class Pole(TreeClass, ABC):
    """Abstract base class for a single 2nd-order ADE pole.

    Concrete subclasses store physically-meaningful parameters
    (e.g. ``delta_epsilon`` for Lorentz, ``omega_p`` for Drude) and
    expose the unified ``(omega_0, gamma, coupling_sq)`` triplet the
    FDTD loop needs via per-axis properties. New pole types can
    subclass :class:`Pole` as long as they fit the 2nd-order
    ODE form.

    Every parameter may differ per grid axis (diagonally anisotropic
    dispersion); the canonical accessors are the ``*_axes`` properties
    returning ``(x, y, z)`` tuples. The scalar accessors (``omega_0`` etc.)
    are a convenience for isotropic poles and raise for per-axis ones.
    Alternatively a pole may carry an :attr:`orientation` unit vector,
    turning it into a single 1D oscillator along that direction
    (off-diagonal coupling tensor :math:`K\\, u u^T`).
    """

    #: Optional oscillator direction ``u`` (normalized on construction).
    #: ``None`` (default) applies the pole isotropically or per-axis. When
    #: set, the pole is a single 1D oscillator along ``u`` with coupling
    #: tensor ``K * u u^T``; all other pole parameters must be scalars.
    orientation: tuple[float, float, float] | None = frozen_field(default=None)

    @property
    def axis_parameters(self) -> tuple[tuple[str, tuple[float, float, float]], ...]:
        """The pole's defining per-axis parameters as ``(name, (x, y, z))`` pairs.

        Used by the isotropy check, the orientation validation and the rotation
        helpers so they stay generic over pole types. Second-order poles report
        the unified triplet; :class:`DebyePole` reports its own two parameters.
        """
        return (
            ("omega_0", self.omega_0_axes),
            ("gamma", self.gamma_axes),
            ("coupling_sq", self.coupling_sq_axes),
        )

    @property
    def coupling_strength_axes(self) -> tuple[float, float, float]:
        """Per-axis coupling strength, exactly zero on axes where the pole is inert.

        Only the zero/non-zero pattern is used (to skip inert axes in the
        stability check and in :meth:`DispersionModel.rotated`); the magnitude
        carries no meaning across pole types.
        """
        return self.coupling_sq_axes

    def recurrence_coefficients_axes(
        self,
        dt: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        """Per-axis discrete-time ADE recurrence coefficients ``(c1, c2, c3)``.

        The default is the second-order (damped harmonic oscillator) recurrence
        of the module docstring; pole types that are not second order override
        this hook. Each returned element is an ``(x, y, z)`` tuple.

        Args:
            dt: Simulation time step (seconds).

        Returns:
            tuple: ``(c1_axes, c2_axes, c3_axes)``.

        Raises:
            ValueError: If the pole is unstable at this time step. The message
                is phrased to read after a ``"Pole {i} ({type})"`` prefix added
                by the caller.
        """
        omega_0 = self.omega_0_axes
        gamma = self.gamma_axes
        coupling_sq = self.coupling_sq_axes
        active = self.coupling_strength_axes
        c1, c2, c3 = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        for ax in range(3):
            gamma_dt = gamma[ax] * dt
            omega0_dt = omega_0[ax] * dt
            # The stability bound only binds on axes where the pole actually
            # couples. A zero-coupling axis (e.g. a Lorentz pole with
            # delta_epsilon = 0 there, the documented way to express an absent
            # resonance) has c3 = 0, so its polarization stays identically
            # zero and its unused omega_0 / gamma are irrelevant.
            axis_active = active[ax] != 0.0
            axis_note = "" if self.is_isotropic else f" on axis {'xyz'[ax]}"
            if axis_active and omega0_dt >= 2.0:
                raise ValueError(
                    f"has omega_0 * dt = {omega0_dt:.4g} >= 2{axis_note}; "
                    "the ADE recurrence roots leave the unit circle (requires omega_0 * dt < 2, "
                    "physically omega_0 * dt << 1). Lower the resonance frequency or reduce the time step."
                )
            half_gamma_dt = 0.5 * gamma_dt
            if half_gamma_dt == 1.0:
                # At exactly gamma * dt = 2 the formula gives c2 = 0, the value
                # the coefficient-only susceptibility inversion reads as "this
                # is a first-order (Debye) pole". Step gamma down by one ulp so
                # c2 lands just below zero instead: the damping changes by a
                # relative 1e-16 (far below fp32 storage), the recurrence is
                # unchanged for every practical purpose, and the flag stays
                # unambiguous. gamma * dt = 2 means a damping time of half a
                # time step, so nothing physical rides on the difference.
                half_gamma_dt = float(np.nextafter(1.0, 0.0))
            denom = 1.0 + half_gamma_dt
            c1[ax] = (2.0 - (omega_0[ax] ** 2) * (dt**2)) / denom
            c2[ax] = -(1.0 - half_gamma_dt) / denom
            c3[ax] = (coupling_sq[ax] * dt**2) / denom
        return (c1[0], c1[1], c1[2]), (c2[0], c2[1], c2[2]), (c3[0], c3[1], c3[2])

    def susceptibility_axes(self, omega: complex | float) -> tuple[complex, complex, complex]:
        """Per-axis analytic complex susceptibility :math:`(\\chi_x, \\chi_y, \\chi_z)`.

        The default is the second-order form
        :math:`K / (\\omega_0^2 - \\omega^2 - i\\gamma\\omega)` in the
        ``exp(-i omega t)`` convention; non-second-order poles override it.

        Args:
            omega: Angular frequency (rad/s).

        Returns:
            tuple: ``(chi_x, chi_y, chi_z)``.
        """
        w = complex(omega)
        omega_0 = self.omega_0_axes
        gamma = self.gamma_axes
        coupling_sq = self.coupling_sq_axes
        out = []
        for ax in range(3):
            denom = omega_0[ax] ** 2 - w * w - 1j * gamma[ax] * w
            out.append(coupling_sq[ax] / denom)
        return (out[0], out[1], out[2])

    def _validate_orientation(self):
        """Normalize and validate :attr:`orientation`. Concrete pole classes call
        this from ``__post_init__`` (which ``autoinit`` only invokes when defined
        directly on the class, not inherited)."""
        if self.orientation is None:
            return
        vec = self.orientation
        if not isinstance(vec, tuple) or len(vec) != 3:
            raise ValueError(f"Pole orientation must be a 3-tuple (x, y, z), got {vec!r}.")
        arr = np.asarray(vec, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Pole orientation components must be finite, got {vec!r}.")
        scale = float(np.max(np.abs(arr)))
        if scale == 0.0:
            raise ValueError("Pole orientation must be a non-zero vector.")
        # scale first so the squared terms cannot overflow for large components
        scaled = arr / scale
        norm = float(np.linalg.norm(scaled))
        object.__setattr__(
            self, "orientation", (float(scaled[0]) / norm, float(scaled[1]) / norm, float(scaled[2]) / norm)
        )
        for name, axes in self.axis_parameters:
            if not _is_uniform(axes):
                raise ValueError(
                    f"Oriented poles are single 1D oscillators and require scalar parameters, "
                    f"but '{name}' differs per axis. Use one oriented pole per direction instead."
                )

    def _uniform_or_raise(self, axes: tuple, name: str) -> float:
        if not _is_uniform(axes):
            raise ValueError(
                f"{type(self).__name__} has per-axis parameters; use the per-axis "
                f"accessor '{name}_axes' instead of the scalar '{name}'."
            )
        return axes[0]

    def _no_second_order_parameters(self, name: str) -> NoReturn:
        raise NotImplementedError(
            f"{type(self).__name__} is not a second-order (damped-oscillator) pole, so it has no "
            f"'{name}'. Use recurrence_coefficients_axes(dt) for the ADE coefficients and "
            "susceptibility_axes(omega) for the analytic susceptibility."
        )

    @property
    def omega_0_axes(self) -> tuple[float, float, float]:
        """Per-axis resonance angular frequency (rad/s). Zero for pure Drude poles.

        Raises ``NotImplementedError`` for poles that are not second order
        (e.g. :class:`DebyePole`).
        """
        self._no_second_order_parameters("omega_0_axes")

    @property
    def gamma_axes(self) -> tuple[float, float, float]:
        """Per-axis damping rate (rad/s).

        Raises ``NotImplementedError`` for poles that are not second order.
        """
        self._no_second_order_parameters("gamma_axes")

    @property
    def coupling_sq_axes(self) -> tuple[float, float, float]:
        """Per-axis effective squared coupling frequency ``K`` (rad^2/s^2).

        ``delta_epsilon * omega_0**2`` for a Lorentz pole and
        ``omega_p**2`` for a Drude pole.

        This is the coefficient ``a`` of the ``E`` driving term in the unified
        2nd-order ODE ``p'' + gamma p' + omega_0**2 p = a E``.

        Raises ``NotImplementedError`` for poles that are not second order.
        """
        self._no_second_order_parameters("coupling_sq_axes")

    @property
    def is_oriented(self) -> bool:
        """Whether the pole is a 1D oscillator along an :attr:`orientation` vector."""
        return self.orientation is not None

    @property
    def is_isotropic(self) -> bool:
        """Whether the pole acts identically on the three axes (and is not oriented)."""
        return self.orientation is None and all(_is_uniform(axes) for _, axes in self.axis_parameters)

    @property
    def omega_0(self) -> float:
        """Resonance angular frequency (rad/s). Zero for pure Drude poles.

        Raises ``ValueError`` for per-axis poles; use :attr:`omega_0_axes`.
        """
        return self._uniform_or_raise(self.omega_0_axes, "omega_0")

    @property
    def gamma(self) -> float:
        """Damping rate (rad/s).

        Raises ``ValueError`` for per-axis poles; use :attr:`gamma_axes`.
        """
        return self._uniform_or_raise(self.gamma_axes, "gamma")

    @property
    def coupling_sq(self) -> float:
        """Effective squared coupling frequency ``K`` (rad^2/s^2).

        Raises ``ValueError`` for per-axis poles; use :attr:`coupling_sq_axes`.
        """
        return self._uniform_or_raise(self.coupling_sq_axes, "coupling_sq")


@autoinit
class LorentzPole(Pole):
    """Lorentz pole parameterised by its physical constants.

    The contribution to the susceptibility is

    .. math::
        \\chi(\\omega) = \\frac{\\Delta\\varepsilon \\cdot \\omega_0^2}{\\omega_0^2 - \\omega^2 - i\\gamma\\omega}.

    Each parameter is either a scalar (isotropic) or a per-axis 3-tuple
    ``(x, y, z)`` for diagonally anisotropic dispersion. An axis without a
    resonance is expressed by a zero ``delta_epsilon`` entry on that axis.
    """

    #: Resonance angular frequency (rad/s). Must be > 0.
    #: Scalar or per-axis 3-tuple.
    resonance_frequency: float | tuple[float, float, float] = frozen_field()

    #: Damping rate (rad/s). Must be >= 0. Scalar or per-axis 3-tuple.
    damping: float | tuple[float, float, float] = frozen_field()

    #: Oscillator strength (dimensionless); the zero-frequency
    #: contribution to the susceptibility. Scalar or per-axis 3-tuple.
    delta_epsilon: float | tuple[float, float, float] = frozen_field()

    def __post_init__(self):
        self._validate_orientation()

    @property
    def omega_0_axes(self) -> tuple[float, float, float]:
        w = _broadcast_axis_param(self.resonance_frequency)
        return (float(w[0]), float(w[1]), float(w[2]))

    @property
    def gamma_axes(self) -> tuple[float, float, float]:
        g = _broadcast_axis_param(self.damping)
        return (float(g[0]), float(g[1]), float(g[2]))

    @property
    def coupling_sq_axes(self) -> tuple[float, float, float]:
        w = self.omega_0_axes
        de = _broadcast_axis_param(self.delta_epsilon)
        return (float(de[0]) * w[0] ** 2, float(de[1]) * w[1] ** 2, float(de[2]) * w[2] ** 2)


@autoinit
class DrudePole(Pole):
    """Drude pole parameterised by its physical constants.

    The contribution to the susceptibility is

    .. math::
        \\chi(\\omega) = -\\frac{\\omega_p^2}{\\omega^2 + i\\gamma\\omega},

    equivalent to a Lorentz pole with ``omega_0 = 0``.

    Each parameter is either a scalar (isotropic) or a per-axis 3-tuple
    ``(x, y, z)`` for diagonally anisotropic dispersion — e.g.
    ``plasma_frequency=(wp, 0.0, 0.0)`` gives a metallic (hyperbolic)
    response only along x.
    """

    #: Plasma angular frequency (rad/s). Must be > 0.
    #: Scalar or per-axis 3-tuple.
    plasma_frequency: float | tuple[float, float, float] = frozen_field()

    #: Damping rate (rad/s). Must be >= 0. Scalar or per-axis 3-tuple.
    damping: float | tuple[float, float, float] = frozen_field()

    def __post_init__(self):
        self._validate_orientation()

    @property
    def omega_0_axes(self) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    @property
    def gamma_axes(self) -> tuple[float, float, float]:
        g = _broadcast_axis_param(self.damping)
        return (float(g[0]), float(g[1]), float(g[2]))

    @property
    def coupling_sq_axes(self) -> tuple[float, float, float]:
        wp = _broadcast_axis_param(self.plasma_frequency)
        return (float(wp[0]) ** 2, float(wp[1]) ** 2, float(wp[2]) ** 2)


@autoinit
class SellmeierPole(Pole):
    """One term of a Sellmeier equation, in the data-sheet parameterisation.

    A Sellmeier data sheet gives the refractive index as

    .. math::
        n^2(\\lambda) = 1 + \\sum_j \\frac{B_j \\lambda^2}{\\lambda^2 - C_j}

    with the vacuum wavelength :math:`\\lambda` and :math:`C_j` in the **same
    squared length unit**. This class takes ``C`` in **m^2** (SI, matching the
    rest of FDTDX); use :meth:`from_micrometres` for the µm^2 numbers printed
    on most data sheets.

    Substituting :math:`\\lambda = 2 \\pi c / \\omega` turns each term into a
    **lossless Lorentz pole**

    .. math::
        \\omega_0 = \\frac{2 \\pi c}{\\sqrt{C}}, \\qquad
        \\Delta\\varepsilon = B, \\qquad \\gamma = 0,

    so this class only supplies the unified ``(omega_0, gamma, coupling_sq)``
    triplet and every downstream path (ADE coefficients, analytic
    susceptibility, the Metal fold) is unchanged. Being lossless, a Sellmeier
    term is only valid away from the material's absorption bands — the
    wavelength range printed with the data-sheet coefficients.

    Each parameter is a scalar (isotropic) or a per-axis 3-tuple ``(x, y, z)``;
    an axis without this term is expressed by ``B = 0`` on that axis (``C``
    must still be positive there).
    """

    #: Sellmeier oscillator strength :math:`B` (dimensionless), equal to the
    #: pole's ``delta_epsilon``. Scalar or per-axis 3-tuple.
    B: float | tuple[float, float, float] = frozen_field()

    #: Sellmeier resonance wavelength squared :math:`C` in **m^2**. Must be > 0.
    #: Scalar or per-axis 3-tuple.
    C: float | tuple[float, float, float] = frozen_field()

    def __post_init__(self):
        self._validate_orientation()

    @classmethod
    def from_micrometres(
        cls,
        B: float | tuple[float, float, float],
        C_um2: float | tuple[float, float, float],
        orientation: tuple[float, float, float] | None = None,
    ) -> "SellmeierPole":
        """Build a pole from data-sheet coefficients with ``C`` in µm^2.

        Args:
            B: Sellmeier strength (dimensionless), scalar or per-axis 3-tuple.
            C_um2: Sellmeier resonance wavelength squared in µm^2, scalar or
                per-axis 3-tuple. (Data sheets that print the resonance
                *wavelength* :math:`\\sqrt{C}` in µm need the square.)
            orientation: Optional oscillator direction, see :class:`Pole`.

        Returns:
            SellmeierPole: The same term with ``C`` converted to m^2.
        """
        scale = 1e-12  # (1e-6 m)^2
        c_axes = _broadcast_axis_param(C_um2)
        c_si: float | tuple[float, float, float]
        if isinstance(C_um2, tuple):
            c_si = (float(c_axes[0]) * scale, float(c_axes[1]) * scale, float(c_axes[2]) * scale)
        else:
            c_si = float(C_um2) * scale
        return cls(B=B, C=c_si, orientation=orientation)

    @property
    def omega_0_axes(self) -> tuple[float, float, float]:
        c_axes = _broadcast_axis_param(self.C)
        out = []
        for ax in range(3):
            c_val = float(c_axes[ax])
            if c_val <= 0.0:
                raise ValueError(
                    f"SellmeierPole C must be > 0 (m^2), got {c_val!r} on axis {'xyz'[ax]}. "
                    "Use SellmeierPole.from_micrometres for data-sheet values in um^2."
                )
            out.append(2.0 * np.pi * c_light / np.sqrt(c_val))
        return (float(out[0]), float(out[1]), float(out[2]))

    @property
    def gamma_axes(self) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    @property
    def coupling_sq_axes(self) -> tuple[float, float, float]:
        w = self.omega_0_axes
        b = _broadcast_axis_param(self.B)
        return (float(b[0]) * w[0] ** 2, float(b[1]) * w[1] ** 2, float(b[2]) * w[2] ** 2)


@autoinit
class DebyePole(Pole):
    """First-order (Debye) relaxation pole.

    The contribution to the susceptibility, in the engine's
    ``exp(-i omega t)`` convention (so a passive medium has
    :math:`\\mathrm{Im}\\,\\chi > 0`), is

    .. math::
        \\chi(\\omega) = \\frac{\\Delta\\varepsilon}{1 - i \\omega \\tau},

    i.e. the first-order ODE :math:`\\tau \\dot{p} + p = \\Delta\\varepsilon E`
    for the normalized polarization. It is *not* a damped oscillator, so this
    class overrides both pole hooks — see
    :meth:`recurrence_coefficients_axes` for the discrete update and its
    accuracy, and :meth:`susceptibility_axes` for the analytic form.

    Each parameter is a scalar (isotropic) or a per-axis 3-tuple ``(x, y, z)``;
    an axis without relaxation is expressed by ``delta_epsilon = 0`` there.
    """

    #: Relaxation strength :math:`\\Delta\\varepsilon` (dimensionless); the
    #: zero-frequency contribution to the susceptibility. Scalar or per-axis
    #: 3-tuple.
    delta_epsilon: float | tuple[float, float, float] = frozen_field()

    #: Relaxation time :math:`\\tau` (seconds). Must be > 0. Scalar or
    #: per-axis 3-tuple.
    relaxation_time: float | tuple[float, float, float] = frozen_field()

    def __post_init__(self):
        self._validate_orientation()

    @property
    def tau(self) -> float:
        """Relaxation time (seconds); alias of :attr:`relaxation_time`.

        Raises ``ValueError`` for per-axis poles; use :attr:`tau_axes`.
        """
        return self._uniform_or_raise(self.tau_axes, "tau")

    @property
    def tau_axes(self) -> tuple[float, float, float]:
        """Per-axis relaxation time (seconds)."""
        t = _broadcast_axis_param(self.relaxation_time)
        return (float(t[0]), float(t[1]), float(t[2]))

    @property
    def delta_epsilon_axes(self) -> tuple[float, float, float]:
        """Per-axis relaxation strength (dimensionless)."""
        d = _broadcast_axis_param(self.delta_epsilon)
        return (float(d[0]), float(d[1]), float(d[2]))

    @property
    def axis_parameters(self) -> tuple[tuple[str, tuple[float, float, float]], ...]:
        return (("delta_epsilon", self.delta_epsilon_axes), ("relaxation_time", self.tau_axes))

    @property
    def coupling_strength_axes(self) -> tuple[float, float, float]:
        return self.delta_epsilon_axes

    def recurrence_coefficients_axes(
        self,
        dt: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        """Exact exponential recurrence for the first-order relaxation ODE.

        Integrating :math:`\\tau \\dot{p} + p = \\Delta\\varepsilon E` exactly
        across one step with ``E`` held constant gives, with
        :math:`\\alpha = \\Delta t / \\tau`,

        .. math::
            c_1 = e^{-\\alpha}, \\qquad c_2 = 0, \\qquad
            c_3 = \\Delta\\varepsilon \\, (1 - e^{-\\alpha}),

        so the recurrence roots are :math:`\\{e^{-\\alpha}, 0\\}` and the update
        is unconditionally stable — there is no ``omega_0 * dt < 2`` bound.

        **Where ``E`` is sampled.** The engine's update
        ``P[n+1] = c1 P[n] + c2 P[n-1] + c3 E[n]`` supplies ``E`` at the
        *left endpoint* of the step, while the exact exponential update wants
        the exponentially-weighted mean of ``E`` over ``[n, n+1]``, which sits
        at the midpoint ``n + 1/2`` to leading order. The realized discrete
        susceptibility is therefore
        :math:`\\chi_\\text{disc} = \\chi(\\omega) (1 + i \\omega \\Delta t / 2)
        + O((\\omega \\Delta t)^2)` — a half-step phase lag, first order in
        ``dt`` (~4.5 % at 40 cells per wavelength with a Courant-limited step).
        The midpoint-consistent coefficient would need ``E[n+1]``, i.e. an
        implicit fold into the E-update that the shared ``c1/c2/c3``
        recurrence (and its Metal counterpart) cannot express; the explicit
        alternatives that do fit the shape carry the same leading error
        (forward Euler ``c1 = 1 - alpha``, trapezoidal
        ``c1 = (2 tau - dt) / (2 tau + dt)``) or are unconditionally unstable
        (the leapfrog-centred form ``c1 = -2 alpha``, ``c2 = 1``). The exact
        exponential form is kept because it alone reproduces the physical
        decay rate for every ``dt``.

        Args:
            dt: Simulation time step (seconds).

        Returns:
            tuple: ``(c1_axes, c2_axes, c3_axes)``.

        Raises:
            ValueError: If a coupling axis has a non-positive relaxation time.
        """
        tau = self.tau_axes
        delta_eps = self.delta_epsilon_axes
        c1, c3 = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        for ax in range(3):
            axis_note = "" if self.is_isotropic else f" on axis {'xyz'[ax]}"
            if tau[ax] <= 0.0:
                raise ValueError(
                    f"has relaxation_time = {tau[ax]:.4g} <= 0{axis_note}; a Debye relaxation time "
                    "must be a positive number of seconds."
                )
            decay = float(np.exp(-dt / tau[ax]))
            c1[ax] = decay
            # -expm1(-x) == 1 - exp(-x), accurate for dt << tau
            c3[ax] = delta_eps[ax] * float(-np.expm1(-dt / tau[ax]))
        return (c1[0], c1[1], c1[2]), (0.0, 0.0, 0.0), (c3[0], c3[1], c3[2])

    def susceptibility_axes(self, omega: complex | float) -> tuple[complex, complex, complex]:
        """Analytic Debye susceptibility :math:`\\Delta\\varepsilon / (1 - i\\omega\\tau)` per axis.

        Args:
            omega: Angular frequency (rad/s).

        Returns:
            tuple: ``(chi_x, chi_y, chi_z)``.
        """
        w = complex(omega)
        tau = self.tau_axes
        delta_eps = self.delta_epsilon_axes
        out = [delta_eps[ax] / (1.0 - 1j * w * tau[ax]) for ax in range(3)]
        return (out[0], out[1], out[2])


@autoinit
class DispersionModel(TreeClass):
    """Linear susceptibility built from a sum of 2nd-order ADE poles.

    The high-frequency permittivity :math:`\\varepsilon_\\infty` is NOT
    stored here - it lives in the parent :class:`~fdtdx.materials.Material`
    as the existing ``permittivity`` field. This keeps a single source of
    truth for the ``inv_permittivities`` array.
    """

    #: Tuple of poles making up the susceptibility model.
    poles: tuple[Pole, ...] = frozen_field(default=())

    @property
    def num_poles(self) -> int:
        """Number of poles in this model."""
        return len(self.poles)

    @property
    def is_isotropic(self) -> bool:
        """Whether every pole applies the same parameters to all three axes."""
        return all(p.is_isotropic for p in self.poles)

    @property
    def has_off_diagonal_coupling(self) -> bool:
        """Whether any pole is oriented (contributing an off-diagonal coupling tensor)."""
        return any(p.is_oriented for p in self.poles)

    def susceptibility_tensor(self, omega: complex | float) -> np.ndarray:
        """Evaluate the full 3x3 complex susceptibility tensor :math:`\\chi_{ij}(\\omega)`.

        Oriented poles contribute :math:`\\chi_p(\\omega)\\, u_p u_p^T`;
        per-axis and isotropic poles contribute diagonal terms. Uses the
        ``exp(-i omega t)`` Fourier convention.

        Args:
            omega: Angular frequency (rad/s).

        Returns:
            Complex numpy array of shape ``(3, 3)``.
        """
        total = np.zeros((3, 3), dtype=np.complex128)
        for p in self.poles:
            chi = p.susceptibility_axes(omega)
            if p.is_oriented:
                assert p.orientation is not None
                u = np.asarray(p.orientation, dtype=np.float64)
                total += chi[0] * np.outer(u, u)
            else:
                for ax in range(3):
                    total[ax, ax] += chi[ax]
        return total

    def permittivity_tensor(
        self,
        omega: complex | float,
        eps_inf: float | tuple = 1.0,
    ) -> np.ndarray:
        """Full 3x3 complex relative permittivity tensor :math:`\\varepsilon_\\infty + \\chi(\\omega)`.

        Args:
            omega: Angular frequency (rad/s).
            eps_inf: High-frequency permittivity — scalar, 3-tuple (diagonal),
                flat 9-tuple or nested 3x3. Defaults to 1.0.

        Returns:
            Complex numpy array of shape ``(3, 3)``.
        """
        eps_arr = np.asarray(eps_inf, dtype=np.complex128)
        if eps_arr.ndim == 0:
            eps_mat = np.eye(3, dtype=np.complex128) * complex(eps_arr)
        elif eps_arr.shape == (3,):
            eps_mat = np.diag(eps_arr)
        elif eps_arr.shape == (9,):
            eps_mat = eps_arr.reshape(3, 3)
        elif eps_arr.shape == (3, 3):
            eps_mat = eps_arr
        else:
            raise ValueError(f"eps_inf must be a scalar, 3-tuple, flat 9-tuple or 3x3, got shape {eps_arr.shape}.")
        return eps_mat + self.susceptibility_tensor(omega)

    def rotated(self, rotation: tuple) -> "DispersionModel":
        """Return a copy of this model with the crystal axes rotated.

        Args:
            rotation: Either a 3x3 rotation matrix as a nested tuple
                ``((r11, r12, r13), ...)`` or a 3-tuple of Euler angles
                ``(alpha, beta, gamma)`` in radians, applied extrinsically as
                ``R = Rz(gamma) @ Ry(beta) @ Rx(alpha)``.

        Returns:
            DispersionModel: Isotropic poles are unchanged; oriented poles have
            their direction rotated; per-axis poles are decomposed into up to
            three oriented poles (one per axis with non-zero coupling), so the
            pole count — and with it the simulation's pole-slot memory — can
            grow. For a rotation that is a signed axis permutation (e.g. 90
            degree rotations), per-axis poles are instead remapped in place and
            keep the cheaper diagonal representation.
        """
        r_mat = _as_rotation_matrix(rotation)
        perm = _signed_permutation(r_mat)
        new_poles: list[Pole] = []
        for p in self.poles:
            if p.is_oriented:
                assert p.orientation is not None
                u = r_mat @ np.asarray(p.orientation, dtype=np.float64)
                new_poles.append(p.aset("orientation", (float(u[0]), float(u[1]), float(u[2]))))
            elif p.is_isotropic:
                new_poles.append(p)
            elif perm is not None:
                new_poles.append(_permute_pole_axes(p, perm))
            else:
                for ax in range(3):
                    if p.coupling_strength_axes[ax] == 0.0:
                        continue
                    direction = (float(r_mat[0, ax]), float(r_mat[1, ax]), float(r_mat[2, ax]))
                    new_poles.append(_oriented_pole_for_axis(p, ax, direction))
        return DispersionModel(poles=tuple(new_poles))

    def susceptibility_axes(self, omega: complex | float) -> tuple[complex, complex, complex]:
        """Evaluate the per-axis complex susceptibility :math:`(\\chi_x, \\chi_y, \\chi_z)`.

        Uses the ``exp(-i omega t)`` Fourier convention (damping appears
        with a ``-i gamma omega`` term in the Lorentz denominator). For an
        isotropic model all three entries are equal.

        Args:
            omega: Angular frequency (rad/s).

        Returns:
            tuple: :math:`\\chi_a(\\omega) = \\sum_p \\chi_{p,a}(\\omega)` for
            each axis ``a`` in ``(x, y, z)``.
        """
        if self.has_off_diagonal_coupling:
            raise ValueError(
                "DispersionModel has oriented poles; use susceptibility_tensor(omega) for the full 3x3 tensor."
            )
        totals = [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]
        for p in self.poles:
            chi = p.susceptibility_axes(omega)
            for ax in range(3):
                totals[ax] = totals[ax] + chi[ax]
        return (totals[0], totals[1], totals[2])

    def susceptibility(self, omega: complex | float) -> complex:
        """Evaluate the complex susceptibility :math:`\\chi(\\omega)`.

        Uses the ``exp(-i omega t)`` Fourier convention (damping appears
        with a ``-i gamma omega`` term in the Lorentz denominator).

        Raises ``ValueError`` for models with per-axis poles; use
        :meth:`susceptibility_axes` for those.

        Args:
            omega: Angular frequency (rad/s).

        Returns:
            complex: :math:`\\chi(\\omega) = \\sum_p \\chi_p(\\omega)`.
        """
        if not self.is_isotropic:
            raise ValueError(
                "DispersionModel has per-axis poles; use susceptibility_axes(omega) for the (x, y, z) values."
            )
        return self.susceptibility_axes(omega)[0]

    def permittivity_axes(
        self,
        omega: complex | float,
        eps_inf: float | tuple[float, float, float] = 1.0,
    ) -> tuple[complex, complex, complex]:
        """Per-axis complex relative permittivity :math:`\\varepsilon_a(\\omega) = \\varepsilon_{\\infty,a} + \\chi_a(\\omega)`.

        Args:
            omega: Angular frequency (rad/s).
            eps_inf: High-frequency permittivity — scalar or per-axis
                3-tuple (the diagonal of the ε∞ tensor). Defaults to 1.0.

        Returns:
            tuple: Relative permittivity at ``omega`` per axis ``(x, y, z)``.
        """
        chi = self.susceptibility_axes(omega)
        e = _broadcast_axis_param(eps_inf)
        return (complex(e[0]) + chi[0], complex(e[1]) + chi[1], complex(e[2]) + chi[2])

    def permittivity(self, omega: complex | float, eps_inf: float = 1.0) -> complex:
        """Complex relative permittivity :math:`\\varepsilon(\\omega) = \\varepsilon_\\infty + \\chi(\\omega)`.

        Raises ``ValueError`` for models with per-axis poles; use
        :meth:`permittivity_axes` for those.

        Args:
            omega: Angular frequency (rad/s).
            eps_inf: High-frequency permittivity. Defaults to 1.0 (vacuum).

        Returns:
            complex: Relative permittivity at ``omega``.
        """
        return eps_inf + self.susceptibility(omega)


def _pole_recurrence_or_raise(
    p: Pole,
    index: int,
    dt: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Call a pole's recurrence hook, prefixing any error with the pole's identity."""
    try:
        return p.recurrence_coefficients_axes(dt)
    except ValueError as exc:
        raise ValueError(f"Pole {index} ({type(p).__name__}) {exc}") from exc


def compute_pole_coefficients_per_axis(
    poles: tuple[Pole, ...],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the per-axis discrete-time ADE recurrence coefficients.

    For each pole and grid axis, returns ``(c1, c2, c3)`` with
    (``D = 1 + gamma dt / 2``)

    .. math::
        c_1 = \\frac{2 - \\omega_0^2 \\Delta t^2}{D}, \\quad
        c_2 = -\\frac{1 - \\gamma \\Delta t / 2}{D}, \\quad
        c_3 = \\frac{K \\Delta t^2}{D},

    where ``K = coupling_sq`` is the ``E`` coupling of the unified ODE:

    :math:`p_p^{n+1} = c_1 p_p^n + c_2 p_p^{n-1} + c_3 E^n`.

    Pole types that are not second order (e.g. :class:`DebyePole`) override
    :meth:`Pole.recurrence_coefficients_axes` and supply their own values;
    this function only dispatches to that hook.

    For isotropic poles the three axis columns are identical.

    Args:
        poles: Tuple of poles (may be empty).
        dt: Simulation time step (seconds).

    Returns:
        Three ``numpy`` arrays of shape ``(len(poles), 3)`` with ``c1``, ``c2``,
        ``c3`` per pole and axis. For an empty pole tuple, returns three
        ``(0, 3)`` arrays.
    """
    n = len(poles)
    c1 = np.zeros((n, 3), dtype=np.float64)
    c2 = np.zeros((n, 3), dtype=np.float64)
    c3 = np.zeros((n, 3), dtype=np.float64)
    for i, p in enumerate(poles):
        if p.is_oriented:
            raise ValueError(
                f"Pole {i} ({type(p).__name__}) is oriented; use compute_pole_coefficients_tensor instead."
            )
        c1[i], c2[i], c3[i] = _pole_recurrence_or_raise(p, i, dt)
    return c1, c2, c3


def compute_pole_coefficients(
    poles: tuple[Pole, ...],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the discrete-time ADE recurrence coefficients of isotropic poles.

    Scalar-per-pole variant of :func:`compute_pole_coefficients_per_axis` (see
    there for the coefficient definitions). Raises ``ValueError`` when any
    pole has per-axis parameters — use the per-axis function for those.

    Args:
        poles: Tuple of isotropic poles (may be empty).
        dt: Simulation time step (seconds).

    Returns:
        Three ``numpy`` arrays of shape ``(len(poles),)`` with ``c1``, ``c2``,
        ``c3``. For an empty pole tuple, returns three empty arrays.
    """
    for i, p in enumerate(poles):
        if not p.is_isotropic:
            raise ValueError(
                f"Pole {i} ({type(p).__name__}) has per-axis parameters or an orientation; "
                "use compute_pole_coefficients_per_axis or compute_pole_coefficients_tensor instead."
            )
    c1, c2, c3 = compute_pole_coefficients_per_axis(poles, dt)
    return c1[:, 0], c2[:, 0], c3[:, 0]


def compute_pole_coefficients_tensor(
    poles: tuple[Pole, ...],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ADE recurrence coefficients with full 3x3 coupling tensors.

    Generalizes :func:`compute_pole_coefficients_per_axis` to oriented poles:
    the recurrence coefficients ``c1``/``c2`` stay per-axis (uniform for an
    oriented pole, whose ``omega_0``/``gamma`` are scalars), while the field
    coupling ``c3`` becomes a row-major 3x3 tensor per pole —
    ``(K dt^2 / D) u u^T`` for a pole oriented along ``u``, diagonal for
    per-axis and isotropic poles.

    Args:
        poles: Tuple of poles (may be empty). Oriented poles must have a
            non-negative coupling (passivity of the ``c3 u u^T`` tensor).
        dt: Simulation time step (seconds).

    Returns:
        Three ``numpy`` arrays: ``c1``, ``c2`` of shape ``(len(poles), 3)`` and
        ``c3`` of shape ``(len(poles), 9)``.
    """
    n = len(poles)
    c1 = np.zeros((n, 3), dtype=np.float64)
    c2 = np.zeros((n, 3), dtype=np.float64)
    c3 = np.zeros((n, 9), dtype=np.float64)
    for i, p in enumerate(poles):
        # Same hook (and therefore the same stability bound and zero-coupling
        # exemption) as compute_pole_coefficients_per_axis; only the coupling
        # is re-shaped into a 3x3 tensor here.
        c1_axes, c2_axes, c3_axes = _pole_recurrence_or_raise(p, i, dt)
        c1[i] = c1_axes
        c2[i] = c2_axes
        if p.is_oriented:
            if c3_axes[0] < 0.0:
                raise ValueError(
                    f"Pole {i} ({type(p).__name__}) has negative coupling c3 = {c3_axes[0]:.4g}; "
                    "oriented poles require a non-negative coupling so the tensor c3 u u^T stays "
                    "positive semi-definite (passivity)."
                )
            assert p.orientation is not None
            u = np.asarray(p.orientation, dtype=np.float64)
            c3[i] = (c3_axes[0] * np.outer(u, u)).reshape(-1)
        else:
            for ax in range(3):
                c3[i, 4 * ax] = c3_axes[ax]
    return c1, c2, c3


def _tensor_from_components(arr: jax.Array) -> jax.Array:
    """Expand a component array ``(1|3|9, *spatial)`` to a matrix field ``(3, 3, *spatial)``."""
    if arr.shape[0] == 9:
        return arr.reshape(3, 3, *arr.shape[1:])
    diag = jnp.broadcast_to(arr, (3, *arr.shape[1:]))
    return jnp.zeros((3, 3, *arr.shape[1:]), dtype=arr.dtype).at[jnp.arange(3), jnp.arange(3)].set(diag)


def _invert_3x3_matrix_field(mat: jax.Array) -> jax.Array:
    """Per-cell inverse of a matrix field ``(3, 3, *spatial)``."""
    moved = jnp.moveaxis(mat, (0, 1), (-2, -1))
    return jnp.moveaxis(jnp.linalg.inv(moved), (-2, -1), (0, 1))


def _eps_matrix_from_inv(inv_eps: jax.Array) -> jax.Array:
    """Per-cell permittivity matrix ``(3, 3, *spatial)`` from stored inverse components."""
    if inv_eps.shape[0] == 9:
        return _invert_3x3_matrix_field(_tensor_from_components(inv_eps))
    return _tensor_from_components(1.0 / inv_eps)


def _expand_recurrence_to_coupling(c: jax.Array, coupling_components: int) -> jax.Array:
    """Expand a recurrence coefficient's component axis to match a 9-component coupling axis.

    A per-axis coefficient ``(P, 3, ...)`` becomes ``(P, 9, ...)`` where the
    row-major coupling entry ``3i+j`` uses the oscillator of row ``i``. Size-1
    axes broadcast as-is.
    """
    if coupling_components != 9 or c.shape[1] != 3:
        return c
    return jnp.repeat(c, 3, axis=1)


def _chi_per_pole_from_coefficients(xp, c1, c2, c3, omega_dt):
    """Per-pole :math:`\\chi(\\omega)` reconstructed from ADE coefficients.

    Shared by the ``jax`` (:func:`susceptibility_from_coefficients`) and
    ``numpy`` (:func:`compute_eps_spectrum_from_coefficients`) paths; ``xp`` is
    the array module. All quantities are ``dt``-normalized, so ``omega_dt`` is
    ``omega * dt`` — it may carry extra leading axes (a frequency sweep), which
    broadcast against the coefficient arrays.

    Cells with no pole (``c1 = c3 = 0``) contribute exactly zero. The pole
    *order* is read off ``c2`` (see the module docstring): ``c2 == 0`` marks a
    first-order (Debye) pole, anything else a second-order one. Coefficient
    arrays that were not produced by this module (``c2 == 0`` with
    ``c1 <= 0``) contribute zero rather than a NaN.
    """
    pole_mask = (c1 != 0.0) | (c3 != 0.0)
    first_order = pole_mask & (c2 == 0.0)
    second_order = pole_mask & ~first_order

    # --- second order: invert (c1, c2, c3) -> (gamma, omega_0^2, K) ---------
    one_minus_c2 = 1.0 - c2
    safe_one_minus_c2 = xp.where(one_minus_c2 == 0.0, 1.0, one_minus_c2)
    gamma_dt = xp.where(second_order, 2.0 * (1.0 + c2) / safe_one_minus_c2, 0.0)
    half_factor = 1.0 + 0.5 * gamma_dt
    omega0_sq_dt2 = xp.where(second_order, 2.0 - c1 * half_factor, 0.0)
    # K*dt^2 = c3*D (numerator of the Lorentzian).
    k_dt2 = xp.where(second_order, c3 * half_factor, 0.0)
    denom_2nd = omega0_sq_dt2 - omega_dt * omega_dt - 1j * gamma_dt * omega_dt
    chi_2nd = xp.where(second_order, k_dt2 / xp.where(second_order, denom_2nd, 1.0 + 0.0j), 0.0 + 0.0j)

    # --- first order: invert (c1, c3) -> (tau, delta_epsilon) ---------------
    # c1 = exp(-dt/tau) and c3 = delta_eps * (1 - c1), so
    #   alpha = dt/tau = -log(c1)  and  chi = c3/(1-c1) * alpha/(alpha - i omega dt).
    # log1p keeps the precision of alpha when c1 -> 1 (dt << tau), and the
    # alpha/(1-c1) ratio is written as one factor because both vanish together
    # there (its limit is 1).
    one_minus_c1 = 1.0 - c1
    valid_1st = first_order & (c1 > 0.0)
    alpha = -xp.log1p(xp.where(valid_1st, -one_minus_c1, 0.0))
    degenerate = one_minus_c1 == 0.0
    ratio = xp.where(degenerate, 1.0, alpha / xp.where(degenerate, 1.0, one_minus_c1))
    denom_1st = alpha - 1j * omega_dt
    take_1st = valid_1st & (denom_1st != 0.0)
    chi_1st = xp.where(take_1st, c3 * ratio / xp.where(take_1st, denom_1st, 1.0 + 0.0j), 0.0 + 0.0j)

    return chi_2nd + chi_1st


def susceptibility_from_coefficients(
    c1: jax.Array,
    c2: jax.Array,
    c3: jax.Array,
    omega: float,
    dt: float,
) -> jax.Array:
    """Evaluate the per-cell complex susceptibility :math:`\\chi(\\omega)` from
    the stored ADE recurrence coefficients.

    The coefficient arrays have shape ``(num_poles, ...)`` where the trailing
    axes are the spatial (and optional component) dimensions. The inversion
    (with ``D = 1 + \\gamma \\Delta t / 2``)

    .. math::
        \\gamma \\Delta t     &= \\frac{2 (1 + c_2)}{1 - c_2},\\\\
        \\omega_0^2 \\Delta t^2 &= 2 - c_1 D,\\\\
        K \\Delta t^2          &= c_3 D

    is applied pointwise, then each pole contributes

    .. math::
        \\chi_p(\\omega) = \\frac{K}{\\omega_0^2 - \\omega^2 - i \\gamma \\omega}

    and the result is summed over the leading pole axis. Cells where the
    coefficients are all zero (no pole) contribute exactly zero.

    Entries with ``c2 == 0`` are first-order (Debye) poles and are inverted
    instead as ``tau = -dt / log(c1)``,
    ``delta_epsilon = c3 / (1 - c1)``, contributing
    :math:`\\Delta\\varepsilon / (1 - i\\omega\\tau)`; see the module
    docstring for why ``c2`` can carry that flag.

    Args:
        c1: ADE coefficient array of shape ``(num_poles, ...)``.
        c2: ADE coefficient array of shape ``(num_poles, ...)``.
        c3: ADE coefficient array of shape ``(num_poles, ...)``.
        omega: Angular frequency (rad/s) at which to evaluate the
            susceptibility.
        dt: Simulation time step (seconds) used to derive the coefficients.

    Returns:
        Complex ``jax.Array`` with shape ``c1.shape[1:]`` — the total
        :math:`\\chi(\\omega)` summed over all poles, in every cell.
    """
    c1 = jnp.asarray(c1)
    c2 = jnp.asarray(c2)
    c3 = jnp.asarray(c3)
    if c1.ndim >= 2 and c3.ndim >= 2 and c3.shape[1] == 9:
        # 9-component coupling (oriented poles): the recurrence coefficients
        # expand so entry (i, j) uses the oscillator of row i; the result is
        # the per-entry chi_ij with shape (9, *spatial).
        c1 = _expand_recurrence_to_coupling(c1, 9)
        c2 = _expand_recurrence_to_coupling(c2, 9)

    chi_per_pole = _chi_per_pole_from_coefficients(jnp, c1, c2, c3, omega * dt)
    return jnp.sum(chi_per_pole, axis=0)


def compute_eps_spectrum_from_coefficients(
    c1: jax.Array | np.ndarray,
    c2: jax.Array | np.ndarray,
    c3: jax.Array | np.ndarray,
    inv_eps_inf: jax.Array | np.ndarray,
    omegas: np.ndarray,
    dt: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Spatially-averaged complex permittivity spectrum for a block of cells.

    For each angular frequency in ``omegas``, evaluates the per-cell
    complex permittivity :math:`\\varepsilon(\\omega) = \\varepsilon_\\infty + \\chi(\\omega)`
    where :math:`\\chi` is reconstructed from the ADE recurrence coefficients,
    and averages over the spatial axes (uniformly or with supplied weights).

    This is the broadband generalization of the single-frequency
    :func:`effective_inv_permittivity` used for carrier-frequency impedance
    matching — callers that need a frequency-dependent impedance (e.g. for
    a convolution-based broadband source correction) use this to build the
    :math:`\\varepsilon(\\omega)` spectrum that feeds
    :func:`compute_impedance_corrected_temporal_profile`.

    Args:
        c1: ADE coefficient array of shape ``(num_poles, num_components, *spatial)``
            as stored on :class:`~fdtdx.fdtd.container.ArrayContainer`, with
            ``num_components in (1, 3)`` (the material-component axis; size 3
            for per-axis anisotropic dispersion). Anisotropic components are
            averaged, mirroring the ``inv_eps_inf`` reduction.
        c2: ADE coefficient array, same shape as ``c1``.
        c3: ADE coefficient array, same shape as ``c1``.
        inv_eps_inf: Per-cell inverse of the high-frequency permittivity,
            shape ``(num_components, *spatial)`` with
            ``num_components in (1, 3, 9)``. For anisotropic tensors
            (9 components) only the diagonal entries are used.
        omegas: 1D array of angular frequencies (rad/s) to evaluate at.
        dt: Simulation time step (seconds) used to derive the coefficients.
        weights: Optional spatial weights with the same shape as the
            trailing axes of ``c1``. If ``None``, uniform averaging.

    Returns:
        Complex numpy array of shape ``(len(omegas),)`` — the volume-averaged
        :math:`\\varepsilon(\\omega)` at each requested frequency.
    """
    c1_np = np.asarray(c1)
    c2_np = np.asarray(c2)
    c3_np = np.asarray(c3)
    inv_eps_np = np.asarray(inv_eps_inf)
    omegas_np = np.asarray(omegas, dtype=np.float64)
    if c3_np.ndim >= 2 and c3_np.shape[1] == 9 and c1_np.shape[1] == 3:
        # 9-component coupling (oriented poles): entry 3i+j uses oscillator row i.
        c1_np = np.repeat(c1_np, 3, axis=1)
        c2_np = np.repeat(c2_np, 3, axis=1)

    # Reduce inv_eps_inf → scalar eps_inf per spatial cell.
    num_components = inv_eps_np.shape[0]
    if num_components == 9:
        # inv_eps_inf stores the inverse tensor and diag(eps) != 1/diag(eps^-1)
        # when off-diagonal terms exist: invert each cell's 3x3 before averaging.
        spatial = inv_eps_np.shape[1:]
        inv_mats = np.moveaxis(inv_eps_np.reshape(3, 3, -1), -1, 0)
        eps_diag_mean = np.trace(np.linalg.inv(inv_mats), axis1=-2, axis2=-1) / 3.0
        eps_inf_per_cell = eps_diag_mean.reshape(spatial)
    elif num_components in (1, 3):
        eps_inf_per_cell = np.mean(1.0 / inv_eps_np, axis=0)
    else:
        raise ValueError(f"Unexpected inv_eps_inf leading dimension {num_components}; expected 1, 3, or 9.")

    # Broadcast: omegas over (M,); coefficient arrays have shape (P, C, *spatial)
    # with C in (1, 3). Right-aligned broadcasting gives (M, P, C, *spatial).
    # The pole-parameter inversion (second and first order) is shared with
    # susceptibility_from_coefficients.
    omega_dt = (omegas_np * dt).reshape((-1,) + (1,) * c1_np.ndim)
    chi_per_pole = _chi_per_pole_from_coefficients(np, c1_np, c2_np, c3_np, omega_dt)
    chi_per_cell = chi_per_pole.sum(axis=1)  # sum over pole axis → (M, C, *spatial)
    # Average the material-component axis (identity for C = 1), mirroring the
    # eps_inf reduction above — this scalar spectrum feeds an impedance filter
    # that has no notion of polarization. For a 9-component coupling only the
    # diagonal entries carry impedance information.
    if chi_per_cell.shape[1] == 9:
        chi_per_cell = chi_per_cell[:, (0, 4, 8)].mean(axis=1)
    else:
        chi_per_cell = chi_per_cell.mean(axis=1)  # → (M, *spatial)

    eps_per_cell = eps_inf_per_cell[None, ...] + chi_per_cell  # (M, *spatial)

    if weights is None:
        flat = eps_per_cell.reshape(eps_per_cell.shape[0], -1)
        return flat.mean(axis=1)

    weights_np = np.asarray(weights, dtype=np.float64).reshape(-1)
    flat = eps_per_cell.reshape(eps_per_cell.shape[0], -1)
    weight_sum = weights_np.sum()
    if weight_sum == 0.0:
        return flat.mean(axis=1)
    return (flat * weights_np).sum(axis=1) / weight_sum


def compute_impedance_corrected_temporal_profile(
    raw_samples: np.ndarray,
    dt: float,
    eps_spectrum: np.ndarray,
    eps_center: complex,
) -> np.ndarray:
    """FIR-filter a raw source temporal profile for broadband impedance matching.

    Given the unfiltered E-side temporal profile ``s(n·dt)`` and the complex
    permittivity spectrum ``eps_spectrum = ε(ω_k)`` at the rFFT frequencies
    of a zero-padded version of ``s``, returns the H-side temporal profile
    ``s_H(n·dt)`` whose spectrum satisfies
    :math:`\\tilde{s}_H(\\omega) = \\tilde{s}(\\omega) \\cdot G(\\omega)` with

    .. math::
        G(\\omega) = \\frac{\\eta(\\omega_c)}{\\eta(\\omega)}
                   = \\sqrt{\\frac{\\varepsilon(\\omega)}{\\varepsilon(\\omega_c)}}

    (assuming a non-dispersive permeability). Injecting the prescribed E and
    H fields as ``E(x,t) = E_spatial(x)·s(t)`` and
    ``H(x,t) = (H_spatial(x)/η(ω_c))·s_H(t)`` then reproduces a physical
    plane wave at every frequency in the pulse bandwidth, not just at
    ``ω_c``. In the non-dispersive limit ``ε(ω) ≡ ε_c`` and ``G`` is the
    identity so ``s_H == s``.

    Implementation: zero-pads to ``M = 2·(len(eps_spectrum) - 1)`` for
    linear convolution, takes a real FFT, multiplies by ``G``, and transforms
    back with :func:`numpy.fft.irfft` (which enforces a real output via
    Hermitian symmetry of the positive-frequency spectrum).

    Args:
        raw_samples: Real 1-D array of the unfiltered temporal profile
            sampled at integer time steps, ``s[n] = s(n·dt)``.
        dt: Simulation time step (seconds). Present for API symmetry; the
            actual time step is encoded in ``eps_spectrum``.
        eps_spectrum: Complex 1-D array of length ``M/2 + 1`` giving
            :math:`\\varepsilon(\\omega)` at
            :math:`\\omega_k = 2\\pi \\cdot k / (M \\cdot \\Delta t)` for
            ``k = 0, ..., M/2``.
        eps_center: Scalar complex :math:`\\varepsilon(\\omega_c)` at the
            source carrier frequency.

    Returns:
        Real 1-D array of length ``len(raw_samples)`` containing ``s_H[n]``.
    """
    del dt
    raw = np.asarray(raw_samples, dtype=np.float64)
    n = raw.shape[0]
    m = (eps_spectrum.shape[0] - 1) * 2
    if m < n:
        raise ValueError(
            f"eps_spectrum of length {eps_spectrum.shape[0]} corresponds to "
            f"M={m} FFT points, which is smaller than the raw profile length {n}."
        )

    padded = np.zeros(m, dtype=np.float64)
    padded[:n] = raw
    spectrum = np.fft.rfft(padded)

    ratio = np.asarray(eps_spectrum, dtype=np.complex128) / complex(eps_center)
    filter_response = np.sqrt(ratio)
    # DC bin: eps(0) can be ill-defined for Drude poles (1/0 in the physical
    # continuum). A real s(t) has a real S(0) anyway, and a real-valued
    # correction there is enough — use G(0)=1 so the filter is the identity
    # at DC. The Nyquist bin must also be real for irfft to produce a real
    # output; take the real part to be safe.
    filter_response[0] = 1.0 + 0.0j
    filter_response[-1] = complex(np.real(filter_response[-1]), 0.0)

    filtered_spectrum = spectrum * filter_response
    filtered = np.fft.irfft(filtered_spectrum, n=m)
    return filtered[:n].astype(np.float64)


def effective_inv_permittivity(
    inv_eps: jax.Array,
    c1: jax.Array | None,
    c2: jax.Array | None,
    c3: jax.Array | None,
    omega: float,
    dt: float,
) -> jax.Array:
    """Per-cell real inverse permittivity :math:`1/\\text{Re}(\\varepsilon_\\infty + \\chi(\\omega))`.

    Sources in FDTDX use a real wave impedance, so only the real part of
    ``ε∞ + χ(ω)`` enters the injected amplitude. The imaginary part describes
    absorption, which is already handled by the ADE update loop (injecting it
    into the source amplitude would double-count).

    Cells with no pole (``c1 = c2 = c3 = 0``) contribute :math:`\\chi = 0` so
    their ``inv_eps`` is returned unchanged.

    Args:
        inv_eps: Per-cell :math:`1/\\varepsilon_\\infty` array. Typically
            has shape ``(num_components, ...)``; any shape broadcast-compatible
            with ``c1.shape[1:]`` works.
        c1: ADE coefficient array of shape ``(num_poles, ...)`` or ``None``.
        c2: ADE coefficient array of shape ``(num_poles, ...)`` or ``None``.
        c3: ADE coefficient array of shape ``(num_poles, ...)`` or ``None``.
        omega: Angular frequency (rad/s) at which to evaluate.
        dt: Simulation time step (seconds).

    Returns:
        Real-valued ``jax.Array`` with the same shape and dtype as
        ``inv_eps``. If any of ``c1``/``c2``/``c3`` is ``None``, returns
        ``inv_eps`` unchanged.
    """
    inv_eps_arr = jnp.asarray(inv_eps)
    coupling_c = jnp.asarray(c3).shape[1] if c3 is not None and jnp.asarray(c3).ndim >= 2 else 1
    if inv_eps_arr.shape[0] == 9 or coupling_c == 9:
        # Tensor path: reconstruct the real permittivity matrix per cell, add
        # the (possibly off-diagonal) susceptibility, and invert per cell.
        # Elementwise 1/inv_eps would divide by the zero off-diagonal entries.
        eps_mat = jnp.real(_eps_matrix_from_inv(inv_eps_arr))
        if c1 is not None and c2 is not None and c3 is not None:
            chi = susceptibility_from_coefficients(c1=c1, c2=c2, c3=c3, omega=omega, dt=dt)
            eps_mat = eps_mat + jnp.real(_tensor_from_components(chi))
        inv_eff = _invert_3x3_matrix_field(eps_mat)
        return inv_eff.reshape(9, *inv_eff.shape[2:]).astype(inv_eps_arr.dtype)

    if c1 is None or c2 is None or c3 is None:
        return inv_eps

    chi = susceptibility_from_coefficients(c1=c1, c2=c2, c3=c3, omega=omega, dt=dt)
    eps_inf = 1.0 / inv_eps_arr
    eps_eff = eps_inf + jnp.real(chi)
    return (1.0 / eps_eff).astype(inv_eps_arr.dtype)


def effective_complex_inv_permittivity(
    inv_eps: jax.Array,
    omega: float,
    dt: float,
    c1: jax.Array | None = None,
    c2: jax.Array | None = None,
    c3: jax.Array | None = None,
    electric_conductivity: jax.Array | None = None,
    conductivity_spacing: float | None = None,
) -> jax.Array:
    r"""Per-cell COMPLEX inverse permittivity :math:`1 / (\varepsilon_\infty + \chi(\omega) + i\sigma/(\varepsilon_0\omega))`.

    Unlike :func:`effective_inv_permittivity` — which returns the real
    ``1/Re(eps)`` for source impedance / energy normalization and deliberately
    drops the imaginary part — this keeps the *full complex* permittivity so the
    mode solver sees the material loss, yielding a complex effective index and a
    lossy mode profile. Use it ONLY for the permittivity handed to the mode
    solver, never for impedance / energy (which would double-count the
    absorption already integrated by the ADE loop and the conductivity update).

    Both loss contributions are added in the ``exp(-i omega t)`` convention
    (positive imaginary part = loss):

    * the dispersive susceptibility :math:`\chi(\omega)` reconstructed from the
      ADE coefficients (omitted when ``c1``/``c2``/``c3`` are ``None``), and
    * the conductivity loss :math:`i\,\sigma_\text{phys} / (\varepsilon_0 \omega)`,
      where :math:`\sigma_\text{phys} = \sigma_\text{array} / \Delta` recovers the
      physical S/m value from the resolution-scaled ``electric_conductivity``
      array (``conductivity_spacing`` is the scaling factor
      :math:`\Delta = c_0 \Delta t / S` applied at initialization).

    Args:
        inv_eps: Per-cell ``1/eps_inf`` (real). Shape ``(num_components, ...)``.
        omega: Angular frequency (rad/s).
        dt: Simulation time step (seconds).
        c1: ADE coefficient array of shape ``(num_poles, ...)`` or ``None``.
        c2: ADE coefficient array of shape ``(num_poles, ...)`` or ``None``.
        c3: ADE coefficient array of shape ``(num_poles, ...)`` or ``None``.
        electric_conductivity: Resolution-scaled conductivity array, or ``None``.
        conductivity_spacing: Scaling factor used to recover the physical
            conductivity. Required when ``electric_conductivity`` is given.

    Returns:
        Complex ``jax.Array`` broadcasting ``inv_eps`` against the loss terms.
    """
    inv_eps = jnp.asarray(inv_eps)
    complex_dtype = jnp.complex128 if inv_eps.dtype == jnp.float64 else jnp.complex64
    coupling_c = jnp.asarray(c3).shape[1] if c3 is not None and jnp.asarray(c3).ndim >= 2 else 1
    if inv_eps.shape[0] == 9 or coupling_c == 9:
        # Diagonal reduction for the mode solver: off-diagonal permittivity and
        # susceptibility entries are dropped, so modal geometry in monoclinic /
        # rotated media is a diagonal approximation.
        eps_mat = _eps_matrix_from_inv(inv_eps)
        eps = jnp.stack([eps_mat[0, 0], eps_mat[1, 1], eps_mat[2, 2]], axis=0).astype(complex_dtype)
        if c1 is not None and c2 is not None and c3 is not None:
            chi = susceptibility_from_coefficients(c1=c1, c2=c2, c3=c3, omega=omega, dt=dt)
            if chi.shape[0] == 9:
                chi = jnp.stack([chi[0], chi[4], chi[8]], axis=0)
            eps = eps + chi
        if electric_conductivity is not None:
            if conductivity_spacing is None:
                raise ValueError("conductivity_spacing is required when electric_conductivity is given.")
            sigma = jnp.asarray(electric_conductivity)
            if sigma.shape[0] == 9:
                sigma = jnp.stack([sigma[0], sigma[4], sigma[8]], axis=0)
            eps = eps + 1j * (sigma / conductivity_spacing) / (omega * eps0)
        return 1.0 / eps

    eps = (1.0 / inv_eps).astype(complex_dtype)
    if c1 is not None and c2 is not None and c3 is not None:
        eps = eps + susceptibility_from_coefficients(c1=c1, c2=c2, c3=c3, omega=omega, dt=dt)
    if electric_conductivity is not None:
        if conductivity_spacing is None:
            raise ValueError("conductivity_spacing is required when electric_conductivity is given.")
        sigma_phys = jnp.asarray(electric_conductivity) / conductivity_spacing
        eps = eps + 1j * sigma_phys / (omega * eps0)
    return 1.0 / eps
