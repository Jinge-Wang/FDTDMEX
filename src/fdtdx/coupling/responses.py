"""The constitutive laws: how one material's permittivity tensor answers to a sampled field.

A response is the physics of a coupling, written once and used twice — by
:mod:`fdtdx.coupling.perturb`, which evaluates it at every bulk point and on both sides of every
blended pixel, and by :mod:`fdtdx.coupling.effects`, where a case declares it. Given the material's
own tensor and the sampled values at ``K`` points it returns ``(K, 3, 3)`` perturbed tensors, and
says which points it leaves exactly alone so the identity there is bit for bit.

Three laws move the real permittivity and nothing else:

* :class:`ThermoOpticResponse`, ``n(T) = n + dn_dT (T - T_ref)`` on every principal index;
* :class:`PockelsResponse`, ``d(1/eps)_I = sum_k r_Ik E_k`` — a field tilts and stretches the index
  ellipsoid;
* :class:`PhotoelasticResponse`, ``d(1/eps)_I = sum_J p_IJ S_J`` — a strain does the same.

:class:`CompositeResponse` puts several of them on one material, applying each in turn to the
running per-point tensor. Each part is applied only where it is not the identity, so a part whose
field sits at its own null value leaves the tensor bit for bit as it was: an unstrained hot pixel is
exactly what the thermo-optic response alone would have written. The order matters in general — a
temperature acts on the index and a strain on the impermeability, and those two do not commute —
and the difference is second order in the two small parameters.

:class:`TensorConstraints` is the other half of the declaration: what a perturbed tensor must
satisfy at every voxel by the physics it models, checked before anything is written.

**Units are part of the law, not of the call site.** Each response states the unit its arithmetic is
written in (``expects_unit``) and reaches it by multiplying the samples by its own ``field_scale``.
:data:`_UNIT_FACTORS` is the table those two are checked against
(:func:`fdtdx.coupling.perturb.check_sample_units`): samples labelled ``"V/um"`` are consistent with
``field_scale=1e6`` and with nothing else. That is the check that catches the factor of 10^6 between
a field solved in micrometres and a response written in metres, which nothing downstream would flag.

A fourth changes how much the medium *absorbs* as well. :class:`LossyResponse` is
:class:`MaterialResponse` plus :meth:`~LossyResponse.conductivity`, which returns what goes into
the loader's second array, ``electric_conductivity`` in siemens per metre, at the same points;
:class:`PlasmaDispersionResponse` is the one implementation — free carriers lower silicon's index
and raise its absorption through the :class:`SorefBennett` power laws, one two-component ``(N, P)``
field driving phase and loss through one complex index. The write itself is in
:mod:`fdtdx.coupling.perturb`, which is also where the loader's no-blend rule for that array is
stated.

The sign convention, taken from the fork and not invented here
--------------------------------------------------------------
:meth:`fdtdx.materials.Material.from_complex_permittivity` splits a complex relative permittivity
``eps' + i eps''`` into the real part it stores and

.. math::  \\sigma = \\omega\\, \\varepsilon_0\\, \\varepsilon''

under the ``exp(-i omega t)`` convention, so a *positive* extinction coefficient ``kappa`` in
``n + i kappa`` is loss. :func:`fdtdx.coupling.export.complex_permittivity_slots` folds the array
back the same way (``eps + i sigma / (omega eps0)``) and so does
``fdtdx.dispersion.complex_permittivity``. This module uses exactly that map in both directions and
nothing else; :func:`sigma_from_extinction` and :func:`extinction_from_sigma` are the two halves.
The time-domain check that the sign is absorbing rather than amplifying is a measurement, not an
assertion: ``tests/unit/coupling/test_conductivity_write.py::test_conductivity_write_decays_like_the_analytic_wave``
runs the fork's own FDTD through a slab this law made lossy and fits ``exp(-alpha z)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np

from fdtdx.constants import c as C_LIGHT
from fdtdx.constants import eps0 as EPS0

#: Relative tolerance for calling a permittivity tensor isotropic.
_ISOTROPY_TOL = 1e-12

#: Sampled-field units each canonical unit accepts, as the factor that converts to it. The response
#: does its arithmetic in the canonical unit and reaches it by multiplying the samples by its own
#: ``field_scale``, so a sample labelled ``"V/um"`` is consistent only with ``field_scale=1e6``.
_UNIT_FACTORS: dict[str, dict[str, float]] = {
    "V/m": {"V/m": 1.0, "kV/m": 1e3, "MV/m": 1e6, "V/mm": 1e3, "V/cm": 1e2, "V/um": 1e6, "V/nm": 1e9},
    "K": {"K": 1.0, "degK": 1.0},
    "1": {"1": 1.0, "-": 1.0, "m/m": 1.0, "dimensionless": 1.0, "%": 1e-2, "ppm": 1e-6, "ustrain": 1e-6},
    # carrier concentrations, read by the plasma-dispersion law below
    "1/cm^3": {"1/cm^3": 1.0, "cm^-3": 1.0, "cm-3": 1.0, "1/m^3": 1e-6, "m^-3": 1e-6, "m-3": 1e-6},
}

#: Unit labels that mean "not stated"; a sample carrying one of them is not checked.
_UNLABELLED: frozenset[str] = frozenset({"", "none", "unknown"})


def _normalise_unit(unit: str) -> str:
    """Strip whitespace and fold the two micro signs onto ``u``; case is kept (mV is not MV)."""
    return str(unit).strip().replace("µ", "u").replace("μ", "u")


class MaterialResponse:
    """How one material's permittivity tensor depends on the sampled field(s).

    Subclasses set ``fields`` (the sample names they read, e.g. ``("T",)`` or ``("E",)``),
    ``expects_unit`` (the unit its arithmetic assumes *after* ``field_scale``) and implement
    :meth:`tensor` and :meth:`unchanged`.
    """

    fields: tuple[str, ...] = ()
    #: Unit of the sampled field the response's own formulas are written in, after ``field_scale``
    #: has been applied. ``None`` disables the unit check
    #: (:func:`fdtdx.coupling.perturb.check_sample_units`).
    expects_unit: ClassVar[str | None] = None

    def tensor(self, base: np.ndarray, values: Mapping[str, np.ndarray]) -> np.ndarray:
        """``(K, 3, 3)`` perturbed tensors from the material's base tensor and the sampled values.

        Args:
            base (np.ndarray): The material's unperturbed permittivity tensor, ``(3, 3)`` when
                every point shares it (what the engine passes) or ``(K, 3, 3)`` when they do not,
                which is how :class:`CompositeResponse` feeds one effect the running tensors of the
                effect before it.
            values (Mapping[str, np.ndarray]): Per field name, ``(K,)`` or ``(K, n)`` samples.

        Returns:
            np.ndarray: ``(K, 3, 3)``.
        """
        raise NotImplementedError

    def unchanged(self, values: Mapping[str, np.ndarray]) -> np.ndarray:
        """``(K,)`` bool: points where the response is exactly the identity (left bit for bit)."""
        raise NotImplementedError

    def is_isotropic_response(self) -> bool:
        """Whether an isotropic base stays isotropic under this response (enables the scalar path)."""
        return False

    def unit_requirements(self) -> tuple[tuple[str, str | None, float], ...]:
        """Per field read: ``(field name, the unit the arithmetic assumes, the scale that reaches it)``.

        One entry per name in :attr:`fields`, all with this response's own ``expects_unit`` and
        ``field_scale``. A response that reads several fields in different units (a composite of
        two effects) overrides this; :func:`fdtdx.coupling.perturb.check_sample_units` reads
        nothing else.
        """
        scale = float(getattr(self, "field_scale", 1.0))
        return tuple((name, self.expects_unit, scale) for name in self.fields)


@dataclass(frozen=True)
class ThermoOpticResponse(MaterialResponse):
    """``n(T) = n + dn_dT (T - T_ref)`` applied to every principal index of the base tensor."""

    dn_dT: float
    reference_temperature: float = 293.15
    fields: tuple[str, ...] = ("T",)
    expects_unit: ClassVar[str | None] = "K"

    def tensor(self, base: np.ndarray, values: Mapping[str, np.ndarray]) -> np.ndarray:
        dT = np.asarray(values["T"], dtype=np.float64).reshape(-1) - self.reference_temperature
        b = np.asarray(base, dtype=np.float64)
        if b.ndim == 2:
            b = b.reshape(3, 3)
            diag = np.diag(b)[None, :]
            out = np.broadcast_to(b, (dT.shape[0], 3, 3)).copy()
        else:
            b = b.reshape(-1, 3, 3)
            diag = np.diagonal(b, axis1=1, axis2=2)
            out = b.copy()
        n = np.sqrt(diag) + self.dn_dT * dT[:, None]
        for c in range(3):
            out[:, c, c] = n[:, c] ** 2
        return out

    def unchanged(self, values: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.asarray(values["T"], dtype=np.float64).reshape(-1) == self.reference_temperature

    def is_isotropic_response(self) -> bool:
        return True


@dataclass(frozen=True)
class PockelsResponse(MaterialResponse):
    """Linear electro-optic effect ``d(1/eps)_ij = sum_k r_ijk E_k`` on the base tensor.

    Attributes:
        r (Sequence[Sequence[float]]): The contracted ``(6, 3)`` electro-optic matrix ``r_{Ik}`` in
            metres per volt, rows in Voigt order ``(xx, yy, zz, yz, xz, xy)``, columns the field
            components in the material's own frame, which must coincide with the grid axes.
        field_scale (float): Multiplies the sampled field before use, so a field solved in volts per
            micrometre becomes volts per metre with ``1e6``.
    """

    r: Sequence[Sequence[float]]
    field_scale: float = 1.0
    fields: tuple[str, ...] = ("E",)
    expects_unit: ClassVar[str | None] = "V/m"

    def _matrix(self) -> np.ndarray:
        m = np.asarray(self.r, dtype=np.float64)
        if m.shape != (6, 3):
            raise ValueError(f"the contracted electro-optic matrix must be (6, 3), got {m.shape}")
        return m

    def tensor(self, base: np.ndarray, values: Mapping[str, np.ndarray]) -> np.ndarray:
        E = np.asarray(values["E"], dtype=np.float64).reshape(-1, 3) * float(self.field_scale)
        delta_voigt = E @ self._matrix().T  # (K, 6): (xx, yy, zz, yz, xz, xy)
        b = np.asarray(base, dtype=np.float64)
        inv_base = np.linalg.inv(b.reshape(3, 3))[None, :, :] if b.ndim == 2 else np.linalg.inv(b.reshape(-1, 3, 3))
        delta = np.zeros((E.shape[0], 3, 3), dtype=np.float64)
        delta[:, 0, 0] = delta_voigt[:, 0]
        delta[:, 1, 1] = delta_voigt[:, 1]
        delta[:, 2, 2] = delta_voigt[:, 2]
        delta[:, 1, 2] = delta[:, 2, 1] = delta_voigt[:, 3]
        delta[:, 0, 2] = delta[:, 2, 0] = delta_voigt[:, 4]
        delta[:, 0, 1] = delta[:, 1, 0] = delta_voigt[:, 5]
        return np.linalg.inv(inv_base + delta)

    def unchanged(self, values: Mapping[str, np.ndarray]) -> np.ndarray:
        E = np.asarray(values["E"], dtype=np.float64).reshape(-1, 3)
        return np.all(E == 0.0, axis=1)


@dataclass(frozen=True)
class PhotoelasticResponse(MaterialResponse):
    """Photoelastic effect ``d(1/eps)_I = sum_J p_IJ S_J`` (Voigt, engineering shear strains).

    Attributes:
        p (Sequence[Sequence[float]]): The ``(6, 6)`` contracted photoelastic matrix, rows and
            columns in Voigt order ``(xx, yy, zz, yz, xz, xy)``, in the material's frame, which
            must coincide with the grid axes.
        field_scale (float): Multiplies the sampled strain (unitless by default).
    """

    p: Sequence[Sequence[float]]
    field_scale: float = 1.0
    fields: tuple[str, ...] = ("S",)
    expects_unit: ClassVar[str | None] = "1"

    def _matrix(self) -> np.ndarray:
        m = np.asarray(self.p, dtype=np.float64)
        if m.shape != (6, 6):
            raise ValueError(f"the contracted photoelastic matrix must be (6, 6), got {m.shape}")
        return m

    def tensor(self, base: np.ndarray, values: Mapping[str, np.ndarray]) -> np.ndarray:
        S = np.asarray(values["S"], dtype=np.float64).reshape(-1, 6) * float(self.field_scale)
        delta_voigt = S @ self._matrix().T
        b = np.asarray(base, dtype=np.float64)
        inv_base = np.linalg.inv(b.reshape(3, 3))[None, :, :] if b.ndim == 2 else np.linalg.inv(b.reshape(-1, 3, 3))
        delta = np.zeros((S.shape[0], 3, 3), dtype=np.float64)
        delta[:, 0, 0] = delta_voigt[:, 0]
        delta[:, 1, 1] = delta_voigt[:, 1]
        delta[:, 2, 2] = delta_voigt[:, 2]
        delta[:, 1, 2] = delta[:, 2, 1] = delta_voigt[:, 3]
        delta[:, 0, 2] = delta[:, 2, 0] = delta_voigt[:, 4]
        delta[:, 0, 1] = delta[:, 1, 0] = delta_voigt[:, 5]
        return np.linalg.inv(inv_base + delta)

    def unchanged(self, values: Mapping[str, np.ndarray]) -> np.ndarray:
        S = np.asarray(values["S"], dtype=np.float64).reshape(-1, 6)
        return np.all(S == 0.0, axis=1)


@dataclass(frozen=True)
class CompositeResponse(MaterialResponse):
    """Several effects on one material at one point, applied in the order given.

    Each part is applied only where it is not the identity, so a part whose field sits at its own
    null value leaves the tensor bit for bit as it was: an unstrained hot pixel is exactly what the
    thermo-optic response alone would have written. The parts after the first are handed the
    running per-point tensors, which is why :meth:`MaterialResponse.tensor` accepts a stacked base.

    The order matters in general -- a temperature acts on the index and a strain on the
    impermeability, and those two do not commute -- and the difference is second order in the two
    small parameters.
    """

    parts: tuple[MaterialResponse, ...] = ()
    expects_unit: ClassVar[str | None] = None  # each part states its own; see unit_requirements

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError("a composite response needs at least one part")
        names: list[str] = []
        for part in self.parts:
            for name in part.fields:
                if name not in names:
                    names.append(name)
        object.__setattr__(self, "fields", tuple(names))

    def _sub(self, part: MaterialResponse, values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {name: values[name] for name in part.fields}

    def tensor(self, base: np.ndarray, values: Mapping[str, np.ndarray]) -> np.ndarray:
        stack = np.asarray(base, dtype=np.float64).reshape(-1, 3, 3)
        if stack.shape[0] != 1:
            raise ValueError("a composite response takes one base tensor per material, not a stack")
        base2d = stack[0]
        count = int(np.asarray(next(iter(values.values()))).shape[0])
        out: np.ndarray | None = None
        for part in self.parts:
            sub = self._sub(part, values)
            changed = ~np.asarray(part.unchanged(sub)).reshape(-1)
            if not changed.any():
                continue
            picked = {name: value[changed] for name, value in sub.items()}
            if out is None:
                out = np.broadcast_to(base2d, (count, 3, 3)).copy()
                out[changed] = part.tensor(base2d, picked)
            else:
                out[changed] = part.tensor(out[changed], picked)
        if out is None:
            out = np.broadcast_to(base2d, (count, 3, 3)).copy()
        return out

    def unchanged(self, values: Mapping[str, np.ndarray]) -> np.ndarray:
        out: np.ndarray | None = None
        for part in self.parts:
            flag = np.asarray(part.unchanged(self._sub(part, values))).reshape(-1)
            out = flag if out is None else (out & flag)
        assert out is not None
        return out

    def is_isotropic_response(self) -> bool:
        return all(part.is_isotropic_response() for part in self.parts)

    def unit_requirements(self) -> tuple[tuple[str, str | None, float], ...]:
        out: list[tuple[str, str | None, float]] = []
        for part in self.parts:
            out.extend(part.unit_requirements())
        return tuple(out)


@dataclass(frozen=True)
class TensorConstraints:
    """What a perturbed permittivity tensor must satisfy at every voxel, by the physics it models.

    A lossless dielectric response (thermo-optic, Pockels, photoelastic) keeps the permittivity
    real, symmetric and positive definite: real because the medium is lossless, symmetric because
    it is reciprocal, positive definite because the stored energy ``E . eps . E / 2`` is positive
    [general knowledge; Landau-Lifshitz ECM ch. 11, Yariv-Yeh ch. 4]. A lossy reciprocal medium is
    complex symmetric with a positive-semidefinite imaginary part (passivity, ``e^{-i omega t}``);
    a lossless gyrotropic (magneto-optic) medium is Hermitian with an antisymmetric imaginary part
    (the fork maps that to an antisymmetric real conductivity). The static loader carries the real
    part only, so the default here is the lossless dielectric set; the flags exist so a future
    response can relax them explicitly rather than silently.

    Attributes:
        real (bool): Imaginary parts must vanish (``|Im| <= tol * scale``).
        symmetric (bool): ``|T - T^T| <= tol * scale``.
        positive_definite (bool): Every eigenvalue of the symmetric part exceeds ``tol * scale``.
        tol (float): Relative tolerance, against the largest entry of each tensor (at least 1).
    """

    real: bool = True
    symmetric: bool = True
    positive_definite: bool = True
    tol: float = 1e-10

    def check(self, tensors: np.ndarray) -> dict[str, Any]:
        """Counts of violations over ``(K, 3, 3)`` tensors, plus the smallest eigenvalue seen."""
        t = np.asarray(tensors).reshape(-1, 3, 3)
        scale = np.maximum(np.max(np.abs(t), axis=(1, 2)), 1.0)
        out: dict[str, Any] = {"num": int(t.shape[0])}
        if self.real:
            imag = np.max(np.abs(np.imag(t)), axis=(1, 2)) if np.iscomplexobj(t) else np.zeros(t.shape[0])
            out["num_complex"] = int(np.count_nonzero(imag > self.tol * scale))
        real_part = np.real(t)
        if self.symmetric:
            asym = np.max(np.abs(real_part - np.swapaxes(real_part, -1, -2)), axis=(1, 2))
            out["num_asymmetric"] = int(np.count_nonzero(asym > self.tol * scale))
        if self.positive_definite:
            sym = 0.5 * (real_part + np.swapaxes(real_part, -1, -2))
            smallest = np.linalg.eigvalsh(sym)[:, 0]
            out["num_not_positive_definite"] = int(np.count_nonzero(smallest <= self.tol * scale))
            out["min_eigenvalue"] = float(smallest.min()) if smallest.size else None
        return out

    def raise_on(self, counts: Mapping[str, Any], where: str) -> None:
        bad = {k: v for k, v in counts.items() if k.startswith("num_") and k != "num" and v}
        if bad:
            raise ValueError(f"perturbed permittivity violates its physical constraints at {where}: {bad}")


# ------------------------------------------------------------------------------------------------
# the sigma <-> Im(eps) map, in one place
# ------------------------------------------------------------------------------------------------
def sigma_from_im_permittivity(im_eps: Any, wavelength: float) -> np.ndarray:
    """``sigma = omega eps0 Im(eps_r)`` in S/m, the fork's own split of a complex permittivity.

    Args:
        im_eps (Any): Imaginary part of the relative permittivity, positive for loss under the
            ``exp(-i omega t)`` convention the fork uses.
        wavelength (float): Free-space wavelength in metres.

    Returns:
        np.ndarray: The equivalent electric conductivity in siemens per metre.

    Raises:
        ValueError: If ``wavelength`` is not positive.
    """
    if wavelength <= 0.0:
        raise ValueError(f"wavelength must be positive, got {wavelength}")
    omega = 2.0 * math.pi * C_LIGHT / float(wavelength)
    return omega * EPS0 * np.asarray(im_eps, dtype=np.float64)


def sigma_from_extinction(n: Any, kappa: Any, wavelength: float) -> np.ndarray:
    """``sigma`` of a medium whose complex index is ``n + i kappa``: ``omega eps0 * 2 n kappa``."""
    return sigma_from_im_permittivity(
        2.0 * np.asarray(n, dtype=np.float64) * np.asarray(kappa, dtype=np.float64), wavelength
    )


def extinction_from_sigma(sigma: Any, eps_real: Any, wavelength: float) -> tuple[np.ndarray, np.ndarray]:
    """Invert the split: ``(n, kappa)`` of the medium the fork stores as ``(eps_real, sigma)``.

    ``eps = eps_real + i sigma/(omega eps0) = (n + i kappa)^2`` gives ``n^2 - kappa^2 = eps_real``
    and ``2 n kappa = Im(eps)``, whose positive-root solution is
    ``n^2 = (eps_real + sqrt(eps_real^2 + Im(eps)^2)) / 2``.

    Args:
        sigma (Any): Electric conductivity in S/m (the physical value, not the loader's scaled one).
        eps_real (Any): The real relative permittivity the loader stores.
        wavelength (float): Free-space wavelength in metres.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(n, kappa)``.
    """
    if wavelength <= 0.0:
        raise ValueError(f"wavelength must be positive, got {wavelength}")
    omega = 2.0 * math.pi * C_LIGHT / float(wavelength)
    im_eps = np.asarray(sigma, dtype=np.float64) / (omega * EPS0)
    re_eps = np.asarray(eps_real, dtype=np.float64)
    n_sq = 0.5 * (re_eps + np.hypot(re_eps, im_eps))
    n = np.sqrt(n_sq)
    kappa = np.where(n > 0.0, im_eps / (2.0 * np.where(n > 0.0, n, 1.0)), 0.0)
    return n, kappa


# ------------------------------------------------------------------------------------------------
# the Soref-Bennett power laws
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SorefBennett:
    r"""The free-carrier index and absorption power laws of silicon, as coefficients.

    .. math::

        \Delta n = a_N\, \Delta N^{p_N} + a_P\, \Delta P^{p_P}, \qquad
        \Delta \alpha = b_N\, \Delta N^{q_N} + b_P\, \Delta P^{q_P}

    with the carrier changes in cm^-3 and ``dalpha`` in cm^-1. ``dn`` is dimensionless.

    The exponents are not integers, so a *negative* carrier change makes each term complex. The
    ``branch`` attribute says what to do with it, because the two readings differ by up to 13 %:

    * ``"comsol_real"`` -- the principal branch's real part, ``Re(c d^p) = c |d|^p cos(pi p)``
      for ``d < 0``. This is literally what the model this coupling was written against evaluates:
      the expression in the registry file is wrapped in ``real(...)``, which is only meaningful
      because the argument can be negative.
    * ``"signed"`` -- ``sign(d) c |d|^p``, i.e. the odd continuation, which is the reading a
      physicist writing the law by hand would usually intend.

    Attributes:
        dn_electron (float): ``a_N``. Negative: electrons lower the index.
        dn_electron_power (float): ``p_N``.
        dn_hole (float): ``a_P``.
        dn_hole_power (float): ``p_P``.
        dalpha_electron (float): ``b_N`` in cm^-1 per (cm^-3)^``q_N``.
        dalpha_electron_power (float): ``q_N``.
        dalpha_hole (float): ``b_P``.
        dalpha_hole_power (float): ``q_P``.
        branch (str): ``"comsol_real"`` or ``"signed"``.
        source (str): Where the numbers came from; carried into a run's report.
    """

    dn_electron: float
    dn_electron_power: float
    dn_hole: float
    dn_hole_power: float
    dalpha_electron: float
    dalpha_electron_power: float
    dalpha_hole: float
    dalpha_hole_power: float
    branch: str = "comsol_real"
    source: str = ""

    def __post_init__(self) -> None:
        if self.branch not in ("comsol_real", "signed"):
            raise ValueError(f"branch must be 'comsol_real' or 'signed', got {self.branch!r}")

    def _power(self, values: np.ndarray, exponent: float) -> np.ndarray:
        """``d^p`` on the branch this object declares, for a ``d`` of either sign."""
        magnitude = np.abs(values) ** float(exponent)
        if self.branch == "signed":
            return np.sign(values) * magnitude
        return np.where(values < 0.0, math.cos(math.pi * float(exponent)) * magnitude, magnitude)

    def delta_index(self, dN: np.ndarray, dP: np.ndarray) -> np.ndarray:
        """``dn`` from carrier changes in cm^-3 (dimensionless)."""
        return self.dn_electron * self._power(dN, self.dn_electron_power) + self.dn_hole * self._power(
            dP, self.dn_hole_power
        )

    def delta_absorption(self, dN: np.ndarray, dP: np.ndarray) -> np.ndarray:
        """``dalpha`` in cm^-1 from carrier changes in cm^-3."""
        return self.dalpha_electron * self._power(dN, self.dalpha_electron_power) + self.dalpha_hole * self._power(
            dP, self.dalpha_hole_power
        )

    def delta_extinction(self, dN: np.ndarray, dP: np.ndarray, wavelength: float) -> np.ndarray:
        """``dk = lambda dalpha / (4 pi)``, with ``lambda`` in metres and ``dalpha`` in cm^-1."""
        if wavelength <= 0.0:
            raise ValueError(f"wavelength must be positive, got {wavelength}")
        wavelength_cm = 100.0 * float(wavelength)
        return wavelength_cm * self.delta_absorption(dN, dP) / (4.0 * math.pi)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dn_electron": float(self.dn_electron),
            "dn_electron_power": float(self.dn_electron_power),
            "dn_hole": float(self.dn_hole),
            "dn_hole_power": float(self.dn_hole_power),
            "dalpha_electron": float(self.dalpha_electron),
            "dalpha_electron_power": float(self.dalpha_electron_power),
            "dalpha_hole": float(self.dalpha_hole),
            "dalpha_hole_power": float(self.dalpha_hole_power),
            "branch": self.branch,
            "source": self.source,
        }


#: The 1.55 um coefficients of the COMSOL silicon-on-insulator modulator model, transcribed from
#: ``repos/kronosaiComsolTestSuite/semiconductorFEM/148411/data/148411_published_values.json``
#: key ``soref_bennett_coupling`` (itself the model PDF's p.10-p.11):
#: ``dn = real(-5.4e-22 dN^1.011 - 1.53e-18 dP^0.838)`` and
#: ``dalpha = real(8.88e-21 dN^1.167 + 5.84e-20 dP^1.109)`` in cm^-1.
SOREF_BENNETT_1550 = SorefBennett(
    dn_electron=-5.4e-22,
    dn_electron_power=1.011,
    dn_hole=-1.53e-18,
    dn_hole_power=0.838,
    dalpha_electron=8.88e-21,
    dalpha_electron_power=1.167,
    dalpha_hole=5.84e-20,
    dalpha_hole_power=1.109,
    branch="comsol_real",
    source=(
        "kronosaiComsolTestSuite/semiconductorFEM/148411/data/148411_published_values.json"
        " key 'soref_bennett_coupling' (model PDF p.10-p.11), 1.55 um"
    ),
)


# ------------------------------------------------------------------------------------------------
# the lossy response: a law that moves the conductivity as well as the tensor
# ------------------------------------------------------------------------------------------------
class LossyResponse(MaterialResponse):
    """A :class:`MaterialResponse` that also moves the conductivity.

    :meth:`MaterialResponse.tensor` keeps its meaning -- the *real*
    permittivity tensor, which is the only thing the loader's inverse-permittivity arrays can hold
    -- and :meth:`conductivity` returns what goes into ``electric_conductivity`` at the same points.
    The two must describe one complex permittivity: a subclass that moves the index without moving
    the loss simply returns the base conductivity.

    Attributes:
        allow_gain (bool): Whether a negative perturbed conductivity is meant. ``False`` on every
            response here, so a sign slip in a loss model is refused where it happens rather than
            quietly amplifying the field; a genuinely amplifying medium sets it on its own response.
    """

    allow_gain: bool = False

    def conductivity(self, base: np.ndarray, base_sigma: np.ndarray, values: Mapping[str, np.ndarray]) -> np.ndarray:
        """``(K, 3)`` perturbed electric conductivity in S/m, one entry per grid axis.

        Args:
            base (np.ndarray): The material's unperturbed *real* permittivity tensor, ``(3, 3)`` or
                ``(K, 3, 3)`` -- the same argument :meth:`tensor` gets.
            base_sigma (np.ndarray): The material's unperturbed conductivity, ``(3,)`` diagonal in
                S/m (the physical value, with the loader's resolution scaling already divided out).
            values (Mapping[str, np.ndarray]): Per field name, ``(K,)`` or ``(K, n)`` samples.

        Returns:
            np.ndarray: ``(K, 3)`` in S/m.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class PlasmaDispersionResponse(LossyResponse):
    r"""Free-carrier (plasma) dispersion: carriers lower the index and raise the absorption.

    The sampled field is one two-component block ``(N, P)`` -- the electron and hole concentration
    *change* against the reference the case differences against -- in cm^-3 after ``field_scale``.
    The coefficients (:class:`SorefBennett`) turn it into ``dn`` and ``dalpha``; ``dn`` goes into
    the permittivity and ``dalpha`` into the conductivity, through the one complex index

    .. math::  \tilde n = (n_0 + \Delta n) + i\,(\kappa_0 + \Delta\kappa),
               \qquad \Delta\kappa = \lambda\,\Delta\alpha / 4\pi

    so that ``eps' = n^2 - kappa^2`` is written to the loader's diagonal and
    ``sigma = omega eps0 * 2 n kappa`` to its conductivity. The ``kappa^2`` term is small (about
    1e-9 of the permittivity of silicon at these losses) but it is kept, because dropping it makes
    a perturbed scene differ from a scene drawn with the perturbed material in the tenth digit and
    the identity tests gate on bit equality.

    ``n_0`` is read from the base tensor the engine passes (``sqrt(eps' + kappa_0^2)`` per axis), so
    a composite that has already moved the index composes correctly. ``kappa_0`` is declared, not
    read, because :meth:`tensor` never sees the conductivity;
    :meth:`~fdtdx.coupling.effects.PlasmaDispersion.check_materials` checks the declared value
    against the scene's own material before anything is sampled.

    Attributes:
        extinction (float): ``kappa_0``, the material's unperturbed extinction coefficient at
            ``wavelength``.
        wavelength (float): Free-space wavelength in metres, the one frequency at which the
            equivalent conductivity reproduces this absorption.
        coefficients (SorefBennett): The power laws.
        field_scale (float): Multiplies the sampled ``(N, P)`` before use, so a field given in
            m^-3 becomes cm^-3 with ``1e-6``.
        allow_gain (bool): Permit a negative perturbed conductivity.
    """

    extinction: float = 0.0
    wavelength: float = 1.55e-6
    coefficients: SorefBennett = SOREF_BENNETT_1550
    field_scale: float = 1.0
    allow_gain: bool = False
    fields: tuple[str, ...] = ("C",)
    expects_unit: ClassVar[str | None] = "1/cm^3"

    def _carriers(self, values: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        block = np.asarray(values["C"], dtype=np.float64)
        if block.ndim != 2 or block.shape[1] != 2:
            raise ValueError(
                f"the carrier field must be two components (electrons, holes) per point, got shape {block.shape}"
            )
        block = block * float(self.field_scale)
        return block[:, 0], block[:, 1]

    def _index_parts(self, base: np.ndarray, values: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """``(n, kappa)``: the perturbed complex index per grid axis, ``(K, 3)`` each."""
        dN, dP = self._carriers(values)
        b = np.asarray(base, dtype=np.float64)
        diag = (
            np.broadcast_to(np.diag(b.reshape(3, 3)), (dN.shape[0], 3))
            if b.ndim == 2
            else np.diagonal(b.reshape(-1, 3, 3), axis1=1, axis2=2)
        )
        kappa0 = float(self.extinction)
        n0 = np.sqrt(diag + kappa0**2)
        dn = self.coefficients.delta_index(dN, dP)
        dk = self.coefficients.delta_extinction(dN, dP, self.wavelength)
        return n0 + dn[:, None], np.full((dN.shape[0], 3), kappa0) + dk[:, None]

    def tensor(self, base: np.ndarray, values: Mapping[str, np.ndarray]) -> np.ndarray:
        n, kappa = self._index_parts(base, values)
        b = np.asarray(base, dtype=np.float64)
        out = np.broadcast_to(b.reshape(3, 3), (n.shape[0], 3, 3)).copy() if b.ndim == 2 else b.reshape(-1, 3, 3).copy()
        for axis in range(3):
            out[:, axis, axis] = n[:, axis] ** 2 - kappa[:, axis] ** 2
        return out

    def conductivity(self, base: np.ndarray, base_sigma: np.ndarray, values: Mapping[str, np.ndarray]) -> np.ndarray:
        del base_sigma  # the declared kappa_0 carries the unperturbed loss; see check_materials
        n, kappa = self._index_parts(base, values)
        return sigma_from_extinction(n, kappa, self.wavelength)

    def unchanged(self, values: Mapping[str, np.ndarray]) -> np.ndarray:
        dN, dP = self._carriers(values)
        return (dN == 0.0) & (dP == 0.0)

    def is_isotropic_response(self) -> bool:
        return True
