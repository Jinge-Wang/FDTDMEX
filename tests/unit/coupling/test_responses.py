"""The constitutive laws that need no scene: the sigma <-> Im(eps) map and the Soref-Bennett laws.

The other responses in :mod:`fdtdx.coupling.responses` are pinned where they are used, through the
engine and the couplings (``test_perturb.py``, ``test_effects.py``, ``test_thermo_optic_pipeline.py``).
These are the ones whose whole claim is arithmetic against an outside source: the split of a complex
permittivity is the fork's own (``Material.from_refractive_index``), and the free-carrier power laws
are the COMSOL 148411 registry's, digit for digit. The write those laws feed, and the sign
convention measured against a running FDTD, are in ``test_conductivity_write.py``.
"""

import math

import numpy as np
import pytest

import fdtdx
from fdtdx.coupling import (
    SOREF_BENNETT_1550,
    LossyResponse,
    PlasmaDispersionResponse,
    SorefBennett,
    extinction_from_sigma,
    sigma_from_extinction,
)

WAVELENGTH = 1.55e-6
N_SI, K_SI = 3.4757, 3.0836e-05  # registry 148411: nSi0 and k0 = lam0 alpha0 / (4 pi)


def _deltas(electrons: float, holes: float, coefficients: SorefBennett = SOREF_BENNETT_1550):
    dN, dP = np.array([electrons]), np.array([holes])
    return (
        float(coefficients.delta_index(dN, dP)[0]),
        float(coefficients.delta_extinction(dN, dP, WAVELENGTH)[0]),
    )


# ------------------------------------------------------------------------------------------------
# the sigma <-> Im(eps) map
# ------------------------------------------------------------------------------------------------
def test_sigma_from_extinction_is_the_forks_own_split():
    """``sigma = omega eps0 2 n kappa`` must be exactly what ``Material.from_refractive_index`` does."""
    material = fdtdx.Material.from_refractive_index(complex(N_SI, K_SI), wavelength=WAVELENGTH)
    ours = float(np.asarray(sigma_from_extinction(N_SI, K_SI, WAVELENGTH)))
    assert ours == pytest.approx(material.electric_conductivity[0], rel=1e-14)
    assert material.permittivity[0] == pytest.approx(N_SI**2 - K_SI**2, rel=1e-14)


def test_extinction_from_sigma_round_trips():
    material = fdtdx.Material.from_refractive_index(complex(N_SI, K_SI), wavelength=WAVELENGTH)
    n, kappa = extinction_from_sigma(material.electric_conductivity[0], material.permittivity[0], WAVELENGTH)
    assert float(n) == pytest.approx(N_SI, rel=1e-12)
    assert float(kappa) == pytest.approx(K_SI, rel=1e-9)


# ------------------------------------------------------------------------------------------------
# the Soref-Bennett coefficients, against the registry
# ------------------------------------------------------------------------------------------------
def test_soref_bennett_coefficients_are_the_registry_numbers():
    """Digit for digit ``148411_published_values.json`` key ``soref_bennett_coupling`` (PDF p.10-11)."""
    c = SOREF_BENNETT_1550
    assert (c.dn_electron, c.dn_electron_power) == (-5.4e-22, 1.011)
    assert (c.dn_hole, c.dn_hole_power) == (-1.53e-18, 0.838)
    assert (c.dalpha_electron, c.dalpha_electron_power) == (8.88e-21, 1.167)
    assert (c.dalpha_hole, c.dalpha_hole_power) == (5.84e-20, 1.109)
    assert c.branch == "comsol_real"
    assert "148411" in c.source


def test_soref_bennett_evaluates_the_registry_expression():
    """``dn`` and ``dalpha`` at one carrier level, against the expression written out by hand."""
    dN, dP = 2.0e18, 5.0e17
    dn_hand = -5.4e-22 * dN**1.011 - 1.53e-18 * dP**0.838
    da_hand = 8.88e-21 * dN**1.167 + 5.84e-20 * dP**1.109
    assert float(SOREF_BENNETT_1550.delta_index(np.array([dN]), np.array([dP]))[0]) == pytest.approx(dn_hand, rel=1e-13)
    assert float(SOREF_BENNETT_1550.delta_absorption(np.array([dN]), np.array([dP]))[0]) == pytest.approx(
        da_hand, rel=1e-13
    )
    # dk = lambda dalpha / (4 pi), with lambda in the same length unit as 1/alpha (cm here).
    dk = float(SOREF_BENNETT_1550.delta_extinction(np.array([dN]), np.array([dP]), WAVELENGTH)[0])
    assert dk == pytest.approx(100.0 * WAVELENGTH * da_hand / (4.0 * math.pi), rel=1e-13)


def test_carriers_lower_the_index_and_raise_the_absorption():
    dn, dk = _deltas(2.0e18, 5.0e17)
    assert dn < 0.0
    assert dk > 0.0


def test_negative_carrier_change_follows_the_declared_branch():
    """The two readings of ``d^p`` for ``d < 0`` differ by ``cos(pi p)``; both are available."""
    dP = -5.0e17
    comsol = float(SOREF_BENNETT_1550.delta_index(np.array([0.0]), np.array([dP]))[0])
    signed = float(
        SorefBennett(
            **{**{k: v for k, v in SOREF_BENNETT_1550.as_dict().items() if k != "branch"}, "branch": "signed"}
        ).delta_index(np.array([0.0]), np.array([dP]))[0]
    )
    # signed gives -|d|^p, the principal branch's real part gives cos(pi p) |d|^p, so the ratio is
    # -cos(pi p) = 0.8729 at p = 0.838: the two readings of the same law differ by 13 % here.
    assert comsol == pytest.approx(-math.cos(math.pi * 0.838) * signed, rel=1e-13)
    assert comsol / signed == pytest.approx(0.8729, abs=1e-3)
    assert comsol > 0.0 and signed > 0.0, "removing holes must raise the index on either reading"


def test_branch_name_is_checked():
    with pytest.raises(ValueError, match="branch must be"):
        SorefBennett(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, branch="principal")


# ------------------------------------------------------------------------------------------------
# the response protocol
# ------------------------------------------------------------------------------------------------
def test_the_response_is_a_lossy_response():
    response = PlasmaDispersionResponse(extinction=K_SI, wavelength=WAVELENGTH)
    assert isinstance(response, LossyResponse)
    assert response.fields == ("C",)
    assert response.expects_unit == "1/cm^3"
