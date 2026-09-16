"""Absorbed-power density from a frequency-domain field and its per-slot complex permittivity.

No electromagnetic engine is involved: the fields are written down and the answer is computed by
hand, so a change of convention (which slot carries which component, which sign of the imaginary
permittivity means loss, the factor one half) fails immediately rather than inside a physics run.
The checks pin, in order: the hand-computed value on a synthetic field; per-slot independence; the
sign convention; that a lossy cell overlapping a PML raises rather than returning a number that was
measured 60-76 % wrong; that the density is zeroed inside the layer; the discrete integral; and the
flux normalisation.
"""

import numpy as np
import pytest

from fdtdx.coupling.heat import (
    absorbed_power_density,
    assert_no_lossy_pml_overlap,
    normalise_to_flux,
    total_absorbed_power,
)

OMEGA = 3.0
EPS0 = 1.0  # nondimensional scene, so the hand computation stays readable


def _field_and_eps():
    """Two cells, every slot carrying a different complex amplitude and a different loss."""
    E = np.zeros((3, 2, 1, 1), dtype=np.complex128)
    E[0, 0, 0, 0] = 2.0 + 0.0j  # |E|^2 = 4
    E[1, 0, 0, 0] = 0.0 + 3.0j  # |E|^2 = 9
    E[2, 0, 0, 0] = 1.0 + 1.0j  # |E|^2 = 2
    E[0, 1, 0, 0] = 0.5 - 0.5j  # |E|^2 = 0.5
    eps = np.zeros((3, 2, 1, 1), dtype=np.complex128)
    eps[0, 0, 0, 0] = 12.0 + 0.25j
    eps[1, 0, 0, 0] = 2.0 + 0.5j
    eps[2, 0, 0, 0] = 4.0 + 0.0j  # lossless slot: contributes nothing
    eps[0, 1, 0, 0] = 1.0 + 2.0j
    return E, eps


def test_the_density_matches_the_hand_computed_value():
    E, eps = _field_and_eps()
    q = absorbed_power_density(E, eps, omega=OMEGA, eps0=EPS0)
    assert q.shape == (2, 1, 1)
    # cell 0: 1/2 * 3 * (0.25*4 + 0.5*9 + 0.0*2) = 1.5 * 5.5 = 8.25
    # cell 1: 1/2 * 3 * (2.0*0.5)               = 1.5 * 1.0 = 1.5
    np.testing.assert_allclose(q[:, 0, 0], [8.25, 1.5], rtol=0.0, atol=1e-15)


def test_each_slot_uses_its_own_permittivity_and_its_own_field():
    E, eps = _field_and_eps()
    base = absorbed_power_density(E, eps, omega=OMEGA, eps0=EPS0)
    # doubling one slot's loss changes only that slot's contribution
    bumped = eps.copy()
    bumped[1, 0, 0, 0] = 2.0 + 1.0j
    got = absorbed_power_density(E, bumped, omega=OMEGA, eps0=EPS0)
    assert got[0, 0, 0] == pytest.approx(base[0, 0, 0] + 0.5 * OMEGA * 0.5 * 9.0)
    assert got[1, 0, 0] == pytest.approx(base[1, 0, 0])
    # and moving a field amplitude between slots changes the answer, so the slots are not summed
    swapped = E.copy()
    swapped[0, 0, 0, 0], swapped[1, 0, 0, 0] = E[1, 0, 0, 0], E[0, 0, 0, 0]
    assert absorbed_power_density(swapped, eps, omega=OMEGA, eps0=EPS0)[0, 0, 0] != pytest.approx(base[0, 0, 0])


def test_a_real_permittivity_absorbs_nothing_and_the_density_is_linear_in_omega():
    E, eps = _field_and_eps()
    lossless = eps.real.astype(np.complex128)
    np.testing.assert_allclose(absorbed_power_density(E, lossless, omega=OMEGA, eps0=EPS0), 0.0, atol=0.0)
    a = absorbed_power_density(E, eps, omega=OMEGA, eps0=EPS0)
    b = absorbed_power_density(E, eps, omega=2.0 * OMEGA, eps0=EPS0)
    np.testing.assert_allclose(b, 2.0 * a, rtol=1e-15)


def test_the_time_convention_flips_the_sign_of_the_loss():
    E, eps = _field_and_eps()
    minus = absorbed_power_density(E, eps, omega=OMEGA, eps0=EPS0, convention="exp(-iwt)")
    plus = absorbed_power_density(E, eps, omega=OMEGA, eps0=EPS0, convention="exp(+iwt)")
    np.testing.assert_allclose(plus, -minus, rtol=1e-15)
    with pytest.raises(ValueError, match="convention must be one of"):
        absorbed_power_density(E, eps, omega=OMEGA, convention="exp(iwt)")
    with pytest.raises(ValueError, match="omega must be positive"):
        absorbed_power_density(E, eps, omega=0.0)


def test_the_slots_can_be_given_as_a_mapping_or_a_stack():
    E, eps = _field_and_eps()
    as_map = {"E0": E[0], "E1": E[1], "E2": E[2]}
    eps_map = {"E0": eps[0], "E1": eps[1], "E2": eps[2]}
    np.testing.assert_allclose(
        absorbed_power_density(as_map, eps_map, omega=OMEGA, eps0=EPS0),
        absorbed_power_density(E, eps, omega=OMEGA, eps0=EPS0),
        rtol=0.0,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="missing the slots"):
        absorbed_power_density({"E0": E[0]}, eps_map, omega=OMEGA)
    with pytest.raises(ValueError, match=r"must have shape \(3, Nx, Ny, Nz\)"):
        absorbed_power_density(E[0], eps, omega=OMEGA)
    with pytest.raises(ValueError, match="they must match"):
        absorbed_power_density(E, eps[:, :1], omega=OMEGA)


# ---------------------------------------------------------------------------
# The PML overlap, which is an error and not a caveat (W1v: 60-76 %)
# ---------------------------------------------------------------------------


def test_a_lossy_cell_inside_the_pml_raises():
    E, eps = _field_and_eps()
    mask = np.zeros((2, 1, 1), dtype=bool)
    mask[1, 0, 0] = True  # the second cell, which carries loss on slot E0
    with pytest.raises(ValueError, match="lossy cells lie inside the PML"):
        absorbed_power_density(E, eps, omega=OMEGA, pml_mask=mask, eps0=EPS0)
    with pytest.raises(ValueError, match=r"slots \['E0'\]"):
        assert_no_lossy_pml_overlap(eps, mask)


def test_a_lossless_pml_cell_is_allowed_and_the_density_is_zeroed_there():
    E, eps = _field_and_eps()
    clean = eps.copy()
    clean[:, 1, 0, 0] = 1.0 + 0.0j  # vacuum inside the layer
    mask = np.zeros((2, 1, 1), dtype=bool)
    mask[1, 0, 0] = True
    q = absorbed_power_density(E, clean, omega=OMEGA, pml_mask=mask, eps0=EPS0)
    assert q[1, 0, 0] == 0.0
    assert q[0, 0, 0] == pytest.approx(8.25)


def test_the_overlap_check_reports_where_and_refuses_a_mismatched_mask():
    _, eps = _field_and_eps()
    ok = np.zeros((2, 1, 1), dtype=bool)
    assert_no_lossy_pml_overlap(eps, ok)  # no overlap: silent
    with pytest.raises(ValueError, match="pml_mask must be"):
        assert_no_lossy_pml_overlap(eps, np.zeros((3, 1, 1), dtype=bool))
    lossy_everywhere = np.ones((2, 1, 1), dtype=bool)
    with pytest.raises(ValueError, match=r"first at index \(0, 0, 0\)"):
        assert_no_lossy_pml_overlap(eps, lossy_everywhere)
    # a tolerance lets a numerically-zero imaginary part through
    tiny = eps.copy()
    tiny.imag[:] = 1e-18
    assert_no_lossy_pml_overlap(tiny, lossy_everywhere, tol=1e-15)


# ---------------------------------------------------------------------------
# The integral and the normalisation
# ---------------------------------------------------------------------------


def test_the_discrete_integral_takes_a_scalar_or_a_per_cell_volume():
    E, eps = _field_and_eps()
    q = absorbed_power_density(E, eps, omega=OMEGA, eps0=EPS0)
    assert total_absorbed_power(q, 0.5) == pytest.approx(0.5 * (8.25 + 1.5))
    per_cell = np.array([[[0.5]], [[2.0]]])
    assert total_absorbed_power(q, per_cell) == pytest.approx(0.5 * 8.25 + 2.0 * 1.5)
    with pytest.raises(ValueError, match="cell_volume must be"):
        total_absorbed_power(q, np.ones((5, 5)))


def test_the_flux_normalisation_scales_without_a_written_unit_conversion():
    q = np.array([[[2.0]], [[4.0]]])
    scaled = normalise_to_flux(q, power_nondimensional=8.0, power_physical=2.0e-4)
    np.testing.assert_allclose(scaled, q * 2.5e-5, rtol=1e-15)
    with pytest.raises(ValueError, match="nothing to normalise"):
        normalise_to_flux(q, 0.0, 1.0)


def test_the_density_runs_unchanged_on_a_jax_array():
    """Array-library-agnostic: only abs, multiply and a reduction, so a traced array works too."""
    jnp = pytest.importorskip("jax.numpy")
    E, eps = _field_and_eps()
    q = absorbed_power_density(jnp.asarray(E), jnp.asarray(eps), omega=OMEGA, eps0=EPS0)
    np.testing.assert_allclose(np.asarray(q)[:, 0, 0], [8.25, 1.5], rtol=1e-6)
