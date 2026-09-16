"""Figures of merit on synthetic inputs, graded against the numbers printed in the source paper.

The reference values come from Jokisch, Christiansen & Sigmund, JOSA B 41(2), A18 (2024): Table 4
(``Delta n_eff = 1.5e-3`` at ``P_in = 20 mW`` over ``L = 200 um``, the phase per length they quote as
``2e-3 pi/um``), Table 5 (``alpha = 1.2227e-4`` for the unheated 4 um reference device) and Table 3
(the continuation schedule). Their heat source is ``P_in`` spread over the heater volume, which for
the 4 um device is ``4 x 0.25 x 200 um^3``.
"""

import math

import numpy as np
import pytest

from fdtdx.coupling import (
    continuation_schedule,
    delta_neff_from_phase,
    heater_power,
    im_neff_from_loss_db_per_cm,
    intensity_fom,
    loss_db_per_cm,
    phase_per_power,
    phase_shift,
    pi_power,
    pi_power_length_product,
    volumetric_heat_rate,
)

LAM = 1.55e-6
LENGTH = 200e-6
DELTA_NEFF = 1.5e-3
POWER = 20e-3


# ------------------------------------------------------------------------------------------------
# phase
# ------------------------------------------------------------------------------------------------


def test_phase_shift_reproduces_the_papers_phase_per_length():
    phase = phase_shift(DELTA_NEFF, LENGTH, LAM)
    assert phase == pytest.approx(1.2162, rel=1e-4)
    # the paper quotes 2e-3 pi per micrometre for both optimised devices
    assert phase / math.pi / (LENGTH * 1e6) == pytest.approx(1.94e-3, rel=1e-2)


def test_phase_and_index_round_trip():
    phase = phase_shift(DELTA_NEFF, LENGTH, LAM)
    assert delta_neff_from_phase(phase, LENGTH, LAM) == pytest.approx(DELTA_NEFF, rel=1e-12)


def test_phase_shift_is_elementwise_on_arrays():
    values = np.array([0.0, 1e-3, 1.5e-3])
    got = phase_shift(values, LENGTH, LAM)
    np.testing.assert_allclose(got, 2 * np.pi * LENGTH * values / LAM, rtol=1e-12)


def test_pi_power_of_the_reference_device():
    assert phase_per_power(DELTA_NEFF, LENGTH, LAM, POWER) == pytest.approx(60.81, rel=1e-3)
    assert pi_power(DELTA_NEFF, LENGTH, LAM, POWER) == pytest.approx(51.66e-3, rel=1e-3)
    # P_pi L in watt-metres; 10.3 mW mm in the units the report uses
    assert pi_power_length_product(DELTA_NEFF, LENGTH, LAM, POWER) == pytest.approx(10.33e-6, rel=1e-3)


def test_pi_power_is_length_independent_at_fixed_index_change_per_watt():
    long_device = pi_power_length_product(DELTA_NEFF, LENGTH, LAM, POWER)
    short_device = pi_power_length_product(DELTA_NEFF, LENGTH / 4, LAM, POWER)
    assert long_device == pytest.approx(short_device, rel=1e-12)


def test_phase_helpers_refuse_non_physical_arguments():
    with pytest.raises(ValueError, match="positive"):
        phase_shift(1e-3, 0.0, LAM)
    with pytest.raises(ValueError, match="positive"):
        phase_shift(1e-3, LENGTH, -LAM)
    with pytest.raises(ValueError, match="power"):
        phase_per_power(1e-3, LENGTH, LAM, 0.0)


# ------------------------------------------------------------------------------------------------
# loss
# ------------------------------------------------------------------------------------------------


def test_loss_matches_the_papers_equation_8_on_its_table_5_attenuation_index():
    assert loss_db_per_cm(1.2227e-4, LAM) == pytest.approx(43.05, rel=1e-3)
    # their own Eq. 8, written out
    expected = -20 * math.log10(math.exp(-2 * math.pi * 1.2227e-4 / (LAM * 100)))
    assert loss_db_per_cm(1.2227e-4, LAM) == pytest.approx(expected, rel=1e-12)


def test_loss_is_signed_so_a_convention_slip_is_visible():
    assert loss_db_per_cm(-1.2227e-4, LAM) == pytest.approx(-43.05, rel=1e-3)


def test_loss_round_trips_through_the_attenuation_index():
    assert im_neff_from_loss_db_per_cm(loss_db_per_cm(3.3e-5, LAM), LAM) == pytest.approx(3.3e-5, rel=1e-12)
    assert im_neff_from_loss_db_per_cm(44.0, LAM) == pytest.approx(1.2497e-4, rel=1e-3)


def test_loss_refuses_a_non_physical_wavelength():
    with pytest.raises(ValueError, match="positive"):
        loss_db_per_cm(1e-4, 0.0)
    with pytest.raises(ValueError, match="positive"):
        im_neff_from_loss_db_per_cm(44.0, -1.0)


# ------------------------------------------------------------------------------------------------
# the driven-solve objective
# ------------------------------------------------------------------------------------------------


# intensity_fom runs on jax.numpy, so without jax_enable_x64 it is float32: 1e-6 is its precision.
def test_intensity_fom_is_the_log_of_the_integrated_intensity():
    field = np.zeros((3, 4, 5), dtype=np.complex128)
    field[0] = 2.0
    field[1] = 1j * 1.0
    total = (4.0 + 1.0) * 20
    assert float(intensity_fom(field)) == pytest.approx(math.log10(total), rel=1e-6)


def test_intensity_fom_honours_the_region_mask_and_the_cell_areas():
    field = np.ones((3, 4, 5), dtype=np.complex128)
    mask = np.zeros((4, 5))
    mask[1:3] = 1.0
    assert float(intensity_fom(field, mask=mask)) == pytest.approx(math.log10(3 * 10), rel=1e-6)
    areas = np.full((4, 5), 0.25)
    assert float(intensity_fom(field, mask=mask, area_weights=areas)) == pytest.approx(math.log10(7.5), rel=1e-6)


def test_intensity_fom_accepts_a_single_component_and_does_not_return_minus_infinity():
    single = np.full((4, 5), 0.5 + 0.5j)
    assert float(intensity_fom(single)) == pytest.approx(math.log10(0.5 * 20), rel=1e-6)
    assert np.isfinite(float(intensity_fom(np.zeros((3, 4, 5)))))


def test_intensity_fom_is_differentiable():
    import jax
    import jax.numpy as jnp

    field = jnp.asarray(np.linspace(0.2, 1.0, 12).reshape(3, 2, 2))

    def objective(scale):
        return intensity_fom(field * scale)

    # log10 of a quadratic in the scale: d/ds = 2 / (s ln 10)
    assert float(jax.grad(objective)(1.0)) == pytest.approx(2.0 / math.log(10.0), rel=1e-6)


# ------------------------------------------------------------------------------------------------
# schedules and drive
# ------------------------------------------------------------------------------------------------


def test_continuation_schedule_follows_the_papers_ramp():
    # beta starts at 5 and multiplies by 1.5 every n_it_cont; alpha is 0, then 1.5, then x1.5
    assert continuation_schedule(5.0, 0.0, 0, every=50) == (5.0, 0.0)
    assert continuation_schedule(5.0, 0.0, 49, every=50) == (5.0, 0.0)
    beta, alpha = continuation_schedule(5.0, 0.0, 50, every=50)
    assert (beta, alpha) == pytest.approx((7.5, 1.5))
    beta, alpha = continuation_schedule(5.0, 0.0, 100, every=50)
    assert (beta, alpha) == pytest.approx((11.25, 2.25))
    beta, alpha = continuation_schedule(5.0, 0.0, 349, every=50)
    assert (beta, alpha) == pytest.approx((5.0 * 1.5**6, 1.5 * 1.5**5))


def test_continuation_schedule_caps_and_ramps_a_non_zero_alpha():
    assert continuation_schedule(5.0, 0.0, 300, every=50, beta_max=20.0)[0] == 20.0
    assert continuation_schedule(5.0, 0.0, 300, every=50, alpha_max=4.0)[1] == 4.0
    beta, alpha = continuation_schedule(10.0, 2.0, 50, every=50)
    assert (beta, alpha) == pytest.approx((15.0, 3.0))


def test_continuation_schedule_refuses_bad_counters():
    with pytest.raises(ValueError, match="positive"):
        continuation_schedule(5.0, 0.0, 10, every=0)
    with pytest.raises(ValueError, match="non-negative"):
        continuation_schedule(5.0, 0.0, -1)


def test_volumetric_heat_rate_of_the_reference_heater():
    volume = 4e-6 * 0.25e-6 * 200e-6  # w_metal x h_metal x L
    assert volumetric_heat_rate(POWER, volume) == pytest.approx(1.0e14, rel=1e-9)
    assert heater_power(1.0e14, volume) == pytest.approx(POWER, rel=1e-12)


def test_heat_rate_helpers_refuse_a_non_physical_volume():
    with pytest.raises(ValueError, match="volume"):
        volumetric_heat_rate(POWER, 0.0)
    with pytest.raises(ValueError, match="volume"):
        heater_power(1.0e14, -1.0)
