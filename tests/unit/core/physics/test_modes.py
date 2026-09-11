from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.core.physics.modes import (
    ModeTupleType,
    compute_mode,
    compute_mode_polarization_fraction,
    compute_mode_symmetry_reduced,
    compute_modes,
    sort_modes,
    tidy3d_mode_computation_wrapper,
)


class TestComputeModePolarizationFraction:
    """Test the compute_mode_polarization_fraction function."""

    def test_te_polarization(self):
        """Test TE polarization fraction calculation."""
        # Create a mode with stronger E1 component (TE-like)
        mode = ModeTupleType(
            neff=1.5,
            Ex=np.array([[2.0, 2.0], [2.0, 2.0]]),  # E1 component (axis 0)
            Ey=np.array([[1.0, 1.0], [1.0, 1.0]]),  # E2 component (axis 1)
            Ez=np.array([[0.0, 0.0], [0.0, 0.0]]),
            Hx=np.array([[0.0, 0.0], [0.0, 0.0]]),
            Hy=np.array([[0.0, 0.0], [0.0, 0.0]]),
            Hz=np.array([[0.0, 0.0], [0.0, 0.0]]),
        )

        fraction = compute_mode_polarization_fraction(mode, (0, 1), "te")

        # Expected: |E1|^2 / (|E1|^2 + |E2|^2) = 16 / (16 + 4) = 0.8
        expected = 16.0 / (16.0 + 4.0)
        assert fraction == pytest.approx(expected)

    def test_tm_polarization(self):
        """Test TM polarization fraction calculation."""
        # Create a mode with stronger E2 component (TM-like)
        mode = ModeTupleType(
            neff=1.5,
            Ex=np.array([[1.0, 1.0], [1.0, 1.0]]),  # E1 component (axis 0)
            Ey=np.array([[3.0, 3.0], [3.0, 3.0]]),  # E2 component (axis 1)
            Ez=np.array([[0.0, 0.0], [0.0, 0.0]]),
            Hx=np.array([[0.0, 0.0], [0.0, 0.0]]),
            Hy=np.array([[0.0, 0.0], [0.0, 0.0]]),
            Hz=np.array([[0.0, 0.0], [0.0, 0.0]]),
        )

        fraction = compute_mode_polarization_fraction(mode, (0, 1), "tm")

        # Expected: |E2|^2 / (|E1|^2 + |E2|^2) = 36 / (4 + 36) = 0.9
        expected = 36.0 / (4.0 + 36.0)
        assert fraction == pytest.approx(expected)

    def test_invalid_pol_raises_error(self):
        """Test that invalid polarization raises ValueError."""
        mode = ModeTupleType(
            neff=1.5,
            Ex=np.array([[1.0]]),
            Ey=np.array([[1.0]]),
            Ez=np.array([[0.0]]),
            Hx=np.array([[0.0]]),
            Hy=np.array([[0.0]]),
            Hz=np.array([[0.0]]),
        )

        with pytest.raises(ValueError, match="pol must be 'te' or 'tm'"):
            compute_mode_polarization_fraction(mode, (0, 1), "invalid_pol")


class TestSortModes:
    """Test the sort_modes function."""

    def create_test_modes(self):
        """Helper function to create test modes."""
        mode1 = ModeTupleType(
            neff=2.0 + 0.1j,
            Ex=np.ones((2, 2)),
            Ey=np.ones((2, 2)) * 0.1,
            Ez=np.zeros((2, 2)),
            Hx=np.zeros((2, 2)),
            Hy=np.zeros((2, 2)),
            Hz=np.zeros((2, 2)),
        )
        mode2 = ModeTupleType(
            neff=1.5 + 0.1j,
            Ex=np.ones((2, 2)) * 0.1,
            Ey=np.ones((2, 2)),
            Ez=np.zeros((2, 2)),
            Hx=np.zeros((2, 2)),
            Hy=np.zeros((2, 2)),
            Hz=np.zeros((2, 2)),
        )
        mode3 = ModeTupleType(
            neff=3.0 + 0.1j,
            Ex=np.ones((2, 2)) * 0.6,
            Ey=np.ones((2, 2)) * 0.4,
            Ez=np.zeros((2, 2)),
            Hx=np.zeros((2, 2)),
            Hy=np.zeros((2, 2)),
            Hz=np.zeros((2, 2)),
        )
        return [mode1, mode2, mode3]

    def test_sort_no_filter(self):
        """Test sorting without polarization filter."""
        modes = self.create_test_modes()
        sorted_modes = sort_modes(modes, None, (0, 1))

        # Should be sorted by descending real part of neff
        expected_order = [3.0, 2.0, 1.5]
        actual_order = [float(np.real(m.neff)) for m in sorted_modes]
        assert actual_order == expected_order

    def test_sort_te_filter(self):
        """Test sorting with TE polarization filter."""
        modes = self.create_test_modes()
        sorted_modes = sort_modes(modes, "te", (0, 1))

        # mode1 has strong Ex (TE-like), mode2 has weak Ex, mode3 is mixed but Ex > Ey
        # TE modes should come first, sorted by neff
        assert sorted_modes[0].neff == 3.0 + 0.1j  # Highest neff TE-like
        assert sorted_modes[1].neff == 2.0 + 0.1j  # TE-like
        assert sorted_modes[2].neff == 1.5 + 0.1j  # TM-like

    def test_sort_tm_filter(self):
        """Test sorting with TM polarization filter."""
        modes = self.create_test_modes()
        sorted_modes = sort_modes(modes, "tm", (0, 1))

        # mode2 has strong Ey (TM-like), should come first among TM modes
        assert sorted_modes[0].neff == 1.5 + 0.1j  # Most TM-like
        # The other modes should follow


class TestComputeMode:
    """Test the compute_mode function."""

    def test_only_bend_radius_raises(self):
        """Setting bend_radius without bend_axis raises ValueError."""
        inv_permittivities = jnp.ones((1, 5, 6, 1))
        with pytest.raises(ValueError, match="both be set or both be None"):
            compute_mode(2e14, inv_permittivities, 1.0, 1e-8, "+", bend_radius=5e-6)

    def test_only_bend_axis_raises(self):
        """Setting bend_axis without bend_radius raises ValueError."""
        inv_permittivities = jnp.ones((1, 5, 6, 1))
        with pytest.raises(ValueError, match="both be set or both be None"):
            compute_mode(2e14, inv_permittivities, 1.0, 1e-8, "+", bend_axis=1)

    def test_invalid_permittivities_shape(self):
        """Test that invalid permittivities shape raises exception."""
        frequency = 2e14
        # 3D array but not squeezable to 2D
        inv_permittivities = jnp.ones((3, 2, 2, 2))
        inv_permeabilities = 1.0
        resolution = 1e-8

        with pytest.raises(Exception, match="Invalid shape of inv_permittivities"):
            compute_mode(frequency, inv_permittivities, inv_permeabilities, resolution, "+")

    def test_invalid_permeabilities_shape(self):
        """Test that invalid permeabilities shape raises exception."""
        frequency = 2e14
        inv_permittivities = jnp.ones((3, 1, 5, 5))
        # Invalid: 4D array with no singleton spatial dim
        inv_permeabilities = jnp.ones((3, 2, 2, 2))
        resolution = 1e-8

        with pytest.raises(Exception, match="Invalid shape of inv_permeabilities"):
            compute_mode(frequency, inv_permittivities, inv_permeabilities, resolution, "+")

    def test_invalid_transverse_coords_shape(self):
        """Transverse coordinate arrays must match the mode cross-section shape."""
        inv_permittivities = jnp.ones((1, 2, 2, 1))

        with pytest.raises(ValueError, match="length 3"):
            compute_mode(
                frequency=2e14,
                inv_permittivities=inv_permittivities,
                inv_permeabilities=1.0,
                resolution=1e-8,
                direction="+",
                transverse_coords=[np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0, 2.0])],
            )

    @patch("fdtdx.core.physics.modes.tidy3d_mode_computation_wrapper")
    @patch("fdtdx.core.physics.modes.normalize_by_poynting_flux")
    def test_transverse_coords_passed_to_tidy3d_and_normalization(self, mock_normalize, mock_tidy3d_wrapper):
        """Non-uniform transverse coordinates are passed through and used for normalization weights."""
        mock_mode = ModeTupleType(
            neff=1.5 + 0.1j,
            Ex=np.ones((3, 3), dtype=np.complex64),
            Ey=np.ones((3, 3), dtype=np.complex64),
            Ez=np.ones((3, 3), dtype=np.complex64),
            Hx=np.ones((3, 3), dtype=np.complex64),
            Hy=np.ones((3, 3), dtype=np.complex64),
            Hz=np.ones((3, 3), dtype=np.complex64),
        )
        mock_tidy3d_wrapper.return_value = [mock_mode]
        mock_normalize.return_value = (
            jnp.ones((3, 3, 3, 1), dtype=jnp.complex64),
            jnp.ones((3, 3, 3, 1), dtype=jnp.complex64),
        )

        compute_mode(
            frequency=2e14,
            inv_permittivities=jnp.ones((1, 3, 3, 1)),
            inv_permeabilities=1.0,
            resolution=1e-8,
            direction="+",
            transverse_coords=[np.asarray([0.0, 1e-6, 3e-6, 4e-6]), np.asarray([0.0, 2e-6, 5e-6, 6e-6])],
            mode_backend="tidy3d",
        )

        wrapper_kwargs = mock_tidy3d_wrapper.call_args.kwargs
        assert np.allclose(wrapper_kwargs["coords"][0], [0.0, 1.0, 3.0, 4.0])
        assert np.allclose(wrapper_kwargs["coords"][1], [0.0, 2.0, 5.0, 6.0])

        normalize_kwargs = mock_normalize.call_args.kwargs
        assert normalize_kwargs["axis"] == 2

        expected_area = jnp.asarray(
            [
                [[2e-12], [3e-12], [1e-12]],
                [[4e-12], [6e-12], [2e-12]],
                [[2e-12], [3e-12], [1e-12]],
            ],
            dtype=jnp.float32,
        )
        assert jnp.allclose(normalize_kwargs["area_weights"], expected_area)

    @patch("fdtdx.core.physics.modes.tidy3d_mode_computation_wrapper")
    @patch("fdtdx.core.physics.modes.normalize_by_poynting_flux")
    def test_transverse_coords_passed_to_tidy3d_and_normalization_2d(self, mock_normalize, mock_tidy3d_wrapper):
        """Non-uniform transverse coordinates are passed through and used for normalization weights."""

        mock_mode = ModeTupleType(
            neff=1.5 + 0.1j,
            Ex=np.ones((5,), dtype=np.complex64),
            Ey=np.ones((5,), dtype=np.complex64),
            Ez=np.ones((5,), dtype=np.complex64),
            Hx=np.ones((5,), dtype=np.complex64),
            Hy=np.ones((5,), dtype=np.complex64),
            Hz=np.ones((5,), dtype=np.complex64),
        )
        mock_tidy3d_wrapper.return_value = [mock_mode]
        mock_normalize.return_value = (
            jnp.ones((3, 2, 5, 1), dtype=jnp.complex64),
            jnp.ones((3, 2, 5, 1), dtype=jnp.complex64),
        )

        compute_mode(
            frequency=2e14,
            inv_permittivities=jnp.ones((1, 2, 5, 1)),
            inv_permeabilities=1.0,
            resolution=1e-8,
            direction="+",
            mode_backend="tidy3d",
            transverse_coords=[
                np.asarray([0.0, 1e-6, 2e-6]),  # Uniform spacing of 1um
                np.asarray([0.0, 2e-6, 5e-6, 6e-6, 8e-6, 9e-6]),  # Non-uniform spacing
            ],
        )

        wrapper_kwargs = mock_tidy3d_wrapper.call_args.kwargs

        # since mode_2d triggers, the mode solver will called for the 1d mode
        assert np.allclose(wrapper_kwargs["coords"][0], [0.0, 1.0])
        assert np.allclose(wrapper_kwargs["coords"][1], [0.0, 2.0, 5.0, 6.0, 8.0, 9.0])

        normalize_kwargs = mock_normalize.call_args.kwargs
        assert normalize_kwargs["axis"] == 2

        expected_area = jnp.asarray(
            [
                [[2e-12], [3e-12], [1e-12], [2e-12], [1e-12]],
                [[2e-12], [3e-12], [1e-12], [2e-12], [1e-12]],
            ],
            dtype=jnp.float32,
        )

        assert jnp.allclose(normalize_kwargs["area_weights"], expected_area)


class TestAnisotropicModeComputation:
    """Test anisotropic material handling in compute_mode."""

    @patch("fdtdx.core.physics.modes.tidy3d_mode_computation_wrapper")
    @patch("fdtdx.core.physics.modes.normalize_by_poynting_flux")
    def test_anisotropic_permittivity_rotation_propagation_axis_0(self, mock_normalize, mock_tidy3d_wrapper):
        """Test that anisotropic permittivity is rotated correctly for propagation along axis 0.

        When propagation is along x (axis 0), tidy3d (x,y,z) maps to physical (y,z,x).
        So physical permittivity (eps_x, eps_y, eps_z) should become tidy3d (eps_y, eps_z, eps_x).
        """
        mock_mode = ModeTupleType(
            neff=1.5 + 0.1j,
            Ex=np.ones((5, 6), dtype=np.complex64),
            Ey=np.ones((5, 6), dtype=np.complex64),
            Ez=np.ones((5, 6), dtype=np.complex64),
            Hx=np.ones((5, 6), dtype=np.complex64),
            Hy=np.ones((5, 6), dtype=np.complex64),
            Hz=np.ones((5, 6), dtype=np.complex64),
        )
        mock_tidy3d_wrapper.return_value = [mock_mode]
        mock_normalize.return_value = (
            jnp.ones((3, 1, 5, 6), dtype=jnp.complex64),
            jnp.ones((3, 1, 5, 6), dtype=jnp.complex64),
        )

        # Anisotropic permittivity with distinct values: eps_x=2, eps_y=3, eps_z=4
        # Shape: (3, 1, 5, 6) - propagation along axis 0
        inv_permittivities = jnp.zeros((3, 1, 5, 6))
        inv_permittivities = inv_permittivities.at[0].set(1 / 2.0)  # eps_x = 2
        inv_permittivities = inv_permittivities.at[1].set(1 / 3.0)  # eps_y = 3
        inv_permittivities = inv_permittivities.at[2].set(1 / 4.0)  # eps_z = 4

        compute_mode(
            frequency=2e14,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=1.0,
            resolution=1e-8,
            direction="+",
            mode_backend="tidy3d",
        )

        # Check what was passed to tidy3d_wrapper
        call_args = mock_tidy3d_wrapper.call_args
        perm_passed = call_args.kwargs["permittivity_cross_section"]

        # For propagation along axis 0: perm_idx = [1, 2, 0]
        # So tidy3d should receive (eps_y, eps_z, eps_x) = (3, 4, 2)
        assert perm_passed.shape[0] == 3
        assert np.allclose(perm_passed[0], 3.0)  # tidy3d x component = physical y
        assert np.allclose(perm_passed[1], 4.0)  # tidy3d y component = physical z
        assert np.allclose(perm_passed[2], 2.0)  # tidy3d z component = physical x

    @patch("fdtdx.core.physics.modes.tidy3d_mode_computation_wrapper")
    @patch("fdtdx.core.physics.modes.normalize_by_poynting_flux")
    def test_anisotropic_permittivity_rotation_propagation_axis_1(self, mock_normalize, mock_tidy3d_wrapper):
        """Test that anisotropic permittivity is rotated correctly for propagation along axis 1.

        When propagation is along y (axis 1), tidy3d (x,y,z) maps to physical (x,z,y).
        So physical permittivity (eps_x, eps_y, eps_z) should become tidy3d (eps_x, eps_z, eps_y).
        """
        mock_mode = ModeTupleType(
            neff=1.5 + 0.1j,
            Ex=np.ones((5, 6), dtype=np.complex64),
            Ey=np.ones((5, 6), dtype=np.complex64),
            Ez=np.ones((5, 6), dtype=np.complex64),
            Hx=np.ones((5, 6), dtype=np.complex64),
            Hy=np.ones((5, 6), dtype=np.complex64),
            Hz=np.ones((5, 6), dtype=np.complex64),
        )
        mock_tidy3d_wrapper.return_value = [mock_mode]
        mock_normalize.return_value = (
            jnp.ones((3, 5, 1, 6), dtype=jnp.complex64),
            jnp.ones((3, 5, 1, 6), dtype=jnp.complex64),
        )

        # Anisotropic permittivity with distinct values: eps_x=2, eps_y=3, eps_z=4
        # Shape: (3, 5, 1, 6) - propagation along axis 1
        inv_permittivities = jnp.zeros((3, 5, 1, 6))
        inv_permittivities = inv_permittivities.at[0].set(1 / 2.0)  # eps_x = 2
        inv_permittivities = inv_permittivities.at[1].set(1 / 3.0)  # eps_y = 3
        inv_permittivities = inv_permittivities.at[2].set(1 / 4.0)  # eps_z = 4

        compute_mode(
            frequency=2e14,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=1.0,
            resolution=1e-8,
            direction="+",
            mode_backend="tidy3d",
        )

        # Check what was passed to tidy3d_wrapper
        call_args = mock_tidy3d_wrapper.call_args
        perm_passed = call_args.kwargs["permittivity_cross_section"]

        # For propagation along axis 1: perm_idx = [0, 2, 1]
        # So tidy3d should receive (eps_x, eps_z, eps_y) = (2, 4, 3)
        assert perm_passed.shape[0] == 3
        assert np.allclose(perm_passed[0], 2.0)  # tidy3d x component = physical x
        assert np.allclose(perm_passed[1], 4.0)  # tidy3d y component = physical z
        assert np.allclose(perm_passed[2], 3.0)  # tidy3d z component = physical y

    @patch("fdtdx.core.physics.modes.tidy3d_mode_computation_wrapper")
    @patch("fdtdx.core.physics.modes.normalize_by_poynting_flux")
    def test_anisotropic_permittivity_rotation_propagation_axis_2(self, mock_normalize, mock_tidy3d_wrapper):
        """Test that anisotropic permittivity is unchanged for propagation along axis 2.

        When propagation is along z (axis 2), tidy3d (x,y,z) maps directly to physical (x,y,z).
        So physical permittivity (eps_x, eps_y, eps_z) stays as (eps_x, eps_y, eps_z).
        """
        mock_mode = ModeTupleType(
            neff=1.5 + 0.1j,
            Ex=np.ones((5, 6), dtype=np.complex64),
            Ey=np.ones((5, 6), dtype=np.complex64),
            Ez=np.ones((5, 6), dtype=np.complex64),
            Hx=np.ones((5, 6), dtype=np.complex64),
            Hy=np.ones((5, 6), dtype=np.complex64),
            Hz=np.ones((5, 6), dtype=np.complex64),
        )
        mock_tidy3d_wrapper.return_value = [mock_mode]
        mock_normalize.return_value = (
            jnp.ones((3, 5, 6, 1), dtype=jnp.complex64),
            jnp.ones((3, 5, 6, 1), dtype=jnp.complex64),
        )

        # Anisotropic permittivity with distinct values: eps_x=2, eps_y=3, eps_z=4
        # Shape: (3, 5, 6, 1) - propagation along axis 2
        inv_permittivities = jnp.zeros((3, 5, 6, 1))
        inv_permittivities = inv_permittivities.at[0].set(1 / 2.0)  # eps_x = 2
        inv_permittivities = inv_permittivities.at[1].set(1 / 3.0)  # eps_y = 3
        inv_permittivities = inv_permittivities.at[2].set(1 / 4.0)  # eps_z = 4

        compute_mode(
            frequency=2e14,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=1.0,
            resolution=1e-8,
            direction="+",
            mode_backend="tidy3d",
        )

        # Check what was passed to tidy3d_wrapper
        call_args = mock_tidy3d_wrapper.call_args
        perm_passed = call_args.kwargs["permittivity_cross_section"]

        # For propagation along axis 2: perm_idx = [0, 1, 2]
        # So tidy3d should receive (eps_x, eps_y, eps_z) = (2, 3, 4) - unchanged
        assert perm_passed.shape[0] == 3
        assert np.allclose(perm_passed[0], 2.0)  # tidy3d x component = physical x
        assert np.allclose(perm_passed[1], 3.0)  # tidy3d y component = physical y
        assert np.allclose(perm_passed[2], 4.0)  # tidy3d z component = physical z

    @patch("fdtdx.core.physics.modes.tidy3d_mode_computation_wrapper")
    @patch("fdtdx.core.physics.modes.normalize_by_poynting_flux")
    def test_anisotropic_permeability_rotation(self, mock_normalize, mock_tidy3d_wrapper):
        """Test that anisotropic permeability is rotated correctly."""
        mock_mode = ModeTupleType(
            neff=1.5 + 0.1j,
            Ex=np.ones((5, 6), dtype=np.complex64),
            Ey=np.ones((5, 6), dtype=np.complex64),
            Ez=np.ones((5, 6), dtype=np.complex64),
            Hx=np.ones((5, 6), dtype=np.complex64),
            Hy=np.ones((5, 6), dtype=np.complex64),
            Hz=np.ones((5, 6), dtype=np.complex64),
        )
        mock_tidy3d_wrapper.return_value = [mock_mode]
        mock_normalize.return_value = (
            jnp.ones((3, 1, 5, 6), dtype=jnp.complex64),
            jnp.ones((3, 1, 5, 6), dtype=jnp.complex64),
        )

        # Anisotropic permittivity and permeability
        # Shape: (3, 1, 5, 6) - propagation along axis 0
        inv_permittivities = jnp.zeros((3, 1, 5, 6))
        inv_permittivities = inv_permittivities.at[0].set(1 / 2.0)  # eps_x = 2
        inv_permittivities = inv_permittivities.at[1].set(1 / 3.0)  # eps_y = 3
        inv_permittivities = inv_permittivities.at[2].set(1 / 4.0)  # eps_z = 4
        # Anisotropic permeability: mu_x=1.1, mu_y=1.2, mu_z=1.3
        inv_permeabilities = jnp.zeros((3, 1, 5, 6))
        inv_permeabilities = inv_permeabilities.at[0].set(1 / 1.1)  # mu_x = 1.1
        inv_permeabilities = inv_permeabilities.at[1].set(1 / 1.2)  # mu_y = 1.2
        inv_permeabilities = inv_permeabilities.at[2].set(1 / 1.3)  # mu_z = 1.3

        compute_mode(
            frequency=2e14,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
            resolution=1e-8,
            direction="+",
            mode_backend="tidy3d",
        )

        # Check what was passed to tidy3d_wrapper
        call_args = mock_tidy3d_wrapper.call_args
        perm_passed = call_args.kwargs["permeability_cross_section"]

        # For propagation along axis 0: perm_idx = [1, 2, 0]
        # So tidy3d should receive (mu_y, mu_z, mu_x) = (1.2, 1.3, 1.1)
        assert perm_passed.shape[0] == 3
        assert np.allclose(perm_passed[0], 1.2, rtol=1e-5)  # tidy3d x = physical y
        assert np.allclose(perm_passed[1], 1.3, rtol=1e-5)  # tidy3d y = physical z
        assert np.allclose(perm_passed[2], 1.1, rtol=1e-5)  # tidy3d z = physical x

    @patch("fdtdx.core.physics.modes.tidy3d_mode_computation_wrapper")
    @patch("fdtdx.core.physics.modes.normalize_by_poynting_flux")
    def test_isotropic_permittivity_no_rotation_needed(self, mock_normalize, mock_tidy3d_wrapper):
        """Test that isotropic permittivity (1 component) is not affected by rotation logic."""
        mock_mode = ModeTupleType(
            neff=1.5 + 0.1j,
            Ex=np.ones((5, 6), dtype=np.complex64),
            Ey=np.ones((5, 6), dtype=np.complex64),
            Ez=np.ones((5, 6), dtype=np.complex64),
            Hx=np.ones((5, 6), dtype=np.complex64),
            Hy=np.ones((5, 6), dtype=np.complex64),
            Hz=np.ones((5, 6), dtype=np.complex64),
        )
        mock_tidy3d_wrapper.return_value = [mock_mode]
        mock_normalize.return_value = (
            jnp.ones((3, 1, 5, 6), dtype=jnp.complex64),
            jnp.ones((3, 1, 5, 6), dtype=jnp.complex64),
        )

        # Isotropic permittivity (1 component): eps=4.0
        # Shape: (1, 1, 5, 6) - propagation along axis 0
        inv_permittivities = jnp.ones((1, 1, 5, 6)) * 0.25  # eps = 4.0

        compute_mode(
            frequency=2e14,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=1.0,
            resolution=1e-8,
            direction="+",
            mode_backend="tidy3d",
        )

        # Check what was passed to tidy3d_wrapper
        call_args = mock_tidy3d_wrapper.call_args
        perm_passed = call_args.kwargs["permittivity_cross_section"]

        # Isotropic case: should have 1 component and value should be preserved
        assert perm_passed.shape[0] == 1
        assert np.allclose(perm_passed[0], 4.0)

    @patch("fdtdx.core.physics.modes.tidy3d_mode_computation_wrapper")
    @patch("fdtdx.core.physics.modes.normalize_by_poynting_flux")
    def test_full_anisotropy_permittivity_9_components(self, mock_normalize, mock_tidy3d_wrapper):
        """Test 9-component full anisotropy permittivity (matrix inversion + rotation)."""
        mock_mode = ModeTupleType(
            neff=1.5 + 0.1j,
            Ex=np.ones((5, 6), dtype=np.complex64),
            Ey=np.ones((5, 6), dtype=np.complex64),
            Ez=np.ones((5, 6), dtype=np.complex64),
            Hx=np.ones((5, 6), dtype=np.complex64),
            Hy=np.ones((5, 6), dtype=np.complex64),
            Hz=np.ones((5, 6), dtype=np.complex64),
        )
        mock_tidy3d_wrapper.return_value = [mock_mode]
        mock_normalize.return_value = (
            jnp.ones((3, 1, 5, 6), dtype=jnp.complex64),
            jnp.ones((3, 1, 5, 6), dtype=jnp.complex64),
        )

        # Full 9-component anisotropy: diagonal inv_permittivity matrix
        # inv_eps = diag(1/2, 1/3, 1/4) => eps = diag(2, 3, 4)
        # Shape: (9, 1, 5, 6) - propagation along axis 0
        inv_permittivities = jnp.zeros((9, 1, 5, 6))
        inv_permittivities = inv_permittivities.at[0].set(1 / 2.0)  # inv_eps_xx
        inv_permittivities = inv_permittivities.at[4].set(1 / 3.0)  # inv_eps_yy
        inv_permittivities = inv_permittivities.at[8].set(1 / 4.0)  # inv_eps_zz

        compute_mode(
            frequency=2e14,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=1.0,
            resolution=1e-8,
            direction="+",
            mode_backend="tidy3d",
        )

        # Verify the wrapper was called with 9-component permittivity
        call_args = mock_tidy3d_wrapper.call_args
        perm_passed = call_args.kwargs["permittivity_cross_section"]
        assert perm_passed.shape[0] == 9
        # Verify that the matrix inversion and axis rotation were applied correctly.
        # Input inv_permittivities diagonal: [0]=1/2 (xx), [4]=1/3 (yy), [8]=1/4 (zz).
        # After inversion: eps_xx=2, eps_yy=3, eps_zz=4.
        # After axis rotation for propagation_axis=0 (perm_idx_full_anisotropy=[4,5,3,7,8,6,1,2,0]):
        #   perm_passed[0] = eps_yy = 3.0
        #   perm_passed[4] = eps_zz = 4.0
        # perm_passed has shape (9, Ny, Nz) = (9, 5, 6)
        perm_yy_after_rotation = float(np.mean(np.array(perm_passed[0])))
        perm_zz_after_rotation = float(np.mean(np.array(perm_passed[4])))
        assert abs(perm_yy_after_rotation - 3.0) < 0.1, (
            f"Expected eps_yy≈3.0 after inversion+rotation, got {perm_yy_after_rotation:.3f}"
        )
        assert abs(perm_zz_after_rotation - 4.0) < 0.1, (
            f"Expected eps_zz≈4.0 after inversion+rotation, got {perm_zz_after_rotation:.3f}"
        )

    @patch("fdtdx.core.physics.modes.tidy3d_mode_computation_wrapper")
    @patch("fdtdx.core.physics.modes.normalize_by_poynting_flux")
    def test_full_anisotropy_permeability_9_components(self, mock_normalize, mock_tidy3d_wrapper):
        """Test 9-component full anisotropy permeability (matrix inversion + rotation)."""
        mock_mode = ModeTupleType(
            neff=1.5 + 0.1j,
            Ex=np.ones((5, 6), dtype=np.complex64),
            Ey=np.ones((5, 6), dtype=np.complex64),
            Ez=np.ones((5, 6), dtype=np.complex64),
            Hx=np.ones((5, 6), dtype=np.complex64),
            Hy=np.ones((5, 6), dtype=np.complex64),
            Hz=np.ones((5, 6), dtype=np.complex64),
        )
        mock_tidy3d_wrapper.return_value = [mock_mode]
        mock_normalize.return_value = (
            jnp.ones((3, 1, 5, 6), dtype=jnp.complex64),
            jnp.ones((3, 1, 5, 6), dtype=jnp.complex64),
        )

        # 3-component permittivity
        inv_permittivities = jnp.zeros((3, 1, 5, 6))
        inv_permittivities = inv_permittivities.at[0].set(1 / 2.0)
        inv_permittivities = inv_permittivities.at[1].set(1 / 3.0)
        inv_permittivities = inv_permittivities.at[2].set(1 / 4.0)

        # Full 9-component anisotropic permeability: diagonal
        # Shape: (9, 1, 5, 6) - propagation along axis 0
        inv_permeabilities = jnp.zeros((9, 1, 5, 6))
        inv_permeabilities = inv_permeabilities.at[0].set(1 / 1.1)  # inv_mu_xx
        inv_permeabilities = inv_permeabilities.at[4].set(1 / 1.2)  # inv_mu_yy
        inv_permeabilities = inv_permeabilities.at[8].set(1 / 1.3)  # inv_mu_zz

        compute_mode(
            frequency=2e14,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
            resolution=1e-8,
            direction="+",
            mode_backend="tidy3d",
        )

        # Verify the wrapper was called with 9-component permeability
        call_args = mock_tidy3d_wrapper.call_args
        perm_passed = call_args.kwargs["permeability_cross_section"]
        assert perm_passed.shape[0] == 9


class TestBackwardModePhaseConvention:
    """Regression tests for the spurious global ±i phase on backward ("-") modes (tidy3d >= 2.9).

    "+" and "-" modes must share the same phase convention exactly.
    """

    frequency = 3e8 / 1.55e-6
    resolution = 100e-9

    def _strip_waveguide_inv_eps(self, num_components: int):
        """Strip waveguide cross-section, propagation along x (axis 0)."""
        shape = (num_components, 1, 20, 15)
        eps = np.ones(shape)
        core = (slice(None), slice(None), slice(7, 13), slice(6, 10))
        if num_components == 1:
            eps[core] = 3.48**2
        elif num_components == 3:
            eps[0][core[1:]] = 3.48**2
            eps[1][core[1:]] = 3.40**2
            eps[2][core[1:]] = 3.45**2
        elif num_components == 9:
            # symmetric (reciprocal) tensor with off-diagonal xy coupling;
            # inv_permittivities holds the INVERSE tensor, so build eps first
            eps_full = np.tile(np.eye(3).reshape(9, 1, 1, 1), (1, *shape[1:]))
            eps_mat = np.array(
                [
                    [3.48**2, 0.3, 0.0],
                    [0.3, 3.40**2, 0.0],
                    [0.0, 0.0, 3.45**2],
                ]
            )
            inv_mat = np.linalg.inv(eps_mat)
            for i in range(9):
                eps_full[i][core[1:]] = inv_mat.reshape(9)[i]
            return jnp.asarray(eps_full)
        return jnp.asarray(1.0 / eps)

    def _compute_both_directions(self, inv_eps):
        results = {}
        for direction in ["+", "-"]:
            E, H, neff = compute_mode(
                frequency=self.frequency,
                inv_permittivities=inv_eps,
                inv_permeabilities=1.0,
                resolution=self.resolution,
                direction=direction,
                mode_index=0,
            )
            results[direction] = (np.asarray(E), np.asarray(H), complex(neff))
        return results

    def _assert_backward_is_reciprocity_transform(self, results, propagation_axis=0):
        Ep, Hp, neff_p = results["+"]
        Em, Hm, neff_m = results["-"]

        assert neff_m == pytest.approx(neff_p, rel=1e-6)

        # reciprocity transform in the physical frame: longitudinal E and
        # transverse H flip sign, transverse E and longitudinal H are unchanged
        expected_E = Ep.copy()
        expected_E[propagation_axis] *= -1
        expected_H = -Hp.copy()
        expected_H[propagation_axis] *= -1

        scale = np.abs(Ep).max()
        np.testing.assert_allclose(Em, expected_E, atol=1e-5 * scale)
        np.testing.assert_allclose(Hm, expected_H, atol=1e-5 * np.abs(Hp).max())

        # real Poynting flux must be -1 (normalized) along the propagation axis
        S = np.cross(np.conj(Em), Hm, axisa=0, axisb=0, axisc=0)
        flux = 0.5 * np.real(S[propagation_axis]).sum()
        assert flux == pytest.approx(-1.0, abs=1e-3)

        # lossless mode: transverse E of the backward mode must be purely real
        # (the ±i bug made it purely imaginary)
        transverse = [ax for ax in range(3) if ax != propagation_axis]
        max_trans = max(np.abs(Em[ax]).max() for ax in transverse)
        for ax in transverse:
            assert np.abs(Em[ax].imag).max() <= 1e-5 * max_trans

    def test_backward_mode_isotropic(self):
        """Isotropic (diagonal solver path): '-' equals reciprocity transform of '+'."""
        results = self._compute_both_directions(self._strip_waveguide_inv_eps(1))
        self._assert_backward_is_reciprocity_transform(results)

    def test_backward_mode_diagonal_anisotropic(self):
        """Diagonal anisotropic (3 components): '-' equals reciprocity transform of '+'."""
        results = self._compute_both_directions(self._strip_waveguide_inv_eps(3))
        self._assert_backward_is_reciprocity_transform(results)

    def test_backward_mode_symmetric_tensorial(self):
        """Symmetric tensorial eps (9 components, reciprocal): tensorial solver path."""
        try:
            results = self._compute_both_directions(self._strip_waveguide_inv_eps(9))
        except Exception as e:  # tidy3d raises inside a jax.pure_callback (JaxRuntimeError)
            if "tensorial mode solver" in str(e):
                pytest.skip("tensorial mode solver requires tidy3d-extras")
            raise
        self._assert_backward_is_reciprocity_transform(results)

    def test_backward_mode_non_reciprocal_raises(self):
        """Asymmetric (non-reciprocal) eps tensor: '-' is not supported and must raise."""
        eps = np.tile(np.eye(3).reshape(9, 1, 1), (1, 20, 15))
        eps[1] = 0.3  # eps_xy != eps_yx
        with pytest.raises(NotImplementedError, match="reciprocity"):
            tidy3d_mode_computation_wrapper(
                frequency=self.frequency,
                permittivity_cross_section=eps,
                coords=[np.arange(21) * 0.1, np.arange(16) * 0.1],
                direction="-",
            )


class TestTidy3DModeComputationWrapper:
    """Test the tidy3d_mode_computation_wrapper function."""

    def create_mock_eh_data(self, shape, num_modes=1):
        """Helper to create properly structured mock EH data."""
        # Create numpy arrays with proper shape and add squeeze method
        if num_modes == 1:
            # Single mode - 2D arrays
            E_data = (
                np.ones(shape, dtype=np.complex64),
                np.ones(shape, dtype=np.complex64),
                np.ones(shape, dtype=np.complex64),
            )
            H_data = (
                np.ones(shape, dtype=np.complex64),
                np.ones(shape, dtype=np.complex64),
                np.ones(shape, dtype=np.complex64),
            )
        else:
            # Multiple modes - 3D arrays with mode dimension last
            E_data = (
                np.ones((*shape, num_modes), dtype=np.complex64),
                np.ones((*shape, num_modes), dtype=np.complex64),
                np.ones((*shape, num_modes), dtype=np.complex64),
            )
            H_data = (
                np.ones((*shape, num_modes), dtype=np.complex64),
                np.ones((*shape, num_modes), dtype=np.complex64),
                np.ones((*shape, num_modes), dtype=np.complex64),
            )

        # Create a mock object that behaves like the expected EH structure
        class MockEH:
            def __init__(self, E_data, H_data):
                self.E_data = E_data
                self.H_data = H_data

            def squeeze(self):
                # Return a tuple that can be unpacked as ((Ex, Ey, Ez), (Hx, Hy, Hz))
                return (self.E_data, self.H_data)

        return MockEH(E_data, H_data)

    @patch("tidy3d.components.mode.solver.compute_modes")
    def test_single_mode(self, mock_compute_modes):
        """Test single mode computation."""
        # Create properly structured mock data
        mock_EH = self.create_mock_eh_data((5, 5), num_modes=1)
        mock_neffs = 1.5 + 0.1j

        mock_compute_modes.return_value = (mock_EH, mock_neffs, None)

        # Test inputs
        frequency = 2e14
        permittivity = np.ones((3, 5, 5))
        coords = [np.linspace(0, 1, 6), np.linspace(0, 1, 6)]  # x, y coordinates

        modes = tidy3d_mode_computation_wrapper(
            frequency=frequency, permittivity_cross_section=permittivity, coords=coords, direction="+", num_modes=1
        )

        assert len(modes) == 1
        assert modes[0].neff == 1.5 + 0.1j
        assert modes[0].Ex.shape == (5, 5)

    @patch("tidy3d.components.mode.solver.compute_modes")
    def test_multiple_modes(self, mock_compute_modes):
        """Test multiple mode computation."""
        # Create properly structured mock data for multiple modes
        mock_EH = self.create_mock_eh_data((5, 5), num_modes=3)
        mock_neffs = np.array([1.5 + 0.1j, 1.4 + 0.1j, 1.3 + 0.1j])

        mock_compute_modes.return_value = (mock_EH, mock_neffs, None)

        frequency = 2e14
        permittivity = np.ones((3, 5, 5))
        coords = [np.linspace(0, 1, 6), np.linspace(0, 1, 6)]

        modes = tidy3d_mode_computation_wrapper(
            frequency=frequency, permittivity_cross_section=permittivity, coords=coords, direction="+", num_modes=3
        )

        assert len(modes) == 3
        assert modes[0].neff == 1.5 + 0.1j
        assert modes[1].neff == 1.4 + 0.1j
        assert modes[2].neff == 1.3 + 0.1j

    @patch("tidy3d.components.mode.solver.compute_modes")
    def test_with_permeability(self, mock_compute_modes):
        """Test mode computation with permeability cross-section."""
        mock_EH = self.create_mock_eh_data((4, 4), num_modes=1)
        mock_neffs = 1.6 + 0.2j

        mock_compute_modes.return_value = (mock_EH, mock_neffs, None)

        frequency = 2e14
        permittivity = np.ones((3, 4, 4))
        permeability = np.ones((3, 4, 4)) * 2.0  # Non-unity permeability
        coords = [np.linspace(0, 1, 5), np.linspace(0, 1, 5)]

        modes = tidy3d_mode_computation_wrapper(
            frequency=frequency,
            permittivity_cross_section=permittivity,
            coords=coords,
            direction="+",
            permeability_cross_section=permeability,
            num_modes=1,
        )

        assert len(modes) == 1
        assert modes[0].neff == 1.6 + 0.2j


class TestComputeModeBendPassthrough:
    """Bend args are converted and passed to tidy3d_mode_computation_wrapper on the Tidy3D backend.

    The pass-through is still the contract, but only for ``mode_backend="tidy3d"``: since the native
    bend (2026-09-09) the fdtdmex backend removes the curvature from the cross-section itself and
    hands the straight problem on, so on that backend nothing bend-shaped reaches any wrapper. Every
    test here therefore selects the backend whose contract it is describing, instead of relying on
    the native backend falling through to Tidy3D as it did before the native bend existed.
    """

    def _make_mock_mode(self, shape=(5, 6)):
        return ModeTupleType(
            neff=1.5 + 0.1j,
            Ex=np.ones(shape, dtype=np.complex64),
            Ey=np.ones(shape, dtype=np.complex64),
            Ez=np.ones(shape, dtype=np.complex64),
            Hx=np.ones(shape, dtype=np.complex64),
            Hy=np.ones(shape, dtype=np.complex64),
            Hz=np.ones(shape, dtype=np.complex64),
        )

    @patch("fdtdx.core.physics.modes.tidy3d_mode_computation_wrapper")
    @patch("fdtdx.core.physics.modes.normalize_by_poynting_flux")
    def test_no_bend_passes_none_to_wrapper(self, mock_normalize, mock_wrapper):
        """Without bend args, None is passed for bend_radius, bend_axis, and plane_center."""
        mock_wrapper.return_value = [self._make_mock_mode()]
        mock_normalize.return_value = (
            jnp.ones((3, 5, 6, 1), dtype=jnp.complex64),
            jnp.ones((3, 5, 6, 1), dtype=jnp.complex64),
        )

        compute_mode(2e14, jnp.ones((1, 5, 6, 1)), 1.0, 1e-8, "+", mode_backend="tidy3d")

        kwargs = mock_wrapper.call_args.kwargs
        assert kwargs["bend_radius"] is None
        assert kwargs["bend_axis"] is None
        assert kwargs["plane_center"] is None

    @patch("fdtdx.core.physics.modes.tidy3d_mode_computation_wrapper")
    @patch("fdtdx.core.physics.modes.normalize_by_poynting_flux")
    def test_bend_radius_converted_meters_to_micrometers(self, mock_normalize, mock_wrapper):
        """bend_radius is divided by 1e-6 before being passed to tidy3d (m → µm)."""
        mock_wrapper.return_value = [self._make_mock_mode()]
        mock_normalize.return_value = (
            jnp.ones((3, 5, 6, 1), dtype=jnp.complex64),
            jnp.ones((3, 5, 6, 1), dtype=jnp.complex64),
        )

        compute_mode(2e14, jnp.ones((1, 5, 6, 1)), 1.0, 1e-8, "+", bend_radius=5e-6, bend_axis=0, mode_backend="tidy3d")

        kwargs = mock_wrapper.call_args.kwargs
        assert kwargs["bend_radius"] == pytest.approx(5.0)  # 5e-6 m = 5.0 µm

    @patch("fdtdx.core.physics.modes.tidy3d_mode_computation_wrapper")
    @patch("fdtdx.core.physics.modes.normalize_by_poynting_flux")
    def test_bend_axis_remapped_z_propagation(self, mock_normalize, mock_wrapper):
        """For z-propagation, transverse axes are [0,1]: physical bend_axis=1 → tidy3d index 1."""
        mock_wrapper.return_value = [self._make_mock_mode()]
        mock_normalize.return_value = (
            jnp.ones((3, 5, 6, 1), dtype=jnp.complex64),
            jnp.ones((3, 5, 6, 1), dtype=jnp.complex64),
        )

        # z-propagation: inv_permittivities shape (1, 5, 6, 1), singleton at dim 3
        compute_mode(
            2e14, jnp.ones((1, 5, 6, 1)), 1.0, 1e-8, "+", bend_radius=10e-6, bend_axis=1, mode_backend="tidy3d"
        )

        kwargs = mock_wrapper.call_args.kwargs
        assert kwargs["bend_axis"] == 1  # transverse_axes=[0,1], index(1)=1

    @patch("fdtdx.core.physics.modes.tidy3d_mode_computation_wrapper")
    @patch("fdtdx.core.physics.modes.normalize_by_poynting_flux")
    def test_bend_axis_remapped_x_propagation(self, mock_normalize, mock_wrapper):
        """For x-propagation, transverse axes are [1,2]: physical bend_axis=2 → tidy3d index 1."""
        mock_wrapper.return_value = [self._make_mock_mode()]
        mock_normalize.return_value = (
            jnp.ones((3, 1, 5, 6), dtype=jnp.complex64),
            jnp.ones((3, 1, 5, 6), dtype=jnp.complex64),
        )

        # x-propagation: inv_permittivities shape (1, 1, 5, 6), singleton at dim 1
        compute_mode(
            2e14, jnp.ones((1, 1, 5, 6)), 1.0, 1e-8, "+", bend_radius=10e-6, bend_axis=2, mode_backend="tidy3d"
        )

        kwargs = mock_wrapper.call_args.kwargs
        assert kwargs["bend_axis"] == 1  # transverse_axes=[1,2], index(2)=1

    @patch("fdtdx.core.physics.modes.tidy3d_mode_computation_wrapper")
    @patch("fdtdx.core.physics.modes.normalize_by_poynting_flux")
    def test_plane_center_is_midpoint_of_transverse_coords(self, mock_normalize, mock_wrapper):
        """plane_center is computed as the midpoint of each transverse coordinate array."""
        mock_wrapper.return_value = [self._make_mock_mode()]
        mock_normalize.return_value = (
            jnp.ones((3, 5, 6, 1), dtype=jnp.complex64),
            jnp.ones((3, 5, 6, 1), dtype=jnp.complex64),
        )

        resolution = 1e-8
        # z-propagation with transverse dims 5 (x) and 6 (y)
        # coords[0] = np.arange(6) * resolution/1e-6, so last = 5 * resolution/1e-6
        # coords[1] = np.arange(7) * resolution/1e-6, so last = 6 * resolution/1e-6
        compute_mode(
            2e14, jnp.ones((1, 5, 6, 1)), 1.0, resolution, "+", bend_radius=5e-6, bend_axis=0, mode_backend="tidy3d"
        )

        kwargs = mock_wrapper.call_args.kwargs
        expected = (0.5 * 5 * resolution / 1e-6, 0.5 * 6 * resolution / 1e-6)
        assert kwargs["plane_center"] == pytest.approx(expected)

    @patch("fdtdx.core.physics.modes.tidy3d_mode_computation_wrapper")
    def test_the_native_backend_consumes_the_bend_instead_of_passing_it_on(self, mock_wrapper):
        """On the fdtdmex backend the bend never reaches a wrapper: it is folded into the material.

        The other side of the contract above. Since 2026-09-09 the curvature is removed from the
        cross-section before the dispatch, so an isotropic bent solve stays native and the Tidy3D
        wrapper is not called at all - which is why the four pass-through tests have to name the
        Tidy3D backend explicitly to test what they mean to test.
        """
        mock_wrapper.side_effect = AssertionError("the native bend path must not call Tidy3D")

        _E, _H, neff = compute_mode(
            2e14,
            jnp.full((1, 12, 10, 1), 1 / 4.0),
            1.0,
            1e-7,
            "+",
            bend_radius=20e-6,
            bend_axis=1,
            mode_backend="fdtdmex",
        )

        mock_wrapper.assert_not_called()
        assert jnp.isfinite(jnp.real(neff))


class TestComputeModeSymmetryReduced:
    """The symmetry-reduced route: mirror the cross-section, solve, project, restrict."""

    def _make_mock_mode(self, shape):
        return ModeTupleType(
            neff=1.5 + 0.1j,
            Ex=np.ones(shape, dtype=np.complex64),
            Ey=np.ones(shape, dtype=np.complex64),
            Ez=np.ones(shape, dtype=np.complex64),
            Hx=np.ones(shape, dtype=np.complex64),
            Hy=np.ones(shape, dtype=np.complex64),
            Hz=np.ones(shape, dtype=np.complex64),
        )

    def _kwargs(self, **overrides):
        # x-propagation (singleton at dim 1), 4 x 3 transverse cells.
        kwargs = dict(
            mirrored_axes=(2,),
            walls={2: 1},
            frequency=2e14,
            inv_permittivities=jnp.ones((1, 1, 4, 3)),
            inv_permeabilities=1.0,
            resolution=1e-8,
            object_name="modesrc",
        )
        kwargs.update(overrides)
        return kwargs

    def test_bend_whose_radial_axis_is_mirrored_raises(self):
        # Convention settled 2026-09-09: bend_axis is normal to the bend plane, so the radius - and
        # the index the bend transform scales - grows along the OTHER transverse axis. Here
        # x-propagation makes the transverse pair (y, z); bend_axis=y puts the radial direction on z,
        # which is exactly the axis config.symmetry mirrors, so the mirrored cross-section is not
        # symmetric and the reduced run cannot represent the mode. Rejected before any solve.
        with pytest.raises(ValueError, match="radius grows along the z-axis"):
            compute_mode_symmetry_reduced(**self._kwargs(bend_radius=5e-6, bend_axis=1))

    @patch("fdtdx.core.physics.modes.tidy3d_mode_computation_wrapper")
    @patch("fdtdx.core.physics.modes.normalize_by_poynting_flux")
    def test_bend_about_the_mirrored_axis_itself_is_allowed(self, mock_normalize, mock_wrapper):
        # The complement of the case above, and the one the old guard wrongly refused: bend_axis=z is
        # the normal of the bend plane, so the radius grows along y and the z mirror plane survives
        # the bend untouched. Allowed, and the solve goes through.
        mock_wrapper.return_value = [self._make_mock_mode((4, 6))]
        mock_normalize.side_effect = lambda E, H, axis, area_weights=None: (E, H)

        mode_E, mode_H, _neff = compute_mode_symmetry_reduced(
            **self._kwargs(bend_radius=5e-6, bend_axis=2, mode_backend="tidy3d")
        )

        assert mode_E.shape == (3, 1, 4, 3)  # solved on the mirrored plane, restricted to the kept half
        assert mode_H.shape == (3, 1, 4, 3)

    @patch("fdtdx.core.physics.modes.tidy3d_mode_computation_wrapper")
    @patch("fdtdx.core.physics.modes.normalize_by_poynting_flux")
    def test_survives_jit(self, mock_normalize, mock_wrapper):
        # A mode source or mode-overlap detector overlapping a Device solves its mode inside
        # apply_params, which callers trace. The parity residual must therefore not be concretized:
        # its diagnostics are skipped under tracing instead.
        mock_wrapper.return_value = [self._make_mock_mode((4, 6))]
        mock_normalize.side_effect = lambda E, H, axis, area_weights=None: (E, H)

        def traced(inv_permittivities):
            mode_E, _mode_H, _neff = compute_mode_symmetry_reduced(
                **self._kwargs(inv_permittivities=inv_permittivities, mode_backend="tidy3d")
            )
            return mode_E

        eager = traced(jnp.ones((1, 1, 4, 3)))
        jitted = jax.jit(traced)(jnp.ones((1, 1, 4, 3)))
        assert jitted.shape == eager.shape
        assert jnp.allclose(jitted, eager)


# ================================================================================================
# Track J phase 0: target_neff, the plural entry point, the 2-D collapse, and precision.
# Every test below runs on the native ("fdtdmex") backend.
# ================================================================================================

LAM = 1.55e-6
FREQ = 299792458.0 / LAM
N_SI, N_SIO2 = 3.48, 1.55
N_TIN, K_TIN = 3.1477, 5.8429
#: TiN at 1.55 um in the exp(+i k0 n z) convention the solver returns, so Im(n_eff) > 0 is loss.
EPS_TIN = (N_TIN**2 - K_TIN**2) + 2j * N_TIN * K_TIN


@pytest.fixture
def float64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _strip(nx: int = 40, ny: int = 30) -> jnp.ndarray:
    """A 40 x 30 cell Si strip in silica at 40 nm; propagation along the first axis."""
    eps = np.full((nx, ny), 2.25)
    eps[14:26, 10:16] = 12.0
    return jnp.asarray(eps)[None, None, :, :]


def _phase_shifter_cross_section(cell_um: float = 0.02, window=(6.0, 2.4), w_metal: float = 4.0):
    """The reference device of Jokisch et al. 2024, section 2: Si guide, SiO2 cladding, TiN heater.

    The same cross-section the mode-adjoint tests use, at the same reduced window.
    """
    nx = round(window[0] / cell_um)
    ny = round(window[1] / cell_um)
    x = -window[0] / 2 + (np.arange(nx) + 0.5) * cell_um
    y = -window[1] / 2 + (np.arange(ny) + 0.5) * cell_um
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    eps = np.full((nx, ny), N_SIO2**2, dtype=np.complex128)
    eps[(np.abs(grid_x) < 5.0) & (np.abs(grid_y) < 0.5)] = N_SI**2
    metal = (grid_x >= -5.0) & (grid_x <= -5.0 + w_metal) & (grid_y >= 0.5) & (grid_y <= 0.75)
    eps[metal] = EPS_TIN
    return jnp.asarray(eps)[None, :, :, None]


class TestTargetNeff:
    """``target_neff`` aims the shift-invert solve and, when given, the ordering."""

    def test_the_default_solve_is_unchanged(self, float64):
        """The regression bar for the whole change: no target, same number as before, to 1e-12.

        The recorded value is the 300 x 120 phase-shifter cross-section at 20 nm solved on
        ``feat/coupling-shared`` at 00e3195, before any of the track J phase 0 edits.
        """
        eps = _phase_shifter_cross_section()
        _E, _H, neff = compute_mode(
            frequency=FREQ,
            inv_permittivities=1.0 / eps,
            inv_permeabilities=1.0,
            resolution=20e-9,
            dtype=jnp.float64,
        )
        recorded = 3.4146019256819313 + 0.00010527080358643824j
        assert complex(neff).real == pytest.approx(recorded.real, abs=1e-12)
        assert complex(neff).imag == pytest.approx(recorded.imag, abs=1e-12)

    def test_it_selects_the_mode_nearest_the_target(self, float64):
        """Aiming at the third mode's index makes it mode_index=0."""
        eps = _strip()
        kwargs = {
            "frequency": FREQ,
            "inv_permittivities": 1.0 / eps,
            "inv_permeabilities": 1.0,
            "resolution": 40e-9,
            "dtype": jnp.float64,
        }
        _E, _H, neffs = compute_modes(num_modes=4, **kwargs)
        aim = float(np.real(neffs[2]))
        assert aim < float(np.real(neffs[0]))  # it is not the mode the default would return

        _E, _H, aimed = compute_mode(mode_index=0, target_neff=aim, **kwargs)
        assert complex(aimed).real == pytest.approx(aim, rel=1e-10)

    @patch("fdtdx.core.physics.modes.tidy3d_mode_computation_wrapper")
    @patch("fdtdx.core.physics.modes.normalize_by_poynting_flux")
    def test_it_reaches_the_backend(self, mock_normalize, mock_wrapper):
        """The value is forwarded, not swallowed by ``compute_mode``."""
        mock_wrapper.return_value = [
            ModeTupleType(
                neff=1.5 + 0.0j,
                **{f"{f}{a}": np.ones((4, 3), dtype=np.complex64) for f in "EH" for a in "xyz"},
            )
        ]
        mock_normalize.side_effect = lambda E, H, axis, area_weights=None: (E, H)

        compute_mode(
            frequency=FREQ,
            inv_permittivities=jnp.ones((1, 1, 4, 3)),
            inv_permeabilities=1.0,
            resolution=1e-8,
            mode_backend="tidy3d",
            target_neff=2.75,
        )
        assert mock_wrapper.call_args.kwargs["target_neff"] == 2.75

    def test_sort_modes_orders_by_distance_to_the_target(self):
        modes = [
            ModeTupleType(neff=n, Ex=None, Ey=None, Ez=None, Hx=None, Hy=None, Hz=None) for n in (3.0, 2.0, 1.2, 0.5)
        ]
        assert [m.neff for m in sort_modes(modes, None, (0, 1))] == [3.0, 2.0, 1.2, 0.5]
        assert [m.neff for m in sort_modes(modes, None, (0, 1), target_neff=1.9)] == [2.0, 1.2, 3.0, 0.5]


class TestComputeModes:
    """The plural entry point returns the list the backend already solved."""

    def test_it_reproduces_compute_mode_index_by_index(self, float64):
        eps = _strip()
        kwargs = {
            "frequency": FREQ,
            "inv_permittivities": 1.0 / eps,
            "inv_permeabilities": 1.0,
            "resolution": 40e-9,
            "dtype": jnp.float64,
        }
        fields_E, fields_H, neffs = compute_modes(num_modes=4, **kwargs)
        assert fields_E.shape == (4, 3, 1, 40, 30)
        assert fields_H.shape == (4, 3, 1, 40, 30)
        assert neffs.shape == (4,)

        for index in range(4):
            single_E, single_H, single_neff = compute_mode(mode_index=index, **kwargs)
            assert complex(single_neff) == pytest.approx(complex(neffs[index]), abs=1e-12)
            scale = float(np.max(np.abs(np.asarray(single_E))))
            assert np.max(np.abs(np.asarray(single_E) - np.asarray(fields_E[index]))) < 1e-10 * scale
            scale_h = float(np.max(np.abs(np.asarray(single_H))))
            assert np.max(np.abs(np.asarray(single_H) - np.asarray(fields_H[index]))) < 1e-10 * scale_h

    def test_it_costs_one_eigen_solve_whatever_the_count(self, float64):
        """The point of the entry point: N candidates, one solve."""
        import fdtdx.core.physics.mode_backend.solve as backend_solve

        eps = _strip()
        kwargs = {
            "frequency": FREQ,
            "inv_permittivities": 1.0 / eps,
            "inv_permeabilities": 1.0,
            "resolution": 40e-9,
            "dtype": jnp.float64,
        }
        original = backend_solve.spl.eigs
        calls = []
        try:
            backend_solve.spl.eigs = lambda *a, **kw: (calls.append(1), original(*a, **kw))[1]
            compute_modes(num_modes=6, **kwargs)
            assert len(calls) == 1
            calls.clear()
            for index in range(6):
                compute_mode(mode_index=index, **kwargs)
            assert len(calls) == 6
        finally:
            backend_solve.spl.eigs = original

    def test_it_refuses_more_modes_than_the_operator_has(self):
        with pytest.raises(ValueError, match="exceeds the"):
            compute_modes(
                frequency=FREQ,
                inv_permittivities=jnp.ones((1, 1, 3, 3)),
                inv_permeabilities=1.0,
                num_modes=100,
                resolution=40e-9,
            )
        with pytest.raises(ValueError, match="at least 1"):
            compute_modes(
                frequency=FREQ,
                inv_permittivities=jnp.ones((1, 1, 8, 8)),
                inv_permeabilities=1.0,
                num_modes=0,
                resolution=40e-9,
            )


def _analytic_slab_te0(
    n_core: float = N_SI,
    thickness: float = 0.5e-6,
    n_clad: float = N_SIO2,
    lam: float = LAM,
) -> float:
    """Effective index of the fundamental even TE mode of a symmetric slab, by bisection.

    Solves ``kappa tan(kappa d / 2) = gamma`` with ``kappa = k0 sqrt(n_core^2 - neff^2)`` and
    ``gamma = k0 sqrt(neff^2 - n_clad^2)`` on the first branch, ``kappa d / 2 < pi / 2``.
    """
    k0 = 2 * np.pi / lam

    def residual(neff: float) -> float:
        kappa = k0 * np.sqrt(n_core**2 - neff**2)
        gamma = k0 * np.sqrt(neff**2 - n_clad**2)
        return kappa * np.tan(kappa * thickness / 2) - gamma

    lo = max(n_clad + 1e-12, np.sqrt(n_core**2 - (np.pi / (k0 * thickness)) ** 2) + 1e-12)
    hi = n_core - 1e-12
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if residual(lo) * residual(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _slab_cross_section(cell: float, collapsed_axis: int, height: float = 4e-6, thickness: float = 0.5e-6):
    """A layered slab, invariant along one transverse axis, given to the 2-D collapse path.

    The invariant axis carries exactly two cells, which is how ``compute_mode`` detects a
    two-dimensional cross-section. Propagation is along the third (length-one) axis.
    """
    n = round(height / cell)
    n_core = round(thickness / cell)
    low = (n - n_core) // 2
    profile = np.full(n, N_SIO2**2)
    profile[low : low + n_core] = N_SI**2
    if collapsed_axis == 1:
        eps = np.broadcast_to(profile[:, None], (n, 2))
    else:
        eps = np.broadcast_to(profile[None, :], (2, n))
    return jnp.asarray(np.array(eps))[None, :, :, None]


class TestTwoDimensionalCollapse:
    """A transverse axis of exactly two cells: the slab path."""

    @pytest.mark.parametrize("collapsed_axis", [0, 1])
    def test_it_returns_the_declared_shape(self, float64, collapsed_axis):
        """It used to raise: the collapsed axis survived the slice and was expanded twice."""
        eps = _slab_cross_section(20e-9, collapsed_axis)
        mode_E, mode_H, neff = compute_mode(
            frequency=FREQ,
            inv_permittivities=1.0 / eps,
            inv_permeabilities=1.0,
            resolution=20e-9,
            dtype=jnp.float64,
        )
        assert mode_E.shape == (3, *eps.shape[1:])
        assert mode_H.shape == (3, *eps.shape[1:])
        # The mode is invariant along the collapsed axis, so the two cells carry the same field.
        first = np.take(np.asarray(mode_E), 0, axis=collapsed_axis + 1)
        second = np.take(np.asarray(mode_E), 1, axis=collapsed_axis + 1)
        assert np.array_equal(first, second)
        assert complex(neff).real > N_SIO2

    @pytest.mark.parametrize("collapsed_axis", [0, 1])
    def test_te0_matches_the_analytic_slab_dispersion_relation(self, float64, collapsed_axis):
        """J1 Slide 18 rung 1, on the collapsed path.

        The finite-difference operator is second order, so a single grid is accurate to ``C h^2``
        (2.1e-4 at 10 nm, 5.3e-5 at 5 nm on this slab): reaching 1e-6 on one grid would need a
        sub-nanometre cell. Two grids and one Richardson step remove the ``h^2`` term, which both
        pins the value to the analytic root and pins the convergence order - a collapse that
        silently changed the effective spacing would break the ratio, not just the value.
        """
        reference = _analytic_slab_te0()
        indices = {}
        for cell in (20e-9, 10e-9, 5e-9):
            eps = _slab_cross_section(cell, collapsed_axis)
            _E, _H, neff = compute_mode(
                frequency=FREQ,
                inv_permittivities=1.0 / eps,
                inv_permeabilities=1.0,
                resolution=cell,
                dtype=jnp.float64,
            )
            indices[cell] = complex(neff).real

        errors = {cell: value - reference for cell, value in indices.items()}
        # second order: the error quarters with each halving of the cell
        assert errors[20e-9] / errors[10e-9] == pytest.approx(4.0, rel=0.05)
        assert errors[10e-9] / errors[5e-9] == pytest.approx(4.0, rel=0.05)

        richardson = (4 * indices[5e-9] - indices[10e-9]) / 3
        assert richardson == pytest.approx(reference, abs=1e-6)


class TestModeSolvePrecision:
    """The mode solve is double precision whatever the simulation runs at."""

    def test_the_operator_is_complex128_from_single_precision_material(self):
        from fdtdx.core.physics.mode_backend.operator import build_derivative_matrices
        from fdtdx.core.physics.mode_backend.solve import assemble_mode_operator

        coords = np.arange(5) * 40e-9
        der_mats = build_derivative_matrices(coords, coords)
        single = np.full(16, 2.25, dtype=np.float32)
        operator = assemble_mode_operator(single, single, single, single, single, single, der_mats, k0=2 * np.pi / LAM)
        assert operator.mat.dtype == np.complex128
        assert operator.qmat.dtype == np.complex128
        assert operator.q_ep.dtype == np.complex128

    def test_a_float32_simulation_gets_the_same_index_as_a_float64_one(self, float64):
        """Every permittivity here is exact in float32, so only the solve's own precision differs."""
        from fdtdx.core.physics.mode_backend import fdtdmex_mode_computation_wrapper

        eps = np.asarray(_strip()[0, 0])  # values 2.25 and 12.0, both exact in float32
        coords_x = np.arange(eps.shape[0] + 1) * 0.04
        coords_y = np.arange(eps.shape[1] + 1) * 0.04

        def solve(dtype):
            modes = fdtdmex_mode_computation_wrapper(
                frequency=FREQ,
                permittivity_cross_section=eps.astype(dtype)[None],
                coords=[coords_x, coords_y],
                direction="+",
                num_modes=4,
            )
            return complex(modes[0].neff)

        assert solve(np.complex64) == solve(np.complex128)
