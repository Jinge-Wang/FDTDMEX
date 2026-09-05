"""Regression tests for the Gaussian plane-source profile on NON-square planes.

``GaussianPlaneSource._gauss_profile`` builds a 2D index grid that must be in the
same ``(horizontal, vertical)`` order as ``center`` and ``radii``. An xy-indexed
meshgrid over ``(height, width)`` returns an array of the right *shape* but with
the two coordinates swapped, so on a square plane nothing is visibly wrong while
on a non-square plane the Gaussian spot lands at the wrong cell -- and once the
misplaced center is more than one radius off the plane, the profile is all zeros
and the normalization by ``profile.sum()`` yields NaN.

Upstream fdtdx fix: commit f4e610c (PR #418).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.objects.sources.linear_polarization import GaussianPlaneSource

# (width, height) pairs: both orderings, so a swap cannot pass by symmetry.
_NON_SQUARE_PLANES = [(30, 12), (12, 30)]


def _profile_2d(width: int, height: int, radii, std: float = 1 / 3):
    """Profile with the spot requested at the plane's center cell, squeezed to 2D.

    The center is an exact cell index (not a half-cell) so ``argmax`` has a
    single unambiguous winner.
    """
    center = (float(width // 2), float(height // 2))
    profile = GaussianPlaneSource._gauss_profile(
        width=width,
        height=height,
        axis=2,
        center=center,
        radii=radii,
        std=std,
    )
    assert profile.shape == (width, height, 1)
    return np.asarray(profile[:, :, 0]), center


@pytest.mark.parametrize(("width", "height"), _NON_SQUARE_PLANES)
def test_gauss_profile_argmax_at_requested_center(width, height):
    """The peak must sit at the requested (horizontal, vertical) center cell."""
    radius = 12.5  # the MRM case: 0.5 um radius on a 40 nm grid
    profile, center = _profile_2d(width, height, (radius, radius))

    assert np.all(np.isfinite(profile)), "profile contains NaN/Inf"
    assert profile.sum() > 0, "profile is all zeros"

    peak = np.unravel_index(int(np.argmax(profile)), profile.shape)
    expected = (int(center[0]), int(center[1]))
    assert peak == expected, f"peak at {peak}, expected {expected} on a {width}x{height} plane"


@pytest.mark.parametrize(("width", "height"), _NON_SQUARE_PLANES)
def test_gauss_profile_not_all_zeros_when_radius_smaller_than_long_side(width, height):
    """A radius that fits the short side must still produce a finite, non-empty spot.

    With the coordinates swapped the center lands off the plane by more than a
    radius here, so the masked profile is empty and ``profile / profile.sum()``
    is NaN everywhere.
    """
    radius = 3.0  # short-side half-extent is 6, long-side offset after a swap is 15
    profile, _ = _profile_2d(width, height, (radius, radius))

    assert np.all(np.isfinite(profile)), "profile contains NaN/Inf (empty mask before normalization)"
    assert profile.sum() > 0, "profile is all zeros"
    assert jnp.isclose(profile.sum(), 1.0, atol=1e-5)


@pytest.mark.parametrize(("width", "height"), _NON_SQUARE_PLANES)
def test_gauss_profile_transposes_with_plane(width, height):
    """Swapping (width, height) must transpose the profile, not reshape it."""
    radius = 12.5
    a, _ = _profile_2d(width, height, (radius, radius))
    b, _ = _profile_2d(height, width, (radius, radius))
    np.testing.assert_allclose(a, b.T, rtol=0, atol=1e-6)


def test_gauss_profile_anisotropic_radii_follow_their_axis():
    """radii[0] scales the horizontal (width) axis, radii[1] the vertical one."""
    width, height = 30, 12
    profile, center = _profile_2d(width, height, (12.0, 3.0))

    h_extent = int((profile.sum(axis=1) > 0).sum())
    v_extent = int((profile.sum(axis=0) > 0).sum())
    assert h_extent > v_extent, f"horizontal extent {h_extent} should exceed vertical extent {v_extent}"
    peak = np.unravel_index(int(np.argmax(profile)), profile.shape)
    assert peak == (int(center[0]), int(center[1]))
