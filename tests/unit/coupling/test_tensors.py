"""Voigt assembly, frame permutation and the stress-optic conversion.

Three things are pinned here. The engineering-shear factor of two, because a photoelastic matrix is
tabulated against it and nothing downstream flags a missing factor. The frame covariance of a
permuted response, because a crystal cut is applied to the response matrix and to the material's own
tensor and getting only one of them right is silent. And the closed-form numbers a photoelastic case
is graded against: the bulk index change of strained silicon, and the ``(B1, B2) -> (p11, p12)``
conversion of the COMSOL stress-optical waveguide model's silica constants.

No DOLFINx and no JAX: this is arithmetic.
"""

import numpy as np
import pytest

from fdtdx.coupling import PhotoelasticResponse, PointTransform, YeeLatticeSamples
from fdtdx.coupling.tensors import (
    VOIGT_LABELS,
    VOIGT_ORDER,
    cubic_photoelastic_matrix,
    isotropic_photoelastic_matrix,
    permute_tensor,
    photoelastic_from_stress_optic,
    stress_optic_from_photoelastic,
    tensor_from_voigt,
    voigt_from_tensor,
    voigt_index_permutation,
    voigt_permute,
    voigt_samples_from_tensor,
)

# Silicon, S1 section 4: p11, p12, p44 and the unstressed index at 1.55 um.
SI_P11, SI_P12, SI_P44 = -0.094, 0.017, -0.051
SI_N0 = 3.4777
# The uniform plane strain of the 4 x 2 um strip under 1 MPa, already in the grid frame.
STRAIN_XX = 8.762727272727273e-06
STRAIN_ZZ = -2.0554545454545456e-06

# Fused silica in the COMSOL stress-optical model: B1, B2 in m^2/N, n, E in Pa, nu.
SILICA_B1, SILICA_B2 = 0.65e-12, 4.2e-12
SILICA_N, SILICA_E, SILICA_NU = 1.445, 78e9, 0.17


def _random_symmetric(rng, count=32):
    t = rng.normal(size=(count, 3, 3))
    return 0.5 * (t + np.swapaxes(t, -1, -2))


def test_the_voigt_order_is_the_one_the_responses_read():
    assert VOIGT_ORDER == ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    assert VOIGT_LABELS == ("xx", "yy", "zz", "yz", "xz", "xy")


def test_the_engineering_shear_is_exactly_twice_the_tensor_component():
    tensor = np.zeros((3, 3))
    tensor[0, 1] = tensor[1, 0] = 3.0
    np.testing.assert_array_equal(voigt_from_tensor(tensor, engineering=True)[5], 6.0)
    np.testing.assert_array_equal(voigt_from_tensor(tensor, engineering=False)[5], 3.0)


def test_assembly_and_expansion_round_trip_to_machine_precision():
    rng = np.random.default_rng(3)
    tensors = _random_symmetric(rng)
    for engineering in (True, False):
        vectors = voigt_from_tensor(tensors, engineering=engineering)
        back = tensor_from_voigt(vectors, engineering=engineering)
        np.testing.assert_allclose(back, tensors, rtol=0.0, atol=1e-15)


def test_a_two_dimensional_tensor_is_read_as_plane_strain():
    tensor = np.array([[1.0, 2.0], [2.0, 3.0]])
    vector = voigt_from_tensor(tensor)
    np.testing.assert_array_equal(vector, np.array([1.0, 3.0, 0.0, 0.0, 0.0, 4.0]))
    # the flattened form a tensor-valued FemField hands over is the same answer
    np.testing.assert_array_equal(voigt_from_tensor(tensor.reshape(4)), vector)


def test_permuting_the_axes_and_contracting_commute():
    rng = np.random.default_rng(4)
    tensors = _random_symmetric(rng)
    for perm, signs in (((0, 1, 2), None), ((0, 2, 1), None), ((2, 0, 1), None), ((1, 0, 2), (1.0, -1.0, 1.0))):
        index, sign = voigt_index_permutation(perm, signs)
        for engineering in (True, False):
            direct = voigt_from_tensor(permute_tensor(tensors, perm, signs), engineering=engineering)
            through_voigt = sign * voigt_from_tensor(tensors, engineering=engineering)[..., index]
            np.testing.assert_allclose(direct, through_voigt, rtol=0.0, atol=1e-15)


def test_the_identity_permutation_changes_nothing():
    rng = np.random.default_rng(5)
    for shape in ((6,), (6, 3), (6, 6)):
        original = rng.normal(size=shape)
        np.testing.assert_array_equal(voigt_permute(original, (0, 1, 2)), original)
    index, sign = voigt_index_permutation((0, 1, 2))
    np.testing.assert_array_equal(index, np.arange(6))
    np.testing.assert_array_equal(sign, np.ones(6))


def test_permuting_a_matrix_twice_returns_it():
    rng = np.random.default_rng(6)
    perm = (2, 0, 1)
    inverse = tuple(int(np.argsort(perm)[i]) for i in range(3))
    for shape in ((6,), (6, 3), (6, 6)):
        original = rng.normal(size=shape)
        np.testing.assert_allclose(
            voigt_permute(voigt_permute(original, perm), inverse), original, rtol=0.0, atol=1e-15
        )


def test_a_cubic_matrix_keeps_its_form_under_any_axis_permutation():
    p = cubic_photoelastic_matrix(SI_P11, SI_P12, SI_P44)
    for perm in ((0, 2, 1), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        np.testing.assert_allclose(voigt_permute(p, perm), p, rtol=0.0, atol=1e-15)


def test_a_permuted_response_gives_the_permuted_tensor():
    rng = np.random.default_rng(7)
    p = rng.normal(size=(6, 6)) * 1e-2
    p = 0.5 * (p + p.T)
    base = np.diag([4.0, 4.4, 5.1])
    strain = _random_symmetric(rng, count=1)[0] * 1e-5
    for perm, signs in (((0, 2, 1), None), ((2, 0, 1), None), ((1, 0, 2), (1.0, -1.0, 1.0))):
        old = PhotoelasticResponse(p=p).tensor(base, {"S": voigt_from_tensor(strain)[None]})[0]
        new = PhotoelasticResponse(p=voigt_permute(p, perm, signs)).tensor(
            permute_tensor(base, perm, signs),
            {"S": voigt_from_tensor(permute_tensor(strain, perm, signs))[None]},
        )[0]
        np.testing.assert_allclose(new, permute_tensor(old, perm, signs), rtol=1e-12, atol=1e-14)


def test_the_bulk_closed_form_of_strained_silicon():
    """S1 section 4: uniform plane strain in silicon, index change against the exact expression."""
    p = cubic_photoelastic_matrix(SI_P11, SI_P12, SI_P44)
    strain = np.diag([STRAIN_XX, 0.0, STRAIN_ZZ])
    S = voigt_from_tensor(strain)
    np.testing.assert_allclose(S, np.array([STRAIN_XX, 0.0, STRAIN_ZZ, 0.0, 0.0, 0.0]))

    delta = p @ S
    np.testing.assert_allclose(delta[:3], np.array([-8.58639091e-07, 1.14023636e-07, 3.42179091e-07]), rtol=1e-8)
    np.testing.assert_array_equal(delta[3:], np.zeros(3))

    tensors = PhotoelasticResponse(p=p).tensor(np.eye(3) * SI_N0**2, {"S": S[None]})
    n_from_response = np.sqrt(np.diag(tensors[0]))
    n_closed_form = 1.0 / np.sqrt(1.0 / SI_N0**2 + delta[:3])
    np.testing.assert_allclose(n_from_response, n_closed_form, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(n_from_response, np.array([3.47771806, 3.47769760, 3.47769280]), atol=5e-9)
    assert n_from_response[0] - n_from_response[2] == pytest.approx(2.5253736e-05, rel=1e-6)


def test_a_shear_strain_makes_an_off_diagonal_permittivity():
    """The entry the 3-component tier cannot hold; the acceptance fixture for the tensor policy."""
    p = cubic_photoelastic_matrix(SI_P11, SI_P12, SI_P44)
    strain = np.array([[STRAIN_XX, 0.0, 1.09e-06], [0.0, 0.0, 0.0], [1.09e-06, 0.0, STRAIN_ZZ]])
    S = voigt_from_tensor(strain)
    assert S[4] == pytest.approx(2.18e-06)  # engineering shear, twice the tensor entry
    tensor = PhotoelasticResponse(p=p).tensor(np.eye(3) * SI_N0**2, {"S": S[None]})[0]
    assert tensor[0, 2] == pytest.approx(1.6264e-05, rel=1e-3)
    assert tensor[0, 2] == pytest.approx(tensor[2, 0], rel=0.0, abs=1e-18)
    assert np.linalg.eigvalsh(tensor)[0] == pytest.approx(12.0943, abs=1e-3)


def test_the_silica_stress_optic_pair_converts_to_the_published_photoelastic_pair():
    """S1 section 7c: COMSOL's (B1, B2) for fused silica become (p11, p12) exactly."""
    p = photoelastic_from_stress_optic(SILICA_B1, SILICA_B2, SILICA_N, SILICA_E, SILICA_NU)
    assert p.shape == (6, 6)
    assert p[0, 0] == pytest.approx(0.1317, abs=5e-5)
    assert p[0, 1] == pytest.approx(0.2886, abs=5e-5)
    assert p[3, 3] == pytest.approx(0.5 * (p[0, 0] - p[0, 1]), rel=0.0, abs=1e-15)
    back = stress_optic_from_photoelastic(p[0, 0], p[0, 1], SILICA_N, SILICA_E, SILICA_NU)
    np.testing.assert_allclose(back, (SILICA_B1, SILICA_B2), rtol=1e-14)


def test_the_conversion_is_refused_where_it_is_singular():
    with pytest.raises(ValueError, match="singular"):
        photoelastic_from_stress_optic(SILICA_B1, SILICA_B2, SILICA_N, SILICA_E, 0.5)


def test_an_isotropic_matrix_is_invariant_under_any_permutation():
    p = isotropic_photoelastic_matrix(0.121, 0.270)
    for perm in ((0, 2, 1), (2, 0, 1), (2, 1, 0)):
        np.testing.assert_allclose(voigt_permute(p, perm), p, rtol=0.0, atol=1e-16)


def test_sampled_tensors_become_voigt_samples_in_the_grid_frame():
    edges = (np.linspace(0.0, 3e-6, 4), np.linspace(0.0, 1e-6, 2), np.linspace(0.0, 2e-6, 3))
    shape = (3, 1, 2)
    mesh_strain = np.array([[STRAIN_XX, 1.09e-06], [1.09e-06, STRAIN_ZZ]])
    values = np.broadcast_to(mesh_strain, (*shape, 2, 2)).copy()
    samples = YeeLatticeSamples(
        edges=edges,
        values={"E0": values},
        covered={"E0": np.ones(shape, dtype=bool)},
        name="eps",
        unit=None,
    )
    out = voigt_samples_from_tensor(samples, PointTransform(collapse_axes=(1,), permute=(0, 2, 1)))
    assert out.name == "S"
    assert out.unit is None
    assert out.values["E0"].shape == (*shape, 6)
    expected = np.array([STRAIN_XX, 0.0, STRAIN_ZZ, 0.0, 2.18e-06, 0.0])
    np.testing.assert_allclose(out.values["E0"].reshape(-1, 6), np.broadcast_to(expected, (6, 6)), atol=1e-18)
    assert out.covered["E0"].all()
    assert out.provenance["voigt"]["engineering_shear"] is True


def test_bad_shapes_are_refused():
    with pytest.raises(ValueError, match="6 entries"):
        tensor_from_voigt(np.zeros(5))
    with pytest.raises(ValueError, match=r"\(6,\), \(6, 3\) and \(6, 6\)"):
        voigt_permute(np.zeros((3, 3)), (0, 1, 2))
    with pytest.raises(ValueError, match="perm must reorder"):
        voigt_permute(np.zeros((6, 6)), (0, 1, 1))
    with pytest.raises(ValueError, match="signs"):
        voigt_permute(np.zeros((6, 6)), (0, 1, 2), (1.0, 2.0, 1.0))
    with pytest.raises(ValueError, match=r"\(\.\.\., 3, 3\)"):
        permute_tensor(np.zeros((2, 2)), (0, 1, 2))
