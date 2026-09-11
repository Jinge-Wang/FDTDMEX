"""Component-frame maps: sampled vector and tensor values must turn with the positions.

``PointTransform.apply`` moves a Yee point into the mesh frame. A scalar sample needs nothing more,
but a vector or a tensor sampled on that mesh carries its components in the mesh frame, and feeding
those straight into a response is a silent correctness bug — the arrays have the right shape and the
wrong axes. These tests pin the one rule that prevents it: the component map and the position map
are built from the same ``permute``, so a value's component along mesh axis ``i`` lands on Yee axis
``permute[i]``, and a two-dimensional mesh's components are lifted to three with the remaining Yee
axis zero.

No DOLFINx here: the transforms are arithmetic. The end-to-end check against a real sampled
finite-element field is in ``test_symmetric_gradient.py``.
"""

import numpy as np
import pytest

from fdtdx.coupling import PointTransform, RadialPlaneTransform, YeeLatticeSamples

CROSS_SECTION = PointTransform(collapse_axes=(1,), permute=(0, 2, 1))


def test_the_component_matrix_is_the_position_permutation_read_the_other_way():
    matrix = CROSS_SECTION.component_matrix()
    # mesh 0 -> Yee x, mesh 1 -> Yee z, mesh 2 (out of plane) -> Yee y.
    np.testing.assert_array_equal(matrix, np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))
    # A pure permutation is orthogonal, so the position map is its transpose.
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    np.testing.assert_allclose(PointTransform(permute=(0, 2, 1)).apply(points), points @ matrix, atol=0.0)


def test_positions_and_rank_one_values_transform_together():
    transform = PointTransform(permute=(1, 2, 0))
    matrix = transform.component_matrix()
    rng = np.random.default_rng(0)
    operator = rng.normal(size=(3, 3))
    yee_points = rng.normal(size=(200, 3))
    mesh_points = transform.apply(yee_points)
    # A field the grid knows as v(x) = A x; the mesh sees the same field in its own frame.
    grid_values = yee_points @ operator.T
    mesh_values = np.einsum("ai,na->ni", matrix, grid_values)
    np.testing.assert_allclose(transform.apply_values(mesh_values, rank=1), grid_values, rtol=0.0, atol=1e-14)
    # and the mesh point really is the same point, which is what makes the pair consistent.
    np.testing.assert_allclose(np.einsum("ia,na->ni", matrix, mesh_points), yee_points, atol=1e-14)


def test_a_rank_two_value_is_conjugated_by_the_same_matrix():
    rng = np.random.default_rng(1)
    mesh_tensors = rng.normal(size=(50, 3, 3))
    mesh_tensors = 0.5 * (mesh_tensors + np.swapaxes(mesh_tensors, -1, -2))
    matrix = CROSS_SECTION.component_matrix()
    expected = np.einsum("ia,nab,jb->nij", matrix, mesh_tensors, matrix)
    np.testing.assert_allclose(CROSS_SECTION.apply_values(mesh_tensors, rank=2), expected, rtol=0.0, atol=0.0)
    # the flattened form FemField.evaluate returns is the same answer
    np.testing.assert_allclose(
        CROSS_SECTION.apply_values(mesh_tensors.reshape(-1, 9), rank=2), expected, rtol=0.0, atol=0.0
    )


def test_a_two_dimensional_mesh_is_lifted_with_zero_on_the_collapsed_axis():
    values = np.array([[1.0, 2.0], [-3.0, 4.0]])
    lifted = CROSS_SECTION.apply_values(values, rank=1)
    # mesh 0 -> x, mesh 1 -> z, and the collapsed propagation axis y gets nothing.
    np.testing.assert_array_equal(lifted, np.array([[1.0, 0.0, 2.0], [-3.0, 0.0, 4.0]]))

    tensor = np.array([[[1.0, 2.0], [2.0, 3.0]]])
    lifted_tensor = CROSS_SECTION.apply_values(tensor, rank=2)
    np.testing.assert_array_equal(lifted_tensor[0], np.array([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [2.0, 0.0, 3.0]]))
    np.testing.assert_array_equal(CROSS_SECTION.apply_values(tensor.reshape(-1, 4), rank=2), lifted_tensor)


def test_a_pure_scale_and_offset_leave_every_component_alone():
    transform = PointTransform(scale=1e6, offset=(1.0, -2.0, 3.0), collapse_axes=(1,))
    values = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(transform.apply_values(values, rank=1), values)
    tensor = np.arange(9.0).reshape(1, 3, 3)
    np.testing.assert_array_equal(transform.apply_values(tensor, rank=2), tensor)
    np.testing.assert_array_equal(transform.apply_values(values, rank=0), values)


def test_a_negative_scale_mirrors_a_vector_and_leaves_a_tensor_alone():
    transform = PointTransform(scale=-1.0)
    values = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(transform.apply_values(values, rank=1), -values)
    tensor = np.arange(9.0).reshape(1, 3, 3)
    np.testing.assert_array_equal(transform.apply_values(tensor, rank=2), tensor)


def test_an_axial_vector_picks_up_the_determinant_of_an_odd_permutation():
    values = np.array([[1.0, 2.0, 3.0]])
    polar = CROSS_SECTION.apply_values(values, rank=1)
    axial = CROSS_SECTION.apply_values(values, rank=1, pseudo=True)
    assert np.linalg.det(CROSS_SECTION.component_matrix()) == pytest.approx(-1.0)
    np.testing.assert_array_equal(axial, -polar)


def test_bad_requests_are_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="2 or 3 components"):
        CROSS_SECTION.apply_values(np.zeros((4, 5)), rank=1)
    with pytest.raises(ValueError, match="rank must be"):
        CROSS_SECTION.apply_values(np.zeros((4, 3)), rank=3)
    with pytest.raises(ValueError, match=r"\(\.\.\., m, m\)"):
        CROSS_SECTION.apply_values(np.zeros((4, 5)), rank=2)
    with pytest.raises(NotImplementedError):
        CROSS_SECTION.apply_values(np.zeros((4, 3, 3)), rank=2, pseudo=True)
    with pytest.raises(ValueError, match="permute must reorder"):
        PointTransform(permute=(0, 0, 1)).component_matrix()


def test_a_radial_transform_turns_a_vector_with_the_azimuth():
    transform = RadialPlaneTransform(center=(1.0, 1.0), z_plane=0.0)
    points = np.array(
        [
            [2.0, 1.0, 0.0],  # due +x from the axis
            [1.0, 2.0, 0.0],  # due +y
            [2.0, 2.0, 0.0],  # 45 degrees
            [1.0, 1.0, 0.0],  # on the axis
        ]
    )
    radial = np.tile(np.array([1.0, 0.0]), (4, 1))
    out = transform.apply_values(radial, rank=1, points=points)
    root = 1.0 / np.sqrt(2.0)
    np.testing.assert_allclose(
        out,
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [root, root, 0.0], [1.0, 0.0, 0.0]]),
        atol=1e-15,
    )
    axial = np.tile(np.array([0.0, 1.0]), (4, 1))
    np.testing.assert_allclose(transform.apply_values(axial, rank=1, points=points), np.tile([0.0, 0.0, 1.0], (4, 1)))
    with pytest.raises(ValueError, match="needs the Yee points"):
        transform.apply_values(radial, rank=1)


def test_a_radial_rank_two_value_is_conjugated_by_the_local_frame():
    transform = RadialPlaneTransform(center=(0.0, 0.0))
    points = np.array([[1.0, 1.0, 0.0]])
    # A purely radial normal stress of 5 at 45 degrees splits equally over xx, yy and xy.
    tensor = np.array([[[5.0, 0.0], [0.0, 0.0]]])
    out = transform.apply_values(tensor, rank=2, points=points)
    np.testing.assert_allclose(out[0], np.array([[2.5, 2.5, 0.0], [2.5, 2.5, 0.0], [0.0, 0.0, 0.0]]), atol=1e-15)
    frames = transform.component_matrix(points)
    np.testing.assert_allclose(frames[0] @ frames[0].T, np.eye(3), atol=1e-15)


def test_the_unit_label_may_state_dimensionless_and_survives_a_round_trip(tmp_path):
    edges = (np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3))
    samples = YeeLatticeSamples(
        edges=edges,
        values={"E0": np.zeros((2, 2, 2, 6))},
        covered={"E0": np.ones((2, 2, 2), dtype=bool)},
        name="S",
        unit=None,
    )
    assert samples.unit is None
    reloaded = YeeLatticeSamples.load(samples.save(tmp_path / "strain.npz"))
    assert reloaded.unit is None
    assert reloaded.name == "S"
