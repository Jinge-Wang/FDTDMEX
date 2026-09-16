"""Moving a polygon's vertices: what is a change of shape and what is a translation.

An ``ExtrudedPolygon`` keeps its vertices centred on their own bounding box and derives its
cross-sectional size from it, so a bodily translation has nowhere to live in the vertex array. These
tests pin the split: a rigid translation comes back as a translation and leaves the shape alone, and
a uniform dilation comes back as a larger shape with no translation.
"""

import numpy as np
import pytest

from fdtdx.coupling import displace_polygon
from fdtdx.materials import Material
from fdtdx.objects.static_material.polygon import ExtrudedPolygon


def _rectangle(width: float, height: float) -> np.ndarray:
    return np.array(
        [
            [-0.5 * width, -0.5 * height],
            [0.5 * width, -0.5 * height],
            [0.5 * width, 0.5 * height],
            [-0.5 * width, 0.5 * height],
        ]
    )


def _polygon(axis: int = 1) -> ExtrudedPolygon:
    grid_shape: list[int | None] = [None, None, None]
    grid_shape[axis] = 8  # only the extrusion axis may be sized explicitly
    return ExtrudedPolygon(
        axis=axis,
        vertices=_rectangle(500e-9, 220e-9),
        material_name="core",
        materials={"core": Material(permittivity=12.25)},
        partial_grid_shape=tuple(grid_shape),
        name="wire",
    )


def test_a_rigid_translation_comes_back_as_a_translation():
    obj = _polygon(axis=1)
    shift = np.array([30e-9, -12e-9])
    moved, translation = displace_polygon(obj, obj.vertices + shift)
    np.testing.assert_allclose(np.asarray(moved.vertices), np.asarray(obj.vertices), atol=1e-21)
    # axis 1 is the extrusion axis, so the polygon plane is (x, z)
    np.testing.assert_allclose(translation, np.array([shift[0], 0.0, shift[1]]), atol=1e-21)
    assert moved.partial_real_shape[0] == pytest.approx(obj.partial_real_shape[0])
    assert moved.partial_real_shape[2] == pytest.approx(obj.partial_real_shape[2])


def test_a_uniform_dilation_changes_the_size_and_translates_nothing():
    obj = _polygon(axis=1)
    stretch = 1.1
    moved, translation = displace_polygon(obj, lambda v: v * stretch)
    np.testing.assert_allclose(np.asarray(moved.vertices), stretch * np.asarray(obj.vertices), rtol=1e-14)
    np.testing.assert_allclose(translation, 0.0, atol=1e-21)
    assert moved.partial_real_shape[0] == pytest.approx(stretch * 500e-9)
    assert moved.partial_real_shape[2] == pytest.approx(stretch * 220e-9)


def test_a_one_sided_stretch_is_half_shape_and_half_translation():
    """Only the +x wall moves: the polygon grows by the full step and its centre by half of it."""
    obj = _polygon(axis=2)
    step = 40e-9
    vertices = np.asarray(obj.vertices).copy()
    vertices[vertices[:, 0] > 0, 0] += step
    moved, translation = displace_polygon(obj, vertices)
    assert moved.partial_real_shape[0] == pytest.approx(500e-9 + step)
    np.testing.assert_allclose(translation, np.array([0.5 * step, 0.0, 0.0]), atol=1e-21)
    assert moved.partial_real_shape[1] == pytest.approx(220e-9)


def test_bad_maps_are_refused():
    obj = _polygon()
    with pytest.raises(ValueError, match="the polygon has"):
        displace_polygon(obj, np.zeros((3, 2)))
    with pytest.raises(ValueError, match="non-finite"):
        displace_polygon(obj, np.full((4, 2), np.nan))
    with pytest.raises(ValueError, match="no extent"):
        displace_polygon(obj, lambda v: v * np.array([1.0, 0.0]))
    with pytest.raises(TypeError, match="ExtrudedPolygon"):
        displace_polygon(object(), np.zeros((4, 2)))
