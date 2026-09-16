"""The PML-overlap alert: an error for a tensorial cross-section, a warning for a diagonal one."""

import numpy as np
import pytest

from fdtdx.core.physics.mode_pml import (
    ModePlanePmlOverlapWarning,
    check_mode_plane_pml_overlap,
    cross_section_is_tensorial,
    pml_boxes_from_boundary_config,
    pml_overlap_cells,
)
from fdtdx.objects.boundaries.initialization import BoundaryConfig

VOLUME = (60, 40, 30)


def _boxes(**overrides):
    config = BoundaryConfig(**overrides)
    return pml_boxes_from_boundary_config(config, VOLUME)


class TestPmlBoxes:
    def test_default_config_gives_six_boxes_of_ten_cells(self):
        boxes = _boxes()
        assert len(boxes) == 6
        names = {name for name, _ in boxes}
        assert names == {"min_x", "max_x", "min_y", "max_y", "min_z", "max_z"}
        by_name = dict(boxes)
        assert by_name["min_x"][0] == (0, 10)
        assert by_name["max_x"][0] == (50, 60)
        assert by_name["min_z"][2] == (0, 10)
        assert by_name["max_z"][2] == (20, 30)

    def test_periodic_and_zero_thickness_sides_are_not_pml(self):
        boxes = _boxes(boundary_type_minx="periodic", thickness_grid_maxy=0)
        names = {name for name, _ in boxes}
        assert "min_x" not in names
        assert "max_y" not in names
        assert "max_x" in names


class TestOverlap:
    def test_a_plane_well_inside_the_domain_does_not_overlap(self):
        plane = ((30, 31), (10, 30), (10, 20))
        assert pml_overlap_cells(plane, _boxes()) == {}

    def test_a_transverse_span_over_the_full_domain_overlaps_four_sides(self):
        """The common case: a mode plane spanning the whole cross-section reaches every side PML."""
        plane = ((30, 31), (0, 40), (0, 30))
        overlaps = pml_overlap_cells(plane, _boxes())
        assert set(overlaps) == {"min_y", "max_y", "min_z", "max_z"}
        assert overlaps["min_y"] == 1 * 10 * 30

    def test_a_plane_inside_the_propagation_pml_overlaps_that_side(self):
        plane = ((5, 6), (12, 28), (12, 18))
        overlaps = pml_overlap_cells(plane, _boxes())
        assert set(overlaps) == {"min_x"}
        assert overlaps["min_x"] == 1 * 16 * 6


class TestTierDetection:
    def test_isotropic_and_diagonal_are_not_tensorial(self):
        assert not cross_section_is_tensorial(np.ones((1, 4, 4, 1)))
        assert not cross_section_is_tensorial(np.ones((3, 4, 4, 1)))

    def test_a_nine_component_diagonal_tensor_is_not_tensorial(self):
        array = np.zeros((9, 4, 4, 1))
        array[0] = array[4] = array[8] = 2.25
        assert not cross_section_is_tensorial(array)

    def test_an_off_diagonal_entry_makes_it_tensorial(self):
        array = np.zeros((9, 4, 4, 1))
        array[0] = array[4] = array[8] = 2.25
        array[1] = array[3] = 0.1
        assert cross_section_is_tensorial(array)


class TestAlert:
    def test_diagonal_tier_warns_and_names_the_overlap_in_cells(self):
        plane = ((30, 31), (0, 40), (0, 30))
        with pytest.warns(ModePlanePmlOverlapWarning, match="min_y: 300 cells"):
            overlaps = check_mode_plane_pml_overlap(
                grid_slice_tuple=plane,
                pml_boxes=_boxes(),
                tensorial=False,
                object_name="bus mode source",
            )
        assert overlaps["min_y"] == 300

    def test_tensorial_tier_raises(self):
        plane = ((30, 31), (0, 40), (0, 30))
        with pytest.raises(ValueError, match="fully tensorial"):
            check_mode_plane_pml_overlap(
                grid_slice_tuple=plane,
                pml_boxes=_boxes(),
                tensorial=True,
                object_name="bus mode source",
            )

    def test_no_overlap_is_silent_for_both_tiers(self, recwarn):
        plane = ((30, 31), (12, 28), (12, 18))
        for tensorial in (False, True):
            assert (
                check_mode_plane_pml_overlap(
                    grid_slice_tuple=plane,
                    pml_boxes=_boxes(),
                    tensorial=tensorial,
                    object_name="bus mode source",
                )
                == {}
            )
        assert [w for w in recwarn if issubclass(w.category, ModePlanePmlOverlapWarning)] == []

    def test_one_warning_per_call_however_many_sides_overlap(self):
        plane = ((30, 31), (0, 40), (0, 30))
        with pytest.warns(ModePlanePmlOverlapWarning) as record:
            check_mode_plane_pml_overlap(
                grid_slice_tuple=plane, pml_boxes=_boxes(), tensorial=False, object_name="source"
            )
        overlap_warnings = [w for w in record if issubclass(w.category, ModePlanePmlOverlapWarning)]
        assert len(overlap_warnings) == 1
        assert "min_z" in str(overlap_warnings[0].message)
        assert "max_z" in str(overlap_warnings[0].message)
