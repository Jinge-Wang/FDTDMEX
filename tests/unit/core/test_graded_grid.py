"""Unit tests for the graded-grid policy and its mesh override regions."""

import numpy as np
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import GradedGrid, RectilinearGrid, RefinementRegion, UniformGrid

# Edge arrays are stored in the solver dtype (float32 by default), so a width that is mathematically
# exact comes back with a few parts per million of rounding. Every comparison below allows for that.
_FLOAT32_SLACK = 1e-4


def _widths(grid: RectilinearGrid, axis: int) -> np.ndarray:
    return np.asarray(grid.cell_widths(axis), dtype=np.float64)


def _edges(grid: RectilinearGrid, axis: int) -> np.ndarray:
    return np.asarray(grid.edges(axis), dtype=np.float64)


def _max_neighbour_ratio(widths: np.ndarray) -> float:
    return float(np.maximum(widths[1:] / widths[:-1], widths[:-1] / widths[1:]).max())


class TestRefinementRegion:
    """The region is a physical-coordinate box with a target width."""

    def test_scalar_spacing_applies_to_every_axis(self):
        region = RefinementRegion(spacing=10e-9, z=(-1e-6, 1e-6))
        assert [region.axis_spacing(axis) for axis in range(3)] == [10e-9] * 3

    def test_per_axis_spacing_is_kept_per_axis(self):
        region = RefinementRegion(spacing=(10e-9, 20e-9, 30e-9))
        assert [region.axis_spacing(axis) for axis in range(3)] == [10e-9, 20e-9, 30e-9]

    def test_axis_bounds_are_none_for_a_whole_axis(self):
        region = RefinementRegion(spacing=10e-9, z=(-1e-6, 2e-6))
        assert region.axis_bounds(0) is None
        assert region.axis_bounds(2) == (-1e-6, 2e-6)

    def test_non_positive_spacing_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            RefinementRegion(spacing=0.0)

    def test_spacing_of_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length-3 sequence"):
            RefinementRegion(spacing=(10e-9, 20e-9))

    def test_bounds_must_increase(self):
        with pytest.raises(ValueError, match="must be increasing"):
            RefinementRegion(spacing=10e-9, x=(1e-6, -1e-6))

    def test_bounds_of_wrong_length_raise(self):
        with pytest.raises(ValueError, match="pair in metres"):
            RefinementRegion(spacing=10e-9, y=(1e-6, 2e-6, 3e-6))


class TestGradedGridWithoutRegions:
    """With nothing to refine the policy has to reproduce the uniform ones exactly."""

    def test_matches_uniform_grid_cell_for_cell(self):
        spacing = 50e-9
        graded = GradedGrid(spacing=spacing).resolve_extent((1e-6, 1e-6, 4e-6))
        uniform = UniformGrid(spacing=spacing).resolve((20, 20, 80))
        assert graded.shape == uniform.shape
        for axis in range(3):
            assert np.array_equal(_edges(graded, axis), _edges(uniform, axis))

    def test_per_axis_background_gives_per_axis_uniform_widths(self):
        graded = GradedGrid(spacing=(50e-9, 100e-9, 25e-9)).resolve_extent((1e-6, 2e-6, 1e-6))
        assert graded.shape == (20, 20, 40)
        for axis, spacing in enumerate((50e-9, 100e-9, 25e-9)):
            assert np.allclose(_widths(graded, axis), spacing, rtol=_FLOAT32_SLACK)

    def test_is_uniform_only_without_regions_and_with_one_spacing(self):
        assert GradedGrid(spacing=50e-9).is_uniform
        assert not GradedGrid(spacing=(50e-9, 50e-9, 25e-9)).is_uniform
        assert not GradedGrid(spacing=50e-9, regions=(RefinementRegion(spacing=10e-9),)).is_uniform

    def test_extent_that_is_not_a_whole_number_of_cells_stays_exact(self):
        grid = GradedGrid(spacing=50e-9).resolve_extent((1e-6, 1e-6, 4.03e-6))
        edges = _edges(grid, 2)
        assert float(edges[-1] - edges[0]) == pytest.approx(4.03e-6, rel=_FLOAT32_SLACK)
        assert np.allclose(_widths(grid, 2), _widths(grid, 2)[0], rtol=_FLOAT32_SLACK)


class TestGradedGridWithRegions:
    """The four properties the generator promises."""

    EXTENT = (1e-6, 1e-6, 4e-6)
    REGION = (-100e-9, 100e-9)
    TARGET = 12.5e-9
    BACKGROUND = 50e-9
    MAX_RATIO = 1.4

    @pytest.fixture
    def grid(self) -> RectilinearGrid:
        policy = GradedGrid(
            spacing=self.BACKGROUND,
            regions=(RefinementRegion(spacing=(self.BACKGROUND, self.BACKGROUND, self.TARGET), z=self.REGION),),
            max_ratio=self.MAX_RATIO,
        )
        return policy.resolve_extent(self.EXTENT)

    def test_cells_inside_the_region_are_at_most_the_target(self, grid):
        edges = _edges(grid, 2)
        widths = _widths(grid, 2)
        inside = (edges[:-1] >= self.REGION[0] * (1 + _FLOAT32_SLACK)) & (
            edges[1:] <= self.REGION[1] * (1 + _FLOAT32_SLACK)
        )
        assert inside.sum() >= 16
        assert widths[inside].max() <= self.TARGET * (1 + _FLOAT32_SLACK)

    def test_neighbouring_widths_obey_the_ratio_bound(self, grid):
        assert _max_neighbour_ratio(_widths(grid, 2)) <= self.MAX_RATIO * (1 + 1e-3)

    def test_region_boundaries_land_on_cell_edges(self, grid):
        edges = _edges(grid, 2)
        for boundary in self.REGION:
            assert float(np.abs(edges - boundary).min()) < _FLOAT32_SLACK * self.TARGET

    def test_the_extent_is_exact(self, grid):
        for axis, length in enumerate(self.EXTENT):
            edges = _edges(grid, axis)
            assert float(edges[-1] - edges[0]) == pytest.approx(length, rel=_FLOAT32_SLACK)
            assert float(edges[0]) == pytest.approx(-length / 2, rel=_FLOAT32_SLACK)

    def test_widths_never_exceed_the_background(self, grid):
        assert _widths(grid, 2).max() <= self.BACKGROUND * (1 + _FLOAT32_SLACK)

    def test_untouched_axes_stay_uniform(self, grid):
        assert np.allclose(_widths(grid, 0), self.BACKGROUND, rtol=_FLOAT32_SLACK)
        assert np.allclose(_widths(grid, 1), self.BACKGROUND, rtol=_FLOAT32_SLACK)

    def test_it_grades_rather_than_stepping_to_the_background(self, grid):
        widths = _widths(grid, 2)
        intermediate = widths[(widths > self.TARGET * 1.05) & (widths < self.BACKGROUND * 0.95)]
        assert intermediate.size >= 3, f"expected a transition ramp, got widths {np.unique(widths.round(12))}"

    def test_an_axis_bound_of_none_refines_the_whole_axis(self):
        policy = GradedGrid(spacing=50e-9, regions=(RefinementRegion(spacing=10e-9, z=(-1e-7, 1e-7)),))
        grid = policy.resolve_extent((1e-6, 1e-6, 4e-6))
        assert np.allclose(_widths(grid, 0), 10e-9, rtol=_FLOAT32_SLACK)
        assert np.allclose(_widths(grid, 1), 10e-9, rtol=_FLOAT32_SLACK)
        assert _widths(grid, 2).max() > 10e-9

    def test_the_finest_target_wins_where_regions_overlap(self):
        policy = GradedGrid(
            spacing=50e-9,
            regions=(
                RefinementRegion(spacing=(50e-9, 50e-9, 25e-9), z=(-500e-9, 500e-9)),
                RefinementRegion(spacing=(50e-9, 50e-9, 6.25e-9), z=(-50e-9, 50e-9)),
            ),
        )
        grid = policy.resolve_extent((1e-6, 1e-6, 4e-6))
        edges = _edges(grid, 2)
        widths = _widths(grid, 2)
        inner = (edges[:-1] >= -50e-9 * (1 - _FLOAT32_SLACK)) & (edges[1:] <= 50e-9 * (1 + _FLOAT32_SLACK))
        assert widths[inner].max() <= 6.25e-9 * (1 + _FLOAT32_SLACK)
        assert _max_neighbour_ratio(widths) <= 1.4 * (1 + 1e-3)

    def test_a_region_coarser_than_the_background_changes_nothing(self):
        policy = GradedGrid(spacing=50e-9, regions=(RefinementRegion(spacing=200e-9, z=(-500e-9, 500e-9)),))
        grid = policy.resolve_extent((1e-6, 1e-6, 4e-6))
        assert np.allclose(_widths(grid, 2), 50e-9, rtol=_FLOAT32_SLACK)

    def test_a_region_outside_the_domain_is_ignored_on_that_axis(self):
        policy = GradedGrid(spacing=50e-9, regions=(RefinementRegion(spacing=10e-9, z=(3e-6, 4e-6)),))
        grid = policy.resolve_extent((1e-6, 1e-6, 4e-6))
        assert np.allclose(_widths(grid, 2), 50e-9, rtol=_FLOAT32_SLACK)
        assert np.allclose(_widths(grid, 0), 10e-9, rtol=_FLOAT32_SLACK)

    def test_center_shifts_the_whole_domain(self):
        policy = GradedGrid(
            spacing=50e-9,
            regions=(RefinementRegion(spacing=(50e-9, 50e-9, 12.5e-9), z=(900e-9, 1100e-9)),),
            center=(0.0, 0.0, 1e-6),
        )
        grid = policy.resolve_extent((1e-6, 1e-6, 4e-6))
        edges = _edges(grid, 2)
        assert float(edges[0]) == pytest.approx(-1e-6, rel=_FLOAT32_SLACK)
        assert float(edges[-1]) == pytest.approx(3e-6, rel=_FLOAT32_SLACK)
        assert float(np.abs(edges - 900e-9).min()) < _FLOAT32_SLACK * 12.5e-9


class TestGradedGridErrors:
    """Invalid input and layouts the generator cannot mesh."""

    def test_max_ratio_must_exceed_one(self):
        with pytest.raises(ValueError, match="max_ratio must be greater than one"):
            GradedGrid(spacing=50e-9, max_ratio=1.0)

    def test_background_spacing_must_be_positive(self):
        with pytest.raises(ValueError, match="must be positive"):
            GradedGrid(spacing=-50e-9)

    def test_regions_must_be_refinement_regions(self):
        with pytest.raises(ValueError, match="must contain RefinementRegion"):
            GradedGrid(spacing=50e-9, regions=({"spacing": 10e-9},))

    def test_resolving_from_a_cell_count_raises(self):
        with pytest.raises(ValueError, match="cannot be resolved from a cell count"):
            GradedGrid(spacing=50e-9).resolve((10, 10, 10))

    def test_a_stretch_too_short_to_grade_raises(self):
        # A 20 nm sliver between the domain edge and a region of 12 nm cells cannot be bridged
        # within a ratio of 1.2: the sliver holds one cell 1.67 times its neighbour.
        policy = GradedGrid(
            spacing=100e-9,
            regions=(RefinementRegion(spacing=(100e-9, 100e-9, 12e-9), z=(-480e-9, 400e-9)),),
            max_ratio=1.2,
        )
        with pytest.raises(ValueError, match="could not honour max_ratio"):
            policy.resolve_extent((1e-6, 1e-6, 1e-6))


class TestGradedGridReporting:
    """min_spacing, summary and the CFL bound."""

    def test_min_spacing_is_the_finest_requested_width(self):
        policy = GradedGrid(
            spacing=50e-9,
            regions=(
                RefinementRegion(spacing=25e-9, z=(-1e-7, 1e-7)),
                RefinementRegion(spacing=(40e-9, 10e-9, 40e-9), x=(-1e-7, 1e-7)),
            ),
        )
        assert policy.min_spacing == 10e-9

    def test_summary_stats_report_the_realized_grid(self):
        policy = GradedGrid(
            spacing=50e-9,
            regions=(RefinementRegion(spacing=(50e-9, 50e-9, 12.5e-9), z=(-100e-9, 100e-9)),),
        )
        stats = policy.summary_stats((1e-6, 1e-6, 4e-6))
        grid = policy.resolve_extent((1e-6, 1e-6, 4e-6))
        assert stats["cells"] == grid.shape
        assert stats["total_cells"] == grid.shape[0] * grid.shape[1] * grid.shape[2]
        assert stats["min_width"][2] == pytest.approx(12.5e-9, rel=_FLOAT32_SLACK)
        assert stats["max_width"][2] <= 50e-9 * (1 + _FLOAT32_SLACK)
        assert stats["max_ratio_observed"][2] <= 1.4 * (1 + 1e-3)
        assert stats["ratio_ok"]
        assert stats["num_regions"] == 1

    def test_summary_text_names_the_cells_and_the_ratio(self):
        policy = GradedGrid(
            spacing=50e-9,
            regions=(RefinementRegion(spacing=(50e-9, 50e-9, 12.5e-9), z=(-100e-9, 100e-9)),),
        )
        text = policy.summary((1e-6, 1e-6, 4e-6))
        assert "GradedGrid" in text
        assert "max ratio 1.4" in text
        assert "ratio bound honoured" in text
        assert text.count("\n") == 4  # header, one line per axis, verdict

    def test_the_unresolved_policy_bounds_the_time_step_by_the_finest_cell(self):
        policy = GradedGrid(spacing=50e-9, regions=(RefinementRegion(spacing=12.5e-9, z=(-1e-7, 1e-7)),))
        config = SimulationConfig(grid=policy, time=10e-15)
        uniform = SimulationConfig(grid=UniformGrid(spacing=12.5e-9), time=10e-15)
        assert config.resolved_grid is None
        assert config.time_step_duration == pytest.approx(uniform.time_step_duration)

    def test_uniform_spacing_raises_once_a_region_exists(self):
        config = SimulationConfig(
            grid=GradedGrid(spacing=50e-9, regions=(RefinementRegion(spacing=10e-9),)),
            time=10e-15,
        )
        with pytest.raises(ValueError, match="no single uniform spacing"):
            config.uniform_spacing()
