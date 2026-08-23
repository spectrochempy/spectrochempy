"""Regression tests for Coord.linearize() linearity detection.

These tests cover the symmetric handling of ascending and descending axes,
and ensure that irregular or duplicated coordinates are never silently
rewritten into an artificial uniform grid.
"""

import numpy as np
import pytest

from spectrochempy import NDDataset
from spectrochempy.core.dataset.coord import Coord


# ==============================================================================
# Regular axes: intentional linearization must be preserved
# ==============================================================================


class TestLinearizeRegularAxes:
    def test_strictly_increasing_regular_is_linear(self):
        values = np.linspace(0.0, 100.0, 51)
        c = Coord(values)
        assert c.linear
        np.testing.assert_allclose(c.data, values)

    def test_strictly_decreasing_regular_is_linear(self):
        values = np.linspace(100.0, 0.0, 51)
        c = Coord(values)
        assert c.linear
        np.testing.assert_allclose(c.data, values)

    @pytest.mark.parametrize("reverse", [False, True])
    def test_near_regular_within_tolerance_is_linear(self, reverse):
        # relative spacing deviation well below the linearization tolerance
        base = np.linspace(0.0, 10.0, 21)
        values = base.copy()
        values[5] += 0.001  # ~0.01% spacing deviation
        if reverse:
            values = values[::-1].copy()
        c = Coord(values)
        assert c.linear

    def test_ascending_descending_equivalence_for_regular_axes(self):
        asc = Coord(np.arange(20) * 0.5)
        desc = Coord((np.arange(20) * 0.5)[::-1].copy())
        assert asc.linear == desc.linear

    def test_linearization_preserves_units_and_title(self):
        c = Coord(np.linspace(0, 10, 11), units="cm^-1", title="wavenumber")
        assert c.units is not None
        assert c.title == "wavenumber"
        assert c.linear


# ==============================================================================
# Irregular axes: values must be preserved exactly, never rewritten
# ==============================================================================


class TestLinearizeIrregularAxes:
    @pytest.mark.parametrize(
        "values",
        [
            np.array([0.0, 1.0, 3.0, 6.0, 10.0]),  # accelerating
            np.array([0.0, 4.0, 7.0, 9.0, 10.0]),  # decelerating
        ],
    )
    def test_irregular_increasing_not_linear(self, values):
        c = Coord(values)
        assert not c.linear
        np.testing.assert_allclose(c.data, values)

    @pytest.mark.parametrize(
        "values",
        [
            np.array([10.0, 9.0, 7.0, 4.0, 0.0]),
            np.array([10.0, 6.0, 3.0, 1.0, 0.0]),
        ],
    )
    def test_irregular_decreasing_not_linear(self, values):
        c = Coord(values)
        assert not c.linear
        np.testing.assert_allclose(c.data, values)

    @pytest.mark.parametrize(
        "values",
        [
            np.array([0.0, 1.0, 3.0, 6.0, 10.0]),
            np.array([0.0, 4.0, 7.0, 9.0, 10.0]),
        ],
    )
    def test_same_geometry_both_orientations(self, values):
        asc = Coord(values)
        desc = Coord(values[::-1].copy())
        assert asc.linear == desc.linear == False  # noqa: E712
        np.testing.assert_allclose(desc.data, values[::-1])

    def test_irregular_data_kept_at_full_precision(self):
        values = np.array([0.0, 0.1, 0.3001, 0.6002, 1.0002])
        c = Coord(values)
        assert not c.linear
        np.testing.assert_allclose(c._data.squeeze(), values)


# ==============================================================================
# Duplicates: must never be erased by an artificial linspace
# ==============================================================================


class TestLinearizeDuplicateValues:
    def test_adjacent_duplicate_increasing(self):
        values = np.array([0.0, 1.0, 1.0, 2.5, 4.0])
        c = Coord(values)
        assert not c.linear
        np.testing.assert_allclose(c._data.squeeze(), values)

    def test_adjacent_duplicate_decreasing(self):
        # historical bug: this axis was silently replaced by a uniform grid
        values = np.array([4.0, 2.5, 1.0, 1.0, 0.0])
        c = Coord(values)
        assert not c.linear
        np.testing.assert_allclose(c._data.squeeze(), values)

    @pytest.mark.parametrize("position", ["start", "middle", "end"])
    def test_duplicate_positions_decreasing(self, position):
        values = np.array([8.0, 6.0, 4.0, 2.0, 0.0])
        idx = {"start": 0, "middle": 2, "end": 3}[position]
        values[idx] = values[idx + 1]
        c = Coord(values)
        assert not c.linear
        np.testing.assert_allclose(c._data.squeeze(), values)

    def test_multiple_duplicates_decreasing(self):
        values = np.array([9.0, 9.0, 7.0, 5.0, 5.0, 2.0, 0.0, 0.0])
        c = Coord(values)
        assert not c.linear
        np.testing.assert_allclose(c._data.squeeze(), values)

    def test_duplicate_symmetry_between_orientations(self):
        asc_values = np.array([0.0, 1.0, 1.0, 3.0, 5.0])
        desc_values = asc_values[::-1].copy()
        asc = Coord(asc_values)
        desc = Coord(desc_values)
        assert asc.linear == desc.linear == False  # noqa: E712
        np.testing.assert_allclose(asc._data.squeeze(), asc_values)
        np.testing.assert_allclose(desc._data.squeeze(), desc_values)

    def test_fully_constant_axis_stays_constant(self):
        # intentional historical behavior: uniform detection keeps the axis
        # linear, and the values are unchanged anyway
        values = np.full(6, 5.0)
        c = Coord(values)
        np.testing.assert_allclose(c.data, values)

# ==============================================================================
# Non-finite values: no artificial linearization, no silent loss
# ==============================================================================


class TestLinearizeNonFiniteValues:
    @pytest.mark.parametrize(
        "values",
        [
            np.array([0.0, 1.0, np.nan, 3.0]),
            np.array([1.0, 2.0, 3.0, np.inf]),
            np.array([-np.inf, 1.0, 2.0, 3.0]),
        ],
    )
    def test_non_finite_values_are_never_linearized(self, values):
        c = Coord(values)
        assert not c.linear
        np.testing.assert_equal(c._data.squeeze(), values)

    def test_non_finite_symmetry_between_orientations(self):
        asc = Coord(np.array([1.0, 2.0, 3.0, np.inf]))
        desc = Coord(np.array([3.0, 2.0, 1.0, -np.inf]))
        assert not asc.linear
        assert not desc.linear


# ==============================================================================
# Integration: NDDataset construction preserves provided coordinates
# ==============================================================================


class TestNDDatasetDescendingDuplicateIntegration:
    def test_dataset_keeps_descending_duplicated_coordinates(self):
        # historical bug: this axis was replaced by a uniform grid at
        # dataset-construction time, before any processing could see it
        values = np.array(
            [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 0.0]
        )
        ds = NDDataset(np.arange(len(values), dtype=float))
        ds.set_coordset(Coord(title="x", data=values))
        np.testing.assert_allclose(ds.x._data.squeeze(), values)
        np.testing.assert_allclose(ds.x.data, values)
        assert not ds.x.linear
