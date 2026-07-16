# SPDX-License-Identifier: BSD-3-Clause
# (see LICENSE.txt for details)

"""
Behavioral tests for Coord.

Tests focus on public API behavior, not private implementation details.
Uses deterministic synthetic data only. No external files, no plotting.
"""

import numpy as np
import pytest

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import DimensionalityError
from spectrochempy.core.units import ur
from spectrochempy.utils.testing import assert_array_equal
from spectrochempy.utils.testing import assert_units_equal

# ==============================================================================
# Construction
# ==============================================================================


class TestCoordConstruction:
    """Coord creation from various inputs."""

    def test_from_list(self):
        c = Coord([1, 2, 3])
        assert c.size == 3
        assert c.shape == (3,)

    def test_from_ndarray(self):
        c = Coord(np.array([10.0, 20.0, 30.0]))
        assert c.size == 3
        assert np.issubdtype(c.dtype, np.floating)

    def test_from_singleton_row_array(self):
        c = Coord(np.array([[1.0, 2.0, 3.0]]))
        assert c.shape == (3,)
        assert_array_equal(c.data, np.array([1.0, 2.0, 3.0]))

    def test_from_singleton_column_array(self):
        c = Coord(np.array([[1.0], [2.0], [3.0]]))
        assert c.shape == (3,)
        assert_array_equal(c.data, np.array([1.0, 2.0, 3.0]))

    def test_from_singleton_row_nddataset(self):
        c = Coord(NDDataset(np.array([[1.0, 2.0, 3.0]])))
        assert c.shape == (3,)
        assert_array_equal(c.data, np.array([1.0, 2.0, 3.0]))

    def test_from_true_2d_array_raises(self):
        with pytest.raises(
            ValueError,
            match="Only one 1D arrays can be used to define coordinates",
        ):
            Coord(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_from_linspace(self):
        c = Coord(np.linspace(0, 10, 11))
        assert c.size == 11
        assert c.linear

    def test_from_arange(self):
        c = Coord(np.arange(5))
        assert c.size == 5

    def test_from_float_list(self):
        c = Coord([1.5, 2.5, 3.5])
        assert c.size == 3
        assert np.issubdtype(c.dtype, np.floating)

    def test_with_name(self):
        c = Coord([1, 2, 3], name="wavelength")
        assert c.name == "wavelength"

    def test_with_units(self):
        c = Coord([1, 2, 3], units="cm^-1")
        assert_units_equal(c.units, ur("cm^-1"))

    def test_with_title(self):
        c = Coord([1, 2, 3], title="Wavenumber")
        assert c.title == "Wavenumber"

    def test_with_labels(self):
        labels = ["a", "b", "c"]
        c = Coord([1, 2, 3], labels=labels)
        assert_array_equal(c.labels, labels)

    def test_empty(self):
        c = Coord([])
        assert c.size == 0
        assert c.is_empty

    def test_single_element(self):
        c = Coord([42.0])
        assert c.size == 1
        assert c.shape == (1,)

    def test_non_monotonic(self):
        c = Coord([3, 1, 4, 1, 5])
        assert c.size == 5
        assert not c.linear

    def test_duplicate_values(self):
        c = Coord([1, 1, 2, 2, 3])
        assert c.size == 5

    def test_dtype_preservation(self):
        c = Coord([1, 2, 3])
        assert np.issubdtype(c.dtype, np.floating)
        c2 = Coord([1.0, 2.0, 3.0])
        assert np.issubdtype(c2.dtype, np.floating)

    def test_name_dimension(self):
        c = Coord([1, 2, 3], name="time")
        dims = c.dims
        assert isinstance(dims, list)
        assert "time" not in dims  # Coord dims are always ["x"], independent of name

    def test_default_dims_list(self):
        c = Coord([1, 2, 3])
        dims = c.dims
        assert isinstance(dims, list)
        assert len(dims) == 1

    def test_non_iterable_data_raises(self):
        with pytest.raises(ValueError):
            Coord(42)


# ==============================================================================
# Properties
# ==============================================================================


class TestCoordProperties:
    """Coord public property behavior."""

    def test_data_returns_numpy_array(self):
        c = Coord([1.0, 2.0, 3.0])
        data = c.data
        assert_array_equal(data, np.array([1.0, 2.0, 3.0]))

    def test_umasked_data(self):
        c = Coord([1.0, 2.0, 3.0])
        d = c.umasked_data
        assert d is not None

    def test_linear_returns_bool(self):
        c = Coord([1, 2, 3, 4, 5])
        assert isinstance(c.linear, bool)

    def test_reversed_returns_bool(self):
        c = Coord([5, 4, 3, 2, 1])
        assert isinstance(c.reversed, bool)

    def test_reversed_true_for_wavenumber(self):
        c = Coord([4000, 3000, 2000, 1000], units="cm^-1")
        assert c.reversed

    def test_reversed_true_for_ppm(self):
        c = Coord([10, 5, 0], units="ppm")
        assert c.reversed

    def test_is_descendant_true_for_decreasing(self):
        c = Coord([5, 4, 3, 2, 1])
        assert c.is_descendant

    def test_spacing_linear(self):
        c = Coord(np.arange(10, dtype=float))
        spacing = c.spacing
        assert spacing is not None

    def test_sigdigits_returns_int(self):
        c = Coord([1.0, 2.0, 3.0])
        digits = c.sigdigits
        assert isinstance(digits, (int, np.integer))

    def test_limits(self):
        c = Coord([1.0, 5.0, 3.0])
        lim = c.limits
        assert len(lim) == 2

    def test_show_datapoints_default(self):
        c = Coord([1, 2, 3])
        assert isinstance(c.show_datapoints, bool)

    def test_show_datapoints_default_false(self):
        c = Coord([1, 2, 3], units="cm^-1")
        # show_datapoints only returns True when laser_frequency is set
        # and units are time or length
        assert isinstance(c.show_datapoints, bool)

    def test_laser_frequency_default_none(self):
        c = Coord([1, 2, 3], units="cm^-1")
        assert c.laser_frequency is None

    def test_laser_frequency_settable(self):
        c = Coord([1, 2, 3], units="cm^-1")
        c.laser_frequency = 15798.0
        assert c.laser_frequency == 15798.0

    def test_has_data(self):
        c = Coord([1, 2, 3])
        assert c.has_data

    def test_has_data_false_empty(self):
        c = Coord([])
        assert not c.has_data

    def test_values_property(self):
        c = Coord([1.0, 2.0, 3.0])
        v = c.values
        assert v is not None

    def test_dtype(self):
        c = Coord([1, 2, 3])
        assert np.issubdtype(c.dtype, np.floating)


# ==============================================================================
# Unit conversion
# ==============================================================================


class TestCoordUnitConversion:
    """Coord unit conversion behavior."""

    def test_to_compatible_units(self):
        c = Coord([1, 2, 3], units="m")
        c2 = c.to("cm")
        assert_units_equal(c2.units, ur("cm"))

    def test_to_incompatible_units_raises(self):
        c = Coord([1, 2, 3], units="cm^-1")
        with pytest.raises(DimensionalityError):
            c.to("s")

    def test_to_with_force(self):
        c = Coord([1, 2, 3], units="cm^-1")
        c2 = c.to("nm^-1", force=True)
        assert c2.units is not None

    def test_original_unchanged_after_to(self):
        c = Coord([1, 2, 3], units="m")
        c.to("cm")
        assert_units_equal(c.units, ur("m"))


# ==============================================================================
# loc2index
# ==============================================================================


class TestCoordLoc2Index:
    """Location-to-index conversion behavior."""

    def test_by_exact_value(self):
        c = Coord([10, 20, 30, 40, 50])
        idx = c.loc2index(30)
        assert idx == 2

    def test_nearest_value(self):
        c = Coord([10.0, 20.0, 30.0, 40.0])
        idx = c.loc2index(25.0)
        assert isinstance(idx, (int, np.integer))

    def test_by_sequence(self):
        c = Coord([10, 20, 30, 40, 50])
        idxs = c.loc2index([20, 40])
        assert len(idxs) == 2

    def test_by_int_location(self):
        c = Coord(np.linspace(0, 100, 101))
        idx = c.loc2index(50)
        assert 0 <= idx < c.size


# ==============================================================================
# linearize / set_laser_frequency
# ==============================================================================


class TestCoordTransform:
    """Coord transformation methods."""

    def test_linearize(self):
        c = Coord([1, 2, 3, 5, 8, 13])
        result = c.linearize()
        assert result is None  # linearize operates in-place
        assert isinstance(c.linear, bool)

    def test_set_laser_frequency(self):
        c = Coord([1, 2, 3], units="cm^-1")
        result = c.set_laser_frequency(15798.0)
        assert result is None  # operates in-place
        assert c.laser_frequency is not None


# ==============================================================================
# Copy
# ==============================================================================


class TestCoordCopy:
    """Coord copy behavior."""

    def test_copy_preserves_name(self):
        c = Coord([1, 2, 3], name="x", units="cm^-1")
        c2 = c.copy()
        assert c2.name == c.name

    def test_copy_preserves_units(self):
        c = Coord([1, 2, 3], name="x", units="cm^-1")
        c2 = c.copy()
        assert_units_equal(c2.units, c.units)

    def test_copy_independence(self):
        c = Coord([1, 2, 3])
        c2 = c.copy()
        c2.name = "changed"
        assert c.name != c2.name

    def test_copy_preserves_data(self):
        c = Coord([1.0, 2.0, 3.0])
        c2 = c.copy()
        assert_array_equal(c2.data, c.data)


# ==============================================================================
# Transpose
# ==============================================================================


class TestCoordTranspose:
    """Coord transpose behavior."""

    def test_transpose_preserves_size(self):
        c = Coord([1, 2, 3])
        ct = c.transpose()
        assert ct.size == c.size

    def test_transpose_returns_coord(self):
        c = Coord([1, 2, 3])
        ct = c.transpose()
        assert isinstance(ct, Coord)


# ==============================================================================
# Edge cases
# ==============================================================================


class TestCoordEdgeCases:
    """Coord edge case behavior."""

    def test_labels_get_set(self):
        c = Coord([1, 2, 3])
        labels = ["a", "b", "c"]
        c.labels = labels
        assert_array_equal(c.labels, labels)

    def test_empty_coord_properties(self):
        c = Coord([])
        assert c.size == 0
        assert c.is_empty
        assert not c.has_data

    def test_integer_to_float_promotion(self):
        c = Coord([1, 2, 3])
        c2 = c.to("m", force=True)
        assert c2 is not None

    def test_linear_returns_true_for_equal_spacing(self):
        c = Coord(np.linspace(100, 200, 50))
        assert c.linear

    def test_linear_returns_false_for_unequal_spacing(self):
        c = Coord([1, 2, 4, 8, 16])
        assert not c.linear

    def test_is_masked_always_false(self):
        c = Coord([1, 2, 3])
        assert not c.is_masked

    def test_mask_always_nomask(self):
        from spectrochempy.core.dataset.coord import NOMASK

        c = Coord([1, 2, 3])
        assert c.mask is NOMASK

    def test_ndim_always_one(self):
        c = Coord([1, 2, 3])
        assert c.ndim == 1

    def test_is_1d_always_true(self):
        c = Coord([1, 2, 3])
        assert c.is_1d

    def test_T_returns_self(self):
        c = Coord([1, 2, 3])
        assert c.T is c


# ==============================================================================
# Arithmetic
# ==============================================================================


class TestCoordArithmetic:
    """Coord arithmetic behavior (operators set via coord.py)."""

    def test_add_with_value(self):
        c = Coord([1, 2, 3])
        c2 = c + 1
        assert c2.size == 3

    def test_sub_with_value(self):
        c = Coord([1, 2, 3])
        c2 = c - 1
        assert c2.size == 3

    def test_mul_with_scalar(self):
        c = Coord([1, 2, 3])
        c2 = c * 2
        assert c2.size == 3

    def test_truediv_with_scalar(self):
        c = Coord([2, 4, 6])
        c2 = c / 2
        assert c2.size == 3

    def test_neg(self):
        c = Coord([1, 2, 3])
        c2 = -c
        assert c2.size == 3

    def test_abs(self):
        c = Coord([-1, -2, 3])
        c2 = abs(c)
        assert c2.size == 3


# ==============================================================================
# Equality
# ==============================================================================


class TestCoordEquality:
    """Coord equality behavior."""

    def test_eq_same_data(self):
        c1 = Coord([1, 2, 3])
        c2 = Coord([1, 2, 3])
        assert c1 == c2

    def test_eq_different_data(self):
        c1 = Coord([1, 2, 3])
        c2 = Coord([4, 5, 6])
        assert c1 != c2

    def test_eq_different_shape(self):
        c1 = Coord([1, 2, 3])
        c2 = Coord([1, 2])
        assert c1 != c2

    def test_eq_with_units(self):
        c1 = Coord([1, 2, 3], units="m")
        c2 = Coord([1, 2, 3], units="m")
        assert c1 == c2

    def test_eq_different_units(self):
        c1 = Coord([1, 2, 3], units="m")
        c2 = Coord([1, 2, 3], units="cm")
        assert c1 != c2


# ==============================================================================
# Indexing & slicing
# ==============================================================================


class TestCoordIndexing:
    """Coord indexing behavior."""

    def test_getitem_int(self):
        c = Coord([10, 20, 30, 40])
        assert c[0] == 10
        assert c[2] == 30

    def test_getitem_slice(self):
        c = Coord([10, 20, 30, 40])
        sliced = c[1:3]
        assert sliced.size == 2
        assert_array_equal(sliced.data, [20, 30])

    def test_getitem_step(self):
        c = Coord([10, 20, 30, 40, 50])
        sliced = c[::2]
        assert sliced.size == 3
        assert_array_equal(sliced.data, [10, 30, 50])

    def test_getitem_name_preserved(self):
        c = Coord([10, 20, 30], name="x")
        sliced = c[0:2]
        assert sliced.name == "x"


class TestCoordCopyLabels:
    """Label preservation across Coord copy operations."""

    def test_coord_copy_preserves_labels(self):
        c = Coord([1, 2, 3], name="x")
        c.labels = ["a", "b", "c"]
        c2 = c.copy()
        assert list(c2.labels) == ["a", "b", "c"]

    def test_coord_copy_labels_independent(self):
        c = Coord([1, 2, 3], name="x")
        c.labels = ["a", "b", "c"]
        c2 = c.copy()
        c2.labels[0] = "modified"
        assert c.labels[0] == "a"
        assert c2.labels[0] == "modified"
