# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
r"""
Regressions for the mask contract of ``NDDataset.trapezoid()`` and
``NDDataset.simpson()``.

A definite integral is only defined when every point that contributed to it is
visible. The contract under test is:

- no masked point in a slice: the slice is integrated normally and the
  corresponding output is not masked;
- at least one masked point in a slice: the corresponding output is masked and
  its raw value is ``NaN``, so an integral derived from excluded points is never
  presented as valid;
- a fully masked slice follows the same rule;
- a 1D input yields a zero-dimensional result with a coherent scalar mask;
- the values hidden under the mask neither influence the result nor reach the
  quadrature, so they cannot overflow it.

These tests use real ``NDDataset`` objects and the public integration API. The
calculation is never mocked.
"""

import numpy as np
import pytest
import scipy.integrate

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset

METHODS = ("trapezoid", "simpson")
X = np.array([0.0, 1.0, 2.0])
X_UNITS = "s"
DATA_UNITS = "V"


def make_1d(values, mask, x=X, title="x"):
    """1D masked dataset with a single named coordinate."""
    return NDDataset(
        np.ma.MaskedArray(np.asarray(values, dtype=float), mask=np.asarray(mask)),
        coordset=[Coord(x, title=title, units=X_UNITS)],
        units=DATA_UNITS,
    )


def make_2d(values, mask, x=X, y_values=(10.0, 20.0)):
    """2D masked dataset with a `y` observation axis and an `x` integration axis."""
    values = np.asarray(values, dtype=float)
    return NDDataset(
        np.ma.MaskedArray(values, mask=np.asarray(mask)),
        coordset=[
            Coord(np.asarray(y_values, dtype=float), title="temperature", units="K"),
            Coord(x, title="x", units=X_UNITS),
        ],
        units=DATA_UNITS,
    )


# ======================================================================================
# COMPLETE SLICES
# ======================================================================================


class TestCompleteSlices:
    """Slices without any masked point keep the ordinary integral."""

    @pytest.mark.parametrize("method", METHODS)
    def test_unmasked_control(self, method):
        # a constant 1 over [0, 2] s integrates to 2.0 s
        ds = make_1d([1.0, 1.0, 1.0], [False, False, False])
        result = getattr(ds, method)()
        assert np.isclose(np.asarray(result.data), 2.0)
        assert not result.is_masked
        assert np.asarray(result.mask).shape == ()

    @pytest.mark.parametrize("method", METHODS)
    def test_all_slices_complete_next_to_an_incomplete_one(self, method):
        # the first spectrum is complete, the second one is not
        ds = make_2d(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[False, False, False], [False, True, False]],
        )
        result = getattr(ds, method)(dim="x")
        assert np.asarray(result.data).shape == (2,)
        np.testing.assert_array_equal(np.asarray(result.mask), [False, True])
        # the complete area is still exactly published
        assert np.isclose(np.asarray(result.data)[0], 2.0)
        assert np.isnan(np.asarray(result.data)[1])

    @pytest.mark.parametrize("method", METHODS)
    def test_ninety_nine_of_one_hundred_spectra_remain_valid(self, method):
        data = np.arange(500.0).reshape(100, 5)
        mask = np.zeros((100, 5), dtype=bool)
        mask[42, 2] = True
        ds = NDDataset(
            np.ma.MaskedArray(data, mask=mask),
            coordset=[
                Coord(np.arange(100.0), title="y"),
                Coord(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), title="x", units=X_UNITS),
            ],
            units=DATA_UNITS,
        )
        result = getattr(ds, method)(dim="x")
        expected = scipy.integrate.trapezoid(
            data, x=np.array([0.0, 1.0, 2.0, 3.0, 4.0]), axis=1
        )
        if method == "simpson":
            expected = scipy.integrate.simpson(
                data, x=np.array([0.0, 1.0, 2.0, 3.0, 4.0]), axis=1
            )
        published = np.asarray(result.data)
        assert int(np.asarray(result.mask).sum()) == 1
        assert np.isnan(published[42])
        # the 99 complete areas match an independent reference exactly
        np.testing.assert_allclose(np.delete(published, 42), np.delete(expected, 42))


# ======================================================================================
# INCOMPLETE SLICES
# ======================================================================================


class TestIncompleteSlices:
    """Slices built from excluded points are published as unavailable."""

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize(
        "mask",
        [
            [False, True, False],  # interior point
            [False, False, True],  # end point
            [True, False, False],  # first point
            [True, True, True],  # fully masked slice
        ],
    )
    def test_incomplete_1d_result_is_masked_nan(self, method, mask):
        ds = make_1d([1.0, 200.0, 3.0], mask)
        result = getattr(ds, method)()
        assert result.is_masked
        assert np.isnan(np.asarray(result.data))
        assert isinstance(result, NDDataset)
        assert result.shape == ()
        assert result.dims == []

    @pytest.mark.parametrize("method", METHODS)
    def test_fully_masked_slice_in_2d(self, method):
        ds = make_2d(
            [[1.0, 1.0, 1.0], [50.0, 60.0, 70.0]],
            [[False, False, False], [True, True, True]],
        )
        result = getattr(ds, method)(dim="x")
        np.testing.assert_array_equal(np.asarray(result.mask), [False, True])
        assert np.isclose(np.asarray(result.data)[0], 2.0)
        assert np.isnan(np.asarray(result.data)[1])

    @pytest.mark.parametrize("method", METHODS)
    def test_partial_mask_on_non_final_dimension(self, method):
        # the excluded point only affects the first output column
        ds = make_2d(
            [[1.0, 1.0, 1.0], [400.0, 400.0, 400.0]],
            [[False, False, False], [True, False, False]],
        )
        result = getattr(ds, method)(dim="y")
        assert result.dims == ["x"]
        np.testing.assert_array_equal(np.asarray(result.mask), [True, False, False])
        assert np.isnan(np.asarray(result.data)[0])
        # the two complete columns keep the exact published areas
        # (mean of 1 and 400 over 10 K)
        assert np.allclose(np.asarray(result.data)[1:], 2005.0)

    @pytest.mark.parametrize("method", METHODS)
    def test_incomplete_slice_on_a_middle_dimension_in_3d(self, method):
        # a 3D dataset has (z, y, x) dimensions: the integration axis is the
        # middle one
        data = np.ones((2, 3, 4))
        mask = np.zeros((2, 3, 4), dtype=bool)
        mask[1, 1, 2] = True
        ds = NDDataset(
            np.ma.MaskedArray(data, mask=mask),
            coordset=[
                Coord(np.array([0.0, 1.0]), title="z"),
                Coord(np.array([0.0, 1.0, 2.0]), title="y"),
                Coord(np.array([0.0, 1.0, 2.0, 3.0]), title="x", units=X_UNITS),
            ],
            units=DATA_UNITS,
        )
        result = getattr(ds, method)(dim="y")
        assert result.shape == (2, 4)
        # exactly one output element is affected: the one whose slice used the
        # excluded point, at the (z=1, x=2) position of the reduced geometry
        published_mask = np.asarray(result.mask)
        assert int(published_mask.sum()) == 1
        assert published_mask[1, 2]
        assert np.isnan(np.asarray(result.data)[1, 2])
        # every complete element keeps the exact published area of ones over 2 s
        complete = np.delete(np.asarray(result.data), 2, axis=1)
        assert np.allclose(complete, 2.0)


# ======================================================================================
# MASK SHAPE COHERENCE
# ======================================================================================


class TestMaskShapeCoherence:
    """The result mask is always compatible with the result shape."""

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize(
        ("make", "dim", "expected"),
        [
            (lambda: make_1d([1.0, 200.0, 3.0], [False, True, False]), "x", ()),
            (
                lambda: make_2d(
                    [[1.0, 1.0, 1.0], [1.0, 200.0, 1.0]],
                    [[False, False, False], [False, True, False]],
                ),
                "x",
                (2,),
            ),
            (
                lambda: make_2d(
                    [[1.0, 1.0, 1.0], [1.0, 200.0, 1.0]],
                    [[False, False, False], [False, True, False]],
                ),
                "y",
                (3,),
            ),
        ],
    )
    def test_mask_shape_matches_data_shape(self, method, make, dim, expected):
        result = getattr(make(), method)(dim=dim)
        assert result.shape == expected
        assert np.asarray(result.mask).shape == result.data.shape
        # the public masked accessor must work on the result
        assert np.ma.isMaskedArray(result.masked_data)
        assert result.is_masked

    @pytest.mark.parametrize("method", METHODS)
    def test_masked_data_hides_the_unavailable_area(self, method):
        ds = make_1d([1.0, 200.0, 3.0], [False, True, False])
        result = getattr(ds, method)()
        masked = result.masked_data
        assert np.ma.isMaskedArray(masked)
        assert bool(np.ma.getmaskarray(masked))
        # the raw value is explicitly unavailable, and never a masked-point area
        assert np.isnan(float(np.asarray(result.data)))

    @pytest.mark.parametrize("method", METHODS)
    def test_zero_dim_result_carries_a_scalar_mask(self, method):
        ds = make_1d([1.0, 200.0, 3.0], [False, True, False])
        result = getattr(ds, method)()
        assert result.ndim == 0
        assert np.ndim(np.asarray(result.mask)) == 0
        # a scalar mask is coercible, which the pre-correction source-shaped mask was not
        assert bool(result.mask) is True

    @pytest.mark.parametrize("method", METHODS)
    def test_explicit_all_false_source_mask_yields_unmasked_result(self, method):
        ds = make_2d([[1.0, 1.0, 1.0]], [[False, False, False]], y_values=(10.0,))
        ds.mask = np.zeros((1, 3), dtype=bool)
        result = getattr(ds, method)(dim="x")
        assert not result.is_masked
        assert np.ndim(np.asarray(result.mask)) == 0
        assert np.isclose(np.asarray(result.data)[0], 2.0)


# ======================================================================================
# INDEPENDENCE FROM THE HIDDEN VALUES
# ======================================================================================


class TestHiddenValueIndependence:
    """The values under a mask never reach the result nor the quadrature."""

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("hidden", [0.0, 200.0, -5.0, 1.0e12])
    def test_result_is_independent_of_the_hidden_value(self, method, hidden):
        reference = getattr(make_1d([1.0, 0.0, 3.0], [False, True, False]), method)()
        other = getattr(make_1d([1.0, hidden, 3.0], [False, True, False]), method)()
        assert np.isnan(np.asarray(reference.data))
        assert np.isnan(np.asarray(other.data))
        np.testing.assert_array_equal(
            np.asarray(reference.mask), np.asarray(other.mask)
        )

    @pytest.mark.parametrize("method", METHODS)
    def test_hidden_values_cannot_overflow_the_calculation(self, method):
        big = 1.0e308
        ds = make_2d(
            [[1.0, 1.0, 1.0], [1.0, big, 1.0]],
            [[False, False, False], [False, True, False]],
        )
        with np.errstate(all="raise"):
            result = getattr(ds, method)(dim="x")
        # the hidden value neither produced an overflow nor perturbed its neighbour
        assert np.isclose(np.asarray(result.data)[0], 2.0)
        assert np.isnan(np.asarray(result.data)[1])

    @pytest.mark.parametrize("method", METHODS)
    def test_fully_masked_slice_of_extreme_values(self, method):
        big = 1.0e308
        ds = make_2d(
            [[1.0, 1.0, 1.0], [big, big, big]],
            [[False, False, False], [True, True, True]],
        )
        with np.errstate(all="raise"):
            result = getattr(ds, method)(dim="x")
        assert np.isclose(np.asarray(result.data)[0], 2.0)
        assert np.isnan(np.asarray(result.data)[1])


# ======================================================================================
# PRESERVED GUARANTEES
# ======================================================================================


class TestPreservedGuarantees:
    """The mask policy must not alter the surrounding integration contract."""

    @pytest.mark.parametrize("method", METHODS)
    def test_source_dataset_is_not_mutated(self, method):
        ds = make_1d([1.0, 200.0, 3.0], [False, True, False])
        data_before = np.array(ds.data, copy=True)
        mask_before = np.array(ds.mask, copy=True)
        result = getattr(ds, method)()
        np.testing.assert_array_equal(np.asarray(ds.data), data_before)
        np.testing.assert_array_equal(np.asarray(ds.mask), mask_before)
        assert ds.shape == (3,)
        assert result.data is not ds.data
        assert result.mask is not ds.mask

    @pytest.mark.parametrize("method", METHODS)
    def test_units_are_unchanged(self, method):
        ds = make_1d([1.0, 200.0, 3.0], [False, True, False])
        result = getattr(ds, method)()
        assert result.units == ds.units * ds.x.units

    @pytest.mark.parametrize("method", METHODS)
    def test_decreasing_axis_still_publishes_no_area(self, method):
        # a reversed axis negates a defined area; an undefined one stays undefined
        defined = make_1d([1.0, 1.0, 1.0], [False] * 3, x=np.array([2.0, 1.0, 0.0]))
        assert np.isclose(np.asarray(getattr(defined, method)().data), -2.0)
        undefined = make_1d(
            [1.0, 200.0, 3.0], [False, True, False], x=np.array([2.0, 1.0, 0.0])
        )
        result = getattr(undefined, method)()
        assert np.isnan(np.asarray(result.data))
        assert result.is_masked

    @pytest.mark.parametrize("method", METHODS)
    def test_surviving_coordinates_are_preserved(self, method):
        ds = make_2d(
            [[1.0, 1.0, 1.0], [1.0, 200.0, 1.0]],
            [[False, False, False], [False, True, False]],
        )
        result = getattr(ds, method)(dim="x")
        assert result.coordset.names == ["y"]
        assert result.dims == ["y"]
        np.testing.assert_allclose(np.asarray(result.y.data), [10.0, 20.0])
        assert result.y.title == ds.y.title
        assert result.y.units == ds.y.units

    @pytest.mark.parametrize("method", METHODS)
    def test_identity_and_history_are_unchanged(self, method):
        ds = make_1d([1.0, 200.0, 3.0], [False, True, False])
        result = getattr(ds, method)()
        assert result.title == "area"
        assert result.description == (
            f"Integration of NDDataset '{ds.name}' along dim: 'x'."
        )
        assert len(result.history) == 1
        assert f"`{method}` method" in result.history[0]

    @pytest.mark.parametrize("method", METHODS)
    def test_partial_then_full_integration_stays_nddataset(self, method):
        ds = make_2d(
            [[1.0, 1.0, 1.0], [1.0, 200.0, 1.0]],
            [[False, False, False], [False, True, False]],
        )
        result = getattr(getattr(ds, method)(dim="x"), method)(dim="y")
        assert isinstance(result, NDDataset)
        assert result.shape == ()
        assert result.dims == []
        assert result.is_masked
        assert np.isnan(np.asarray(result.data))
