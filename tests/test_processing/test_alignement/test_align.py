# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""Tests for the interpolate module"""

import importlib
import sys

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.processing.alignment.align import align
from spectrochempy.utils.exceptions import UnitsCompatibilityError

# align
# -------


def test_legacy_align_import_is_deprecated():
    sys.modules.pop("spectrochempy.processing.alignement.align", None)
    with pytest.deprecated_call(match="processing.alignement.align") as record:
        legacy_module = importlib.import_module(
            "spectrochempy.processing.alignement.align"
        )
    assert legacy_module.align is align
    messages = [str(item.message) for item in record]
    assert any(
        "spectrochempy.processing.alignment.align" in message for message in messages
    )
    assert all("0.13.0" not in message for message in messages)
    assert any("deprecation policy is satisfied" in message for message in messages)


def test_align(ds1, ds2):
    ds1c = ds1.copy()
    dss = ds1c.align(ds2, dim="x")  # first syntax
    # TODO: flag copy=False raise an error
    assert dss is not None
    ds3, ds4 = dss  # a tuple is returned

    assert ds3.shape == (10, 100, 6)

    # TODO: labels are not aligned

    dss2 = align(ds1, ds2, dim="x")  # second syntax
    assert dss2 == dss
    assert dss2[0] == dss[0]
    ds5, ds6 = dss2

    dss_root = scp.align(ds1, ds2, dim="x")
    assert dss_root == dss

    # align another dim
    dss3 = align(ds1, ds2, dim="z")  # by default it would be the 'x' dim
    ds7, ds8 = dss3
    assert ds7.shape == (17, 100, 3)

    # align two dims
    dss4 = align(ds1, ds2, dims=["x", "z"])
    ds9, ds10 = dss4

    # align inner
    a, b = align(ds1, ds2, method="inner")


@pytest.mark.parametrize("keep_row", [False, True])
@pytest.mark.parametrize("second_labels", [None, ["q", "r", "s"]])
def test_align_inner_spectra(keep_row, second_labels):
    first_x = scp.Coord([1.0, 2.0, 3.0], labels=["a", "b", "c"])
    second_x = scp.Coord([2.0, 3.0, 4.0], labels=second_labels)
    first_values = np.array([10.0, 20.0, 30.0])
    second_values = np.array([2.0, 3.0, 4.0])
    if keep_row:
        first_values = first_values[None, :]
        second_values = second_values[None, :]
    first = scp.NDDataset(first_values)
    second = scp.NDDataset(second_values)
    first.x = first_x
    second.x = second_x

    aligned_first, aligned_second = scp.align(first, second, method="inner")

    np.testing.assert_array_equal(aligned_first.x.data, [2.0, 3.0])
    np.testing.assert_array_equal(aligned_second.x.data, [2.0, 3.0])
    np.testing.assert_array_equal(aligned_first.x.labels, ["b", "c"])
    if second_labels is None:
        assert aligned_second.x.labels is None
    else:
        np.testing.assert_array_equal(aligned_second.x.labels, ["q", "r"])
    np.testing.assert_array_equal(
        (aligned_first - aligned_second).data.ravel(), [18.0, 27.0]
    )


def test_align_inner_rejects_disjoint_coordinates():
    first = scp.NDDataset([10.0, 20.0], coordset=[scp.Coord([1.0, 2.0])])
    second = scp.NDDataset([30.0, 40.0], coordset=[scp.Coord([3.0, 4.0])])

    with pytest.raises(ValueError, match="No shared coordinates along dimension 'x'"):
        scp.align(first, second, method="inner")


@pytest.mark.parametrize(
    ("first_grid", "second_grid"),
    [
        ([1000.0, 1000.001, 1001.0], [1000.0, 1000.001, 1001.0]),
        ([1000.0, 1000.001, 1001.0], [1000.0, 1001.0]),
        ([1000.0, 1001.0], [1000.0, 1000.001, 1001.0]),
    ],
)
def test_align_inner_rejects_colliding_rounded_coordinates(first_grid, second_grid):
    first = scp.NDDataset(first_grid, coordset=[scp.Coord(first_grid)])
    second = scp.NDDataset(second_grid, coordset=[scp.Coord(second_grid)])

    with pytest.raises(ValueError, match="Coordinate rounding.*dimension 'x'"):
        scp.align(first, second, method="inner")


def test_align_inner_preserves_each_spectrum_mask():
    first = scp.NDDataset(
        [10.0, 20.0, 30.0],
        mask=[True, False, True],
        coordset=[scp.Coord([1.0, 2.0, 3.0])],
    )
    second = scp.NDDataset(
        [2.0, 3.0, 4.0],
        mask=[True, False, False],
        coordset=[scp.Coord([2.0, 3.0, 4.0])],
    )

    aligned_first, aligned_second = scp.align(first, second, method="inner")

    np.testing.assert_array_equal(aligned_first.data, [20.0, 30.0])
    np.testing.assert_array_equal(aligned_second.data, [2.0, 3.0])
    np.testing.assert_array_equal(aligned_first.mask, [False, True])
    np.testing.assert_array_equal(aligned_second.mask, [True, False])


def test_align_inner_preserves_complex_values():
    first = scp.NDDataset(
        [10.0 + 1.0j, 20.0 + 2.0j, 30.0 + 3.0j],
        coordset=[scp.Coord([1.0, 2.0, 3.0])],
    )
    second = scp.NDDataset([2.0, 3.0, 4.0], coordset=[scp.Coord([2.0, 3.0, 4.0])])

    aligned_first, aligned_second = scp.align(first, second, method="inner")

    np.testing.assert_array_equal(aligned_first.data, [20.0 + 2.0j, 30.0 + 3.0j])
    np.testing.assert_array_equal(aligned_second.data, [2.0, 3.0])


def test_align_inner_preserves_integer_values_and_labels():
    first = scp.NDDataset(
        np.array([1, 2**53 + 1, 3], dtype=np.int64),
        dtype=np.int64,
        coordset=[scp.Coord([1.0, 2.0, 3.0], labels=np.array([100, 200, 300]))],
    )
    second = scp.NDDataset([2.0, 3.0, 4.0], coordset=[scp.Coord([2.0, 3.0, 4.0])])

    aligned_first, aligned_second = scp.align(first, second, method="inner")

    assert aligned_first.data.dtype == np.int64
    assert aligned_first.data.tolist() == [2**53 + 1, 3]
    np.testing.assert_array_equal(aligned_first.x.labels, [200, 300])
    assert aligned_second.x.labels is None


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.complex64])
def test_align_inner_keeps_exact_sentinel_for_narrow_dtypes(dtype):
    first = scp.NDDataset(
        np.array([10.0, np.nan, 30.0], dtype=dtype),
        dtype=dtype,
        mask=[False, True, False],
        coordset=[scp.Coord([3.0, 2.0, 1.0])],
    )
    second = scp.NDDataset([2.0, 3.0, 4.0], coordset=[scp.Coord([2.0, 3.0, 4.0])])

    aligned_first, _ = scp.align(first, second, method="inner")

    missing = aligned_first.x.data == 2.0
    np.testing.assert_array_equal(aligned_first.mask, missing)
    np.testing.assert_array_equal(aligned_first.data[missing], [99999999999999.0])
    np.testing.assert_array_equal(aligned_first.data[~missing], [10.0])


@pytest.mark.parametrize("dtype", [object, "U3", "S3"])
@pytest.mark.parametrize(
    ("other_grid", "expected"),
    [([2.0, 3.0, 4.0], [20.0, 30.0]), ([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])],
)
def test_align_inner_accepts_numeric_arrays_convertible_to_float(
    dtype, other_grid, expected
):
    first = scp.NDDataset(
        np.array([10, 20, 30], dtype=dtype),
        dtype=dtype,
        coordset=[scp.Coord([1.0, 2.0, 3.0])],
    )
    second = scp.NDDataset([2.0, 3.0, 4.0], coordset=[scp.Coord(other_grid)])

    aligned_first, _ = scp.align(first, second, method="inner")

    np.testing.assert_array_equal(aligned_first.data, expected)


def test_align_rejects_incompatible_coord_units_with_clear_message():
    ds1 = scp.NDDataset([1.0, 2.0], coordset=[scp.Coord([100.0, 200.0], units="cm^-1")])
    ds2 = scp.NDDataset([3.0, 4.0], coordset=[scp.Coord([1.0, 2.0], units="s")])

    with pytest.raises(UnitsCompatibilityError) as exc:
        align(ds1, ds2, dim="x")
    message = str(exc.value)
    assert "Cannot align datasets" in message
    assert "dimension 'x'" in message
    assert "s" in message
    assert "cm" in message
    assert "Convert the coordinates to compatible units before retrying." in message


@pytest.mark.parametrize("method", ["first", "last"])
@pytest.mark.parametrize("dim", ["x", "y"])
def test_align_reference_join_uses_only_matching_points(method, dim):
    coordinates = scp.Coord([0.0, 1.0])

    def spectra(grid, values):
        axis = scp.Coord(grid)
        if dim == "x":
            return scp.NDDataset(np.array(values).T, coordset=[coordinates, axis])
        return scp.NDDataset(values, coordset=[axis, coordinates])

    first = spectra([1.0, 2.0, 3.0], [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    second = spectra([2.0, 3.0, 4.0], [[40.0, 41.0], [50.0, 51.0], [60.0, 61.0]])

    left, right = scp.align(first, second, dim=dim, method=method)

    actual = right if method == "first" else left
    data = actual.data if dim == "y" else actual.data.T
    mask = actual.mask if dim == "y" else actual.mask.T
    if method == "first":
        np.testing.assert_array_equal(data[1:], [[40.0, 41.0], [50.0, 51.0]])
        np.testing.assert_array_equal(mask[:, 0], [True, False, False])
    else:
        np.testing.assert_array_equal(data[:2], [[20.0, 21.0], [30.0, 31.0]])
        np.testing.assert_array_equal(mask[:, 0], [False, False, True])


def test_align_outer_preserves_complex_order_and_independent_labels():
    first = scp.NDDataset(
        [30.0 + 3j, 10.0 + 1j, 20.0 + 2j],
        coordset=[scp.Coord([3.0, 1.0, 2.0], labels=np.array([300, 100, 200]))],
    )
    second = scp.NDDataset([40.0, 41.0], coordset=[scp.Coord([4.0, 5.0])])

    left, right = scp.align(first, second, method="outer")

    np.testing.assert_array_equal(left.x.data, [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(left.data[:3], [10.0 + 1j, 20.0 + 2j, 30.0 + 3j])
    np.testing.assert_array_equal(left.mask, [False, False, False, True, True])
    assert left.x.labels.tolist() == [100, 200, 300, "--", "--"]
    assert right.x.labels is None


@pytest.mark.parametrize("method", ["linear", "pchip"])
@pytest.mark.parametrize("dims", [["x", "y"], ["y", "x"]])
def test_align_interpolation_masks_outside_range_on_both_axes(method, dims):
    reference = scp.NDDataset(
        np.ones((3, 3)),
        coordset=[scp.Coord([1.0, 2.0, 3.0]), scp.Coord([1.0, 2.0, 3.0])],
    )
    source = scp.NDDataset(
        np.ones((3, 3)),
        coordset=[scp.Coord([2.0, 3.0, 4.0]), scp.Coord([2.0, 3.0, 4.0])],
    )

    _, aligned = scp.align(
        reference, source, dims=dims, method="interpolate", interpolate_method=method
    )

    np.testing.assert_array_equal(
        aligned.mask, [[True, True, True], [True, False, False], [True, False, False]]
    )
    np.testing.assert_allclose(aligned.masked_data.compressed(), np.ones(4))


@pytest.mark.parametrize("method", ["first", "last"])
def test_align_reference_join_masks_entirely_disjoint_spectrum(method):
    first = scp.NDDataset([10.0, 20.0], coordset=[scp.Coord([2.0, 1.0])])
    second = scp.NDDataset([30.0, 40.0], coordset=[scp.Coord([4.0, 3.0])])

    left, right = scp.align(first, second, method=method)

    reference, other = (left, right) if method == "first" else (right, left)
    expected_grid = [2.0, 1.0] if method == "first" else [4.0, 3.0]
    np.testing.assert_array_equal(reference.x.data, expected_grid)
    np.testing.assert_array_equal(other.x.data, expected_grid)
    assert other.mask.all()


@pytest.mark.parametrize("single_input", [False, True])
def test_align_interpolate_uses_explicit_target_for_each_input(single_input):
    first = scp.NDDataset(
        [10.0, 20.0, 30.0], coordset=[scp.Coord([1.0, 2.0, 3.0], units="cm")]
    )
    second = scp.NDDataset(
        [20.0, 40.0, 60.0], coordset=[scp.Coord([1.0, 2.0, 3.0], units="cm")]
    )
    target = scp.Coord([15.0, 25.0], units="mm")

    inputs = (first,) if single_input else (first, second)
    outputs = scp.align(*inputs, coord=target, method="interpolate")

    assert len(outputs) == len(inputs)
    for factor, output in enumerate(outputs, start=1):
        np.testing.assert_allclose(output.x.data, [1.5, 2.5])
        np.testing.assert_allclose(output.data, np.array([15.0, 25.0]) * factor)
    np.testing.assert_array_equal(target.data, [15.0, 25.0])


@pytest.mark.parametrize("coords", [None, [1.0, np.nan, 3.0], [1.0, np.inf, 3.0]])
def test_align_rejects_missing_or_nonfinite_coordinates(coords):
    first = scp.NDDataset([10.0, 20.0, 30.0])
    if coords is not None:
        first.x = scp.Coord(coords)
    second = scp.NDDataset([10.0, 20.0, 30.0], coordset=[scp.Coord([1.0, 2.0, 3.0])])

    with pytest.raises(ValueError, match="numeric coordinates along dimension 'x'"):
        scp.align(first, second, method="outer")


def test_align_outer_retains_small_coordinates_when_union_expands_the_range():
    first = scp.NDDataset([10.0, 20.0], coordset=[scp.Coord([0.00014, 0.00116])])
    second = scp.NDDataset([30.0, 40.0], coordset=[scp.Coord([1.0, 2.0])])
    third = scp.NDDataset([50.0, 60.0], coordset=[scp.Coord([2.0, 3.0])])

    left, _, _ = scp.align(first, second, third, method="outer")

    np.testing.assert_array_equal(left.x.data, [0.00014, 0.00116, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(left.mask, [False, False, True, True, True])
    np.testing.assert_array_equal(left.masked_data.compressed(), [10.0, 20.0])


def test_align_interpolate_rejects_nonadjacent_duplicate_source_coordinates():
    first = scp.NDDataset([10.0, 20.0], coordset=[scp.Coord([1.0, 2.0])])
    second = scp.NDDataset([30.0, 40.0, 50.0], coordset=[scp.Coord([1.0, 2.0, 1.0])])

    with pytest.raises(ValueError, match="Duplicate or near-duplicate"):
        scp.align(first, second, method="interpolate")


def test_align_linear_interpolation_keeps_exact_singleton_row():
    x = scp.Coord([1.0, 2.0, 3.0])
    reference = scp.NDDataset(np.ones((2, 3)), coordset=[scp.Coord([2.0, 3.0]), x])
    source = scp.NDDataset([[3.0, 4.0, 5.0]], coordset=[scp.Coord([2.0]), x])

    _, aligned = scp.align(reference, source, dim="y", method="interpolate")

    np.testing.assert_array_equal(aligned.mask, [[False] * 3, [True] * 3])
    np.testing.assert_array_equal(aligned.masked_data.compressed(), [3.0, 4.0, 5.0])


@pytest.mark.parametrize("method", ["outer", "first", "last"])
def test_align_keeps_distinct_coordinates_on_wide_grids(method):
    first = scp.NDDataset([10.0, 20.0], coordset=[scp.Coord([1.0, 1e5])])
    second = scp.NDDataset([30.0, 40.0], coordset=[scp.Coord([2.0, 1e5])])

    left, right = scp.align(first, second, method=method)

    if method == "outer":
        np.testing.assert_array_equal(left.x.data, [1.0, 2.0, 1e5])
        np.testing.assert_array_equal(left.mask, [False, True, False])
        np.testing.assert_array_equal(right.mask, [True, False, False])
    else:
        target, aligned = (left, right) if method == "first" else (right, left)
        np.testing.assert_array_equal(
            target.x.data, [1.0 if method == "first" else 2.0, 1e5]
        )
        np.testing.assert_array_equal(aligned.mask, [True, False])
        assert aligned.data[-1] == (40.0 if method == "first" else 20.0)


@pytest.mark.parametrize("method", ["outer", "first", "last"])
def test_align_matches_converted_units_with_rounding_disabled(method):
    first = scp.NDDataset(
        [10.0, 20.0],
        coordset=[scp.Coord([0.007, 0.01], units="m", rounding=False)],
    )
    second = scp.NDDataset(
        [30.0, 40.0],
        coordset=[scp.Coord([0.7, 1.0], units="cm", rounding=False)],
    )

    left, right = scp.align(first, second, method=method)

    np.testing.assert_array_equal(left.x.data, right.x.data)
    np.testing.assert_array_equal(left.data, [10.0, 20.0])
    np.testing.assert_array_equal(right.data, [30.0, 40.0])
    assert not left.is_masked and not right.is_masked


def test_align_mixed_units_preserves_distinct_unrounded_points():
    first = scp.NDDataset(
        [10.0, 20.0],
        coordset=[scp.Coord([1e5, 1e5 + 1e-10], units="m", rounding=False)],
    )
    second = scp.NDDataset(
        [30.0, 40.0], coordset=[scp.Coord([5.0, 6.0], units="cm", rounding=False)]
    )

    left, _ = scp.align(first, second, method="outer")

    np.testing.assert_array_equal(left.x.data, [0.05, 0.06, 1e5, 1e5 + 1e-10])
    np.testing.assert_array_equal(left.masked_data.compressed(), [10.0, 20.0])


@pytest.mark.parametrize("method", ["outer", "first", "last"])
def test_align_rounded_slices_match_small_spectroscopic_coordinate_drift(method):
    first = scp.NDDataset([10.0, 20.0], coordset=[scp.Coord([3500.685, 3502.685])])
    second = scp.NDDataset([30.0, 40.0], coordset=[scp.Coord([3500.686, 3502.686])])

    left, right = scp.align(first, second, method=method)

    np.testing.assert_array_equal(left.x.data, right.x.data)
    np.testing.assert_array_equal(left.data, [10.0, 20.0])
    np.testing.assert_array_equal(right.data, [30.0, 40.0])
    assert not left.is_masked and not right.is_masked


def test_align_does_not_merge_unrounded_neighbors_after_unit_conversion():
    first = scp.NDDataset(
        [10.0, 20.0],
        coordset=[scp.Coord([1e5, 1e6], units="m", rounding=False)],
    )
    second = scp.NDDataset(
        [30.0, 40.0],
        coordset=[scp.Coord([(1e5 + 1e-10) * 100, 2e8], units="cm", rounding=False)],
    )

    left, right = scp.align(first, second, method="outer")

    assert len(left.x.data) == 4
    np.testing.assert_array_equal(left.mask, [False, True, False, True])
    np.testing.assert_array_equal(right.mask, [True, False, True, False])


def test_align_rejects_coordinate_overflow_after_unit_conversion():
    first = scp.NDDataset(
        [10.0, 20.0],
        coordset=[scp.Coord([1.0, 2.0], units="nm", rounding=False)],
    )
    second = scp.NDDataset(
        [30.0, 40.0],
        coordset=[scp.Coord([1e290, 1e308], units="m", rounding=False)],
    )

    with pytest.raises(ValueError, match="non-finite coordinates"):
        scp.align(first, second, method="outer")


@pytest.mark.parametrize("method", ["outer", "first", "last"])
@pytest.mark.parametrize(
    ("first_grid", "first_unit", "second_grid", "second_unit"),
    [
        ([1000.0, 2000.0], "cm^-1", [10000.0, 5000.0], "nm"),
        ([1.0, 2.0], "absorbance", [0.1, 0.01], "absolute_transmittance"),
    ],
)
def test_align_matches_unrounded_coordinates_with_contextual_units(
    method, first_grid, first_unit, second_grid, second_unit
):
    first = scp.NDDataset(
        [10.0, 20.0],
        coordset=[scp.Coord(first_grid, units=first_unit, rounding=False)],
    )
    second = scp.NDDataset(
        [30.0, 40.0],
        coordset=[scp.Coord(second_grid, units=second_unit, rounding=False)],
    )

    left, right = scp.align(first, second, method=method)

    np.testing.assert_array_equal(left.x.data, right.x.data)
    np.testing.assert_array_equal(right.data - left.data, [20.0, 20.0])
    assert not left.is_masked and not right.is_masked


@pytest.mark.parametrize("method", ["outer", "first", "last", "interpolate"])
def test_align_matches_unrounded_feet_and_meters_on_either_reference(method):
    metric = scp.NDDataset(
        [10.0, 20.0], coordset=[scp.Coord([0.3048, 0.6096], units="m", rounding=False)]
    )
    imperial = scp.NDDataset(
        [30.0, 40.0], coordset=[scp.Coord([1.0, 2.0], units="ft", rounding=False)]
    )
    for inputs in [(metric, imperial), (imperial, metric)]:
        left, right = scp.align(*inputs, method=method)
        np.testing.assert_array_equal(left.x.data, right.x.data)
        np.testing.assert_array_equal(left.data, inputs[0].data)
        np.testing.assert_array_equal(right.data, inputs[1].data)
        assert not left.is_masked and not right.is_masked


def test_align_preserves_distinct_unrounded_neighbors_across_spectral_units():
    first = scp.NDDataset(
        [10.0], coordset=[scp.Coord([1000.0], units="cm^-1", rounding=False)]
    )
    second = scp.NDDataset(
        [20.0],
        coordset=[
            scp.Coord([np.nextafter(10000.0, np.inf)], units="nm", rounding=False)
        ],
    )

    left, right = scp.align(first, second, method="outer")

    assert left.x.size == right.x.size == 2
    assert np.count_nonzero(left.mask) == np.count_nonzero(right.mask) == 1
    assert not np.any(left.mask == right.mask)


@pytest.mark.parametrize("method", ["outer", "first", "interpolate"])
def test_align_does_not_round_converted_coordinates_again(method):
    first = scp.NDDataset([10.0, 20.0], coordset=[scp.Coord([0.0, 1000.0], units="m")])
    second = scp.NDDataset([30.0, 40.0], coordset=[scp.Coord([1.0, 1e12], units="nm")])

    left, right = scp.align(first, second, method=method)

    if method == "outer":
        np.testing.assert_allclose(left.x.data, [0.0, 1e-9, 1000.0], atol=0.0)
        np.testing.assert_array_equal(right.mask, [True, False, False])
        np.testing.assert_array_equal(right.masked_data.compressed(), [30.0, 40.0])
    else:
        np.testing.assert_array_equal(right.x.data, [0.0, 1000.0])
        np.testing.assert_array_equal(right.mask, [True, False])
        np.testing.assert_array_equal(right.masked_data.compressed(), [40.0])


@pytest.mark.parametrize("method", ["linear", "pchip"])
@pytest.mark.parametrize("explicit_target", [False, True])
def test_align_interpolate_keeps_converted_endpoints_with_default_rounding(
    method, explicit_target
):
    reference = scp.NDDataset(
        [10.0, 20.0], coordset=[scp.Coord([0.3048, 0.6096], units="m")]
    )
    source = scp.NDDataset([30.0, 40.0], coordset=[scp.Coord([1.0, 2.0], units="ft")])
    target = scp.Coord([1.0, 2.0], units="ft") if explicit_target else None

    left, right = scp.align(
        reference, source, method="interpolate", interpolate_method=method, coord=target
    )

    np.testing.assert_array_equal(right.x.data, reference.x.data)
    np.testing.assert_array_equal(left.data, reference.data)
    np.testing.assert_array_equal(right.data, source.data)
    assert not left.is_masked and not right.is_masked


@pytest.mark.parametrize("method", ["outer", "first", "last"])
def test_align_preserves_extended_precision_complex_intensities(method):
    first = scp.NDDataset(
        np.array([1.0 + 3.0j, 2.0 + 4.0j], dtype=np.clongdouble),
        dtype=np.clongdouble,
        coordset=[scp.Coord([0.0, 1.0])],
    )
    second = scp.NDDataset([30.0, 40.0], coordset=[scp.Coord([0.0, 1.0])])

    left, _ = scp.align(first, second, method=method)

    assert left.data.dtype == first.data.dtype
    np.testing.assert_array_equal(left.data, first.data)


@pytest.mark.parametrize("method", ["outer", "first", "last"])
def test_align_matches_mixed_coordinate_dtypes_without_losing_integer_precision(method):
    first = scp.NDDataset(
        [10.0, 20.0],
        coordset=[scp.Coord([2**53, 2**53 + 1], dtype=np.int64, rounding=False)],
    )
    second = scp.NDDataset(
        [30.0, 40.0], coordset=[scp.Coord([1e18, 2e18], rounding=False)]
    )
    if (
        method == "outer"
        and np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant
    ):
        with pytest.raises(ValueError, match="losing coordinate precision"):
            scp.align(first, second, method=method)
        return

    left, right = scp.align(first, second, method=method)

    if method == "outer":
        assert [int(point) for point in left.x.data[:2]] == [2**53, 2**53 + 1]
        np.testing.assert_array_equal(left.mask, [False, False, True, True])
        np.testing.assert_array_equal(right.mask, [True, True, False, False])
    elif method == "first":
        assert [int(point) for point in left.x.data] == [2**53, 2**53 + 1]
        assert right.mask.all()
    else:
        np.testing.assert_array_equal(right.x.data, [1e18, 2e18])
        assert left.mask.all()
    if method != "last":
        np.testing.assert_array_equal(
            np.ma.asarray(left.masked_data).compressed(), [10.0, 20.0]
        )
    if method != "first":
        np.testing.assert_array_equal(
            np.ma.asarray(right.masked_data).compressed(), [30.0, 40.0]
        )


@pytest.mark.parametrize("method", ["outer", "first", "last"])
def test_align_converts_large_integer_coordinates_exactly(method):
    meters = [2**53, 2**53 + 1]
    first = scp.NDDataset(
        [10.0, 20.0],
        coordset=[scp.Coord(meters, units="m", dtype=np.int64, rounding=False)],
    )
    second = scp.NDDataset(
        [30.0, 40.0],
        coordset=[
            scp.Coord(
                [100 * point for point in meters],
                units="cm",
                dtype=np.int64,
                rounding=False,
            )
        ],
    )

    left, right = scp.align(first, second, method=method)

    expected = second.x.data if method == "last" else first.x.data
    np.testing.assert_array_equal(left.x.data, expected)
    np.testing.assert_array_equal(right.x.data, expected)
    np.testing.assert_array_equal(left.data, first.data)
    np.testing.assert_array_equal(right.data, second.data)
    assert not left.is_masked and not right.is_masked


@pytest.mark.parametrize("method", ["outer", "first", "last"])
def test_align_rounds_large_integer_keys_without_coercing_to_double(method):
    points = [2**53, 2**53 + 1]
    first = scp.NDDataset(
        [10.0, 20.0],
        coordset=[scp.Coord(points, dtype=np.int64, sigdigits=17)],
    )
    second = scp.NDDataset(
        [30.0, 40.0],
        coordset=[scp.Coord([2.0, 3.0], sigdigits=17)],
    )
    if (
        method == "outer"
        and np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant
    ):
        with pytest.raises(ValueError, match="losing coordinate precision"):
            scp.align(first, second, method=method)
        return

    left, right = scp.align(first, second, method=method)

    if method != "last":
        assert [int(point) for point in left.x.data[-2:]] == points
        np.testing.assert_array_equal(
            np.ma.asarray(left.masked_data).compressed(), first.data
        )
    if method != "first":
        np.testing.assert_array_equal(
            np.ma.asarray(right.masked_data).compressed(), second.data
        )


@pytest.mark.parametrize("method", ["outer", "first", "last"])
@pytest.mark.parametrize("dtype", [np.int64, np.longdouble])
def test_align_preserves_wide_intensities_when_inserting_missing_points(method, dtype):
    points = (
        [2**53 + 1, 2**53 + 3]
        if dtype is np.int64
        else [np.nextafter(np.longdouble(1), np.longdouble(2)), np.longdouble(2)]
    )
    source = scp.NDDataset(points, dtype=dtype, coordset=[scp.Coord([0.0, 1.0])])
    reference = scp.NDDataset([30.0, 40.0, 50.0], coordset=[scp.Coord([0.0, 1.0, 2.0])])
    inputs = [source, reference] if method == "last" else [reference, source]

    result = scp.align(*inputs, method=method)[0 if method == "last" else 1]

    assert result.data.dtype == source.data.dtype
    np.testing.assert_array_equal(result.data[:2], source.data)
    np.testing.assert_array_equal(result.mask, [False, False, True])
    assert result.data[2] == 99999999999999


@pytest.mark.parametrize("method", ["linear", "pchip"])
@pytest.mark.parametrize("dim", ["x", "y"])
def test_align_interpolate_does_not_expose_sentinel_from_outer_join(method, dim):
    unchanged = scp.Coord([0.0, 1.0])

    def spectrum(data, points):
        sampled = scp.Coord(points)
        if dim == "y":
            return scp.NDDataset(np.asarray(data).T, coordset=[sampled, unchanged])
        return scp.NDDataset(data, coordset=[unchanged, sampled])

    source = spectrum([[1.0, 3.0], [2.0, 6.0]], [1.0, 3.0])
    other = spectrum([[2.0, 3.0], [4.0, 6.0]], [2.0, 3.0])
    joined, _ = scp.align(source, other, dim=dim, method="outer")
    target = scp.Coord([1.0, 1.25, 2.0, 2.75, 3.0])

    (result,) = scp.align(
        joined, dim=dim, method="interpolate", interpolate_method=method, coord=target
    )

    if dim == "y":
        result = result.T
    np.testing.assert_array_equal(result.mask, [[False, True, True, True, False]] * 2)
    np.testing.assert_array_equal(result.masked_data.compressed(), [1.0, 3.0, 2.0, 6.0])
    assert np.all(result.data[result.mask] == 99999999999999)


def test_align_pchip_masks_intervals_adjacent_to_a_masked_sample():
    source = scp.NDDataset(
        [0.0, 1.0, np.nan, 9.0, 16.0, 25.0],
        mask=[False, False, True, False, False, False],
        coordset=[scp.Coord([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])],
    )
    target = scp.Coord([0.5, 1.0, 3.5, 4.0, 4.5])

    (result,) = scp.align(
        source, method="interpolate", interpolate_method="pchip", coord=target
    )

    np.testing.assert_array_equal(result.mask, [True, False, True, False, False])
    np.testing.assert_array_equal(result.data[[1, 3]], [1.0, 16.0])
    assert 16.0 < result.data[-1] < 25.0
