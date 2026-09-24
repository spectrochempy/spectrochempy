# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Characterize coordinate-aware NDDataset broadcasting before its redesign."""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.utils.exceptions import CoordinatesMismatchError


def _axes():
    samples = scp.Coord.arange(3, title="sample")
    variables = scp.Coord.linspace(
        1000.0,
        1200.0,
        4,
        title="wavenumber",
        units="cm^-1",
    )
    return samples, variables


def _requested_operands():
    samples, variables = _axes()
    concentration = scp.linspace(
        0.1,
        0.3,
        3,
        coordset=[samples],
        dims=["y"],
    )
    column = concentration.reshape((3, 1), dims=("y", "x"))
    profile = scp.linspace(
        0.0,
        1.0,
        4,
        coordset=[variables],
        dims=["x"],
    )
    return concentration, column, profile


def test_reshape_adds_an_empty_coordinate_for_the_new_singleton_dimension():
    concentration, column, _ = _requested_operands()

    assert column.shape == (3, 1)
    assert column.dims == ["y", "x"]
    np.testing.assert_array_equal(column.y.data, concentration.y.data)
    assert column.y is not concentration.y
    assert column.x.is_empty
    assert column.x.data is None
    assert column.x.labels is None


def test_scalar_and_trailing_vector_broadcast_keep_left_geometry():
    samples, variables = _axes()
    matrix = scp.NDDataset(
        np.arange(1.0, 13.0).reshape(3, 4),
        dims=["y", "x"],
        coordset=[samples, variables],
    )
    profile = scp.NDDataset(
        [1.0, 2.0, 3.0, 4.0],
        dims=["x"],
        coordset=[variables.copy()],
    )

    scaled = matrix * scp.NDDataset(2.0)
    multiplied = matrix * profile

    assert scaled.shape == multiplied.shape == (3, 4)
    assert scaled.dims == multiplied.dims == ["y", "x"]
    np.testing.assert_allclose(multiplied.data, matrix.data * profile.data)
    np.testing.assert_allclose(multiplied.y.data, samples.data)
    np.testing.assert_allclose(multiplied.x.data, variables.data)


def test_rank_expanding_broadcast_currently_keeps_stale_left_dimensions():
    """A 1D left operand can currently return 2D data with only one dim."""
    matrix = scp.NDDataset(np.arange(1.0, 13.0).reshape(3, 4), dims=["y", "x"])
    profile = scp.NDDataset([1.0, 2.0, 3.0, 4.0], dims=["x"])

    forward = matrix - profile
    reversed_ = profile - matrix

    np.testing.assert_allclose(forward.data, matrix.data - profile.data)
    np.testing.assert_allclose(reversed_.data, profile.data - matrix.data)
    assert forward.shape == reversed_.shape == (3, 4)
    assert forward.dims == ["y", "x"]
    assert reversed_.dims == ["x"]
    assert len(reversed_.dims) != reversed_.ndim


def test_column_and_vector_broadcast_without_coordinates():
    column = scp.NDDataset([[1.0], [2.0], [3.0]], dims=["y", "x"])
    profile = scp.NDDataset([1.0, 2.0, 3.0, 4.0], dims=["x"])

    result = column * profile

    assert result.shape == (3, 4)
    assert result.dims == ["y", "x"]
    assert result.coordset is None
    np.testing.assert_allclose(
        result.data,
        [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], [3.0, 6.0, 9.0, 12.0]],
    )


def test_two_complementary_singleton_axes_broadcast_without_coordinates():
    column = scp.NDDataset([[1.0], [2.0], [3.0]], dims=["y", "x"])
    row = scp.NDDataset([[1.0, 2.0, 3.0, 4.0]], dims=["y", "x"])

    forward = column * row
    reversed_ = row * column

    assert forward.shape == reversed_.shape == (3, 4)
    assert forward.dims == reversed_.dims == ["y", "x"]
    np.testing.assert_allclose(forward.data, reversed_.data)


def test_requested_empty_singleton_coordinate_is_rejected_before_broadcast():
    _, column, profile = _requested_operands()

    with pytest.raises(CoordinatesMismatchError):
        column * profile
    with pytest.raises(CoordinatesMismatchError):
        profile * column
    with pytest.raises(CoordinatesMismatchError):
        np.multiply(column, profile)


def test_absent_coordinate_is_tolerated_but_secondary_coordinate_is_not_adopted():
    _, variables = _axes()
    column = scp.NDDataset([[1.0], [2.0], [3.0]], dims=["y", "x"])
    profile = scp.NDDataset(
        [1.0, 2.0, 3.0, 4.0],
        dims=["x"],
        coordset=[variables],
    )

    result = column * profile

    assert result.shape == (3, 4)
    assert result.coordset is None


def test_significant_singleton_coordinates_are_not_treated_as_expandable():
    samples, variables = _axes()
    value_column = scp.NDDataset(
        [[1.0], [2.0], [3.0]],
        dims=["y", "x"],
        coordset=[samples, scp.Coord([999.0], title="slot")],
    )
    numeric_profile = scp.NDDataset(
        [1.0, 2.0, 3.0, 4.0],
        dims=["x"],
        coordset=[variables],
    )
    label_column = scp.NDDataset(
        [[1.0], [2.0], [3.0]],
        dims=["y", "x"],
        coordset=[samples, scp.Coord(labels=["slot"], title="slot")],
    )
    label_profile = scp.NDDataset(
        [1.0, 2.0, 3.0, 4.0],
        dims=["x"],
        coordset=[scp.Coord(labels=["a", "b", "c", "d"])],
    )

    with pytest.raises(CoordinatesMismatchError):
        value_column * numeric_profile
    with pytest.raises(CoordinatesMismatchError):
        label_column * label_profile


def test_matching_and_mismatching_label_only_coordinates():
    labels = scp.Coord(labels=["a", "b", "c", "d"], title="bands")
    left = scp.NDDataset(np.ones(4), dims=["x"], coordset=[labels])
    matching = scp.NDDataset(np.ones(4), dims=["x"], coordset=[labels.copy()])
    mismatching = scp.NDDataset(
        np.ones(4),
        dims=["x"],
        coordset=[scp.Coord(labels=["a", "b", "c", "z"], title="bands")],
    )

    result = left + matching

    assert result.x.labels.tolist() == ["a", "b", "c", "d"]
    with pytest.raises(CoordinatesMismatchError):
        left + mismatching


def test_physically_equivalent_coordinates_with_different_units_currently_succeed():
    left_coord = scp.Coord(
        [1000.0, 1100.0, 1200.0, 1300.0],
        units="cm^-1",
    )
    right_coord = scp.Coord(
        [100000.0, 110000.0, 120000.0, 130000.0],
        units="m^-1",
    )
    left = scp.NDDataset(np.ones(4), dims=["x"], coordset=[left_coord])
    right = scp.NDDataset(np.ones(4), dims=["x"], coordset=[right_coord])

    result = left + right

    np.testing.assert_allclose(result.x.data, left_coord.data)
    assert result.x.units == scp.ur("cm^-1")


def test_incompatible_unitless_coordinates_are_rejected():
    left = scp.NDDataset(
        np.ones(4),
        dims=["x"],
        coordset=[scp.Coord([1000.0, 1100.0, 1200.0, 1300.0])],
    )
    right = scp.NDDataset(
        np.ones(4),
        dims=["x"],
        coordset=[scp.Coord([2000.0, 2100.0, 2200.0, 2300.0])],
    )

    with pytest.raises(CoordinatesMismatchError):
        left + right


def test_incompatible_unitful_coordinates_currently_escape_validation():
    """Unit-bearing coordinate differences currently pass the last-axis guard."""
    left_coord = scp.Coord(
        [1000.0, 1100.0, 1200.0, 1300.0],
        units="cm^-1",
    )
    right_coord = scp.Coord(
        [2000.0, 2100.0, 2200.0, 2300.0],
        units="cm^-1",
    )
    left = scp.NDDataset(np.ones(4), dims=["x"], coordset=[left_coord])
    right = scp.NDDataset(np.ones(4), dims=["x"], coordset=[right_coord])

    result = left + right

    np.testing.assert_allclose(result.x.data, left_coord.data)


def test_dimension_names_do_not_align_operands():
    samples, variables = _axes()
    matrix = scp.NDDataset(
        np.ones((3, 4)),
        dims=["y", "x"],
        coordset=[samples, variables],
    )
    differently_named_profile = scp.NDDataset(
        np.arange(4.0),
        dims=["z"],
        coordset=[variables.copy()],
    )

    result = matrix + differently_named_profile

    assert result.shape == (3, 4)
    assert result.dims == ["y", "x"]


def test_incompatible_shapes_raise_after_last_coordinate_validation():
    samples, variables = _axes()
    left = scp.NDDataset(
        np.ones((3, 4)),
        dims=["y", "x"],
        coordset=[samples, variables],
    )
    right = scp.NDDataset(
        np.ones((2, 4)),
        dims=["y", "x"],
        coordset=[scp.Coord([1.0, 2.0]), variables.copy()],
    )

    with pytest.raises(ArithmeticError, match="could not be broadcast"):
        left + right


def test_masks_units_and_sources_follow_existing_broadcast_contract():
    column = scp.NDDataset(
        [[1.0], [2.0], [3.0]],
        dims=["y", "x"],
        mask=[[False], [True], [False]],
        units="mol/L",
    )
    profile = scp.NDDataset(
        [1.0, 2.0, 3.0, 4.0],
        dims=["x"],
        mask=[False, False, True, False],
        units="absorbance",
    )
    column_mask = column.mask.copy()
    profile_mask = profile.mask.copy()

    result = column * profile

    np.testing.assert_array_equal(
        result.mask,
        [
            [False, False, True, False],
            [True, True, True, True],
            [False, False, True, False],
        ],
    )
    assert result.units == column.units * profile.units
    np.testing.assert_array_equal(column.mask, column_mask)
    np.testing.assert_array_equal(profile.mask, profile_mask)


def test_result_coordinates_are_copied_from_the_left_operand():
    samples, variables = _axes()
    left = scp.NDDataset(
        np.ones((3, 4)),
        dims=["y", "x"],
        coordset=[samples, variables],
    )
    right = scp.NDDataset(
        np.ones((3, 4)),
        dims=["y", "x"],
        coordset=[samples.copy(), variables.copy()],
    )

    result = left + right

    assert result.y is not left.y
    assert result.x is not left.x
    assert result.y is not right.y
    assert result.x is not right.x
    result.x = scp.Coord([42.0, 43.0, 44.0, 45.0], title="changed")
    assert left.x.data[0] == 1000.0
    assert right.x.data[0] == 1000.0


def test_equal_same_dimension_coordinate_sets_are_preserved():
    samples, variables = _axes()
    alternate = scp.Coord([9.0, 10.0, 11.0, 12.0], title="index")
    multi = scp.CoordSet(variables, alternate, sorted=False)
    left = scp.NDDataset(
        np.ones((3, 4)),
        dims=["y", "x"],
        coordset=[samples, multi],
    )
    right = left.copy()

    result = left + right

    assert result.x.is_same_dim
    assert result.x.names == left.x.names
    assert result.x.default == left.x.default
    assert result.x is not left.x


def test_shared_coordinate_references_are_copied_with_left_geometry():
    shared = scp.Coord([1.0, 2.0, 3.0, 4.0], title="shared")
    coordset = scp.CoordSet(x=shared, y="x")
    left = scp.NDDataset(
        np.ones((4, 4)),
        dims=["y", "x"],
        coordset=coordset,
    )
    right = left.copy()

    result = left + right

    assert result.coordset.references == {"y": "x"}
    np.testing.assert_allclose(result.y.data, result.x.data)
    assert result.coordset is not left.coordset
    assert result.x is not left.x


def test_inplace_broadcast_requires_the_target_shape_to_stay_unchanged():
    samples, variables = _axes()
    matrix = scp.NDDataset(
        np.ones((3, 4)),
        dims=["y", "x"],
        coordset=[samples, variables],
    )
    profile = scp.NDDataset(
        [1.0, 2.0, 3.0, 4.0],
        dims=["x"],
        coordset=[variables.copy()],
    )
    profile_before = profile.copy()

    matrix *= profile

    assert matrix.shape == (3, 4)
    assert matrix.dims == ["y", "x"]
    np.testing.assert_allclose(matrix.data, np.broadcast_to(profile.data, (3, 4)))
    np.testing.assert_allclose(profile.data, profile_before.data)


def test_failed_expanding_inplace_broadcast_currently_changes_history_only():
    column = scp.NDDataset([[1.0], [2.0], [3.0]], dims=["y", "x"])
    profile = scp.NDDataset([1.0, 2.0, 3.0, 4.0], dims=["x"])
    data_before = column.data.copy()
    history_before = list(column.history)

    with pytest.raises(ArithmeticError, match="non-broadcastable output operand"):
        column *= profile

    assert column.shape == (3, 1)
    assert column.dims == ["y", "x"]
    np.testing.assert_array_equal(column.data, data_before)
    assert column.history != history_before
