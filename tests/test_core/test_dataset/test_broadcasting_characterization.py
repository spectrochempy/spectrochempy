# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Characterize coordinate-aware NDDataset broadcasting before its redesign."""

import copy as cpy
import operator

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.core.units import DimensionalityError
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


def _observable_state(dataset):
    """Capture arithmetic state that NDDataset equality deliberately ignores."""
    coordset = cpy.deepcopy(dataset._coordset)
    defaults = {}
    if dataset._coordset is not None:
        defaults = {
            dim: getattr(dataset._coordset[dim], "default_index", None)
            for dim in dataset.dims
        }
    return {
        "data_object": dataset._data,
        "data": dataset._data.copy(),
        "dtype": dataset.dtype,
        "mask_object": dataset._mask,
        "mask": np.asarray(dataset.mask).copy(),
        "units_object": dataset._units,
        "units": dataset.units,
        "dims_object": dataset._dims,
        "dims": list(dataset.dims),
        "coordset_object": dataset._coordset,
        "coordset": coordset,
        "references": (
            None if dataset._coordset is None else dict(dataset._coordset.references)
        ),
        "defaults": defaults,
        "title": dataset.title,
        "name": dataset.name,
        "meta_object": dataset._meta,
        "meta": cpy.deepcopy(dataset.meta),
        "history_object": dataset._history,
        "history": list(dataset._history),
        "rendered_history": list(dataset.history),
        "author": dataset.author,
        "description": dataset.description,
        "origin": dataset.origin,
        "created": dataset._created,
        "modified": dataset._modified,
        "transposed": dataset._transposed,
    }


def _assert_observable_state(dataset, before):
    """Assert values, metadata, dates, and backing objects are unchanged."""
    assert dataset._data is before["data_object"]
    np.testing.assert_array_equal(dataset.data, before["data"])
    assert dataset.dtype == before["dtype"]
    assert dataset._mask is before["mask_object"]
    np.testing.assert_array_equal(np.asarray(dataset.mask), before["mask"])
    assert dataset._units is before["units_object"]
    assert dataset.units == before["units"]
    assert dataset._dims is before["dims_object"]
    assert dataset.dims == before["dims"]
    assert dataset._coordset is before["coordset_object"]
    if before["coordset"] is None:
        assert dataset._coordset is None
    else:
        assert dataset._coordset == before["coordset"]
        assert dataset._coordset.references == before["references"]
        assert {
            dim: getattr(dataset._coordset[dim], "default_index", None)
            for dim in dataset.dims
        } == before["defaults"]
    assert dataset.title == before["title"]
    assert dataset.name == before["name"]
    assert dataset._meta is before["meta_object"]
    assert dataset.meta == before["meta"]
    assert dataset._history is before["history_object"]
    assert dataset._history == before["history"]
    assert dataset.history == before["rendered_history"]
    assert dataset.author == before["author"]
    assert dataset.description == before["description"]
    assert dataset.origin == before["origin"]
    assert dataset._created == before["created"]
    assert dataset._modified == before["modified"]
    assert dataset._transposed == before["transposed"]


def _stateful_dataset(data, *, dims=None, coordset=None, mask=None, units="m"):
    kwargs = {
        "units": units,
        "title": "signal",
        "name": "atomicity-target",
    }
    if dims is not None:
        kwargs["dims"] = dims
    if coordset is not None:
        kwargs["coordset"] = coordset
    if mask is not None:
        kwargs["mask"] = mask
    dataset = scp.NDDataset(data, **kwargs)
    dataset.author = "maintainer"
    dataset.description = "in-place atomicity regression"
    dataset.origin = "synthetic"
    dataset.meta.experiment = {"kind": "atomicity", "replicate": 1}
    dataset.history = "Initial state"
    return dataset


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


def test_rank_expanding_broadcast_reconstructs_result_dimensions():
    matrix = scp.NDDataset(np.arange(1.0, 13.0).reshape(3, 4), dims=["y", "x"])
    profile = scp.NDDataset([1.0, 2.0, 3.0, 4.0], dims=["x"])

    forward = matrix - profile
    reversed_ = profile - matrix

    np.testing.assert_allclose(forward.data, matrix.data - profile.data)
    np.testing.assert_allclose(reversed_.data, profile.data - matrix.data)
    assert forward.shape == reversed_.shape == (3, 4)
    assert forward.dims == reversed_.dims == ["y", "x"]
    assert len(forward.dims) == forward.ndim
    assert len(reversed_.dims) == reversed_.ndim


def test_broadcast_dimension_name_collision_is_rejected():
    column = scp.NDDataset([[1.0], [2.0], [3.0]], dims=["y", "x"])
    same_named_profile = scp.NDDataset([1.0, 2.0, 3.0, 4.0], dims=["y"])

    with pytest.raises(ValueError, match="collision.*'y'.*axes 0/1"):
        column * same_named_profile
    with pytest.raises(ValueError, match="collision.*'y'.*axes 0/1"):
        same_named_profile * column


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


def test_complementary_singleton_axes_use_non_singleton_provider_coordinates():
    samples, variables = _axes()
    column = scp.NDDataset(
        [[1.0], [2.0], [3.0]],
        dims=["y", "x"],
        coordset=[samples, scp.Coord([999.0], title="slot")],
    )
    row = scp.NDDataset(
        [[1.0, 2.0, 3.0, 4.0]],
        dims=["y", "x"],
        coordset=[scp.Coord(labels=["reference"]), variables],
    )

    for result in (column * row, row * column):
        assert result.shape == (3, 4)
        assert result.ndim == 2
        assert result.dims == ["y", "x"]
        np.testing.assert_allclose(result.data, column.data * row.data)
        np.testing.assert_allclose(result.y.data, samples.data)
        np.testing.assert_allclose(result.x.data, variables.data)

    np.testing.assert_allclose(column.x.data, [999.0])
    assert row.y.labels.tolist() == ["reference"]


def test_empty_singleton_coordinate_inherits_the_expanding_axis():
    _, column, profile = _requested_operands()

    expected = column.data * profile.data
    for result in (
        column * profile,
        profile * column,
        np.multiply(column, profile),
        np.multiply(profile, column),
    ):
        assert result.shape == (3, 4)
        assert result.ndim == 2
        assert result.dims == ["y", "x"]
        assert result.y.size == 3
        assert result.x.size == 4
        np.testing.assert_allclose(result.data, expected)
        np.testing.assert_allclose(result.y.data, column.y.data)
        np.testing.assert_allclose(result.x.data, profile.x.data)


def test_absent_coordinate_inherits_the_expanding_axis_coordinate():
    _, variables = _axes()
    column = scp.NDDataset([[1.0], [2.0], [3.0]], dims=["y", "x"])
    profile = scp.NDDataset(
        [1.0, 2.0, 3.0, 4.0],
        dims=["x"],
        coordset=[variables],
    )

    result = column * profile

    assert result.shape == (3, 4)
    assert result.y.is_empty
    np.testing.assert_allclose(result.x.data, variables.data)


@pytest.mark.parametrize(
    "operation, expected",
    [
        pytest.param(
            lambda left, right: left - right,
            lambda left, right: left - right,
            id="subtract",
        ),
        pytest.param(
            lambda left, right: left / right,
            lambda left, right: left / right,
            id="divide",
        ),
    ],
)
def test_non_commutative_broadcast_preserves_operand_order(operation, expected):
    column = scp.NDDataset([[2.0], [4.0], [8.0]], dims=["y", "x"])
    profile = scp.NDDataset([1.0, 2.0, 4.0, 8.0], dims=["x"])

    forward = operation(column, profile)
    reversed_ = operation(profile, column)

    np.testing.assert_allclose(forward.data, expected(column.data, profile.data))
    np.testing.assert_allclose(reversed_.data, expected(profile.data, column.data))
    assert forward.dims == reversed_.dims == ["y", "x"]


def test_numeric_and_labeled_singleton_coordinates_can_expand():
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

    numeric_results = (
        value_column * numeric_profile,
        numeric_profile * value_column,
    )
    for result in numeric_results:
        assert result.shape == (3, 4)
        assert result.dims == ["y", "x"]
        np.testing.assert_allclose(
            result.data, value_column.data * numeric_profile.data
        )
        np.testing.assert_allclose(result.y.data, samples.data)
        np.testing.assert_allclose(result.x.data, variables.data)

    label_results = (label_column * label_profile, label_profile * label_column)
    for result in label_results:
        assert result.shape == (3, 4)
        assert result.dims == ["y", "x"]
        np.testing.assert_allclose(result.data, label_column.data * label_profile.data)
        assert result.y.data.tolist() == samples.data.tolist()
        assert result.x.labels.tolist() == ["a", "b", "c", "d"]

    numeric_results[0].x[0] = 42.0
    label_results[0].x.labels[0] = "changed"

    np.testing.assert_allclose(value_column.x.data, [999.0])
    np.testing.assert_allclose(numeric_profile.x.data, variables.data)
    assert label_column.x.labels.tolist() == ["slot"]
    assert label_profile.x.labels.tolist() == ["a", "b", "c", "d"]


def test_same_dimension_singleton_group_can_expand_without_recursion():
    samples, variables = _axes()
    singleton_group = scp.CoordSet(
        scp.Coord([999.0], title="slot value"),
        scp.Coord(labels=["slot"], title="slot label"),
        sorted=False,
    )
    provider_group = scp.CoordSet(
        variables,
        scp.Coord(labels=["a", "b", "c", "d"], title="bands"),
        sorted=False,
    )
    provider_group.select(2)
    column = scp.NDDataset(
        [[1.0], [2.0], [3.0]],
        dims=["y", "x"],
        coordset=[samples, singleton_group],
    )
    profile = scp.NDDataset([1.0, 2.0, 3.0, 4.0], dims=["x"], coordset=[provider_group])

    for result in (column * profile, profile * column):
        assert result.shape == (3, 4)
        assert result.dims == ["y", "x"]
        np.testing.assert_allclose(result.data, column.data * profile.data)
        assert result.x.is_same_dim
        assert result.x.names == profile.x.names
        assert result.x.default_index == profile.x.default_index
        np.testing.assert_allclose(result.x.default.data, profile.x.default.data)
        assert result.x is not profile.x

    assert column.x.sizes == 1
    np.testing.assert_allclose(column.x.coords[1].data, [999.0])
    assert column.x.coords[0].labels.tolist() == ["slot"]


def test_selected_reference_spectrum_broadcasts_without_squeeze():
    samples, variables = _axes()
    matrix = scp.NDDataset(
        np.arange(12.0).reshape(3, 4),
        dims=["y", "x"],
        coordset=[samples, variables],
    )
    selected = matrix[0]

    matrix_before = matrix.copy()
    selected_before = selected.copy()
    operations = (
        (matrix - selected, matrix.data - selected.data),
        (selected - matrix, selected.data - matrix.data),
        (np.subtract(matrix, selected), matrix.data - selected.data),
        (np.subtract(selected, matrix), selected.data - matrix.data),
    )

    for result, expected in operations:
        assert result.shape == matrix.shape
        assert result.ndim == matrix.ndim
        assert result.dims == matrix.dims
        np.testing.assert_allclose(result.data, expected)
        np.testing.assert_allclose(result.y.data, matrix.y.data)
        np.testing.assert_allclose(result.x.data, matrix.x.data)

    operations[0][0].y[0] = 42.0
    np.testing.assert_allclose(matrix.data, matrix_before.data)
    np.testing.assert_allclose(selected.data, selected_before.data)
    np.testing.assert_allclose(matrix.y.data, matrix_before.y.data)
    np.testing.assert_allclose(selected.y.data, selected_before.y.data)
    assert selected.shape == (1, 4)


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


@pytest.mark.parametrize(
    "right_units",
    [
        pytest.param("m^-1", id="same-dimension-units"),
        pytest.param("s", id="incompatible-units"),
        pytest.param(None, id="unit-on-one-side"),
    ],
)
def test_other_coordinate_unit_policies_remain_unresolved(right_units):
    """The same-unit fix does not define other coordinate-unit policies."""
    left_coord = scp.Coord([1.0, 2.0, 3.0, 4.0], units="cm^-1")
    right_coord = scp.Coord([1.0, 2.0, 3.0, 4.0], units=right_units)
    left = scp.NDDataset(np.ones(4), dims=["x"], coordset=[left_coord])
    right = scp.NDDataset(np.ones(4), dims=["x"], coordset=[right_coord])

    result = left + right

    np.testing.assert_array_equal(result.data, np.full(4, 2.0))
    np.testing.assert_array_equal(result.x.data, left_coord.data)


def test_one_sided_coordinate_unit_still_uses_raw_value_validation():
    left = scp.NDDataset(
        np.ones(4),
        dims=["x"],
        coordset=[scp.Coord([1.0, 2.0, 3.0, 4.0], units="cm^-1")],
    )
    right = scp.NDDataset(
        np.ones(4),
        dims=["x"],
        coordset=[scp.Coord([2.0, 3.0, 4.0, 5.0])],
    )

    with pytest.raises(CoordinatesMismatchError):
        left + right


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


def test_matching_unitless_coordinates_are_accepted():
    coordinate = scp.Coord([1000.0, 1100.0, 1200.0, 1300.0])
    left = scp.NDDataset(np.ones(4), dims=["x"], coordset=[coordinate])
    right = scp.NDDataset(np.full(4, 2.0), dims=["x"], coordset=[coordinate.copy()])

    result = left + right

    np.testing.assert_array_equal(result.data, np.full(4, 3.0))


def test_identical_same_unit_coordinates_are_accepted():
    coordinate = scp.Coord(
        [1000.0, 1100.0, 1200.0, 1300.0],
        units="cm^-1",
    )
    left = scp.NDDataset(np.ones(4), dims=["x"], coordset=[coordinate])
    right = scp.NDDataset(np.full(4, 2.0), dims=["x"], coordset=[coordinate.copy()])
    left_before = left.x.copy()
    right_before = right.x.copy()

    operator_result = left + right
    ufunc_result = np.add(left, right)

    np.testing.assert_array_equal(operator_result.data, np.full(4, 3.0))
    np.testing.assert_array_equal(ufunc_result.data, np.full(4, 3.0))
    np.testing.assert_array_equal(left.x.data, left_before.data)
    np.testing.assert_array_equal(right.x.data, right_before.data)
    assert left.x.units == left_before.units
    assert right.x.units == right_before.units


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda left, right: left + right, id="operator"),
        pytest.param(lambda left, right: right + left, id="reversed-operator"),
        pytest.param(lambda left, right: np.add(left, right), id="ufunc"),
        pytest.param(lambda left, right: np.add(right, left), id="reversed-ufunc"),
    ],
)
def test_incompatible_same_unit_coordinates_are_rejected(operation):
    """Same-unit coordinate differences fail the last-axis guard."""
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
    left_before = left.x.copy()
    right_before = right.x.copy()

    with pytest.raises(CoordinatesMismatchError):
        operation(left, right)

    np.testing.assert_array_equal(left.x.data, left_before.data)
    np.testing.assert_array_equal(right.x.data, right_before.data)
    assert left.x.units == left_before.units
    assert right.x.units == right_before.units


def test_same_unit_coordinate_validation_preserves_existing_tolerance():
    """Coord precision exposes differences on either side of decimal=3."""
    left_coord = scp.Coord([0.0, 1.0], units="cm^-1")
    close_coord = scp.Coord([0.0014, 1.0014], units="cm^-1")
    far_coord = scp.Coord([0.0016, 1.0016], units="cm^-1")
    left = scp.NDDataset(np.ones(2), dims=["x"], coordset=[left_coord])
    close = scp.NDDataset(np.ones(2), dims=["x"], coordset=[close_coord])
    far = scp.NDDataset(np.ones(2), dims=["x"], coordset=[far_coord])

    np.testing.assert_array_equal(close.x.data, [0.001, 1.001])
    np.testing.assert_array_equal(far.x.data, [0.002, 1.002])
    np.testing.assert_array_equal((left + close).data, np.full(2, 2.0))
    with pytest.raises(CoordinatesMismatchError):
        left + far


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
    labeled_variables = scp.Coord(
        variables.data.copy(),
        labels=["a", "b", "c", "d"],
        title=variables.title,
        units=variables.units,
    )
    left = scp.NDDataset(
        np.ones((3, 4)),
        dims=["y", "x"],
        coordset=[samples, labeled_variables],
    )
    right = scp.NDDataset(
        np.ones((3, 4)),
        dims=["y", "x"],
        coordset=[samples.copy(), labeled_variables.copy()],
    )

    result = left + right

    assert result.y is not left.y
    assert result.x is not left.x
    assert result.y is not right.y
    assert result.x is not right.x
    result.x[0] = 42.0
    result.x.labels[0] = "changed"
    assert left.x.data[0] == 1000.0
    assert right.x.data[0] == 1000.0
    assert left.x.labels.tolist() == ["a", "b", "c", "d"]
    assert right.x.labels.tolist() == ["a", "b", "c", "d"]


def test_inherited_coordinate_values_and_labels_are_deep_copied():
    samples, variables = _axes()
    labeled_variables = scp.Coord(
        variables.data.copy(),
        labels=["a", "b", "c", "d"],
        title=variables.title,
        units=variables.units,
    )
    column = scp.NDDataset(
        [[1.0], [2.0], [3.0]],
        dims=["y", "x"],
        coordset=[samples, None],
    )
    profile = scp.NDDataset(
        [1.0, 2.0, 3.0, 4.0],
        dims=["x"],
        coordset=[labeled_variables],
    )

    result = column * profile
    result.x[0] = 42.0
    result.x.labels[0] = "changed"

    assert profile.x.data[0] == 1000.0
    assert profile.x.labels.tolist() == ["a", "b", "c", "d"]
    assert column.y.data[0] == 0


def test_inherited_same_dimension_coordinate_group_keeps_default():
    samples, variables = _axes()
    alternate = scp.Coord([9.0, 10.0, 11.0, 12.0], title="index")
    multi = scp.CoordSet(variables, alternate, sorted=False)
    multi.select(2)
    column = scp.NDDataset(
        [[1.0], [2.0], [3.0]],
        dims=["y", "x"],
        coordset=[samples, None],
    )
    profile = scp.NDDataset([1.0, 2.0, 3.0, 4.0], dims=["x"], coordset=[multi])

    result = column * profile

    assert result.x.is_same_dim
    assert result.x.names == profile.x.names
    assert result.x.default_index == profile.x.default_index
    assert result.x is not profile.x


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


def test_references_between_selected_axes_survive_mixed_geometry():
    shared = scp.Coord([1.0, 2.0], title="shared")
    left = scp.NDDataset(
        np.ones((2, 2, 1)),
        dims=["z", "y", "x"],
        coordset=scp.CoordSet(z="y", y=shared, x=None),
    )
    profile = scp.NDDataset(
        [1.0, 2.0, 3.0],
        dims=["x"],
        coordset=[scp.Coord([10.0, 20.0, 30.0])],
    )

    for result in (left * profile, profile * left):
        assert result.shape == (2, 2, 3)
        assert result.dims == ["z", "y", "x"]
        assert result.coordset.references == {"z": "y"}
        np.testing.assert_allclose(result.z.data, result.y.data)
        np.testing.assert_allclose(result.x.data, profile.x.data)


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


def test_failed_expanding_inplace_broadcast_is_atomic():
    column = _stateful_dataset(
        [[1.0], [2.0], [3.0]],
        dims=["y", "x"],
        mask=[[False], [True], [False]],
    )
    profile = _stateful_dataset([1.0, 2.0, 3.0, 4.0], dims=["x"])
    column_before = _observable_state(column)
    profile_before = _observable_state(profile)

    with pytest.raises(ArithmeticError, match="non-broadcastable output operand"):
        column *= profile

    _assert_observable_state(column, column_before)
    _assert_observable_state(profile, profile_before)


def test_failed_inplace_incompatible_shapes_are_atomic():
    target = _stateful_dataset(np.arange(6.0).reshape(2, 3), dims=["y", "x"])
    other = _stateful_dataset(np.ones((2, 2)), dims=["y", "x"])
    target_before = _observable_state(target)
    other_before = _observable_state(other)

    with pytest.raises(ArithmeticError, match="could not be broadcast"):
        target += other

    _assert_observable_state(target, target_before)
    _assert_observable_state(other, other_before)


def test_failed_inplace_same_unit_coordinate_mismatch_is_atomic():
    target_coord = scp.Coord([1000.0, 1001.0, 1002.0], units="cm^-1")
    other_coord = scp.Coord([1000.0, 1001.0, 1002.01], units="cm^-1")
    target = _stateful_dataset([1.0, 2.0, 3.0], dims=["x"], coordset=[target_coord])
    other = _stateful_dataset([3.0, 2.0, 1.0], dims=["x"], coordset=[other_coord])
    target_before = _observable_state(target)
    other_before = _observable_state(other)

    with pytest.raises(CoordinatesMismatchError):
        target -= other

    _assert_observable_state(target, target_before)
    _assert_observable_state(other, other_before)


def test_failed_inplace_incompatible_units_are_atomic():
    target = _stateful_dataset([1.0, 2.0], units="m")
    other = _stateful_dataset([1.0, 2.0], units="s")
    target_before = _observable_state(target)
    other_before = _observable_state(other)

    with pytest.raises(DimensionalityError):
        target += other

    _assert_observable_state(target, target_before)
    _assert_observable_state(other, other_before)


def test_failed_inplace_numeric_conversion_is_atomic_and_preserves_shared_buffer():
    source = np.array([1.0, 2.0])
    target = _stateful_dataset(source)
    incompatible = np.array(["not-a-number", "still-not-a-number"])
    source_before = source.copy()
    incompatible_before = incompatible.copy()
    target_before = _observable_state(target)
    assert np.shares_memory(target.data, source)

    with pytest.raises(TypeError):
        target += incompatible

    _assert_observable_state(target, target_before)
    np.testing.assert_array_equal(source, source_before)
    np.testing.assert_array_equal(incompatible, incompatible_before)


def test_late_inplace_trait_notification_failure_rolls_back_complete_state():
    primary = scp.Coord([1000.0, 1001.0], units="cm^-1")
    alternate = scp.Coord([10.0, 20.0], title="alternate")
    spectral_group = scp.CoordSet(primary, alternate, sorted=False)
    coordset = scp.CoordSet(x=spectral_group, y="x")
    target = _stateful_dataset(
        [[1.0, 2.0], [3.0, 4.0]],
        dims=["y", "x"],
        coordset=coordset,
        mask=[[False, False], [False, False]],
    )
    other = _stateful_dataset(
        [[1.0, 1.0], [1.0, 1.0]],
        dims=["y", "x"],
        coordset=coordset.copy(),
        mask=[[False, True], [False, False]],
    )
    target_before = _observable_state(target)
    other_before = _observable_state(other)

    def fail_after_assignment(_change):
        raise RuntimeError("controlled late trait notification failure")

    target.observe(fail_after_assignment, names="_mask")
    try:
        with pytest.raises(RuntimeError, match="controlled late trait"):
            target += other
    finally:
        target.unobserve(fail_after_assignment, names="_mask")

    _assert_observable_state(target, target_before)
    _assert_observable_state(other, other_before)


@pytest.mark.parametrize(
    ("inplace_op", "plain_op", "scalar"),
    [
        (operator.iadd, operator.add, 2.0),
        (operator.isub, operator.sub, 2.0),
        (operator.imul, operator.mul, 2.0),
        (operator.itruediv, operator.truediv, 2.0),
    ],
)
def test_successful_inplace_scalar_operations_commit_once(inplace_op, plain_op, scalar):
    samples = scp.Coord([0.0, 1.0], title="sample")
    variables = scp.Coord([1000.0, 1001.0], units="cm^-1")
    target = _stateful_dataset(
        [[2.0, 4.0], [6.0, 8.0]],
        dims=["y", "x"],
        coordset=[samples, variables],
    )
    expected = plain_op(target.copy(), scalar)
    identity = id(target)
    history_length = len(target._history)
    coordset = target._coordset
    meta = target._meta

    result = inplace_op(target, scalar)

    assert result is target
    assert id(target) == identity
    np.testing.assert_allclose(target.data, expected.data)
    np.testing.assert_array_equal(target.mask, expected.mask)
    assert target.dtype == expected.dtype
    assert target.units == expected.units
    assert target.title == expected.title
    assert target._coordset is coordset
    assert target._meta is meta
    assert len(target._history) == history_length + 1
    assert "Inplace binary op" in target.history[-1]


def test_successful_inplace_selected_reference_preserves_geometry_and_masks():
    samples, variables = _axes()
    values = np.arange(1.0, 13.0).reshape(3, 4)
    mask = np.zeros_like(values, dtype=bool)
    mask[0, 1] = True
    mask[2, 3] = True
    target = _stateful_dataset(
        values.copy(),
        dims=["y", "x"],
        coordset=[samples, variables],
        mask=mask,
    )
    reference = target[0]
    target_coordset = target._coordset
    target_dims = target._dims
    target_meta = target._meta
    reference_before = _observable_state(reference)
    history_length = len(target._history)
    expected_mask = mask | np.broadcast_to(reference.mask, target.shape)
    expected_data = values - reference.data

    result = operator.isub(target, reference)

    assert result is target
    np.testing.assert_allclose(target.data, expected_data)
    np.testing.assert_array_equal(target.mask, expected_mask)
    assert target.units == scp.ur.m
    assert target._coordset is target_coordset
    assert target._dims is target_dims
    assert target._meta is target_meta
    np.testing.assert_array_equal(target.y.data, samples.data)
    np.testing.assert_array_equal(target.x.data, variables.data)
    assert len(target._history) == history_length + 1
    _assert_observable_state(reference, reference_before)
