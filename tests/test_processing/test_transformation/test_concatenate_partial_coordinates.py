# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: S101

"""Regression tests for partially present concatenation coordinates."""

import warnings

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.utils.exceptions import DimensionsCompatibilityError


def _dataset(rows, offset, coordinate_kind, *, masked_row=None):
    data = np.arange(offset, offset + rows * 2, dtype=float).reshape(rows, 2)
    mask = np.zeros_like(data, dtype=bool)
    if masked_row is not None:
        mask[masked_row, 1] = True

    dataset = scp.NDDataset(np.ma.MaskedArray(data, mask=mask))
    x = scp.Coord([100.0, 200.0], title="spectral", units="cm^-1")
    dataset.name = f"block-{offset}"
    labels = [f"row-{offset + index}" for index in range(rows)]

    if coordinate_kind == "numeric":
        y = scp.Coord(
            np.arange(offset, offset + rows, dtype=float),
            labels=labels,
            title="observation",
            units="s",
        )
        dataset.set_coordset(y=y, x=x)
    elif coordinate_kind == "labels":
        dataset.set_coordset(
            y=scp.Coord(labels=labels, title="observation"),
            x=x,
        )
    elif coordinate_kind == "empty":
        dataset.set_coordset(y=scp.Coord(None, size=rows), x=x)
    elif coordinate_kind == "absent":
        dataset.set_coordset(x=x)
    else:
        raise ValueError(f"Unknown coordinate kind: {coordinate_kind}")

    return dataset


def _assert_sources_unchanged(sources, snapshots):
    for source, snapshot in zip(sources, snapshots, strict=True):
        np.testing.assert_array_equal(source.data, snapshot.data)
        np.testing.assert_array_equal(source.mask, snapshot.mask)
        assert source.coordset == snapshot.coordset


def test_concatenate_complete_coordinates_preserves_geometry_and_masks():
    first = _dataset(2, 10, "numeric", masked_row=1)
    second = _dataset(3, 20, "numeric", masked_row=2)
    snapshots = [first.copy(), second.copy()]

    result = scp.concatenate(first, second, dims="y")

    assert result.shape == (5, 2)
    np.testing.assert_array_equal(
        result.data,
        np.concatenate([first.data, second.data], axis=0),
    )
    np.testing.assert_array_equal(
        result.mask,
        np.concatenate([first.mask, second.mask], axis=0),
    )
    np.testing.assert_array_equal(result.y.data, [10.0, 11.0, 20.0, 21.0, 22.0])
    assert result.y.labels.tolist() == [
        "row-10",
        "row-11",
        "row-20",
        "row-21",
        "row-22",
    ]
    assert result.y.size == result.shape[0]
    assert result.x == first.x
    assert result[-1].y.data.tolist() == [22.0]
    assert result[-1].y.labels.tolist() == ["row-22"]
    _assert_sources_unchanged([first, second], snapshots)


def test_concatenate_without_dimension_coordinates_preserves_other_axis():
    first = _dataset(2, 10, "absent")
    second = _dataset(3, 20, "absent")

    result = scp.concatenate(first, second, dims="y")

    assert result.shape == (5, 2)
    assert result.coord("y") is None
    assert result.x == first.x
    np.testing.assert_array_equal(result.data[-1], second.data[-1])


def test_concatenate_empty_dimension_coordinates_remains_empty():
    first = _dataset(2, 10, "empty")
    second = _dataset(3, 20, "empty")

    result = scp.concatenate(first, second, dims="y")

    assert result.shape == (5, 2)
    assert result.y.is_empty
    assert result.y.data is None
    assert result.y.labels is None
    assert result.x == first.x


def test_concatenate_label_only_coordinates_are_valid():
    first = _dataset(2, 10, "labels")
    second = _dataset(3, 20, "labels")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = scp.concatenate(first, second, dims="y")

    assert not caught
    assert result.shape == (5, 2)
    assert result.y.data is None
    assert result.y.labels.tolist() == [
        "row-10",
        "row-11",
        "row-20",
        "row-21",
        "row-22",
    ]
    assert result.y.size == result.shape[0]
    assert result[-1].y.labels.tolist() == ["row-22"]


@pytest.mark.parametrize(
    "coordinate_kinds",
    [
        ("numeric", "empty"),
        ("empty", "numeric"),
        ("numeric", "absent", "numeric"),
    ],
)
def test_concatenate_rejects_partially_present_coordinates(coordinate_kinds):
    sources = [
        _dataset(index + 2, 10 * (index + 1), kind, masked_row=0)
        for index, kind in enumerate(coordinate_kinds)
    ]
    snapshots = [source.copy() for source in sources]

    with pytest.raises(
        DimensionsCompatibilityError,
        match=(
            "coordinates.*dimension 'y'.*provided by every input or removed from "
            "all inputs"
        ),
    ):
        scp.concatenate(*sources, dims="y")

    _assert_sources_unchanged(sources, snapshots)


def test_concatenate_rejects_mixed_numeric_and_label_only_coordinates():
    numeric = _dataset(2, 10, "numeric")
    labels = _dataset(3, 20, "labels")

    with pytest.raises(
        DimensionsCompatibilityError,
        match="coordinate representations.*dimension 'y'",
    ):
        scp.concatenate(numeric, labels, dims="y")


def test_stack_path_remains_valid_with_missing_existing_coordinates():
    first = _dataset(2, 10, "absent")
    second = _dataset(2, 20, "absent")

    result = scp.stack(first, second)

    assert result.shape == (2, 2, 2)
    assert result[result.dims[0]].labels.tolist() == [first.name, second.name]
    assert result.x == first.x
