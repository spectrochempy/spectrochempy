# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Regression tests for ``align(..., interpolate_sampling=...)``."""

import numpy as np
import pytest

import spectrochempy as scp


def _datasets(*, identical_grids=False, masked=False):
    first_x = scp.Coord([0.0, 2.0, 4.0], title="distance", units="m")
    if identical_grids:
        second_x = scp.Coord([0.0, 2.0, 4.0], title="distance", units="m")
        second_data = [2.0, 6.0, 10.0]
    else:
        second_x = scp.Coord([0.0, 100.0, 300.0, 400.0], title="distance", units="cm")
        second_data = [1.0, 3.0, 7.0, 9.0]

    first = scp.NDDataset(
        [1.0, 5.0, 9.0],
        coordset=[first_x],
        title="signal",
        units="absorbance",
        meta={"source": "first"},
    )
    second = scp.NDDataset(
        second_data,
        coordset=[second_x],
        title="signal",
        units="absorbance",
        meta={"source": "second"},
    )
    first.history = "first source"
    second.history = "second source"
    if masked:
        first.mask = [False, True, False]
        second.mask = [False, False, True, False]
    return first, second


def _assert_unchanged(dataset, before):
    np.testing.assert_array_equal(dataset.data, before.data)
    np.testing.assert_array_equal(dataset.mask, before.mask)
    np.testing.assert_array_equal(dataset.x.data, before.x.data)
    assert dataset.shape == before.shape
    assert dataset.dims == before.dims
    assert dataset.units == before.units
    assert dataset.title == before.title
    assert dataset.x.units == before.x.units
    assert dataset.x.title == before.x.title
    assert dataset.meta == before.meta
    assert dataset.history_entries == before.history_entries


def test_align_interpolate_default_and_auto_use_first_grid():
    first, second = _datasets()
    first_before = first.copy()
    second_before = second.copy()

    default = scp.align(first, second, dim="x", method="interpolate")
    automatic = first.align(
        second,
        dim="x",
        method="interpolate",
        interpolate_sampling="auto",
    )

    for first_result, second_result in (default, automatic):
        np.testing.assert_allclose(first_result.data, [1.0, 5.0, 9.0])
        np.testing.assert_allclose(second_result.data, [1.0, 5.0, 9.0])
        np.testing.assert_array_equal(first_result.x.data, [0.0, 2.0, 4.0])
        np.testing.assert_array_equal(second_result.x.data, [0.0, 2.0, 4.0])
        assert first_result.x.units == scp.ur.m
        assert second_result.x.units == scp.ur.m

    _assert_unchanged(first, first_before)
    _assert_unchanged(second, second_before)


def test_align_interpolate_auto_preserves_identical_grids():
    first, second = _datasets(identical_grids=True)

    first_result, second_result = scp.align(
        first,
        second,
        dim="x",
        method="interpolate",
        interpolate_sampling="auto",
    )

    np.testing.assert_array_equal(first_result.data, first.data)
    np.testing.assert_array_equal(second_result.data, second.data)
    np.testing.assert_array_equal(first_result.x.data, first.x.data)
    np.testing.assert_array_equal(second_result.x.data, first.x.data)


@pytest.mark.parametrize("sampling", [2, 0.5, 0, False, None])
def test_align_rejects_unsupported_sampling_without_mutation(sampling):
    first, second = _datasets(masked=True)
    first_before = first.copy()
    second_before = second.copy()

    with pytest.raises(
        NotImplementedError,
        match="only supports interpolate_sampling='auto'",
    ):
        scp.align(
            first,
            second,
            dim="x",
            method="interpolate",
            interpolate_sampling=sampling,
        )

    _assert_unchanged(first, first_before)
    _assert_unchanged(second, second_before)


def test_align_noninterpolating_method_rejects_sampling_request():
    first, second = _datasets(masked=True)
    first_before = first.copy()
    second_before = second.copy()

    with pytest.raises(
        NotImplementedError,
        match="only supports interpolate_sampling='auto'",
    ):
        scp.align(first, second, method="outer", interpolate_sampling=2)

    _assert_unchanged(first, first_before)
    _assert_unchanged(second, second_before)
