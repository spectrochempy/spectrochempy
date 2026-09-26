# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np
import pytest

from spectrochempy import Coord
from spectrochempy import NDDataset
from spectrochempy.processing.fft.shift import cs
from spectrochempy.processing.fft.shift import ls
from spectrochempy.processing.fft.shift import roll
from spectrochempy.processing.fft.shift import rs
from spectrochempy.utils.testing import assert_array_equal


def test_rs_shifts_last_axis():
    """rs uses ``axis=-1`` so 2D input shifts only the last dim."""
    data = np.arange(12.0).reshape(3, 4)
    ds = NDDataset(data)
    shifted = rs(ds, pts=2)
    expected = np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 4, 5],
            [0, 0, 8, 9],
        ]
    )
    assert_array_equal(shifted.data, expected)


def test_ls_shifts_last_axis():
    """ls uses ``axis=-1`` so 2D input shifts only the last dim."""
    data = np.arange(12.0).reshape(3, 4)
    ds = NDDataset(data)
    shifted = ls(ds, pts=2)
    expected = np.array(
        [
            [2, 3, 0, 0],
            [6, 7, 0, 0],
            [10, 11, 0, 0],
        ]
    )
    assert_array_equal(shifted.data, expected)


def test_roll_shifts_last_axis():
    """roll uses ``axis=-1`` so 2D input shifts only the last dim."""
    data = np.arange(12.0).reshape(3, 4)
    ds = NDDataset(data)
    shifted = roll(ds, pts=2)
    expected = np.array(
        [
            [2, 3, 0, 1],
            [6, 7, 4, 5],
            [10, 11, 8, 9],
        ]
    )
    assert_array_equal(shifted.data, expected)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_zero_left_shift_preserves_values(explicit, inplace):
    source = NDDataset(np.array([1.0, 2.0, 3.0, 4.0]))
    original = source.data.copy()
    kwargs = {"inplace": inplace}
    if explicit:
        kwargs["pts"] = 0

    result = source.ls(**kwargs)

    assert_array_equal(result.data, original)
    assert (result is source) is inplace


def test_roll_moves_mask_with_values_and_preserves_masked_mean():
    source = NDDataset(np.array([1.0, 99.0, 3.0, 4.0]))
    source.mask = np.array([False, True, False, False])
    source_data = source.data.copy()
    source_mask = source.mask.copy()
    source_mean = np.ma.array(source.data, mask=source.mask).mean()

    result = source.roll(pts=1)

    assert_array_equal(result.data, [4.0, 1.0, 99.0, 3.0])
    assert_array_equal(result.mask, [False, False, True, False])
    assert np.ma.array(result.data, mask=result.mask).mean() == source_mean
    assert_array_equal(source.data, source_data)
    assert_array_equal(source.mask, source_mask)


@pytest.mark.parametrize(
    ("method", "expected_data", "expected_mask"),
    [
        (rs, [0.0, 1.0, 99.0, 3.0], [False, False, True, False]),
        (ls, [99.0, 3.0, 4.0, 0.0], [True, False, False, False]),
        (cs, [4.0, 1.0, 99.0, 3.0], [False, False, True, False]),
    ],
)
def test_discrete_shifts_move_mask_and_leave_zero_fill_unmasked(
    method, expected_data, expected_mask
):
    source = NDDataset(np.array([1.0, 99.0, 3.0, 4.0]))
    source.mask = np.array([False, True, False, False])

    result = method(source, pts=1)

    assert_array_equal(result.data, expected_data)
    assert_array_equal(result.mask, expected_mask)
    assert result.shape == source.shape


def test_roll_moves_mask_on_selected_axis_and_preserves_source():
    data = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    mask = np.array([[False, True, False], [True, False, False]])
    source = NDDataset(
        data.copy(),
        coordset=[Coord([0.0, 1.0]), Coord([10.0, 20.0, 30.0])],
    )
    source.mask = mask.copy()

    result = source.roll(pts=1, axis=0)

    assert_array_equal(result.data, [[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])
    assert_array_equal(result.mask, [[True, False, False], [False, True, False]])
    assert result.shape == source.shape
    assert result.dims == source.dims
    assert_array_equal(source.data, data)
    assert_array_equal(source.mask, mask)


def test_roll_moves_mask_inplace():
    source = NDDataset(np.array([1.0, 99.0, 3.0, 4.0]))
    source.mask = np.array([False, True, False, False])

    result = source.roll(pts=1, inplace=True)

    assert result is source
    assert_array_equal(result.data, [4.0, 1.0, 99.0, 3.0])
    assert_array_equal(result.mask, [False, False, True, False])


def test_negated_roll_moves_mask_without_negating_it():
    source = NDDataset(np.array([1.0, 99.0, 3.0, 4.0]))
    source.mask = np.array([False, True, False, False])

    result = source.roll(pts=1, neg=True)

    assert_array_equal(result.data, [-4.0, 1.0, 99.0, 3.0])
    assert_array_equal(result.mask, [False, False, True, False])


@pytest.mark.parametrize("name", ["roll", "cs"])
@pytest.mark.parametrize("inplace", [False, True])
def test_zero_negated_circular_shift_is_noop_with_single_history(name, inplace):
    source = NDDataset(np.array([1.0, 99.0, 3.0, 4.0]))
    source.mask = np.array([False, True, False, False])
    source.annotate("Synthetic source")
    original_data = source.data.copy()
    original_mask = source.mask.copy()
    original_history = source.history_entries

    result = getattr(source, name)(pts=0, neg=True, inplace=inplace)

    assert (result is source) is inplace
    assert_array_equal(result.data, original_data)
    assert_array_equal(result.mask, original_mask)
    assert result.history_entries[:-1] == original_history
    assert len(result.history_entries) == len(original_history) + 1
    assert "`roll` shift performed" in result.history_entries[-1]["message"]
    if not inplace:
        assert_array_equal(source.data, original_data)
        assert_array_equal(source.mask, original_mask)
        assert source.history_entries == original_history


def test_cs_matches_roll_and_adds_one_history_entry():
    source = NDDataset(np.array([1.0, 99.0, 3.0, 4.0]))
    source.mask = np.array([False, True, False, False])
    source.annotate("Synthetic source")
    original_history = source.history_entries

    rolled = source.roll(pts=1, neg=True)
    shifted = source.cs(pts=1, neg=True)

    assert_array_equal(shifted.data, rolled.data)
    assert_array_equal(shifted.mask, rolled.mask)
    for result in (rolled, shifted):
        assert result.history_entries[:-1] == original_history
        assert len(result.history_entries) == len(original_history) + 1
        assert "`roll` shift performed" in result.history_entries[-1]["message"]
    assert source.history_entries == original_history
