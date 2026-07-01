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
