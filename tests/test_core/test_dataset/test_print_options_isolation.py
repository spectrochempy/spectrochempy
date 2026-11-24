# ======================================================================================
# Copyright (c) 2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Displaying objects must not alter NumPy's global print options (#1974)."""

import numpy as np
import pytest

import spectrochempy as scp


@pytest.fixture
def user_printoptions():
    saved = np.get_printoptions()
    np.set_printoptions(threshold=1000)
    yield
    np.set_printoptions(**saved)


def test_display_keeps_global_numpy_printoptions(user_printoptions):
    ds = scp.NDDataset(np.random.default_rng(0).random((6, 20)))
    ds.y = scp.Coord(labels=list("abcdef"))
    before = np.get_printoptions()
    for obj in (ds, ds.x, ds.y, ds.coordset):
        str(obj)
        repr(obj)
        if hasattr(obj, "_repr_html_"):
            obj._repr_html_()
    assert np.get_printoptions() == before
    assert np.get_printoptions()["threshold"] == 1000
    assert str(np.arange(12.0)) == "[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11.]"


def test_display_output_independent_of_user_printoptions(user_printoptions):
    ds = scp.NDDataset(np.arange(40.0).reshape(4, 10))
    text = str(ds)
    np.set_printoptions(threshold=10, precision=1, edgeitems=1)
    assert str(ds) == text
