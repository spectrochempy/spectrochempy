# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np
import pytest

from spectrochempy import diag
from spectrochempy.processing.transformation.npy import dot


def test_npy(ds1):
    # functions that keep units

    # DIAG
    with pytest.raises(ValueError):
        df = diag(ds1)  # work only for 1d or 2D dataset

    ds = ds1[0].squeeze()
    assert ds.ndim == 2
    df = diag(ds)
    assert df.units == ds1.units
    assert df.ndim == 1
    assert df.size == ds.x.size

    d = ds[0].squeeze()
    assert d.ndim == 1
    df = diag(d)
    assert df.units == ds1.units
    assert df.ndim == 2
    assert df.size == d.x.size**2

    df = diag(ds.data)
    assert df._implements("NDDataset")

    # DOT
    a = ds  # 2D dataset
    b = ds1[3].squeeze()  # second 2D dataset
    b.ito("km", force=True)  # put some units to b
    x = dot(a.T, b)
    assert x.units == a.units * b.units
    assert x.shape == (a.x.size, b.x.size)

    # allow mixing numpy object with dataset
    x = dot(a.T, b.data)
    assert x.units == a.units

    # if no dataset then is it equivalent to np.dot
    x = dot(a.data.T, b.data)
    assert isinstance(x, np.ndarray)
