# -*- coding: utf-8 -*-
# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import numpy as np
import pytest

from spectrochempy.core.dataset.npy import (identity, eye, full, ones, zeros, dot, diag, empty_like,
                                            zeros_like, ones_like, full_like, )
from spectrochempy.units import ur


def test_npy(ds1):
    # functions that keep units

    df = full_like(ds1, dtype=np.complex128, fill_value=2.5)
    assert df.units == ds1.units

    df = zeros_like(ds1, dtype=np.complex128)
    assert df.units == ds1.units

    df = ones_like(ds1, dtype=np.complex128)
    assert df.units == ds1.units

    df = empty_like(ds1, dtype=np.complex128)
    assert df.units == ds1.units

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
    assert df.size == d.x.size ** 2

    # if no dataset then it is equivalent to np.diag
    df = diag(ds.data)
    assert isinstance(df, np.ndarray)

    # DOT
    a = ds  # 2D dataset
    b = ds1[3].squeeze()  # second 2D dataset
    b.ito('km', force=True)  # put some units to b
    x = dot(a.T, b)
    assert x.units == a.units * b.units
    assert x.shape == (a.x.size, b.x.size)

    # allow mixing numpy object with dataset
    x = dot(a.T, b.data)
    assert x.units == a.units

    # if no dataset then is it equivalent to np.dot
    x = dot(a.data.T, b.data)
    assert isinstance(x, np.ndarray)


def test_creation_function():
    df = zeros((2, 3), dtype='int64', units='km')
    assert df.shape == (2, 3)
    assert df.dtype == 'int64'
    assert df.units == ur.km

    df = ones((2, 3), dtype='complex128', units='km')
    assert df.shape == (2, 3)
    assert df.dtype == 'complex128'
    assert df.units == ur.km

    df = full((2, 3), 100, dtype='float32', units='km')
    assert df.shape == (2, 3)
    assert df.dtype == 'float32'
    assert df.units == ur.km

    df = eye(3, k=0, dtype='float64', units='eV')
    assert df.shape == (3, 3)
    assert df.dtype == 'float64'
    assert df.units == ur.eV

    assert df[0, 0].data.squeeze() == 1
    assert df[0, 1].data.squeeze() == 0

    df = eye(3, k=1, dtype='float64', units='m')
    assert df.shape == (3, 3)
    assert df.dtype == 'float64'
    assert df.units == ur.m

    assert df[0, 0].data.squeeze() == 0
    assert df[0, 1].data.squeeze() == 1

    df = identity(2, units='m')
    df.units = ur.m
    assert np.all(df.data == np.array([[1., 0.], [0., 1.]]))


# ============================================================================
if __name__ == '__main__':
    pass

# end of module
