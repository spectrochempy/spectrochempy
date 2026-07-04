# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np
import pytest

from spectrochempy import NDDataset
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

    # isinstance typo regression: first arg numpy, second arg NDDataset
    x = dot(a.data.T, b)
    assert x.shape == (a.x.size, b.x.size)


def test_dot_strict_mask_propagation():
    # `strict` must be honoured and forwarded to numpy.ma.dot.
    # With a masked entry in row 0 of `a`, strict propagation masks the whole
    # corresponding row of the result; non-strict treats masked values as 0.
    a = np.ma.array([[1.0, 2.0], [3.0, 4.0]], mask=[[0, 1], [0, 0]])
    b = np.ma.array([[5.0, 6.0], [7.0, 8.0]], mask=[[0, 0], [0, 0]])
    da = NDDataset(a)
    db = NDDataset(b)

    # numpy reference behaviour
    ref_loose = np.ma.dot(a, b, strict=False)
    ref_strict = np.ma.dot(a, b, strict=True)
    assert not np.ma.getmaskarray(ref_loose).any()
    assert np.array_equal(
        np.ma.getmaskarray(ref_strict), [[True, True], [False, False]]
    )

    # strict=True must mask the whole result row impacted by the masked value
    r_strict = dot(da, db, strict=True)
    assert np.array_equal(
        np.ma.getmaskarray(np.ma.array(r_strict.data, mask=r_strict.mask)),
        [[True, True], [False, False]],
    )

    # strict=False keeps every entry unmasked and matches numpy
    r_loose = dot(da, db, strict=False)
    assert not np.ma.getmaskarray(np.ma.array(r_loose.data, mask=r_loose.mask)).any()
    np.testing.assert_allclose(r_loose.data, ref_loose.data)

    # strict actually changes the outcome, so the two must differ
    assert not np.array_equal(
        np.ma.getmaskarray(np.ma.array(r_strict.data, mask=r_strict.mask)),
        np.ma.getmaskarray(np.ma.array(r_loose.data, mask=r_loose.mask)),
    )

    # default must preserve the historical (non-strict) behaviour
    r_default = dot(da, db)
    assert np.array_equal(
        np.ma.getmaskarray(np.ma.array(r_default.data, mask=r_default.mask)),
        np.ma.getmaskarray(np.ma.array(r_loose.data, mask=r_loose.mask)),
    )
