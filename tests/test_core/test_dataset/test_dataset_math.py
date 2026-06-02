# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.utils.testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)


# ======================================================================================
# Utility and other tests
# ======================================================================================


def test_nddataset_basic_math():
    nd1 = scp.NDDataset([1.0, 2.0, 3.0])
    nd2 = scp.NDDataset([0.1, 0.2, 0.3])

    # Test basic math operations
    assert_array_almost_equal((nd1 + nd2).data, [1.1, 2.2, 3.3])
    assert_array_almost_equal((nd1 * nd2).data, [0.1, 0.4, 0.9])
    assert_array_almost_equal((nd1 / nd2).data, [10.0, 10.0, 10.0])


def test_nddataset_reduction():
    nd = scp.NDDataset([[1.0, 2.0], [3.0, 4.0]])

    # Test reduction operations
    assert float(nd.sum()) == 10.0  # Convert to float for comparison
    assert float(nd.mean()) == 2.5
    assert float(nd.std()) == pytest.approx(1.118033988749895)


def test_nddataset_broadcasting():
    nd = scp.NDDataset([[1.0, 2.0], [3.0, 4.0]])
    scalar = scp.NDDataset(2.0)

    # Test broadcasting
    result = nd * scalar
    assert_array_equal(result.data, [[2.0, 4.0], [6.0, 8.0]])


def test_nddataset_comparison():
    ndd = scp.NDDataset([1.0, 2.0 + 1j, 3.0])
    val = ndd * 1.2 - 10.0
    val = np.abs(val)
    # Numerical validation
    expected = np.abs(np.array([1.0, 2.0 + 1j, 3.0]) * 1.2 - 10.0)
    assert_array_equal(val.data, expected)
    assert np.all(val >= 6.0)


def test_nddataset_swapdims(nd1d, nd2d, ref_ds, ds1):
    nd1 = nd1d.copy()
    nd2 = nd2d.copy()
    nd3 = ds1.copy()
    # swapdims needs 2D at least
    assert nd1.shape == (10,)
    nd1s = nd1.swapdims(1, 0)
    assert_equal(nd1s.data, nd1.data)
    nd2s = nd2.swapdims(1, 0)
    assert nd2s.dims == nd2.dims[::-1]
    assert nd3.shape == ref_ds.shape
    nd3s = nd3.swapdims(1, 0)
    ref = ref_ds
    refs = np.swapaxes(ref, 1, 0)
    assert nd3.shape == ref.shape  # original unchanged
    assert nd3s.shape == refs.shape
    assert nd3s is not nd3
    assert nd3s.dims[:2] == nd3.dims[:2][::-1]
    nd3s = nd3.swapdims(1, 0, inplace=True)
    assert nd3.shape == refs.shape  # original changed
    assert nd3s is nd3  # objects should be the same
    # use of the numpy method
    nd3s = np.swapaxes(nd3, 1, 0)
    assert nd3.shape == refs.shape  # original unchanged (but was already
    # swapped)
    assert nd3s.shape == ref.shape
    assert (
        nd3s is not nd3
    )  # TODO: add check for swapdims of all elements  # of a dataset such as meta


def test_nddataset_transpose_swapdims(ds1):
    nd = ds1.copy()
    ndt = nd.T
    assert nd[1] == ndt[..., 1].T
    # fix a bug with loc indexation
    nd1 = nd[4000.0:3000.0]
    assert str(nd1) == "NDDataset: [float64] a.u. (shape: (z:4, y:100, x:3))"
    nd2 = ndt[..., 4000.0:3000.0]
    assert str(nd2) == "NDDataset: [float64] a.u. (shape: (x:3, y:100, z:4))"
    assert nd1 == nd2.T


def test_nddataset_apply_funcs(dsm):
    nd = dsm.copy()
    # convert to masked array preserves shape
    ma = np.ma.array(nd)
    assert isinstance(ma, np.ma.MaskedArray)
    assert ma.shape == nd.shape

    nd[1] = scp.MASKED
    ma_masked = np.ma.array(nd)
    assert isinstance(ma_masked, np.ma.MaskedArray)
    assert ma_masked.mask.any()

    ar = np.array(nd)
    assert isinstance(ar, np.ndarray)
    assert ar.shape == nd.shape


def test_take(dsm):
    nd = dsm.copy()
    # take by index
    taken = nd.take([0, 2])
    assert taken.shape[0] == 2
    assert_array_equal(taken.data[0], nd.data[0])
    assert_array_equal(taken.data[1], nd.data[2])


if __name__ == "__main__":
    pytest.main([__file__])
