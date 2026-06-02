# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from zoneinfo import ZoneInfoNotFoundError

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.utils.system import get_user_and_node
from spectrochempy.utils.testing import (
    assert_array_equal,
    assert_equal,
)


# test minimal constructeur and dtypes
adata = (
    [],
    [None, 1.0],
    [np.nan, np.inf],
    [0, 1, 2],
    [0.0, 1.0, 3.0],
    [0.0 + 1j, 10.0 + 3.0j],
    [0.0 + 1j, np.nan + 3.0j],
)


def test_nddataset_docstring():
    from spectrochempy.utils import docutils as chd

    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy.core.dataset.nddataset"
    chd.check_docstrings(
        module,
        obj=scp.NDDataset,
        # exclude some errors - remove whatever you want to check
        exclude=["SA01", "EX01", "ES01", "GL11", "GL08", "PR01"],
    )


@pytest.mark.parametrize("a", adata)
def test_nddataset_1D_NDDataset(a):
    # 1D
    for arr in [a, np.array(a)]:
        ds = scp.NDDataset(arr)
        assert ds.size == len(arr)
        assert ds.shape == (ds.size,)
        if ds.size == 0:
            assert ds.dtype is None
            assert ds.dims == []
        else:
            assert ds.dtype in [np.float64, np.complex128]
            assert ds.dims == ["x"]
        # force dtype
        ds = scp.NDDataset(arr, dtype=np.float32)
        if ds.size == 0:
            assert ds.dtype is None
        else:
            assert ds.dtype == np.float32
        assert ds.title == "<untitled>"
        assert ds.mask == scp.NOMASK
        assert ds.meta == {}
        assert ds.name.startswith("NDDataset")
        assert ds.author == get_user_and_node()
        assert ds.description == ""
        assert ds.history == []


arrdata = (
    np.array([[1, 1.0], [0, np.nan]]),
    np.random.rand(2, 3).astype("int64"),
    np.random.rand(2, 4),
)


@pytest.mark.parametrize("arr", arrdata)
def test_nddataset_2D_NDDataset(arr):
    # 2D
    ds = scp.NDDataset(arr)
    assert ds.size == arr.size
    assert ds.shape == arr.shape
    if ds.size == 0:
        assert ds.dtype is None
        assert ds.dims == []
    else:
        assert ds.dtype == np.float64
        assert ds.dims == ["y", "x"][-ds.ndim :]
    assert ds.title == "<untitled>"
    assert ds.mask == scp.NOMASK
    assert ds.meta == {}
    assert ds.name.startswith("NDDataset")
    assert ds.author == get_user_and_node()
    assert not ds.history
    assert ds.description == ""
    # force dtype
    ds = scp.NDDataset(arr, dtype=np.float32)
    if ds.size == 0:
        assert ds.dtype is None
    else:
        assert ds.dtype == np.float32
    if ds.shape[-1] % 2 == 0:  # must be even
        ds = scp.NDDataset(arr, dtype=np.complex128)
        if ds.size == 0:
            assert ds.dtype is None
        else:
            assert ds.dtype == np.complex128
    else:
        with pytest.raises(ValueError):
            ds = scp.NDDataset(arr, dtype=np.complex128)


def test_nddataset_conversion():
    nd = scp.NDDataset(np.array([[1, 2, 3], [4, 5, 6]]))
    assert nd.data.size == 6
    assert nd.data.dtype == np.dtype("float64")


def test_nddataset_copy_ref():
    """
    Tests to ensure that creating a new NDDataset object copies by *reference*.
    """
    a = np.ones((10, 10))
    nd_ref = scp.NDDataset(a)
    a[0, 0] = 0
    assert nd_ref.data[0, 0] == 0


def test_nddataset_creationdate():
    ndd = scp.NDDataset([1.0, 2.0, 3.0])
    ndd2 = np.sqrt(ndd)
    assert ndd2._date is not None


def test_nddataset_title():
    ndd = scp.NDDataset([1.0, 2.0, 3.0], title="xxxx")
    assert ndd.title == "xxxx"
    ndd2 = scp.NDDataset(ndd, title="yyyy")
    assert ndd2.title == "yyyy"
    ndd2.title = "zzzz"
    assert ndd2.title == "zzzz"


def test_dataset_filename():
    from pathlib import Path

    ndd = scp.NDDataset([1.0, 2.0, 3.0])
    assert ndd.filename == Path(ndd.name).with_suffix(".scp")

    ndd.filename = "xxxx"
    assert ndd.filename == Path("xxxx")

    ndd1 = scp.NDDataset([6.0, 2.0], filename="yyy")
    assert ndd1.filename == Path("yyy")

    ndd2 = scp.NDDataset(ndd, filename="xyxy")
    assert ndd2.filename == Path("xyxy")

    ndd2.filename = "zzzz"
    assert ndd2.filename == Path("zzzz")


def test_nddataset_deepcopy():
    """Deep copy via copy.deepcopy or copy() isolates all mutable state."""
    import copy

    ds = scp.NDDataset(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        units="m",
        title="original",
    )
    # Use copy.deepcopy (function, not method)
    ds_copy = copy.deepcopy(ds)
    # Modify original
    ds.data[0, 0] = 99.0
    ds.title = "modified"
    # Copy should be isolated
    assert_array_equal(ds_copy.data, np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert ds_copy.title == "original"

    # copy() method should also be isolated
    ds2 = scp.NDDataset(np.array([[5.0, 6.0], [7.0, 8.0]]), units="s")
    ds2_copy = ds2.copy()
    ds2.data[0, 0] = 99.0
    assert_array_equal(ds2_copy.data, np.array([[5.0, 6.0], [7.0, 8.0]]))


if __name__ == "__main__":
    pytest.main([__file__])
