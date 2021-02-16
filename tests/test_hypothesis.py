import numpy as np
import pytest
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hen

import spectrochempy as scp
from spectrochempy.utils import get_user_and_node


# test minimal constructeur and dtypes
@given(st.lists(st.none()), st.lists(st.integers()),
       st.lists(st.floats()), st.lists(st.complex_numbers()))
def test_1D_NDDataset(a, b, c, d):
    # 1D
    for arr in [a, b, c, d,
                np.array(a), np.array(b), np.array(c), np.array(d)]:

        ds = scp.NDDataset(arr)
        assert ds.size == len(arr)
        assert ds.shape == (ds.size,)
        if ds.size == 0:
            assert ds.dtype is None
            assert ds.dims == []
        else:
            assert ds.dtype in [np.float64, np.complex128]
            assert ds.dims == ['x']

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


@given(hen.arrays(float, st.tuples(st.integers(1,3), st.integers(1,3))))
def test_2D_NDDataset(arr):
    # 2D
    ds = scp.NDDataset(arr)
    assert ds.size == arr.size
    assert ds.shape == arr.shape
    if ds.size == 0:
        assert ds.dtype is None
        assert ds.dims == []
    else:
        assert ds.dtype == np.float64
        assert ds.dims == ['y','x'][-ds.ndim:]

    assert ds.title == "<untitled>"
    assert ds.mask == scp.NOMASK
    assert ds.meta == {}
    assert ds.name.startswith("NDDataset")
    assert ds.author == get_user_and_node()
    assert ds.description == ""

    # force dtype
    ds = scp.NDDataset(arr, dtype=np.float32)
    if ds.size == 0:
        assert ds.dtype is None
    else:
        assert ds.dtype == np.float32

    if ds.shape[-1] % 2 == 0: # must be even
        ds = scp.NDDataset(arr, dtype=np.complex)
        if ds.size == 0:
            assert ds.dtype is None
        else:
            assert ds.dtype == np.complex128
    else:
        with pytest.raises(ValueError):
            ds = scp.NDDataset(arr, dtype=np.complex)

    if (arr.shape[-1] % 2) == 0 and (arr.shape[-2] % 2) == 0 and arr.ndim==2:
        ds = scp.NDDataset(arr, dtype=np.quaternion)
        if ds.size == 0:
            assert ds.dtype is None
        else:
            assert ds.dtype == np.quaternion
    else:
        with pytest.raises(ValueError):
            ds = scp.NDDataset(arr, dtype=np.quaternion)
