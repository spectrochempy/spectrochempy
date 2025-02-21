# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from copy import copy, deepcopy

import numpy as np
import pytest
from quaternion import as_float_array, as_quat_array, quaternion

from spectrochempy.core.dataset.baseobjects.ndcomplex import NDComplexArray
from spectrochempy.core.units import Quantity, ur
from spectrochempy.utils.testing import (
    assert_approx_equal,
    assert_array_equal,
    assert_equal,
)

# TODO: a lot of repetition - must be simplified with some logics

typequaternion = np.dtype(np.quaternion)


def test_ndarray_comparison(ndarray, ndarrayunit, ndarraycplx, ndarrayquaternion):
    # test comparison

    nd1 = ndarray.copy()

    assert nd1 == ndarray
    assert nd1 is not ndarray

    nd2 = ndarrayunit.copy()
    assert nd2 == ndarrayunit

    assert nd1 != nd2
    assert not nd1 == nd2

    nd3 = ndarraycplx.copy()
    assert nd3 == ndarraycplx

    nd4 = ndarrayquaternion.copy()
    assert nd4 == ndarrayquaternion

    assert nd1 != "xxxx"

    nd2n = nd2.to(None, force=True)
    assert nd2n != nd2


def test_ndcomplex_init_quaternion_witharray():
    d = np.arange(24).reshape(3, 2, 4)

    d = as_quat_array(d)
    d0 = NDComplexArray(d)

    assert d0.shape == (3, 2)
    assert_array_equal(d0.real.data, [[0, 4], [8, 12], [16, 20]])

    d1 = NDComplexArray(d)
    d1 = d1.set_quaternion()
    assert_array_equal(d1.real.data, [[0, 4], [8, 12], [16, 20]])

    d1 = d0.swapdims(1, 0)
    assert d1.shape == (2, 3)
    assert_array_equal(d1.real.data, [[0, 8, 16], [4, 12, 20]])
    assert d1[0, 0].values == quaternion(0, 2, 1, 3)


def test_ndcomplex_init_quaternion():
    # test with complex data in all dimension

    np.random.seed(12345)
    d = np.random.random((4, 3)) * np.exp(0.1j)
    d0 = NDComplexArray(
        d,
        units=ur.Hz,
        mask=[[False, True, False], [True, False, False]],
        dtype=typequaternion,
    )  # with units & mask
    assert d0.shape == (2, 3)
    assert "NDComplexArray: [quaternion] Hz" in repr(d0)


def test_ndcomplex_init_quaternion_error1():
    # test with complex data in all dimension but odd number of comlumn (should raise an error
    d = np.random.random((3, 3)) * np.exp(0.1j)
    with pytest.raises(ValueError):
        NDComplexArray(d, dtype=typequaternion)  # with units & mask


def test_ndcomplex_init_quaternion_error2():
    # test with complex data in all dimension but odd number of rows(should raise an error
    d = np.random.random((3, 4))
    with pytest.raises(ValueError):
        NDComplexArray(d, dtype=typequaternion)  # with units & mask


def test_ndcomplex_init_complex_with_copy_of_ndarray():
    # test with complex from copy of another ndArray

    d = np.ones((2, 2)) * np.exp(0.1j)
    d1 = NDComplexArray(d)
    d2 = NDComplexArray(d1)
    assert d1._data is d2._data
    assert np.all(d1.data == d2.data)
    assert d2.has_complex_dims
    assert not d2.is_quaternion
    assert d2.shape == (2, 2)


def test_ndcomplex_init_complex_with_mask():
    # test with complex with mask and units

    np.random.seed(12345)
    d = np.random.random((2, 2)) * np.exp(0.1j)

    d3 = NDComplexArray(
        d, units=ur.Hz, mask=[[False, True], [False, False]]
    )  # with units & mask

    # internal representation (interleaved)
    assert d3.shape == (2, 2)
    assert d3._data.shape == (2, 2)
    assert d3.data.shape == (2, 2)
    assert d3.size == 4

    assert (d3.real.data == d.real).all()
    assert np.all(d3.data.real == d.real)

    assert d3.dtype == np.complex128
    assert d3.has_complex_dims
    assert d3.mask.shape[-1] == d3.shape[-1]
    d3RR = d3.component("RR")
    assert not d3RR.has_complex_dims
    assert d3RR._data.shape == (2, 2)
    assert d3RR._mask.shape == (2, 2)

    assert isinstance(d3[1, 1].values, Quantity)
    assert d3[1, 1].values.magnitude == d[1, 1]


def test_ndcomplex_swapdims():
    np.random.seed(12345)
    d = np.random.random((4, 3)) * np.exp(0.1j)
    d3 = NDComplexArray(
        d,
        units=ur.Hz,
        mask=[
            [False, True, False],
            [False, True, False],
            [False, True, False],
            [True, False, False],
        ],
    )  # with units & mask
    assert d3.shape == (4, 3)
    assert d3._data.shape == (4, 3)
    assert d3.has_complex_dims
    assert not d3.is_quaternion
    assert d3.dims == ["y", "x"]
    d4 = d3.swapdims(0, 1)
    assert d4.dims == ["x", "y"]
    assert d4.shape == (3, 4)
    assert d4._data.shape == (3, 4)
    assert d4.has_complex_dims
    assert not d4.is_quaternion


def test_ndcomplex_ndarraycplx_fixture2(ndarraycplx):
    nd = ndarraycplx.copy()
    # some checking
    assert nd.size == 40
    assert nd.data.size == 40
    assert nd.shape == (10, 4)
    assert nd.has_complex_dims
    assert nd.data.dtype == np.complex128
    assert nd.ndim == 2


def test_ndcomplex_init_complex_with_a_ndarray():
    # test with complex data in the last dimension

    d = np.array([[1, 2], [3, 4]]) * np.exp(0.1j)
    d0 = NDComplexArray(d)
    assert d0.dtype == np.complex128
    assert d0.has_complex_dims
    assert d0.shape == (2, 2)
    assert d0.size == 4

    assert "NDComplexArray: [complex128]" in repr(d0)


def test_ndcomplex_quaternion_fixture(ndarrayquaternion):
    nd = ndarrayquaternion.copy()

    # some checking
    assert nd.size == 20
    assert nd.data.size == 20
    assert nd.shape == (5, 4)
    assert nd.has_complex_dims
    assert nd.is_quaternion
    assert nd.data.dtype == typequaternion
    assert nd.dtype == typequaternion
    assert nd.ndim == 2


def test_ndcomplex_real_imag():
    np.random.seed(12345)
    d = np.random.random((2, 2)) * np.exp(0.1j)
    d3 = NDComplexArray(d)
    new = d3.copy()
    new.data = d3.real.data + 1j * d3.imag.data
    assert_equal(d3.data, new.data)


def test_ndcomplex_set_with_complex(ndarraycplx):
    nd = ndarraycplx.copy()
    nd.units = "meter/hour"
    assert nd.units == ur.meter / ur.hour


def test_ndcomplex_copy_of_ndarray(ndarraycplx):
    nd1 = ndarraycplx
    nd2 = copy(ndarraycplx)
    assert nd2 is not nd1
    assert nd2.shape == nd1.shape
    assert nd2.is_complex == nd1.is_complex
    assert nd2.ndim == nd1.ndim


def test_ndcomplex_deepcopy_of_ndarray(ndarraycplx):
    # for this example there is no diif with copy (write another test for this)
    nd1 = ndarraycplx.copy()
    nd2 = deepcopy(nd1)
    assert nd2 is not nd1
    assert nd2.data.size == 40


def test_ndcomplex_len_and_sizes_cplx(ndarraycplx):
    ndc = ndarraycplx.copy()
    assert ndc.has_complex_dims
    assert ndc.is_complex
    assert len(ndc) == 10  # len is the number of rows
    assert ndc.shape == (10, 4)
    assert ndc.size == 40
    assert ndc.ndim == 2


# def test_ndcomplex_slicing_byindex_cplx(ndarraycplx):
#     ndc = ndarraycplx.copy()
#     ndc1 = ndc[1, 1]
#     assert_equal(ndc1.values, ndc.RR[1, 1].values + ndc.RI[1, 1].values * 1.j)


def test_ndcomplex_slicing_byindex_quaternion(ndarrayquaternion):
    ndc = ndarrayquaternion.copy()
    ndc1 = ndc[1, 1].real
    assert_approx_equal(ndc1.values.magnitude, 4.646475973719301, 3)


def test_ndcomplex_complex(ndarraycplx):
    nd = ndarraycplx.copy()

    ndr = nd.real
    assert_array_equal(ndr.data, nd.data.real)
    assert ndr.size == nd.size
    assert not ndr.is_complex


def test_ndcomplex_str_representation_for_complex():
    nd1 = NDComplexArray([1.0 + 2.0j, 2.0 + 3.0j])
    assert "NDComplexArray: [complex128] unitless" in repr(nd1)


def test_ndcomplex_quaternion_str_representation():
    np.random.seed(12345)
    d = np.random.random((4, 2)) * np.exp(0.1j)
    NDComplexArray(d, dtype=typequaternion)


def test_ndcomplex_real_imag_quaternion():
    np.random.seed(12345)
    d = np.random.random((2, 2)) * np.exp(0.1j)
    d3 = NDComplexArray(d, dtype=typequaternion)
    d3r = d3.real
    assert d3r.dtype == np.float64
    assert d3r.shape == (1, 2)
    d3i = d3.imag
    assert d3i.dtype == typequaternion


def test_ndcomplex_swapdims_quaternion():
    np.random.seed(12345)
    d = np.random.random((4, 3)) * np.exp(0.1j)

    d3 = NDComplexArray(
        d,
        units=ur.Hz,
        mask=[[False, True, False], [True, False, False]],
        dtype=typequaternion,
    )  # quaternion with units & mask

    assert d3.shape == (2, 3)
    assert d3._data.shape == (2, 3)
    assert d3.has_complex_dims
    assert d3.is_quaternion

    w, x, y, z = as_float_array(d3.data).T

    d4 = d3.swapdims(0, 1)

    assert d4.shape == (3, 2)
    assert d4._data.shape == (3, 2)
    assert d4.has_complex_dims
    assert d4.is_quaternion

    wt, yt, xt, zt = as_float_array(d4.data).T
    assert_array_equal(xt, x.T)
    assert_array_equal(yt, y.T)
    assert_array_equal(zt, z.T)
    assert_array_equal(wt, w.T)


def test_ndcomplex_squeeze(ndarrayunit):
    nd = NDComplexArray(ndarrayunit)
    assert nd.shape == (10, 8)

    d = nd[..., 0]
    d = d.squeeze()
    assert d.shape == (10,)

    d = nd[0]
    d = d.squeeze()
    assert d.shape == (8,)

    nd1 = nd.set_complex()
    assert nd1.shape == (10, 4)
    nd1._repr_html_()

    d = nd[..., 0]
    d = d.squeeze()
    assert d.shape == (10,)

    d = nd[0]
    assert d.shape == (1, 8)
    d1 = d.squeeze()
    assert d1.shape == (8,)
    assert d1 is not d

    # TODO: test a revoir  # d = nd[..., 0].real  # assert np.all(d == nd[..., 0].RR)
    # assert d.shape == (10, 1)  # d1 = d.squeeze("x")  # assert d1.shape == (10,)
    # assert d1 is not d  #  # # inplace  # d = nd[..., 0:1]  # assert d.shape == (10, 1)
    # d1 = d.squeeze(dims=1, inplace=True)  # assert d1.shape == (10,)  # assert d1 is d  #
    # d = nd[0:1]  # assert d.shape == (1, 8)  # d1 = d.squeeze(dims=0, inplace=True)
    # assert d1.shape == (8,)  # assert d1 is d
