# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

# flake8: noqa


from copy import copy, deepcopy
from quaternion import as_quat_array, as_float_array, quaternion
import numpy as np
import pytest
from pint.errors import DimensionalityError

from spectrochempy.core.dataset.ndarray import NDArray, NDComplexArray
from spectrochempy.core.units import ur, Quantity
from spectrochempy.utils import (
    SpectroChemPyWarning,
    INPLACE,
    MASKED,
    TYPE_INTEGER,
    TYPE_FLOAT,
)
from spectrochempy.utils.testing import (
    assert_equal,
    assert_array_equal,
    assert_approx_equal,
    assert_produces_warning,
    RandomSeedContext,
)


# TODO: a lot of repetition - must be simplified with some logics

typequaternion = np.dtype(np.quaternion)

# ------------------------------------------------------------------
# create reference arrays
# ------------------------------------------------------------------

with RandomSeedContext(12345):
    ref_data = 10.0 * np.random.random((10, 8)) - 5.0
    ref3d_data = 10.0 * np.random.random((10, 100, 3)) - 5.0
    ref3d_2_data = np.random.random((9, 50, 4))
    ref_mask = ref_data < -4
    ref3d_mask = ref3d_data < -3
    ref3d_2_mask = ref3d_2_data < -2


# ------------------------------------------------------------------
# Fixtures: some NDArray's
# ------------------------------------------------------------------


@pytest.fixture(scope="function")
def refarray():
    return ref_data.copy()


@pytest.fixture(scope="function")
def refmask():
    return ref_mask.copy()


@pytest.fixture(scope="function")
def ndarray():
    # return a simple ndarray with some data
    return NDArray(ref_data, desc="An array", copy=True).copy()


@pytest.fixture(scope="function")
def ndarrayunit():
    # return a simple ndarray with some data and units
    return NDArray(ref_data, units="m/s", copy=True).copy()


@pytest.fixture(scope="function")
def ndarraymask():
    # return a simple ndarray with some data and units
    return NDArray(
        ref_data, mask=ref_mask, units="m/s", history="Creation with mask", copy=True
    ).copy()


# ------------------------------------------------------------------
# Fixtures: Some NDComplex's array
# ------------------------------------------------------------------


@pytest.fixture(scope="function")
def ndarraycplx():
    # return a complex ndarray
    return NDComplexArray(ref_data, units="m/s", dtype=np.complex128, copy=True).copy()


@pytest.fixture(scope="function")
def ndarrayquaternion():
    # return a quaternion ndarray
    return NDComplexArray(ref_data, units="m/s", dtype=np.quaternion, copy=True).copy()


# ------------------------------------------------------------------
#  NDARRAY INITIALIZATION
# ------------------------------------------------------------------


def test_ndarray_init(refarray, refmask, ndarray, ndarraymask):
    # initialisation with null array

    d0 = NDArray(description="testing ndarray")

    assert d0.implements("NDArray")
    assert d0.implements() == "NDArray"

    assert isinstance(d0, NDArray)

    assert d0.is_empty
    assert len(d0) == 0
    assert d0.shape == ()
    assert d0.id.startswith("NDArray")
    assert d0.name == d0.id
    assert d0.long_name == "<untitled>"
    assert d0.ndim == 0
    assert d0.size is None
    assert not d0.is_masked
    assert d0.dtype is None
    assert d0.unitless
    assert not d0.dims
    assert not d0.meta
    assert hash(d0) is not None
    assert repr(d0) == "NDArray: empty (size: 0)"

    # assignment to basic write allowed properties

    d0.data = [1, 2, 3]  # put some data
    assert_array_equal(d0.data, np.array([1, 2, 3]))
    assert d0.dtype in TYPE_INTEGER
    assert d0._created.astype("datetime64[D]") == np.datetime64("now", "D")
    d0.name = "xxxx"
    assert d0.name == "xxxx"
    d0.long_name = "yyyy"
    assert d0.long_name == "yyyy"
    d0.meta = []
    d0.meta.something = "a_value"
    assert d0.meta.something == "a_value"
    assert d0[1].value == 2  # only a single element so we get a squeezed array
    d0.units = "absorbance"
    assert d0.units == ur.absorbance
    assert d0[2] == 3 * ur.absorbance
    assert d0.dims == ["x"]

    # initialisation with a scalar quantity

    d1 = NDArray(25)
    assert d1.data == np.array(25)
    assert d1.data.dtype in TYPE_INTEGER

    d1 = NDArray(13.0 * ur.tesla)
    assert d1.data == np.array(13.0)
    assert d1.data.dtype in TYPE_FLOAT
    assert d1.shape == ()
    assert d1.ndim == 0
    assert not d1.dims
    assert d1.units == "tesla"
    assert d1.values == 13.0 * ur.tesla

    # initialisation with a 1D array  quantity

    d2 = NDArray([13.0] * ur.tesla)
    assert d2.data == np.array([13.0])
    assert d2.shape == (1,)
    assert d2.ndim == 1
    assert d2.dims == ["x"]
    assert d2.units == "tesla"
    assert d2.values == 13.0 * ur.tesla

    # initialisation with a 1D vector quantity

    d3 = NDArray([[13.0, 20.0]] * ur.tesla)
    assert_array_equal(d3.data, np.array([[13.0, 20.0]]))
    assert d3.shape == (1, 2)
    assert d3.ndim == 2
    assert d3.dims == ["y", "x"]
    assert d3.units == "tesla"

    # initialisation with a sequence

    d4 = NDArray((2, 3, 4))
    assert d4.shape == (3,)
    assert d4.size == 3
    assert d4.dims == ["x"]
    assert not d4.is_masked

    # initialization with an array

    d5 = NDArray(refarray)
    assert d5.shape == refarray.shape
    assert d5.size == refarray.size
    assert not d5.is_masked

    # initialization with an NDArray object

    d6 = NDArray(ndarraymask)
    assert d6.long_name == "<untitled>"
    assert d6.shape == refarray.shape
    assert d6.dims == ["y", "x"]
    assert d6.size == refarray.size
    assert_array_equal(d6.data, refarray)
    assert d6._data is ndarraymask._data  # by default we do not copy
    # d6.data and ndarraym ask.data are however different due
    # to the addition of un offset
    assert d6.is_masked
    assert_array_equal(d6.mask, refmask)
    assert d6.mask is ndarraymask.mask  # no copy by default

    # initialization with an NDArray object with copy
    d7 = NDArray(ndarraymask, copy=True)
    assert_array_equal(d7.data, refarray)
    assert d7.data is not ndarraymask.data  # by default we do not copy
    assert_array_equal(d7.mask, refmask)
    assert d7.mask is not ndarraymask.mask  # no copy by default

    # initialisation with a sequence and a mask

    d0mask = NDArray([2, 3, 4, 5], mask=[1, 0, 0, 0], dtype="int64")
    assert d0mask.shape == (4,)
    assert d0mask.is_masked
    assert d0mask.mask.shape == d0mask.shape

    # initialisation with a sequence and a mask

    d1mask = NDArray([2.0, 3.0, 4.0, 5.1], mask=[1, 0, 0, 0])
    assert d1mask.shape == (4,)
    assert d1mask.is_masked
    assert d1mask.mask.shape == d1mask.shape

    # dtype specified

    d8 = NDArray(ndarraymask, desc="with mask", dtype=np.int64)
    assert d8.shape == refarray.shape
    assert d8.data.dtype == np.int64
    assert d8.dims == ["y", "x"]
    assert d8.long_name == "<untitled>"
    assert d8.source == "with mask"
    assert d8.desc == d8.source
    assert len(ndarraymask.history) == 1  # one line already in
    assert len(d8.history) == 2  # copy added

    # initialisation with only labels

    d9 = NDArray(labels="a b c d e f g h i j".split(), title="labeled")
    assert d9.is_labeled

    # changing dims name
    d11 = NDArray(
        labels="a b c d e f g h i j".split(),
        title="labeled",
        dims=["q"],
        author="Blake",
        history="Created from scratch",
    )
    assert d11.dims == ["q"]
    assert d11.author == "Blake"

    assert "[  a   b ...   i   j]" in d11._repr_html_()  # comparison


def test_ndarray_copy():
    d0 = NDArray(
        np.linspace(4000, 1000, 10),
        labels="a  b  c  d  e  f  g  h  i  j".split(),
        units="s",
        mask=False,
        title="wavelength",
    )
    d0[5] = MASKED

    d1 = d0.copy()
    assert d1 is not d0
    assert d1 == d0
    assert not (d1 != d0)
    assert d1.units == d0.units
    assert_array_equal(d1.labels, d0.labels)
    assert_array_equal(d1.mask, d0.mask)

    d0 = NDArray(
        np.linspace(4000, 1000, 10),
        labels=[
            "a  b  c  d  e  f  g  h  i  j".split(),
            "bc cd de ef ab fg gh hi ja ij".split(),
        ],
        units="s",
        mask=False,
        title="wavelength",
    )
    d0[5] = MASKED

    d1 = d0.copy()
    assert d1 is not d0
    assert d1 == d0
    assert d1.units == d0.units
    assert_array_equal(d1.labels, d0.labels)
    assert_array_equal(d1.mask, d0.mask)

    d2 = copy(d0)
    assert d2 == d0

    d3 = deepcopy(d0)
    assert d3 == d0


def test_ndarray_sort():
    # labels and sort

    d0 = NDArray(
        np.linspace(4000, 1000, 10),
        labels="a b c d e f g h i j".split(),
        units="s",
        mask=False,
        title="wavelength",
    )

    assert d0.is_labeled

    d1 = d0._sort()
    assert d1.data[0] == 1000
    assert d1.data[-1] == d0.data[0]

    # check inplace
    d2 = d0._sort(inplace=True)
    assert d0.data[0] == d2.data[0] == 1000
    assert d2 is d0

    # check descend
    d0._sort(descend=True, inplace=True)
    assert d0.data[0] == 4000

    # check sort using label
    d3 = d0._sort(by="label", descend=True)
    assert d3.labels[0] == "j"
    assert d3 is not d0

    # multilabels
    # add a row of labels to d0
    d0.labels = "bc cd de ef ab fg gh hi ja ij ".split()

    d1 = d0._sort()
    assert d1.data[0] == 1000
    assert_array_equal(d1.labels[0], ["j", "ij"])

    d1._sort(descend=True, inplace=True)
    assert d1.data[0] == 4000
    assert_array_equal(d1.labels[0], ["a", "bc"])

    d1 = d1._sort(by="label[1]", descend=True)
    assert np.all(d1.labels[0] == ["i", "ja"])

    # other way
    d2 = d1._sort(by="label", pos=1, descend=True)
    assert np.all(d2.labels[0] == d1.labels[0])

    d3 = d1.copy()
    d3._labels = None
    d3._sort(
        by="label", pos=1, descend=True
    )  # no label! generate a warning but no error


def test_ndarray_methods(refarray, ndarray, ndarrayunit):
    ref = refarray
    nd = ndarray.copy()
    assert nd.data.size == ref.size
    assert nd.shape == ref.shape
    assert nd.size == ref.size
    assert nd.ndim == 2
    assert nd.data[1, 1] == ref[1, 1]
    assert nd.dims == ["y", "x"]
    assert nd.unitless  # no units
    assert not nd.dimensionless  # no unit so dimensionless has no sense

    with assert_produces_warning(
        SpectroChemPyWarning,
        raise_on_extra_warnings=False,
        match="There is no units for this NDArray!",
    ):
        # try to change to an array with units
        nd.to("m")  # should not change anything (but raise a warning)

    assert nd.unitless

    nd.units = "m"
    assert nd.units == ur.meter

    nd1 = nd.to("km")
    assert nd.units != ur.kilometer  # not inplace
    assert nd1.units == ur.kilometer
    nd.ito("m")
    assert nd.units == ur.meter

    # change of units - ok if it can be casted to the current one

    nd.units = "cm"

    # cannot change to incompatible units

    with pytest.raises(TypeError):
        nd.units = "radian"

    # we can force them

    nd.ito("radian", force=True)

    # check dimensionless and scaling

    assert 1 * nd.units == 1.0 * ur.dimensionless
    assert nd.units.dimensionless
    assert nd.dimensionless
    with pytest.raises(DimensionalityError):
        nd1 = nd1.ito("km/s")  # should raise an error
    nd.units = "m/km"
    assert nd.units.dimensionless
    assert nd.units.scaling == 0.001
    nd.to(1 * ur.m, force=True)
    assert nd.dims == ["y", "x"]

    # check units compatibility

    nd.ito("m", force=True)
    nd2 = ndarray.copy()
    assert nd2.dims == ["y", "x"]
    nd2.units = "km"
    assert nd.is_units_compatible(nd2)
    nd2.ito("radian", force=True)
    assert not nd.is_units_compatible(nd2)

    # check masking

    assert not nd.is_masked
    repr(nd)
    assert repr(nd).startswith("NDArray: ")
    nd[0] = MASKED
    assert nd.is_masked
    assert nd.dims == ["y", "x"]

    # check len and size

    assert len(nd) == ref.shape[0]
    assert nd.shape == ref.shape
    assert nd.size == ref.size
    assert nd.ndim == 2
    assert nd.dims == ["y", "x"]

    # a vector is a 1st rank tensor. Internally (it will always be represented
    # as a 1D matrix.

    v = NDArray([[1.0, 2.0, 3.0]])
    assert v.ndim == 2
    assert v.shape == (1, 3)
    assert v.dims == ["y", "x"]
    assert_array_equal(v.data, np.array([[1.0, 2.0, 3.0]]))

    vt = v.transpose()
    assert vt.shape == (3, 1)
    assert vt.dims == ["x", "y"]
    assert_array_equal(vt.data, np.array([[1.0], [2.0], [3.0]]))

    # test repr

    nd = ndarrayunit.copy()
    h, w = ref.shape
    assert nd.__repr__() == f"NDArray: [float64] m.s^-1 (shape: (y:{h}, x:{w}))"
    nd[1] = MASKED
    assert nd.is_masked

    # test repr_html
    assert "<table style='background:transparent'>" in nd._repr_html_()

    # test iterations

    nd = ndarrayunit.copy()
    nd[1] = MASKED

    # force units to change

    np.random.seed(12345)
    ndd = NDArray(
        data=np.random.random((3, 3)),
        mask=[[True, False, False], [False, True, False], [False, False, True]],
        units="meters",
    )

    with pytest.raises(Exception):
        ndd.to("second")
    ndd.to("second", force=True)

    # swapdims

    np.random.seed(12345)
    d = np.random.random((4, 3))
    d3 = NDArray(
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
    assert d3.dims == ["y", "x"]
    d4 = d3.swapdims(0, 1)
    assert d4.dims == ["x", "y"]
    assert d4.shape == (3, 4)
    assert d4._data.shape == (3, 4)

    # test iter
    for i, item in enumerate(ndd):
        assert item == ndd[i]

    ndz = NDArray()
    assert not list(item for item in ndz)

    assert str(ndz) == repr(ndz) == "NDArray: empty (size: 0)"


################
# TEST SLICING #
################


def test_ndarray_slicing(refarray, ndarray):

    ref = refarray
    nd = ndarray.copy()
    assert not nd.is_masked
    assert nd.dims == ["y", "x"]

    # slicing is different in scpy than with numpy. We always return
    # unsqueezed dimensions, except for array of size 1, which are considered as scalar

    nd1 = nd[0, 0]
    assert_equal(nd1.data, nd.data[0:1, 0:1])
    assert nd1 is not nd[0, 0]
    assert nd1.ndim == 2  # array not reduced
    assert nd1.size == 1
    assert nd1.shape == (1, 1)
    assert isinstance(nd1, NDArray)
    assert isinstance(nd1.data, np.ndarray)
    assert isinstance(nd1.values, TYPE_FLOAT)

    nd1b = nd.__getitem__(
        (0, 0),
    )
    assert nd1b == nd1

    nd1a = nd[0, 0:2]
    assert_equal(nd1a.data, nd.data[0:1, 0:2])
    assert nd1a is not nd[0, 0:2]
    assert nd1a.ndim == 2
    assert nd1a.size == 2
    assert nd1a.shape == (1, 2)
    assert isinstance(nd1a, NDArray)
    assert nd1a.dims == ["y", "x"]

    # returning none if empty when slicing
    nd1b = nd[11:, 11:]
    assert nd1b is None

    # nd has been changed, restore it before continuing
    nd = ndarray.copy()

    nd2 = nd[7:10]
    assert_equal(nd2.data, nd.data[7:10])
    assert not nd.is_masked

    nd3 = nd2[1]
    assert nd3.shape == (1, ref.shape[1])
    assert nd3.dims == ["y", "x"]

    nd4 = nd2[:, 1]
    assert nd4.shape == (3, 1)
    assert nd4.dims == ["y", "x"]

    # squezzing
    nd5 = nd4.squeeze()
    assert nd5.shape == (3,)
    assert nd5.dims == ["y"]

    # set item
    nd[1] = 2.0
    assert nd[1, 0] == 2

    # set item mask
    nd[1] = MASKED
    assert nd.is_masked

    # boolean indexing
    nd = ndarray.copy()
    nd[nd.data > 0]

    # fancy indexing
    df = nd.data[[-1, 1]]

    ndf = nd[[-1, 1]]
    assert_array_equal(ndf.data, df)

    ndf = nd[
        [-1, 1], INPLACE
    ]  # TODO: check utility of this (I remember it should be related to setitem)
    assert_array_equal(ndf.data, df)

    # use with selection from other numpy functions
    am = np.argmax(nd.data, axis=1)
    assert_array_equal(am, np.array([7, 3]))
    amm = nd.data[..., am]
    assert_array_equal(nd[..., am].data, amm)

    # slicing only-label array

    d0 = NDArray(labels="a b c d e f g h i j".split(), title="labelled")
    assert d0.is_labeled

    assert d0.ndim == 1
    assert d0.shape == (10,)
    assert d0[1].labels == ["b"]
    assert d0[1].values == "b"
    assert d0["b"].values == "b"
    assert d0["c":"d"].shape == (2,)
    assert_array_equal(d0["c":"d"].values, np.array(["c", "d"]))


def test_dim_names_specified(ndarray):
    nd = ndarray.copy()
    assert not nd.is_masked
    assert nd.dims == ["y", "x"]

    # set dim names
    nd.dims = ["t", "y"]

    assert nd.dims == ["t", "y"]

    assert nd.dims == ["t", "y"]


def test_ndarray_issue_23():
    nd = NDArray(np.ones((10, 10)))
    assert nd.shape == (10, 10)
    assert nd.dims == ["y", "x"]
    # slicing
    nd1 = nd[1]
    assert nd1.shape == (1, 10)
    assert nd1.dims == ["y", "x"]
    # transposition
    ndt = nd1.T
    assert ndt.shape == (10, 1)
    assert ndt.dims == ["x", "y"]
    # squeezing
    nd2 = nd1.squeeze()
    assert nd2.shape == (10,)
    assert nd2.dims == ["x"]

    nd = NDArray(np.ones((10, 10, 2)))
    assert nd.shape == (10, 10, 2)
    assert nd.dims == ["z", "y", "x"]
    # slicing
    nd1 = nd[:, 1]
    assert nd1.shape == (10, 1, 2)
    assert nd1.dims == ["z", "y", "x"]
    # transposition
    ndt = nd1.T
    assert ndt.shape == (2, 1, 10)
    assert ndt.dims == ["x", "y", "z"]
    # squeezing
    nd2 = nd1.squeeze()
    assert nd2.shape == (10, 2)
    assert nd2.dims == ["z", "x"]


# Bugs Fixes


def test_ndarray_bug_13(ndarrayunit):
    nd = ndarrayunit[0]

    assert isinstance(nd[0], NDArray)

    # reproduce our bug (now solved)
    nd[0] = Quantity("10 cm.s^-1")

    with pytest.raises(DimensionalityError):
        nd[0] = Quantity("10 cm")


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
    # test with complex data in all dimension but odd number of comlumn
    # (should raise an error
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
