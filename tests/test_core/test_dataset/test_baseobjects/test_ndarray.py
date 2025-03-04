# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from copy import copy, deepcopy

import numpy as np
import pytest
from pint.errors import DimensionalityError

from spectrochempy.core.dataset.baseobjects.ndarray import NDArray
from spectrochempy.core.units import Quantity, ur
from spectrochempy.utils.constants import INPLACE, MASKED
from spectrochempy.utils.misc import TYPE_FLOAT, TYPE_INTEGER
from spectrochempy.utils.testing import (
    assert_array_equal,
    assert_equal,
    catch_warnings,
    raises,
)


# --------------------------------------------------------------------------------------
#  NDARRAY INITIALIZATION
# --------------------------------------------------------------------------------------
def test_ndarray_init(refarray, refmask, ndarray, ndarraymask):
    # initialisation with null array

    d0 = NDArray(description="testing ndarray")

    assert d0._implements("NDArray")
    assert d0._implements() == "NDArray"

    assert isinstance(d0, NDArray)

    assert d0.is_empty
    assert len(d0) == 0
    assert d0.shape == ()
    assert d0.id.startswith("NDArray")
    assert d0.name == d0.id
    assert d0.title == "<untitled>"
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
    d0.name = "xxxx"
    assert d0.name == "xxxx"
    d0.title = "yyyy"
    assert d0.title == "yyyy"
    d0.meta = {}
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
    assert d6.title == "<untitled>"
    assert d6.shape == refarray.shape
    assert d6.dims == ["y", "x"]
    assert d6.size == refarray.size
    assert_array_equal(d6.data, refarray)
    assert d6._data is ndarraymask._data  # by default we do not copy
    # d6.data and ndarraym ask.data are however different due to the addition of un offset
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
    assert d8.title == "<untitled>"

    # initialisation with only labels

    d9 = NDArray(labels="a b c d e f g h i j".split(), title="labeled")
    assert d9.is_labeled

    # changing dims name
    d11 = NDArray(
        labels="a b c d e f g h i j".split(),
        title="labeled",
        dims=["q"],
        author="Blake",
    )
    assert d11.dims == ["q"]

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

    # check inplace
    d2 = d0._sort(inplace=True)
    assert d0.data[0] == 1000
    assert d2 is d0

    # check descend
    d0._sort(descend=True, inplace=True)
    assert d0.data[0] == 4000

    # check sort using label
    d3 = d0._sort(by="label", descend=True)
    assert d3.labels[0] == "j"

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

    with catch_warnings() as w:
        # try to change to an array with units
        nd.to("m")  # should not change anything (but raise a warning)
        assert w[-1].category == UserWarning

    assert nd.unitless

    nd.units = "m"
    assert nd.units == ur.meter

    nd1 = nd.to("km")
    assert nd.units != ur.kilometer  # not inplace
    assert nd1.units == ur.kilometer
    nd.ito("m")
    assert nd.units == ur.meter

    # change of units - ok if it can be cast to the current one

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
    with raises(DimensionalityError):
        nd1 = nd1.ito("km/s")  # should raise an error
    nd.units = "m/km"
    assert nd.units.dimensionless
    # assert nd.units.scaling == 0.001
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
    assert nd.__repr__() == f"NDArray: [float64] m⋅s⁻¹ (shape: (y:{h}, x:{w}))"
    nd[1] = MASKED
    assert nd.is_masked

    # test repr_html
    assert (
        "<div class='scp-output'><details><summary>NDArray: [float64]"
        in nd._repr_html_()
    )

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

    with raises(Exception):
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
