# -*- coding: utf-8 -*-
# flake8: noqa


from copy import copy

import numpy as np
import pytest
from pint.errors import DimensionalityError

from spectrochempy.core.dataset.coord import Coord, LinearCoord
from spectrochempy.core.units import ur, Quantity
from spectrochempy.core import debug_

from spectrochempy.utils.testing import (
    assert_array_equal,
    assert_equal_units,
    assert_approx_equal,
    assert_produces_warning,
)

# ======================================================================================================================
# Coord
# ======================================================================================================================


def test_coord():
    # simple coords

    a = Coord([1, 2, 3], name="x")
    assert a.id is not None
    assert not a.is_empty
    assert not a.is_masked
    assert_array_equal(a.data, np.array([1, 2, 3]))
    assert not a.is_labeled
    assert a.units is None
    assert a.unitless
    debug_(a.meta)
    assert not a.meta
    assert a.name == "x"

    # set properties

    a.title = "xxxx"
    assert a.title == "xxxx"
    a.name = "y"
    assert a.name == "y"
    a.meta = None
    a.meta = {"val": 125}  # need to be an OrderedDic
    assert a.meta["val"] == 125

    # now with labels

    x = np.arange(10)
    y = list("abcdefghij")
    a = Coord(x, labels=y, title="processors", name="x")
    assert a.title == "processors"
    assert isinstance(a.data, np.ndarray)
    assert isinstance(a.labels, np.ndarray)

    # any kind of object can be a label

    assert a.labels.dtype == "O"
    # even an array
    a._labels[3] = x
    assert a._labels[3][2] == 2

    # coords can be defined only with labels

    y = list("abcdefghij")
    a = Coord(labels=y, title="processors")
    assert a.title == "processors"
    assert isinstance(a.labels, np.ndarray)
    assert_array_equal(a.values, a.labels)
    # any kind of object can be a label
    assert a.labels.dtype == "O"
    # even an array
    a._labels[3] = range(10)
    assert a._labels[3][2] == 2

    # coords with datetime in labels

    x = np.arange(10)
    y = [np.datetime64(f"2017-06-{(2 * (i + 1)):02d}") for i in x]

    a = Coord(x, labels=y, title="time")
    assert a.title == "time"
    assert isinstance(a.data, np.ndarray)
    assert isinstance(a.labels, np.ndarray)
    b = a._sort(by="label", descend=True)
    assert_array_equal(b.data, a.data[::-1])
    b = a._sort(by="label", descend=True, inplace=True)
    assert_array_equal(b.data, a.data)

    # actually y can also be data
    c = Coord(y, title="time")
    assert c.title == "time"
    assert isinstance(c.data, np.ndarray)
    assert isinstance(c.data[0], np.datetime64)
    assert c.dtype == np.dtype("datetime64[D]")
    c._sort(descend=True, inplace=True)
    assert_array_equal(b.labels, c.data)

    # but coordinates must be 1D

    with pytest.raises(ValueError):
        # should raise an error as coords must be 1D
        Coord(data=np.ones((2, 10)))

    # unitless coordinates

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        mask=None,
        units=None,
        title="wavelength",
    )
    assert coord0.units is None
    assert coord0.data[0] == 4000.0
    assert repr(coord0) == "Coord: [float64] unitless (size: 10)"

    # dimensionless coordinates

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        mask=None,
        units=ur.dimensionless,
        title="wavelength",
    )
    assert coord0.units.dimensionless
    assert coord0.units.scaling == 1.0
    assert coord0.data[0] == 4000.0
    assert repr(coord0) == "Coord: [float64]  (size: 10)"

    # scaled dimensionless coordinates

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        mask=None,
        units="m/km",
        title="wavelength",
    )
    assert coord0.units.dimensionless
    assert (
        coord0.data[0] == 4000.0
    )  # <- displayed data to be multiplied by the scale factor
    assert repr(coord0) == "Coord: [float64] scaled-dimensionless (0.001) (size: 10)"

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        mask=None,
        units=ur.m / ur.km,
        title="wavelength",
    )

    assert coord0.units.dimensionless
    assert (
        coord0.data[0] == 4000.0
    )  # <- displayed data to be multiplied by the scale factor
    assert repr(coord0) == "Coord: [float64] scaled-dimensionless (0.001) (size: 10)"

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        mask=None,
        units="m^2/s",
        title="wavelength",
    )
    assert not coord0.units.dimensionless
    assert coord0.units.scaling == 1.0
    assert coord0.data[0] == 4000.0
    assert repr(coord0) == "Coord: [float64] m^2.s^-1 (size: 10)"

    # comparison

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        mask=None,
        title="wavelength",
    )
    coord0b = Coord(
        data=np.linspace(4000, 1000, 10),
        labels="a b c d e f g h i j".split(),
        mask=None,
        title="wavelength",
    )
    coord1 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels="a b c d e f g h i j".split(),
        mask=None,
        title="titi",
    )
    coord2 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels="b c d e f g h i j a".split(),
        mask=None,
        title="wavelength",
    )
    coord3 = Coord(
        data=np.linspace(4000, 1000, 10), labels=None, mask=None, title="wavelength"
    )

    assert coord0 == coord0b
    assert coord0 != coord1  # different title
    assert coord0 != coord2  # different labels
    assert coord0 != coord3  # one coord has no label

    # init from another coord

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        units="s",
        mask=None,
        title="wavelength",
    )

    coord1 = Coord(coord0)
    assert coord1._data is coord0._data
    coord1 = Coord(coord0, copy=True)
    assert coord1._data is not coord0._data
    assert_array_equal(coord1._data, coord0._data)
    assert isinstance(coord0, Coord)
    assert isinstance(coord1, Coord)

    # sort

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        units="s",
        mask=None,
        title="wavelength",
    )
    assert coord0.is_labeled
    ax = coord0._sort()
    assert ax.data[0] == 1000
    coord0._sort(descend=True, inplace=True)
    assert coord0.data[0] == 4000
    ax1 = coord0._sort(by="label", descend=True)
    assert ax1.labels[0] == "j"

    # copy

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        units="s",
        mask=None,
        title="wavelength",
    )

    coord1 = coord0.copy()
    assert coord1 is not coord0

    assert_array_equal(coord1.data, coord0.data)
    assert_array_equal(coord1.labels, coord0.labels)
    assert coord1.units == coord0.units

    coord2 = copy(coord0)
    assert coord2 is not coord0

    assert_array_equal(coord2.data, coord0.data)
    assert_array_equal(coord2.labels, coord0.labels)
    assert coord2.units == coord0.units

    # automatic reversing for wavenumbers

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10), units="cm^-1", mask=None, title="wavenumbers"
    )
    assert coord0.reversed

    assert not coord0.is_complex
    assert not coord0.is_empty
    assert coord0.T == coord0
    assert_array_equal(coord0.masked_data, coord0.data)


def test_coord_slicing():
    # slicing by index

    coord0 = Coord(data=np.linspace(4000, 1000, 10), mask=None, title="wavelength")

    assert coord0[0] == 4000.0

    coord1 = Coord(
        data=np.linspace(4000, 1000, 10), units="cm^-1", mask=None, title="wavelength"
    )
    c1 = coord1[0]
    assert isinstance(c1.values, Quantity)
    assert coord1[0].values == 4000.0 * (1.0 / ur.cm)

    # slicing with labels

    labs = list("abcdefghij")

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=labs,
        units="cm^-1",
        mask=None,
        title="wavelength",
    )

    assert coord0[0].values == 4000.0 * (1.0 / ur.cm)
    assert isinstance(coord0[0].values, Quantity)

    assert coord0[2] == coord0["c"]
    assert coord0["c":"d"] == coord0[2:4]  # label included

    # slicing only-labels coordinates

    y = list("abcdefghij")
    a = Coord(labels=y, name="x")
    assert a.name == "x"
    assert isinstance(a.labels, np.ndarray)
    assert_array_equal(a.values, a.labels)


# Math
# ------------------------------------------------------------------

# first operand has units km, second is a scalar with units m
@pytest.mark.parametrize(
    ("operation", "result_units"),
    [
        ("__add__", ur.km),
        ("__sub__", ur.km),
        ("__mul__", ur.km * ur.m),
        ("__truediv__", ur.km / ur.m),
    ],
)
def test_coord_unit_conversion_operators_a(operation, result_units):
    print(operation, result_units)
    in_km = Coord(
        data=np.linspace(4000, 1000, 10), units="km", mask=None, title="something"
    )

    scalar_in_m = 2.0 * ur.m

    operator_km = in_km.__getattribute__(operation)

    combined = operator_km(scalar_in_m)

    assert_equal_units(combined.units, result_units)


UNARY_MATH = [
    "fabs",
    "ceil",
    "floor",
    "negative",
    "reciprocal",
    "rint",
    "sqrt",
    "square",
]


@pytest.mark.parametrize("name", UNARY_MATH)
def test_coord_unary_ufuncs_simple_data(name):
    coord0 = Coord(
        data=np.linspace(4000, 1000, 10), units="km", mask=None, title="something"
    )

    f = getattr(np, name)
    r = f(coord0)
    assert isinstance(r, Coord)


# first operand has units km, second is a scalar unitless
@pytest.mark.parametrize(
    ("operation", "result_units"), [("__mul__", ur.km), ("__truediv__", ur.km)]
)
def test_coord_unit_conversion_operators(operation, result_units):
    in_km = Coord(
        data=np.linspace(4000, 1000, 10), units="km", mask=None, title="something"
    )

    scalar = 2.0

    operator_km = in_km.__getattribute__(operation)

    combined = operator_km(scalar)
    debug_(f"{operation}, {combined}")
    assert_equal_units(combined.units, result_units)


NOTIMPL = [
    "average",
    "clip",
    "mean",
    "pipe",
    "remove_masks",
    "std",
    "cumsum",
    "sum",
    "swapdims",
    "swapaxes",
    "squeeze",
    "random",
    "empty",
    "empty_like",
    "var",
    "ones",
    "ones_like",
    "full",
    "diag",
    "diagonal",
    "full_like",
    "identity",
    "eye",
    "zeros",
    "zeros_like",
    "coordmin",
    "coordmax",
    "conjugate",
    "conj",
    "abs",
    "absolute",
    "all",
    "any",
    "argmax",
    "argmin",
    "asfortranarray",
    "origin",
]


@pytest.mark.parametrize("name", NOTIMPL)
def test_coord_not_implemented(name):
    coord0 = Coord(
        data=np.linspace(4000, 1000, 10), units="cm^-1", mask=None, title="wavelength"
    )
    with pytest.raises(NotImplementedError):
        getattr(coord0, name)()


def test_linearcoord():
    coord1 = Coord([1, 2.5, 4, 5])

    coord2 = Coord(np.array([1, 2.5, 4, 5]))
    assert coord2 == coord1

    coord3 = Coord(range(10))

    coord4 = Coord(np.arange(10))
    assert coord4 == coord3

    coord5 = coord4.copy()
    coord5 += 1
    assert np.all(coord5.data == coord4.data + 1)

    assert coord5 is not None
    coord5.linear = True

    coord6 = Coord(linear=True, offset=2.0, increment=2.0, size=10)
    assert np.all(coord6.data == (coord4.data + 1.0) * 2.0)

    LinearCoord(offset=2.0, increment=2.0, size=10)

    coord0 = LinearCoord.linspace(
        200.0,
        300.0,
        3,
        labels=["cold", "normal", "hot"],
        units="K",
        title="temperature",
    )
    coord1 = LinearCoord.linspace(
        0.0, 60.0, 100, labels=None, units="minutes", title="time-on-stream"
    )
    coord2 = LinearCoord.linspace(
        4000.0, 1000.0, 100, labels=None, units="cm^-1", title="wavenumber"
    )

    assert coord0.size == 3
    assert coord1.size == 100
    assert coord2.size == 100

    coordc = coord0.copy()
    assert coord0 == coordc

    coordc = coord1.copy()
    assert coord1 == coordc

    assert_approx_equal(coord1.spacing.m, 0.606060606)

    assert coord1.author is None
    assert not coord1.history

    assert not coord1.descendant
    assert coord2.descendant

    assert coord1.is_1d

    assert coord0.transpose() == coord0


def test_datetime64_coordinates(IR_dataset_2D):

    X = IR_dataset_2D
    assert X.y.dtype == np.dtype(
        "datetime64[ns]"
    )  # in spectrochempy internal datetime are in units of nanoseconds
    assert (
        X.y.units == None
    )  # there is no units for this object as it is defined internally

    # with assert_produces_warning(match="method `to` cannot be used with datetime object. Ignored!"):
    X.y.to("tesla")  # can not change to a foreign unit (of course)

    # subtract a datetime to a datetime array --> should be a float
    X.y = X.y - X.y[0]  # subtract the acquisition timestamp of the first spectrum
    assert X.y.data.dtype == np.dtype("float")

    X.y = X.y.to("minute")  # convert to minutes
    assert X.y.units == ur.minute
    X.y += 2  # add 2 minutes
    assert X.y.units == ur.minute
    assert X.y[0].data == [
        120
    ]  # check that the addition is correctly done 2 min (internally 120s)
