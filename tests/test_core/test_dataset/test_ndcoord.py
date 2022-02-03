# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

# flake8: noqa


from copy import copy

import numpy as np
import pytest
from pint.errors import DimensionalityError


from traitlets import TraitError, HasTraits

from spectrochempy.core.dataset.coord import (
    Coord,
    LinearCoord,
    CoordSet,
    DEFAULT_DIM_NAME,
)
from spectrochempy.core.units import ur, Quantity
from spectrochempy.core import debug_
from spectrochempy.core.dataset.coord import trim_ranges
from spectrochempy.utils import Range

from spectrochempy.utils.testing import (
    assert_array_equal,
    assert_units_equal,
    assert_approx_equal,
    # assert_produces_warning
)

from spectrochempy.utils.exceptions import (
    InvalidDimensionNameError,
    InvalidCoordinatesTypeError,
    InvalidCoordinatesSizeError,
)

# ========
# FIXTURES
# ========


@pytest.fixture(scope="function")
def coord0():
    c = Coord(
        data=np.linspace(4000.0, 1000.0, 10),
        labels=list("abcdefghij"),
        units="cm^-1",
        long_name="wavenumber (coord0)",
    )
    return c


@pytest.fixture(scope="function")
def coord1():
    c = LinearCoord(
        data=np.linspace(0.0, 60.0, 100), units="s", long_name="time-on-stream (coord1)"
    )
    return c


@pytest.fixture(scope="function")
def coord2():
    c = Coord(
        data=np.linspace(200.0, 300.0, 3),
        labels=["cold", "normal", "hot"],
        units="K",
        long_name="temperature (coord2)",
    )
    return c


# ==========
# Coord Test
# ==========


def test_coord_init():

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

    assert_units_equal(combined.units, result_units)


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
    assert_units_equal(combined.units, result_units)


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
    assert X.y.dtype == np.dtype("datetime64[s]")
    assert X.y.units is None
    # there is no units for this object as it is defined internally

    # with assert_produces_warning(match="method `to` cannot be used with datetime object. Ignored!"):
    X.y.to("tesla")
    # can not change to a foreign unit (of course)

    # subtract a datetime to a datetime array --> should be a float
    X.y = X.y - X.y[0]  # subtract the acquisition timestamp of the first spectrum
    assert X.y.dtype == np.dtype("float")
    assert X.y._units == ur.second

    X.y = X.y.to("minute")  # convert to minutes
    assert X.y.units == ur.minute
    X.y += 2
    # add 2 minutes
    assert X.y.units == ur.minute
    assert X.y[0].data == [2]


# ==============
# CoordSet Tests
# ==============


def test_coordset_init(coord0, coord1, coord2):

    coord3 = coord2.copy()
    coord3.long_name = "My name is titi"

    # First syntax
    # Coordinates are sorted in the coordset
    coordsa = CoordSet(coord0, coord3, coord2)
    # by convention if the names are not specified, then the coordinantes follows the order of dims, so they are in reverse order with respect of the coorset where the coords are ordered by names alphabetical order.

    assert coordsa.names == ["x", "y", "z"]
    assert coordsa.long_names == [
        "temperature (coord2)",
        "My name is titi",
        "wavenumber (coord0)",
    ]
    assert coordsa.long_names == coordsa.titles

    # Dims specified
    coordsa = CoordSet(coord0, coord3, coord2, dims=["x", "y", "z"])
    assert coordsa.names == ["x", "y", "z"]
    assert coordsa.titles == [
        "wavenumber (coord0)",
        "My name is titi",
        "temperature (coord2)",
    ]

    # Wrong dim specified
    with pytest.raises(InvalidDimensionNameError) as e:
        coordsa = CoordSet(coord0, coord3, coord2, dims=["toto", "y", "z"])
    assert "dim=`toto` was given!" in str(e.value)

    # Second syntax with a tuple of coordinates
    coordsb = CoordSet((coord0, coord3, coord2))
    assert coordsb.names == ["x", "y", "z"]

    # But warning
    # A list means that it is a sub-coordset (different meaning)
    coordsa1 = CoordSet([coord0[:3], coord3[:3], coord2[:3]])
    assert coordsa1.names == ["x"]
    assert coordsa1.x.names == ["_1", "_2", "_3"]

    # Third syntax
    coordsc = CoordSet(x=coord2, y=coord3, z=coord0)
    assert coordsc.names == ["x", "y", "z"]

    coordsc1 = CoordSet({"x": coord2, "y": coord3, "z": coord0})
    assert coordsc1.names == ["x", "y", "z"]

    coordsd = CoordSet(coord3, x=coord2, y=coord3, z=coord0)
    # conflict (keyw replace args)
    assert coordsb == coordsd
    assert coordsb == coordsc1

    coord4 = copy(coord2)
    coordsc = CoordSet([coord1[:3], coord2[:3], coord4[:3]])
    assert coordsa != coordsc

    # coordset as coordinates
    coordse = CoordSet(x=(coord1[:3], coord2[:3]), y=coord3, z=coord0)
    assert coordse["x"].titles == CoordSet(coord1, coord2).titles
    assert coordse["x_1"] == coord2
    assert coordse["My name is titi"] == coord3

    # iteration
    for coord in coordsa:
        assert isinstance(coord, Coord)

    for i, coord in enumerate(coordsa):
        assert isinstance(coord, Coord)

    assert repr(coord0) == "Coord: [float64] cm^-1 (size: 10)"

    coords = CoordSet(coord0.copy(), coord0)

    assert repr(coords).startswith("CoordSet: [x:wavenumber")

    with pytest.raises(InvalidCoordinatesTypeError):
        _ = CoordSet(2, 3)  # Coord in CoordSet cannot be simple scalar

    coords = CoordSet(x=coord2, y=coord3, z=None)
    assert coords.names == ["x", "y", "z"]
    assert coords.z.is_empty

    coords = CoordSet(x=coord2, y=coord3, z=np.array((1, 2, 3)))
    assert coords.names == ["x", "y", "z"]
    assert coords.z.size == 3

    with pytest.raises(InvalidDimensionNameError):
        _ = CoordSet(x=coord2, y=coord3, fx=np.array((1, 2, 3)))
        # wrong key (must be a single char)

    with pytest.raises(InvalidCoordinatesTypeError):
        _ = CoordSet(x=coord2, y=coord3, z=3)
        # wrong coordinate value

    # set a coordset from another one
    coords = CoordSet(**coordse)
    assert coordse.names == ["x", "y", "z"]
    assert coords.names == ["x", "y", "z"]
    assert coords == coordse

    # not recommended
    coords2 = CoordSet(*coordse)
    # loose the names so the ordering may be different
    assert coords2.names == ["x", "y", "z"]
    assert coords.x == coords2.z

    # invalid size
    with pytest.raises(InvalidCoordinatesSizeError):  # size for same dim does not match
        _ = CoordSet(coord0, [coord1, coord2])


def test_coordset_implements_method(coord0, coord1):
    c = CoordSet(coord0, coord1)
    assert c.implements("CoordSet")
    assert c.implements() == "CoordSet"


# read_only properties


def test_coordset_available_names_property(coord0, coord1, coord2):
    c = CoordSet(coord0, coord1, coord2)
    assert c.available_names == DEFAULT_DIM_NAME[:-3]

    c = CoordSet(coord0, coord0.copy(), [coord1, coord1.copy()])
    assert c.available_names == DEFAULT_DIM_NAME[:-3]


def test_coordset_coords_property(coord0, coord1, coord2):
    c = CoordSet(coord0, coord1, coord2)
    assert c.coords == [coord2, coord1, coord0]


def test_coordset_has_defined_names_property(coord0, coord1, coord2):
    c = CoordSet(coord0, coord1, coord2)
    assert not c.has_defined_name
    c.name = "Toto"
    assert c.has_defined_name


def test_coordset_is_empty_property():
    c = CoordSet()
    assert c.implements("CoordSet")
    assert c.is_empty


def test_coordset_is_same_dim_property(coord0):
    c = CoordSet([coord0, coord0.copy()])
    assert c.x.is_same_dim


def test_coordset_references_property():
    c = CoordSet(x=LinearCoord.arange(10), y="x")
    assert isinstance(c.references, dict)
    assert c.references == dict(y="x")


def test_coordset_sizes_property(coord0, coord1, coord2):
    c = CoordSet(x=coord0, y="x")
    assert c.sizes == [coord0.size]

    c = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    assert c.sizes == [coord1.size, coord0.size, coord2.size]

    # alias
    assert c.size == [coord1.size, coord0.size, coord2.size]


def test_coordset_names_property(coord0, coord1, coord2):
    c = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    assert c.names == ["x", "y", "z"]


def test_coordset_default_property(coord0, coord1):
    c = CoordSet([coord0, coord0.copy()], coord1)
    assert c.default is None
    cx = c.x
    assert cx.default is cx
    cy = c.y
    assert cy.default.name == "_1"


# mutable properties


def test_coordset_data_property(coord0, coord1):
    c = CoordSet([coord0, coord0.copy()], coord1)
    assert_array_equal(c.y.data, coord0.data)
    assert_array_equal(c.x.data, coord1.data)
    assert c.data is None


def test_coordset_name_property(coord0):
    c = CoordSet(coord0)
    assert c.name == c.id

    c.name = "Toto"
    assert c.name == "Toto"


def test_coordset_titles_property(coord0, coord1):
    c = CoordSet(coord0, coord1)
    assert c.titles == ["time-on-stream (coord1)", "wavenumber (coord0)"]

    with pytest.deprecated_call():
        _ = c.titles


def test_coordset_long_names_property(coord0, coord1, coord2):
    c = CoordSet(coord0, [coord1, coord1.copy()], coord2)
    assert c.long_names == [
        "temperature (coord2)",
        ["time-on-stream (coord1)", "time-on-stream (coord1)"],
        "wavenumber (coord0)",
    ]


def test_coordset_labels_property(coord0, coord1, coord2):
    c = CoordSet(coord0, [coord1, coord1.copy()], coord2)
    assert_array_equal(c.labels[0], np.array(["cold", "normal", "hot"]))


def test_coordset_unit_property(coord0, coord1, coord2):
    c = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    assert c.units == ["second", ["1 / centimeter", "1 / centimeter"], "kelvin"]
    assert c.y.units == ["1 / centimeter", "1 / centimeter"]


# public methods


def test_coordset_keys_method(coord0, coord1, coord2):
    c = CoordSet(x=coord0, z="x", y=[coord2, coord2.copy()])
    assert c.names == c.keys()[:2]  # 'z' is not included in names but in keys()


def test_coordset_select_method(coord0, coord1):
    c = CoordSet([coord0, coord0.copy()], coord1)
    cy = c.y
    assert cy.default.name == "_1"
    cy.select(2)
    assert cy.default.name == "_2"


def test_coordset_set_method(coord0, coord1, coord2):
    c = CoordSet(x=coord0)
    c0 = c.copy()
    assert c == c0
    assert c.x == coord0

    c0.set()
    assert c == c0, "nothing happens"

    # set with kwargs

    c0.set(x=coord1)
    assert c0.x == coord1, "x is replaced by a new coordinate!"

    c0.set(y=coord2)
    assert c0.x == coord1
    assert c0.y == coord2

    # set with args
    # append
    c0.set(coord0)
    assert c0.coords == [coord1, coord2, coord0]

    # owerwrite existing
    c0.set(coord0, coord0.copy(), mode="w")
    assert c0.coords == [coord0, coord0]

    # replace selected
    c0.set(y=coord1)
    assert c0.coords == [coord0, coord1]

    # with a coordset

    ca = CoordSet([coord0[:3], coord1[:3], coord2[:3]])
    c0.set(ca, mode="w")
    assert c0.coords == ca.coords


def test_coordset_set_long_names_method(coord0, coord1, coord2):
    coord0b = coord0.copy()
    coord0b.long_name = "bibi"
    c = CoordSet(coord2, [coord0, coord0b], coord1)
    assert c.long_names == [
        "time-on-stream (coord1)",
        ["wavenumber (coord0)", "bibi"],
        "temperature (coord2)",
    ]

    c.set_titles("time", "dddd", "celsius")
    assert (
        str(c) == "CoordSet: [x:time, y:[_1:wavenumber (coord0), _2:bibi], z:celsius]"
    )

    c.set_long_names(x="time", z="celsius", y_1="length")
    assert str(c) == repr(c) == "CoordSet: [x:time, y:[_1:length, _2:bibi], z:celsius]"

    c.set_long_names("t", ("l", "g"), x="x")
    assert str(c) == "CoordSet: [x:x, y:[_1:l, _2:g], z:celsius]"

    c.set_long_names(("t", ("l", "g")), z="z")
    assert str(c) == "CoordSet: [x:t, y:[_1:l, _2:g], z:z]"

    c.set_long_names()  # nothing happens
    assert str(c) == "CoordSet: [x:t, y:[_1:l, _2:g], z:z]"


def test_coordset_set_units_method(coord0, coord1, coord2):
    coord0b = coord0.copy()
    coord0b.long_name = "bibi"
    c = CoordSet(coord2, [coord0, coord0b], coord1)

    assert c.units == ["second", ["1 / centimeter", "1 / centimeter"], "kelvin"]

    with pytest.raises(DimensionalityError) as e:  # because units doesn't match
        c.set_units(("s", ("m", "s")), z="radian")
    assert (
        str(e.value)
        == "Cannot convert from '1 / centimeter' (1 / [length]) to 'second' ([time])"
    )

    c.set_units(("km/s", ("m", "s")), z="radian", force=True)  # force change
    assert c.y_1.units == ur("m")


def test_coordset_update_method(coord0, coord1, coord2):
    coord0b = coord0.copy()
    coord0b.long_name = "bibi"

    c = CoordSet(coord2, [coord0, coord0b], coord1)
    assert c.names == ["x", "y", "z"]
    assert (
        str(c.coords)
        == "[LinearCoord: [float64] s (size: 100), CoordSet: [_1:wavenumber (coord0), _2:bibi], Coord: [float64] K (size: 3)]"
    )

    c.update(x=coord2)
    assert (
        str(c.coords)
        == "[Coord: [float64] K (size: 3), CoordSet: [_1:wavenumber (coord0), _2:bibi], Coord: [float64] K (size: 3)]"
    )

    c.update(y_1=coord1)
    assert (
        str(c.coords)
        == "[Coord: [float64] K (size: 3), CoordSet: [_1:time-on-stream (coord1), _2:bibi], Coord: [float64] K (size: 3)]"
    )

    with pytest.raises(InvalidDimensionNameError):
        c.update(k=coord1)


def test_coordset_loc2index_method(coord0, coord1, coord2):
    coord0b = coord0.copy()
    coord0b.long_name = "bibi"

    c = CoordSet(coord2, [coord0, coord1[:10]], coord0)

    with pytest.raises(TypeError):
        c.loc2index(3000.0)

    idx = c.y.loc2index(3000.0)
    assert idx == 3
    assert c.y_1[idx].value == 3000 * ur("1/cm")

    c.y.select(2)  # change default
    idx2 = c.y.loc2index(4.8)
    assert_approx_equal(c.y_2[idx2].m, 4.848, significant=3)

    assert c.y.loc2index(20) == c.y_2.loc2index(20.0)


def test_coordset_call_method(coord0, coord1):
    coordsa = CoordSet(coord0, coord1)

    assert (
        str(coordsa) == "CoordSet: [x:time-on-stream (coord1), y:wavenumber (coord0)]"
    )
    a = coordsa(1, 0)
    assert a == coordsa

    b = coordsa(1)
    assert b == coord0  # reordering

    c = coordsa("x")
    assert c == coord1

    d = coordsa("time-on-stream (coord1)")
    assert d == coord1

    with pytest.raises(KeyError):
        coordsa("x_a")  # do not exit

    coordsa("y_a")


def test_coordset_get_item_method(coord0, coord1, coord2):
    c = CoordSet(coord2, [coord0[:5], coord0[5:]], coord1)

    coord = c["temperature (coord2)"]
    assert str(coord) == "Coord: [float64] K (size: 3)"
    assert coord.name == "z"

    coord = c["wavenumber (coord0)"]
    assert coord.name == "_1"

    coord = c["y_2"]
    assert coord.name == "_2"

    coord = c["_1"]
    assert coord.name == "_1"

    with pytest.raises(TypeError):
        c[3000.0]

    val = c.y[3000.0]
    assert val == 3000 * ur("1/cm")

    c.y.select(2)  # change default
    val = c.y[3000.0]  # out of limits
    assert int(val.m) == 2333


def test_coordset_del_item_method(coord0, coord1, coord2):
    c = CoordSet(coord2, [coord0, coord0.copy()], coord1)

    assert (
        str(c)
        == repr(c)
        == "CoordSet: [x:time-on-stream (coord1), y:[_1:wavenumber (coord0), _2:wavenumber (coord0)], z:temperature (coord2)]"
    )

    del c["temperature (coord2)"]
    assert (
        str(c)
        == repr(c)
        == "CoordSet: [x:time-on-stream (coord1), y:[_1:wavenumber (coord0), _2:wavenumber (coord0)]]"
    )

    del c.y["wavenumber (coord0)"]
    assert (
        str(c)
        == repr(c)
        == "CoordSet: [x:time-on-stream (coord1), y:[_2:wavenumber (coord0)]]"
    )

    c = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    del c["wavenumber (coord0)"]
    assert (
        str(c)
        == repr(c)
        == "CoordSet: [x:time-on-stream (coord1), y:[_2:wavenumber (coord0)], z:temperature (coord2)]"
    )

    c = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    del c.y_2
    assert (
        str(c)
        == repr(c)
        == "CoordSet: [x:time-on-stream (coord1), y:[_1:wavenumber (coord0)], z:temperature (coord2)]"
    )

    c = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    del c.y._1
    assert (
        str(c)
        == repr(c)
        == "CoordSet: [x:time-on-stream (coord1), y:[_2:wavenumber (coord0)], z:temperature (coord2)]"
    )

    # execute some uncovered part of the coordset methods

    # __dir__
    assert "is_same_dim" in c.__dir__()

    # __setattr__ error
    with pytest.raises(AttributeError):
        del c.temperature

    # __getattr__ error
    with pytest.raises(AttributeError):
        x = c.temperature

    # _repr_html
    assert c._repr_html_().startswith("<table style='background:transparent'>")


def test_coordset_copy_method(coord0, coord1):
    coord2 = LinearCoord.linspace(200.0, 300.0, 3, units="K", long_name="temperature")

    coordsa = CoordSet(coord0, coord1, coord2)

    coordsb = coordsa.copy()
    assert coordsa == coordsb
    assert coordsa is not coordsb
    assert coordsa(1) == coordsb(1)
    assert coordsa(1).name == coordsb(1).name

    # copy
    coords = CoordSet(coord0, coord0.copy())
    coords1 = coords[:]
    assert coords is not coords1

    import copy

    coords2 = copy.deepcopy(coords)
    assert coords == coords2  #


def test_coordset_str_repr_method(coord0, coord1, coord2):
    coords = CoordSet(coord2, [coord0, coord0.copy()], coord1)

    assert (
        str(coords)
        == repr(coords)
        == "CoordSet: [x:time-on-stream (coord1), y:[_1:wavenumber (coord0), _2:wavenumber (coord0)], z:temperature (coord2)]"
    )
    assert repr(coords) == str(coords)


def test_coordset_set_item_method(coord0, coord1, coord2):
    coords = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    assert (
        str(coords)
        == repr(coords)
        == "CoordSet: [x:time-on-stream (coord1), y:[_1:wavenumber (coord0), _2:wavenumber (coord0)], z:temperature (coord2)]"
    )
    coords.set_long_names("time-on-stream", ("wavenumber", "wavenumber"), "temperature")

    # set item

    coords["z"] = coord2
    assert (
        str(coords)
        == "CoordSet: [x:time-on-stream, y:[_1:wavenumber, _2:wavenumber], z:temperature (coord2)]"
    )

    with pytest.raises(KeyError):
        coords["temperature"] = coord1

    coords["temperature (coord2)"] = coord1
    assert (
        str(coords)
        == "CoordSet: [x:time-on-stream, y:[_1:wavenumber, _2:wavenumber], z:time-on-stream (coord1)]"
    )

    coords["y_2"] = coord2
    assert (
        str(coords)
        == "CoordSet: [x:time-on-stream, y:[_1:wavenumber, _2:temperature (coord2)], z:time-on-stream (coord1)]"
    )

    coords["_1"] = coord2
    assert (
        str(coords)
        == "CoordSet: [x:time-on-stream, y:[_1:temperature (coord2), _2:temperature (coord2)], z:time-on-stream (coord1)]"
    )

    coords["t"] = coord2
    assert (
        str(coords)
        == "CoordSet: [x:time-on-stream, y:[_1:temperature (coord2), _2:temperature (coord2)], z:time-on-stream (coord1), t:temperature (coord2)]"
    )

    coord2.long_name = "zaza"
    coords["temperature (coord2)"] = coord2
    assert (
        str(coords)
        == "CoordSet: [x:time-on-stream, y:[_1:temperature (coord2), _2:temperature (coord2)], z:time-on-stream (coord1), t:zaza]"
    )

    coords["temperature (coord2)"] = coord2
    assert (
        str(coords)
        == "CoordSet: [x:time-on-stream, y:[_1:zaza, _2:temperature (coord2)], z:time-on-stream (coord1), t:zaza]"
    )

    coords.set(coord1, coord0, coord2, mode="w")
    assert (
        str(coords)
        == "CoordSet: [x:zaza, y:wavenumber (coord0), z:time-on-stream (coord1)]"
    )

    coords.z = coord0
    assert (
        str(coords)
        == "CoordSet: [x:zaza, y:wavenumber (coord0), z:wavenumber (coord0)]"
    )

    coords.zaza = coord0
    assert (
        str(coords)
        == "CoordSet: [x:wavenumber (coord0), y:wavenumber (coord0), z:wavenumber (coord0)]"
    )


def test_issue_310():

    import spectrochempy as scp
    import numpy as np

    D = scp.NDDataset(np.zeros((10, 5)))
    c5 = scp.Coord.linspace(start=0.0, stop=1000.0, num=5, name="xcoord")
    c10 = scp.Coord.linspace(start=0.0, stop=1000.0, num=10, name="ycoord")

    assert c5.name == "xcoord"
    D.set_coordset(x=c5, y=c10)
    assert c5.name == "x"
    assert str(D.coordset) == "CoordSet: [x:<untitled>, y:<untitled>]"
    assert D.dims == ["y", "x"]

    E = scp.NDDataset(np.ndarray((10, 5)))
    E.dims = ["t", "s"]
    E.set_coordset(s=c5, t=c10)
    assert c5.name == "s"
    assert str(E.coordset) == "CoordSet: [s:<untitled>, t:<untitled>]"
    assert E.dims == ["t", "s"]

    E = scp.NDDataset(np.ndarray((5, 10)))
    E.dims = ["s", "t"]
    E.set_coordset(s=c5, t=c10)
    assert c5.name == "s"
    assert str(E.coordset) == "CoordSet: [s:<untitled>, t:<untitled>]"
    assert E.dims == ["s", "t"]

    assert str(D.coordset) == "CoordSet: [x:<untitled>, y:<untitled>]"
    assert D.dims == ["y", "x"]

    assert str(D[:, 1]) == "NDDataset: [float64] unitless (shape: (y:10, x:1))"


# ======================================================================================================================
# trim_ranges
# ======================================================================================================================


def test_trim_ranges():
    r = trim_ranges()
    assert r == []

    r = trim_ranges(3, 2)
    assert r[0] == [2, 3]

    r = trim_ranges((3, 2), (4.4, 10), (4, 5))
    assert r[-1] == [4, 10]
    assert r == [[2, 3], [4, 10]]

    r = trim_ranges((3, 2), (4.4, 10), (4, 5), reversed=True)
    assert r == [[10, 4], [3, 2]]


# ======================================================================================================================
# Range
# ======================================================================================================================


def test_range():
    class MyClass(HasTraits):
        r = Range()  # Initialized with some default values

    c = MyClass()
    c.r = [10, 5]
    assert c.r == [5, 10]
    with pytest.raises(TraitError):
        c.r = [10, 5, 1]
