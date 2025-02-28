# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from copy import copy

import numpy as np
import pytest

from spectrochempy.application import debug_
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.units import Quantity, ur
from spectrochempy.utils.testing import (
    assert_approx_equal,
    assert_array_equal,
    assert_units_equal,
)
from spectrochempy.utils.warnings import assert_produces_warning

# --------------------------------------------------------------------------------------
# Coord
# --------------------------------------------------------------------------------------


def test_coord():
    # simple coords
    # init with invalid data
    with pytest.raises(ValueError):
        # value must be iterable
        Coord(1)

    # init with a list of values
    a = Coord([1, 2, 3], name="x")
    assert_array_equal(a.data, np.array([1, 2, 3]))
    assert a.dtype == np.dtype(float)
    assert a.is_1d
    assert not a.is_descendant
    assert a.id is not None
    assert not a.is_empty
    assert not a.is_masked
    assert a.size == 3
    assert a.ndim == 1
    assert a.shape == (3,)
    assert a.is_labeled is False
    assert a.units is None
    assert a.unitless is True
    assert not a.meta
    assert a.name == "x"
    assert a.title == "<untitled>"

    # larmor frequency
    a = Coord([1, 2, 3], name="x", units="ppm", larmor=104.7 * ur.MHz)
    assert a.larmor == 104.7 * ur.MHz
    assert a.reversed is True
    assert a.units == ur.ppm

    # init with size 1
    a = Coord([1.001], name="x")
    assert a.data[0] == 1.001, "signdigit default=4"

    a = Coord([1.0001])
    assert a.data[0] == 1.0, "signdigit default=4"

    # init with size 0 object
    a = Coord([])
    assert np.all(a.data == np.array([]))

    # set properties
    a.title = "xxxx"
    assert a.title == "xxxx"
    a.name = "y"
    assert a.name == "y"
    a.meta = None
    a.meta = {"val": 125}
    assert a.meta["val"] == 125

    # init now with labels and values
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

    # coords with datetime

    from datetime import datetime

    x = np.arange(10)
    y = [datetime(2017, 6, 2 * (i + 1)) for i in x]

    a = Coord(x, labels=y, title="time")
    assert a.title == "time"
    assert isinstance(a.data, np.ndarray)
    assert isinstance(a.labels, np.ndarray)
    a._sort(by="label", descend=True)

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
    #   assert coord0.units.scaling == 1.0
    assert coord0.data[0] == 4000.0
    assert repr(coord0) == "Coord: [float64] dimensionless (size: 10)"

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
    # assert repr(coord0) == "Coord: [float64] scaled-dimensionless (0.001) (size: 10)"

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
    #  assert repr(coord0) == "Coord: [float64] scaled-dimensionless (0.001) (size: 10)"

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        mask=None,
        units="m^2/s",
        title="wavelength",
    )
    assert not coord0.units.dimensionless
    # assert coord0.units.scaling == 1.0
    assert coord0.data[0] == 4000.0
    assert repr(coord0) == "Coord: [float64] m²⋅s⁻¹ (size: 10)"

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
    assert coord1._data is not coord0._data  # data are always copied in this case
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

    # test default
    assert a.default is a

    # loc2index
    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        units="s",
        mask=None,
        title="wavelength",
    )
    assert coord0.loc2index(3500.0) == 1

    # check few additional methods

    # transpose (no effect)
    coord1 = coord0.T
    assert coord1 is coord0
    coord1 = coord0.transpose()
    assert coord1 is coord0

    # default and coord attributes (when they are call but it is not a CoordSet
    assert coord1 is coord1.default
    assert coord1 is coord1.coords

    # related to issue #643
    ### The following code works perfectly

    c0 = Coord.linspace(
        start=8000.0, stop=1250.0, num=6, labels=None, units="cm^-1", title="wavenumber"
    )

    from spectrochempy import NDDataset

    ds = NDDataset(np.random.random((6,)), coords=[c0])
    ds.x = c0
    assert ds.x.linear == True

    ds.x.ito("nm")  # check ito works
    assert ds.x.linear == False, "ito should have changed the coord to non-linear"

    assert ds.x.units == ur.nm, "ito should have changed the coord units to nm"
    assert (
        ds.x[0].value == 1250.0 * ur.nm
    ), "ito should have changed the coord values to nm"

    x = ds.x.to("cm^-1")  # check to works  (back to linear)
    assert x.linear == True, "to should have changed the coord to linear"
    assert x.units == ur.cm**-1, "to should have changed the coord units to cm^-1"
    assert (
        x[0].value == 8000.0 * ur.cm**-1
    ), "to should have changed the coord values to cm^-1"

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
    a = Coord(labels=labs, name="x")
    assert a.name == "x"
    assert isinstance(a.labels, np.ndarray)
    assert_array_equal(a.values, a.labels)

    # fancy indexing
    c = coord0[[0, 2, 4]]
    assert isinstance(c, Coord)
    assert c.shape == (3,)
    assert c[0] == coord0[0]
    assert c[1] == coord0[2]
    assert c[2] == coord0[4]

    #  slicing result to None (when the resulting coord is empty)
    c = coord0[5000.0:5500.0]
    assert c is None

    # axis reversed?
    coord0.units = "cm^-1"
    assert coord0.reversed

    # printing
    s = str(coord0)
    assert "Coord: [float64] cm" in s or "Coord: [float64] 1/cm" in s
    assert "(size: 10)" in s

    s = coord0._repr_html_()
    assert "<div class='label'>[  a   b ...   i   j]</div>" in s

    c = Coord()
    assert "<div class='numeric'>Undefined</div>" in c._repr_html_()

    # several row of label
    coord0.labels = list("klmnopqrst")
    s = coord0._repr_html_()
    assert "<br/>          [  k   l ...   s   t]]" in s

    # spacing
    sp = coord0.spacing
    assert sp == -333.3 * ur.cm**-1


# Math
# --------------------------------------------------------------------------------------
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
    "get_axis",
]


@pytest.mark.parametrize("name", NOTIMPL)
def test_coord_not_implemented(name):
    coord0 = Coord(
        data=np.linspace(4000, 1000, 10), units="cm^-1", mask=None, title="wavelength"
    )
    with pytest.raises(NotImplementedError):
        getattr(coord0, name)()


def test_linearcoord():
    from spectrochempy.core.dataset.coord import LinearCoord

    with assert_produces_warning(DeprecationWarning, check_stacklevel=False):
        _ = LinearCoord(offset=0, increment=10, size=10)

    # test it for creation using offset and increment
    coord0 = LinearCoord(offset=1, increment=10, size=10)
    assert isinstance(coord0, LinearCoord)
    assert coord0[0] == 1
    assert_array_equal(coord0.data, Coord.arange(1, 100, 10).data)
