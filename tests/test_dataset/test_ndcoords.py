# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL FREE SOFTWARE LICENSE AGREEMENT (Version 2.1)
# See full LICENSE agreement in the root directory
# =============================================================================



"""

"""

from copy import copy

import numpy as np
import pytest

from spectrochempy.api import *

from spectrochempy.utils.traittypes import Range

from traitlets import HasTraits
from traitlets import TraitError


from tests.utils import (assert_array_equal,
                         assert_equal_units)


# warnings.simplefilter(action="ignore", category=FutureWarning)


class MinimalCoordSetSubclass(Coord):
    pass


def test_coordarray_subclass():
    a = MinimalCoordSetSubclass([1, 2, 3])
    assert a.name is not None
    assert not a.is_empty
    assert not a.is_masked
    assert_array_equal(a.data, np.array([1, 2, 3]))
    assert not a.is_labeled
    assert a.units is None
    assert a.unitless
    assert not a.is_uncertain
    print((a.meta))
    assert not a.meta

    # set
    a.title = 'xxxx'
    a.meta = None
    a.meta = {'val': 125}  # need to be an OrderedDict
    assert a.meta['val'] == 125
    pass


# noinspection PyProtectedMember
def test_coordarray_withlabels():
    x = np.arange(10)
    y = [i for i in 'abcdefghij']
    a = MinimalCoordSetSubclass(x, labels=y, title='processors')
    assert a.title == 'processors'
    assert isinstance(a.data, np.ndarray)
    assert isinstance(a.labels, np.ndarray)
    # any knid of object can be a label
    assert a.labels.dtype == 'O'
    # even an array
    a._labels[3] = x
    assert a._labels[3][2] == 2

def test_coordarray_withonlylabels():
    y = [i for i in 'abcdefghij']
    a = MinimalCoordSetSubclass(labels=y, title='processors')
    assert a.title == 'processors'
    assert not a.data
    assert isinstance(a.labels, np.ndarray)
    # any kind of object can be a label
    assert a.labels.dtype == 'O'
    # even an array
    a._labels[3] = range(10)
    assert a._labels[3][2] == 2

def test_coordarray_with_datetime():
    from datetime import datetime
    x = np.arange(10)
    y = [datetime(2017, 6, 2 * (i + 1)) for i in x]

    a = MinimalCoordSetSubclass(x, labels=y, title='time')
    assert a.title == 'time'
    assert isinstance(a.data, np.ndarray)
    assert isinstance(a.labels, np.ndarray)
    b = a._sort(by='label', descend=True)
    # print(b)


def test_coord_init_unitless():
    # unitless
    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels='a b c d e f g h i j'.split(),
                   mask=None,
                   units=None,
                   title='wavelength')
    assert coord0.units is None
    assert coord0.data[0] == 4000.
    assert repr(
        coord0) == "Coord: [4000.000, 3666.667, ..., 1333.333, 1000.000] unitless"


def test_coord_init_dimensionless():
    # dimensionless specified
    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels='a b c d e f g h i j'.split(),
                   mask=None,
                   units=ur.dimensionless,
                   title='wavelength')
    assert coord0.units.dimensionless
    assert coord0.units.scaling == 1.
    assert coord0.data[0] == 4000.
    assert repr(
        coord0) == "Coord: [4000.000, 3666.667, ..., 1333.333, 1000.000] dimensionless"


def test_coord_init_dimensionless_scaled():
    # scaled dimensionless implicit
    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels='a b c d e f g h i j'.split(),
                   mask=None,
                   units='m/km',
                   title='wavelength')
    assert coord0.units.dimensionless
    assert coord0.data[
               0] == 4000.  # <- displayed data to be multiplied by the scale factor
    assert repr(
        coord0) == "Coord: [4000.000, 3666.667, ..., 1333.333, 1000.000] scaled-dimensionless (0.001)"

    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels='a b c d e f g h i j'.split(),
                   mask=None,
                   units=ur.m / ur.km,
                   title='wavelength')

    assert coord0.units.dimensionless
    assert coord0.data[
               0] == 4000.  # <- displayed data to be multiplied by the scale factor
    assert repr(
        coord0) == "Coord: [4000.000, 3666.667, ..., 1333.333, 1000.000] scaled-dimensionless (0.001)"


def test_coord_init_specific_units():
    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels='a b c d e f g h i j'.split(),
                   mask=None,
                   units="m^2/s",
                   title='wavelength')
    assert not coord0.units.dimensionless
    assert coord0.units.scaling == 1.
    assert coord0.data[0] == 4000.
    assert repr(
        coord0) == "Coord: [4000.000, 3666.667, ..., 1333.333, 1000.000] m^2.s^-1"


def test_coord_equal():
    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels='a b c d e f g h i j'.split(),
                   mask=None,
                   title='wavelength')
    coord0b = Coord(data=np.linspace(4000, 1000, 10),
                    labels='a b c d e f g h i j'.split(),
                    mask=None,
                    title='wavelength')
    coord1 = Coord(data=np.linspace(4000, 1000, 10),
                   labels='a b c d e f g h i j'.split(),
                   mask=None,
                   title='titi')
    coord2 = Coord(data=np.linspace(4000, 1000, 10),
                   labels='b c d e f g h i j a'.split(),
                   mask=None,
                   title='wavelength')
    coord3 = Coord(data=np.linspace(4000, 1000, 10),
                   labels=None,
                   mask=None,
                   title='wavelength')

    assert coord0 == coord0b
    assert coord0 == coord1  # but different title (not important)
    assert coord0 != coord2  # different labels
    assert coord0 != coord3  # one coord has no label


def test_coords_equal():
    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels='a b c d e f g h i j'.split(),
                   mask=None,
                   title='wavelength')
    coord1 = Coord(data=np.linspace(4000, 1000, 10),
                   labels='a b c d e f g h i j'.split(),
                   mask=None,
                   title='titi')
    coord2 = Coord(data=np.linspace(4000, 1000, 10),
                   labels='b c d e f g h i j a'.split(),
                   mask=None,
                   title='wavelength')
    coord3 = Coord(data=np.linspace(4000, 1000, 10),
                   labels=None,
                   mask=None,
                   title='wavelength')

    coordsa = CoordSet([coord0,  coord3, coord2])
    coordsb = CoordSet([coord0, coord3, coord2])
    assert coordsa == coordsb

    coord4 = copy(coord2)
    coordsc = CoordSet([coord1, coord2, coord4])
    assert coordsa != coordsc


def test_set_coord_from_another_coord():
    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels='a b c d e f g h i j'.split(),
                   units='s',
                   mask=None,
                   title='wavelength')

    coord1 = Coord(coord0)
    assert coord1._data is coord0._data
    coord1 = Coord(coord0, copy=True)
    assert coord1._data is not coord0._data
    assert_array_equal(coord1._data, coord0._data)
    assert isinstance(coord0, Coord)
    assert isinstance(coord1, Coord)


def test_coord_sort():
    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels='a b c d e f g h i j'.split(),
                   units='s',
                   mask=None,
                   title='wavelength')
    assert coord0.is_labeled
    ax = coord0._sort()
    assert (ax.data[0] == 1000)
    coord0._sort(descend=True, inplace=True)
    assert (coord0.data[0] == 4000)
    ax1 = coord0._sort(by='label', descend=True)
    assert (ax1.labels[0] == 'j')


def test_coord_copy():
    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels='a b c d e f g h i j'.split(),
                   units='s',
                   mask=None,
                   title='wavelength')

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


def test_coords_str_repr():
    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels='a b c d e f g h i j'.split(),
                   units='cm^-1',
                   mask=None,
                   title='wavelength')

    assert str(coord0).split('\n')[0].strip() == 'title: Wavelength'
    assert repr(
        coord0) == "Coord: [4000.000, 3666.667, ..., 1333.333, 1000.000] cm^-1"

    coords = CoordSet([coord0, coord0.copy()])
    assert str(coords) == "[wavelength, wavelength]"
    assert repr(coords).startswith("CoordSet object <<object ")

    print(coord0)


def test_coords_copy():
    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels='a b c d e f g h i j'.split(),
                   units='cm^-1',
                   mask=None,
                   title='wavelength')

    coords = CoordSet([coord0, coord0.copy()])
    coords1 = coords[:]
    assert coords is not coords1


def test_coords_slicing():
    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   # labels='a b c d e f g h i j'.split(),
                   units='cm^-1',
                   mask=None,
                   title='wavelength')
    c0 = coord0[0]
    assert coord0[0].data == 4000.0
    assert coord0[0] == 4000.0 * (1. / ur.cm)


def test_coords_slicing_with_labels():
    labs = 'a b c d e f g h i j'.split()

    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels=labs,
                   units='cm^-1',
                   mask=None,
                   title='wavelength')

    assert coord0[0].data == 4000.0
    assert coord0[0] == 4000.0 * (1. / ur.cm)
    assert type(coord0[0]) == Coord

    print((Quantity("4000 cm^-1").m))
    assert isinstance(coord0[0].data, float)

    print((coord0[0].values))
    assert isinstance(coord0[0].values, Quantity)
    assert isinstance(coord0.values, Quantity)

    assert coord0[2].labels == labs[2]


# first operand has units km, second is a scalar with units m
@pytest.mark.parametrize(('operation', 'result_units'), [
    ('__add__', ur.km),
    ('__sub__', ur.km),
    ('__mul__', ur.km * ur.m),
    ('__truediv__', ur.km / ur.m)])
def test_unit_conversion_operators(operation, result_units):
    in_km = Coord(data=np.linspace(4000, 1000, 10),
                  units='km',
                  mask=None,
                  title='something')

    scalar_in_m = 2. * ur.m

    operator_km = in_km.__getattribute__(operation)

    combined = operator_km(scalar_in_m)
    ##print(operation, combined)
    assert_equal_units(combined.units, result_units)


# first operand has units km, second is an array with units m
@pytest.mark.parametrize(('operation', 'result_units'), [
    ('__add__', ur.km),
    ('__sub__', ur.km),
    ('__mul__', ur.km * ur.m),
    ('__truediv__', ur.km / ur.m)])
def test_unit_conversion_operators(operation, result_units):
    in_km = Coord(data=np.linspace(4000, 1000, 10),
                  units='km',
                  mask=None,
                  title='something')

    array_in_m = np.arange(in_km.size) * ur.m

    operator_km = in_km.__getattribute__(operation)

    combined = operator_km(array_in_m)
    # print(operation, combined)
    assert_equal_units(combined.units, result_units)
    assert isinstance(combined, Coord)


UNARY_MATH = ["fabs", "ceil", "floor", "negative", "reciprocal",
              "rint", "sqrt", "square"]


@pytest.mark.parametrize('name', UNARY_MATH)
def test_unary_ufuncs_simple_data(name):
    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   units='km',
                   mask=None,
                   title='something')

    f = getattr(np, name)
    r = f(coord0)
    assert isinstance(r, Coord)


# first operand has units km, second is a scalar unitless
@pytest.mark.parametrize(('operation', 'result_units'), [
    ('__mul__', ur.km),
    ('__truediv__', ur.km)])
def test_unit_conversion_operators(operation, result_units):
    in_km = Coord(data=np.linspace(4000, 1000, 10),
                  units='km',
                  mask=None,
                  title='something')

    scalar = 2.

    operator_km = in_km.__getattribute__(operation)

    combined = operator_km(scalar)
    # print(operation, combined)
    assert_equal_units(combined.units, result_units)


class testRangeTrait(HasTraits):
    x = Range()

    def __init__(self):
        assert not self.x

        self.x = [2, 1.]
        assert self.x == [1, 2]

        self.x = (2, 1.)
        assert self.x == [1, 2]

        self.x.reverse()
        assert self.x == [2, 1]

        assert self.x[0] == 2

        with pytest.raises(TraitError):
            self.x = [2]

        self.x = []
        assert not self.x

        pass


def testinterval():
    testRangeTrait()


def test_coordrange():
    r = CoordRange()
    assert r.ranges == []

    r = CoordRange(3, 2)
    assert r.ranges[0] == [2, 3]

    r = CoordRange((3, 2), (4.4, 10), (4, 5))
    assert r.ranges[-1] == [4, 10]
    assert r.ranges == [[2, 3], [4, 10]]

    r.reverse()
    assert r.ranges == [[10, 4], [3, 2]]



