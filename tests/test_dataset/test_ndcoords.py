# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT 
# See full LICENSE agreement in the root directory
# ======================================================================================================================


"""

"""

from copy import copy

import numpy as np
import pytest

from traitlets import TraitError, HasTraits
from spectrochempy.core.dataset.ndcoords import Coord, CoordRange, CoordSet, Range
from spectrochempy.units import ur, Quantity
from spectrochempy.core.dataset.ndarray import NDArray

from spectrochempy.utils.testing import (assert_array_equal,
                                         assert_equal_units, raises)
from spectrochempy.core import log


# ======================================================================================================================
# Coord
# ======================================================================================================================

def test_coord():

    # simple coords

    a = Coord([1, 2, 3], name='x')
    assert a.id is not None
    assert not a.is_empty
    assert not a.is_masked
    assert_array_equal(a.data, np.array([1, 2, 3]))
    assert not a.is_labeled
    assert a.units is None
    assert a.unitless
    log.debug(a.meta)
    assert not a.meta
    assert a.name == 'x'

    # set properties

    a.title = 'xxxx'
    assert a.title == 'xxxx'
    a.name = 'y'
    assert a.name == 'y'
    a.meta = None
    a.meta = {'val': 125}  # need to be an OrderedDic
    assert a.meta['val'] == 125

    # now with labels

    x = np.arange(10)
    y = list('abcdefghij')
    a = Coord(x, labels=y, title='processors', name='x')
    assert a.title == 'processors'
    assert isinstance(a.data, np.ndarray)
    assert isinstance(a.labels, np.ndarray)

    # any kind of object can be a label

    assert a.labels.dtype == 'O'
    # even an array
    a._labels[3] = x
    assert a._labels[3][2] == 2

    # coords can be defined only with labels

    y = list('abcdefghij')
    a = Coord(labels=y, title='processors')
    assert a.title == 'processors'
    assert isinstance(a.labels, np.ndarray)
    assert_array_equal(a.values,a.labels)
    # any kind of object can be a label
    assert a.labels.dtype == 'O'
    # even an array
    a._labels[3] = range(10)
    assert a._labels[3][2] == 2
    log.info('\n'+str(a))
    log.info('\n'+repr(a))

    # coords with datetime

    from datetime import datetime
    x = np.arange(10)
    y = [datetime(2017, 6, 2 * (i + 1)) for i in x]

    a = Coord(x, labels=y, title='time')
    assert a.title == 'time'
    assert isinstance(a.data, np.ndarray)
    assert isinstance(a.labels, np.ndarray)
    b = a._sort(by='label', descend=True)
    log.info('\n'+str(b))


    # but coordinates must be 1D

    with pytest.raises(ValueError) as e_info:
        # should raise an error as coords must be 1D
        Coord(data=np.ones((2, 10)))

    # unitless coordinates

    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels= list('abcdefghij'),
                   mask=None,
                   units=None,
                   title='wavelength')
    assert coord0.units is None
    assert coord0.data[0] == 4000.
    assert repr(coord0) == "Coord: [float64] unitless"

    log.info('\n'+str(coord0))

    # dimensionless coordinates

    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels=list('abcdefghij'),
                   mask=None,
                   units=ur.dimensionless,
                   title='wavelength')
    assert coord0.units.dimensionless
    assert coord0.units.scaling == 1.
    assert coord0.data[0] == 4000.
    assert repr(coord0) == "Coord: [float64] dimensionless"

    # scaled dimensionless coordinates

    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels=list('abcdefghij'),
                   mask=None,
                   units='m/km',
                   title='wavelength')
    assert coord0.units.dimensionless
    assert coord0.data[0] == 4000.  # <- displayed data to be multiplied by the scale factor
    assert repr(coord0) == "Coord: [float64] scaled-dimensionless (0.001)"

    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels=list('abcdefghij'),
                   mask=None,
                   units=ur.m / ur.km,
                   title='wavelength')

    assert coord0.units.dimensionless
    assert coord0.data[0] == 4000.  # <- displayed data to be multiplied by the scale factor
    assert repr(coord0) == "Coord: [float64] scaled-dimensionless (0.001)"
    log.info('\n'+str(coord0))

    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels=list('abcdefghij'),
                   mask=None,
                   units="m^2/s",
                   title='wavelength')
    assert not coord0.units.dimensionless
    assert coord0.units.scaling == 1.
    assert coord0.data[0] == 4000.
    assert repr(coord0) == "Coord: [float64] m^2.s^-1"

    # comparison

    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels=list('abcdefghij'),
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
    assert coord0 != coord1  # different title
    assert coord0 != coord2  # different labels
    assert coord0 != coord3  # one coord has no label

    # init from another coord

    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels=list('abcdefghij'),
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

    # sort

    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels=list('abcdefghij'),
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

    # copy

    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels=list('abcdefghij'),
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


    # automatic reversing for wavenumbers

    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   units='cm^-1',
                   mask=None,
                   title='wavenumbers')
    assert coord0.reversed

    # not implemented

    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   units='cm^-1',
                   mask=None,
                   title='wavelength')
    with pytest.raises(AttributeError):
        c = coord0.real

def test_coord_slicing():

    # slicing by index

    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   mask=None,
                   title='wavelength')
    c0 = coord0[0]
    assert coord0[0] == 4000.0

    coord1 = Coord(data=np.linspace(4000, 1000, 10),
                   units='cm^-1',
                   mask=None,
                   title='wavelength')
    c1 = coord1[0]
    assert isinstance(c1.values, Quantity)
    assert coord1[0].values == 4000.0 * (1. / ur.cm)


    # slicing with labels

    labs = list('abcdefghij')

    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels=labs,
                   units='cm^-1',
                   mask=None,
                   title='wavelength')

    assert coord0[0].values == 4000.0 * (1. / ur.cm)
    assert isinstance(coord0[0].values, Quantity)

    assert coord0[2] == coord0['c']
    assert coord0['c':'d'] == coord0[2:4]   # label included

    # slicing only-labels coordinates

    y = list('abcdefghij')
    a = Coord(labels=y, name='x')
    assert a.name == 'x'
    assert isinstance(a.labels, np.ndarray)
    assert_array_equal(a.values, a.labels)


#########
# Math
#########

# first operand has units km, second is a scalar with units m
@pytest.mark.parametrize(('operation', 'result_units'), [
    ('__add__', ur.km),
    ('__sub__', ur.km),
    ('__mul__', ur.km * ur.m),
    ('__truediv__', ur.km / ur.m)])
def test_coord_unit_conversion_operators(operation, result_units):
    in_km = Coord(data=np.linspace(4000, 1000, 10),
                  units='km',
                  mask=None,
                  title='something')

    scalar_in_m = 2. * ur.m

    operator_km = in_km.__getattribute__(operation)

    combined = operator_km(scalar_in_m)
    log.debug(f'{operation}, {combined}')
    assert_equal_units(combined.units, result_units)


# first operand has units km, second is an array with units m
@pytest.mark.parametrize(('operation', 'result_units'), [
    ('__add__', ur.km),
    ('__sub__', ur.km),
    ('__mul__', ur.km * ur.m),
    ('__truediv__', ur.km / ur.m)])
def test_coord_unit_conversion_operators(operation, result_units):
    in_km = Coord(data=np.linspace(4000, 1000, 10),
                  units='km',
                  mask=None,
                  title='something')

    array_in_m = np.arange(in_km.size) * ur.m

    operator_km = in_km.__getattribute__(operation)

    combined = operator_km(array_in_m)
    log.debug(f'{operation}, {combined}')
    assert_equal_units(combined.units, result_units)
    assert isinstance(combined, Coord)


UNARY_MATH = ["fabs", "ceil", "floor", "negative", "reciprocal",
              "rint", "sqrt", "square"]


@pytest.mark.parametrize('name', UNARY_MATH)
def test_coord_unary_ufuncs_simple_data(name):
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
def test_coord_unit_conversion_operators(operation, result_units):
    in_km = Coord(data=np.linspace(4000, 1000, 10),
                  units='km',
                  mask=None,
                  title='something')

    scalar = 2.

    operator_km = in_km.__getattribute__(operation)

    combined = operator_km(scalar)
    log.debug(f'{operation}, {combined}')
    assert_equal_units(combined.units, result_units)


NOTIMPL = ['cumsum', 'mean',
           'pipe', 'remove_masks',
           'std', 'sum', 'swapaxes'
           ]


@pytest.mark.parametrize('name', NOTIMPL)
def test_coord_not_implemented(name):
    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   units='cm^-1',
                   mask=None,
                   title='wavelength')
    with pytest.raises(NotImplementedError):
        f = getattr(coord0, name)()

# ======================================================================================================================
# CoordSet
# ======================================================================================================================

def test_coordset_init():
    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels=list('abcdefghij'),
                   mask=None,
                   title='wavelength0')
    coord1 = Coord(data=np.linspace(4000, 1000, 10),
                   labels=list('abcdefghij'),
                   mask=None,
                   title='titi')
    coord2 = Coord(data=np.linspace(4000, 1000, 10),
                   labels=list('abcdefghij'),
                   mask=None,
                   title='wavelength2')
    coord3 = Coord(data=np.linspace(4000, 1000, 10),
                   labels=None,
                   mask=None,
                   title='wavelength3')

    coordsa = CoordSet([coord0, coord3, coord2])   # one syntax
    coordsb = CoordSet(coord0, coord3, coord2)     # a second syntax : equivalent
    coordsc = CoordSet(x=coord2, y=coord3,z=coord0) # third syntax
    coordsc1 = CoordSet({'x':coord2, 'y':coord3, 'z':coord0})
    coordsd = CoordSet(coord3, x=coord2, y=coord3, z=coord0) # conflict (keyw replace args)
    assert coordsa == coordsb
    assert coordsa == coordsc
    assert coordsa == coordsd
    assert coordsa == coordsc1
    c = coordsa["x"]
    assert c == coord2
    c = coordsa["y"]
    assert c == coord3
    assert coordsa['wavelength0'] == coord0

    coord4 = copy(coord2)
    coordsc = CoordSet([coord1, coord2, coord4])
    assert coordsa != coordsc

    coordse = CoordSet(x=[coord1,coord2], y=coord3, z=coord0) # coordset as coordinates
    assert coordse['x'] == CoordSet(coord1,coord2)
    assert coordse['x_1'] == coord2
    assert coordse['titi'] == coord1

    # iteration
    for coord in coordsa:
        assert isinstance(coord, Coord)

    for i, coord in enumerate(coordsa):
        assert isinstance(coord, Coord)



    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels='a b c d e f g h i j'.split(),
                   units='cm^-1',
                   mask=None,
                   title='wavelength')

    log.debug(str(coord0))
    log.debug(repr(coord0))
    assert repr(
        coord0) == "Coord: [float64] cm^-1"

    coords = CoordSet([coord0, coord0.copy()])
    log.debug(str(coords))

    assert repr(coords).startswith("CoordSet: [y:wavelength, x:wavelength]")

    # copy

    coord0 = Coord(data=np.linspace(4000, 1000, 10),
                   labels='a b c d e f g h i j'.split(),
                   units='cm^-1',
                   mask=None,
                   title='wavelength')

    coords = CoordSet([coord0, coord0.copy()])
    coords1 = coords[:]
    assert coords is not coords1


def test_coords_multicoord_for_a_single_dim():
    # normal coord (single numerical array for a axis)

    coord1 = NDArray(data=np.linspace(1000., 4000., 5),
                     labels='a b c d e'.split(), mask=None, units='cm^1',
                     title='wavelengths')

    coord0 = NDArray(data=np.linspace(20, 500, 5),
                     labels='very low-low-normal-high-very high'.split('-'),
                     mask=None, units='K', title='temperature')

    # pass as a list of coord
    coordsa = CoordSet([coord1, coord0])
    assert repr(coordsa) == 'CoordSet: [y:wavelengths, x:temperature]'
    assert not coordsa.is_same_dim

    # try to pass as an CoordSet
    coordsb = CoordSet(coordsa)
    assert not coordsb.is_same_dim

    # try to pass a arguments, each being an coord
    coordsc = CoordSet(coord1, coord0)
    assert not coordsc.is_same_dim
    assert repr(coordsc) == 'CoordSet: [y:wavelengths, x:temperature]'
    assert not coordsa.is_same_dim

    # try to pass arguments where each are a coords
    coordsd = CoordSet(coordsa, coordsc)
    assert repr(coordsd) == "CoordSet: [y:[_0:wavelengths, _1:temperature], x:[_0:wavelengths, _1:temperature]]"

    assert not coordsd.is_same_dim
    assert np.all([item.is_same_dim for item in coordsd])

    coordse = CoordSet(coordsa, coord1)
    assert repr(coordse) == "CoordSet: [y:[_0:wavelengths, _1:temperature], x:wavelengths]"

    assert not coordse.is_same_dim
    assert coordse[0].is_same_dim

    # bug with copy (lost name in copy)

    co = coordse[-1]
    assert isinstance(co, Coord)

    co = coordse[-1:]
    assert isinstance(co, CoordSet)
    assert co.names == ['x']  # should keep the original name (solved)
    assert co.x == coord1



# ======================================================================================================================
# CoordRange
# ======================================================================================================================

def test_coordrange():
    r = CoordRange()
    assert r == []

    r = CoordRange(3, 2)
    assert r[0] == [2, 3]

    r = CoordRange((3, 2), (4.4, 10), (4, 5))
    assert r[-1] == [4, 10]
    assert r == [[2, 3], [4, 10]]

    r = CoordRange((3, 2), (4.4, 10), (4, 5),
                   reversed=True)
    assert r == [[10, 4], [3, 2]]


# ======================================================================================================================
# Range
# ======================================================================================================================

def test_range():

    class MyClass(HasTraits):
        r = Range()  # Initialized with some default values

    c = MyClass()
    c.r = [10, 5]
    assert c.r == [5,10]
    with raises(TraitError):
        c.r = [10, 5, 1]
