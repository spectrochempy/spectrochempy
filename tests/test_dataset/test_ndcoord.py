# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

from copy import copy

import numpy as np
import pytest

from spectrochempy.core.dataset.coord import Coord, LinearCoord
from spectrochempy.core.readers.importer import read
from spectrochempy.units import ur, Quantity
from spectrochempy.utils.testing import assert_array_equal, assert_equal_units
from spectrochempy.core import info_, print_, debug_


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
    debug_(a.meta)
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
    assert_array_equal(a.values, a.labels)
    # any kind of object can be a label
    assert a.labels.dtype == 'O'
    # even an array
    a._labels[3] = range(10)
    assert a._labels[3][2] == 2
    info_('\n' + str(a))
    info_('\n' + repr(a))

    # coords with datetime

    from datetime import datetime
    x = np.arange(10)
    y = [datetime(2017, 6, 2 * (i + 1)) for i in x]

    a = Coord(x, labels=y, title='time')
    assert a.title == 'time'
    assert isinstance(a.data, np.ndarray)
    assert isinstance(a.labels, np.ndarray)
    b = a._sort(by='label', descend=True)
    info_('\n' + str(b))

    # but coordinates must be 1D

    with pytest.raises(ValueError):
        # should raise an error as coords must be 1D
        Coord(data=np.ones((2, 10)))

    # unitless coordinates

    coord0 = Coord(data=np.linspace(4000, 1000, 10), labels=list('abcdefghij'), mask=None, units=None,
                   title='wavelength')
    assert coord0.units is None
    assert coord0.data[0] == 4000.
    assert repr(coord0) == 'Coord: [float64] unitless (size: 10)'

    info_('\n' + str(coord0))

    # dimensionless coordinates

    coord0 = Coord(data=np.linspace(4000, 1000, 10), labels=list('abcdefghij'), mask=None, units=ur.dimensionless,
                   title='wavelength')
    assert coord0.units.dimensionless
    assert coord0.units.scaling == 1.
    assert coord0.data[0] == 4000.
    assert repr(coord0) == 'Coord: [float64]  (size: 10)'

    # scaled dimensionless coordinates

    coord0 = Coord(data=np.linspace(4000, 1000, 10), labels=list('abcdefghij'), mask=None, units='m/km',
                   title='wavelength')
    assert coord0.units.dimensionless
    assert coord0.data[0] == 4000.  # <- displayed data to be multiplied by the scale factor
    assert repr(coord0) == 'Coord: [float64] scaled-dimensionless (0.001) (size: 10)'

    coord0 = Coord(data=np.linspace(4000, 1000, 10), labels=list('abcdefghij'), mask=None, units=ur.m / ur.km,
                   title='wavelength')

    assert coord0.units.dimensionless
    assert coord0.data[0] == 4000.  # <- displayed data to be multiplied by the scale factor
    assert repr(coord0) == 'Coord: [float64] scaled-dimensionless (0.001) (size: 10)'
    info_('\n' + str(coord0))

    coord0 = Coord(data=np.linspace(4000, 1000, 10), labels=list('abcdefghij'), mask=None, units="m^2/s",
                   title='wavelength')
    assert not coord0.units.dimensionless
    assert coord0.units.scaling == 1.
    assert coord0.data[0] == 4000.
    assert repr(coord0) == 'Coord: [float64] m^2.s^-1 (size: 10)'

    # comparison

    coord0 = Coord(data=np.linspace(4000, 1000, 10), labels=list('abcdefghij'), mask=None, title='wavelength')
    coord0b = Coord(data=np.linspace(4000, 1000, 10), labels='a b c d e f g h i j'.split(), mask=None,
                    title='wavelength')
    coord1 = Coord(data=np.linspace(4000, 1000, 10), labels='a b c d e f g h i j'.split(), mask=None, title='titi')
    coord2 = Coord(data=np.linspace(4000, 1000, 10), labels='b c d e f g h i j a'.split(), mask=None,
                   title='wavelength')
    coord3 = Coord(data=np.linspace(4000, 1000, 10), labels=None, mask=None, title='wavelength')

    assert coord0 == coord0b
    assert coord0 != coord1  # different title
    assert coord0 != coord2  # different labels
    assert coord0 != coord3  # one coord has no label

    # init from another coord

    coord0 = Coord(data=np.linspace(4000, 1000, 10), labels=list('abcdefghij'), units='s', mask=None,
                   title='wavelength')

    coord1 = Coord(coord0)
    assert coord1._data is coord0._data
    coord1 = Coord(coord0, copy=True)
    assert coord1._data is not coord0._data
    assert_array_equal(coord1._data, coord0._data)
    assert isinstance(coord0, Coord)
    assert isinstance(coord1, Coord)

    # sort

    coord0 = Coord(data=np.linspace(4000, 1000, 10), labels=list('abcdefghij'), units='s', mask=None,
                   title='wavelength')
    assert coord0.is_labeled
    ax = coord0._sort()
    assert (ax.data[0] == 1000)
    coord0._sort(descend=True, inplace=True)
    assert (coord0.data[0] == 4000)
    ax1 = coord0._sort(by='label', descend=True)
    assert (ax1.labels[0] == 'j')

    # copy

    coord0 = Coord(data=np.linspace(4000, 1000, 10), labels=list('abcdefghij'), units='s', mask=None,
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

    coord0 = Coord(data=np.linspace(4000, 1000, 10), units='cm^-1', mask=None, title='wavenumbers')
    assert coord0.reversed


def test_coord_slicing():
    # slicing by index

    coord0 = Coord(data=np.linspace(4000, 1000, 10), mask=None, title='wavelength')

    assert coord0[0] == 4000.0

    coord1 = Coord(data=np.linspace(4000, 1000, 10), units='cm^-1', mask=None, title='wavelength')
    c1 = coord1[0]
    assert isinstance(c1.values, Quantity)
    assert coord1[0].values == 4000.0 * (1. / ur.cm)

    # slicing with labels

    labs = list('abcdefghij')

    coord0 = Coord(data=np.linspace(4000, 1000, 10), labels=labs, units='cm^-1', mask=None, title='wavelength')

    assert coord0[0].values == 4000.0 * (1. / ur.cm)
    assert isinstance(coord0[0].values, Quantity)

    assert coord0[2] == coord0['c']
    assert coord0['c':'d'] == coord0[2:4]  # label included

    # slicing only-labels coordinates

    y = list('abcdefghij')
    a = Coord(labels=y, name='x')
    assert a.name == 'x'
    assert isinstance(a.labels, np.ndarray)
    assert_array_equal(a.values, a.labels)


# Math
# ----------------------------------------------------------------------------------------------------------------------

# first operand has units km, second is a scalar with units m
@pytest.mark.parametrize(('operation', 'result_units'),
                         [('__add__', ur.km), ('__sub__', ur.km), ('__mul__', ur.km * ur.m),
                          ('__truediv__', ur.km / ur.m)])
def test_coord_unit_conversion_operators_a(operation, result_units):
    print(operation, result_units)
    in_km = Coord(data=np.linspace(4000, 1000, 10), units='km', mask=None, title='something')

    scalar_in_m = 2. * ur.m

    operator_km = in_km.__getattribute__(operation)

    combined = operator_km(scalar_in_m)
    info_(f'{operation}, {combined}')
    assert_equal_units(combined.units, result_units)


UNARY_MATH = ["fabs", "ceil", "floor", "negative", "reciprocal", "rint", "sqrt", "square"]


@pytest.mark.parametrize('name', UNARY_MATH)
def test_coord_unary_ufuncs_simple_data(name):
    coord0 = Coord(data=np.linspace(4000, 1000, 10), units='km', mask=None, title='something')

    f = getattr(np, name)
    r = f(coord0)
    assert isinstance(r, Coord)


# first operand has units km, second is a scalar unitless
@pytest.mark.parametrize(('operation', 'result_units'), [('__mul__', ur.km), ('__truediv__', ur.km)])
def test_coord_unit_conversion_operators(operation, result_units):
    in_km = Coord(data=np.linspace(4000, 1000, 10), units='km', mask=None, title='something')

    scalar = 2.

    operator_km = in_km.__getattribute__(operation)

    combined = operator_km(scalar)
    debug_(f'{operation}, {combined}')
    assert_equal_units(combined.units, result_units)


NOTIMPL = ['mean', 'pipe', 'remove_masks', 'std', 'sum', 'swapdims']


@pytest.mark.parametrize('name', NOTIMPL)
def test_coord_not_implemented(name):
    coord0 = Coord(data=np.linspace(4000, 1000, 10), units='cm^-1', mask=None, title='wavelength')
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
    print_('Linear coordinate', coord5)

    coord6 = Coord(linear=True, offset=2.0, increment=2.0, size=10)
    assert np.all(coord6.data == (coord4.data + 1.0) * 2.)

    # %%
    coord7 = LinearCoord(offset=2.0, increment=2.0, size=10)
    print_(coord7)

    # %%
    X = read('irdata/CO@Mo_Al2O3.SPG')
    print_(X.x)
    XX = X.x.copy()
    X.x.linear = True
    print_((X.x.data - XX.data).ptp(), X.x.increment,
           (X.x.data - XX.data).ptp() * 100 / X.x.increment)  # TODO : why such a difference!!!!
    print(X.x._data)

    print(X.x.offset)
    X.x.ito('m^-1')
    print(X.x.offset_value)

    coord0 = LinearCoord.linspace(200., 300., 3, labels=['cold', 'normal', 'hot'], units="K", title='temperature')
    coord1 = LinearCoord.linspace(0., 60., 100, labels=None, units="minutes", title='time-on-stream')
    coord2 = LinearCoord.linspace(4000., 1000., 100, labels=None, units="cm^-1", title='wavenumber')

    assert coord0.size==3
    assert coord1.size==100
    assert coord2.size==100

    coordc = coord0.copy()
    assert coord0 == coordc

    coordc = coord1.copy()
    assert coord1 == coordc
