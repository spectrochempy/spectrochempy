# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================

"""

"""

from copy import copy

import numpy as np
import pytest

from spectrochempy.api import *
from spectrochempy.core.dataset.ndaxes import AxisWarning

from tests.utils import (assert_array_equal,
                         assert_equal_units)


# warnings.simplefilter(action="ignore", category=FutureWarning)


class MinimalAxesSubclass(Axis):
    pass


def test_axisarray_subclass():
    a = MinimalAxesSubclass([1, 2, 3])
    assert a.name is not None
    assert a.is_untitled
    assert not a.is_empty
    assert not a.is_masked
    assert_array_equal(a.coords, np.array([1,2,3]) )
    assert not a.is_labeled
    assert a.units is None
    assert a.unitless
    assert not a.is_uncertain
    print(a.meta)
    assert not a.meta

    # set
    a.title = 'xxxx'
    a.meta = None
    a.meta = {'val':125}  # need to be an OrderedDict
    assert a.meta['val']==125
    pass


# noinspection PyProtectedMember
def test_axisarray_withlabels():
    x = np.arange(10)
    y =[i for i in 'abcdefghij']
    a = MinimalAxesSubclass(x, labels=y, title='processors')
    assert a.title == 'processors'
    assert isinstance(a.coords, np.ndarray)
    assert isinstance(a.labels, np.ndarray)
    # any knid of object can be a label
    assert a.labels.dtype == 'O'
    # even an array
    a._labels[3] = x
    assert a._labels[3][2] == 2


def test_axisarray_with_datetime():
    from datetime import datetime
    x = np.arange(10)
    y =[datetime(2017, 6, 2*(i+1)) for i in x]

    a = MinimalAxesSubclass(x, labels=y, title='time')
    assert a.title == 'time'
    assert isinstance(a.coords, np.ndarray)
    assert isinstance(a.labels, np.ndarray)
    b = a._sort(by='label', descend=True)
    #print(b)




def test_axis_init_unitless():
    # unitless
    axe0 = Axis(coords = np.linspace(4000, 1000, 10),
                labels = 'a b c d e f g h i j'.split(),
                mask = None,
                units = None,
                title = 'wavelength')
    assert axe0.units is None
    assert axe0.data[0] == 4000.
    assert axe0.coords[0] == 4000.
    assert repr(axe0) == "Axis([   4e+03, 3.67e+03, ..., 1.33e+03,    1e+03]) unitless"


def test_axis_init_dimensionless():
    # dimensionless specified
    axe0 = Axis(coords = np.linspace(4000, 1000, 10),
                labels = 'a b c d e f g h i j'.split(),
                mask = None,
                units = ur.dimensionless,
                title = 'wavelength')
    assert axe0.units.dimensionless
    assert axe0.units.scaling == 1.
    assert axe0.coords[0] == 4000.
    assert repr(axe0) == "Axis([   4e+03, 3.67e+03, ..., 1.33e+03,    1e+03]) dimensionless"


def test_axis_init_dimensionless_scaled():
    # scaled dimensionless implicit
    axe0 = Axis(coords = np.linspace(4000, 1000, 10),
                labels = 'a b c d e f g h i j'.split(),
                mask = None,
                units = 'm/km',
                title = 'wavelength')
    assert axe0.units.dimensionless
    assert axe0.coords[0] == 4000.     # <- displayed data to be multiplied by the scale factor
    assert repr(axe0) == "Axis([   4e+03, 3.67e+03, ..., 1.33e+03,    1e+03]) scaled-dimensionless (0.001)"

    axe0 = Axis(coords=np.linspace(4000, 1000, 10),
                labels='a b c d e f g h i j'.split(),
                mask=None,
                units=ur.m / ur.km,
                title='wavelength')

    assert axe0.units.dimensionless
    assert axe0.coords[0] == 4000.  # <- displayed data to be multiplied by the scale factor
    assert repr(axe0) == "Axis([   4e+03, 3.67e+03, ..., 1.33e+03,    1e+03]) scaled-dimensionless (0.001)"


def test_axis_init_specific_units():
    axe0 = Axis(coords = np.linspace(4000, 1000, 10),
                labels = 'a b c d e f g h i j'.split(),
                mask = None,
                units = "m^2/s",
                title = 'wavelength')
    assert not axe0.units.dimensionless
    assert axe0.units.scaling == 1.
    assert axe0.coords[0] == 4000.
    assert repr(axe0) == "Axis([   4e+03, 3.67e+03, ..., 1.33e+03,    1e+03]) m^2.s^-1"


def test_axis_equal():
    axe0 = Axis(coords = np.linspace(4000, 1000, 10),
                labels = 'a b c d e f g h i j'.split(),
                mask = None,
                title = 'wavelength')
    axe1 = Axis(coords = np.linspace(4000, 1000, 10),
                labels = 'a b c d e f g h i j'.split(),
                mask = None,
                title = 'titi')
    axe2 = Axis(coords = np.linspace(4000, 1000, 10),
                labels = 'a b c d e f g h i j a'.split(),
                mask = None,
                title = 'wavelength')
    axe3 = Axis(coords = np.linspace(4000, 1000, 10),
                labels = None,
                mask = None,
                title = 'wavelength')

    assert axe0 == axe1
    assert axe0 != axe2 # different labels
    assert axe0 == axe3 # one axe has no label (ignored)


def test_axes_equal():
    axe0 = Axis(coords = np.linspace(4000, 1000, 10),
                labels = 'a b c d e f g h i j'.split(),
                mask = None,
                title = 'wavelength')
    axe1 = Axis(coords = np.linspace(4000, 1000, 10),
                labels = 'a b c d e f g h i j'.split(),
                mask = None,
                title = 'titi')
    axe2 = Axis(coords = np.linspace(4000, 1000, 10),
                labels = 'a b c d e f g h i j a'.split(),
                mask = None,
                title = 'wavelength')
    axe3 = Axis(coords = np.linspace(4000, 1000, 10),
                labels = None,
                mask = None,
                title = 'wavelength')

    axesa = Axes([axe0, axe1, axe2])
    axesb = Axes([axe1, axe3, axe2])
    assert axesa == axesb

    axe4 = copy(axe2)
    axesc = Axes([axe1, axe2, axe4])
    assert axesa != axesc

def test_set_axis_from_another_axis():

    axe0 = Axis(coords = np.linspace(4000, 1000, 10),
                labels = 'a b c d e f g h i j'.split(),
                units= 's',
                mask = None,
                title = 'wavelength')


    axe1 = Axis(axe0)  # no copy by default
    assert axe1.coords is axe0.coords  # todo: is this really necessary?

    axe1 = Axis(axe0, iscopy=True)
    assert axe1.coords is not axe0.coords
    assert_array_equal(axe1.coords, axe0.coords)
    assert isinstance(axe0, Axis)
    assert isinstance(axe1, Axis)

def test_axis_sort():

    axe0 = Axis(coords = np.linspace(4000, 1000, 10),
                labels = 'a b c d e f g h i j'.split(),
                units= 's',
                mask = None,
                title = 'wavelength')
    assert axe0.is_labeled
    ax = axe0._sort()
    assert(ax.data[0]==1000)
    axe0._sort(descend=True, inplace=True)
    assert (axe0.data[0] == 4000)
    ax1 = axe0._sort(by='label', descend=True)
    assert (ax1.labels[0] == 'j')

def test_axis_copy():

    axe0 = Axis(coords = np.linspace(4000, 1000, 10),
                labels = 'a b c d e f g h i j'.split(),
                units= 's',
                mask = None,
                title = 'wavelength')

    axe1 = axe0.copy()
    assert axe1 is not axe0

    assert_array_equal(axe1.coords, axe0.coords)
    assert_array_equal(axe1.labels, axe0.labels)
    assert axe1.units == axe0.units

    axe2 = copy(axe0)
    assert axe2 is not axe0

    assert_array_equal(axe2.coords, axe0.coords)
    assert_array_equal(axe2.labels, axe0.labels)
    assert axe2.units == axe0.units

def test_axes_str_repr():
    axe0 = Axis(coords=np.linspace(4000, 1000, 10),
                labels='a b c d e f g h i j'.split(),
                units='cm^-1',
                mask=None,
                title='wavelength')

    assert str(axe0).split('\n')[0].strip() == 'title: Wavelength'
    assert repr(axe0) == "Axis([   4e+03, 3.67e+03, ..., 1.33e+03,    1e+03]) cm^-1"

    axes = Axes([axe0, axe0.copy()])
    assert str(axes)=="([wavelength], [wavelength])"
    assert repr(axes).startswith("Axes object <<Axis object ")

    print (axe0)

def test_axes_copy():
    axe0 = Axis(coords=np.linspace(4000, 1000, 10),
                labels='a b c d e f g h i j'.split(),
                units='cm^-1',
                mask=None,
                title='wavelength')

    axes = Axes([axe0, axe0.copy()])
    axes1 = axes[:]
    assert axes is not axes1

def test_axes_slicing():
    axe0 = Axis(coords=np.linspace(4000, 1000, 10),
                #labels='a b c d e f g h i j'.split(),
                units='cm^-1',
                mask=None,
                title='wavelength')

    assert axe0[0].coords == 4000.0
    assert axe0[0] == 4000.0 * (1./ur.cm)

def test_axes_slicing_with_labels():

    labs= 'a b c d e f g h i j'.split()

    axe0 = Axis(coords=np.linspace(4000, 1000, 10),
                labels=labs,
                units='cm^-1',
                mask=None,
                title='wavelength')

    assert axe0[0].data == 4000.0
    assert axe0[0] == 4000.0 * (1./ur.cm)
    assert type(axe0[0])==Axis

    print(Quantity("4000 cm^-1").m)
    assert isinstance(axe0[0].coords, np.ndarray)

    print (axe0[0].values)
    assert isinstance(axe0[0].values, Quantity)
    assert isinstance(axe0.values, Quantity)

    print (axe0[2].labels)
    assert axe0[2].labels == labs[2]


# first operand has units km, second is a scalar with units m
@pytest.mark.parametrize(('operation', 'result_units'), [
    ('__add__', ur.km),
    ('__sub__', ur.km),
    ('__mul__', ur.km * ur.m),
    ('__truediv__', ur.km / ur.m)])
def test_unit_conversion_operators(operation, result_units):

    in_km = Axis(coords=np.linspace(4000, 1000, 10),
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

    in_km = Axis(coords=np.linspace(4000, 1000, 10),
                units='km',
                mask=None,
                title='something')

    array_in_m = np.arange(in_km.size) * ur.m

    operator_km = in_km.__getattribute__(operation)

    combined = operator_km(array_in_m)
    #print(operation, combined)
    assert_equal_units(combined.units, result_units)
    assert isinstance(combined, Axis)


UNARY_MATH = ["fabs", "ceil", "floor", "negative", "reciprocal",
              "rint", "sqrt", "square"]
@pytest.mark.parametrize('name', UNARY_MATH)
def test_unary_ufuncs_simple_data(name):
    axe0 = Axis(coords=np.linspace(4000, 1000, 10),
                 units='km',
                 mask=None,
                 title='something')

    f = getattr(np, name)
    r = f(axe0)
    assert isinstance(r, Axis)

# first operand has units km, second is a scalar unitless
@pytest.mark.parametrize(('operation', 'result_units'), [
    ('__mul__', ur.km),
    ('__truediv__', ur.km)])
def test_unit_conversion_operators(operation, result_units):

    in_km = Axis(coords=np.linspace(4000, 1000, 10),
                units='km',
                mask=None,
                title='something')

    scalar = 2.

    operator_km = in_km.__getattribute__(operation)

    combined = operator_km(scalar)
    #print(operation, combined)
    assert_equal_units(combined.units, result_units)


from spectrochempy.core.api import AxisRange,  AxisRangeError
from spectrochempy.utils.traittypes import Range

from traitlets import HasTraits
from traitlets import TraitError

class testRangeTrait(HasTraits):

    x = Range()

    def __init__(self):

        assert not self.x

        self.x = [2,1.]
        assert self.x == [1,2]

        self.x = (2,1.)
        assert self.x == [1,2]

        self.x.reverse()
        assert self.x == [2,1]

        assert self.x[0] == 2

        with pytest.raises(TraitError):
            self.x = [2]

        self.x = []
        assert not self.x


        pass

def testinterval():
    testRangeTrait()

def test_axisrange():

    r = AxisRange()
    assert r.ranges == []

    r = AxisRange(3,2)
    assert r.ranges[0] == [2,3]

    r = AxisRange((3, 2), (4.4,10), (4,5))
    assert r.ranges[-1] == [4, 10]
    assert r.ranges == [[2,3],[4, 10]]

    r.reverse()
    assert r.ranges == [[10, 4], [3, 2]]

######

# multiaxes

def test_multiaxis_for_a_single_dim():

    # normal axis (single numerical array for a anxis)

    axe0 = Axis(coords = np.linspace(1000., 4000., 5),
                labels = 'a b c d e'.split(),
                mask = None,
                units='cm^1',
                title = 'wavelengths')

    axe1 = Axis(coords = np.linspace(20, 500, 5),
                labels = 'very low-low-normal-high-very high'.split('-'),
                mask = None,
                units = 'K',
                title = 'temperature')

    # pass as a list of axis
    axesa = Axes([axe0, axe1])
    assert str(axesa) == '([wavelengths], [temperature])'

    # try to pass as an Axes
    axesb = Axes(axesa)
    assert str(axesb) == '([wavelengths], [temperature])'

    # try to pass a arguments, each being an axis
    axesc = Axes(axe0, axe1)
    assert not axesc.issamedim
    assert str(axesc) == '([wavelengths], [temperature])'


    # try to pass a arguments, each being an axes
    axesc._transpose()
    axesd = Axes(axesa, axesc)
    assert str(axesd) == "([['wavelengths', 'temperature']], " \
                          "[['temperature', 'wavelengths']])"

    assert not axesd.issamedim
    assert np.all([item.issamedim for item in axesd])

    axesd._transpose()
    assert str(axesd) == "([['temperature', 'wavelengths']], " \
                          "[['wavelengths', 'temperature']])"

    with pytest.warns(AxisWarning):
        axesd[0]._transpose()



def test_axes_manipulation(IR_source_1):

    source = IR_source_1
    axe0 = source.axes[0]

    axe0 -= axe0[0]

    print(axe0)
