# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
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

from copy import copy, deepcopy

import numpy as np
import pytest
from traitlets import TraitError

from pint import DimensionalityError
from spectrochempy.api import NDArray
from spectrochempy.api import ur
from tests.utils import (raises)
from ...utils import NumpyRNGContext


#warnings.simplefilter(action="ignore", category=FutureWarning)


class MinimalSubclass(NDArray):
    # in principle NDArray is not used directly and should be subclassed
    pass

def test_init_ndarray_subclass():
    # test initialization of an empty ndarray
    # check some of its properties
    a = MinimalSubclass()
    assert isinstance(a, NDArray)
    assert a.name != u'<no name>'  # must be the uuid in this case
    assert a.is_empty
    assert not a.is_masked
    assert not a.is_uncertain
    assert a.unitless
    assert a.is_untitled
    assert not a.meta


# noinspection PyProtectedMember,PyProtectedMember
def test_set_ndarray_subclass():
    # test of setting some attributes of an empty ndarray
    # check some of its properties
    a = MinimalSubclass()
    a._name = 'xxxx'
    assert a.name == u'xxxx'
    a._title = 'yyyy'
    assert not a.is_untitled
    assert a.title == u"yyyy"
    with pytest.raises(TraitError):
        a._meta = []  # need to be a Meta class
    a._meta.something = "a_value"
    assert a._meta.something == "a_value"

@pytest.fixture(scope="module")
def ndarraysubclass():
    # return a simple ndarray
    # with some data
    with NumpyRNGContext(12345):
        dx = 10.*np.random.random((10, 10))-5.
    _nd = MinimalSubclass()
    _nd._data = dx
    return _nd.copy()

def test_set_simple_ndarray(ndarraysubclass):
    nd = ndarraysubclass
    assert nd.data.size == 100
    assert nd.shape == (10, 10)
    assert nd.size == 100
    assert nd.ndim == 2
    assert nd.is_complex == [False, False]
    assert nd.data[1,1] == 4.6130673607282127

def test_set_ndarray_with_units(ndarraysubclass):
    nd = ndarraysubclass

    assert nd.unitless # ,o units
    assert not nd.dimensionless # no unit so no dimension has no sense


    #with catch_warnings() as w:
    nd.to('m')  # should not change anything (but raise a warning)
    #assert w[0].category == SpectroChemPyWarning

    assert nd.unitless

    nd.units = 'm'
    assert nd.units == ur.meter

    nd1 = nd.to('km')
    assert nd.units == ur.kilometer
    assert nd1.units == ur.kilometer
    #with catch_warnings() as w:
    nd.ito('m')
    #assert w[0].category == SpectroChemPyDeprecationWarning
    nd.to('m')
    assert nd.units == ur.meter

    # change of units - ok if it can be casted to the current one
    nd.units = 'cm'

    # cannot chnage to incompatible units
    with pytest.raises(ValueError):
        nd.units = 'radian'

    # we can force them
    nd.change_units('radian')

    assert 1 * nd.units == 1. * ur.dimensionless
    assert nd.units.dimensionless
    assert nd.dimensionless
    with raises(DimensionalityError):
        nd1 = nd1.ito('km/s')  # should raise an error
    nd.units = 'm/km'
    assert nd.units.dimensionless
    assert nd.units.scaling == 0.001

@pytest.fixture(scope="module")
def ndarraysubclasscplx():
    # return a complex ndarray
    # with some complex data
    with NumpyRNGContext(1245):
        dx = np.random.random((10, 20))
    _nd = MinimalSubclass()
    _nd._data = dx
    _nd.set_complex(axis=-1)  # this means that the data are complex in
    # the last dimension
    return _nd.copy()

def test_set_ndarray_with_complex(ndarraysubclasscplx):
    nd = ndarraysubclasscplx.copy()
    assert nd.data.size == 200
    assert nd.size == 100
    assert nd.data.shape == (10, 20)
    assert nd.shape == (10, 10)  # the real shape
    assert nd.is_complex == [False, True]
    assert nd.ndim == 2
    nd.units = 'meter'
    assert nd.units == ur.meter


def test_copy_of_ndarray(ndarraysubclasscplx):
    nd = copy(ndarraysubclasscplx)
    assert nd is not ndarraysubclasscplx
    assert nd.data.size == 200
    assert nd.size == 100
    assert nd.data.shape == (10, 20)
    assert nd.shape == (10, 10)  # the real shape
    assert nd.is_complex == [False, True]
    assert nd.ndim == 2


def test_deepcopy_of_ndarray(ndarraysubclasscplx):
    # for this example there is no diif with copy (write another test for this)
    nd1 = ndarraysubclasscplx.copy()
    nd = deepcopy(nd1)
    assert nd is not nd1
    assert nd.data.size == 200
    assert nd.size == 100
    assert nd.data.shape == (10, 20)
    assert nd.shape == (10, 10)  # the real shape
    assert nd.is_complex == [False, True]
    assert nd.ndim == 2


# noinspection PyProtectedMember
def test_ndarray_with_uncertaincy(ndarraysubclass):
    nd = ndarraysubclass.copy()
    assert not nd.is_uncertain
    assert repr(nd).startswith('NDArray:')
    nd._uncertainty = np.abs(nd._data * .01)
    assert nd.is_uncertain
    assert str(nd).startswith('NDArray:')


# noinspection PyProtectedMember
def test_ndarray_with_mask(ndarraysubclass):
    nd = ndarraysubclass.copy()
    assert not nd.is_masked
    assert str(nd).startswith('NDArray:')
    nd._mask[5] = True
    assert nd.is_masked


# noinspection PyProtectedMember
def test_ndarray_with_uncertaincy_and_units(ndarraysubclass):
    nd = ndarraysubclass.copy()
    nd.change_units('m')
    assert nd.units == ur.meter
    assert not nd.is_uncertain
    assert repr(nd).startswith('NDArray:')
    nd._uncertainty = np.abs(nd._data * .01)
    assert nd.is_uncertain
    assert str(nd).startswith('NDArray:')
    #print(nd)


# noinspection PyProtectedMember,PyProtectedMember
def test_ndarray_with_uncertaincy_and_units_being_complex(ndarraysubclasscplx):
    nd = ndarraysubclasscplx.copy()
    nd.units = 'm'
    assert nd.units == ur.meter
    assert not nd.is_uncertain
    assert repr(nd).startswith('NDArray:')
    nd._uncertainty = nd._data * .01
    assert nd.is_uncertain
    assert str(nd).startswith('NDArray:')
    #print(nd)
    assert nd._uncertainty.size == nd.data.size


def test_ndarray_len_and_sizes(ndarraysubclass, ndarraysubclasscplx):

    nd = ndarraysubclass
    #print(nd.is_complex)
    assert not nd.is_complex[0]
    assert len(nd) == 10
    assert nd.shape == (10, 10)
    assert nd.size == 100
    assert nd.ndim == 2

    nd = ndarraysubclasscplx
    #print(nd.is_complex)
    assert nd.is_complex[1]
    assert len(nd) == 10
    assert nd.shape == (10, 10)
    assert nd.size == 100
    assert nd.ndim == 2

