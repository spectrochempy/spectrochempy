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

from copy import copy, deepcopy

import numpy as np
import pytest
from datetime import datetime

from pint import DimensionalityError
from spectrochempy.core.dataset.ndarray import NDArray
from spectrochempy.core.units import ur
from spectrochempy.utils import SpectroChemPyWarning, \
    SpectroChemPyDeprecationWarning
from tests.utils import (assert_equal, assert_array_equal,
                         assert_array_almost_equal, assert_equal_units,
                         raises)
from tests.utils import NumpyRNGContext, catch_warnings
from spectrochempy.extern.traittypes import Array

### TEST INITIALIZATION

def test_init_ndarray_void():

    d0 = NDArray(None) # void initialization
    assert d0.shape == (0,)
    assert d0.size == 0
    assert not d0.is_masked
    assert d0.dtype == 'float64'
    assert not d0.has_complex_dims
    assert (repr(d0)=='NDArray: [] unitless')
    assert (str(d0) == '[]')

def test_init_ndarray_quantity():

    d0 = NDArray(13. * ur.tesla) #initialisation with a quantity
    assert d0.units == 'tesla'
    d0 = NDArray((2,3,4)) # initialisation with a sequence
    assert d0.shape == (3,)
    assert d0.size == 3
    assert not d0.has_complex_dims
    assert not d0.is_masked
    assert str(d0) == '[       2        3        4]'

def test_init_ndarray_sequence():

    d0 = NDArray([2,3,4,5]) # initialisation with a sequence
    assert d0.shape == (4,)
    assert not d0.is_masked
    assert not d0.has_complex_dims

def test_init_ndarray_array():

    d1 = NDArray(np.ones((5, 5))) #initialization with an array
    assert d1.shape == (5, 5)
    assert d1.size == 25
    assert not d1.is_masked
    assert not d1.has_complex_dims

def test_init_ndarray_NDArray():

    d1 = NDArray(np.ones((5, 5)))
    assert d1.shape == (5, 5)
    d2 = NDArray(d1) # initialization with an NDArray object
    assert d2.shape == (5,5)
    assert not d2.has_complex_dims
    assert d2.size == 25
    assert not d2.is_masked
    assert d1.data is d2.data # by default we do not copy

def test_init_ndarray_NDArray_copy():

    d1 = NDArray(np.ones((5, 5)))
    d2 = NDArray(d1, copy=True) # initialization with an NDArray object
    assert d2.shape == (5,5)
    assert not d2.has_complex_dims
    assert d2.size == 25
    assert not d2.is_masked
    assert d1.data is not d2.data # we have forced a copy

def test_init_ndarray_with_a_mask():

    d0mask = NDArray([2, 3, 4, 5], mask=[1,0,0,0])  # sequence + mask
    assert d0mask.shape == (4,)
    assert not d0mask.has_complex_dims
    assert d0mask.is_masked
    assert d0mask.mask.shape == d0mask.shape
    assert str(d0mask).startswith('[  --        3        4        5]')
    assert repr(d0mask).startswith(
            'NDArray: [  --,        3,        4,        5]')

def test_init_ndarray_with_a_mask_and_uncertainty():
    d0unc = NDArray([2, 3, 4, 5], uncertainty=[.1,.2,.15,.21],
                     mask=[1,0,0,0])  # sequence + mask + uncert
    assert d0unc.shape == (4,)
    assert not d0unc.has_complex_dims
    assert d0unc.is_masked
    assert d0unc.is_uncertain
    assert str(d0unc).startswith('[  --    3.000+/-0.200 ')
    assert repr(d0unc).startswith(
            'NDArray: [  --,    3.000+/-0.200,    4.000+/-0.150,')

def test_init_complex_ndarray():

    # test with complex data in the last dimension
    d = np.ones((2, 2))*np.exp(.1j)
    d1 = NDArray(d)
    assert d1.has_complex_dims
    assert d1.is_complex[-1]
    assert d1.shape == (2, 2)
    assert d1.size == 4
    assert repr(d1).startswith('NDArray: [[   0.995,    0.100, ')

    d2 = NDArray(d1)
    assert d1.data is d2.data
    assert np.all(d1.data == d2.data)
    assert np.all(d2.is_complex==[False, True])
    assert d2.shape == (2,2)
    assert str(d2).startswith('RR[[   0.995    0.995]')
    assert 'RI[[   0.100    0.100]' in str(d2)

    np.random.seed(12345)
    d = np.random.random((2, 2)) * np.exp(.1j)
    d3 = NDArray(d, units=ur.Hz,
                 mask=[[False, True], [False, False]])  # with units & mask
    assert d3.shape == (2, 2)
    assert d3._data.shape == (2,4)
    assert d3.size == 4
    assert d3.dtype == np.complex
    assert d3.has_complex_dims
    assert d3.mask.shape[-1] == d3.shape[-1] * 2
    d3RR = d3.part('RR')
    assert not d3RR.has_complex_dims
    assert d3RR._data.shape == (2,2)
    assert d3RR._mask.shape == (2, 2)
    assert str(d3).startswith("RR[[   0.925   --]")
    assert str(d3).endswith("[   0.018    0.020]] Hz")
    assert d3[1, 1].data == d[1,1]

def test_labels_and_sort():
    d0 = NDArray(np.linspace(4000, 1000, 10),
                 labels='a b c d e f g h i j'.split(),
                 units='s',
                 mask=False,
                 title='wavelength')
    assert d0.is_labeled
    d1 = d0._sort()
    assert (d1.data[0] == 1000)
    d0._sort(descend=True, inplace=True)
    assert (d0.data[0] == 4000)
    d1 = d0._sort(by='label', descend=True)
    assert (d1.labels[0] == 'j')

def test_real_imag():
    np.random.seed(12345)
    d = np.random.random((2, 2)) * np.exp(.1j)
    d3 = NDArray(d)
    d3r = d3.real
    d3i = d3.imag
    from spectrochempy.utils.arrayutils import interleave
    new = d3.copy()
    new.data = d3.real.data + 1j * d3.imag.data
    assert_equal( d3.data, new.data)

def test_init_with_complex_information():
    np.random.seed(12345)
    d = np.random.random((4, 3)) * np.exp(.1j)
    d3 = NDArray(d, units=ur.Hz,
                 mask=[[False, True, False],
                       [True, False, False]],
                 is_complex= [True, True]

                 )  # with units & mask
    assert d3.shape == (2, 3)
    assert d3._data.shape == (4, 6)


def test_init_ndarray_subclass():
    # test initialization of an empty ndarray
    # check some of its properties
    a = NDArray()
    assert isinstance(a, NDArray)
    assert a.name != u'<no name>'  # must be the uuid in this case
    assert a.is_empty
    assert not a.is_masked
    assert not a.is_uncertain
    assert a.unitless
    assert a.is_untitled
    assert not a.meta
    assert a.date == datetime(1970, 1, 1, 0, 0)
    a.date = datetime(2005,10,12)
    a.date = "25/12/2025"
    assert a.date == datetime(2025, 12, 25, 0, 0)

def test_set_ndarray_subclass():
    # test of setting some attributes of an empty ndarray
    # check some of its properties
    a = NDArray()
    a.name = 'xxxx'
    assert a.name == u'xxxx'
    a.title = 'yyyy'
    assert not a.is_untitled
    assert a.title == u"yyyy"
    a.meta = []
    a.meta.something = "a_value"
    assert a.meta.something == "a_value"

def test_set_simple_ndarray(ndarray):
    nd = ndarray.copy()
    assert nd.data.size == 100
    assert nd.shape == (10, 10)
    assert nd.size == 100
    assert nd.ndim == 2
    assert not nd.has_complex_dims
    assert nd.data[1,1] == 4.6130673607282127

def test_set_ndarray_with_units(ndarray):
    nd = ndarray.copy()

    assert nd.unitless # ,o units
    assert not nd.dimensionless # no unit so no dimension has no sense

    with catch_warnings() as w:
        nd.to('m')  # should not change anything (but raise a warning)
        assert w[0].category == SpectroChemPyWarning

    assert nd.unitless

    nd.units = 'm'
    assert nd.units == ur.meter

    nd1 = nd.to('km')
    assert nd.units == ur.kilometer
    assert nd1.units == ur.kilometer
    with catch_warnings() as w:
        nd.ito('m')
        assert w[0].category == SpectroChemPyDeprecationWarning
    nd.to('m')
    assert nd.units == ur.meter

    # change of units - ok if it can be casted to the current one
    nd.units = 'cm'

    # cannot change to incompatible units
    with pytest.raises(ValueError):
        nd.units = 'radian'

    # we can force them
    nd.to('radian', force=True)

    assert 1 * nd.units == 1. * ur.dimensionless
    assert nd.units.dimensionless
    assert nd.dimensionless
    with raises(DimensionalityError):
        nd1 = nd1.ito('km/s')  # should raise an error
    nd.units = 'm/km'
    assert nd.units.dimensionless
    assert nd.units.scaling == 0.001

    #with raises(TypeError):
    nd.to(1 * ur.m, force=True)


def test_set_ndarray_with_complex(ndarraycplx):
    nd = ndarraycplx.copy()
    assert nd.data.size == 200
    assert nd.size == 100
    assert nd.data.shape == (10, 20)
    assert nd.shape == (10, 10)  # the real shape
    assert nd.is_complex == [False, True]
    assert nd.ndim == 2
    nd.units = 'meter'
    assert nd.units == ur.meter


def test_copy_of_ndarray(ndarraycplx):
    nd = copy(ndarraycplx)
    assert nd is not ndarraycplx
    assert nd.data.size == 200
    assert nd.size == 100
    assert nd.data.shape == (10, 20)
    assert nd.shape == (10, 10)  # the real shape
    assert nd.is_complex == [False, True]
    assert nd.ndim == 2


def test_deepcopy_of_ndarray(ndarraycplx):
    # for this example there is no diif with copy (write another test for this)
    nd1 = ndarraycplx.copy()
    nd = deepcopy(nd1)
    assert nd is not nd1
    assert nd.data.size == 200
    assert nd.size == 100
    assert nd.data.shape == (10, 20)
    assert nd.shape == (10, 10)  # the real shape
    assert nd.is_complex == [False, True]
    assert nd.ndim == 2


def test_ndarray_with_uncertainty(ndarray):
    nd = ndarray.copy()
    assert not nd.is_uncertain
    assert repr(nd).startswith('NDArray: ')
    nd._uncertainty = np.abs(nd._data * .01)
    nd.to('second', force=True) # force a change of units
    assert nd.is_uncertain
    assert repr(nd).startswith('NDArray: ')
    assert str(nd.values[0,0]) == "4.30+/-0.04 second"


def test_ndarray_with_mask(ndarray):
    nd = ndarray.copy()
    assert not nd.is_masked
    assert repr(nd).startswith('NDArray: ')
    nd.mask[0] = True
    assert nd.is_masked


def test_ndarray_units(ndarray):
    nd = ndarray.copy()
    nd2 = ndarray.copy()
    nd.units = 'm'
    nd2.units = 'km'
    assert nd.is_units_compatible(nd2)
    nd2.to('radian',  force=True)
    assert not nd.is_units_compatible(nd2)


def test_ndarray_with_uncertaincy_and_units(ndarray):
    nd = ndarray.copy()
    nd.to('m', force=True)
    assert nd.units == ur.meter
    assert not nd.is_uncertain
    assert repr(nd).startswith('NDArray: ')
    nd._uncertainty = np.abs(nd._data * .01)
    assert nd.is_uncertain
    assert repr(nd).startswith('NDArray: ')
    units = nd.units
    nd.units = None # should change nothing
    assert nd.units == units
    nd.mask[1,1] = True
    assert nd.is_masked
    nd.mask[1, 2] = True
    print(nd)

def test_ndarray_with_uncertaincy_and_units_being_complex(ndarraycplx):
    nd = ndarraycplx.copy()
    nd.units = 'm'
    assert nd.units == ur.meter
    assert not nd.is_uncertain
    assert repr(nd).startswith('NDArray: ')
    nd._uncertainty = nd._data * .01
    assert nd.is_uncertain
    assert repr(nd).startswith('NDArray: ')
    assert nd._uncertainty.size == nd.data.size


def test_ndarray_len_and_sizes(ndarray, ndarraycplx):
    nd = ndarray.copy()
    assert not nd.has_complex_dims
    assert len(nd) == 10
    assert nd.shape == (10, 10)
    assert nd.size == 100
    assert nd.ndim == 2

    ndc = ndarraycplx.copy()
    assert ndc.has_complex_dims
    assert ndc.is_complex[1]
    assert len(ndc) == 10
    assert ndc.shape == (10, 10)
    assert ndc.size == 100
    assert ndc.ndim == 2


### TEST SLICING

def test_slicing_byindex(ndarray, ndarraycplx):

    nd = ndarray.copy()
    assert not np.any(nd.is_complex) and not nd.is_masked and not nd.is_uncertain
    nd1 = nd[0,0]
    assert_equal(nd1.data, nd.data[0,0])
    nd2 = nd[7:10]
    assert_equal(nd2.data,nd.data[7:10])
    assert not nd.is_masked

    # set item
    nd[1] = 2.
    assert nd[1,0] == 2

    nd.mask[1] = True
    assert nd.is_masked

    ndc = ndarraycplx.copy()
    ndc1 = ndc[1,1]
    assert_equal(ndc1.data, ndc.RR[1,1].data + ndc.RI[1,1].data*1.j)

    ndc.set_complex

### TEST __REPR__

def test_repr(ndarray, ndarrayunit):
    nd = ndarray.copy()
    assert '-1.836,' in nd.__repr__()
    nd = ndarrayunit.copy()
    assert '-1.836,' in nd.__repr__()
    nd.mask[1] = True
    nd.uncertainty = np.abs(nd._data * .1)
    assert nd.is_masked
    assert nd.is_uncertain
    assert '4.296+/-0.430,' in nd.__repr__()

### TEST_COMPARISON

def test_comparison(ndarray, ndarrayunit):
    nd1 = ndarray.copy()
    print(nd1)
    nd2 = ndarrayunit.copy()
    print(nd2)
    assert nd1 != nd2
    assert not nd1==nd2

### TEST ITERATIONS

def test_iteration(ndarrayunit):
    nd = ndarrayunit.copy()
    nd.mask[1] = True
    nd._uncertainty = np.abs(nd._data * .01)
    for item in nd:
        print(item)

#### Squeezing #################################################################
def test_squeeze(ndarrayunit):  # ds2 is defined in conftest

    nd = ndarrayunit.copy()
    assert nd.shape == (10, 10)

    d = nd[..., 0]
    d = d.squeeze()
    assert d.shape == (10, )

    d = nd[0]
    d = d.squeeze()
    assert d.shape == (10, )

    nd.set_complex(-1)
    assert nd.shape == (10, 5)

    d = nd[..., 0]
    d = d.squeeze() # cannot be squeezed in the complex dimension (in reality 2 values)
    assert d.shape == (10, 1)

    d = nd[0]
    d1 = d.squeeze()
    assert d1.shape == (5, )
    assert d1 is not d

    d = nd[...,0:1].RR
    d1 = d.squeeze(inplace=True, axis=-1)
    assert d1.shape == (10, )
    assert d1 is d

    nd.set_complex(-1)
    assert nd.shape == (10, 5)

    nd.set_real(-1)
    assert nd.shape == (10, 10)

    nd.set_complex(0)
    assert nd.shape == (5, 10)

    d = nd[0:1].RR
    d1 = d.squeeze(inplace=True, axis=0)
    assert d1.shape == (10, )
    assert d1 is d

def test_with_units_and_forc_to_change():

    np.random.seed(12345)
    ndd = NDArray(data=np.random.random((3, 3)),
                  mask = [[True, False, False],
                        [False, True, False],
                        [False, False, True]],
                  units = 'meters')

    with raises(Exception):
        ndd.to('second')
    ndd.to('second', force=True)