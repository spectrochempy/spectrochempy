# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT 
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""Tests for the nddataset module

"""
import os
import pandas as pd
import pytest
import numpy as np
from numpy.random import rand

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndcoordset import CoordSet
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.units import ur, Quantity
from spectrochempy.core import info_, debug_, warning_, error_, print_
from spectrochempy import *

from pint.errors import (UndefinedUnitError, DimensionalityError)
from spectrochempy.utils import (MASKED, NOMASK, TYPE_FLOAT, TYPE_INTEGER,
                                 Meta, SpectroChemPyException)
from spectrochempy.utils.testing import (assert_equal, assert_array_equal, raises, RandomSeedContext)

from quaternion import quaternion

typequaternion = np.dtype(np.quaternion)


def test_nddataset_fix_crossvalidate_bug():
    assert hasattr(NDDataset(), '_%s_validate' % '_iscopy') is False


# ======================================================================================================================
# Init
# ======================================================================================================================

def test_nddataset_init_nothing():
    # init with nothing

    nd0 = NDDataset()
    assert 'empty' in str(nd0)
    assert not nd0.dims
    assert nd0.shape == ()

def test_nddataset_init_simple_list():
    # init with a simple list

    inp = [1, 2, 3]
    nd1 = NDDataset(inp)
    info_(nd1)
    info_(repr(nd1))
    assert (np.array(inp) == nd1.data).all()
    assert nd1.dims == ['x']
    assert nd1.ndim == 1
    assert nd1.size == 3
    assert nd1.dtype in TYPE_INTEGER
    assert not nd1.is_complex
    assert nd1.coords is None  # no coordinates
    assert nd1.x is None  # no coordinates
    with pytest.raises(AttributeError):
        y = nd1.y

    # masking

    nd1[1] = MASKED
    assert nd1.is_masked
    assert "[int64]" in str(nd1) or "[int32]" in str(nd1)
    info_(nd1)


def test_nddataset_dims_attribute_at_creation():
    inp = [1, 2, 3]
    nd1 = NDDataset(inp, dims='q' )
    assert nd1.dims == ['q']

    ndd1 = NDDataset(np.ones((5, 3)), dims=['u', 'w'])
    assert ndd1.ndim == 2
    assert ndd1.dims == ['u', 'w']
    
    
def test_nddataset_init_ndarray():
    # init with a ndarray

    ndd1 = NDDataset(np.ones((5, 3)), title='toto')
    assert ndd1.ndim == 2
    assert ndd1.dims == ['y', 'x']

def test_nddataset_init_quantity():
    # initializing with a quantity

    d = 5
    units = ur.meter
    ndd = NDDataset(5 * units)
    assert ndd.units == units
    assert ndd.values == 5 * units
    info_(ndd)

    d = np.ones((5, 3))
    ndd = NDDataset(d * units)
    assert ndd.units == units
    assert_array_equal(ndd.data, d)
    info_(ndd)

    # Should raise an error if unit is not known
    with pytest.raises(UndefinedUnitError):
        NDDataset([1, 2, 3], units=ur.toto)


def test_nddataset_init_using_dataset(nd1d, nd2d, ds1):
    
    assert nd1d.data.shape == (10,)
    assert nd1d.data.size == 10
    assert nd1d.ndim == 1
    assert nd1d.dtype in TYPE_FLOAT
    assert nd1d.dims == ['x']
    info_(nd1d)

    # init from another dataset

    nd2 = NDDataset(nd1d, copy=False)
    assert nd2.data is nd1d.data

    # check copy flag
    nd3 = NDDataset(nd2d, copy=True)
    assert nd3.data is not nd2d.data

    # check no coordinates
    assert nd3.coords is None
    assert nd3.x is None
    assert nd3.y is None
    with pytest.raises(AttributeError):
        z = nd3.z

    # attribute of another dataset are properly passed
    nd = NDDataset(ds1)
    assert nd.id != ds1.id
    assert nd.title == ds1.title
    assert nd.coords == ds1.coords
    assert nd.mask == ds1.mask
    assert nd.units == ds1.units
    assert nd.meta == ds1.meta
    assert nd is not ds1
    assert nd == ds1


# ----------------------------------------------------------------------------------------------------------------------
# Coordinates
# ----------------------------------------------------------------------------------------------------------------------

def test_nddataset_init_with_coordinates():
    
    # intializing with coordinates
    ndd3 = NDDataset(np.array([1, 2, 3]), coords=([0, 1, 2],))
    info_(ndd3)


def test_nddataset_coords():
    
    # init coordinates at NDDataset initialization

    dx = np.random.random((10, 7, 3))
    coord0 = np.arange(10)
    coord1 = np.arange(7)
    coord2 = np.arange(3) * 100.
    
    da = NDDataset(dx,
                   coords=(coord0, coord1, coord2),
                   title='absorbance',
                   coordtitles=['wavelength', 'time-on-stream', 'temperature'],
                   coordunits=['cm^-1', 's', 'K'],
                   )

    assert da.shape == (10, 7, 3)
    assert da.coords.titles == ['temperature', 'time-on-stream', 'wavelength']
    assert da.coords.names == ['x', 'y', 'z']
    assert da.coords.units == [ur.Unit('K'), ur.Unit('s') ,ur.Unit('cm^-1')]

    # order of dims follow data shape, but not necessarily the coord list (which is ordered by name)
    assert da.dims == ['z', 'y', 'x']
    assert da.coords.names == sorted(da.dims)

    info_(da)

    # transpose

    dat = da.T
    assert dat.dims == ['x', 'y', 'z']
    info_(dat)

    # dims changed but did not change coords order !
    assert dat.coords.names == sorted(dat.dims)
    assert dat.coords.titles == da.coords.titles
    assert dat.coords.units == da.coords.units
    
    # too many coordinates

    cadd = Coord(labels=['d%d' % i for i in range(6)])
    coordtitles = ['wavelength', 'time-on-stream', 'temperature']
    coordunits = ['cm^-1', 's', None]
    daa = NDDataset(dx,
                    coords=[coord0, coord1, coord2, cadd, coord2.copy()],
                    title='absorbance',
                    coordtitles=coordtitles,
                    coordunits=coordunits,
                    )
    assert daa.coords.titles == coordtitles[::-1]
    assert daa.dims == ['z', 'y', 'x']
    info_(daa)

    # with a CoordSet
    c0, c1 = Coord(labels=['d%d' % i for i in range(6)]), \
             Coord(data=[1, 2, 3, 4, 5, 6])
    cc = CoordSet(c0, c1)
    print_(cc)
    cd = CoordSet(x=cc, y=c1)
    print_(cd)
    
    ds = NDDataset([1, 2, 3, 6, 8, 0],
                   coords=cd,
                   units='m')
    assert ds.dims == ['x']
    assert ds.x == cc
    ds.history = 'essai: 1'
    ds.history = 'try:2'
    print_(ds)
    
    # wrong type
    with pytest.raises(TypeError):
        c = ds.coord[1.3]

    # extra coordinates
    with pytest.raises(AttributeError):
        c = ds.y

    #invalid_length
    coord1 = Coord(np.arange(9), title='wavelengths')  # , units='m')
    coord2 = Coord(np.arange(20), title='time')  # , units='s')
    with pytest.raises(ValueError):
        ndd1 = NDDataset(np.random.random((10, 20)), coords=(coord1, coord2))


# ----------------------------------------------------------------------------------------------------------------------
# indexing
# ----------------------------------------------------------------------------------------------------------------------

def test_nddataset_coords_indexer():
    dx = np.random.random((10, 100, 10))
    coord0 = np.linspace(4000, 1000, 10)
    coord1 = np.linspace(0, 60, 10)  # wrong length
    coord2 = np.linspace(20, 30, 10)

    with pytest.raises(ValueError): # wrong lenth
        da = NDDataset(dx,
                       coords=[coord0, coord1, coord2],
                       title='absorbance',
                       coordtitles=['wavelength', 'time-on-stream', 'temperature'],
                       coordunits=['cm^-1', 's', 'K'],
                       )

    coord1 = np.linspace(0, 60, 100)
    da = NDDataset(dx,
                   coords=[coord0, coord1, coord2],
                   title='absorbance',
                   coordtitles=['wavelength', 'time-on-stream', 'temperature'],
                   coordunits=['cm^-1', 's', 'K'],
                   )

    assert da.shape == (10, 100, 10)
    coords = da.coords
    assert len(coords) == 3

    assert_array_equal(da.coords[2].data, coord0, "get axis by index failed")
    assert_array_equal(da.coords['wavelength'].data, coord0, "get axis by title failed")
    assert_array_equal(da.coords['time-on-stream'].data, coord1, "get axis by title failed")
    assert_array_equal(da.coords['temperature'].data, coord2)

    da.coords['temperature'] += 273.15 *ur.K
    assert_array_equal(da.coords['temperature'].data, coord2+273.15)


# Methods

def test_nddataset_str():
    arr1d = NDDataset([1, 2, 3])
    print(arr1d)
    info_(arr1d)
    assert '[int64]' in str(arr1d) or '[int32]' in str(arr1d)
    arr2d = NDDataset(np.array([[1, 2], [3, 4]]))
    assert str(arr2d) == 'NDDataset: [int64] unitless (shape: (y:2, x:2))' or str(arr2d) == 'NDDataset: [int32] unitless (shape: (y:2, x:2))'


def test_nddataset_str_repr(ds1):
    arr1d = NDDataset(np.array([1, 2, 3]))
    assert repr(arr1d).startswith('NDDataset')
    arr2d = NDDataset(np.array([[1, 2], [3, 4]]))
    assert repr(arr2d).startswith('NDDataset')
    print(repr(ds1))
    info_(repr(ds1))
    print(ds1)
    info_(ds1)
    print_(ds1)


def test_nddataset_mask_valid():
    NDDataset(np.random.random((10, 10)),
              mask=np.random.random((10, 10)) > 0.5)


def test_nddataset_copy_ref():
    """
    Tests to ensure that creating a new NDDataset object copies by *reference*.
    """
    a = np.ones((10, 10))
    nd_ref = NDDataset(a)
    a[0, 0] = 0
    assert nd_ref.data[0, 0] == 0


def test_nddataset_conversion():
    nd = NDDataset(np.array([[1, 2, 3], [4, 5, 6]]))
    assert nd.data.size == 6
    assert nd.data.dtype == np.dtype(int)


def test_nddataset_invalid_units():
    with pytest.raises(UndefinedUnitError):
        d = NDDataset(np.ones((5, 5)), units="NotAValidUnit")


def test_nddataset_units(nd1d):
    nd = nd1d.copy()
    nd = np.fabs(nd)
    nd.units = 'm'
    nd2 = np.sqrt(nd)
    assert isinstance(nd2, type(nd))
    assert nd2.data[1] == np.sqrt(nd.data[1])
    assert nd2.units == ur.m ** .5

    nd.units = 'cm'
    nd2 = np.sqrt(nd)
    nd.ito('m')
    nd2 = np.sqrt(nd)
    assert isinstance(nd2, type(nd))
    assert nd2.data[1] == np.sqrt(nd.data[1])
    assert nd2.units == ur.m ** .5


def test_nddataset_masked_array_input():
    a = np.random.randn(100)
    marr = np.ma.masked_where(a > 0, a)

    nd = NDDataset(marr)

    # check that masks and data match
    assert_array_equal(nd.mask, marr.mask)
    assert_array_equal(nd.data, marr.data)

    # check that they are both by reference
    marr.mask[10] = ~marr.mask[10]
    marr.data[11] = 123456789

    assert_array_equal(nd.mask, marr.mask)
    assert_array_equal(nd.data, marr.data)


def test_nddataset_swapaxes(nd1d, nd2d, ref_ds, ds1):
    nd1 = nd1d.copy()
    nd2 = nd2d.copy()
    nd3 = ds1.copy()

    # swapaxes needs 2D at least
    info_("swap 1D - should do nothing")
    assert nd1.shape == (10,)
    nd1s = nd1.swapaxes(1, 0)
    assert_equal(nd1s.data, nd1.data)

    info_("swap 2D no coords \n" + str(nd2))
    nd2s = nd2.swapaxes(1, 0)
    assert nd2s.dims == nd2.dims[::-1]

    info_("swap 2D \n", nd3)
    assert nd3.shape == ref_ds.shape
    nd3s = nd3.swapaxes(1, 0)
    ref = ref_ds
    refs = np.swapaxes(ref, 1, 0)
    assert nd3.shape == ref.shape  # original unchanged
    assert nd3s.shape == refs.shape
    assert nd3s is not nd3
    assert nd3s.dims[:2] == nd3.dims[:2][::-1]

    info_("swap 2D inplace")
    nd3s = nd3.swapaxes(1, 0, inplace=True)
    assert nd3.shape == refs.shape  # original changed
    assert nd3s is nd3  # objects should be the same
    info_(nd3s)

    # use of the numpy method
    info_("swap 2D using the numpy method")
    nd3s = np.swapaxes(nd3, 1, 0)
    assert nd3.shape == refs.shape  # original unchanged (but was already swapped)
    assert nd3s.shape == ref.shape
    assert nd3s is not nd3
    info_(nd3s)

    # TODO: add check for swapaxes of all elements of a dataset such as meta


################################### TEST SLICING################################

def test_nddataset_loc2index(ref_ds, ds1):
    da = ds1
    ref = ref_ds

    info_(da)
    assert da.shape == ref.shape
    coords = da.coords
    assert len(coords) == 3

    assert da._loc2index(3990.0, dim=0) == 0
    assert da._loc2index('j', dim=0) == 9
    with pytest.raises(IndexError):
        da._loc2index('z', 0)  # labels doesn't exist
    da._loc2index(5000, 0)
    assert da._loc2index(5000, 0) == (0, 'out_of_limits')  # return the low limit
    assert da._loc2index(0, 0) == (9, 'out_of_limits')  # return the high limit index


def test_nddataset_slicing_by_index(ref_ds, ds1):
    da = ds1
    ref = ref_ds
    # case where the index is an integer: the selection is by index starting at zero
    assert da.shape == ref.shape
    coords = da.coords

    plane0 = da[0]
    # should return a dataset of with dimension x of size 1
    assert type(plane0) == type(da)
    assert plane0.ndim == 3
    assert plane0.shape == (1, 100, 3)
    assert plane0.size == 300
    assert plane0.dims == ['z', 'y', 'x']
    assert_array_equal(plane0.z, da.z[0])

    da1 = plane0.squeeze()
    assert da1.shape == (100, 3)
    assert da1.dims == ['y', 'x']

    plane1 = da[:, 0]
    assert type(plane1) == type(da)
    assert plane1.ndim == 3
    assert plane1.shape == (10, 1, 3)
    assert plane1.size == 30
    assert plane1.dims == ['z', 'y', 'x']

    da2 = plane1.squeeze()
    assert da2.dims == ['z', 'x']
    assert da2.z == coords[-1]   # remember coordinates are ordered by name!
    assert da2.x == coords[0]

    # another selection
    row0 = plane0[:, 0]
    assert type(row0) == type(da)
    assert row0.shape == (1, 1, 3)

    # and again selection
    element = row0[..., 0]
    assert type(element) == type(da)
    assert element.dims == ['z', 'y', 'x']
    info_("element : \n", element)

    # squeeze
    row1 = row0.squeeze()
    assert row1.mask == NOMASK
    row1[0] = MASKED
    assert row1.dims == ['x']
    assert row1.shape == (3,)
    assert row1.mask.shape == (3,)
    info_("row1 : \n", row1)
    element = row1[..., 0]
    assert element.x == coords[0][0]

    # now a slicing in multi direction
    matrix = da[1:4, 10:40:2, :2]
    info_("matrix : %s" % matrix)

    # now again a slicing in multi direction (full selection)
    matrix = da[:, :, -2]
    info_("matrix : %s" % matrix)

    # now again a slicing in multi direction (ellipsis)
    matrix = da[..., -1]
    info_("matrix : %s" % matrix)

    matrix = da[-1, ...]
    info_("matrix : %s" % matrix)


def test_nddataset_slicing_by_label(ref_ds, ds1):
    da = ds1
    ref = ref_ds
    print_(da)
    
    # selection
    planeb = da['b']
    assert type(planeb) == type(da)
    plane1 = da[1]
    assert_equal(planeb.data, plane1.data)
    assert planeb.ndim == 3
    assert planeb.size == 300
    bd = da['b':'f']  # the last index is included
    assert bd.shape == (5, 100, 3)
    info_(bd)
    b1 = da[1:6]
    assert_equal(bd.data, b1.data)
    bc = da['b':'f', :, "hot"]
    assert bc.shape == (5, 100, 1)
    assert bc.z.labels[0] == 'b'

    hot = da[..., "hot"]
    print_(hot)

    # TODO: find a way to use such syntax
    # hot2 = da["x.hot"]
    # assert hot == hot2


def test_nddataset_slicing_by_values(ref_ds, ds1):
    da = ds1
    ref = ref_ds

    x = da[3000.]
    info_(x)
    assert x.shape == (1, 100, 3)

    y = da[3000.0:2000.0, :, 210.]
    info_(y)
    assert y.shape == (4, 100, 1)

    # slicing by values should also work using reverse order
    yr = da[2000.0:3000.0, :, 210.]


def test_nddataset_slicing_out_limits(caplog, ref_ds, ds1):
    import logging
    logger = logging.getLogger('SpectroChemPy')
    logger.propagate = True
    caplog.set_level(logging.DEBUG)

    da = ds1
    ref = ref_ds
    y1 = da[2000.]
    assert str(y1) == 'NDDataset: [float64] a.u. (shape: (z:1, y:100, x:3))'

    y2 = da[2000]
    assert y2 is None  # as we are out of limits
    assert caplog.records[-1].levelname == 'ERROR'
    assert caplog.records[-1].message.startswith('Empty array of shape (0, 100, 3) resulted from slicing.')

    y3 = da[:, 95:105]
    assert str(y3) == 'NDDataset: [float64] a.u. (shape: (z:10, y:5, x:3))'

    info_(da)
    y4 = da[5000.:4001.]
    assert y2 is None  # as we are out of limits
    assert caplog.records[-1].levelname == 'ERROR'
    assert caplog.records[-1].message.startswith('Empty array of shape (0, 100, 3) resulted from slicing.')

    y5 = da[5000.:3000.]
    assert str(y5) == 'NDDataset: [float64] a.u. (shape: (z:4, y:100, x:3))'


@raises(IndexError)
def test_nddataset_slicing_toomanyindex(ref_ds, ds1):
    da = ds1
    ref = ref_ds
    y = da[:, 3000.:2000., :, 210.]


def test_nddataset_slicing_by_index_nocoords(ref_ds, ds1):
    da = ds1
    ref = ref_ds
    # case where the index is an integer:
    # the selection is by index starting at zero

    da.delete_coords()  # clear coords
    plane0 = da[1]
    assert type(plane0) == type(da)  # should return a dataset
    assert plane0.ndim == 3
    assert plane0.size == 300
    info_("Plane0 : %s" % plane0)


def test_nddataset_slicing_by_location_but_nocoords(ref_ds, ds1):
    da = ds1
    ref = ref_ds
    # case where the index is an integer:
    # the selection is by index starting at zero

    da.delete_coords()  # clear coords

    # this cannot work (no coords for location)
    with pytest.raises(SpectroChemPyException):
        plane1 = da[3666.7]


# slicing tests
def test_nddataset_simple_slicing():
    d1 = NDDataset(np.ones((5, 5)))
    assert d1.data.shape == (5, 5)
    assert d1.shape == (5, 5)

    d2 = d1[2:3, 2:3]
    assert d2.shape == (1, 1)
    assert (d1 is not d2)
    d3 = d1[2, 2]
    assert d3.shape == (1, 1)
    assert d3.squeeze().shape == ()

    d3 = d1[0]
    assert d3.shape == (1, 5)


def test_nddataset_slicing_with_mask():
    mask = np.zeros((5, 5)).astype(bool)
    mask[1, 1] = True
    d1 = NDDataset(np.ones((5, 5)), mask=mask)
    assert d1[1].shape == (1, 5)
    assert d1[1, 1].mask


def test_nddataset_slicing_with_mask_units():
    u1 = np.ones((5, 5)) * 3
    mask = np.zeros((5, 5)).astype(bool)
    mask[1, 1] = True
    d1 = NDDataset(np.ones((5, 5)), mask=mask, units='m')
    assert d1[0].shape == (1, 5)


def test_nddataset_slicing_with_coords(ref_ds, ds1):
    da = ds1.copy()
    ref = ref_ds.copy()

    da00 = da[0, 0]
    assert da00.shape == (1, 1, 3)
    assert da00.coords['x'] == da00.coords[0]
    assert da00.coords['x'] == da.coords[0]


def test_nddataset_mask_array_input():
    marr = np.ma.array([1., 2., 5.])  # Masked array with no masked entries
    nd = NDDataset(marr)
    assert not nd.is_masked

    marr = np.ma.array([1., 2., 5.], mask=[True, False, False])  # Masked array
    nd = NDDataset(marr)
    assert nd.is_masked


def test_nddataset_unmasked_in_operation_with_masked_numpy_array():
    ndd = NDDataset(np.array([1, 2, 3]))
    np_data = -np.ones_like(ndd)
    np_mask = np.array([True, False, True])
    np_arr_masked = np.ma.array(np_data, mask=np_mask)

    result1 = ndd * np_arr_masked
    assert result1.is_masked
    assert np.all(result1.mask == np_mask)
    # TODO: IndexError: in the future, 0-d boolean arrays will be interpreted as a valid boolean index
    # assert np.all(result1[~result1.mask].data == -ndd.data[~np_mask])

    result2 = np_arr_masked * ndd
    # Numpy masked  array return a masked array in this case
    # assert result2.is_masked
    assert np.all(result2.mask == np_mask)
    # assert np.all(result2[~result2.mask].data == -ndd.data[~np_mask])


@pytest.mark.parametrize('shape', [(10,), (5, 5), (3, 10, 10)])
def test_nddataset_mask_invalid_shape(shape):
    with pytest.raises(ValueError) as exc:
        with RandomSeedContext(789):
            NDDataset(np.random.random((10, 10)),
                      mask=np.random.random(shape) > 0.5)
    assert exc.value.args[0] == 'mask {} and data (10, 10) shape mismatch!'.format(shape)


@pytest.mark.parametrize('mask_in', [
    np.array([True, False]),
    np.array([1, 0]),
    [True, False],
    [1, 0]])
def test_nddataset_mask_init_without_np_array(mask_in):
    ndd = NDDataset(np.array([1, 1]), mask=mask_in)
    assert (ndd.mask == mask_in).all()


def test_nddataset_with_mask_acts_like_masked_array():
    # test for #2414
    input_mask = np.array([True, False, False])
    input_data = np.array([1., 2., 3.])
    ndd_masked = NDDataset(input_data.copy(), mask=input_mask.copy())
 #   ndd_masked = np.sqrt(ndd_masked)
    other = - np.ones_like(input_data)
    
    result1 = np.multiply(ndd_masked,other)
    result2 = ndd_masked * other
    result3 = other * ndd_masked
    result4 = other / ndd_masked
    # Test for both orders of multiplication
    for result in [result1, result2, result3, result4]:
        assert result.is_masked
        # Result mask should match input mask because other has no mask
        assert np.all(result.mask == input_mask)
        # TODO:IndexError: in the future, 0-d boolean arrays will be interpreted as a valid boolean index
        # assert np.all(result[~result.mask].data == - input_data[~input_mask])


def test_nddataset_creationdate():
    ndd = NDDataset([1., 2., 3.])
    ndd2 = np.sqrt(ndd)
    info_(ndd2)
    assert ndd2._date is not None


def test_nddataset_title():
    ndd = NDDataset([1., 2., 3.], title='xxxx')
    assert ndd.title == 'xxxx'

    ndd2 = NDDataset(ndd, title='yyyy')
    assert ndd2.title == 'yyyy'

    ndd2.title = 'zzzz'
    assert ndd2.title == 'zzzz'


def test_nddataset_real_imag():
    na = np.array(
        [[1. + 2.j, 2. + 0j],
         [1.3 + 2.j, 2. + 0.5j],
         [1. + 4.2j, 2. + 3j]])

    nd = NDDataset(na)

    # in the last dimension
    assert_array_equal(nd.real, na.real)
    assert_array_equal(nd.imag, na.imag)


def test_nddataset_comparison():
    ndd = NDDataset([1., 2. + 1j, 3.])

    val = ndd * 1.2 - 10.
    val = np.abs(val)
    info_(val)
    info_(val >= 7.)

    assert np.all(val >= 6.)
    info_(str('ok = behave like a bool'))


def test_nddataset_repr_html():
    nd = NDDataset.read_omnic(os.path.join('irdata', 'nh4y-activation.spg'))
    x = nd.T._repr_html_()

    dx = np.random.random((10, 100, 3))

    coord0 = Coord(data=np.linspace(4000., 1000., 10),
                   labels='a b c d e f g h i j'.split(),
                   mask=None,
                   units="cm^-1",
                   title='wavelength')

    coord1 = Coord(data=np.linspace(0., 60., 100),
                   labels=None,
                   mask=None,
                   units="s",
                   title='time-on-stream')

    coord2 = Coord(data=np.linspace(200., 300., 3),
                   labels=['cold', 'normal', 'hot'],
                   mask=None,
                   units="K",
                   title='temperature')

    da = NDDataset(dx,
                   coords=[coord0, coord1, coord2],
                   title='absorbance',
                   units='absorbance'
                   )

    info_(da)
    a = da._repr_html_()
    info_(a)
    # assert "<strong>Units</strong></td><td style='text-align:left'>absorbance</td>" in a


#### Metadata ################################################################
def test_nddataset_with_meta(ds1):
    da = ds1.copy()

    meta = Meta()
    meta.essai = ['try_metadata', 10]
    da.meta = meta

    # check copy of meta
    dac = da.copy()
    assert dac.meta == da.meta


#### sorting #################################################################
def test_nddataset_sorting(ds1):  # ds1 is defined in conftest

    dataset = ds1[:3, :3, 0].copy()
    info_(dataset)
    dataset.sort(inplace=True, dim='z')
    labels = np.array(list('abc'))
    assert_array_equal(dataset.coords['z'].labels, labels)
    # nochange because the  axis is naturally iversed to force it
    # we need to specify descend
    
    dataset.sort(inplace=True, descend=False, dim='z')  # order value in increasing order
    info_(dataset)
    labels = np.array(list('cba'))
    assert_array_equal(dataset.coords['z'].labels, labels)

    dataset.sort(inplace=True, dim='z')
    info_(dataset)
    new = dataset.copy()
    new = new.sort(descend=False, inplace=False, dim='z')
    info_(new)
    assert_array_equal(new.data, dataset.data[::-1])
    assert (new[0, 0] == dataset[-1, 0])
    assert_array_equal(new.coords['z'].labels, labels)
    assert_array_equal(new.coords["z"].data, dataset.coords['z'].data[::-1])

    # check for another dimension

    dataset = ds1.copy()
    new = ds1.copy()
    new.sort(dim='y', inplace=True, descend=False)
    assert_array_equal(new.data, dataset.data)
    assert (new[0, 0, 0] == dataset[0, 0, 0])

    new = dataset.copy()
    new.sort(dim='y', inplace=True, descend=True)
    assert_array_equal(new.data, dataset.data[:, ::-1, :])
    assert (new[0, -1, 0] == dataset[0, 0, 0])


#### multiple axis #############################################################

def test_nddataset_multiple_axis(ref_ds, coord0, coord1, coord2, coord2b, dsm):  # dsm is defined in conftest

    ref = ref_ds
    da = dsm.copy()
    coordm = CoordSet(coord2, coord2b)

    info_(da)

    # check indexing
    assert da.shape == ref.shape
    coords = da.coords
    assert len(coords) == 3

    assert coords['z'] == coord0
    assert da.z == coord0

    assert da.coords['wavenumber'] == coord0
    assert da.wavenumber == coord0
    assert da['wavenumber'] == coord0
    
    # for multiple coordinates
    assert da.coords['x'] == coordm
    assert da['x'] == coordm
    assert da.x == coordm
    
    # but we can also specify, which axis should be returned explicitely
    # by an index or a label
    assert da.coords['x_1'] == coord2b
    assert da.coords['x_2'] == coord2
    assert da.coords['x'][1] == coord2
    assert da.coords['x']._1 == coord2b
    assert da.x['_1'] == coord2b
    assert da['x_1'] == coord2b
    assert da.x_1 == coord2b
    
    
    x = da.coords['x']
    assert x['temperature'] == coord2
    assert da.coords['x']['temperature'] == coord2
    
    # even simpler we can specify any of the axis title and get it ...
    assert da.coords['time-on-stream'] == coord1

    assert da.coords['temperature'] == coord2

    da.coords['magnetic field'] += 100 * ur.millitesla

    assert da.coords['magnetic field'] == coord2b + 100 * ur.millitesla


def test_nddataset_coords_manipulation(IR_dataset_2D):
    dataset = IR_dataset_2D.copy()
    coord0 = dataset.coords['y']
    coord0 -= coord0[0] # remove first element
    info_(coord0)

def test_nddataset_square_dataset_with_identical_coordinates():
    
    a = np.random.rand(3,3)
    c = Coord(np.arange(3)*.25, title='time', units='us')
    nd = NDDataset(a, coords=CoordSet(x=c, y='x'))
    info_(nd)
    assert nd.x == nd.y
    
#### Test masks ######

def test_nddataset_use_of_mask(IR_dataset_1D):
    nd = IR_dataset_1D.copy()
    info_(nd)
    nd[950.:1260.] = MASKED
    info_(nd)


# ----------------------------------------------------------------------------------------------------------------------
# additional tests made following some bug fixes
# ----------------------------------------------------------------------------------------------------------------------


def test_nddataset_repr_html_bug_undesired_display_complex():
    da = NDDataset([1, 2, 3])
    da.title = 'intensity'
    da.description = 'Some experimental measurements'
    da.units = 'dimensionless'
    assert "(complex)" not in da._repr_html_()
    pass


def test_nddataset_bug_fixe_figopeninnotebookwithoutplot():
    da = NDDataset([1, 2, 3])
    da2 = np.sqrt(da ** 3)
    assert da2._fig is None  # no figure should open


def test_nddataset_max_min_with_1D_real(IR_dataset_1D):
    # test on a 1D NDDataset
    nd1 = IR_dataset_1D
    nd1[1] = True
    assert nd1.is_masked
    info_(nd1)
    assert "[float32]" in str(nd1)
    mx = nd1.max()
    assert mx == Quantity(6.0, 'absorbance')
    mx = nd1.max(keepdims=1)
    assert isinstance(mx, NDDataset)
    #assert mx.data == pytest.approx(6.0)


def test_nddataset_max_with_2D_real(IR_dataset_2D):
    # test on a 2D NDDataset
    nd2 = IR_dataset_2D
    nd2 = nd2[:, 4000.:1300.]
    ndmt = nd2.min()  # no axis specified
    info_(ndmt)
    nd2m = nd2.max('y')  # axis selected
    info_(nd2m)
    nd2m2 = nd2.min('x')  # axis selected
    info_(nd2m2)
    pass


def test_nddataset_fancy_indexing():
    # numpy vs dataset
    rand = np.random.RandomState(42)
    x = rand.randint(100, size=10)

    # single value indexing
    info_(x[3], x[7], x[2])
    dx = NDDataset(x)
    assert (dx[3].data, dx[7].data, dx[2].data) == (x[3], x[7], x[2])

    # slice indexing
    info_(x[3:7])
    assert_array_equal(dx[3:7].data, x[3:7])

    # boolean indexing
    info_(x[x > 52])
    assert_array_equal(dx[x > 52].data, x[x > 52])

    # fancy indexing
    ind = [3, 7, 4]
    info_(x[ind])
    assert_array_equal(dx[ind].data, x[ind])

    ind = np.array([[3, 7], [4, 5]])
    info_(x[ind])
    assert_array_equal(dx[ind].data, x[ind])

    with RandomSeedContext(1234):
        a = np.random.random((3, 5)).round(1)
    c = (np.arange(3), np.arange(5))
    nd = NDDataset(a, coords=c)
    info_(nd)
    a = nd[[1, 0, 2]]
    info_(a)
    a = nd[np.array([1, 0])]
    info_(a)


def test_nddataset_extrema():
    with RandomSeedContext(1234):
        a = np.random.random((3, 5)).round(1)
    c = (np.arange(3)*10.0 * ur.s, np.arange(5)*7.0 * ur.kg)
    nd = NDDataset(a, coords=c, units='m')
    info_(nd)
    
    mi = nd.min()
    assert mi== Quantity(0.2, 'meter')
    
    ma = nd.max()
    assert ma== Quantity(1.0, 'meter')
    
    ma = np.max(nd, keepdims=True)
    assert isinstance(ma, NDDataset)
    assert ma.shape == (1,1)
    assert ma.x.data == np.array([21])
    assert ma.y.data == np.array([10])

    mi = nd.min(keepdims=True)
    assert isinstance(mi, NDDataset)

    info_('_____________________________')
    mi1 = nd.min(dim='y')
    info_('minimum', mi1)
    # ma1 = nd.max('x')
    # info_('X :', ma1)
    # ma2 = nd.max('y')
    # info_('Y :', ma2)


def test_nddataset_bug_par_arnaud():
    import spectrochempy as scp
    import numpy as np

    x = scp.Coord(data=np.linspace(1000., 4000., num=6000), title='x')
    y = scp.Coord(data=np.linspace(0., 10, num=5), title='y')

    data = np.random.rand(x.size, y.size)

    ds = scp.NDDataset(data, coords=[x, y])

    ds2 = ds[2000.0:3200.0, :]

    assert ds2.coords.y.data.shape[0] == 2400, 'taille axe 0 doit être 2400'
    assert ds2.data.shape[0] == 2400, "taille dimension 0 doit être 2400"

    info_('taille axe 0 : ' + str(ds2.coords.y.data.shape[0]))
    info_('taille dimension 0:' + str(ds2.data.shape[0]))


def test_nddataset_bug_13(IR_dataset_1D):
    nd = IR_dataset_1D
    info_(nd)
    info_(type(nd.x).__name__)
    info_('\n coord %s' % nd.x)

    nd.x[0] = 0  # here we assume that the value have the same units
    info_('\nnew coord\n%s' % nd.x)

    # reproduce our bug  (now fixed)
    nd.x[0] = Quantity('0 m^-1')
    info_('\nnew coord\n%s' % nd.x)

    v = Quantity('0.2 m')
    nd.x[0] = v
    info_('\nnew coord\n%s' % nd.x)

    with pytest.raises(DimensionalityError):
        v = Quantity('0.1 s')
        nd.x[0] = v


################# Complex and Quaternion, and NMR ##################

def test_nddataset_create_from_complex_data():
    # 1D (complex)
    nd = NDDataset([1. + 2.j, 2. + 0j])
    assert nd.data.size == 2
    assert nd.size == 2
    assert nd.data.shape == (2,)
    assert nd.shape == (2,)

    # 2D (complex in the last dimension - automatic detection)
    nd = NDDataset([[1. + 2.j, 2. + 0j],
                    [1.3 + 2.j, 2. + 0.5j],
                    [1. + 4.2j, 2. + 3j]
                    ])

    assert nd.data.size == 6
    assert nd.size == 6
    assert nd.data.shape == (3, 2)
    assert nd.shape == (3, 2)

    # 2D quaternion
    nd = NDDataset([[1., 2.],
                    [1.3, 2.],
                    [1., 2.],
                    [1., 2.],
                    ], dtype=typequaternion)

    assert nd.data.size == 2
    assert nd.size == 2
    assert nd.data.shape == (2, 1)
    assert nd.shape == (2, 1)

    # take real part
    ndr = nd.real
    assert ndr.shape == (2, 1)
    assert not ndr.is_quaternion


def test_nddataset_set_complex_1D_during_math_op():
    nd = NDDataset([1., 2.], coords=[Coord([10, 20])], units='meter')
    assert nd.data.size == 2
    assert nd.size == 2
    assert nd.shape == (2,)
    assert nd.units == ur.meter
    assert not nd.is_complex

    ndj = nd * 1j
    assert ndj.data.size == 2
    assert ndj.is_complex


def test_nddataset_create_from_complex_data_with_units():
    # 1D
    nd = NDDataset([1. + 2.j, 2. + 0j])
    assert nd.data.size == 2
    assert nd.size == 2
    assert nd.data.shape == (2,)
    assert nd.shape == (2,)

    # add units
    nd.units = 'm**-1'
    nd.ito('cm^-1')

    # 2D
    nd2 = NDDataset(
        [[1. + 2.j, 2. + 0j], [1.3 + 2.j, 2. + 0.5j], [1. + 4.2j, 2. + 3j]])

    assert nd2.data.size == 6
    assert nd2.size == 6
    assert nd2.data.shape == (3, 2)
    assert nd2.shape == (3, 2)

    # add units
    nd2.units = 'm**-1'
    nd2.ito('cm^-1')


def test_nddataset_real_imag_quaternion():
    na = np.array(
        [[1. + 2.j, 2. + 0j, 1.3 + 2.j],
         [2. + 0.5j, 1. + 4.2j, 2. + 3j]])

    nd = NDDataset(na)

    # in the last dimension
    assert_array_equal(nd.real, na.real)
    assert_array_equal(nd.imag, na.imag)

    # in another dimension
    nd.set_quaternion(inplace=True)
    assert nd.is_quaternion
    assert nd.shape == (1, 3)

    na = np.array(
        [[1. + 2.j, 2. + 0j],
         [1.3 + 2.j, 2. + 0.5j],
         [1. + 4.2j, 2. + 3j],
         [5. + 4.2j, 2. + 3j]])

    nd = NDDataset(na)
    nd.set_quaternion(inplace=True)
    assert nd.is_quaternion

    assert_array_equal(nd.real.data, na[::2, :].real)

    nb = np.array(
        [[0. + 2.j, 0. + 0j],
         [1.3 + 2.j, 2. + 0.5j],
         [0. + 4.2j, 0. + 3j],
         [5. + 4.2j, 2. + 3j]])
    ndj = NDDataset(nb, dtype=quaternion)

    assert nd.imag == ndj


def test_nddataset_quaternion():
    na0 = np.array([[1., 2., 2., 0., 0., 0.],
                    [1.3, 2., 2., 0.5, 1., 1.],
                    [1, 4.2, 2., 3., 2., 2.],
                    [5., 4.2, 2., 3., 3., 3.]])

    nd = NDDataset(na0)
    assert nd.shape == (4, 6)
    nd.dims = ['v','u']
    nd.set_coords(v=np.linspace(-1, 1, 4), u=np.linspace(-10., 10., 6))
    
    nd.set_quaternion()

    # test swapaxes
    nds = nd.swapaxes(0, 1)
    info_(nds)

    assert_array_equal(nd.data.T, nds.data)
    assert nd.coords[0] == nds.coords[0]  # we do not swap the coords

    # test transpose
    nds = nd.T
    info_(nds)

    assert_array_equal(nd.data.T, nds.data)
    assert nd.coords[0] == nds.coords[0]

@pytest.mark.skip()
def test_nddataset_max_with_2D_quaternion(NMR_dataset_2D):
    # test on a 2D NDDataset
    nd2 = NMR_dataset_2D
    assert nd2.is_quaternion
    ndmt = nd2.max()  # no axis specified
    info_(ndmt)
    nd2m = nd2.max(dim=0)  # axis selected
    info_(nd2m)

@pytest.mark.skip()
def test_nddataset_max_min_with_1D(NMR_dataset_1D):
    # test on a 1D NDDataset
    nd1 = NMR_dataset_1D
    nd1[4] = MASKED
    assert nd1.is_masked
    info_(nd1)
    mx = nd1.max()
    info_(mx)
    assert (mx.real, mx.imag) == pytest.approx((2283.5096153847107, -2200.383064516033))
    # check if it works for real
    mx1 = nd1.real.max()
    assert mx1 == pytest.approx(2283.5096153847107)
    mi = nd1.min()
    info_(mi)
    assert (mi.real, mi.imag) == pytest.approx((-408.29714640199626, 261.1864143920416))

@pytest.mark.skip()
def test_nddataset_comparison_of_dataset(NMR_dataset_1D):
    # bug in notebook

    nd1 = NMR_dataset_1D.copy()
    nd2 = NMR_dataset_1D.copy()

    lb1 = nd1.em(lb=100. * ur.Hz)
    lb2 = nd2.em(lb=100. * ur.Hz)

    assert nd1 is not nd2
    assert nd1 == nd2

    assert lb1 is not lb2
    assert lb1 == lb2

    info_(lb1)
    info_(lb2)


def test_nddataset_complex_dataset_slicing_by_index():
    na0 = np.array([1. + 2.j, 2., 0., 0., -1.j, 1j] * 4)
    nd = NDDataset(na0)
    assert nd.shape == (24,)
    assert nd.data.shape == (24,)
    coords = (np.linspace(-10., 10., 24), )
    nd.set_coords(coords)
    x1 = nd.x.copy()
    info_(nd)
    nd.coords = coords
    x2 = nd.x.copy()
    assert x1 == x2
    info_(nd)

    # slicing
    nd1 = nd[0]
    assert nd1.shape == (1,)
    assert nd1.data.shape == (1,)
    info_(nd1)

    # slicing range
    nd2 = nd[1:6]
    assert nd2.shape == (5,)
    assert nd2.data.shape == (5,)
    info_(nd2)

    na0 = na0.reshape(6, 4)
    nd = NDDataset(na0)
    coords = CoordSet(np.linspace(-10., 10., 6), np.linspace(-1., 1., 4))
    nd.set_coords(**coords)
    assert nd.shape == (6, 4)
    assert nd.data.shape == (6, 4)
    nd.coords = coords
    info_(nd)


    # slicing 2D
    nd1 = nd[0]
    assert nd1.shape == (1, 4)
    assert nd1.data.shape == (1, 4)
    info_(nd1)

    # slicing range
    nd1 = nd[1:3]
    assert nd1.shape == (2, 4)
    assert nd1.data.shape == (2, 4)
    info_(nd1)

    # slicing range
    nd1 = nd[1:3, 0:2]
    assert nd1.shape == (2, 2)
    assert nd1.data.shape == (2, 2)
    info_(nd1)

    nd.set_complex()
    assert nd.shape == (6, 4)
    assert nd.data.shape == (6, 4)
    info_(nd)


def test_nddataset_init_complex_1D_with_mask():
    # test with complex with mask and units

    np.random.seed(12345)
    d = np.random.random((5)) * np.exp(.1j)
    d1 = NDDataset(d, units=ur.Hz)  # with units
    d1[1] = MASKED
    assert d1.shape == (5,)
    assert d1._data.shape == (5,)
    assert d1.size == 5
    assert d1.dtype == np.complex
    assert d1.has_complex_dims
    assert d1.mask.shape[-1] == 5
    assert d1[2].data == d[2]
    d1R = d1.real
    assert not d1R.has_complex_dims
    assert d1R._data.shape == (5,)
    assert d1R._mask.shape == (5,)


def test_nddataset_transpose_swapaxes(ds1):
    nd = ds1.copy()
    info_(nd)
    ndt = nd.T
    info_(ndt)
    info_(ndt[-1])
    assert nd[1] == ndt[..., 1].T

    # fix a bug with loc indexation
    nd1 = nd[4000.:3000.]
    assert str(nd1) == 'NDDataset: [float64] a.u. (shape: (z:4, y:100, x:3))'
    nd2 = ndt[..., 4000.:3000.]
    assert str(nd2) == 'NDDataset: [float64] a.u. (shape: (x:3, y:100, z:4))'
    assert nd1 == nd2.T


def test_nddataset_set_coordinates(nd2d, ds1):
    # set coordinates all together
    nd = nd2d.copy()
    ny, nx = nd.shape
    nd.set_coords(x=np.arange(nx), y=np.arange(ny))
    assert nd.dims == ['y', 'x']
    assert nd.x == np.arange(nx)
    nd.transpose(inplace=True)
    assert nd.dims == ['x', 'y']
    assert nd.x == np.arange(nx)

    # set coordinates from tuple
    nd = nd2d.copy()
    ny, nx = nd.shape
    nd.set_coords(np.arange(ny), np.arange(nx))
    assert nd.dims == ['y', 'x']
    assert nd.x == np.arange(nx)
    nd.transpose(inplace=True)
    assert nd.dims == ['x', 'y']
    assert nd.x == np.arange(nx)

    # set coordinate with one set to None: should work!
    # set coordinates from tuple
    nd = nd2d.copy()
    ny, nx = nd.shape
    nd.set_coords(np.arange(ny), None)
    assert nd.dims == ['y', 'x']
    assert nd.y == np.arange(ny)
    assert nd.x.is_empty
    info_('______________')
    info_(nd)
    nd.transpose(inplace=True)
    assert nd.dims == ['x', 'y']
    assert nd.y == np.arange(ny)
    assert nd.x.is_empty
    assert nd.coords == CoordSet(np.arange(ny), None)

    nd = nd2d.copy()
    ny, nx = nd.shape
    nd.set_coords(None, np.arange(nx))
    assert nd.dims == ['y', 'x']
    assert nd.x == np.arange(nx)
    assert nd.y.is_empty

    nd.set_coords(y=np.arange(ny), x=None)

    # set up a single coordinates
    nd = nd2d.copy()
    ny, nx = nd.shape
    nd.x = np.arange(nx)
    nd.x = np.arange(nx)  # do it again - fix  a bug
    nd.set_coordtitles(y='intensity', x='time')
    assert repr(nd.coords) == 'CoordSet: [x:time, y:intensity]'
    info_(nd)

    # validation
    with pytest.raises(ValueError):
        nd.x = np.arange(nx + 5)
    with pytest.raises(AttributeError):
        nd.z = None

def test_nddataset_set_coordinates_withnames(nd2d, ds1):
    # set coordinates all together
    nd = nd2d.copy()
    ny, nx = nd.shape
    nd.coords = CoordSet(u=np.arange(nx), v=np.arange(ny))
    assert nd.dims != ['u', 'v'] # dims = ['y','x']

    #set dim names
    nd.dims = ['u','v']
    nd.set_coords(**CoordSet(u=np.arange(ny), v=np.arange(nx)))
    assert nd.dims == ['u', 'v']

    info_(nd)
    assert nd.dims == ['u', 'v']

    info_(nd[1])
    assert nd.dims == ['u', 'v']

### issue 29
def test_nddataset_issue_29_mulitlabels():

    DS = NDDataset(rand(3,4))

    with pytest.raises(ValueError):
        # shape data and label mismatch
        DS.set_coords(DS.y, Coord(title='xaxis', units='s', data=[1, 2, 3, 4], labels=['a', 'b', 'c']))

    c = Coord(title='xaxis', units='s', data=[1, 2, 3, 4], labels=['a', 'b', 'c', 'd'])
    DS.set_coords(x=c)
    c = Coord(title='xaxis', units='s', data=[1, 2, 3, 4], labels=[['a', 'c', 'b', 'd'],['e', 'f', 'g', 'h']])
    d = DS.y
    DS.set_coords(d, c)
    print_(DS.x)
    DS.x.labels = ['alpha', 'beta', 'omega', 'gamma']
    print_(DS)
    print_(DS.x)
    assert DS.x.labels.shape == (4,3)

    # sort
    DS1 = DS.sort(axis=1, by='value', descend=True)
    assert_array_equal(DS1.x,[4,3,2,1])

    # sort
    assert DS.dims==['y','x']
    DS1 = DS.sort(dim='x', by='label', descend=False)
    assert_array_equal(DS1.x,[1,3,2,4])

    DS1 = DS.sort(dim='x', by='label', pos=2, descend=False)
    assert_array_equal(DS1.x,[1,2,4,3])

    DS2 = DS.sort(dim='y')

    DS.y.labels = ['alpha', 'omega', 'gamma']
    DS2 = DS.sort(dim='y')
    assert_array_equal(DS2.y.labels ,['alpha',  'gamma', 'omega'])

    # indexing
    print_(DS[..., 'a':'c'])
    print_(DS['alpha', 'e':'f'])

def test_nddataset_apply_funcs(IR_dataset_1D):
    
    # convert to masked array
    out = np.ma.array(IR_dataset_1D)
    IR_dataset_1D[1] = MASKED
    out = np.ma.array(IR_dataset_1D)
    out = array(IR_dataset_1D)
    
    print(out)
    
# ----------------------------------------------------------------------------------------------------------------------
# Pandas
# ----------------------------------------------------------------------------------------------------------------------

def test_nddataset_init_pandas(series, dataframe):
    
    # init with pandas
    dx = series
    da = NDDataset(dx)
    assert isinstance(da, NDDataset)
    assert_equal(da.dtype, dx.values.dtype)
    assert_equal(da.shape, dx.shape)
    assert_array_equal(da.data, dx.values)
    
    dx = dataframe
    da = NDDataset(dx)
    assert isinstance(da, NDDataset)
    for i, coord in enumerate(da.coords):
        assert isinstance(coord, Coord)
    assert da.coords.titles == ['temperature', 'time']  # sort was applied on the coordinates
    assert da.dims == ['y', 'x']
    assert_equal(da.dtype, dx.values.dtype)
    assert_equal(da.shape, dx.shape)
    assert_array_equal(da.data, dx.values)
    info_(da)
    
    # Panel was removed from Panda
    # # init with a panel directly (get the coords)
    # dx = panel
    # da = NDDataset(dx)
    # assert isinstance(da, NDDataset)
    # assert_equal(da.dtype, dx.values.dtype)
    # assert_equal(da.shape, dx.shape)
    # assert_array_equal(da.data, dx.values)
    #
    # assert len(da.data) == 7
    # assert da.coords.titles == ['axe2', 'axe1', 'axe0']
    # assert da.dims == ['z', 'y', 'x']
    #
    # # various mode of access to the coordinates
    # assert_array_equal(da.coords[2].data, panel.axes[0].values) # not recommended (except if one know that the coordinates are ordered)
    # assert_equal(da.coords['x'].data, panel.axes[2].values)
    # assert_equal(da.coords(axis=1).data, panel.axes[1].values)
    # assert_equal(da.coords(1).data, panel.axes[1].values)
    # assert_equal(da.coords(axis='z').data, panel.axes[0].values)
    #
    # # selection of the axis
    # assert isinstance(da.coords[1], Coord)
    # assert isinstance(da.coords[1:], CoordSet)
    # assert isinstance(da.coords(0, 2), CoordSet)
    # assert isinstance(da.coords(0, 2)[0], Coord)
    # assert isinstance(da.coords(2), Coord)

# TODO: write test for to_pandas conversion

# ----------------------------------------------------------------------------------------------------------------------
# Xarray
# ----------------------------------------------------------------------------------------------------------------------

# TODO:  dataset from xarray

def test_nddataset_xarray_export(IR_dataset_2D):
    nd = IR_dataset_2D.copy()
    da = nd.to_xarray()
    info_(da)


def test_nddataset_xarray_export_w_spa():
    na = NDDataset.read_omnic(os.path.join('irdata', 'subdir', '7_CZ0-100 Pd_101.SPA'))
    info_(na)
    da = na.to_xarray()
    info_(da)

