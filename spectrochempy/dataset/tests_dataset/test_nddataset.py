# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT 
# See full LICENSE agreement in the root directory
# =============================================================================

"""Tests for the nddataset module

"""

import pandas as pd
import pytest
import numpy as np


# from spectrochempy import *
from spectrochempy.dataset.nddataset import NDDataset
from spectrochempy.dataset.ndcoords import Coord, CoordSet
from spectrochempy.units import ur
from spectrochempy.api import masked, show

from spectrochempy.extern.pint.errors import (UndefinedUnitError,
                                              DimensionalityError)
from spectrochempy.utils import Meta, SpectroChemPyWarning
from spectrochempy.utils.testing import (assert_equal, assert_array_equal,
                         assert_array_almost_equal, assert_equal_units,
                         raises)
from spectrochempy.utils.testing import RandomSeedContext


def test_fix_crossvalidate_bug():
    assert hasattr(NDDataset(), '_%s_validate' % '_iscopy') is False


def test_nddataset_init_with_nonarray():
    inp = [1, 2, 3]
    nd = NDDataset(inp)
    assert (np.array(inp) == nd.data).all()


def test_nddataset_init_with_ndarray(ndarray):

    nd = NDDataset(ndarray)
    nd.mask[0] = True    # with a mask
    assert nd.is_masked
    print(nd)


def test_nddataset_simple(nd):
    assert nd.data.shape == (10, 10)
    assert nd.data.size == 100
    assert nd.data.dtype == np.dtype(float)


# @pytest.mark.xfail(True, reason='not fully implemented ')
def test_nddataset_str():
    arr1d = NDDataset([1, 2, 3])
    # print(arr1d)
    assert str(arr1d).splitlines()[0].strip().startswith('name/id:')

    arr2d = NDDataset(np.array([[1, 2], [3, 4]]))
    assert str(arr2d).splitlines()[0].strip().startswith('name/id:')

    d0unc = NDDataset([2, 3, 4, 5], uncertainty=[.1,.2,.15,.21],
                     mask=[1,0,0,0])  # sequence + mask + uncert
    assert d0unc.shape == (4,)
    assert not d0unc.has_complex_dims
    assert d0unc.is_masked
    assert d0unc.is_uncertain
    assert str(d0unc).splitlines()[9].strip().startswith(
                                              '[   --    3.000+/-0.200')
    assert repr(d0unc).startswith(
            'NDDataset: [   --,    3.000+/-0.200,    4.000+/-0.150,')


def test_nddataset_repr():
    arr1d = NDDataset(np.array([1, 2, 3]))
    assert repr(arr1d).startswith('NDDataset')
    arr2d = NDDataset(np.array([[1, 2], [3, 4]]))
    assert repr(arr2d).startswith('NDDataset')


def test_nddataset_axes_invalid():
    coord1 = Coord(np.arange(10), name='hello')  # , units='m')
    coord2 = Coord(np.arange(20), name='hello')  # , units='s')
    with pytest.raises(ValueError):
        # a different name must be provided for each axis
        print((coord1.name, coord2.name))
        ndd1 = NDDataset(np.random.random((10, 20)), coordset=[coord1, coord2])


def test_nddataset_coords_valid():
    coord1 = Coord(np.arange(10), title='wavelengths')  # , units='m')
    coord2 = Coord(np.arange(20), title='time')  # , units='s')
    ndd1 = NDDataset(np.random.random((10, 20)), coordset=[coord1, coord2])


def test_nddataset_coords_with_units_valid():
    coord1 = Coord(np.arange(10), title='wavelengths', units='cm^-1')
    coord2 = Coord(np.arange(20), title='time', units='s')
    ndd1 = NDDataset(np.random.random((10, 20)), title='absorbance',
                     coordset=[coord1, coord2])


def test_nddataset_coords_invalid_length():
    coord1 = Coord(np.arange(9), title='wavelengths')  # , units='m')
    coord2 = Coord(np.arange(20), title='time')  # , units='s')
    with pytest.raises(ValueError):
        ndd1 = NDDataset(np.random.random((10, 20)), coordset=[coord1, coord2])


def test_nddataset_mask_valid():
    NDDataset(np.random.random((10, 10)),
              mask=np.random.random((10, 10)) > 0.5)


def test_nddataset_uncertainty_init():
    ar = np.ones((5, 5))
    un = np.abs(ar * .01)
    d = NDDataset(ar, uncertainty=un)


def test_nddataset_init_from_nddatasetset_argument_only():
    ndd1 = NDDataset(np.array([1]), title='toto')
    ndd2 = NDDataset(ndd1)
    assert ndd2.id != ndd1.id
    assert ndd2.title == ndd1.title
    assert ndd2.coords == ndd1.coords
    assert ndd2.uncertainty == ndd1.uncertainty
    assert ndd2.mask == ndd1.mask
    assert ndd2.units == ndd1.units
    assert ndd2.meta == ndd1.meta
    assert ndd2 is not ndd1
    assert ndd2 == ndd1


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


def test_invalid_units():
    with pytest.raises(UndefinedUnitError):
        d = NDDataset(np.ones((5, 5)), units="NotAValidUnit")


def test_uncertainties_and_units(nd1d):
    nd = nd1d.copy()
    nd = np.fabs(nd)
    nd.units = 'm'
    nd._uncertainty = np.abs(nd._data * .01)
    nd2 = np.sqrt(nd)
    assert isinstance(nd2, type(nd))
    assert nd2.data[1] == np.sqrt(nd.data[1])
    assert nd2.units == ur.m ** .5
    assert np.round(nd2.uncertainty[1], 4) == 0.0071

    nd.units = 'cm'
    nd2 = np.sqrt(nd)
    nd.ito('m')
    nd2 = np.sqrt(nd)
    assert isinstance(nd2, type(nd))
    assert nd2.data[1] == np.sqrt(nd.data[1])
    assert nd2.units == ur.m ** .5
    assert np.round(nd2.uncertainty[1], 4) == 0.0007

def test_initializing_from_nddataset():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = NDDataset(d1)
    assert d1.data is d2.data # by default we do not perform a copy of data

    d1 = NDDataset(np.ones((2, 2)))
    d2 = NDDataset(d1, copy=True)   # change the default behavior
    assert d1.data is not d2.data

# Test an array and a scalar because a scalar Quantity does not always
# behaves the same way as an array.
@pytest.mark.parametrize('data', [np.array([1, 2, 3]), 5])
def test_initializing_nddataset_from_quantity(data):
    # Until nddataset and quantity are integrated initializing with a quantity
    # should raise an error.
    units = ur.meter
    ndd = NDDataset(data * units)
    assert ndd.units == units
    np.testing.assert_array_equal(ndd.data, np.array(data))


def test_initializing_nddataset_from_quantity_and_units_raises_error():
    # Should raise an error if a Quantity is provided for the data and
    # an explicit units is given.
    with pytest.raises(UndefinedUnitError):
        NDDataset([1, 2, 3] * ur.adu, units=ur.adu)


def test_masked_array_input():
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


# initi with panda structure
# Some panda structure for dataset initialization
@pytest.fixture()
def series():
    with RandomSeedContext(2345):
        arr = pd.Series(np.random.randn(4), index=np.arange(4) * 10.)
    arr.index.name = 'un nom'
    return arr


@pytest.fixture()
def dataframe():
    with RandomSeedContext(23451):
        arr = pd.DataFrame(np.random.randn(6, 4), index=np.arange(6) * 10.,
                           columns=np.arange(4) * 10.)
    for ax, name in zip(arr.axes, ['y', 'x']):
        ax.name = name
    return arr


@pytest.fixture()
def panel():
    shape = (7, 6, 5)
    with RandomSeedContext(23452):
        # TODO: WARNING: pd.Panel is deprecated in pandas
        arr = pd.Panel(np.random.randn(*shape), items=np.arange(shape[0]) * 10.,
                       major_axis=np.arange(shape[1]) * 10.,
                       minor_axis=np.arange(shape[2]) * 10.)

    for ax, name in zip(arr.axes, ['z', 'y', 'x']):
        ax.name = name
    return arr


@pytest.fixture()
def panelnocoordname():
    shape = (7, 6, 5)
    with RandomSeedContext(2452):
        arr = pd.Panel(np.random.randn(*shape), items=np.arange(shape[0]) * 10.,
                       major_axis=np.arange(shape[1]) * 10.,
                       minor_axis=np.arange(shape[2]) * 10.)
    return arr


def test_init_series(series):
    # init with a panel directly (get the coords)
    dx = series
    da = NDDataset(dx)
    assert isinstance(da, NDDataset)
    assert_equal(da.dtype, dx.values.dtype)
    assert_equal(da.shape, dx.shape)
    assert_array_equal(da.data, dx.values)


def test_init_dataframe(dataframe):
    # init with a panel directly (get the coords)
    dx = dataframe
    da = NDDataset(dx)
    assert isinstance(da, NDDataset)
    assert_equal(da.dtype, dx.values.dtype)
    assert_equal(da.shape, dx.shape)
    assert_array_equal(da.data, dx.values)


def test_init_panel(panelnocoordname, panel):
    # init with a panel directly (get the coords)
    dx = panel
    da = NDDataset(dx)
    assert isinstance(da, NDDataset)
    assert_equal(da.dtype, dx.values.dtype)
    assert_equal(da.shape, dx.shape)
    assert_array_equal(da.data, dx.values)

    assert len(da.data) == 7
    assert da.coordset.titles == ['z', 'y', 'x']

    # various mode of access to the coordinates
    assert_array_equal(da.coordset[-1].data, panel.axes[-1].values)
    assert_equal(da.coordset['z'].data, panel.axes[0].values)
    assert_equal(da.coordset(axis=1).data, panel.axes[1].values)
    assert_equal(da.coordset(1).data, panel.axes[1].values)
    assert_equal(da.coordset(axis='z').data, panel.axes[0].values)

    # selection of the axis
    assert isinstance(da.coordset[1], Coord)
    assert isinstance(da.coordset[1:], CoordSet)
    assert isinstance(da.coordset(0, 2), CoordSet)
    assert isinstance(da.coordset(0, 2)[0], Coord)
    assert isinstance(da.coordset(2), Coord)


def test_set_coords_parameters(panel):
    dx = panel.values
    coords = panel.axes
    da = NDDataset(dx,
                   coords,
                   title='absorbance',
                   )

    da.coordset.titles = ['wavelength', 'time-on-stream', 'another']
    da.coordset.units = ['cm^-1', 's', 'm**2/s']

    # print(da)


def test_uncertainties(nd1d):
    nd = nd1d.copy()
    nd._uncertainty = np.abs(nd._data * .01)
    nd2 = np.cos(nd)
    assert isinstance(nd2, type(nd))
    assert nd2.data[0] == np.cos(nd.data[0])
    assert nd2.units is None
    assert np.round(nd2.uncertainty[0], 4) == 0.0084


def test_ndarray_swapaxes(nd1d, nd2d):
    nd1 = nd1d.copy()
    nd2 = nd2d.copy()

    # swapaxes needs 2D at least
    assert nd1.shape == (4,)
    nd1s = nd1.swapaxes(1, 0)
    assert_equal(nd1s.data, nd1.data)

    assert nd2.shape == (2, 4)
    nd2s = nd2.swapaxes(1, 0)
    assert nd2.shape == (2, 4)  # original unchanged
    assert nd2s.shape == (4, 2)
    assert nd2s is not nd2

    nd2s = nd2.swapaxes(1, 0, inplace=True)
    assert nd2.shape == (4, 2)  # original changed
    assert nd2s is nd2  # objects should be the same

    # use of the numpy method
    nd2s = np.swapaxes(nd2, 1, 0)
    assert nd2.shape == (4, 2)  # original unchanged
    assert nd2s.shape == (2, 4)
    assert nd2s is not nd2

    # TODO: add check for swscpng of all elements of a dataset such as meta


def test_set_coords_parameters_at_init():
    dx = np.random.random((10, 10, 10))
    coord0 = np.arange(10)
    coord1 = np.arange(10)
    coord2 = np.arange(10)
    da = NDDataset(dx,
                   coordset=[coord0, coord1, coord2],
                   title='absorbance',
                   coordtitles=['wavelength', 'time-on-stream', 'another'],
                   coordunits=['cm^-1', 's', None],
                   )
    assert da.coordset.titles == ['wavelength', 'time-on-stream', 'another']
    assert da.coordset.units == [ur.Unit('cm^-1'), ur.Unit('s'), None]
    # print(da)

    # da[0]


def test_coords_indexer():
    dx = np.random.random((10, 100, 10))
    coord0 = np.linspace(4000, 1000, 10)
    coord1 = np.linspace(0, 60, 10)  # wrong length
    coord2 = np.linspace(20, 30, 10)

    with pytest.raises(ValueError):
        da = NDDataset(dx,
                       coordset=[coord0, coord1, coord2],
                       title='absorbance',
                       coordtitles=['wavelength', 'time-on-stream',
                                   'temperature'],
                       coordunits=['cm^-1', 's', 'K'],
                       )

    coord1 = np.linspace(0, 60, 100)
    da = NDDataset(dx,
                   coordset=[coord0, coord1, coord2],
                   title='absorbance',
                   coordtitles=['wavelength', 'time-on-stream', 'temperature'],
                   coordunits=['cm^-1', 's', 'K'],
                   )

    assert da.shape == (10, 100, 10)
    coordset = da.coordset
    assert len(coordset) == 3

    assert_array_equal(da.coordset[0].data, coord0, "get axis by index failed")
    assert_array_equal(da.coordset['wavelength'].data, coord0,
                       "get axis by title failed")
    assert_array_equal(da.coordset['time-on-stream'].data, coord1,
                       "get axis by title failed")
    assert_array_equal(da.coordset['temperature'].data, coord2,
                       "get axis by title failed")

    da.coordset['temperature'].data += 273.15
    with pytest.raises(
            AssertionError):  # because the original data is also modified
        assert_array_equal(da.coordset['temperature'].data, coord2 + 273.15,
                           "get axis by title failed")

    # this is ok
    assert_array_equal(da.coordset['temperature'].data, coord2,
                       "get axis by title failed")


################################### TEST SLICING################################

def test_loc2index(dataset3d):
    da = dataset3d
    # print(da)
    assert da.shape == (10, 100, 3)
    coordset = da.coordset
    assert len(coordset) == 3

    assert da._loc2index(3990.0, axis=0) == 0
    assert da._loc2index('j', axis=0) == 9
    with pytest.raises(IndexError):
        da._loc2index('z', 0)  # labels doesn't exist
    with pytest.warns(SpectroChemPyWarning):
        da._loc2index(5000, 0)
    assert da._loc2index(5000, 0) == 0  # return the low limit
    assert da._loc2index(0, 0) == 9  # return the high limit index


def test_dataset_slicing_by_index(dataset3d):
    # case where the index is an integer: the selection is by index starting at zero
    da = dataset3d
    assert da.shape == (10, 100, 3)

    plane0 = da[0]
    assert type(plane0) == type(da)  # should return a dataset
    assert plane0.ndim == 2
    assert plane0.shape == (100,3)
    assert plane0.size == 300
    # print("Plane0: ", plane0)

    # a plane but without reduction
    plane1 = da[1:2]
    assert type(plane1) == type(da)
    assert plane1.ndim == 2
    assert plane1.size == 300
    # print("Plane1: ", plane1)

    # another selection
    row0 = plane0[:, 0]
    assert type(row0) == type(da)
    assert row0.shape == (100, )

    # and again selection
    element = row0[..., 0]
    assert type(element) == type(da)
    print(("element: ", element))

    # now a slicing in multi direction
    matrix = da[1:4, 10:40:2, :2]
    # print("matrix: ", matrix)

    # now again a slicing in multi direction (full selection)
    matrix = da[:, :, -2]
    # print("matrix: ", matrix)

    # now again a slicing in multi direction (ellipsis)
    matrix = da[..., -1]
    # print("matrix: ", matrix)

    matrix = da[-1, ...]
    # print("matrix: ", matrix)


def test_dataset_slicing_by_label(dataset3d):
    da = dataset3d

    # selection
    planeb = da['b']
    assert type(planeb) == type(da)
    plane1 = da[1]
    assert_equal(planeb.data, plane1.data)
    assert planeb.ndim == 2
    assert planeb.size == 300
    bd = da['b':'f']  # the last index is included
    assert bd.shape == (5, 100, 3)
    # print(bd)
    b1 = da[1:6]
    assert_equal(bd.data, b1.data)
    bc = da['b':'f', :, "hot"]
    assert bc.shape == (5, 100)
    assert bc.coordset(0).labels[0] == 'b'


def test_dataset_slicing_by_values(dataset3d):
    da = dataset3d

    x = da[3000.]
    # print(x)
    assert x.shape == (100, 3)

    y = da[3000.0:2000.0, :, 210.]
    # print(y)
    assert y.shape == (4, 100)

    # slicing by values should also work using reverse order
    yr = da[2000.0:3000.0, :, 210.]


@raises(IndexError)
def test_dataset_slicing_out_limits(dataset3d):
    da = dataset3d
    y = da[3000:2000, :, 210.]


@raises(IndexError)
def test_dataset_slicing_toomanyindex(dataset3d):
    da = dataset3d
    y = da[:, 3000.:2000., :, 210.]


def test_dataset_slicing_by_index_nocoords(dataset3d):
    # case where the index is an integer:
    # the selection is by index starting at zero
    da = dataset3d
    da._coords = None  # clear coords
    plane0 = da[1]
    assert type(plane0) == type(da)  # should return a dataset
    assert plane0.ndim == 2
    assert plane0.size == 300
    # print("Plane0: ", plane0)
    plane1 = da[3666.7]
    # print("Plane1: ", plane1)
    assert_array_equal(plane0.data, plane1.data)


# Uncertainty tests
def test_nddataset_uncertainty_init_invalid_shape_1():
    un = np.ones((6, 6))
    with pytest.raises(ValueError) as exc:
        NDDataset(np.ones((5, 5)), uncertainty=un)
    assert exc.value.args[
               0] == 'uncertainty (6, 6) and data (5, 5) shape mismatch!'

# slicing tests
def test_nddataset_simple_slicing():

    d1 = NDDataset(np.ones((5, 5)))
    assert d1.data.shape == (5, 5)
    assert d1.shape == (5, 5)

    d2 = d1[2:3, 2:3]
    assert d2.data.shape == ()
    assert (d1 is not d2)
    d3 = d1[2, 2]
    assert d3.data.shape == ()

    d3 = d1[0]
    assert d3.shape == (5,)

def test_ndataset_slicing_with_uncertainty():
    u1 = np.ones((5, 5)) * 3
    d1 = NDDataset(np.ones((5, 5)), uncertainty=u1)
    assert d1[0].shape==(5,)

def test_ndataset_slicing_with_mask():
    mask = np.zeros((5, 5)).astype(bool)
    mask[1,1]=True
    d1 = NDDataset(np.ones((5, 5)), mask=mask)
    assert d1[1].shape==(5,)
    assert d1[1,1].mask

def test_ndataset_slicing_with_mask_and_uncertainty_and_units():
    u1 = np.ones((5, 5)) * 3
    mask = np.zeros((5, 5)).astype(bool)
    mask[1, 1] = True
    d1 = NDDataset(np.ones((5, 5)), uncertainty=u1, mask=mask, units='m')
    assert d1[0].shape==(5,)


def test_slicing_with_coords(dataset3d):
    da = dataset3d
    assert da[0, 0].shape == (3,)
    assert_array_equal(da[0, 0].coordset[-1].data, da.coordset[-1].data)



def test_mask_array_input():
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
    # assert_array_equal(result1.data, result2.data)  # warning: masked array!
    # they cannot be compared based on the data only
    assert result2.is_masked
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


def test_ndddata_with_mask_acts_like_masked_array():
    # test for #2414
    input_mask = np.array([True, False, False])
    input_data = np.array([1., 2., 3.])
    ndd_masked = NDDataset(input_data.copy(), mask=input_mask.copy())
    other = - np.ones_like(input_data)
    result1 = ndd_masked * other
    result2 = other * ndd_masked
    # Test for both orders of multiplication
    for result in [result1, result2]:
        assert result.is_masked
        # Result mask should match input mask because other has no mask
        assert np.all(result.mask == input_mask)
        # TODO:IndexError: in the future, 0-d boolean arrays will be interpreted as a valid boolean index
        # assert np.all(result[~result.mask].data == - input_data[~input_mask])


def test_creationdate():
    ndd = NDDataset([1., 2., 3.])
    ndd2 = np.sqrt(ndd)
    print(ndd2)
    assert ndd2._date is not None


def test_title():
    ndd = NDDataset([1., 2., 3.], title='xxxx')
    assert ndd.title == 'xxxx'

    ndd2 = NDDataset(ndd, title='yyyy')
    assert ndd2.title == 'yyyy'

    ndd2.title = 'zzzz'
    assert ndd2.title == 'zzzz'


def test_create_from_complex_data():
    # 1D (complex)
    nd = NDDataset([1. + 2.j, 2. + 0j])
    assert nd.data.size == 4
    assert nd.size == 2
    assert nd.data.shape == (4,)
    assert nd.shape == (2,)

    # 2D (complex in the last dimension - automatic detection)
    nd = NDDataset([[1. + 2.j, 2. + 0j],
                    [1.3 + 2.j, 2. + 0.5j],
                    [1. + 4.2j, 2. + 3j]
                    ])

    assert nd.data.size == 12
    assert nd.size == 6
    assert nd.data.shape == (3, 4)
    assert nd.shape == (3, 2)

    # 2D (complex but not in the last dimension )
    nd = NDDataset([[1., 2.],
                    [1.3, 2.],
                    [1., 2.],
                    [1., 2.],
                    ], is_complex=[True, False])

    assert nd.data.size == 8
    assert nd.size == 4
    assert nd.data.shape == (4, 2)
    assert nd.shape == (2, 2)

    # 2D (complex in all dimension )
    nd = NDDataset([[1.j, 2.],
                    [1.3, 2.j],
                    [1.j, 2.],
                    [1., 2.],
                    ], is_complex=[True, True])

    assert nd.data.size == 16
    assert nd.size == 4
    assert nd.data.shape == (4, 4)
    assert nd.shape == (2, 2)

    # take real part
    ndr = nd.real
    assert ndr.shape == (2,2)
    assert ndr.is_complex == [True, False]


    pass


def test_set_complex_1D_during_math_op():
    nd = NDDataset([1., 2.], coordset=[Coord([10, 20])], units='meter')
    assert nd.data.size == 2
    assert nd.size == 2
    assert nd.shape == (2,)
    assert nd.is_complex == [False, ]

    ndj = nd * 1j
    assert ndj.data.size == 4
    assert ndj.is_complex[-1]


def test_create_from_complex_data_with_units_and_uncertainties():
    # 1D
    nd = NDDataset([1. + 2.j, 2. + 0j])
    assert nd.data.size == 4
    assert nd.size == 2
    assert nd.data.shape == (4,)
    assert nd.shape == (2,)

    # add units
    nd.units = 'm**-1'
    nd.ito('cm^-1')

    # add uncertainties
    # u1 = np.ones_like(nd) / 100.  #TODO: look how to implement this complex
    # uncertainty correctly -
    # nd.uncertainty = u1

    # 2D
    nd2 = NDDataset(
            [[1. + 2.j, 2. + 0j], [1.3 + 2.j, 2. + 0.5j], [1. + 4.2j, 2. + 3j]])

    assert nd2.data.size == 12
    assert nd2.size == 6
    assert nd2.data.shape == (3, 4)
    assert nd2.shape == (3, 2)

    # add units
    nd2.units = 'm**-1'
    nd2.ito('cm^-1')

    # add uncertainties
    #u2 = np.ones_like(nd2) / 100.
    #nd2.uncertainty = u2


def test_real_imag():
    na = np.array(
            [[1. + 2.j, 2. + 0j],
             [1.3 + 2.j, 2. + 0.5j],
             [1. + 4.2j, 2. + 3j]])

    nd = NDDataset(na)

    # in the last dimension
    assert_array_equal(nd.real, na.real)
    assert_array_equal(nd.imag, na.imag)

    # in another dimension
    with raises(ValueError):
        nd.set_complex(axis=0)  # cannot be complex as the number of row
        # doesn't match an even number

    na = np.array(
            [[1. + 2.j, 2. + 0j], [1.3 + 2.j, 2. + 0.5j],
             [1. + 4.2j, 2. + 3j], [5. + 4.2j, 2. + 3j]])

    nd = NDDataset(na)
    nd.set_complex(axis=0)
    assert nd.is_complex == [True, True]
    assert_array_equal(nd.real, na.real)
    assert_array_equal(nd.imag, na.imag)

    na0 = np.array([[1., 2., 2., 0.],
                    [1.3, 2., 2., 0.5],
                    [1, 4.2, 2., 3.],
                    [5., 4.2, 2., 3.]])

    assert_array_equal(nd.data, na0)

    nareal0 = np.array([[1., 2., 2., 0.],
                        [1, 4.2, 2., 3.]])

    assert_array_equal(nd.part("R*"), nareal0)

    naimag0 = np.array([[1.3, 2., 2., 0.5],
                        [5., 4.2, 2., 3.]])

    assert_array_equal(nd.part("I*"), naimag0)


def test_complex_full():
    na0 = np.array([[1., 2., 2., 0., 0., 0.],
                    [1.3, 2., 2., 0.5, 1., 1.],
                    [1, 4.2, 2., 3., 2., 2.],
                    [5., 4.2, 2., 3., 3., 3.]])

    nd = NDDataset(na0)
    coordset = CoordSet([np.linspace(-1, 1, 4), np.linspace(-10., 10., 6)])
    assert nd.shape == (4, 6)
    nd.coordset = coordset
    # print(nd)
    nd.set_complex(axis=0)
    # print(nd)

    # test swapaxes
    nds = nd.swapaxes(0, 1)
    # print(nds)

    assert_array_equal(nd.data.T, nds.data)
    assert nd.coordset[1] == nds.coordset[0]

    # test transpose
    nds = nd.T
    # print(nds)

    assert_array_equal(nd.data.T, nds.data)
    assert nd.coordset[1] == nds.coordset[0]


def test_nddataset_from_scp():
    pass


def test_complex_dataset_slicing_by_index():
    na0 = np.array([1. + 2.j, 2., 0., 0., -1.j, 1j] * 4)
    nd = NDDataset(na0)
    print(nd)
    coordset = CoordSet([np.linspace(-10., 10., 24)])
    nd.coordset = coordset

    assert nd.shape == (24,)
    assert nd.data.shape == (48,)
    # print(nd)

    # slicing
    nd1 = nd[0]
    assert nd1.shape == (1,)  ###TODO: To check
    assert nd1.data.shape == ()
    # print(nd1)


    # slicing range
    nd2 = nd[1:6]
    assert nd2.shape == (5,)
    assert nd2.data.shape == (10,)
    # print(nd2)

    na0 = na0.reshape(6, 4)
    nd = NDDataset(na0)
    coordset = CoordSet([np.linspace(-10., 10., 6), np.linspace(-1., 1., 4)])
    nd.coordset = coordset
    assert nd.shape == (6, 4)
    assert nd.data.shape == (6, 8)
    # print(nd)

    # slicing 2D
    nd1 = nd[0]
    assert nd1.shape == (4,)
    assert nd1.data.shape == (8,)
    # print(nd1)

    # slicing range
    nd1 = nd[1:3]
    assert nd1.shape == (2, 4)
    assert nd1.data.shape == (2, 8)
    # print(nd1)

    # slicing range
    nd1 = nd[1:3, 0:2]
    assert nd1.shape == (2, 2)
    assert nd1.data.shape == (2, 4)
    # print(nd1)

    nd.set_complex(0)
    assert nd.shape == (3, 4)
    assert nd.data.shape == (6, 8)
    # print(nd)


def test_comparison():
    ndd = NDDataset([1., 2. + 1j, 3.])

    val = ndd * 1.2 - 10.
    val = np.abs(val)
    # print(val)
    # print("######")
    # print(val >= 7.)

    assert np.all(val >= 6.)
    # print( 'ok = behave like a bool')


def test_nddataset_repr_html():

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
                   coordset=[coord0, coord1, coord2],
                   title='absorbance',
                   units='absorbance'
                   )

    print(da)
    a = da._repr_html_()
    print(a)
    #assert "<strong>Units</strong></td><td style='text-align:left'>absorbance</td>" in a


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
    dataset.sort(inplace=True)
    labels = np.array('c b a'.split())
    assert_array_equal(dataset.coordset[0].labels, labels)
    print(dataset)

    dataset.sort(inplace=True)
    print(dataset)
    new = dataset.copy()
    new = new.sort(descend=True, inplace=False)
    print(new)
    assert_array_equal(new.data, dataset.data[::-1])
    assert (new[0, 0] == dataset[-1, 0])
    assert_array_equal(new.coordset[0].labels, labels[::-1])
    assert_array_equal(new.coordset[0].data, dataset.coordset[0].data[::-1])

    new = dataset.copy()
    new.sort(inplace=True, descend=False)
    assert_array_equal(new.data, dataset.data)
    assert (new[0, 0] == dataset[0, 0])
    assert_array_equal(new.coordset[0].labels, labels)

    # check for another dimension

    dataset = ds1.copy()
    new = ds1.copy()
    new.sort(axis=1, inplace=True, descend=False)
    assert_array_equal(new.data, dataset.data)
    assert (new[0, 0, 0] == dataset[0, 0, 0])

    new = dataset.copy()
    new.sort(axis=1, inplace=True, descend=True)
    assert_array_equal(new.data, dataset.data[:, ::-1, :])
    assert (new[0, -1, 0] == dataset[0, 0, 0])


#### multiple axis #############################################################
def test_multiple_axis(dsm):  # dsm is defined in conftest

    da = dsm.copy()
    print(da)  # TODO: improve output in this case of multiple axis

    # check slicing
    assert da.shape == (9, 50)
    coordset = da.coordset
    assert len(coordset) == 2

    #
    assert_array_equal(coordset[0].data, np.linspace(4000., 1000., 9),
                       "get a Coord from Coordset by index failed")

    assert_array_equal(da.coordset['wavenumber'].data,
                       np.linspace(4000., 1000., 9),
                       "get a Coord from Coordset by title failed")

    # for multiple coordinates
    assert_array_equal(da.coordset[1].coords[0].data , np.linspace(0., 60., 50),
                       "get a Coord from Coordset by index failed")

    # but we can also specify, which axis shuld be returned explicitely
    # by an index or a label
    assert_array_equal(da.coordset[1][1].data, np.logspace(1., 4., 50),
                       "get a Coord from Coordset by index failed")

    assert_array_equal(da.coordset[1]['temperature'].data,
                       np.logspace(1., 4., 50),
                       "get a Coord from Coordset by index failed")

    # even simlper we can specify any of the axis title and get it ...
    assert_array_equal(da.coordset['time-on-stream'].data,
                       np.linspace(0., 60., 50),
                       "get a Coord from Coordset by title failed")

    assert_array_equal(da.coordset['temperature'].data, np.logspace(1., 4., 50),
                       "get a Coord from Coordset by title failed")

    da.coordset['temperature'].data += 273.15

    assert_array_equal(da.coordset['temperature'].data,
                       np.logspace(1., 4., 50) + 273.15,
                       "get a Coord from Coordset by title failed")

def test_coords_manipulation(IR_dataset_2D):
    dataset = IR_dataset_2D.copy()
    coord0 = dataset.coordset[0]

    coord0 -= coord0[0]

    print(coord0)


def test_ndarray_plusminus():
    ds = NDDataset([1.,2.,3.])
    dsu = ds.plus_minus(.1)
    assert repr(ds[0]) == "NDDataset:    1.000 unitless" # not  inplace
    assert repr(dsu[0])== "NDDataset:    1.000+/-0.100 unitless"
    np.random.seed(12345)
    ndd = NDDataset(data=np.random.random((3, 3)), units='m')
    ndd.plus_minus(.2, inplace=True)
    assert repr(ndd).startswith('NDDataset: [[   0.930+/-0.200')
    assert repr(ndd).endswith('0.749+/-0.200]] m')

#### Test masks ######

def test_nddataset_use_of_mask(IR_dataset_1D):

    nd = IR_dataset_1D.copy()
    nd[950.:1260.] = masked
    print(nd)
    nd.plot()
    show()

# ----------------------------------------------------------------------------
# additional tests made following some bug fixes
# ----------------------------------------------------------------------------

def test_bug1():
    ds = NDDataset([[1, 2, 3, 6, 8, 0]],
                   coordset=[Coord(data=[1, 2, 3, 4, 5, 6])], units='m')
    print(ds)


def test_repr_html_bug_undesired_display_complex():

    da = NDDataset([1, 2, 3])
    da.title = 'intensity'
    da.description = 'Some experimental measurements'
    da.units = 'dimensionless'
    assert "(complex)" not in da._repr_html_()
    pass

def test_bug_fixe_figopeninnotebookwithoutplot():
    da = NDDataset([1, 2, 3])
    da2 = np.sqrt(da ** 3)
    assert da2._fig is None  # no figure should open

# issue #7 - max()

def test_init_complex_1D_with_mask():
    # test with complex with mask and units

    np.random.seed(12345)
    d = np.random.random((5)) * np.exp(.1j)
    d1 = NDDataset(d, units=ur.Hz)  # with units
    d1.mask[1] = True
    assert d1.shape == (5,)
    assert d1._data.shape == (1,10)
    assert d1.size == 5
    assert d1.dtype == np.complex
    assert d1.has_complex_dims
    assert d1.mask.shape[-1] == d1.shape[-1] * 2
    d3RR = d1.part('RR')
    assert not d3RR.has_complex_dims
    assert d3RR._data.shape == (2, 2)
    assert d3RR._mask.shape == (2, 2)
    assert str(d1).startswith("RR[[   0.925       --]")
    assert str(d1).endswith(     "[   0.018    0.020]] Hz")
    assert d1[1, 1].data == d[1, 1]

def test_max_with_ndarray(ndarray):

    # test on a NDDataset without CoordSet
    nd = NDDataset(ndarray)
    nd.mask[0] = True    # with a mask
    assert nd.is_masked
    mx = nd.max()
    assert mx == 4.940145858999619

def test_max_with_1D(NMR_dataset_1D):
    # test on a 1D NDDataset
    nd1 = NMR_dataset_1D
    nd1.mask[1] = True
    assert nd1.is_masked
    print(nd1)
    mx = nd1.cdata.max()
    assert mx == 821.4872828784091+80.80955334991164j
    mx = nd1.data.max()
    assert mx == 821.4872828784091
    am = nd1.max()
    print(am)

def test_max_with_2D(NMR_dataset_2D):
    # test on a 2D NDDataset
    nd1 = NMR_dataset_1D
    print(nd2.max())
    print(nd2.max(axis=0))
    print(nd1.ndmax())