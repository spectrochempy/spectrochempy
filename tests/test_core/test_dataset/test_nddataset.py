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



"""Tests for the nddataset module

"""

import numpy as np
import pandas as pd
import pytest

from pint import DimensionalityError, UndefinedUnitError
from spectrochempy.api import (NDDataset, CoordSet, Coord, CoordsError, Meta)
from spectrochempy.core.units import ur
from spectrochempy.utils import SpectroChemPyWarning
from tests.utils import (assert_equal, assert_array_equal,
                         assert_array_almost_equal, assert_equal_units,
                         raises)
from tests.utils import NumpyRNGContext


def test_fix_crossvalidate_bug():
    assert hasattr(NDDataset(), '_%s_validate' % '_iscopy') is False


def test_nddataset_init_with_nonarray():
    inp = [1, 2, 3]
    nd = NDDataset(inp)
    assert (np.array(inp) == nd.data).all()


def test_nddataset_simple(nd):
    assert nd.data.shape == (10, 10)
    assert nd.data.size == 100
    assert nd.data.dtype == np.dtype(float)


# @pytest.mark.xfail(True, reason='not fully implemented ')
def test_nddataset_str():
    arr1d = NDDataset([1, 2, 3])
    # print(arr1d)
    assert str(arr1d).splitlines()[2].strip().startswith('author:')

    arr2d = NDDataset(np.array([[1, 2], [3, 4]]))
    assert str(arr2d).splitlines()[2].strip().startswith('author:')


def test_nddataset_repr():
    arr1d = NDDataset(np.array([1, 2, 3]))
    assert repr(arr1d).startswith('NDDataset')
    arr2d = NDDataset(np.array([[1, 2], [3, 4]]))
    assert repr(arr2d).startswith('NDDataset')


def test_nddataset_axes_invalid():
    coord1 = Coord(np.arange(10), name='hello')  # , units='m')
    coord2 = Coord(np.arange(20), name='hello')  # , units='s')
    with pytest.raises(CoordsError):
        # a different name must be provided for each axis
        # print(coord1.name, coord2.name)
        ndd1 = NDDataset(np.random.random((10, 20)), coords=[coord1, coord2])


def test_nddataset_coords_valid():
    coord1 = Coord(np.arange(10), title='wavelengths')  # , units='m')
    coord2 = Coord(np.arange(20), title='time')  # , units='s')
    ndd1 = NDDataset(np.random.random((10, 20)), coords=[coord1, coord2])


def test_nddataset_coords_with_units_valid():
    coord1 = Coord(np.arange(10), title='wavelengths', units='cm^-1')
    coord2 = Coord(np.arange(20), title='time', units='s')
    ndd1 = NDDataset(np.random.random((10, 20)), title='absorbance',
                     coords=[coord1, coord2])


def test_nddataset_coords_invalid_length():
    coord1 = Coord(np.arange(9), title='wavelengths')  # , units='m')
    coord2 = Coord(np.arange(20), title='time')  # , units='s')
    with pytest.raises(CoordsError):
        ndd1 = NDDataset(np.random.random((10, 20)), coords=[coord1, coord2])


def test_nddataset_mask_valid():
    NDDataset(np.random.random((10, 10)),
              mask=np.random.random((10, 10)) > 0.5)


def test_nddataset_uncertainty_init():
    ar = np.ones((5, 5))
    un = np.abs(ar * .01)
    d = NDDataset(ar, uncertainty=un)


def test_nddataset_init_from_nddatasetset_argument_only():
    ndd1 = NDDataset(np.array([1]))
    ndd2 = NDDataset(ndd1)
    assert ndd2.name != ndd1.name
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
    nd.to('m')
    nd2 = np.sqrt(nd)
    assert isinstance(nd2, type(nd))
    assert nd2.data[1] == np.sqrt(nd.data[1])
    assert nd2.units == ur.m ** .5
    assert np.round(nd2.uncertainty[1], 4) == 0.0071

def test_initializing_from_nddataset():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = NDDataset(d1)
    assert d1.data is not d2.data # by default perfomr a copy of data

    d1 = NDDataset(np.ones((2, 2)))
    d2 = NDDataset(d1, iscopy=False)   # change the default behavior
    assert d1.data is d2.data

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
    with NumpyRNGContext(2345):
        arr = pd.Series(np.random.randn(4), index=np.arange(4) * 10.)
    arr.index.name = 'un nom'
    return arr


@pytest.fixture()
def dataframe():
    with NumpyRNGContext(23451):
        arr = pd.DataFrame(np.random.randn(6, 4), index=np.arange(6) * 10.,
                           columns=np.arange(4) * 10.)
    for ax, name in zip(arr.axes, ['y', 'x']):
        ax.name = name
    return arr


@pytest.fixture()
def panel():
    shape = (7, 6, 5)
    with NumpyRNGContext(23452):
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
    with NumpyRNGContext(2452):
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

    assert len(da.coords) == 3
    assert da.coords.titles == ['z', 'y', 'x']

    # various mode of access to the coordinates
    assert_array_equal(da.coords[-1].coords, panel.axes[-1].values)
    assert_equal(da.coords['z'].data, panel.axes[0].values)
    assert_equal(da.coords(axis=1), panel.axes[1].values)
    assert_equal(da.coords(1), panel.axes[1].values)
    assert_equal(da.coords(axis='z'), panel.axes[0].values)

    # selection of the axis
    assert isinstance(da.coords[1], Coord)
    assert isinstance(da.coords[1:], CoordSet)
    assert isinstance(da.coords(0, 2), CoordSet)
    assert isinstance(da.coords(0, 2)[0], Coord)
    assert isinstance(da.coords(2), Coord)


def test_set_coords_parameters(panel):
    dx = panel.values
    coords = panel.axes
    da = NDDataset(dx,
                   coords,
                   title='absorbance',
                   )

    da.coords.titles = ['wavelength', 'time-on-stream', 'another']
    da.coords.units = ['cm^-1', 's', 'm**2/s']

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

    # TODO: add check for swaping of all elements of a dataset such as meta


def test_set_coords_parameters_at_init():
    dx = np.random.random((10, 10, 10))
    coord0 = np.arange(10)
    coord1 = np.arange(10)
    coord2 = np.arange(10)
    da = NDDataset(dx,
                   coords=[coord0, coord1, coord2],
                   title='absorbance',
                   coordstitles=['wavelength', 'time-on-stream', 'another'],
                   coordsunits=['cm^-1', 's', None],
                   )
    assert da.coords.titles == ['wavelength', 'time-on-stream', 'another']
    assert da.coords.units == [ur.Unit('cm^-1'), ur.Unit('s'), None]
    # print(da)

    # da[0]


def test_coords_indexer():
    dx = np.random.random((10, 100, 10))
    coord0 = np.linspace(4000, 1000, 10)
    coord1 = np.linspace(0, 60, 10)  # wrong length
    coord2 = np.linspace(20, 30, 10)

    with pytest.raises(CoordsError):
        da = NDDataset(dx,
                       coords=[coord0, coord1, coord2],
                       title='absorbance',
                       coordstitles=['wavelength', 'time-on-stream',
                                   'temperature'],
                       coordsunits=['cm^-1', 's', 'K'],
                       )

    coord1 = np.linspace(0, 60, 100)
    da = NDDataset(dx,
                   coords=[coord0, coord1, coord2],
                   title='absorbance',
                   coordstitles=['wavelength', 'time-on-stream', 'temperature'],
                   coordsunits=['cm^-1', 's', 'K'],
                   )

    assert da.shape == (10, 100, 10)
    coords = da.coords
    assert len(coords) == 3

    assert_array_equal(da.coords[0].coords, coord0, "get axis by index failed")
    assert_array_equal(da.coords['wavelength'].coords, coord0,
                       "get axis by title failed")
    assert_array_equal(da.coords['time-on-stream'].coords, coord1,
                       "get axis by title failed")
    assert_array_equal(da.coords['temperature'].coords, coord2,
                       "get axis by title failed")

    da.coords['temperature'].coords += 273.15
    with pytest.raises(
            AssertionError):  # because the original data is also modified
        assert_array_equal(da.coords['temperature'].coords, coord2 + 273.15,
                           "get axis by title failed")

    # this is ok
    assert_array_equal(da.coords['temperature'].coords, coord2,
                       "get axis by title failed")


################################### TEST SLICING################################

# Datasets and CoordSet
@pytest.fixture()
def dataset1d():
    # create a simple 1D
    length = 10.
    x_axis = Coord(np.arange(length) * 1000.,
                  title='wavelengths',
                  units='cm^-1')
    with NumpyRNGContext(125):
        ds = NDDataset(np.random.randn(length),
                       coords=[x_axis],
                       title='absorbance',
                       units='dimensionless')
    return ds.copy()


@pytest.fixture()
def dataset3d():
    with NumpyRNGContext(12345):
        dx = np.random.random((10, 100, 3))

    coord0 = Coord(coords=np.linspace(4000., 1000., 10),
                labels='a b c d e f g h i j'.split(),
                mask=None,
                units="cm^-1",
                title='wavelength')

    coord1 = Coord(coords=np.linspace(0., 60., 100),
                labels=None,
                mask=None,
                units="s",
                title='time-on-stream')

    coord2 = Coord(coords=np.linspace(200., 300., 3),
                labels=['cold', 'normal', 'hot'],
                mask=None,
                units="K",
                title='temperature')

    da = NDDataset(dx,
                   coords=[coord0, coord1, coord2],
                   title='absorbance',
                   units='dimensionless',
                   uncertainty=dx * 0.1,
                   mask=np.zeros_like(dx)  # no mask
                   )
    return da.copy()


def test_loc2index(dataset3d):
    da = dataset3d
    # print(da)
    assert da.shape == (10, 100, 3)
    coords = da.coords
    assert len(coords) == 3

    axis = coords[0]
    assert da._loc2index(3990.0, axis) == 0
    assert da._loc2index('j', axis) == 9
    with pytest.raises(ValueError):
        da._loc2index('z', axis)  # labels doesn't exist
    with pytest.warns(SpectroChemPyWarning):
        da._loc2index(5000, axis)
    assert da._loc2index(5000, axis) == 0  # return the low limit
    assert da._loc2index(0, axis) == 9  # return the high limit index


def test_dataset_slicing_by_index(dataset3d):
    # case where the index is an integer: the selection is by index starting at zero
    da = dataset3d

    plane0 = da[0]
    assert type(plane0) == type(da)  # should return a dataset
    assert plane0.ndim == 3
    assert plane0.size == 300
    # print("Plane0: ", plane0)

    # a plane but without reduction
    plane1 = da[1:2]
    assert type(plane1) == type(da)  # should return a dataset
    assert plane1.ndim == 3  # no reduction
    assert plane1.size == 300
    # print("Plane1: ", plane1)

    # another selection
    row0 = plane0[:, 0]
    assert type(row0) == type(da)
    # print("row0: ", row0)

    # and again selection
    element = row0[..., 0]
    assert type(element) == type(da)
    # print("element: ", element)

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
    assert planeb.ndim == 3
    assert planeb.size == 300
    bd = da['b':'f']  # the last index is included
    assert bd.shape == (5, 100, 3)
    # print(bd)
    b1 = da[1:6]
    assert_equal(bd.data, b1.data)
    bc = da['b':'f', :, "hot"]
    assert bc.shape == (5, 100, 1)
    assert bc.coords(0).labels[0] == 'b'
    assert bc['b'].squeeze(axis=-1).coords(0).labels == 'b'


def test_dataset_slicing_by_values(dataset3d):
    da = dataset3d

    x = da[3000.]
    # print(x)
    assert x.shape == (1, 100, 3)

    y = da[3000.0:2000.0, :, 210.]
    # print(y)
    assert y.shape == (4, 100, 1)


@raises(IndexError)
def test_dataset_slicing_out_limits(dataset3d):
    da = dataset3d
    y = da[3000:2000, :, 210.]


@raises(IndexError)
def test_dataset_slicing_toomanyindex(dataset3d):
    da = dataset3d
    y = da[:, 3000.:2000., :, 210.]


def test_dataset_slicing_by_index_nocoords(dataset3d):
    # case where the index is an integer: the selection is by index starting at zero
    da = dataset3d
    da._coords = None  # clear coords
    plane0 = da[5]
    assert type(plane0) == type(da)  # should return a dataset
    assert plane0.ndim == 3
    assert plane0.size == 300
    # print("Plane0: ", plane0)
    plane1 = da[5.0]
    # print("Plane1: ", plane1)
    assert_array_equal(plane0.data, plane1.data)


# Uncertainty tests
def test_nddataset_uncertainty_init_invalid_shape_1():
    un = np.ones((6, 6))
    with pytest.raises(ValueError) as exc:
        NDDataset(np.ones((5, 5)), uncertainty=un)
    assert exc.value.args[
               0] == 'uncertainty shape does not match array data shape'


# slicing tests
def test_simple_slicing():
    u1 = np.ones((5, 5)) * 3
    d1 = NDDataset(np.ones((5, 5)), uncertainty=u1)
    assert d1.data.shape == (5, 5)
    assert d1.shape == (5, 5)
    d2 = d1[2:3, 2:3]
    assert d2.data.shape == (1, 1)
    assert d2.uncertainty.shape == (1, 1)
    assert_array_equal(d2.uncertainty, u1[2:3, 2:3])
    assert (d1 is not d2)
    d3 = d1[2, 2]
    assert d3.data.shape == (1, 1)


def test_slicing_with_mask():
    ndd = NDDataset(np.array([1., 2., 3.]),
                    mask=np.array([False, False, False]))
    assert ndd[0].shape == (1,)
    assert ndd[0].mask.shape == (1,)
    assert not ndd[0].mask


def test_slicing_with_coords(dataset3d):
    da = dataset3d
    assert da[0, 0].shape == (1, 1, 3)
    assert_array_equal(da[0, 0].coords[-1].data, da.coords[-1].data)


def test_ufunc_method(nd):
    assert isinstance(nd, NDDataset)
    nd2 = np.sin(nd)
    assert nd2 is not nd
    assert nd2._data is not nd._data
    sinnd = np.sin(nd._data)
    assert_array_equal(nd2._data, sinnd)

    nd2.units = 'm'
    nds = np.sqrt(nd2)
    assert nds.units == ur.m ** .5
    assert_array_equal(nds._data, np.sqrt(nd2._data))

    nds._data = nds._data[:4, :2]
    # print(nds.shape)
    ndsw = np.swapaxes(nds, 1, 0)
    # print(ndsw.shape)
    pass


def test_unary_ops():
    # UNARY_OPS = ['neg', 'pos', 'abs', 'invert']
    d1 = NDDataset(np.ones((5, 5)))
    d2 = +d1  # pos
    assert isinstance(d2, NDDataset)
    assert np.all(d2.data == 1.)
    d2 = -d1  # neg
    assert isinstance(d2, NDDataset)
    assert np.all(d2.data == -1.)
    d3 = abs(d2)  # abs
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 1.)


def test_unary_ops_with_units():
    # UNARY_OPS = ['neg', 'pos', 'abs']
    d1 = NDDataset(np.ones((5, 5)), units='m')
    d2 = +d1  # pos
    assert isinstance(d2, NDDataset)
    assert np.all(d2.data == 1.)
    assert d2.units == ur.m
    d2 = -d1  # neg
    assert isinstance(d2, NDDataset)
    assert np.all(d2.data == -1.)
    assert d2.units == ur.m
    d3 = abs(d2)  # abs
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 1.)
    assert d2.units == ur("m")


def test_nddataset_add():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = NDDataset(np.ones((5, 5)))
    d3 = -d1
    d3 = d1 * .5 + d2
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 1.5)


def test_nddataset_add_with_numpy_array():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = np.ones((5, 5))
    d3 = d1 * .5 + d2
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 1.5)
    # should commute!
    d3 = d2 + d1 * .5
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 1.5)


def test_nddataset_add_inplace():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = NDDataset(np.ones((5, 5)))
    d1 += d2 * .5
    assert np.all(d1.data == 1.5)


def test_nddataset_add_mismatch_coords():
    coord1 = Coord(np.arange(5.))
    coord2 = Coord(np.arange(1., 5.5, 1.))
    d1 = NDDataset(np.ones((5, 5)), coords=[coord1, coord2])
    d2 = NDDataset(np.ones((5, 5)), coords=[coord2, coord1])
    with pytest.raises(ValueError) as exc:
        d3 = d1 + d2
    assert exc.value.args[0] == "coords properties do not match"
    with pytest.raises(ValueError) as exc:
        d1 += d2
    assert exc.value.args[0] == "coords properties do not match"


def test_nddataset_add_mismatch_units():
    d1 = NDDataset(np.ones((5, 5)), units='cm^2')
    d2 = NDDataset(np.ones((5, 5)), units='cm')
    with pytest.raises(DimensionalityError) as exc:
        d3 = d1 + d2
    with pytest.raises(DimensionalityError) as exc:
        d1 += d2


def test_nddataset_add_mismatch_shape():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = NDDataset(np.ones((6, 6)))
    with pytest.raises(ValueError) as exc:
        d1 += d2
    assert exc.value.args[0].startswith(
            "operands could not be broadcast together")


def test_nddataset_add_with_masks():
    # numpy masked arrays mask the result of binary operations if the
    # mask of either operand is set.
    # Does NDData?
    ndd1 = NDDataset(np.array([1, 2]))
    ndd2 = NDDataset(np.array([2, 1]))
    result = ndd1 + ndd2
    assert_array_equal(result.data, np.array([3, 3]))

    ndd1 = NDDataset(np.array([1, 2]), mask=np.array([True, False]))
    other_mask = ~ ndd1.mask
    ndd2 = NDDataset(np.array([2, 1]), mask=other_mask)
    result = ndd1 + ndd2
    # The result should have all entries masked...
    assert result.mask.all()


def test_nddataset_add_uncertainties():
    u1 = np.ones((5, 5)) * 3
    u2 = np.ones((5, 5))
    d1 = NDDataset(np.ones((5, 5)), uncertainty=u1)
    d2 = NDDataset(np.ones((5, 5)), uncertainty=u2)
    d3 = d1 + d2
    assert np.all(d3.data == 2.)
    assert_array_equal(d3.uncertainty, np.sqrt(10.))


def test_nddataset_subtract():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = NDDataset(np.ones((5, 5)) * 2.)
    d3 = d1 - d2
    assert np.all(d3.data == -1.)


def test_nddataset_substract_with_numpy_array():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = np.ones((5, 5))
    d3 = d1 * .5 - d2
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == -0.5)
    # should commute!
    d3 = d2 - d1 * .5
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 0.5)


def test_nddataset_subtract_mismatch_coords():
    coord1 = Coord(np.arange(5.))
    coord2 = Coord(np.arange(1., 5.5, 1.))
    d1 = NDDataset(np.ones((5, 5)), coords=[coord1, coord2])
    d2 = NDDataset(np.ones((5, 5)), coords=[coord2, coord1])
    with pytest.raises(ValueError) as exc:
        d1 -= d2
    assert exc.value.args[0] == "coords properties do not match"


def test_nddataset_subtract_mismatch_units():
    d1 = NDDataset(np.ones((5, 5)), units='m')
    d2 = NDDataset(np.ones((5, 5)) * 2., units='m/s')
    with pytest.raises(DimensionalityError) as exc:
        d1 -= d2
        # TODO: generate a clear message


def test_nddataset_subtract_mismatch_shape():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = NDDataset(np.ones((6, 6)) * 2.)
    with pytest.raises(ValueError) as exc:
        d1 -= d2
    assert exc.value.args[0].startswith(
            "operands could not be broadcast together")


def test_nddataset_subtract_uncertainties():
    u1 = np.ones((5, 5)) * 3
    u2 = np.ones((5, 5))
    d1 = NDDataset(np.ones((5, 5)), uncertainty=u1)
    d2 = NDDataset(np.ones((5, 5)) * 2., uncertainty=u2)
    d3 = d1 - d2
    assert np.all(d3.data == -1.)
    assert_array_equal(d3.uncertainty, np.sqrt(10.))


def test_nddataset_multiply_uncertainties():
    u1 = np.ones((5, 5)) * 3
    u2 = np.ones((5, 5))
    d1 = NDDataset(np.ones((5, 5)), uncertainty=u1)
    d2 = NDDataset(np.ones((5, 5)) * 2., uncertainty=u2)
    d3 = d1 * d2
    assert np.all(d3.data == 2.)
    assert_array_equal(d3.uncertainty, 2 * np.sqrt(9.25))


def test_nddataset_multiply_with_numpy_array():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = np.ones((5, 5)) * 2.
    d3 = d1 * d2
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 2.)
    # should commute!
    d3 = d2 * d1
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 2.)


def test_nddataset_divide_uncertainties():
    u1 = np.ones((5, 5)) * 3
    u2 = np.ones((5, 5))
    d1 = NDDataset(np.ones((5, 5)), uncertainty=u1)
    d2 = NDDataset(np.ones((5, 5)) * 2., uncertainty=u2)
    d3 = d1 / d2
    assert np.all(d3.data == 0.5)
    assert_array_equal(d3.uncertainty, 0.5 * np.sqrt(9.25))


def test_nddataset_divide_with_numpy_array():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = np.ones((5, 5)) * 2.
    d3 = d1 / d2
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 0.5)
    # should commute!
    d3 = d2 / d1
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 2.)


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
        with NumpyRNGContext(789):
            NDDataset(np.random.random((10, 10)),
                      mask=np.random.random(shape) > 0.5)
    assert exc.value.args[0] == 'dimensions of mask do not match data'


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


# first operand has units km, second has units m
@pytest.mark.parametrize(('operation', 'result_units'), [
    ('__add__', ur.km),
    ('__sub__', ur.km),
    ('__mul__', ur.km * ur.m),
    ('__truediv__', ur.km / ur.m)])
def test_uncertainty_unit_conversion_operators(operation, result_units):
    in_km = NDDataset(np.array([1, 1]), units=ur.km,
                      uncertainty=np.array([.1, .1]))
    in_m = NDDataset(in_km.data * 1000, units=ur.m)
    in_m.uncertainty = np.array(in_km.uncertainty * 1000)
    operator_km = in_km.__getattribute__(operation)
    combined = operator_km(in_m)
    assert_equal_units(combined.units, result_units)
    if operation in ['__add__', '__sub__']:
        # uncertainty is not scaled by result values
        assert_array_almost_equal(combined.uncertainty,
                                  np.sqrt(2) * in_km.uncertainty)
    else:
        # uncertainty is scaled by result
        assert_array_almost_equal(combined.uncertainty,
                                  np.sqrt(
                                          2) * in_km.uncertainty * combined.data)


@pytest.mark.parametrize(('unit1', 'unit2', 'op', 'result_units'), [
    (None, None, '__add__', None),
    (None, None, '__mul__', None),
    (None, ur.m, '__mul__', ur.m),
    (ur.dimensionless, None, '__mul__',
     ur.dimensionless),
    (ur.eV, ur.eV, '__add__', ur.eV),
    (ur.eV, ur.eV, '__sub__', ur.eV),
    (ur.eV, ur.eV, '__truediv__', ur.dimensionless),
    (ur.eV, ur.m, '__mul__', ur.m * ur.eV)
])
def test_arithmetic_unit_calculation(unit1, unit2, op, result_units):
    # Test for #2413
    ndd1 = NDDataset(np.array([1]), units=unit1)
    ndd2 = NDDataset(np.array([1]), units=unit2)
    ndd1_method = ndd1.__getattribute__(op)
    result = ndd1_method(ndd2)
    assert result.units == result_units


UNARY_MATH = ["fabs", "ceil", "floor", "negative", "reciprocal",
              "rint", "sqrt", "square"]


@pytest.mark.parametrize('name', UNARY_MATH)
def test_unary_ufuncs_simple_data(nd, name):
    assert nd.unitless
    f = getattr(np, name)
    r = f(nd)
    assert isinstance(r, NDDataset)


@pytest.mark.parametrize('name', UNARY_MATH)
def test_unary_ufuncs_data_w_uncertainties(nd, name):
    nd._uncertainty = np.abs(nd._data * .01)
    assert nd.unitless
    f = getattr(np, name)
    r = f(nd)
    assert isinstance(r, NDDataset)


@pytest.mark.parametrize('name', UNARY_MATH)
def test_unary_ufuncs_data_w_uncertainties(nd, name):
    nd.units = ur.m
    nd._uncertainty = np.abs(nd._data * .01)
    f = getattr(np, name)
    r = f(nd)
    assert isinstance(r, NDDataset)


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


def test_ndarray_complex(ndcplx):
    nd = ndcplx
    assert nd.data.size == 100
    assert nd.size == 50
    assert nd.data.shape == (10, 10)
    assert nd.shape == (10, 5)  # the real shape
    assert nd.is_complex == [False, True]
    assert nd.ndim == 2

    ndr = nd.real()
    assert_array_equal(ndr.data, nd.data[..., ::2])
    assert ndr.size == nd.size
    assert ndr.is_complex == [False, False]


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
    ndr = nd.real()
    assert ndr.shape == (2,2)
    assert ndr.is_complex == [True, False]


    pass


def test_make_complex_1D_during_math_op():
    nd = NDDataset([1., 2.], coords=[Coord([10, 20])], units='meter')
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
    u1 = np.ones_like(nd) / 100.
    nd.uncertainty = u1

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
    u2 = np.ones_like(nd2) / 100.
    nd2.uncertainty = u2


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
    coords = CoordSet([np.linspace(-1, 1, 4), np.linspace(-10., 10., 6)])
    assert nd.shape == (4, 6)
    nd.coords = coords
    # print(nd)
    nd.make_complex(axis=0)
    # print(nd)

    # test swapaxes
    nds = nd.swapaxes(0, 1)
    # print(nds)

    assert_array_equal(nd.data.T, nds.data)
    assert nd.coords[1] == nds.coords[0]

    # test transpose
    nds = nd.T
    # print(nds)

    assert_array_equal(nd.data.T, nds.data)
    assert nd.coords[1] == nds.coords[0]


def test_nddataset_from_api():
    pass


def test_complex_dataset_slicing_by_index():
    na0 = np.array([1. + 2.j, 2., 0., 0., -1.j, 1j] * 4)

    nd = NDDataset(na0)
    coords = CoordSet([np.linspace(-10., 10., 24)])
    nd.coords = coords

    assert nd.shape == (24,)
    assert nd.data.shape == (48,)
    # print(nd)

    # slicing
    nd1 = nd[0]
    assert nd1.shape == (1,)
    assert nd1.data.shape == (2,)
    # print(nd1)


    # slicing range
    nd2 = nd[1:6]
    assert nd2.shape == (5,)
    assert nd2.data.shape == (10,)
    # print(nd2)

    na0 = na0.reshape(6, 4)
    nd = NDDataset(na0)
    coords = CoordSet([np.linspace(-10., 10., 6), np.linspace(-1., 1., 4)])
    nd.coords = coords
    assert nd.shape == (6, 4)
    assert nd.data.shape == (6, 8)
    # print(nd)

    # slicing 2D
    nd1 = nd[0]
    assert nd1.shape == (1, 4)
    assert nd1.data.shape == (1, 8)
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

    nd.make_complex(0)
    assert nd.shape == (3, 4)
    assert nd.data.shape == (6, 8)
    # print(nd)


def test_absolute_of_complex():
    ndd = NDDataset([1., 2. + 1j, 3.])

    val = np.abs(ndd)
    # print(val)

    val = ndd[1] * 1.2 - 10.
    val = np.abs(val)
    # print(val)

    na0 = np.array([[1., 2., 2., 0., 0., 0.],
                    [1.3, 2., 2., 0.5, 1., 1.],
                    [1, 4.2, 2., 3., 2., 2.],
                    [5., 4.2, 2., 3., 3., 3.]])
    nd = NDDataset(na0)
    coords = CoordSet([np.linspace(-1, 1, 4), np.linspace(-10., 10., 6)])
    assert nd.shape == (4, 6)
    nd.coords = coords
    nd.make_complex(axis=0)
    # print(nd)

    val = np.abs(nd)  # this dimension (the last is ot complex)
    # print(val)

    val = np.fabs(nd)  # this dimension (the last is ot complex)
    # print(val)   # should work the same  (works only on the last dimension

    val = nd.abs(axis=0)  # the np.abs works only on the last dimension
    # print(val)

    # TODO: add more testings


def test_comparison():
    ndd = NDDataset([1., 2. + 1j, 3.])

    val = ndd * 1.2 - 10.
    val = np.abs(val)
    # print(val)
    # print("######")
    # print(val >= 7.)

    assert np.all(val >= 6.)
    # print( 'ok = behave like a bool')


def test_repr_html():
    #
    #
    #
    dx = np.random.random((10, 100, 3))

    coord0 = Coord(coords=np.linspace(4000., 1000., 10),
                labels='a b c d e f g h i j'.split(),
                mask=None,
                units="cm^-1",
                title='wavelength')

    coord1 = Coord(coords=np.linspace(0., 60., 100),
                labels=None,
                mask=None,
                units="s",
                title='time-on-stream')

    coord2 = Coord(coords=np.linspace(200., 300., 3),
                labels=['cold', 'normal', 'hot'],
                mask=None,
                units="K",
                title='temperature')

    da = NDDataset(dx,
                   coords=[coord0, coord1, coord2],
                   title='absorbance',
                   units='dimensionless'
                   )

    # print (da._repr_html_())


#### Squeezing #################################################################
def test_squeeze(ds1):  # ds2 is defined in conftest

    d = ds1[..., 0]
    d = d.squeeze()
    assert d.shape == (10, 100)


#### Metadata ##################################################################
def test_dataset_with_meta(ds1):
    da = ds1.copy()

    meta = Meta()
    meta.essai = ['try_metadata', 10]
    da.meta = meta

    # check copy of meta
    dac = da.copy()
    assert dac.meta == da.meta


#### sorting ###################################################################
def test_sorting(ds1):  # ds1 is defined in conftest

    source = ds1[:3, :3, 0].copy()
    source = source.squeeze()
    source.sort(inplace=True)
    labels = np.array('c b a'.split())
    assert_array_equal(source.coords[0].labels, labels)
    print(source)

    source.sort(inplace=True)
    print(source)
    new = source.copy()
    new = new.sort(descend=True, inplace=False)
    print(new)
    assert_array_equal(new.data, source.data[::-1])
    assert (new[0, 0] == source[-1, 0])
    assert_array_equal(new.coords[0].labels, labels[::-1])
    assert_array_equal(new.coords[0].data, source.coords[0].data[::-1])

    new = source.copy()
    new.sort(inplace=True, descend=False)
    assert_array_equal(new.data, source.data)
    assert (new[0, 0] == source[0, 0])
    assert_array_equal(new.coords[0].labels, labels)

    # check for another dimension

    source = ds1.copy()
    new = ds1.copy()
    new.sort(axis=1, inplace=True, descend=False)
    assert_array_equal(new.data, source.data)
    assert not new.coords(1).is_reversed
    assert (new[0, 0, 0] == source[0, 0, 0])

    new = source.copy()
    new.sort(axis=1, inplace=True, descend=True)
    assert_array_equal(new.data, source.data[:, ::-1, :])
    assert new.coords(1).is_reversed
    assert (new[0, -1, 0] == source[0, 0, 0])


#### multiple axis #############################################################
def test_multiple_axis(dsm):  # dsm is defined in conftest

    da = dsm.copy()
    print(da)  # TODO: improve output in this case of multiple axis

    # check slicing
    assert da.shape == (9, 50)
    coords = da.coords
    assert len(coords) == 2

    #
    assert_array_equal(da.coords[0].coords, np.linspace(4000., 1000., 9),
                       "get axis by index failed")

    assert_array_equal(da.coords['wavenumber'].coords,
                       np.linspace(4000., 1000., 9),
                       "get axis by title failed")

    # for mulitple axis by default, the retruned numerical coordiantes
    # will be the first axis (in the axis set)
    assert_array_equal(da.coords[1].coords, np.linspace(0., 60., 50),
                       "get axis by index failed")

    # but we can also specify, which axis shuld be returned explicitely
    # by an index or a label
    assert_array_equal(da.coords[1][1].coords, np.logspace(1., 4., 50),
                       "get axis by index failed")

    assert_array_equal(da.coords[1]['temperature'].coords,
                       np.logspace(1., 4., 50),
                       "get axis by index failed")

    # even simlper we can specify any of the axis title and get it ...
    assert_array_equal(da.coords['time-on-stream'].coords,
                       np.linspace(0., 60., 50),
                       "get axis by title failed")

    assert_array_equal(da.coords['temperature'].coords, np.logspace(1., 4., 50),
                       "get axis by title failed")

    da.coords['temperature'].coords += 273.15

    assert_array_equal(da.coords['temperature'].coords,
                       np.logspace(1., 4., 50) + 273.15,
                       "get axis by title failed")


def test_bug_fixe_figopeninnotebookwithoutplot():
    da = NDDataset([1, 2, 3])
    da2 = np.sqrt(da ** 3)
    assert da2._fig is None  # no figure should open


def test_coords_manipulation(IR_source_1):
    source = IR_source_1
    coord0 = source.coords[0]

    coord0 -= coord0[0]

    print(coord0)
