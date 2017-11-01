# coding=utf-8
import matplotlib as mpl
import sys

if 'builddocs' in sys.argv[1]:
    mpl.use('agg')

matplotlib_backend = mpl.get_backend()

import numpy as np
import pandas as pd

from spectrochempy.extern.pint import DimensionalityError

from tests.utils import (assert_equal, assert_array_equal,
                         assert_array_almost_equal, assert_equal_units,
                         raises)

from tests.utils import NumpyRNGContext

import pytest
import numpy as np
import os
import sys

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndarray import CoordSet, NDArray
from spectrochempy.core.dataset.ndcoords import Coord
from spectrochempy.api import data, plotoptions

plotoptions.do_not_block = True

#########################
# FIXTURES: some arrays
#########################

@pytest.fixture(scope="module")
def ndarray(): #ndarraysubclass():
    # return a simple ndarray with some data
    with NumpyRNGContext(12345):
        dx = 10.*np.random.random((10, 10))-5.
    _nd = NDArray()
    _nd.data = dx
    return _nd.copy()

@pytest.fixture(scope="module")
def ndarrayunit(): #ndarraysubclassunit():
    # return a simple ndarray with some data
    with NumpyRNGContext(12345):
        dx = 10.*np.random.random((10, 10))-5.
    _nd = NDArray()
    _nd.data = dx
    _nd.units = 'm/s'
    return _nd.copy()

@pytest.fixture(scope="module")
def ndarraycplx():

    # return a complex ndarray
    # with some complex data

    with NumpyRNGContext(12345):
        dx = np.random.random((10, 10))
    nd = NDArray()
    nd.data = dx
    nd.set_complex(axis=-1)  # this means that the data are complex in
                              # the last dimension

    # some checking
    assert nd.data.size == 100
    assert nd.size == 50
    assert nd.data.shape == (10, 10)
    assert nd.shape == (10, 5)  # the real shape
    assert nd.is_complex == [False, True]
    assert nd.ndim == 2

    # return
    return nd.copy()

#########################
# FIXTURES: some datasets
#########################
@pytest.fixture()
def ndcplx():
    # return a complex ndarray
    _nd = NDDataset()
    with NumpyRNGContext(1234):
        _nd._data = np.random.random((10, 10))
    _nd.set_complex(axis=-1)  # this means that the data are complex in
    # the last dimension
    return _nd


@pytest.fixture()
def nd1d():
    # a simple ndarray with negative elements
    _nd = NDDataset()
    _nd._data = np.array([1., 2., 3., -0.4])
    return _nd

@pytest.fixture()
def nd2d():
    # a simple 2D ndarray with negative elements
    _nd = NDDataset()
    _nd._data = np.array([[1., 2., 3., -0.4], [-1., -.1, 1., 2.]])
    return _nd


@pytest.fixture()
def nd():
    # return a simple (positive) ndarray
    _nd = NDDataset()
    with NumpyRNGContext(145):
        _nd._data = np.random.random((10, 10))
    return _nd.copy()

@pytest.fixture()
def ds1():
    with NumpyRNGContext(12345):
        dx = np.random.random((10, 100, 3))
        # make complex along first dimension
        is_complex = [False, False, False]  # TODO: check with complex

    coord0 = Coord(data=np.linspace(4000., 1000., 10),
                 labels='a b c d e f g h i j'.split(),
                 units="cm^-1",
                 title='wavenumber')

    coord1 = Coord(data=np.linspace(0., 60., 100),
                 units="s",
                 title='time-on-stream')

    coord2 = Coord(data=np.linspace(200., 300., 3),
                 labels=['cold', 'normal', 'hot'],
                 units="K",
                 title='temperature')

    da = NDDataset(dx,
                   is_complex=is_complex,
                   coordset=[coord0, coord1, coord2],
                   title='Absorbance',
                   units='absorbance',
                   uncertainty=dx * 0.1,
                   )
    return da.copy()


@pytest.fixture()
def ds2():
    with NumpyRNGContext(12345):
        dx = np.random.random((9, 50, 4))
        # make complex along first dimension
        is_complex = [False, False, False]  # TODO: check with complex

    coord0 = Coord(data=np.linspace(4000., 1000., 9),
                 labels='a b c d e f g h i'.split(),
                 units="cm^-1",
                 title='wavenumber')

    coord1 = Coord(data=np.linspace(0., 60., 50),
                 units="s",
                 title='time-on-stream')

    coord2 = Coord(data=np.linspace(200., 1000., 4),
                 labels=['cold', 'normal', 'hot', 'veryhot'],
                 units="K",
                 title='temperature')

    da = NDDataset(dx,
                   is_complex=is_complex,
                   coordset=[coord0, coord1, coord2],
                   title='Absorbance',
                   units='absorbance',
                   uncertainty=dx * 0.1,
                   )
    return da.copy()


@pytest.fixture()
def dsm():  # dataset with coords containing several axis

    with NumpyRNGContext(12345):
        dx = np.random.random((9, 50))
        # make complex along first dimension
        is_complex = [False, False]  # TODO: check with complex

    coord0 = Coord(data=np.linspace(4000., 1000., 9),
                 labels='a b c d e f g h i'.split(),
                 units="cm^-1",
                 title='wavenumber')

    coord11 = Coord(data=np.linspace(0., 60., 50),
                  units="s",
                  title='time-on-stream')

    coord12 = Coord(data=np.logspace(1., 4., 50),
                  units="K",
                  title='temperature')

    coordmultiple = CoordSet(coord11, coord12)
    da = NDDataset(dx,
                   is_complex=is_complex,
                   coordset=[coord0, coordmultiple],
                   title='Absorbance',
                   units='absorbance',
                   uncertainty=dx * 0.1,
                   )
    return da.copy()


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
                       coordset=[x_axis],
                       title='absorbance',
                       units='dimensionless')
    return ds.copy()


@pytest.fixture()
def dataset3d():
    with NumpyRNGContext(12345):
        dx = np.random.random((10, 100, 3))

    coord0 = Coord(np.linspace(4000., 1000., 10),
                labels='a b c d e f g h i j'.split(),
                mask=None,
                units="cm^-1",
                title='wavelength')

    coord1 = Coord(np.linspace(0., 60., 100),
                labels=None,
                mask=None,
                units="s",
                title='time-on-stream')

    coord2 = Coord(np.linspace(200., 300., 3),
                labels=['cold', 'normal', 'hot'],
                mask=None,
                units="K",
                title='temperature')

    da = NDDataset(dx,
                   coordset=[coord0, coord1, coord2],
                   title='absorbance',
                   units='dimensionless',
                   uncertainty=dx * 0.1,
                   mask=np.zeros_like(dx)  # no mask
                   )
    return da.copy()


############################
# Fixture:  IR spectra (SPG)
############################
@pytest.fixture(scope="function")
def IR_source_1():
    source = NDDataset.read_omnic(
            os.path.join(data, 'irdata', 'NH4Y-activation.SPG'))
    return source


# Fixture:  IR spectra
@pytest.fixture(scope="function")
def IR_scp_1():
    source = NDDataset.load(
            os.path.join(data, 'irdata', 'NH4Y-activation.scp'))
    return source


# Fixture : NMR spectra
@pytest.fixture(scope="function")
def NMR_source_1D():
    path = os.path.join(data, 'nmrdata', 'bruker', 'tests', 'nmr',
                        'bruker_1d')
    source = NDDataset.read_bruker_nmr(
            path, expno=1, remove_digital_filter=True)
    return source


# Fixture : NMR spectra
@pytest.fixture(scope="function")
def NMR_source_1D_1H():
    path = os.path.join(data, 'nmrdata', 'bruker', 'tests', 'nmr',
                        'tpa')
    source = NDDataset.read_bruker_nmr(
            path, expno=10, remove_digital_filter=True)
    return source


@pytest.fixture(scope="function")
def NMR_source_2D():
    path = os.path.join(data, 'nmrdata', 'bruker', 'tests', 'nmr',
                        'bruker_2d')
    source = NDDataset.read_bruker_nmr(
            path, expno=1, remove_digital_filter=True)
    return source


# TODO: rationalise all this fixtures

