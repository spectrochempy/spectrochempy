# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

import sys
import os
import numpy as np
import pandas as pd
import pytest

# initialize a ipython session before calling spectrochempy
# ----------------------------------------------------------------------------------------------------------------------

from IPython.testing.globalipapp import start_ipython


@pytest.fixture(scope='session')
def session_ip():
    try:
        return start_ipython()
    except:
        return None


@pytest.fixture(scope='module')
def ip(session_ip):
    yield session_ip

from spectrochempy.core import app
try:
    # work only if spectrochempy is installed
    from spectrochempy.core import app
except ModuleNotFoundError as e:
    raise ModuleNotFoundError('You must install spectrochempy and its dependencies '
                              'before executing tests!')

# ======================================================================================================================
# FIXTURES
# ======================================================================================================================

from spectrochempy.core.dataset.ndarray import NDArray
from spectrochempy.core.dataset.ndcomplex import NDComplexArray
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndpanel import NDPanel
from spectrochempy.core.dataset.ndcoordset import CoordSet
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.utils.testing import RandomSeedContext
from spectrochempy.core import general_preferences as prefs


# Handle command line argument for spectrochempy
# ----------------------------------------------------------------------------------------------------------------------

def pytest_cmdline_preparse(config, args):
    for item in args[:]:
        for k in list(app.flags.keys()):
            if item.startswith("--" + k) or k in ['--help', '--help-all']:
                args.remove(item)
            continue
        for k in list(app.aliases.keys()):
            if item.startswith("-" + k) or k in ['h', ]:
                args.remove.append(item)


# create reference arrays
# ----------------------------------------------------------------------------------------------------------------------

with RandomSeedContext(12345):
    ref_data = 10. * np.random.random((10, 8)) - 5.
    ref3d_data = 10. * np.random.random((10, 100, 3)) - 5.
    ref3d_2_data = np.random.random((9, 50, 4))

ref_mask = ref_data < -4
ref3d_mask = ref3d_data < -3
ref3d_2_mask = ref3d_2_data < -2


# Fixtures: some NDArray's
# ----------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="function")
def refarray():
    return ref_data.copy()


@pytest.fixture(scope="function")
def refmask():
    return ref_mask.copy()


@pytest.fixture(scope="function")
def ndarray():
    # return a simple ndarray with some data
    return NDArray(ref_data, copy=True).copy()


@pytest.fixture(scope="function")
def ndarrayunit():
    # return a simple ndarray with some data and units
    return NDArray(ref_data, units='m/s', copy=True).copy()


@pytest.fixture(scope="function")
def ndarraymask():
    # return a simple ndarray with some data and units
    return NDArray(ref_data, mask=ref_mask, units='m/s', copy=True).copy()


# Fixtures: Some NDComplex's array
# ----------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="function")
def ndarraycplx():
    # return a complex ndarray
    return NDComplexArray(ref_data, units='m/s', dtype=np.complex128, copy=True).copy()


@pytest.fixture(scope="function")
def ndarrayquaternion():
    # return a quaternion ndarray
    return NDComplexArray(ref_data, units='m/s', dtype=np.quaternion, copy=True).copy()


# Fixtures: Some NDDatasets
# ----------------------------------------------------------------------------------------------------------------------

coord0_ = Coord(data=np.linspace(4000., 1000., 10), labels=list('abcdefghij'), units="cm^-1", title='wavenumber')
@pytest.fixture(scope="function")
def coord0():
    return coord0_.copy()

coord1_ = Coord(data=np.linspace(0., 60., 100), units="s", title='time-on-stream')
@pytest.fixture(scope="function")
def coord1():
    return coord1_.copy()

coord2_ = Coord(data=np.linspace(200., 300., 3), labels=['cold', 'normal', 'hot'], units="K", title='temperature')
@pytest.fixture(scope="function")
def coord2():
    return coord2_.copy()

coord2b_ = Coord(data=np.linspace(1., 20., 3), labels=['low', 'medium', 'high'], units="tesla", title='magnetic field')
@pytest.fixture(scope="function")
def coord2b():
    return coord2b_.copy()

coord0_2_ = Coord(data=np.linspace(4000., 1000., 9), labels=list('abcdefghi'), units="cm^-1", title='wavenumber')
@pytest.fixture(scope="function")
def coord0_2():
    return coord0_2_.copy()

coord1_2_ = Coord(data=np.linspace(0., 60., 50), units="s", title='time-on-stream')
@pytest.fixture(scope="function")
def coord1_2():
    return coord1_2_.copy()

coord2_2_ = Coord(data=np.linspace(200., 1000., 4), labels=['cold', 'normal', 'hot', 'veryhot'], units="K",
                 title='temperature')
@pytest.fixture(scope="function")
def coord2_2():
    return coord2_2_.copy()

@pytest.fixture(scope="function")
def nd1d():
    # a simple ddataset
    return NDDataset(ref_data[:, 1].squeeze()).copy()


@pytest.fixture(scope="function")
def nd2d():
    # a simple 2D ndarrays
    return NDDataset(ref_data).copy()


@pytest.fixture(scope="function")
def ref_ds():
    # a dataset with coordinates
    return ref3d_data.copy()


@pytest.fixture(scope="function")
def ds1():
    # a dataset with coordinates
    return NDDataset(ref3d_data, coords=[coord0_, coord1_, coord2_], title='Absorbance', units='absorbance').copy()


@pytest.fixture(scope="function")
def ds2():
    # another dataset
    return NDDataset(ref3d_2_data, coords=[coord0_2_, coord1_2_, coord2_2_], title='Absorbance', units='absorbance').copy()


@pytest.fixture(scope="function")
def dsm():
    # dataset with coords containing several axis and a mask

    coordmultiple = CoordSet(coord2_, coord2b_)
    return NDDataset(ref3d_data, coords=[coord0_, coord1_, coordmultiple], mask=ref3d_mask, title='Absorbance',
                     units='absorbance').copy()

# NDPanel
@pytest.fixture(scope="function")
def pnl():
    with RandomSeedContext(12345):
        arr1 = np.random.rand(10,20)
        arr2 = np.random.rand(20,12)
    cy1 = Coord(np.arange(10), title='ty', units='s')
    cy2 = Coord(np.arange(12), title='ty', units='s')
    cx = Coord(np.arange(20), title='tx', units='km')
    nd1 = NDDataset(arr1, coords=(cy1, cx), name='arr1')
    nd2 = NDDataset(arr2, coords=(cy2, cx), dims=['x', 'y'], name='arr2')
    pnl = NDPanel(nd1, nd2)
    assert pnl.dims == ['x', 'y']
    return pnl.copy()

# Fixtures:  IR spectra (SPG)
# ----------------------------------------------------------------------------------------------------------------------

directory = prefs.datadir
dataset = NDDataset.read_omnic(os.path.join(directory, 'irdata', 'nh4y-activation.spg'))


@pytest.fixture(scope="function")
def IR_dataset_2D():
    return dataset.copy()


@pytest.fixture(scope="function")
def IR_dataset_1D():
    return dataset[0].squeeze().copy()


@pytest.fixture(scope="function")
def IR_scp_1():
    directory = prefs.datadir
    dataset = NDDataset.load(os.path.join(directory, 'irdata', 'nh4.scp'))
    return dataset.copy()


# Fixture: NMR spectra
# ----------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="function")
def NMR_dataset_1D():
    directory = prefs.datadir
    path = os.path.join(directory, 'nmrdata', 'bruker', 'tests', 'nmr', 'bruker_1d')
    dataset = NDDataset.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    return dataset.copy()


@pytest.fixture(scope="function")
def NMR_dataset_1D_1H():
    directory = prefs.datadir
    path = os.path.join(directory, 'nmrdata', 'bruker', 'tests', 'nmr', 'tpa')
    dataset = NDDataset.read_bruker_nmr(path, expno=10, remove_digital_filter=True)
    return dataset.copy()


@pytest.fixture(scope="function")
def NMR_dataset_2D():
    directory = prefs.datadir
    path = os.path.join(directory, 'nmrdata', 'bruker', 'tests', 'nmr', 'bruker_2d')
    dataset = NDDataset.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    return dataset.copy()


# Some panda structure for dataset initialization
# ----------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="function")
def series():
    with RandomSeedContext(2345):
        arr = pd.Series(np.random.randn(4), index=np.arange(4) * 10.)
    arr.index.name = 'un nom'
    return arr.copy()


@pytest.fixture(scope="function")
def dataframe():
    with RandomSeedContext(23451):
        arr = pd.DataFrame(np.random.randn(6, 4), index=np.arange(6) * 10., columns=np.arange(4) * 10.)
    for ax, name in zip(arr.axes, ['time', 'temperature']):
        ax.name = name
    return arr.copy()

# Panel is removed from Panda
# @pytest.fixture(scope="function")
# def panel():
#     shape = (7, 6, 5)
#     with RandomSeedContext(23452):
#         arr = pd.Panel(data = np.random.randn(*shape), items=np.arange(shape[0]) * 10.,
#                        major_axis=np.arange(shape[1]) * 10.,
#                        minor_axis=np.arange(shape[2]) * 10.)
#     for ax, name in zip(arr.axes, ['axe0', 'axe1', 'axe2']):
#         ax.name = name
#     return arr.copy()

# GUI Fixtures
# ----------------------------------------------------------------------------------------------------------------------

# from pyqtgraph import mkQApp

# @pytest.fixture(scope="module")
# def app():
#    return mkQApp()
