import numpy as np
import pandas as pd

from pint import DimensionalityError


from tests.utils import (assert_equal, assert_array_equal,
                         assert_array_almost_equal, assert_equal_units,
                         raises)
from tests.utils import NumpyRNGContext

import pytest
import numpy as np
import os


from spectrochempy.api import NDDataset, Axes, Axis, data_dir

# An IR spectra
@pytest.fixture(scope="function")
def IR_source_1():
    source = NDDataset.read_omnic(
        os.path.join(data_dir, 'irdata', 'NH4Y-activation.SPG'))
    return source


@pytest.fixture(scope="function")
def IR_scp_1():
    source = NDDataset.read_omnic(
        os.path.join(data_dir, 'irdata', 'NH4Y-activation.SPG'))
    return source


# some datasets
@pytest.fixture()
def ds1():
    with NumpyRNGContext(12345):
        dx = np.random.random((10, 100, 3))
        # make complex along first dimension
        is_complex = [False, False, False]  # TODO: check with complex

    axe0 = Axis(coords=np.linspace(4000., 1000., 10),
                labels='a b c d e f g h i j'.split(),
                units="cm^-1",
                title='wavenumber')

    axe1 = Axis(coords=np.linspace(0., 60., 100),
                units="s",
                title='time-on-stream')

    axe2 = Axis(coords=np.linspace(200., 300., 3),
                labels=['cold', 'normal', 'hot'],
                units="K",
                title='temperature')

    da = NDDataset(dx,
                   is_complex=is_complex,
                   axes=[axe0, axe1, axe2],
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

    axe0 = Axis(coords=np.linspace(4000., 1000., 9),
                labels='a b c d e f g h i'.split(),
                units="cm^-1",
                title='wavenumber')

    axe1 = Axis(coords=np.linspace(0., 60., 50),
                units="s",
                title='time-on-stream')

    axe2 = Axis(coords=np.linspace(200., 1000., 4),
                labels=['cold', 'normal', 'hot', 'veryhot'],
                units="K",
                title='temperature')

    da = NDDataset(dx,
                   is_complex=is_complex,
                   axes=[axe0, axe1, axe2],
                   title='Absorbance',
                   units='absorbance',
                   uncertainty=dx * 0.1,
                   )
    return da.copy()


@pytest.fixture()
def dsm():  # dataset with axes containing several axis

    with NumpyRNGContext(12345):
        dx = np.random.random((9, 50))
        # make complex along first dimension
        is_complex = [False, False]  # TODO: check with complex

    axe0 = Axis(coords=np.linspace(4000., 1000., 9),
                labels='a b c d e f g h i'.split(),
                units="cm^-1",
                title='wavenumber')

    axe11 = Axis(coords=np.linspace(0., 60., 50),
                 units="s",
                 title='time-on-stream')

    axe12 = Axis(coords=np.logspace(1., 4., 50),
                 units="K",
                 title='temperature')

    axemultiple = Axes(axe11, axe12)
    da = NDDataset(dx,
                   is_complex=is_complex,
                   axes=[axe0, axemultiple],
                   title='Absorbance',
                   units='absorbance',
                   uncertainty=dx * 0.1,
                   )
    return da.copy()

