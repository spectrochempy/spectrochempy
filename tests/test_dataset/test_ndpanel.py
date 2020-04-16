# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================
"""


"""
import numpy as np
from copy import copy, deepcopy
from datetime import datetime
import pytest

from pint.errors import DimensionalityError

from spectrochempy.core.dataset.ndarray import NDArray
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.core.dataset.ndcoordset import CoordSet
from spectrochempy.core.dataset.ndpanel import NDPanel
from spectrochempy.core.dataset.nddataset import NDDataset

from spectrochempy.core import info_, debug_, warning_, error_, print_
from spectrochempy.units import ur, Quantity
from spectrochempy.utils import (SpectroChemPyWarning,
                                 INPLACE, MASKED,
                                 TYPE_INTEGER, TYPE_FLOAT)
from spectrochempy.utils.testing import (assert_equal, assert_array_equal,
                                         raises, catch_warnings,
                                         assert_approx_equal, RandomSeedContext)


def test_ndpanel_init():
    
    # void
    panel = NDPanel()
    assert panel.datasets == {}
    assert not panel.coords
    assert not panel.meta
    assert panel.name == panel.id
    assert panel.dims == []
    
    # without coordinates
    arr1 = np.random.rand(2,4)
    panel = NDPanel(arr1)
    assert panel.dims == ['x','y']
    assert panel.names == ['data_0']
    assert panel['data_0'].dims == ['y','x']
    assert panel.x is None
    with raises(AttributeError): # not in dims
        z = panel.z
    
    # without coordinates, multiple datasets, merge dimension
    arr2 = np.random.rand(3,4)
    na2 = NDArray(arr2, name='na2')
    panel = NDPanel(arr1, na2)          #  merge=True by default
    assert panel.dims == ['x','y','z']  #  then merge is automatic
    assert panel.names == ['data_0', 'na2']
    assert panel['data_0'].dims == ['y','x']
    assert panel['na2'].dims == ['z','x']

    # without coordinates, multiple datasets, dimensions not merged
    panel = NDPanel(arr1, na2, merge=False)
    assert panel.dims == ['u', 'x', 'y', 'z']  # dims are ordered by name
    assert panel.names == ['data_0', 'na2']
    assert panel['data_0'].dims == ['y','x']
    assert panel['na2'].dims == ['u','z']
    
    # with coordinates, multiple sets, dimensions not merged
    arr1 = np.random.rand(10,4)
    cx = Coord(np.arange(4), title='tx', units='s')
    cy = Coord(np.arange(10), title='ty', units='km')
    nd1 = NDDataset(arr1, coords=(cy, cx), name='arr1')
    
    arr2 = np.random.rand(15,4)
    cy2 = Coord(np.linspace(0,10,15)*1000., title='ty2', units='m')
    nd2 = NDDataset(arr2, coords=(cy2, cx), name='arr2')

    panel = NDPanel(nd1, nd2, merge=False)
    assert panel.dims == ['u', 'x', 'y', 'z']
    assert panel['arr1'].dims == ['y', 'x']
    assert panel['arr2'].dims == ['u', 'z']
    
    # with coordinates, multiple set, dimension merged
    panel = NDPanel(nd1, nd2)
    assert panel.dims == ['x','y','z']
    assert panel['arr1'].dims == ['y', 'x']
    assert panel['arr2'].dims == ['z', 'x']
    
    # dataset alignment during init (cy and cy2 can be aligned) but the title must be the same
    nd2.y.title = nd1.y.title
    panel = NDPanel(nd1, nd2, align='outer')
    assert panel.dims == ['x','y']
    assert panel['arr1'].dims == ['y', 'x']
    assert panel['arr2'].dims == ['y', 'x']
    
    info_('\n-----nd1')
    info_(nd1)
    info_('\n-----nd2')
    info_(nd2)
    # test print
    #print(nd1)
    #print(panel)
    #print_(panel)
    info_('\n\nouter')
    info_(panel)
    
    #TODO: check alignement errors
    panel = NDPanel(nd1, nd2, align='inner')
    assert panel.dims == ['x','y']
    assert panel['arr1'].dims == ['y', 'x']
    assert panel['arr2'].dims == ['y', 'x']
    info_('\n\ninner')
    info_(panel)
    
def test_ndpanel_fixture():
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
    return pnl
    
# ============================================================================
if __name__ == '__main__':
    pass

# end of module