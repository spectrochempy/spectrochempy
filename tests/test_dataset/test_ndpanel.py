# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
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

from spectrochempy.core import log
from spectrochempy.units import ur, Quantity
from spectrochempy.utils import (SpectroChemPyWarning, info_, debug_,
                                 INPLACE, MASKED,
                                 TYPE_INTEGER, TYPE_FLOAT)
from spectrochempy.utils.testing import (assert_equal, assert_array_equal,
                                         raises, catch_warnings,
                                         assert_approx_equal)


def test_ndpanel_init():
    
    # void
    panel = NDPanel()
    assert panel.datasets == {}
    assert not panel.coords
    assert not panel.meta
    assert panel.name == panel.id
    assert panel.dims == []
    assert panel.dim_properties == {}
    
    # without coordinates
    arr1 = np.random.rand(2,4)
    panel = NDPanel(arr1)
    assert panel.dims == ['x','y']
    assert panel.dim_properties == {'x':[4, None, None, None, None], 'y':[2, None, None, None, None]}
    assert panel.names == ['data_0']
    assert panel['data_0'].dims == ['y','x']
    assert panel.x is None
    with raises(AttributeError): # not in dims
        z = panel.z
    
    # without coordinates, multiple datasets, merge dimension
    arr2 = np.random.rand(3,4)
    na2 = NDArray(arr2, name='na2')
    panel = NDPanel(arr1, na2)
    assert panel.dims == ['x','y','z']  # by default alignement is automatic
    assert panel.names == ['data_0', 'na2']
    assert panel['data_0'].dims == ['y','x']
    assert panel['na2'].dims == ['z','x']

    # without coordinates, muliple datasets, dimensions not merged
    panel = NDPanel(arr1, na2, merge=False)
    assert panel.dims == ['u', 'x', 'y', 'z']  # dims are ordered by name
    assert panel.names == ['data_0', 'na2']
    assert panel['data_0'].dims == ['y','x']
    assert panel['na2'].dims == ['u','z']
    
    # with coordinates, multiple sets, dimensions not merged
    arr1 = np.random.rand(2,4)
    cx = Coord(np.arange(4), title='tx', units='s')
    cy = Coord(np.arange(2), title='ty', units='km')
    nd1 = NDDataset(arr1, coords=(cy, cx), name='arr1')
    
    arr2 = np.random.rand(3,4)
    cy2 = Coord(np.arange(3), title='ty2', units='eV')
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
    
    


# ============================================================================
if __name__ == '__main__':
    pass

# end of module