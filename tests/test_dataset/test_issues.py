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




# ======================================================================================================================

def test_fix_issue_20():
    
    # Description of bug #20
    # -----------------------
    # X = read_omnic(os.path.join('irdata', 'CO@Mo_Al2O3.SPG'))
    #
    # # slicing a NDDataset with an integer is OK for the coord:
    # X[:,100].x
    # Out[4]: Coord: [float64] cm^-1
    #
    # # but not with an integer array :
    # X[:,[100, 120]].x
    # Out[5]: Coord: [int32] unitless
    #
    # # on the other hand, the explicit slicing of the coord is OK !
    # X.x[[100,120]]
    # Out[6]: Coord: [float64] cm^-1
    
    X = read_omnic(os.path.join('irdata', 'CO@Mo_Al2O3.SPG'))
    assert X.__str__() == 'NDDataset: [float32] a.u. (shape: (y:19, x:3112))'

    # slicing a NDDataset with an integer is OK for the coord:
    assert X[:,100].x.__str__() == 'Coord: [float64] cm^-1'

    # The explicit slicing of the coord is OK !
    assert X.x[[100,120]].__str__() == 'Coord: [float64] cm^-1'
    
    # slicing the NDDataset with an integer array is also OK (fixed #20)
    assert X[:,[100, 120]].x.__str__() == X.x[[100,120]].__str__()
    
    