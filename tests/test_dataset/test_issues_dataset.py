# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import os

import numpy as np

from spectrochempy import read_omnic
from spectrochempy.units import ur

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
    assert X[:, 100].x.__str__() == 'Coord: [float64] cm^-1'

    # The explicit slicing of the coord is OK !
    assert X.x[[100, 120]].__str__() == 'Coord: [float64] cm^-1'

    # slicing the NDDataset with an integer array is also OK (fixed #20)
    assert X[:, [100, 120]].x.__str__() == X.x[[100, 120]].__str__()


def test_fix_issue_58():
    X = read_omnic(os.path.join('irdata', 'CO@Mo_Al2O3.SPG'))
    X.y = X.y - X.y[0]  # subtract the acquisition timestamp of the first spectrum
    X.y = X.y.to('minute')  # convert to minutes
    assert X.y.units == ur.minute
    X.y += 2  # add 2 minutes
    assert X.y.units == ur.minute
    assert X.y[0].data == [2]  # check that the addition is correctly done 2 min
