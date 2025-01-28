# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
import os

import numpy as np

from spectrochempy import read_omnic
from spectrochempy.core.units import ur
from spectrochempy.utils.plots import show

typequaternion = np.dtype(np.quaternion)


# ======================================================================================
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

    X = read_omnic(os.path.join("irdata", "CO@Mo_Al2O3.SPG"))
    assert X.__str__() == "NDDataset: [float64] a.u. (shape: (y:19, x:3112))"

    # slicing a NDDataset with an integer is OK for the coord:
    assert X[:, 100].x.__str__() in [
        "Coord: [float64] cm⁻¹ (size: 1)",
        "Coord: [float64] 1/cm (size: 1)",
    ]

    # The explicit slicing of the coord is OK !
    assert X.x[[100, 120]].__str__() in [
        "Coord: [float64] cm⁻¹ (size: 2)",
        "Coord: [float64] 1/cm (size: 2)",
    ]

    # slicing the NDDataset with an integer array is also OK (fixed #20)
    assert X[:, [100, 120]].x.__str__() == X.x[[100, 120]].__str__()


def test_fix_issue_58():
    X = read_omnic(os.path.join("irdata", "CO@Mo_Al2O3.SPG"))
    X.y = X.y - X.y[0]  # subtract the acquisition timestamp of the first spectrum
    X.y = X.y.to("minute")  # convert to minutes
    assert X.y.units == ur.minute
    X.y += 2  # add 2 minutes
    assert X.y.units == ur.minute
    assert X.y[0].data == [2]  # check that the addition is correctly done 2 min


def test_fix_issue_186():
    import spectrochempy as scp

    da = scp.read_omnic(os.path.join("irdata", "nh4y-activation.spg"))
    da -= da.min()
    da.plot()

    dt = da.to("transmittance")
    dt.plot()

    dt.ito("absolute_transmittance")
    dt.plot()

    da = dt.to("absorbance")
    da.plot()

    dt.ito("transmittance")
    dt.plot()

    da = dt.to("absorbance")
    da.plot()

    show()


def test_issue_668():
    import spectrochempy as scp

    s = scp.read("irdata/CO@Mo_Al2O3.SPG")[-1, 2300.0:1900.0]

    # print(s[:, [0, 1, 2, 7, 15]].x.data)
    # # wrongly returns [    2300     2296     2293     2289     2285]
    # print(s[:, [0, 1, 5, 10,
    #             15]].x.data)
    # # despite the change of inner index, it returns the same as above...
    # print(s.x[[0, 1, 2, 7,
    #            15]].data)
    # # returns the expected values of the first line above[    2300     2299     2298     2293     2285]

    assert np.all(s[:, [0, 1, 2, 7, 15]].x.data == s.x[[0, 1, 2, 7, 15]].data)
