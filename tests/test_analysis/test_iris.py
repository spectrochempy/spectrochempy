# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

import os
import numpy as np

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.core.analysis.iris import IRIS
from spectrochempy.utils import show
from spectrochempy.core import info_

def test_IRIS():
    print('start test IRIS')

    X = NDDataset.read_omnic(os.path.join('irdata', 'CO@Mo_Al2O3.SPG'))

    p = [0.00300, 0.00400, 0.00900, 0.01400, 0.02100, 0.02600, 0.03600,
         0.05100, 0.09300, 0.15000, 0.20300, 0.30000, 0.40400, 0.50300,
         0.60200, 0.70200, 0.80100, 0.90500, 1.00400]

    X.coords.update(y=Coord(p, title='pressure', units='torr'))
    # Using the `update` method is mandatory because it will preserve the name.
    # Indeed, setting using X.coords[0] = Coord(...) fails unless name is specified: Coord(..., name='y')

    ############################################################
    # set the optimization parameters, perform the analysis
    # and plot the results

    param = {'epsRange': [-8, -1, 20],
             'lambdaRange': [-7, -5, 3],
             'kernel': 'langmuir'}

    X_ = X[:, 2250.:1950.]
    X_.plot()

    iris = IRIS(X_, param, verbose=True)

    f = iris.f
    X_hat = iris.reconstruct()

    iris.plotlcurve(scale='ln')
    f[0].plot(method='map', plottitle=True)
    X_hat[0].plot(plottitle=True)

    show()


# =============================================================================
if __name__ == '__main__':
    pass
