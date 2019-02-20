# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================
"""
2D-IRIR analysis example
---------------------
In this example, we perform the 2D IRIS analysis of CO adsorption on a sulfide
 catalyst.

"""

import os
from spectrochempy import Coord, NDDataset,IRIS, show

############################################################
# Upload dataset and add pressure coordinates

X = NDDataset.read_omnic(os.path.join('irdata', 'CO@Mo_Al2O3.SPG'))

p = [0.00300, 0.00400, 0.00900, 0.01400, 0.02100, 0.02600, 0.03600,
     0.05100, 0.09300, 0.15000, 0.20300, 0.30000, 0.40400, 0.50300,
     0.60200, 0.70200, 0.80100, 0.90500, 1.00400]

X.coordset[0] = Coord(p, title='pressure')

############################################################
# set the optimization parameters, perform the analysis
# and plot the results

param = {'epsRange': [-8, -1, 50],
         'lambdaRange': [-10, 1, 12],
         'guess':'random',
         'kernel':'langmuir'}

iris = IRIS(X[:, 2250.:1950.], param, verbose=True)
f = iris.transform()
X_hat = iris.inverse_transform()

iris.plotlcurve()
f[-5].plot(method='map')
X_hat[-5].plot()

show()