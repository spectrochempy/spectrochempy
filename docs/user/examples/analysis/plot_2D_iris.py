# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================
"""
2D-IRIS analysis example
------------------------
In this example, we perform the 2D IRIS analysis of CO adsorption on a sulfide
 catalyst.

"""

import spectrochempy as scp
import matplotlib.pyplot as plt
import os

########################################################################################################################
# Upload dataset and add pressure coordinates

X = scp.NDDataset.read_omnic(os.path.join('irdata', 'CO@Mo_Al2O3.SPG'))

p = [0.00300, 0.00400, 0.00900, 0.01400, 0.02100, 0.02600, 0.03600,
     0.05100, 0.09300, 0.15000, 0.20300, 0.30000, 0.40400, 0.50300,
     0.60200, 0.70200, 0.80100, 0.90500, 1.00400]

X.coords.update(y=scp.Coord(p, title='pressure', units='torr'))
# Using the `update` method is mandatory because it will preserve the name.
# Indeed, setting using X.coords[0] = Coord(...) fails unless name is specified: Coord(..., name='y')

###############################
# Select and plot the spectral range of interest
X_ = X[:, 2250.:1950.]
X_.plot()

########################################################################################################################
# Perform IRIS without regularization and plots results
param = {'epsRange': [-8, -1, 50],
         'kernel':'langmuir'}

iris = scp.IRIS(X_, param, verbose=True)
iris.plotdistribution()
iris.plotmerit()

########################################################################################################################
# Perform  IRIS with regularization, manual search
param = {'epsRange': [-8, -1, 50],
         'lambdaRange': [-10, 1, 12],
         'kernel':'langmuir'}

iris = scp.IRIS(X_, param, verbose=True)
iris.plotlcurve()
iris.plotdistribution(-7)
iris.plotmerit(-7)

########################################################################################################################
# Now try an automatic search of the regularization parameter:

param = {'epsRange': [-8, -1, 50],
         'lambdaRange': [-10, 1],
         'kernel':'langmuir'}

iris = scp.IRIS(X_, param, verbose=True)
iris.plotlcurve()
iris.plotdistribution(-1)
iris.plotmerit(-1)

#plt.show() # uncomment to show plot if needed()