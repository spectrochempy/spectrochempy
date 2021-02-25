# -*- coding: utf-8 -*-
# flake8: noqa
# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""
2D-IRIS analysis example
=========================

In this example, we perform the 2D IRIS analysis of CO adsorption on a sulfide catalyst.

"""

import spectrochempy as scp

########################################################################################################################
# ## Uploading dataset

X = scp.read('irdata/CO@Mo_Al2O3.SPG')

########################################################################################################################
# ``X`` has two coordinates: 
# * `wavenumbers` named "x" 
# * and `timestamps` (*i.e.,* the time of recording) named "y".
print(X.coordset)

########################################################################################################################
# ## Setting new coordinates
#
# Each experiments corresponding to a timestamp correspond to a given pressure of CO in the intrared cell. 
#
# Hence it would be interesting to replace the "useless" timestamps (``y``) by a pressure coordinates:

pressures = [0.00300, 0.00400, 0.00900, 0.01400, 0.02100, 0.02600, 0.03600,
             0.05100, 0.09300, 0.15000, 0.20300, 0.30000, 0.40400, 0.50300,
             0.60200, 0.70200, 0.80100, 0.90500, 1.00400]

""
c_pressures = scp.Coord(pressures, title='pressure', units='torr')

###############################################################################
# Now we can set multiple coordinates:

c_times = X.y.copy()  # the original coordinate
X.y = [c_times, c_pressures]
print(X.y)

###############################################################################
# By default, the current coordinate is the first one (here `c_times`). For example, it will be used for plotting:

prefs = X.preferences
prefs.figure.figsize = (7,3)
_ = X.plot(colorbar=True)
_ = X.plot_map(colorbar=True)

###############################################################################
# To seamlessly work with the second coordinates (pressures), we can change the default coordinate:

X.y.select(2)    # to select coordinate ``_2``
X.y.default

########################################################################################################################
# Let's now plot the spectral range of interest. The default coordinate is now used:
X_ = X[:, 2250.:1950.]
print(X_.y.default)
_ = X_.plot()
_ = X_.plot_map()

###############################################################################
# ## IRIS analysis without regularization

########################################################################################################################
# Perform IRIS without regularization (the verbose flag can be set to True to have information on the running process)
param = {
        'epsRange': [-8, -1, 50],
        'kernel': 'langmuir'
        }
iris = scp.IRIS(X_, param, verbose=False)

########################################################################################################################
# Plots the results
iris.plotdistribution()
_ = iris.plotmerit()

###############################################################################
# ## With regularization and a manual seach

########################################################################################################################
# Perform  IRIS with regularization, manual search
param = {
        'epsRange': [-8, -1, 50],
        'lambdaRange': [-10, 1, 12],
        'kernel': 'langmuir'
        }

iris = scp.IRIS(X_, param, verbose=False)
iris.plotlcurve()
iris.plotdistribution(-7)
_ = iris.plotmerit(-7)

###############################################################################
# ## Automatic search

########################################################################################################################
# Now try an automatic search of the regularization parameter:

param = {
        'epsRange': [-8, -1, 50],
        'lambdaRange': [-10, 1],
        'kernel': 'langmuir'
        }

iris = scp.IRIS(X_, param, verbose=False)
iris.plotlcurve()
iris.plotdistribution(-1)
_ = iris.plotmerit(-1)

""
# scp.show()  # uncomment to show plot if needed (not necessary in jupyter notebook)
