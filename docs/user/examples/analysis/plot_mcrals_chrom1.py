# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================
"""
MCR ALS optimization example (original example from Jaumot)
------------------------------------------------------------
In this example, we perform the MCR ALS optimization of a dataset 
corresponding to a HPLC-DAD run, from Jaumot et al. Chemolab, 76 (2005), 
pp. 101-110 and Jaumot et al. Chemolab, 140 (2015) pp. 1-12.
This dataset (and others) can be loaded from the "Multiovariate Curve Resolution Homepage"
at https://mcrals.wordpress.com/download/example-data-sets.

For the user convenience, this dataset is present in the 'datadir'
of spectrochempy in 'als2004dataset.MAT'

"""

import spectrochempy as scp

############################################################
# Load the dataset

datasets = scp.read_matlab("matlabdata/als2004dataset.MAT")

# As the .mat file contains 6 matrices, 6 NDDataset objects are returned:

print('\n NDDataset names: ' + str([ds.name for ds in datasets]))

##########################################################
# We are interested in the first dataset ('m1') that contains a single HPLS-DAD run (51x96)  dataset.
# As usual, the 51 rows correspond to the 'time axis' of the HPLC run, and the 96 columns to the 'wavelength' axis
# of the UV spectra. The original dataset does not contain information as to the actual time and wavelength coordinates.
#
# MCR-ALS needs also an initial guess for either concentration profiles or pure spectra concentration profiles.
# The second dataset 'spure' is a (4x96) guess of spectral profiles.
#
# Load the experimental data as X and the guess:
X = datasets[0]
guess = datasets[1]

##############################################################
# Create a MCR-ALS object with the default settings and the verbose optionto get a summary of optimization steps:
mcr = scp.MCRALS(X, guess, verbose=True)

##############################################################
# The optimization has converged. We can get the concentration (C) and pure spectra profiles (St)
# and plot them


mcr.C.T.plot()
mcr.St.plot()


###############################################################
# Finally, plots the reconstructed dataset  (X_hat = C St) vs original dataset (X)
# an residuals. The fit is good and comparable with the original paper.
X_hat = mcr.plotmerit()


#show() # uncomment to show plot if needed()