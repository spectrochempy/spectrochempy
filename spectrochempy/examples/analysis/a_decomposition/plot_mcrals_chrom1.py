# -*- coding: utf-8 -*-
# %%
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
MCR-ALS example (adapted from Jaumot et al. 2005)
=================================================

In this example, we perform the MCR ALS optimization of a dataset
corresponding to a HPLC-DAD run, from :cite:t:`jaumot:2005` and :cite:t:`jaumot:2015`\ .

This dataset (and others) can be downloaded from the
`Multivariate Curve Resolution Homepage
<https://mcrals.wordpress.com/download/example-data-sets>`_\ .

For the user convenience, this dataset is present in the test data directory
`scp.preferences.datadir` of SpectroChemPy as ``als2004dataset.MAT``\ .
"""
# %%
# Import the spectrochempy API package
import spectrochempy as scp

# %%
# Loading the example dataset
# ---------------------------
#
# The file type (matlab) is inferred from the extension ``.mat``\ , so we
# can use the generic API function `read`\ .  Alternatively, one can be more
# specific by using the `read_matlab` function. Both have exactly the same behavior.
datasets = scp.read("matlabdata/als2004dataset.MAT")

# %%
# As the ``.mat`` file contains 6 matrices, 6 `NDDataset` objects are returned.
print("NDDataset names:")
for ds in datasets:
    print(f"{ds.name} : {ds.shape}")

# %%
# We are interested in the last dataset (``"m1"``\ ) that contains a single HPLS-DAD run
# ``(51x96)``  dataset.
#
# As usual, the 51 rows correspond to the ``time axis`` of the HPLC run, and the 96
# columns to the ``wavelength`` axis of the UV spectra. The original dataset does not
# contain information as to the actual time and wavelength coordinates.
#
# MCR-ALS needs also an initial guess for either concentration profiles or pure spectra
# concentration profiles.
# The 4th dataset in the example (``"spure"``) contains (4x96) guess of spectral
# profiles.
#
# The experimental data as :math:`X` (``X``) and the ``guess`` are thus:
X = datasets[-1]
guess = datasets[3]

# %%
# Plot of X and of the guess:
_ = X.plot()
_ = guess.plot()

# %%
# Create a MCR-ALS object
# -----------------------
#
# We first create a MCR-ALS object named here ``mcr``\ .
#
# The `log_level` option can be set to ``"INFO"`` to get verbose ouput of
# the MCR-ALS optimization steps.
mcr = scp.MCRALS(log_level="INFO")

# %%
# Fit the MCR-ALS model
# -----------------------
#
# Then we execute the optimization process using the `fit` method with
# the ``X`` and ``guess`` dataset as input arguments.
_ = mcr.fit(X, guess)

# %%
# Plotting the results
# --------------------
#
# The optimization has converged. We can get the concentration :math:`C` (C) and
# pure spectra profiles :math:`S^T` (St) and plot them

_ = mcr.C.T.plot()
_ = mcr.St.plot()

# %%
# Finally, plots the reconstructed dataset (:math:`\hat{X} = C.S^T`\ )
# *vs.* original dataset
# (:math:`X`\ ) as well as the residuals (:math:`E`\ ) for few spectra.
#
# The fit is good and comparable to the original paper (:cite:t:`jaumot:2005`).
_ = mcr.plotmerit(nb_traces=5)

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
