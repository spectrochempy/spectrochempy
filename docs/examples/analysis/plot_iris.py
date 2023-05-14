# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
2D-IRIS analysis example
=========================

In this example, we perform the 2D IRIS analysis of CO adsorption on a sulfide catalyst.
"""
# %%
import spectrochempy as scp

# %%
# Uploading dataset
# -----------------
X = scp.read("irdata/CO@Mo_Al2O3.SPG")

# %%
# `X` has two coordinates:
# * `wavenumbers` named "x"
# * and `timestamps` (*i.e.,* the time of recording) named "y".

print(X.coordset)

# %%
# Setting new coordinates
# -----------------------
#
# The `y` coordinates of the dataset is the acquisition timestamp.
# However, each spectrum has been recorded with a given pressure of CO
# in the infrared cell.
#
# Hence, it would be interesting to add pressure coordinates to the `y` dimension:

pressures = [
    0.003,
    0.004,
    0.009,
    0.014,
    0.021,
    0.026,
    0.036,
    0.051,
    0.093,
    0.150,
    0.203,
    0.300,
    0.404,
    0.503,
    0.602,
    0.702,
    0.801,
    0.905,
    1.004,
]

c_pressures = scp.Coord(pressures, title="pressure", units="torr")

# %%
# Now we can set multiple coordinates:

c_times = X.y.copy()  # the original coordinate
X.y = [c_times, c_pressures]
print(X.y)

# %%
# To get a detailed
# a rich display of these coordinates. In a jupyter notebook, just type:

X.coordset

# %%
# By default, the current coordinate is the first one (here `c_times` ).
# For example, it will be used by default for
# plotting:

prefs = X.preferences
prefs.figure.figsize = (7, 3)
_ = X.plot(colorbar=True)
_ = X.plot_map(colorbar=True)

# %%
# To seamlessly work with the second coordinates (pressures), we can change the default
# coordinate:

X.y.select(2)  # to select coordinate `_2`
X.y.default

# %%
# Let's now plot the spectral range of interest. The default coordinate is now used:
X_ = X[:, 2250.0:1950.0]
print(X_.y.default)
_ = X_.plot()
_ = X_.plot_map()

# %%
# IRIS analysis without regularization
# ------------------------------------
# Perform IRIS without regularization (the loglevel can be set to `INFO` to have
# information on the running process)
iris1 = scp.IRIS(log_level="INFO")

# %%
# first we compute the kernel object
K = scp.IrisKernel(X_, "langmuir", q=[-8, -1, 50])

# %%
# The actual kernel is given by the `kernel` attribute
kernel = K.kernel
kernel

# %%
# Now we fit the model - we can pass either the Kernel object or the kernel NDDataset
iris1.fit(X_, K)

# %%
# Plots the results
iris1.plotdistribution()
_ = iris1.plotmerit()

# %%
# With regularization and a manual search
# ---------------------------------------
# Perform  IRIS with regularization, manual search
iris2 = scp.IRIS(reg_par=[-10, 1, 12])

# %%
# We keep the same kernel object as previously - performs the fit.
iris2.fit(X_, K)

iris2.plotlcurve(title="L curve, manual search")
iris2.plotdistribution(-7)
_ = iris2.plotmerit(-7)

# %%
# Automatic search
# ----------------
# %%
# Now try an automatic search of the regularization parameter:

iris3 = scp.IRIS(reg_par=[-10, 1])
iris3.fit(X_, K)
iris3.plotlcurve(title="L curve, automated search")

# %%
# The data corresponding to the largest curvature of the L-curve
# are at the second last position of output data.

# sphinx_gallery_thumbnail_number = 11

iris3.plotdistribution(-2)
_ = iris3.plotmerit(-2)

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
