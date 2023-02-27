# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
MCR-ALS optimization example with kinetic constraints
===========================================================

In this example, we perform the MCR ALS optimization of the UV-vis of spectra resulting from a three-component
reaction A -> B -> C which was investigated by UV–Vis spectroscopy. Full details of the reaction and data acquisition
conditions can be found in ...

The data has been provided by the Biosystems Data Analysis Group of the University of Amsterdam [32].
Here we will focus on a signlme run. For the user convenience, this dataset is present in the spectrochempy_data
 as 'METING9.mat' .

"""
# %%
import spectrochempy as scp
import numpy as np

# %%
# Load the dataset
ds = scp.read_matlab("matlabdata/METING9.MAT")

# %%
#  The first array contains the time in seconds since the start of the reaction (t=0). The first column of the matrix
#  contains the wavelength axis and the remaining columns are the measured UV-VIS spectra (wavelengths x timepoints)
print("\n NDDataset names: " + str([d.name for d in ds]))

# %%
# We load the experimental spectra (in `ds[1]`) and add the `y` (time) a,d `x` (wavelength) coordinates :

D = scp.NDDataset(ds[1][:, 1:].data.T)
D.y = scp.Coord(ds[0].data.squeeze(), title="time") / 60
D.x = scp.Coord(ds[1][:, 0].data.squeeze(), title="wavelength / cm$^{-1}$")

_ = D.plot()
# %%
# A first estimate of the concentrations can be obtained by EFA
# %%
print("compute EFA...")
efa = scp.EFA()
efa.fitD[:, 300.0:500.0]
efa.used_components = 3
C0 = efa.transform()
C0 = C0 / C0.max(dim="y") * 5.0
_ = C0.T.plot()

# %%
# We can get a better estimate of the concentration (C) and pure spectra profiles (St)
# by soft MCR-ALS:
# %%
scp.set_loglevel("INFO")
mcr_1 = scp.MCRALS()
_ = mcr_1.fit(D, C0)

mcr_1.C.T.plot()
mcr_1.St.plot()

# %%
# Kinetic constraints can be added, i.e. imposing that the concentratuion profiles obay
# a kinetic model.To do so we first define an ActionMAssKinetics object with
# roghly estimated rate constants:
# %%

reactions = ("A -> B", "B -> C")
species_concentrations = {"A": 5.0, "B": 0.0, "C": 0.0}
k0 = np.array(((0.5, 0.0), (0.05, 0.0)))

kin = scp.ActionMassKinetics(reactions, species_concentrations, k0)

# %%
# The concentration profile obtained with this approximate model can be computed and
# compared with thoise of the soft MCR-ALS:
# %%

Ckin, meta = kin.integrate(D.y.data)
_ = mcr_1.C.T.plot(linestyle="-", cmap=None)
_ = Ckin.T.plot(clear=False, cmap=None)

# %%
# Even though very approximate, the same values can be used to run a hard-soft MCR-ALS:
# %%
X = D[:, 300.0:500.0]
param_to_optimize = {"k[0].A": 0.5, "k[1].A": 0.05}
mcr_2 = scp.MCRALS()
mcr_2.hardConc = [0, 1, 2]
mcr_2.getConc = kin.fit_to_concentrations
mcr_2.argsGetConc = ([0, 1, 2], [0, 1, 2], param_to_optimize)
mcr_2.fit(X, Ckin)

# %%
# Now, let's compare the concentration profile of the hard-soft MCAR-ALS (C = X(C$_{kin}^+$ X)$^+$) with
# that of the optimized kinetic model (C$_{kin}$):
# %%
_ = mcr_2.C.T.plot()
_ = mcr_2.C_hard.T.plot(clear=False)

# %%
# Finally, let's plot the pure spectra profiles St, and the
#  reconstructed dataset  (X_hat = C St) vs original dataset (X)
# and residuals.
# %%
_ = mcr_2.St.plot()
X_hat = mcr_2.inverse_transform()
_ = mcr_2.plotmerit(X, X_hat)

scp.show()  # uncomment to show plot if needed (not necessary in jupyter notebook)
