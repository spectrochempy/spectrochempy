# -*- coding: utf-8 -*-
# %%
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
SIMPLISMA example
-----------------
In this example, we perform the PCA dimensionality reduction of a spectra
dataset

"""
# %%
# Import the spectrochempy API package (and the SIMPLISMA model independently)
import spectrochempy as scp

# %%
# Load a matlab datasets
print("Dataset (Jaumot et al., Chemometr. Intell. Lab. 76 (2005) 101-110)):")
lnd = scp.read_matlab("matlabdata/als2004dataset.MAT", merge=False)
for mat in lnd:
    print("    " + mat.name, str(mat.shape))

ds = lnd[-1]
_ = ds.plot()

# %%
# Add some metadata for a nicer display
ds.title = "absorbance"
ds.units = "absorbance"
ds.set_coordset(None, None)
ds.y.title = "elution time"
ds.x.title = "wavelength"
ds.y.units = "hours"
ds.x.units = "nm"

# %%
# Fit the SIMPLISMA model
print("Fit SIMPLISMA on {}\n".format(ds.name))
simpl = scp.SIMPLISMA(max_components=20, tol=0.2, noise=3, log_level="INFO")
simpl.fit(ds)

# %%
# Plot concentration
_ = simpl.C.T.plot(title="Concentration")

# %%
# Plot components (St)

# sphinx_gallery_thumbnail_number = 3

_ = simpl.components.plot(title="Pure profiles")

# %%
# Show the plot of merit
# after reconstruction oto the original data space
simpl.plotmerit(offset=0, nb_traces=5)

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
