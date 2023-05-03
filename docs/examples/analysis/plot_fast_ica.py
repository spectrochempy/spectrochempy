# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
NDDataset FastICA example
=========================

"""
# %%
# Import the spectrochempy API package
import spectrochempy as scp

# %%
# Prepare the dataset to NMF factorize
# -------------------------------------

# %%
# Here we use a dataset from Jaumot et al., Chemometr. Intell. Lab. 76 (2005) 101-110)

X = scp.read("matlabdata/als2004dataset.MAT")[-1]

X.title = "absorbance"
X.units = "absorbance"
X.set_coordset(None, None)
X.y.title = "elution time"
X.x.title = "wavelength"
X.y.units = "hours"
X.x.units = "cm^-1"


# %%
# Create and fit a FastICA object
# -------------------
#
# As argument of the object constructor we define log_level to ``"INFO"`` to
# obtain verbose output during fit, and we set the number of component to use at 4.
ica = scp.FastICA(n_components=4)
_ = ica.fit(X)

# Get the mixing system and source spectral profiles
# --------------------------------------------------
#
# The mixing system :math:`A` and the source spectral profiles :math:`S^T` can
# be obtained as follows (the Sklearn equivalents - also valid with Scpy - are
# indicated as comments
A = ica.A  # or model.transform()
St = ica.St  # or model.mixing.T

# %%
# Plot them
# ---------

# sphinx_gallery_thumbnail_number = 3

_ = A.T.plot(title="Mixing System")
_ = St.plot(title="Sources spectral profiles", colormap=None)

# %%
# Reconstruct the dataset
# -----------------------
#
# The dataset can be reconstructed from these matrices anf the mean: :
X_hat_a = scp.dot(A, St) + X.mean(dim=0).data
_ = X_hat_a.plot(title=r"$\hat{X} = \bar{X} + A S^t$")

# %%
# Or using the transform() method:
X_hat_b = ica.inverse_transform()
_ = X_hat_b.plot(title="$\hat{X} =$ ica.inverse_transform()")

# %%
# Finally, the qualityu of the reconstriction can be checked by `plotmerit()`
_ = ica.plotmerit(nb_traces=15)

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script

# %%
# scp.show()
