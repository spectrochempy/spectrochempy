# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
NDDataset PCA analysis example
-------------------------------
In this example, we perform a PLS regression to predict the composition of corn samples
from their NIR spectra.
"""

# %%
# Import the spectrochempy API package
import spectrochempy as scp
import matplotlib.pyplot as plt

# %%
# Load a dataset
ds_list = scp.download("http://www.eigenvector.com/data/Corn/corn.mat")
for ds in ds_list:
    print(ds.name)
# %%
# The dataset named `'m5spec'`, contains the NIR spectra of 80 corn samples recorded on the same
# instrument. Let's assign this NDDataset specta to `X`, add few informations and plot it:
# %%
X = ds_list[4]
X.title = "reflectance"
X.x.title = "Wavelength"
X.x.units = "nm"
X.plot(cmap=None)

# %%
# The values of the properties we want to predict are in the `'propval'` dataset:
Y = ds_list[3]
_ = Y.T.plot(cmap=None, legend=Y.x.labels)

# We are interested to predict the moisture content:
y = Y[:, 0]

# We select arbitrarily the 57 first samples to train the model and the 27 remaining ones
# to test the model.
X_train = X[:57]
X_test = X[57:]
y_train = y[:57]
y_test = y[57:]


# %%
# Create a PLS object and fit the train datasets:
pls = scp.PLS(used_components=5)
pls.fit(X_train, y_train)

y_train_hat = pls.predict()
y_test_hat = pls.predict(X_test)

plt.figure()
plt.plot(y_train.data, y_train_hat.data, "o")
plt.plot(y_test.data, y_test_hat.data, "x")


# %%
