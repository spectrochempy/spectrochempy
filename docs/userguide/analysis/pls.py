# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.8
# ---
# %% [markdown]
# # Principal Component Analysis

# %%
import spectrochempy as scp

from sklearn.cross_decomposition import PLSRegression


# %% [markdown]
# ## Introduction
#
# PLS (standing for Partial Least Squares regression ) is a statistical method to estimate
# $nxl$ dependant or predicted variables $Y$ from $nxm$ explanatory or observed varaibles $X$ by
# projecting both of them on new spaces spanned by $k$ latent variables, according to:
# $$ X = S L^T + E $$
# $$ Y = P Q^T + F $$
# Where $S$ and $P$  matrices are such that the product of the first column of $S$ - the score vector $s_1$ -
# by the first line of $L^T$ - the loading vector $l_1$ - are those that best explain the variance
# of the dataset. These score and loading vectors are together called the ‘first component’. The
# second component best explain the remaining variance, etc...
#
# The implementation of PLS in spectrochempy is based on the [Scikit-Learn](https://scikit-learn.org/)
# implementation of
# [PLS](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.htm)
# with similar methods and attributes on the one hands, and some that are specific to spectrochempy.
#
# ## Loading of the dataset
# Here we show how PLSis implemented in Scpy on a dataset consisting of 80 samples of corn measured
# on 3 different NIR spectrometers, together with moisture, oil,
# protein and starch values for each of the samples. This dataset (and others) can be loaded from
# [http://www.eigenvector.com](http://www.eigenvector.com/data/).
# %%
A = scp.download("http://www.eigenvector.com/data/Corn/corn.mat")

# %% [markdown]
# The .mat file contains 7 eigenvectors's datasets which are thus returned in A as a list of 7
# NDDatasets. We print the names and dimensions of these datasets:
# %%
for a in A:
    print(f"{a.name} : {a.shape}")

# %% [markdown]
# In this tutorial, we are first interested in the dataset named `'m5spec'`, corresponding to the 80 spectra
# on one of the instruments and `'propvals'` giving the property values of the 80 corn samples.
#
# Let's name the specta NDDataset `X` , add few informations about the x-scale and plot it:

# %%
X = A[-3]
X.title = "absorbance"
X.x.title = "Wavelength"
X.x.units = "nm"
X.plot(cmap=None)

# %%
X_ = X.detrend()
_ = X_.plot(cmap=None)
# %% [markdown]
# Let's plot the properties of the sample:
Y = A[3]
_ = Y.T.plot(cmap=None, legend=Y.x.labels)

Y_std = (Y - Y.mean(dim=0)) / Y.std(dim=0)
_ = Y_std.T.plot(cmap=None, legend=Y.x.labels)

n_targets = Y.shape[1]
# %%

# %% [markdown]
pls = PLSRegression(n_components=27)
pls.fit(X_.data, Y_std.data)
#
# Yhat = pls.predict(X_.data)
# plt.figure()
# c = ["b", "g", "r", "grey"]
# for prop, i in zip(Y.x.labels, range(n_targets)):
#     print(i)
#     _ = plt.scatter(
#         Y_std.data[:, i],
#         Yhat[:, i],
#         s=50,
#         c=c[i],
#         alpha=0.3,
#         label=prop,
#     )
#     _ = plt.legend()

pls.transform(X_.data, Y_std.data)
# %%
scp_pls = scp.PLS(used_components=27)
scp_pls.fit(X_, Y_std)

scp_pls.x_loadings.plot()
scp_pls.x_weights.plot()
scp_pls.x_rotations.plot()
scp_pls.x_scores.T.plot()

# %%
scp_pls.y_loadings.plot()
scp_pls.y_weights.plot()
scp_pls.y_rotations.plot()
scp_pls.y_scores.T.plot()

# %%
scp_pls.coef.plot()
scp.show()

scp_pls.intercept.plot()
