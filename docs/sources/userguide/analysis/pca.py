# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
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

# %% [markdown]
# ## Introduction
#
# PCA (standing for Principal Component Analysis) is one of the most popular
# method for dimensionality reduction by factorising a dataset $X$ into score ($S$) and
# loading ($L^T$) matrices of a limited number of components and minimize the error $E$:
# $$ X = S L^T + E $$
# These matrices are such that the product of the first column of $S$ - the score vector $s_1$ -
# by the first line of $L^T$ - the loading vector $l_1$ - are those that best explain the variance
# of the dataset. These score and loading vectors are together called the ‘first component’. The
# second component best explain the remaining variance, etc...
#
# The implementation of PCA in spectrochempy is based on the [Scikit-Learn](https://scikit-learn.org/)
# implementation of [PCA](https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca).
# with similar methods and attributes on the one hands, and some that are specific to spectrochempy.
#
# ## Loading of the dataset
# Here we show how PCA is implemented in Scpy on a dataset corresponding to a HPLC-DAD run,
# from Jaumot et al. Chemolab, 76 (2005), pp. 101-110 and Jaumot et al. Chemolab, 140 (2015)
# pp. 1-12. This dataset (and others) can be loaded from the [Multivariate Curve Resolution
# Homepage](https://mcrals.wordpress.com/download/example-data-sets). For the user convenience,
# this dataset is present in the 'datadir' of spectrochempy in 'als2004dataset.MAT' and can be
# read as follows in Scpy:
# %%
A = scp.read_matlab("matlabdata/als2004dataset.MAT")

# %% [markdown]
# The .mat file contains 6 matrices which are thus returned in A as a list of 6
# NDDatasets. We print the names and dimensions of these datasets:
# %%
for a in A:
    print(f"{a.name} : {a.shape}")

# %% [markdown]
# In this tutorial, we are first interested in the dataset named ('m1') that contains
# a singleHPLC-DAD run(s).
# As usual, the rows correspond to the 'time axis' of the HPLC run(s), and the columns
# to the 'wavelength' axis of the UV spectra.
#
# Let's name it `X` (as in the matrix equation above), display its content and plot it:

# %%
X = A[-1]
X

# %%
X.plot()

# %% [markdown]
# This original matrix ('m1') does not contain information as to the actual elution time, wavelength,
# and data units. Hence, the resulting NDDataset has no coordinates and on the plot,
# only the matrix line and row # indexes are indicated.
# For the clarity of the tutorial, we add: (i) a proper title to the data, (ii)
# the default coordinates (index) do the NDDataset and (iii) a proper name for these
# coordinates:
# %%
X.title = "absorbance"
X.y = scp.Coord.arange(51, title="elution time", labels=[str(i) for i in range(51)])
X.x = scp.Coord.arange(96, title="wavelength")

# %% [markdown]
# From now on, these names will be taken into account by Scpy in the plots as well as
# in the analysis treatments (PCA, EFA, MCR-ALS ...). For instance to plot X as a surface:
# %%
surf = X.plot_surface(linewidth=0.0, ccount=100, figsize=(10, 5), autolayout=False)

# %% [markdown]
# ## Running a PCA
# First, we create a PCA object with default parameters and we compute the components with the fit() method:
# %%
pca = scp.PCA()
pca.fit(X)

# %% [markdown]
# The default number of components is given by min(X.shape). As often in spectroscopy
# the number of observations/spectra is lower that the number of wavelength/features, the number of components
# often equals the number of spectra.
# %%
pca.n_components

# %% [markdown]
# As the main purpose of PCA is dimensionality reduction, we generally limit the PCA to a limited number of components.
# This can be done by either reseting the number of components of an existing object:
# %%
pca.n_components = 8
pca.fit(X)

# %% [markdown]
# Or directly by creating a PCA instance with the desired number of components:
# %%
pca = scp.PCA(n_components=8)
pca.fit(X)

# %% [markdown]
# The choice of the optimum number of components to describe a dataset is always a delicate matter. It can be based on:
# - examination of the explained variance
# - examination of the scores and loadings
# - comparison of the experimental vs. reconstructed dataset
#
# The so-called figures of merit of the PCA can be obtained with the `printev()` method:
# pca.printev()

# %% [markdown]
# The data of the two last columns are stored in the `PCA.explained_variance_ratio' and
# `PCA.cumulative_explained_variance' attributes. They can be plotted directly as a scree plot:
# %%
_ = pca.screeplot()

# %% [markdown]
# The number of significant PC's is clearly larger or equal to 2. It is, however,
# difficult to determine whether
# it should be set to 3 or 4...  Let's look at the score and loading matrices.
#
# The scores and loadings can be obtained using the `scores` and `loadings` PCA attribute or
# obtained by the Scikit-Learn-like methods/attributes `pca.transform()` and `pca.components`,
# respectively.
#
# Scores and Loadings can be plotted using the usual plot() method, with prior transpositon
# for the scores:
# %%
pca.scores.T.plot()
pca.loadings.plot()

# %% [markdown]
# Examination of the plots above indicate that the 4th component has a structured,
# nonrandom loading and score, while they are random fopr the next ones. Hence, one could
# reasonably assume that 4 PC are enough to correctly account of the dataset.
#
# Another possibility can be a visual comparison of the modeled dataset $\hat{X} = S L^T $,
# the original dataset $X$ and the resitua,s $E = X - \hat{X}$. This can be done using
# the `plotmerit()` method which plots both $X$, $\hat{X}$ (in dotted lines) and the residuals (in red):
# %%
pca = scp.PCA(n_components=4)
pca.fit(X)
pca.plotmerit()

# %% [markdown]
# The number of spectra can be limited by the `nb_traces` attributes:
# %%
pca.plotmerit(nb_traces=5)

# %% [markdown]
# and if needed both datasets can be shifted using the `offset` attribute (in percet of the fullscale):
# %%
pca.plotmerit(nb_traces=5, offset=100.0)

# %% [markdown]
# Score plots can be used to see the projection of each observation/spectrum
# onto the span of the principal components:
# %%
_ = pca.scoreplot(1, 2, show_labels=True, labels_every=5)
_ = pca.scoreplot(1, 2, 3)
