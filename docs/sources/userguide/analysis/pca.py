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
# from the Multivariate Curve Resolution example collection.
#
# ```{=rst}
# See :cite:t:`jaumot:2005,jaumot:2015` for the original publications.
# ```
#
# This dataset (and others) can be loaded from the [Multivariate Curve Resolution
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
_ = X.plot()

# %% [markdown]
# This original matrix ('m1') does not contain information as to the actual elution time, wavelength,
# and data units. Hence, the resulting NDDataset has no coordinates and on the plot,
# only the matrix line and row # indexes are indicated.
# For the clarity of the tutorial, we add: (i) a proper title to the data, (ii)
# the default coordinates (index) do the NDDataset and (iii) a proper name for these
# coordinates:
# %%
X.title = "absorbance"
X.set_coordset(None, None)
X.set_coordtitles(y="elution time", x="wavelength")

# %% [markdown]
# From now on, these names will be taken into account by Scpy in the plots as well as
# in the analysis treatments (PCA, EFA, MCR-ALS ...). For instance to plot X as a surface:
# %%
surf = X.plot_surface(linewidth=0.0)

# %% [markdown]
# ## Running a PCA
# First, we create a PCA object with default parameters and we compute the components with the fit() method:
# %%
pca = scp.PCA()
_ = pca.fit(X)

# %% [markdown]
# The default number of components is given by min(X.shape). As often in spectroscopy
# the number of observations/spectra is lower that the number of wavelength/features, the number of components
# often equals the number of spectra.
# %%
pca.n_components

# %% [markdown]
# As the main purpose of PCA is dimensionality reduction, we generally limit the PCA to a limited number of components.
# This can be done by either resetting the number of components of an existing object:
# %%
pca.n_components = 8
_ = pca.fit(X)

# %% [markdown]
# Or directly by creating a PCA instance with the desired number of components:
# %%
pca = scp.PCA(n_components=8)
_ = pca.fit(X)

# %% [markdown]
# The grouped PCA outputs are available from `pca.result`. This is the most
# convenient grouped API when you want related fitted outputs together:
# %%
scores = pca.result.scores
loadings = pca.result.loadings
variance = pca.result.explained_variance

# %% [markdown]
# The mapping form, such as `pca.result.outputs["scores"]`, remains available
# for introspection and programmatic access. Historical direct estimator
# accessors such as `pca.scores`, `pca.loadings`, and `pca.transform()` also
# remain supported in this release, so `.result` is promoted without being
# mandatory.

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
_ = pca.plot_scree()

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
_ = pca.scores.T.plot()
_ = pca.loadings.plot()

# %% [markdown]
# Examination of the plots above indicate that the 4th component has a structured,
# nonrandom loading and score, while they are random fopr the next ones. Hence, one could
# reasonably assume that 4 PC are enough to correctly account of the dataset.
#
# Another possibility can be a visual comparison of the modeled dataset $\hat{X} = S L^T $,
# the original dataset $X$ and the residuals $E = X - \hat{X}$. This can be done using
# the `plot_merit()` method which plots both $X$, $\hat{X}$ (in orange) and the residuals (in grey):
# %%
pca = scp.PCA(n_components=4)
_ = pca.fit(X)
_ = pca.plot_merit()


# %% [markdown]
# Score plots can be used to see the projection of each observation/spectrum
# onto the span of the principal components:
# %%
_ = pca.plot_score()
_ = pca.plot_score(components=(1, 2, 3))


# %% [markdown]
# ## Compare PCA before and after masking a saturated band
#
# A large contribution to the variance can come from the instrument. In the
# NH₄Y activation series, detector saturation affects the band between 882 and
# 1280 cm⁻¹. Compare the PCA results on the same subset of spectra with and
# without that interval. Float slice bounds select wavenumbers.
#
# Use `n_components=0.999` to retain enough components to explain at least
# 99.9% of the variance. Compare the number of components retained by the
# two fits. Each percentage refers to the variance of its own input matrix.

# %%
spectra = scp.read_omnic("irdata/nh4y-activation.spg")[::5]
masked = spectra.copy()
masked[:, 882.0:1280.0] = scp.MASKED

full_pca = scp.PCA(n_components=0.999)
masked_pca = scp.PCA(n_components=0.999)
_ = full_pca.fit(spectra)
_ = masked_pca.fit(masked)
print(f"Components retained before masking: {full_pca.n_components}")
print(f"Components retained after masking: {masked_pca.n_components}")

# %% [markdown]
# With 11 spectra, centering limits the rank to at most 10. The threshold
# selects a number of components for this dataset and preprocessing. To
# interpret the result, use the loadings to locate the wavenumbers that
# contribute to each component and the scores to compare the spectra.
# Assigning a component to a chemical species requires independent evidence.

# %%
_ = masked_pca.result.loadings.plot(legend=True)
_ = masked_pca.plot_score(components=(1, 2))

# %% [markdown]
# The [short IR gallery example](../../gettingstarted/examples/gallery/auto_examples_analysis/a_decomposition/plot_pca_spec.rst)
# shows the fit after masking and its score plot.


# %% [markdown]
# ## Explore the iris measurements
#
# Each row describes a flower, with sepal and petal length and width in
# the columns. The PCA uses the measurements on their original scale.
# Species labels are used to color the scores after fitting.
#
# ```{=rst}
# The iris measurements were used by :cite:t:`fisher:1936`.
# ```
#
# Compare two ways to choose how many components to retain. The maximum
# likelihood estimate (`n_components="mle"`) requires at least as many
# observations as variables. A fraction such as `0.999` sets a target
# for cumulative explained variance.

# %%
flowers = scp.load_iris()
mle_pca = scp.PCA(n_components="mle")
variance_pca = scp.PCA(n_components=0.999)
_ = mle_pca.fit(flowers)
_ = variance_pca.fit(flowers)
for label, model in [
    ("Maximum likelihood", mle_pca),
    ("99.9% threshold", variance_pca),
]:
    retained = model.cumulative_explained_variance[-1].value.magnitude
    print(f"{label}: {model.n_components} components, {retained:.2f}% variance")

# %% [markdown]
# The scores give the coordinates of each flower in the projection.
# The loadings describe how the original measurements contribute to
# those axes. The scree plot shows the variance explained by each axis.

# %%
iris_scores = variance_pca.result.scores
iris_loadings = variance_pca.result.loadings
print(f"Scores: {iris_scores.shape}, loadings: {iris_loadings.shape}")
iris_loadings

# %%
_ = variance_pca.plot_scree()

# %% [markdown]
# The first two axes separate setosa from the other species in this
# projection. Versicolor and virginica still overlap. Use a third axis
# to inspect their separation from another angle. Explained variance
# measures how much variation the projection retains across all flowers.

# %%
_ = variance_pca.plot_score(components=(1, 2), color_mapping="labels")

# %%
ax = variance_pca.plot_score(components=(1, 2, 3), color_mapping="labels")
_ = ax.view_init(10, 75)

# %% [markdown]
# The [short iris gallery example](../../gettingstarted/examples/gallery/auto_examples_analysis/a_decomposition/plot_pca_iris.rst)
# fits two components directly and shows the 2D score plot.
