# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MCR ALS

# %%
from spectrochempy import *

# %% [markdown]
# ## 1. Introduction 
#
# MCR-ALS (standing for Multivariate Curve Resolution - Alternating Least Squares) is a popular method for resolving a set (or several sets) of spectra $X$ of an evolving mixture (or a set of mixtures) into the spectra $S^t$ of 'pure' species and their concentration profiles $C$. In term of matrix equation:
# $$ X = C S^t + E $$
# The ALS algorithm allows applying soft or hard constraints (e.g. non negativity, unimodality, equality to a given profile) to the spectra or concentration profiles of pure species. This property makes MCR-ALS an extremely flexible and powerful method. Its current implementation in Scpy is limited to soft constraints but is exected to cover more advanced features in further releases.
#
# In, this tutorial the application of MCS-ALS as implemented in Scpy to a 'classical' dataset form the literature is presented.
#
# ## 2. The (minimal) dataset
#
# In this example, we perform the MCR ALS optimization of a dataset corresponding to a HPLC-DAD run, from Jaumot et al. Chemolab, 76 (2005), 
# pp. 101-110 and Jaumot et al. Chemolab, 140 (2015) pp. 1-12. This dataset (and others) can be loaded from the "Multivariate Curve Resolution Homepage"
# at https://mcrals.wordpress.com/download/example-data-sets. For the user convenience, this dataset is present in the 'datadir' of spectrochempy in 'als2004dataset.MAT' and can be read as follows in Scpy: 

# %%
A = read_matlab("matlabdata/als2004dataset.MAT")

# %% [markdown]
# The .mat file contains 6 matrices which are thus returned in A as a list of 6 NDdatasets. We print the names and dimensions of these datasets:

# %%
for a in A:
    print(a.name + ': ' + str(a.shape))

# %% [markdown]
# In this tutorial, we are first interested in the dataset named ('m1') that contains a singleHPLC-DAD run(s).
# As usual, the rows correspond to the 'time axis' of the HPLC run(s), and the columns to the 'wavelength' axis
# of the UV spectra. 
#
# Let's name it 'X' (as in the matrix equation above), display its content and plot it:

# %%
X = A[0]
X

# %%
_ = X.plot()

# %% [markdown]
# The original dataset is the 'm1' matrix and does not contain information as to the actual elution time, wavelength, and data units. Hence the resulting NDDataset has no coordinates and on the plot, only the matrix line and row indexes are indicated. For the clarity of the tutorial, we add: (i) a proper title to the data, (ii) the default coordinates (index) do the NDDataset and (iii) a proper name for these coordinates:

# %%
X.title = 'absorbance'
X.set_coords(None, None)
X.set_coordtitles(y='elution time', x='wavelength')
X

# %% [markdown]
# From now on, these names will be taken into account by Scpy in the plottings as well as in the analysis treatments (PCA, EFA, MCR-ALs, ...). For instance to plot X as a surface:

# %%
surf = X.plot_surface(colorbar=True, linewidth=.2, ccount=100, figsize=(10,5))

# %% [markdown]
# ## 3 Initial guess and MCR ALS optimization
#
# The ALS optimization of the MCR equation above requires the input of a guess for either the concentration matrix $C_0$ or the spectra matrix $S^t_0$. Given the data matrix $X$, the lacking initial matrix ($S^t_0$ or $C_0$, respectively) is computed by:
# $$ S^t_0 = \left( C_0^tC_0 \right)^{-1} C_0^t X $$
#
# or
#
# $$ C_0 = X {S^t_0}^t  \left( S_0^t {S_0^t}^t \right)^{-1} $$
#
# ### 3.1. Case of initial spectral profiles
# The matrix spure provided in the initial dataset is a guess for the spectral profiles. Let's name it 'St0', and plot it:  

# %%
St0 = A[1]
_ = St0.plot()

# %% [markdown]
# Note that, again, no information has been given as to the ordinate and abscissa data. We could add them as previously but this is not very important. The key point is that the 'wavelength' dimension is compatible with the data 'X', which is indeed the case (both have a legth of 95). If it was not, an error would be generated in the following.  
#
# #### 3.1.1 ALS Optimization
# With this guess 'St0' and the dataset 'X' we can create a MCRALS object. At this point of the tutorial, we will use all the default parameters except for the 'verbose' option which is swiched on to have a summary of the ALS iterations: 

# %%
mcr = MCRALS(X, St0, verbose='True')

# %% [markdown]
# The optimization has converged within few iterations. The figures reported for each iteration are defined as follows:
#
# - 'Error/PCA' is the standard deviation of the residuals with respect to data reconstructed by a PCA with as many components as pure species (4 in this example),
#
# - 'Error/exp': is the standard deviation of the residuals with respect to the experimental data X,
#
# - '%change': is the percent change of 'Error/exp' between 2 iterations
#
# The default is to stop when this %change between two iteration is negative (so that the solution is improving), but with an absolute value lower than 0.1% (so that the improvement is considered negligible). This parameter - as well as several other parameters affecting the ALS optimization can be changed by the setting the 'tol' value in a python dictionary using the key 'tol'. For instance: 

# %%
mcr = MCRALS(X, St0, param={'tol':0.01}, verbose='True')

# %% [markdown]
# As could be expected more iterations have been necessary to reach this stricter convergence criterion.  The other convergence criterion that can be fixed by the user is 'maxdiv', the maximum number of successive diverging iterations. It is set to 5 by default and allows for stopping the ALS algorithm when it is no converging. If for instance the 'tol' is set very low, the optimization will be stopped when either the maximum number of iterations is reached (maxit, 50 by default) or when no improvement is during 5 successive iterations (maxdiv).   

# %%
mcr = MCRALS(X, St0, param={'tol':0.001}, verbose='True')

# %% [markdown]
# Now if 'maxit' is set to 10:  

# %%
mcr = MCRALS(X, St0, param={'tol':0.001, 'maxit':10}, verbose='True')

# %% [markdown]
# #### 3.1.2 Solutions
#
# The solutions of the MCR ALS optimization are the optimized concentration and pure spectra matrices. They can be obtained by the MCRALS.transform() method. let's remake an MCRALS object with the default settings, ('tol' = 0.1 and verbose = False), and get C and St.

# %%
mcr1 = MCRALS(X, St0)

# %% [markdown]
# As the dimensions of C are such that the rows direction (C.y) corresponds to the elution time and the columns direction (C.x) correspond to the four pure species, it is necessary to transpose it before plotting in order to plot the concentration vs. the elution time.

# %%
_ = mcr1.C.T.plot()

# %% [markdown]
# On the other hand, the spectra of the pure species can be plot directly:

# %%
_ = mcr1.St.plot()

# %% [markdown]
# #### 3.1.3 A basic illustration of the rotational ambiguity
# We have thus obtained the elution profiles of the four pure species. Note that the 'concentration' values are very low. This results from the fact that the absorbance values in X are on the order of 0-1 while the absorbances of the initial pure spectra are of the order of 10^4. As can be seen above, the absorbance of the final spectra is of the same order of magnitude.
#
# It is possible to normalize the intensity of the spectral profiles by setting the 'normSpec' parameter to True. With thios option, the specra are normalized such that their euclidian norm is 1. The other normalization option is n ormspec = 'max', whereby the maximum intyensity of tyhe spectra is 1. Let's look at the effect of both normalizations:

# %%
mcr2 = MCRALS(X, St0, param={'normSpec': 'euclid'})

mcr3 = MCRALS(X, St0, param={'normSpec': 'max'})

_ = mcr1.St.plot()
_ = mcr2.St.plot()
_ = mcr3.St.plot()

# %%
_ = mcr1.C.T.plot()
_ = mcr2.C.T.plot()
_ = mcr3.C.T.plot()

# %% [markdown]
# It is clear that the normalization affects the relative intensity of the spectra and of the concentration. This is a basic example of the well known rotational ambiguity of the MCS ALS solutions.

# %% [markdown]
# ### 3.2 Guessing the concentration profile with PCA + EFA
#
# Generally, in MCR ALS, the initial guess cannot be obtained independently of the experimental data 'x'. In such a case, one has to rely on 'X' to obtained (i) the number of pure species  and (ii) their initial concentrations or spectral profiles. The number of of pure species can be assessed by carrying out a PCA on the data while the concentrations or spectral profiles can be estimated using procedures such EFA of SIMPLISMA. The following will illustrate the use of PCA followed by EFA
#
# #### 3.2.1 Use of PCA to assess the number of pure species
#
# Let's first analyse our dataset using PCA and plot a screeplot:

# %%
pca = PCA(X)
pca.printev(n_pc=10)
_ = pca.screeplot(n_pc=8)

# %% [markdown]
# The number of significant PC's is clearly larger or equal to 2. It is, however, difficult tto determine whether it should be set to 3 or 4...  Let's look at the score and loading matrices:
#

# %%
S, LT = pca.reduce(n_pc=8)
_ = S.T.plot()
_ = LT.plot()

# %% [markdown]
# Examination of the scores and loadings indicate that the 4th component has structured, non random scores and loadings. Hence we will fix the number of pure species to 4.
#
# NB: The PCA.transform() can also be used with n_pc='auto' to determine automatically the number of components using the method of Thomas P. Minka (Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604). This type of methods, however, often lead to too many PC's for the chemist because they recover all contributions to the data variance: chemical AND non-chemical, thus including non-gaussian noise, baseline changes, background absorption...
#
# 32 in the present case:

# %%
S3, LT3 = pca.reduce(n_pc='auto')
S3.shape

# %% [markdown]
# #### 3.2.2 determination of initial concentrations using EFA
#
# Once the number of components has been determined, the initial concentration matrix is obtained very easily using EFA:
#

# %%
efa = EFA(X)
C0 = efa.get_conc(n_pc=4)
_ = C0.T.plot()

# %% [markdown]
# The MCR ALS can then be launched using this new guess:

# %%
mcr4 = MCRALS(X, guess=C0, param={'maxit':100, 'normSpec':'euclid'}, verbose=True) 

# %%
_ = mcr4.C.T.plot()
_ = mcr4.St.plot()

# %% [markdown]
# ## 4. Augmented datasets

# %% [markdown]
# The 'MATRIX' dataset is a columnwise augmented dataset consisting into 5 successive runs:

# %%
A[3]

# %% [markdown]
# Let's plot it as a map, and as a surface:

# %%
X2 = A[3]
X2.title = 'absorbance'
X2.set_coords(None, None)
X2.set_coordtitles(y='elution time', x='wavelength')

surf = X2.plot_surface(colorbar=True, linewidth=.2, ccount=100, figsize=(10,5))

_ = X2.plot(method='map')



# %%
mcr5 = MCRALS(X2, guess=St0, param={'unimodConc': [0] * 4}, verbose=True)

# %%
_ = mcr5.C.T.plot()

# %%
_ = mcr5.St.plot()

# %% [markdown]
# [To be continued...]
