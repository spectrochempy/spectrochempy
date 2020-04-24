# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
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

# %% [markdown] {"toc-hr-collapsed": false}
# # Introduction to NMR processing

# %%
from spectrochempy import *

# %% [markdown]
# ## Import data
#
# Here we import two dataset, one is 1D and the other is 2D
#
# Because , we will sometimes need to recall the original dataset, we create two getting functions

# %%
# 1D dataset getting function 
datadir = general_preferences.datadir
def get_dataset1D():
    dataset1D = NDDataset()
    path = os.path.join(datadir,'nmrdata','bruker', 'tests', 'nmr','bruker_1d')
    dataset1D.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    return dataset1D

# 2D dataset getting function
def get_dataset2D():
    dataset2D = NDDataset()
    path = os.path.join(datadir,'nmrdata','bruker', 'tests', 'nmr','bruker_2d')
    dataset2D.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    return dataset2D

# %%
# get the 1D dataset
dataset1D = get_dataset1D()
# display info
dataset1D 

# %%
# get the 2D dataset
dataset2D = get_dataset2D()
# display info
dataset2D

# %% [markdown]
# Note that the internal type of the data is Quaterniion (Hypercomplex, i.e., complex in two dimensions)  
#
# (For now only 2D hypercomplex dataset are handled in spectrochempy - Future work may include datasets with higher dimensionality)

# %%
dataset2D.dtype

# %% [markdown]
# ### Getting real part only

# %% [markdown]
# For 1D or 2D data

# %%
dataset1D.real

# %%
dataset2D.real

# %% [markdown]
# The imaginary part can be extracted as well

# %%
dataset1D.imag

# %%
dataset2D.imag

# %% [markdown]
# Note that for 2D, we get a Quaternion's array (with a real part set to zero).
#
# If one want only a part of the 2D array, e.g., Real in first dimension, but Imaginary in the second dimension, then one can use `RI` attributes (or `RR`, `IR` or `II` for any of the other parts)

# %%
dataset2D.RI

# %% [markdown]
# ### Math on quaternion arrays
#
# To be completed ! 

# %%
dataset2D * 2. - 1.j

# %% [markdown]
# ## Plot the 1D dataset raw data

# %%
# restore the original dataset (useful in case of mutiple execution of part of the code, 
# to avoid unexpected cumulative processing)
dataset1D = get_dataset1D()

# plot the real data
dataset1D.plot(color='blue', xlim=(0,15000.)) 

# plot the imaginary data on the same plot
# Note that we assign the results of this function to `_` in order to avoid output such as :
# <matplotlib.axes._subplots.AxesSubplot at 0x1c1b2fbac8>
_ = dataset1D.plot(imag=True, color='red', ls='--', ylim=(-600,600), 
                   data_only=True, clear=False)

# Note the two additional flags:
# ------------------------------
# `clear=False`to plot on the previous plot (without this a new plot is created)
# `data_only =True` to plot only the additional data, without updating the figure setting 
#  such as xlim and so on, except if they are specifically indicated.

# %% [markdown]
# To display the imaginary part, one can also simply use the show_complex commands.

# %%
_ = dataset1D.plot(show_complex=True, color='green', xlim=(0.,25000.))

# %% [markdown]
# ## Plot the 2D dataset raw data

# %%
# restore the original dataset
dataset2D = get_dataset2D()
# plot the dataset as a contour map
_ = dataset2D.plot_map(xlim=(0.,25000.))

# %% [markdown]
# Multiple display are also possible for 2D dataset

# %%
dataset2D.plot_map(xlim=(0.,6000.), ylim=(0.,6000.))
# here we plot the transposed dataset (dataset2D.T) on the same figure.
_ = dataset2D.T.plot_map(cmap='magma', data_only=True, clear=False)

# %% [markdown]
# ## Apodization

# %% [markdown]
# ### Introduction to apodization processing

# %% [markdown]
# In most situation, there is two syntaxes to perform operation on the dataset.
#
# For instance, to perform the apodization using an exponential multiplication:
#
# 1. newdataset = dataset.em(lb=...)
# 2. newdataset = em(dataset, lb=...)

# %%
# restore the original dataset
dataset1D = get_dataset1D()

# plot it
dataset1D.plot() 

# Create the apodized dataset usint exponential multiplication (em)
lb_dataset = dataset1D.em(lb=100.*ur.Hz, inplace=True)

# plot the apodized dataset on the same figure
lb_dataset.plot(xlim=(0,25000), zlim=(-100,100), color='r', clear=False)

# add some text
_ = lb_dataset.ax.text(12500,90,'Dual display (original & apodized fids)', 
                       ha='center', fontsize=10)

# %% [markdown]
# Note that the apodized dataset actually replace the original data

# %%
# check that both dataset are the same (an error should be issued by the assert command if it is not the case)
assert lb_dataset is dataset1D  # note here, that the original data are modified by default 
                                # when applying apodization function. 

# %% [markdown]
# We can use the `inplace` keyword set to *False* to modify this behavior

# %%
# restore original
dataset1D = get_dataset1D()

# create a apodized copy of the dataset
lb_dataset = dataset1D.em(lb=100.*ur.Hz, inplace=False)

# this is a new dataset
assert lb_dataset is not dataset1D

# %% [markdown]
# We can use of the second syntax.

# %%
# restore original
dataset1D = get_dataset1D() 

# Create the apodized dataset
lb2_dataset = em(dataset1D, lb=100.*ur.Hz, inplace=False)

# check that lb2_dataset and the previous lb_dataset are equal
assert lb2_dataset == lb_dataset

# %% [markdown]
# We can also get the apodization function together with the processed dataset

# %%
# restore original
dataset1D = get_dataset1D()

# create the apodized dataset 
# when retfunc is True, the apodization function is also returned together witht he apodized dataset
lb_dataset, apodfunc = dataset1D.em(lb=100.*ur.Hz, retfunc=True, inplace=False)

# plot the 3 datasets on the same figure 
dataset1D.plot() 
(apodfunc*200.).plot(color='r', clear=False)
lb_dataset.plot(data_only=True, xlim=(0,25000), zlim=(-200,200), color='g', clear=False) 

_ = dataset1D.ax.text(12500,180,'Multiple display (original & em apodized fids + apod.function)', ha='center', fontsize=10)

# %% [markdown]
# ### Available apodization functions

# %% [markdown]
# #### em

# %%
# restore original
dataset1D = get_dataset1D()[:10000.0]  # take a selection

# normalize amplitude
dataset1D /= dataset1D.max()

# apodize
LB = 100.*ur.Hz
lb_dataset, apodfunc = dataset1D.em(lb=LB, retfunc=True, inplace=False) 

# Plot
dataset1D.plot(lw=1, color='gray') 
apodfunc.plot(color='r', clear=False)
lb_dataset.plot(color='r', ls='--', clear=False) 

# shifted
lbshifted_dataset, apodfuncshifted = dataset1D.em(lb=LB, shifted=3000, retfunc=True, inplace=False) 
apodfuncshifted.plot(color='b')
lbshifted_dataset.plot(ylim=(-1,1), color='b', ls='--', clear=False) 

# rev
lbrev_dataset, apodfuncrev = dataset1D.em(lb=LB, rev=True, retfunc=True, inplace=False) 
apodfuncrev.plot(color='g')
lbrev_dataset.plot(ylim=(-1,1), color='g', ls='--', clear=False) 

# inv
lbinv_dataset, apodfuncinv = dataset1D.em(lb=LB, inv=True, retfunc=True, inplace=False)
apodfuncinv.plot(color='m')
_ = lbinv_dataset.plot(ylim=(-1.5,1.5), color='m', ls='--', clear=False) 

# %% [markdown]
# #### gm

# %%
# restore original
dataset1D = get_dataset1D()[:10000.0]  # take a selection 
dataset1D /= dataset1D.max()

# apodize
LB = -100.*ur.Hz
GB = 300.*ur.Hz
gb_dataset, apodfunc = dataset1D.gm(gb=GB, lb=LB, retfunc=True, inplace=False)

# plot 
dataset1D.plot() 
apodfunc.plot(color='r', clear=False)
_ = gb_dataset.plot(xlim=(0,10000), zlim=(-1.5,1.5), color='r', ls='--', clear=False) 

# shifted
LB = 10.*ur.Hz
GB = 300.*ur.Hz
gbsh_dataset, apodfuncsh  = dataset1D.gm(gb=GB, lb=LB, shifted=2000, retfunc=True, inplace=False) 

# plot 
apodfuncsh.plot(color='g', clear=False)
_ = gbsh_dataset.plot(zlim=(-1.5,1.5), color='g', ls='--', clear=False) 

# %% [markdown]
# #### sp (shifted sine-bell apodization)

# %% [markdown]
# $\text{sp}(x) = \sin(\frac{(\pi - \phi) t }{\text{aq}} + \phi)^{p}$ where 
#         
# where $0 < t < \text{aq}$ and  $\phi = \pi ⁄ \text{sbb}$ when $\text{ssb} \ge 2$ or $\phi = 0$ when $\text{ssb} < 2$.
#         
# $\text{aq}$ is an acquisition status parameter and $\text{ssb}$ is a processing parameter (see the `SSB` parameter definition below) and $\text{pow}$ is an exponent equal to 1 for a sine bell window or 2 for a squared sine bell window.
#     
# SSB: This processing parameter mimics the behaviour of the SSB parameter on bruker TOPSPIN software: Typical values are 1 for a pure sine function and 2 for a pure cosine function. Values greater than 2 give a mixed sine/cosine function. Note that all values smaller than 2, for example 0, have the same effect as $\text{ssb}=1$, namely a pure sine function.

# %%
dataset1D = get_dataset1D()[:10000.0]  # take a selection
dataset1D /= dataset1D.max()

ssb = 2.
pow = 1.

sp_dataset, apodfunc = dataset1D.sp(ssb=ssb, pow=pow, retfunc=True, inplace = False)

# plot
dataset1D.plot() 
apodfunc.plot(color='r', clear=False)
_ = sp_dataset.plot(color='r', ls='--', clear=False, xlim=(0,10000))

# %%
dataset1D = get_dataset1D()[:10000.0]  # take a selection
dataset1D /= dataset1D.max()

ssb = 3.
pow = 2.

sp_dataset, apodfunc = dataset1D.sp(ssb=ssb, pow=pow, retfunc=True, inplace = False)

# plot
dataset1D.plot() 
apodfunc.plot(color='r', clear=False)
_ = sp_dataset.plot(color='r', ls='--', clear=False, xlim=(0,10000))

# %% [markdown]
# ### Apodization of 2D data

# %%
dataset2D = get_dataset2D()
dataset2D.plot_map(xlim=(0.,25000.))

LB = 200.*ur.Hz
dataset2D.em(lb=LB)
dataset2D.em(lb=LB/2, axis=0)  
_ = dataset2D.plot_map(data_only=True, cmap='copper', clear=False)

# %% [markdown]
# ## Time-frequency transforms : FFT

# %%
dataset1D = get_dataset1D() # restore original
LB = 10.*ur.Hz
dataset1D.em(lb=LB)
transf1 = dataset1D.fft(size=32000) 
_ = transf1.plot(xlim=[30,-30])

# %% [markdown]
# by default, the frequency axis unit is in ppm when data originate from NMR. Note that the axis is automatically reversed as it is usual for ppm scale in NMR. 

# %% [markdown]
# To get frequency scale, then set `ppm` flag to False

# %%
dataset1D = get_dataset1D() # restore original
LB = 10.*ur.Hz
GB = 50.*ur.Hz
dataset1D.gm(gb=GB, lb=LB)
transf1 = dataset1D.fft(size=32000, ppm=False) 
assert transf1 is not dataset1D
_ = transf1.plot(xlim=[5000,-5000])

# %% [markdown]
# As the new dataset is transformed, any function that apply only to time data such as apodization (*e.g.*, **em** or **gm**) should not work

# %%
# This generate an error
#_ = transf1.em(lb=10*ur.Hz)

# %%
# and also this generate an error
#_ = transf1.fft()

# %% [markdown]
# One can perform the inverse fourier transform using **fft** with `inv=True` keyword argument, or **ifft**

# %%
#transf2 = transf1.ifft()
#_ = transf2.plot()

# %% [markdown]
# ## Phasing
#
# Our spectra can be phased using 

# %%
transf1 = dataset1D.fft(size=32000) 
_ = transf1.plot()

# %%
pt = transf1.pk()
_ = pt.plot(xlim=[50,-50])

# %%
# automatic phasing
transfph3 = transf1.apk(verbose=True)

pt.plot(xlim=(20,-20))
transfph3.plot(xlim=(20,-20), clear=False, color='b')

# %%
# automatic phasing
transfph3 = transf1.apk(verbose=True, fit_phc1=True, mode='negmin+entropy', ediff=2, gamma=1)

pt.plot(xlim=(20,-20))
transfph3.plot(xlim=(100,-100), clear=False, color='b')

# %%

# %%

# %%

# %%

# %%
