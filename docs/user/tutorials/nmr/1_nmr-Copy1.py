# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] {"toc-hr-collapsed": false}
# # FFT

# %%
from spectrochempy import *


# %%
def _ps(data, p0=0.0, p1=0.0):
    """
    Linear phase correction

    Parameters
    ----------
    data : ndarray
        Array of NMR data.
    p0 : float
        Zero order phase in degree
    p1 : float
        First order phase in degree

    Returns
    -------
    out : ndarray
        Phased NMR data.

    """
    p0r = p0.copy().to('rad').m
    p1r = p1.copy().to('rad').m
    size = data.shape[-1]
    out = data * np.exp(1.0j * (p0r + p1r * np.arange(size)/ size)).astype(data.dtype)
    
    return out


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
from ipywidgets import interactive, fixed, FloatSlider
from IPython.display import display
#import matplotlib.pyplot as plt

def interact_pk(dataset, dim=-1, **kwargs):
    
    verbose = kwargs.get('verbose', False)
    if verbose:
        print_(' ')
        print_('INTERACTIVE PHASING MODE')
    
    new = dataset.copy()
    
    # On which axis do we want to phase? (get axis from arguments)
    # The last dimension is always the dimension on which we apply the phases.
    # If needed, we swap the dimensions to be sure to be in this situation
    axis, dim = new.get_axis(dim, negative_axis=True)
    swaped = False
    if axis != -1:
        new.swapaxes(axis, -1, inplace=True)  # must be done in  place
        swaped = True
    
    # select the last coordinates and check the unit validity
    lastcoord = new.coords[dim]
    if (lastcoord.units.dimensionality != '1/[time]' and lastcoord.units != 'ppm'):
        error_('Phasing apply only to dimensions with [frequency] dimensionality or with ppm units\n'
               'Phasing processing was thus cancelled')
        return new
    
    # get initial absolute phase and pivot
    # get the initial phase setting
    if verbose:
        print_(f'Current phases : {new.meta.phc0[-1]}, {new.meta.phc1[-1]}')
    
    phc0 = new.meta.phc0[-1]
    phc1 = new.meta.phc1[-1]
    pivot = float(abs(new).max().coords[-1].data) 
    ppivot = lastcoord.loc2index(pivot)
    
    ax = new.plot(xlim=[50,-50]).axes
    l = ax.lines[0]
    p = ax.axvline(pivot, color='r', alpha=0.5)
    
    def _phasing( ph0, ph1, pivot):
        
        rphc0 = (ph0 - phc0) * ur.deg
        rphc1 = (ph1 - phc1) * ur.deg
        ppivot = lastcoord.loc2index(pivot)
        
        data = _ps(new.data, rphc0, rphc1)
        
        l.set_ydata(data)
        p.set_xdata([pivot,pivot])
        
    w = interactive(_phasing,
                 ph0=FloatSlider(min=phc0-45,
                                 max=phc0+45,
                                 step=0.001,
                                 value=phc0,
                                 continuous_update=True),
                 ph1=FloatSlider(min=phc1-20,
                                 max=phc1+20,
                                 step=0.001,
                                 value=phc1,
                                 continuous_update=True),
                 pivot=FloatSlider(min=lastcoord[-1].values,
                                   max=lastcoord[0].values,
                                   value=pivot,
                                   continuous_update=True))
    
    return w
    
    


# %%
# %matplotlib widget

# %%
w = interact_pk(transf1)
w


# %%
w.children[0].value + dataset1D.meta.phc0[-1]

# %%
x = transf1.plot()


# %%
x.axes.lines[0].set_ydata(0)

# %%
ax = transf1.plot(xlim=[50,-50]).axes
l = ax.lines[0]
p = ax.axvline(0.2, color='r', alpha=0.5)

# %%
p.get_xdata()

# %%

# %%
