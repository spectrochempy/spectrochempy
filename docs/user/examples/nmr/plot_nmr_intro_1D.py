# coding: utf-8
"""
Introduction to NMR processing
===========================================

Here we explain how to display and perform basic processing of NMR file

"""
import os
import spectrochempy as scp

###############################################################################
#
# Here we import a 1D NMR dataset
# 
# Because we will sometimes need to recall the original dataset,
# we create a getting functions

###############################################################################
# Loading the NMR data
# --------------------
# Let's define the 1D dataset getting function
datadir = scp.datadir.path
def get_dataset1D():
    """Read the 1D dataset"""
    s1D = scp.NDDataset()
    path = os.path.join(datadir, 'nmrdata', 'bruker', 'tests', 'nmr',
                        'bruker_1d')
    s1D.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    return s1D


###############################################################################
# Now get the 1D dataset
dataset1D = get_dataset1D()
print(dataset1D)


###############################################################################
# Plot the 1D dataset raw data
# ----------------------------
# We plot the real data and the imaginary data on the same plot
dataset1D.plot(xlim=(0, 25000), style='paper')
dataset1D.plot(imag=True, data_only=True, color='red', clear=False)

###############################################################################
# `data_only=True` to plot only the additional data, without updating the
# figure setting such as xlim and so on.


###############################################################################
# To display the imaginary part, one can also simply use the show_complex
# commands.

ax = dataset1D.plot(show_complex=True, color='green', xlim=(0., 20000.),
                   zlim=(-2., 2.))


###############################################################################
# Apodization

dataset1D = get_dataset1D()  # restore original
p = dataset1D.plot()

# create the apodized dataset
ur = scp.ur  # the unit registry
lb_dataset = dataset1D.em(lb=100. * ur.Hz)

p = lb_dataset.plot(xlim=(0, 25000), zlim=(-2, 2))

lb_dataset.ax.text(12500, 1.70, 'Dual display (original & apodized fids)',
                  ha='center', fontsize=10)


###############################################################################
# Note that the apodized dataset actually replace the original data
# check that both dataset are the same
print(lb_dataset is dataset1D)

###############################################################################
# note here, that the original data are modified by default when applying apodization function.
#
# If we want to avoid this behavior and create a new dataset instead,
# we use the `inplace` flag.
dataset1D = get_dataset1D()
lb2_dataset = dataset1D.em(lb=100. * ur.Hz, inplace=False)

# check that both dataset are different
print(lb2_dataset is not dataset1D)

###############################################################################
# We can also get only the apodization function

dataset1D = get_dataset1D()  # restore original
p = dataset1D.plot()


###############################################################################
# create the apodized dataset (if apply is False, the apodization function
# is not applied to the dataset,
# but returned)

apodfunc = dataset1D.em(lb=100. * ur.Hz, apply=False)

apodfunc.plot(xlim=(0, 25000), zlim=(-2, 2))

dataset1D.em(lb=100. * ur.Hz, apply=True)
dataset1D.plot(data_only=True, clear=False)
dataset1D.ax.text(12500, 1.70,
                 'Multiple display (original & em apodized fids + '
                 'apod.function)',
                 ha='center', fontsize=10)

###############################################################################
# Apodization functions can be em, gm, sp ...
dataset1D = get_dataset1D()  # restore original
p = dataset1D.plot()

###############################################################################
# gm apodization:
LB = 50. * ur.Hz
GB = 100. * ur.Hz
apodfunc = dataset1D.gm(gb=GB, lb=LB, apply=False)

apodfunc.plot(xlim=(0, 25000), clear=False, zlim=(-2, 2))

dataset1D.gm(gb=GB, lb=LB)  # apply=True by default
dataset1D.plot(data_only=True, clear=False)

dataset1D.ax.text(25000, 500,
                 'Multiple display (original & gm apodized fids + '
                 'apod.function)',
                 ha='center', fontsize=10)

###############################################################################
# sp apodization:

###############################################################################
# **TODO**: sp function


scp.show()

