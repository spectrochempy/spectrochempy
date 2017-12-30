# coding: utf-8
"""
Introduction to NMR processing
===========================================

Here we explain how to display and perform basic processing of NMR file

"""
import os
import spectrochempy as scp

##########################################################
#
# Here we import two datasets, one is 1D and the other is 2D
# 
# Because we will sometimes need to recall the original dataset,
# we create two getting functions

##########################################################
# 1D dataset getting function
# ---------------------------
datadir = scp.datadir.path

def get_source1D():
    source1D = scp.NDDataset()
    path = os.path.join(datadir, 'nmrdata', 'bruker', 'tests', 'nmr', 'bruker_1d')
    source1D.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    return source1D


##########################################################
# 2D dataset getting function
# ---------------------------

def get_source2D():
    source2D = scp.NDDataset()
    path = os.path.join(datadir, 'nmrdata', 'bruker', 'tests', 'nmr', 'bruker_2d')
    source2D.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    return source2D


##########################################################
# get the 1D dataset
# -------------------
source1D = get_source1D()
source1D

##########################################################
# get the 2D dataset
# -------------------
source2D = get_source2D()
source2D

##########################################################
# Plot the 1D dataset raw data
# ----------------------------
# plot the real data
source1D.plot(xlim=(0, 25000), style='paper')

# plot the imaginary data on the same plot
source1D.plot(imag=True, data_only=True, hold=True)
# `data_only=True` to plot only the additional data, without updating the figure setting
# such as xlim and so on.
scp.show()

##########################################################
# To display the imaginary part, one can also simply use the show_complex commands.

ax = source1D.plot(show_complex=True, color='green',
                   xlim=(0., 20000.), zlim=(-2., 2.))

scp.show()

###############################
# Plot the 2D dataset raw data

source2D = get_source2D()
ax = scp.plot(source2D, xlim=(0., 25000.))

##############################
# probably less util, but multiple display is also possible for 2D

source2D.plot()
ax = source2D.plot(imag=True, cmap='jet', data_only=True, hold=True)
scp.show()

#################
# Apodization

source1D = get_source1D()  # restore original
p = source1D.plot()

# create the apodized dataset
ur = scp.ur # the unit registry
lb_source = source1D.em(lb=100. * ur.Hz)

p = lb_source.plot(xlim=(0, 25000), zlim=(-2, 2))

lb_source.ax.text(12500, 1.70, 'Dual display (original & apodized fids)', ha='center',
           fontsize=16)

scp.show()

############################
# Note that the apodized dataset actually replace the original data
# check that both dataset are the same
lb_source is source1D  # note here, that the original data are modified by default
# when applying apodization function.
# Use the `inplace` keyword to modify this behavior

#################################
# If we want to avoid this behavior and create a new dataset instead, we use the `inplace` flag.

source1D = get_source1D()

lb2_source = source1D.em(lb=100. * ur.Hz, inplace=False)

# check that both dataset are different
lb2_source is not source1D

###############################################
# We can also get only the apodization function

source1D = get_source1D()  # restore original
p = source1D.plot()

scp.show()

################################################
# create the apodized dataset (if apply is False, the apodization function is not applied to the dataset,
# but returned)

apodfunc = source1D.em(lb=100. * ur.Hz, apply=False)

apodfunc.plot(xlim=(0, 25000), zlim=(-2, 2))

source1D.em(lb=100. * ur.Hz, apply=True)
source1D.plot(data_only=True, hold=True)
source1D.ax.text(12500, 1.70,
           'Multiple display (original & em apodized fids + apod.function)',
           ha='center', fontsize=14)
scp.show()

######################################
# Apodization function can be em, gm, sp ...

source1D = get_source1D()  # restore original
p = source1D.plot()

LB = 50. * ur.Hz
GB = 100. * ur.Hz
apodfunc = source1D.gm(gb=GB, lb=LB, apply=False)

apodfunc.plot(xlim=(0, 25000), hold=True, zlim=(-2, 2))

source1D.gm(gb=GB, lb=LB)  # apply=True by default
source1D.plot(data_only=True, hold=True)

source1D.ax.text(12500, 1.70,
           'Multiple display (original & gm apodized fids + apod.function)',
           ha='center', fontsize=14)

scp.show()

# **TODO**: sp function

################################################
# Apodization of 2D data

source2D = get_source2D()
source2D.plot(xlim=(0., 5000.))

LB = 20. * ur.Hz
source2D.em(lb=LB)
source2D.em(lb=LB / 2, axis=0)
source2D.plot(data_only=True, xlim=(0, 5000), cmap='copper', hold=True)

scp.show()





