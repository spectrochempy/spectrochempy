# coding: utf-8
"""
Introduction to NMR processing
===========================================

Here we explain how to display and perform basic processing of NMR file

"""
import os
import spectrochempy as scp

ur = scp.ur

###############################################################################
#
# Here we import a 2D NMR dataset
# 
# Because we will sometimes need to recall the original dataset,
# we create two getting functions

###############################################################################
# Loading the NMR data
# --------------------
# Let's define the 2D dataset getting function
datadir = scp.datadir.path
def get_dataset2D():
    """Read the 2D dataset"""
    s2D = scp.NDDataset()
    path = os.path.join(datadir, 'nmrdata', 'bruker', 'tests', 'nmr',
                        'bruker_2d')
    s2D.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    return s2D


###############################################################################
# Now get the 1D dataset
dataset2D = get_dataset2D()
print(dataset2D)

###############################################################################
# Plot the 2D dataset raw data

dataset2D = get_dataset2D()
ax = scp.plot(dataset2D, xlim=(0., 25000.))

###############################################################################
# Multiple display is possible for 2D spectra
dataset2D.plot()
ax = dataset2D.plot(imag=True, cmap='jet', data_only=True, clear=False)


###############################################################################
# Apodization of 2D data

dataset2D = get_dataset2D()
dataset2D.plot(xlim=(0., 5000.))

LB = 20. * ur.Hz
dataset2D.em(lb=LB)
dataset2D.em(lb=LB / 2, axis=0)
dataset2D.plot(data_only=True, xlim=(0, 5000), cmap='copper', clear=False)

scp.show()

