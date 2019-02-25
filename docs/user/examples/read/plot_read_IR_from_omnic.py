# coding: utf-8
"""
Loading an IR (omnic SPG) experimental file
============================================

Here we load an experimental SPG file (OMNIC) and plot it.

"""

import spectrochempy as scp
import os

###################################################################
# Loading and stacked plot of the original

datadir = scp.general_preferences.datadir

dataset = scp.NDDataset.read_omnic(os.path.join(datadir,
                                              'irdata', 'nh4y-activation.spg'))

dataset.plot_stack(style='paper')

##################################################################
# change the unit of y axis, the y origin as well as the title of the axis

dataset.y.to('hour')
dataset.y -= dataset.y[0]
dataset.y.title = 'acquisition time'

dataset.plot_stack()

#show() # uncomment to show plot if needed()