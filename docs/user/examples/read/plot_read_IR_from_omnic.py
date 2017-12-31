# coding: utf-8
"""
Loading an IR (omnic SPG) experimental file
============================================

Here we load an experimental SPG file (OMNIC) and plot it.

"""

import os
import spectrochempy as scp

###################################################################
# Loading and stacked plot of the original

datadir = scp.datadir.path

dataset = scp.NDDataset.read_omnic(os.path.join(datadir,
                                              'irdata', 'NH4Y-activation.SPG'))

dataset.plot_stack(style='paper')
scp.show()

##################################################################
# change the unit of y axis, the y origin as well as the title of the axis

dataset.y.to('hour')
dataset.y -= dataset.y[0]
dataset.y.title = 'acquisition time'

dataset.plot_stack()
scp.show()
