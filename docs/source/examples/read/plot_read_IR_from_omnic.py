# coding: utf-8
"""
Loading an IR (omnic SPG) exerimental file
===========================================

Here we load an experimental SPG file (OMNIC) and plot it.

"""

import os
from spectrochempy import api

###################################################################
# Loading and stacked plot of the original

datadir = api.preferences.datadir

dataset = api.NDDataset.read_omnic(os.path.join(datadir,
                                           'irdata', 'NH4Y-activation.SPG'))

dataset.plot_stack(style='paper')
api.show()

##################################################################
# change the unit of y axis, the y origin as well as the title of the axis

dataset.y.to('hour')
dataset.y -= dataset.y[0]
dataset.y.title = 'acquisition time'

dataset.plot_stack()
api.show()
