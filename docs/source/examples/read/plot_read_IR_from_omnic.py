# coding: utf-8
"""
Loading an IR (omnic SPG) exerimental file
===========================================

Here we load an experimental SPG file (OMNIC) and plot it.

"""
from spectrochempy.api import *

###################################################################
# Loading and stacked plot of the original

source = NDDataset.read_omnic(os.path.join(scpdata,
                                           'irdata', 'NH4Y-activation.SPG'))

source.plot_stack(style='paper')
show()

##################################################################
# change the unit of y axis, the y origin as well as the title of the axis

source.y.to('hour')
source.y -= source.y[0]
source.y.title = 'acquisition time'

source.plot_stack()
show()
