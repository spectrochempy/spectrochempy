# coding: utf-8
"""

Loading of experimental 1D NMR data
===================================

In this example, we load a NMR dataset (in the Bruker format) and plot it.

"""

from spectrochempy.api import *

##########################################################
# ``data`` contains the path to a default data directory.

path = os.path.join(scpdata, 'nmrdata', 'bruker', 'tests', 'nmr', 'bruker_1d')

##########################################################
# load the data in a new dataset

ndd = NDDataset.read_bruker_nmr(path, expno=1, remove_digital_filter=True)

##########################################################
# view it...

plot(ndd, style='paper')
show()
