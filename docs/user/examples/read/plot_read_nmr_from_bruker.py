# coding: utf-8
"""

Loading of experimental 1D NMR data
===================================

In this example, we load a NMR dataset (in the Bruker format) and plot it.

"""

import spectrochempy as scp
import os

##########################################################
# `datadir.path` contains the path to a default data directory.

datadir = scp.general_preferences.datadir

path = os.path.join(datadir, 'nmrdata', 'bruker', 'tests', 'nmr', 'bruker_1d')

##########################################################
# load the data in a new dataset

ndd = scp.NDDataset.read_bruker_nmr(path, expno=1, remove_digital_filter=True)

##########################################################
# view it...

scp.plot(ndd, style='paper')

#show() # uncomment to show plot if needed()
