# coding: utf-8
"""

Loading of experimental 1D NMR data
===================================

In this example, we load a NMR dataset (in the Bruker format) and plot it.

"""

import os
from spectrochempy import core

##########################################################
# `preferences.datadir` contains the path to a default data directory.

datadir = core.preferences.datadir

path = os.path.join(datadir, 'nmrdata', 'bruker', 'tests', 'nmr', 'bruker_1d')

##########################################################
# load the data in a new dataset

ndd = core.NDDataset.read_bruker_nmr(path, expno=1, remove_digital_filter=True)

##########################################################
# view it...

core.plot(ndd, style='paper')
core.show()
