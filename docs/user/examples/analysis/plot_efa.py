# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================
"""
EFA analysis example
---------------------
In this example, we perform the Evolving Factor Analysis

"""
import os
import numpy as np
import spectrochempy as scp
# sphinx_gallery_thumbnail_number = 4

############################################################
# Upload and preprocess a dataset

datadir = scp.general_preferences.datadir
dataset = scp.read_omnic(os.path.join(datadir, 'irdata',
                                      'nh4y-activation.spg'))


# columns masking
dataset[:, 1230.0:920.0] = scp.masked  # do not forget to use float in slicing
dataset[:, 5997.0:5993.0] = scp.masked

# row masking (just for an example
# dataset[10:16] = scp.masked

# difference spectra
# dataset -= dataset[-1]

dataset.plot_stack()

############################################################
#  Evolving Factor Analysis

efa = scp.EFA(dataset)

f = efa.get_forward(npc=6, plot=True)
b = efa.get_backward(npc=6, plot=True)


npc = 4
cut = np.max(f[:, npc].data)

c = efa.get_conc(npc, plot=True)

scp.show()
