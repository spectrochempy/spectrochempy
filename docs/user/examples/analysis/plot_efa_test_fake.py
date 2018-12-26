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

############################################################
# Upload and preprocess a dataset

dataset = scp.load("irdata/nh4y-activation.spg")


# columns masking
#dataset[:, 1230.0:920.0] = scp.masked  # do not forget to use float in slicing
#dataset[:, 5997.0:5993.0] = scp.masked

# row masking (just for an example
#dataset[10:16] = scp.masked

dataset.plot_stack()

############################################################
#  Evolving Factor Analysis

efa = scp.EFA(dataset)


f = efa.get_forward(npc=7, plot=True)
b = efa.get_backward(npc=7, plot=True)

#scp.show()

##############################################################################
# Clearly we can retain 4 components, in agreement with what was used to
# generate the data - we set the cutof of the 5th components
#

npc = 4
cut = np.max(f[:, npc].data)

f = efa.get_forward(npc=4, cutoff=cut, plot=True)
b = efa.get_backward(npc=4, cutoff=cut, plot=True)

#scp.show()

c = efa.get_conc(npc, cutoff=cut, plot=True)

scp.show()