# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
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
from spectrochempy import core

############################################################
# Upload and preprocess a dataset

datadir = core.preferences.datadir
dataset = core.read_omnic(os.path.join(datadir, 'irdata',
                                      'NH4Y-activation.SPG'))
# columns masking
dataset[:, 1230.0:920.0] = core.masked  # do not forget to use float in slicing
# row masking (just for an example
dataset[10:11] = core.masked

# difference spectra
dataset -= dataset[-1]

dataset.plot_stack()

############################################################
#  Evolving Factor Analysis

efa = core.EFA(dataset)

npc = 3
c = efa.get_conc(npc, plot=True)


