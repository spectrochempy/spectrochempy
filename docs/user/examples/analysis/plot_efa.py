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
import spectrochempy as scp

############################################################
# Upload and preprocess a dataset

datadir = scp.datadir.path
dataset = scp.read_omnic(os.path.join(datadir, 'irdata',
                                      'NH4Y-activation.SPG'))
# columns masking
dataset[:, 1230.0:920.0] = scp.masked  # do not forget to use float in slicing
# row masking (just for an example
dataset[10:11] = scp.masked

# difference spectra
dataset -= dataset[-1]

dataset.plot_stack()

############################################################
#  Evolving Factor Analysis

efa = scp.EFA(dataset)

npc = 3
c = efa.get_conc(npc, plot=True)


