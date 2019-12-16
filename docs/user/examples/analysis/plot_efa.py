# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
EFA analysis example
---------------------
In this example, we perform the Evolving Factor Analysis

"""
import spectrochempy as scp
import matplotlib.pyplot as plt
import os

# sphinx_gallery_thumbnail_number = 2

########################################################################################################################
# Upload and preprocess a dataset

datadir = scp.general_preferences.datadir
dataset = scp.read_omnic(os.path.join(datadir, 'irdata',
                                      'nh4y-activation.spg'))

########################################################################################################################
# columns masking

dataset[:, 1230.0:920.0] = scp.MASKED  # do not forget to use float in slicing
dataset[:, 5997.0:5993.0] = scp.MASKED

########################################################################################################################
# difference spectra

dataset -= dataset[-1]
dataset.plot_stack()   # figure 1

########################################################################################################################
# column masking for bad columns

dataset[10:12] = scp.MASKED

########################################################################################################################
#  Evolving Factor Analysis

efa = scp.EFA(dataset)


########################################################################################################################
# Show results

npc = 4
c = efa.get_conc(npc)
c.T.plot()

# plt.show() # uncomment to show plot if needed()
