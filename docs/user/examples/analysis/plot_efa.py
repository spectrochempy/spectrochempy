# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
EFA analysis example
---------------------
In this example, we perform the Evolving Factor Analysis

"""
from spectrochempy import *

# sphinx_gallery_thumbnail_number = 4

########################################################################################################################
# Upload and preprocess a dataset

datadir = general_preferences.datadir
dataset = read_omnic(os.path.join(datadir, 'irdata',
                                      'nh4y-activation.spg'))

########################################################################################################################
# columns masking

dataset[:, 1230.0:920.0] = MASKED  # do not forget to use float in slicing
dataset[:, 5997.0:5993.0] = MASKED

########################################################################################################################
# difference spectra

dataset -= dataset[-1]
dataset.plot_stack()   # figure 1

########################################################################################################################
# column masking for bad columns

dataset[10:12] = MASKED

########################################################################################################################
#  Evolving Factor Analysis

efa = EFA(dataset)


########################################################################################################################
# Show results

npc = 4
c = efa.get_conc(npc, plot=True)

#show() # uncomment to show plot if needed()
