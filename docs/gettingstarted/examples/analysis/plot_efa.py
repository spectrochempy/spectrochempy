# -*- coding: utf-8 -*-
# flake8: noqa
# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
EFA analysis example
======================

In this example, we perform the Evolving Factor Analysis

"""
########################################################################################################################
# sphinx_gallery_thumbnail_number = 2

import spectrochempy as scp
import os

########################################################################################################################
# Upload and preprocess a dataset

datadir = scp.preferences.datadir
dataset = scp.read_omnic(os.path.join(datadir, "irdata", "nh4y-activation.spg"))

########################################################################################################################
# columns masking

dataset[:, 1230.0:920.0] = scp.MASKED  # do not forget to use float in slicing
dataset[:, 5997.0:5993.0] = scp.MASKED

########################################################################################################################
# difference spectra

dataset -= dataset[-1]
dataset.plot_stack()

########################################################################################################################
# column masking for bad columns

dataset[10:12] = scp.MASKED

########################################################################################################################
#  Evolving Factor Analysis

efa = scp.EFA(dataset)

########################################################################################################################
# Show results
#
npc = 4
c = efa.get_conc(npc)
c.T.plot()

# scp.show()  # Uncomment to show plot if needed (not necessary in jupyter notebook)
