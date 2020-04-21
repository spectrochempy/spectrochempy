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
import numpy as np

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


f = efa.cut_f(n_pc=7)
f.plot()
b = efa.cut_b(n_pc=7)

f.T.plot(yscale="log", labels= f.y.labels, legend='best')
b.T.plot(yscale="log")

##############################################################################
# Clearly we can retain 4 components, in agreement with what was used to
# generate the data - we set the cutof of the 5th components
#

npc = 4
cut = np.max(f[:, npc].data)

f = efa.cut_f(n_pc=4, cutoff=cut)
b = efa.cut_b(n_pc=4, cutoff=cut)
# we concatenate the datasets to plot them in a single figure
both = scp.concatenate(f, b)
both.T.plot(yscale="log")

c = efa.get_conc(npc, cutoff=cut)
c.T.plot()

#show() # uncomment to show plot if needed()