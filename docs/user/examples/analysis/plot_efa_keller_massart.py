# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
EFA analysis (Keller and Massart original example)
--------------------------------------------------
In this example, we perform the Evolving Factor Analysis of a TEST dataset
(ref. Keller and Massart, Chemometrics and Intelligent Laboratory Systems,
12 (1992) 209-224 )

"""
import spectrochempy as scp
import numpy as np

# sphinx_gallery_thumbnail_number = 5

########################################################################################################################
# Generate a test dataset
# ----------------------------------------------------------------------------------------------------------------------
# 1) Simulated Chromatogram
# *************************

t = scp.Coord(np.arange(15), units='minutes', title='time')  # time coordinates
c = scp.Coord(range(2), title = 'components')                                      # component coordinates

data = np.zeros((2,15), dtype=np.float64)
data[0, 3:8] = [1,3,6,3,1] # compound 1
data[1, 5:11] = [1,3,5,3,1,0.5] #compound 2

dsc = scp.NDDataset(data=data, coords=[c, t])

########################################################################################################################
# 2) Adsorption spectra
# **********************


spec = np.array([[2.,3.,4.,2.],[3.,4.,2.,1.]])
w = scp.Coord(np.arange(1,5,1), units='nm', title='wavelength')

dss = scp.NDDataset(data=spec, coords=[c, w])

########################################################################################################################
# 3) Simulated data matrix
# ************************

dataset = scp.dot(dsc.T, dss)
dataset.data = np.random.normal(dataset.data,.2)
dataset.title = 'intensity'

dataset.plot_stack()

########################################################################################################################
# 4) Evolving Factor Analysis
# ***************************

efa = scp.EFA(dataset)

########################################################################################################################
# Plots of the log(EV) for the forward and backward analysis
#

f = efa.get_forward(plot=True)
b = efa.get_backward(plot=True)

########################################################################################################################
# Looking at these EFA curves, it is quite obvious that only two components
# are really significant, and this correspond to the data that we have in
# input.
# We can consider that the third EFA components is mainly due to the noise,
# and so we can use it to set a cut of values

n_pc = 2
cut = np.max(f[:, n_pc].data)

efa.get_forward(plot=True, n_pc=2, cutoff=cut, legend='upper right')
efa.get_backward(plot=True, n_pc=2, cutoff=cut, clear=False, legend='lower right')
# with clear=False, we will plot the two graphs on the same figure
# TODO: solve the problem with legends when two plots on the same figures


########################################################################################################################
# Get the abstract concentration profile based on the FIFO EFA analysis
#

c = efa.get_conc(n_pc, cutoff=cut, plot=True)
