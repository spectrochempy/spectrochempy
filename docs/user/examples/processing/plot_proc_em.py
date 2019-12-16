# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
Exponential window multiplication
=====================================================

In this example, we exponential window multiplication to apodize a NMR signal in the time domain.

"""

import spectrochempy as scp
import os
Hz = scp.ur.Hz
us = scp.ur.us

dataset1D = scp.NDDataset()
path = os.path.join(scp.general_preferences.datadir, 'nmrdata', 'bruker', 'tests', 'nmr', 'bruker_1d')
dataset1D.read_bruker_nmr(path, expno=1, remove_digital_filter=True)

########################################################################################################################
# Normalize the dataset values and reduce the time domain

dataset1D /= dataset1D.real.data.max()  # normalize
dataset1D = dataset1D[0.:15000.]

########################################################################################################################
# Apply exponential window apodization

new1, curve1 = scp.em(dataset1D.copy(), lb=20*Hz, retfunc=True, inplace=False)

########################################################################################################################
# Apply a shifted exponential window apodization
# defualt units are HZ for broadening and microseconds for shifting

new2, curve2 = dataset1D.copy().em(lb=100*Hz, shifted=10000*us , retfunc=True, inplace=False)



########################################################################################################################
# Plotting

p = dataset1D.plot(zlim=(-2, 2), color='k')

curve1.plot(color='r')
new1.plot(color='r', clear=False, label=' em = 20 hz')


curve2.plot(color='b', clear=False)
new2.plot(dcolor='b', clear=False, label=' em = 30 HZ, shifted = ' )



scp.show()