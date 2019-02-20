# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# ======================================================================================================================




""" Tests for the ndplugin module

"""
import numpy as np
import pandas as pd

from spectrochempy.core.processors.autosub import autosub
from spectrochempy.api import show
from spectrochempy.utils import SpectroChemPyWarning
from spectrochempy.utils.testing import (assert_equal, assert_array_equal,
                         assert_array_almost_equal, assert_equal_units,
                         raises)

import pytest
import numpy as np
import os


# autosub
#------

def test_autosub(IR_dataset_2D):

    dataset = IR_dataset_2D

    ranges = [5000., 5999.], [1940., 1820.]


    s1 = dataset.copy()
    ref = s1[-1].squeeze()

    dataset.plot_stack()
    ref.plot(clear=False, linewidth=2., color='r')


    s2 = dataset.copy()

    s3 = s2.autosub(ref, *ranges, dim=-1, method='varfit', inplace=False)
    s3.plot()

    # inplace = False
    assert np.round(s2.data[-1,0],4) != 0.0000
    assert np.round(s3.data[-1,0],4) == 0.0000
    s3.name="varfit"

    s3.plot_stack()

    s4 = dataset.copy()
    s4.autosub(ref, *ranges, method='chi2', inplace=True)
    s4.name = "chi2, inplace"
    assert np.round(s4.data[-1,0],4) == 0.0000

    s4.plot_stack()  #true avoid blocking due to graphs

    s4 = dataset.copy()
    s = autosub(s4, ref, *ranges, method='chi2')
    assert np.round(s4.data[-1, 0], 4) != 0.0000
    assert np.round(s.data[-1, 0], 4) == 0.0000
    s.name = 'chi2 direct call'

    s.plot_stack()

    s5 = dataset.copy()
    ref2 = s5[:,0].squeeze()
    ranges2 = [0,5], [45, 54]

    # TODO : not yet implemented
    #s6 = s5.autosub(ref2, *ranges2, dim='y', method='varfit', inplace=False)
    #s6.plot()


    show()

